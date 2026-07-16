# Clock-Drift Correction via Stop Bookmark — Design

**Date:** 2026-07-16
**Status:** approved, pending implementation
**Related:** `docs/video-sync-alignment-bugs.md` (start-bookmark latency),
`docs/alignment-decisions.md` (why not sipper-in-cap anchoring)

## Problem

Cap↔video alignment currently anchors on a single event: the video bookmark at
the sensor Start click. The capacitance trace runs on the host clock
(`time.time()`); the video runs on the Pi's `SensorTimestamp` clock. Two free-
running crystals drift relative to each other (~10–50 ppm typical). Over a ~2 h
session (reference recording: 8127 s) that is **~0.15–0.4 s** of accumulated
skew — 1–3 inter-lick intervals (ILI ≈ 100–140 ms). A single-point anchor cannot
correct it: the misalignment grows linearly across the clip.

A second anchor far from the first lets us fit the clock *rate*, not just the
offset. We add a **Stop bookmark** (mirroring the existing Start bookmark) and
consume it as a two-point linear fit in the render/crop pipeline.

## Non-goals

- FP `alignment_from_bookmark` stays single-anchor for now (follow-up).
- No sipper-based anchoring (see `docs/alignment-decisions.md` — rejected).
- No new recording benefits retroactively: existing files have no stop bookmark
  and render exactly as today (`slope = 1.0`).

## Timing model

Session time τ is host-seconds since `start_time` (τ = 0 ≡ Start click), the same
axis the capacitance trace uses. For a video frame with PTS `pts_sec` (Pi video
clock, seconds):

```
τ(pts_sec) = latency + slope · (pts_sec − pts_start_sec)
```

- `pts_start_sec` — PTS of the Start-bookmark frame, `pts_ns[frame_index] / 1e9`.
- `latency` — `bookmark_latency(host_before, host_after, start_time)` (existing;
  0.0 when the Start host bracket is absent). The Start-bookmark frame was
  captured ~`latency` s after `start_time`, so it sits at τ = `latency`.
- `slope` — host-seconds per video-second:
  ```
  slope = (mid_stop − mid_start) / (pts_stop_sec − pts_start_sec)
  ```
  where `mid_x = (host_before_x + host_after_x) / 2` and `pts_stop_sec =
  pts_ns[stop_frame_index] / 1e9`. Defaults to **1.0** when the Stop bookmark is
  absent.

**Identity check:** at `slope = 1`, τ(pts_sec) = latency + (pts_sec −
pts_start_sec), which is algebraically identical to the current
`(pts − pts[0])/1e9 − video_base_eff` with `video_base_eff = pts_start_sec −
latency`. Existing recordings are unaffected.

Derivation: a frame's true host time is `mid_start + slope·(pts_sec −
pts_start_sec)` (anchor at the Start bookmark, scale video-elapsed by the rate).
τ = host_time − start_time, and `mid_start − start_time = latency`.

## Components

### 1. Acquisition — record the Stop bookmark

Mirror the Start-bookmark path, which already runs the wireless round-trip on a
daemon thread so acquisition never stalls (see `docs/video-sync-alignment-bugs.md`
fix 0).

- **`components/sensor_card.py`** — `stop_sensor()`: for the designated camera
  sensor, fire a threaded bookmark bracketed with `host_before`/`host_after`,
  writing stop-side datasets for the sensor's current cycle. The camera session
  spans the whole global session, so it is still active here.
- **`components/session_controls.py`** — `stop_recording()` stops still-recording
  sensors directly (it does not call `stop_sensor`). Fire the same camera-sensor
  stop bookmark there too, **before** the `_camera_stop_and_fetch` thread sends
  `STOP_SESSION` (afterwards there is no active session to bookmark).
- Factor the stop bookmark into **one shared helper** (e.g.
  `sensor_card.bookmark_stop(sensor_id, cycle)`) called from both paths so they
  cannot drift apart. It returns the thread so tests can join it, matching
  `start_sensor`.
- **`recording/recorder.py`** — `write_video_metadata()` gains stop-side params
  (`stop_frame_index`, `stop_pts`, `stop_host_before`, `stop_host_after`) →
  new per-cycle datasets `video_stop_frame_index{suffix}`, `video_stop_pts{suffix}`,
  `video_stop_bookmark_host_before{suffix}`, `video_stop_bookmark_host_after{suffix}`.
  Any None is not written (backward compatible, same rule as the start fields).

### 2. Timing core — `video/trimcrop.py`

Introduce a `SessionClock` dataclass carrying `(pts_start_sec, latency, slope)`
with a `session_time(pts_sec)` method implementing the formula above (numpy-
broadcast). A builder `session_clock(anchor, pts_ns)` computes:
- `pts_start_sec = pts_ns[anchor.video_frame_index] / 1e9`
- `latency = anchor.latency`
- `slope = anchor.drift_slope(pts_ns)`

Replace the scalar `video_base` / `video_base_eff` interface:
- `frame_session_times(clock, pts_ns)` — session time of every sidecar frame.
- `compute_trim_frames(clock, pts_ns, start, end)` — frame selection uses the
  drift-corrected times.
- `trim_window_seconds(clock, pts_ns, start, end)` — unchanged return contract
  (`start_frame, stop_frame, start_sec, end_sec`); the seconds remain PTS-based
  original-timeline seconds for ffmpeg seeking (`(pts_ns[sf]−pts_ns[0])/1e9`),
  only frame *selection* is drift-aware.
- `probe_frame_session_times(path, clock)` — applies the same `clock.session_time`
  to the ffprobe PTS, unifying the sidecar and ffprobe timing paths (they
  currently duplicate the base subtraction).

`VideoAnchor` gains: `stop_frame_index`, `stop_host_before`, `stop_host_after`
(all `float | None`) and a method `drift_slope(pts_ns) -> float` returning 1.0
when any stop field is absent. `read_video_anchor` reads the stop datasets from
the **same resolved cycle** as the start fields (same suffix discipline already
enforced there).

### 3. Consumers

- **`make_sync_video.py`** — `clip_trim_window` / `render_clip` build one
  `SessionClock` and thread it through `trim_window_seconds` and
  `probe_frame_session_times`. `Recording` carries the clock (or the anchor +
  pts_ns to build it) instead of the bare `video_base` + `bookmark_latency`.
- **`crop_video.py`** — `compute_crop_window` builds the same clock. Crop and
  render therefore select the identical window (existing shared invariant via
  `trim_window_seconds`).

## Error handling / backward compatibility

- **No stop bookmark** (every existing file; camera disabled; a stop round-trip
  that failed): `drift_slope` returns 1.0 → identical to current behavior.
- **Stop round-trip drops on the wireless link**: the daemon bookmark thread
  catches, logs a warning, writes nothing. Stop is never blocked. Slope falls
  back to 1.0.
- **Degenerate fit** (`pts_stop_sec == pts_start_sec`, i.e. same frame): guard
  and fall back to slope 1.0 rather than divide-by-zero.

## Testing (TDD)

1. **Identity:** `SessionClock` with `slope=1, latency=0` reproduces current
   `frame_session_times` / `probe_frame_session_times` output on existing
   fixtures (byte-for-byte on the numbers).
2. **Recovered slope:** fabricate `pts_ns` with a known ppm skew and two
   bookmarks; assert `drift_slope` recovers it within tolerance, and a
   mid-session frame maps to the correct τ (off by the full drift when slope
   forced to 1.0, correct when fit).
3. **Metadata:** `write_video_metadata` writes the four stop datasets when
   supplied, omits each when None; multi-cycle suffixes correct.
4. **Anchor round-trip:** `read_video_anchor` returns the stop fields;
   `drift_slope` = 1.0 when they are absent.
5. **Stop-bookmark thread:** failure path writes nothing and does not raise
   (mirror the existing start-bookmark integration test); shared helper called
   by both `stop_sensor` and the global-stop path.
6. **Window lockstep:** `crop_video` and `make_sync_video` select the same frames
   under a non-unit slope.

## Follow-ups (not this change)

- Wire the two-point drift into FP `alignment_from_bookmark` /
  `video_relative_to_abs` so the automated TP/FP comparison benefits.
