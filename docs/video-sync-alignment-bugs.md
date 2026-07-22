# Video ↔ Capacitance Alignment Bugs (acquisition anchor)

**Date:** 2026-07-14
**Reference recording:** `Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.{mp4,h5,txt}`

## Symptom

In the side-by-side demo clip (`make_sync_video.py`) the mouse video ran ahead
of the capacitance trace by a roughly constant ~2.5 s: a lick was visible in the
video panel before its capacitance deflection reached the centered dot.

## Where the error actually is

Not in the clip renderer. `make_sync_video.py`'s math is self-consistent — both
panels are driven by session time τ (τ = 0 ≡ `start_time`):

- **Cap side:** `data_analysis.filter_data` zero-bases `time_data` at `start_time`.
- **Video side:** frame session = `sidecar_pts − video_base`, with
  `video_base = sidecar[frame_index] − sidecar[0]`.
- The mp4-internal PTS (read via `-copyts` + ffprobe) vs the `.txt` sidecar PTS
  (used for `video_base`) drift only ~0.02 s over 30 s — negligible.

So the two panels line up **iff** the anchor holds: *the bookmarked frame
coincides with `start_time`*. That single link was wrong, for two reasons on the
**acquisition** side. `make_sync_video` faithfully propagated the error.

### Bug 1 — bookmark round-trip latency (dominant, the ~2.5 s)

`components/sensor_card.py` stamps `start_time = time.time()` on the host, then a
few lines later calls `client.bookmark()`, a synchronous round-trip to the Pi.
The Pi returns its *current* frame count, captured L seconds later (one-way
latency). So the bookmarked frame actually occurred at `start_time + L`, but the
pipeline treated it as `start_time`. Every video frame was therefore labeled L
seconds too early → the video panel leads the trace by L (~2.5 s here).

**Worse, the blocking call also destroyed data.** `start_sensor` runs on the
Solara asyncio event loop — the *same* loop that drives `recorder.record_sensors`
(`recording/recorder.py:71`, `await asyncio.sleep(0)` per iteration). A
synchronous socket round-trip on that loop freezes the loop, so **no sensor is
sampled for the whole round-trip**. Confirmed on the reference file: **every one
of the 24 sensors on all four boards has an identical ~5.019 s gap at
`start_time`** — by far the largest gap in the 8127 s recording (next largest
0.23 s). The full round-trip (~5 s) is the gap; the one-way (~2.5 s) is the video
lead. (The camera `start_session` call is *not* the culprit: the bookmark frame
sits ~32 s into the video, so `start_session` ran ~32 s before `start_time`.)

L was **not recorded anywhere**: only `video_frame_index` and `video_pts` were
persisted. The cap clock is host-unix; the video clock is Pi `SensorTimestamp`;
with no second shared event, L could not be recovered post-hoc — which is why no
display change fixed it and a manual `--sync-offset` was the only lever for an
already-captured file.

### Bug 2 — off-by-one in the bookmark (~8 ms, minor)

`pi/camera_backend.py::_on_frame` writes a frame's timestamp to the sidecar
(0-based line = frame index) and *then* increments `_frame_count`. `bookmark`
returned `frame_index = _frame_count` (the next, not-yet-written frame) while
`pts = _last_frame_ts_ns` (the frame just written, `_frame_count − 1`). So
`video_frame_index` pointed one frame past the pts it was paired with.

## Fixes applied (this change)

0. **Non-blocking bookmark (stops the data loss)** — `components/sensor_card.py`
   now runs the bookmark round-trip on its own `daemon` thread (mirroring the
   existing stop-path pattern in `session_controls.py`) instead of on the event
   loop, so `record_sensors` keeps sampling during the round-trip and the
   ~5 s acquisition gap disappears. `write_video_metadata` is already h5-lock
   guarded; `start_sensor` returns the thread so tests can join it.
   (`tests/test_camera_integration.py::test_bookmark_does_not_block_the_caller`)

1. **Off-by-one** — `pi/camera_backend.py::bookmark` now returns
   `frame_index = _frame_count - 1`, matching the reported `pts`.
   (`tests/test_camera_session_lifecycle.py::test_bookmark_index_and_pts_describe_same_frame`)

2. **Record the latency** — the bookmark round-trip is now bracketed with host
   wall-clock, and the Pi clock is persisted, so L is measurable from the file:
   - `components/sensor_card.py` captures `host_before`/`host_after` around
     `client.bookmark()` and forwards them plus `pi_monotonic`.
   - `recording/recorder.py::write_video_metadata` writes new per-cycle
     datasets: `video_pi_monotonic`, `video_bookmark_host_before`,
     `video_bookmark_host_after` (each omitted when the caller doesn't supply it,
     so older callers are unaffected).
     (`tests/test_video_metadata.py`, `tests/test_camera_integration.py`)

   Recovery formula for a future recording:
   `L ≈ (host_before + host_after) / 2 − start_time`, and the corrected video
   anchor adds L to every frame's session time (equivalently, subtracts L from
   `video_base`).

3. **Consume the latency in the renderer** — `make_sync_video.py::load_recording`
   reads the bracket (via `bookmark_latency()`) and stores `rec.bookmark_latency`
   (0.0 when the fields are absent). `render_clip` corrects the anchor with
   `video_base_eff = rec.video_base − rec.bookmark_latency`, used for both frame
   trimming and per-frame timing, so a recording that carries the bracket aligns
   with no manual `--sync-offset`. `--sync-offset` remains only as a residual
   nudge. (`tests/test_make_sync_video.py`)

## Renderer bug — decode/pts frame-count drift (separate from the anchor)

**Symptom.** Two clips of very different length covering the same session showed
the *same* capacitance trace but a *different* video frame (~1 s apart, growing
with clip length) — a per-clip inconsistency the constant anchor offset can't
explain.

**Cause.** `make_sync_video.TrimmedFrameSource` times each frame by the ffprobe
pts list (`frame_sess`) but *counts* frames by imageio's sequential decode. This
footage is VFR (coded 240 fps, real ~120); imageio's default reader forces CFR
and **duplicates** frames, so it decodes more frames than the pts list has (e.g.
37200 vs 37066 over 300 s). Each duplicate advances the decode counter without a
matching pts, slipping the frame↔session map ~1 s per ~300 s — worse the longer
the clip, so two clips of different length disagree at the same session.

**Fix.** Open the reader in passthrough (`imageio.get_reader(..., output_params=
["-vsync", "0"])`) so exactly one decoded frame maps to each pts entry. Verified:
a 200 s clip and a 10 s clip now show the identical video frame at the same
session (0.0000 s difference). (`tests/test_make_sync_video.py::
test_trimmed_frame_source_decode_matches_pts`)

## Correction (2026-07-16) — the latency was the END of the bracket, not the midpoint

**Symptom.** A new finger-touch test recording
(`Lickometry Data/raw_data_2026-07-16_09-56-26`, which *does* carry the host
bracket, so `latency` = 2.50 s was applied automatically) still showed the video
leading the trace by ~2.5 s in `make_sync_video`. The midpoint correction removed
only half the true lag.

**Root cause.** `bookmark_latency` placed the bookmarked frame at
`(host_before + host_after) / 2`, which is right only if the round-trip delay is
symmetric *network* latency. It isn't. The round-trip on this file is **4.71 s** —
that is the Pi blocked/queued, not the wire. `camera_backend.bookmark()` reads
`_last_frame` when the call *runs* on the Pi, which is at the **end** of the
round-trip (~`host_after`), and the camera kept capturing during the block, so
`_last_frame` advanced the whole 4.7 s. Proof from the file itself: `pi_monotonic`
(bookmark exec, Pi clock) − `video_pts` (grabbed frame, Pi clock) = **0.016 s**, so
the frame was grabbed ~16 ms before exec ≈ `host_after`, not 2.5 s before it.

**Evidence.** Cross-correlating the video's per-frame motion energy against
|d cap/dt| over the 9 rhythmic finger presses: residual video-lead **+2.41 s**
with the midpoint latency (2.50 s), **+0.05 s** with `host_after − start_time`
(4.86 s), **+0.07 s** with `host_after − start_time − (pi_monotonic − video_pts)`
(4.84 s).

**Fix.**
- `bookmark_latency(host_after, start_time, pi_monotonic=None, pts=None)` now
  returns `(host_after − start_time) − (pi_monotonic − pts)` via the new
  `_frame_host_time` helper (the `pi_monotonic − pts` gap is dropped when the Pi
  clocks aren't recorded, leaving `host_after − start_time`).
- `VideoAnchor.drift_slope` uses the same end-of-bracket host time at **both**
  bookmark endpoints (was the midpoint at each).
- `VideoAnchor` gained `pi_monotonic`/`video_pts` (start) and
  `stop_pi_monotonic`/`stop_pts` (stop); `read_video_anchor` reads them.
- `recorder.write_video_metadata` now also persists `video_stop_pi_monotonic`;
  `sensor_card.bookmark_stop` forwards it. (Start already persisted
  `video_pi_monotonic` + `video_pts`.)
- `false_positive_analysis.alignment_from_bookmark` updated to the same formula
  and gained an optional `pi_monotonic` arg.
- Tests updated in `tests/test_trimcrop.py`, `test_make_sync_video.py`,
  `test_crop_video.py`, `test_bookmark_alignment.py`.

**Residual.** The only unmodelled term is the Pi→host response one-way latency
(~ms). No manual `--sync-offset` is needed for recordings that carry the bracket.

## Still open (follow-ups)

- **This reference file (`ACG-26-3`, 2026-07-13) has no recorded L** and a real
  ~5 s hole in every sensor at `start_time` (permanent data loss — the samples
  were never taken). It predates the fix, so `rec.bookmark_latency` is 0.0 and
  its only alignment lever remains a one-time `--sync-offset` measured from a
  single clearly-visible lick (≈ half the 5 s gap ≈ 2.5 s).
- Threading removes the acquisition stall but the bookmarked frame is still
  grabbed ~one-way-latency after `start_time`; the recorded bracket (fix 2) is
  what makes that residual recoverable, so future recordings still depend on the
  renderer correction (fix 3). If the wireless one-way latency itself can be cut,
  the residual shrinks toward zero.
- The mock client (`hardware/pi_camera_mock.py`) still returns a synthetic
  1-based `frame_index`; it has no sidecar, so `make_sync_video` never runs on
  mock output. Left as-is intentionally.
