# Sync Video Composite — Design

**Date:** 2026-07-14
**Status:** Approved (design), pending implementation plan

## Goal

Produce a side-by-side demo video from a recording that captured mouse video
concurrently with capacitance: left panel = mouse video, right panel = the
sensor's capacitance trace with a sliding window and a centered dot marking the
current time. This visualizes individual lick deflections against the raw
behavior.

Reference recording: `Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.{mp4,h5,txt}`.

## Scope

- Single script, `make_sync_video.py`.
- Output covers a **configurable clip window** (`--start`/`--end`, required — no
  auto default).
- Only clip-length windows are a design target; full-session (~2 hr) rendering
  is possible but slow and not optimized for.

## Data model (established facts)

Raw h5 layout: `board_FT232H{0..3}/sensor_{n}/` groups. Each sensor has:

- `cap_data` (int64, capacitance samples), `time_data` (float64, unix-epoch
  seconds, ~45.7 Hz).
- Session bookmarks: `start_time`, `stop_time` (unix seconds, set by the
  Start/Stop buttons). Some sensors have numbered variants (`start_time1`,
  `stop_time1`, …); the highest-numbered pair is the one used (matches
  `filter_data` logic).
- Volume/weight: `start_vol`, `stop_vol`, `weight`.

**Exactly one** sensor per recording carries video-sync metadata:

- `video_filename` (bytes), `video_frame_index` (int64), `video_pts` (float64,
  seconds).

The `.txt` sidecar is one PTS per video frame in **nanoseconds** (frame index =
line number, 0-based). Verified: `txt[video_frame_index]/1e9` equals stored
`video_pts` within ~8 ms (one frame). Reference video is 1280×720, ~120–240 fps,
video duration ~8127 s.

Layout csv (e.g. `layout_w_controls.csv`): rows `sensor_number,animal_id`, maps
sensor → animal. For the reference recording the video sensor is
`board_FT232H0/sensor_1` → `ACG-26-3-1`.

## Lick detection — reuse, do not reimplement

Detection lives in `data_analysis.py`:

- `filter_data(raw_h5f, filtered_h5f, sensor_animal_map, logfile, ...)` reads the
  raw h5, trims each sensor to its `start_time`/`stop_time`, **zero-bases**
  `time_data` (0 = start_time), trims to `recording_length` (default 2 hr),
  maps sensor→animal, runs `basic_algorithm`, and writes results per animal.
- Result per animal (in the written/filtered h5): `cap_data`, `time_data`
  (session-relative, 0-based), `lick_times` (session-relative seconds),
  `lick_indices` (indices into the session `cap_data`), plus `optimal_*` variants.

The script runs `filter_data` into a **temp h5**, then reads back the video
animal's `cap_data`, `time_data`, and `lick_indices`. Because the displayed
trace and the markers come from the same session arrays, indices align with no
extra bookkeeping. Markers use the primary `lick_indices` (basic_threshold
algorithm), not the optimal-threshold variant.

`sensor_animal_map` is a pandas object indexed by sensor number with animal id
as the value (as consumed by `filter_data`'s `iterrows()` / `row.name` /
`row.item()`), loaded from the layout csv.

## Time references

- **Session-relative time `τ`**: seconds since `start_time` (the Start button).
  This is the reference for `--start`/`--end` and matches the zero-based
  `time_data` returned by `filter_data`.
- **Raw absolute time**: `unix_t = start_time + τ`.
- **Raw first sample**: `t0_raw = raw time_data[0]` (untrimmed), read from the
  raw h5 before `filter_data` trims.

## Video sync

Assume the stored `(video_frame_index, video_pts)` sync point pins to `t0_raw`
(the first raw cap sample). Then for session time `τ`:

```
video_sec(τ) = video_base + (start_time + τ - t0_raw) + sync_offset
video_base   = pts[video_frame_index]/1e9 - pts[0]/1e9
```

`pts[]` from the `.txt` sidecar. `video_sec` is seconds from the start of the
video file.

The cap-time↔frame correspondence (pinning to `t0_raw`) is **inferred**, not
provable from the data. `--sync-offset` (seconds, default 0) is a manual nudge
to correct any residual constant offset by eyeballing the output. Clock drift
between the PTS clock and the unix clock over long spans is a known limitation;
for clip-length windows it is negligible.

## Architecture — single matplotlib animation

One `matplotlib.animation.FuncAnimation` over a figure with two axes; one
timeline drives both panels so sync is automatic (no cross-pipeline frame
matching). Written to mp4 via the ffmpeg writer.

### Components

1. **Arg parsing / config**
   - `--h5` (required), `--video` (default: resolve `video_filename` relative to
     the h5's directory), `--pts-txt` (default: video path with `.txt`),
     `--layout` (required), `--out` (required), `--start` / `--end` (required,
     session-relative seconds), `--fps` (default 30), `--window` (default 2.5,
     half-width seconds), `--sync-offset` (default 0).
   - Validate: `start < end`, both within `[0, session_duration]`.

2. **Recording loader** — `load_recording(h5_path, layout_path)`
   - Open raw h5, find the group with `video_filename` → record board id, sensor
     number, `video_frame_index`, `video_pts`, `start_time`, and `t0_raw`.
   - Load layout csv into the `sensor_animal_map` shape `filter_data` expects;
     resolve the video sensor's animal id.
   - Run `filter_data` into a temp h5 (temp dir, cleaned up); read back the video
     animal's session `cap_data`, `time_data`, `lick_indices`.
   - Load PTS sidecar (`np.loadtxt`, int64 ns) → compute `video_base`.
   - Return a small dataclass/dict with everything the animator needs.

3. **Sync mapper** — pure function `video_sec(τ)` as defined above.

4. **Video frame source** — `FrameGrabber`
   - Wraps `cv2.VideoCapture`. For output frame `i` at `τ_i = start + i/fps`,
     returns the source frame nearest `video_sec(τ_i)`.
   - Sequential decode: seek once to the clip's first `video_sec`, then read
     forward, emitting the frame whose timestamp is nearest each `τ_i` (output
     fps ≤ source fps, so this is decode-and-skip). Convert BGR→RGB for
     matplotlib.

5. **Animator** — builds the figure (left `imshow` axis, right trace axis),
   defines `init` and `update(i)`:
   - Left: update `imshow` data with the grabbed frame.
   - Right: x-limits `[τ_i - window, τ_i + window]`, plot the session trace
     restricted to that window, a dot at `(τ_i, cap(τ_i))`, and lick markers
     (from `lick_indices`/`lick_times`) whose times fall in the window.
   - Uses blitting where practical.
   - `FuncAnimation(..., frames=n_frames).save(out, writer=ffmpeg, fps=fps)`.

### Data flow

```
raw h5 ─┬─► video-sync metadata (frame_index, pts, start_time, t0_raw)
        └─► filter_data (temp h5) ─► session cap_data / time_data / lick_indices
pts.txt ─► video_base
                    │
            video_sec(τ) mapper
                    │
 for each output frame i:  τ_i ─► video_sec ─► FrameGrabber ─► left panel
                          τ_i ─► trace window + dot + markers ─► right panel
                    │
            FuncAnimation.save ─► out.mp4
```

## Error handling

- No sensor with `video_filename` → error, name the h5.
- Resolved video/pts file missing → error with the resolved path.
- `--start`/`--end` outside session or inverted → error with the valid range.
- Video animal not found via layout (sensor not in layout, or `filter_data`
  produced no licks) → proceed with trace + empty markers; warn.
- ffmpeg not on PATH → error early with install hint.

## Testing

- **Sync-map unit test**: known `video_frame_index`, `video_pts`, `pts[]`,
  `start_time`, `t0_raw` → assert `video_sec(0)` equals the expected video
  second; assert linearity and `--sync-offset` shift.
- **Frame-count test**: for a small window and fps, assert the animation emits
  `round((end-start)*fps)` frames.
- **Marker-window test**: given synthetic `lick_times`, assert only in-window
  markers are selected for a sample `τ`.
- **Smoke test**: run end-to-end on a ~5 s window of the reference recording,
  assert a non-empty mp4 is written with the expected duration (±1 frame).

## Out of scope / non-goals

- Full-session render optimization.
- Audio.
- Multiple sensors in one composite (only one has video metadata by design).
- Threshold line / running lick counter (not requested).
- PTS-vs-unix clock-drift correction beyond the constant `--sync-offset`.

## Dependencies

`h5py`, `numpy`, `pandas`, `matplotlib` (ffmpeg writer), `opencv-python` (cv2),
`data_analysis.py` (import `filter_data`), system `ffmpeg` on PATH.
