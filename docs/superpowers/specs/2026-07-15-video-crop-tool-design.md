# Standalone Video Crop Tool — Design

**Date:** 2026-07-15
**Status:** Approved (design), pending implementation plan

## Goal

A standalone tool that trims a Pi video recording to its capacitance-recording
window and crops it to a square region the user positions by hand. Produces a
`<video>_cropped.mp4` that `make_sync_video.py` then consumes directly.

Today `make_sync_video.py` trims and center-crops on every render. The crop
region is hard-coded to the frame center, which is rarely where the sipper is,
and the (expensive, re-encoding) trim+crop is repeated for every clip rendered
from the same recording. This design moves that work into a one-time
interactive step.

Reference recording:
`Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.{mp4,h5,txt}`
(1280x720, nominal 240 fps).

## Scope

- New root script `crop_video.py`: CLI + matplotlib GUI.
- New module `video/trimcrop.py`: shared pure functions and ffmpeg wrappers,
  extracted from `make_sync_video.py`.
- `make_sync_video.py` stops trimming and cropping; it consumes the cropped
  file and only cheaply subclips (stream copy) to its `--start`/`--end` window.

Out of scope: batch/headless cropping, persisting crop positions across
recordings, resizable crop boxes in the GUI, cropping non-square regions.

## Established facts

- Raw h5 layout: `board_FT232H{0..3}/sensor_{n}/`. Exactly one sensor carries
  `video_filename`, `video_frame_index`, `video_pts` — found by
  `find_video_sensor()`.
- Session window comes from that sensor's `start_time`/`stop_time`; numbered
  variants (`start_time1`, …) exist for multi-cycle recordings and the
  highest-numbered pair wins (`_resolve_start_stop()`, mirrors `filter_data`).
- The PTS sidecar `<video>.txt` holds one nanosecond timestamp per frame of the
  **original** video.
- `video_base = (pts[video_frame_index] - pts[0]) / 1e9` — seconds from the
  first frame to the bookmarked (sipper-insertion) frame.
- The bookmark round-trip lagged `start_time`, so the raw anchor runs early by
  `bookmark_latency(host_before, host_after, start_time)`; the corrected anchor
  is `video_base_eff = video_base - latency`. Older recordings lack the host
  bracket and get latency 0.
- ffmpeg's input `-ss` does not land frame-accurately on this footage. Every
  trim uses `-copyts` so output frames keep original PTS, and frames are timed
  by their real PTS (read back with ffprobe) — never by where the seek landed.

## Architecture

### `video/trimcrop.py` (new)

Pure functions and ffmpeg wrappers, no GUI, no argparse. Moved verbatim from
`make_sync_video.py` unless noted:

- `find_video_sensor(raw_h5)`
- `_resolve_start_stop(group)`
- `read_session_window(h5_path)` — returns `(start_time, stop_time)`.
  `read_session_duration()` in `make_sync_video.py` becomes a thin caller.
- `compute_video_base(pts_ns, frame_index)`
- `bookmark_latency(host_before, host_after, start_time)`
- `frame_session_times(pts_ns, video_base)`
- `compute_trim_frames(pts_ns, video_base, start, end)`
- `probe_frame_session_times(path, video_base)`
- `read_video_anchor(h5_path)` — new: reads `video_frame_index`, the host
  bracket, and start/stop in one open; returns everything needed to compute
  `video_base_eff`. Both scripts need this, neither should re-open the h5.

Changed signature:

- `trim_and_crop(video_path, start_sec, end_sec, out_path, crop_x, crop_y,
  size, seek_margin=5.0)` — **crop origin is now explicit**, replacing the
  centered `(iw-w)/2` expression. Filter becomes
  `crop={size}:{size}:{crop_x}:{crop_y}`. Re-encodes (libx264, veryfast,
  crf 18, yuv420p) because a filter is applied.

New:

- `probe_start_pts(path)` — new: the input's first presentation timestamp, via
  `ffprobe -show_entries format=start_time`.
- `subclip_copy(video_path, start_sec, end_sec, out_path, seek_margin=5.0)` —
  `ffmpeg -ss <coarse> -copyts -i <in> -to <end> -an -c copy <out>`. No filter,
  no re-encode, near-instant. Stream copy cuts at the keyframe at or before
  the seek target, so the clip may carry leading frames earlier than
  `start_sec`; that is harmless because consumers time frames by PTS and skip
  past them.

  **Input `-ss` is file-relative; `-to` with `-copyts` is absolute.** Verified
  experimentally on the reference recording: a cropped file whose PTS start at
  30 s yields zero frames for `-ss 35`, and seeks correctly for `-ss 5`, while
  `-to 40` cuts at absolute 40 s either way. So the seek must be

      coarse = max(0.0, start_sec - probe_start_pts(video_path) - seek_margin)

  `start_sec` is always an original-video-timeline second. The original video's
  `start_time` is 0.0, so this reduces to today's behavior there; the cropped
  file's `start_time` is the session start (~32 s), where the subtraction is
  what makes the seek land. `trim_and_crop` reads the original video only and
  keeps the plain `start_sec - seek_margin` seek.
- `clamp_origin(x, y, frame_w, frame_h, size)` — clamps a proposed crop origin
  so the box stays inside the frame, and rounds each coordinate down to an even
  number (yuv420p chroma subsampling requires even offsets). Raises `ValueError`
  if `size` exceeds either frame dimension. Pure; unit-tested.

### `crop_video.py` (new root script)

```
python crop_video.py --h5 <raw.h5> [--video X] [--pts-txt Y] [--size 360]
                     [--out Z] [--force]
```

Flow:

1. `shutil.which("ffmpeg")` check; error out early if missing.
2. Resolve video and PTS sidecar (see "Path resolution" below). `crop_video.py`
   always operates on the **original** video, never on an existing
   `_cropped.mp4`; if resolution lands on one, error out.
3. `read_video_anchor(h5)` → `video_base_eff = compute_video_base(pts,
   frame_index) - bookmark_latency(...)`.
4. `session_duration = stop_time - start_time`;
   `compute_trim_frames(pts, video_base_eff, 0.0, session_duration)` →
   `(start_frame, stop_frame)` → video-seconds
   `start_sec = (pts[start_frame]-pts[0])/1e9`, `end_sec` likewise `+ 0.3`
   (same tail margin `render_clip` uses today).
5. Preview frame: `imageio.get_reader(video).get_data(mid_frame)` where
   `mid_frame = (start_frame + stop_frame) // 2` — the midpoint of the **trim
   window**, not of the file, so the mouse is present and the sipper is in.
6. Launch GUI (below). Window closed without pressing Crop → exit 0, no output
   written, print `cancelled`.
7. On Crop: close the figure, then
   `trim_and_crop(video, start_sec, end_sec, out, x, y, size)`. Print the
   window, the crop origin, and the output path.

Default `--out` is `<video-base>_cropped.mp4` (matches the file already on disk
in `Lickometry Data/ACG-26-3/`). Refuse to overwrite an existing output unless
`--force`. The original video is never modified.

### GUI

Matplotlib only — no new dependencies, and matplotlib is already used
throughout the repo.

- `ax.imshow(frame)`; `Rectangle((x0, y0), size, size, fill=False, lw=2,
  edgecolor="lime")` starting centered.
- `button_press_event`: if the click is inside the rect, record the grab offset.
- `motion_notify_event`: while grabbed, move the rect to
  `clamp_origin(event.xdata - dx, event.ydata - dy, W, H, size)`.
- `button_release_event`: release the grab.
- `Button` widget labelled "Crop" in the lower right; sets a result flag and
  closes the figure.
- Axes title shows the live crop origin, e.g. `crop 360x360 @ (452, 180)`.

Drag only — the box is locked to `--size` (default 360). No resize handles:
a fixed size keeps output dimensions consistent across recordings and removes a
whole class of accidental-resize mistakes.

### `make_sync_video.py` (changed)

- Deletes its own copies of the functions now in `video/trimcrop.py` and
  imports them instead.
- `render_clip` calls `subclip_copy` where it called `trim_and_crop`. It no
  longer crops — whatever crop the input file has is what the panel shows.
- Drops `--crop-w` / `--crop-h`.
- `--intermediate` keeps its meaning (the subclip file, still kept).
- Everything downstream is unchanged: `probe_frame_session_times` reads the
  subclip's preserved PTS, `TrimmedFrameSource` matches frames by session time.

## Path resolution

Shared helper `resolve_paths(h5_path, video, pts_txt, prefer_cropped)`:

1. Base video name always comes from the h5's `video_filename`, joined to the
   h5's directory. Call this `<base>.mp4`.
2. **PTS sidecar is always `<base>.txt`** — derived from the original video
   name, never from the resolved video path. `--pts-txt` overrides.
3. Video path:
   - `--video X` → X verbatim.
   - `prefer_cropped=True` (make_sync_video) → `<base>_cropped.mp4` if it
     exists, else `<base>.mp4` plus a stderr note:
     `note: using uncropped video <path> (no _cropped.mp4; run crop_video.py first)`.
   - `prefer_cropped=False` (crop_video) → `<base>.mp4`.

The sidecar rule matters: today's `resolve_paths` derives it as
`splitext(video)[0] + ".txt"`, which would look for a nonexistent
`<base>_cropped.txt` the moment the cropped file is preferred. The cropped
video has no sidecar of its own and needs none — it carries original PTS via
`-copyts`, and `probe_frame_session_times` reads them back from the file.

The uncropped fallback renders correctly; the video panel is just the full
1280x720 frame instead of a 360x360 crop.

## Error handling

Matches the existing style: `video/trimcrop.py` raises `ValueError` /
`RuntimeError` with a specific message; `main()` catches
`(ValueError, FileNotFoundError, KeyError, OSError, RuntimeError)`, prints
`error: {e}` to stderr, returns 1.

Specific cases:

- ffmpeg not on PATH → early check in `main()`.
- No sensor with `video_filename` → `find_video_sensor` raises.
- No frames in the session window → `compute_trim_frames` raises.
- Output exists and no `--force` → `ValueError`.
- `--video` points at a `_cropped.mp4` → `ValueError` (double-cropping).
- ffmpeg non-zero exit → `RuntimeError` with the last 800 chars of stderr.

## Testing

New `tests/test_trimcrop.py`, pure functions only:

- `clamp_origin`: inside frame unchanged (modulo even rounding); each edge
  clamps; oversized `size` vs frame errors; odd inputs round down to even.
- `compute_trim_frames`: window edges inclusive; empty window raises.
- `_resolve_start_stop`: picks the highest-numbered pair; unnumbered pair; no
  pair falls back to `time_data` ends.
- `resolve_paths`: sidecar stays `<base>.txt` when the cropped video is
  preferred; `--video` override; `prefer_cropped` picks cropped when present
  and falls back when absent.
- `trim_and_crop` / `subclip_copy`: assert the constructed ffmpeg argv
  (crop filter string, `-copyts`, `-c copy` vs `-c:v libx264`) with
  `subprocess.run` patched. ffmpeg is not executed in tests.

`tests/test_make_sync_video.py` is updated for the moved functions and the
dropped `--crop-w`/`--crop-h` flags.

The GUI is deliberately thin (event handlers delegating to `clamp_origin`) and
is not unit-tested; it is verified by running the tool on the reference
recording.

## Verification

Run against `Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.h5`:

1. `python crop_video.py --h5 <ref>.h5` → drag box over the sipper → Crop.
   Confirm `<ref>_cropped.mp4` is 360x360 and its duration matches
   `stop_time - start_time`.
2. `python make_sync_video.py --h5 <ref>.h5 --layout <layout> --start 60
   --end 70 --out /tmp/clip.mp4` → confirm it picks the cropped file, the
   panel shows the chosen region, and licks still line up with the trace.
3. Move `<ref>_cropped.mp4` aside; rerun step 2 → confirm the fallback note
   prints and the clip renders with the full uncropped frame.
