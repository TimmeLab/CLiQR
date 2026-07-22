# Concurrent Video Capture (Pi Camera)

CLiQR can record video from a Raspberry Pi 5 + Pi Camera 3 in sync with
capacitive recording. When the operator clicks a designated sensor's **Start**
button (sipper inserted), the desktop GUI bookmarks the current video frame so
capacitive data and video share a common time reference.

## Topology

- One Pi 5 + Camera 3 films one cage.
- Desktop GUI talks to the Pi over **TCP/LAN** (one link for trigger, ack, and
  file copy). No USB serial/gadget is used — Pi 5 does not support USB device
  mode.
- The camera is **optional and non-blocking**: if the Pi is unreachable,
  capacitive recording proceeds normally.

## Timing model: pre-roll + bookmark

The camera starts recording at **global session Start** and logs every frame's
`SensorTimestamp` to a `.txt` file (one nanosecond value per line). The
per-sensor Start click sends a `BOOKMARK`; the Pi replies with the current
frame index and PTS, stored in the HDF5 as `video_frame_index` / `video_pts` /
`video_filename`. Sync is bounded by one frame interval (~33 ms @ 30 fps), with
no camera-warmup slop.

## Pi setup (one-time)

1. Raspberry Pi OS (Bookworm or later) on the Pi 5.
2. Install picamera2 (preinstalled on recent Pi OS; otherwise
   `sudo apt install -y python3-picamera2`).
3. Copy this repository (or at least the `pi/` and `video/` packages) to the Pi.
4. Ensure the Pi and desktop are on the same LAN; note the Pi's hostname/IP.

## Running the Pi server

```bash
./pi/run_server.sh --port 8770 --output-dir ~/cliqr_clips
```

The wrapper redirects stdout/stderr to `~/cliqr_camera_server.log` (previous run
kept as `.log.1`; override with `CLIQR_LOG`).

**Do not run `python -m pi.pi_camera_server` directly in an interactive
terminal.** ffmpeg is a child process that inherits those handles. If the
tty/ssh consumer stops draining its output, ffmpeg blocks on the stderr write,
stops reading its stdin pipe, and the back-pressure reaches picamera2's request
loop and stalls the camera. That is what silently ended the 2026-07-21
recording 44 min into a 2 h 19 min session.

The server pre-rolls on `START_SESSION`, bookmarks on `BOOKMARK`, finalizes the
MP4 + `.txt` on `STOP_SESSION`, and serves files on `GET_FILE`.

### Stall watchdog

A watchdog thread polls the frame counter while a session records. If no frame
arrives for `STALL_TIMEOUT_S` (3 s, ~360 frames at 120 fps), it stops the
segment, re-acquires the camera, and resumes into `<session>_part2.mp4` /
`.txt`, then `_part3`, and so on. A stall therefore costs a few seconds of
video instead of the rest of the run, and never fails the session — the
capacitance trace is what must survive.

`STOP_SESSION` reports every stall in its reply, the GUI logs each one and shows
a persistent warning, and `BOOKMARK` returns `frames_stale_s` so a frozen camera
is visible *during* the session rather than the next day.

Each segment carries its own `.txt` sidecar whose SensorTimestamps are absolute
Pi boot-clock nanoseconds, so later segments still align to the capacitance
trace through the original Start bookmark — they need no bookmark of their own.

The watchdog also restarts a segment when ffmpeg *exits* mid-session (a full
disk kills the muxer, not the camera — frames keep filling the sidecar while
the mp4 is silently truncated, which frame-staleness alone would never catch).

## Disk budget on the Pi

Video is `-c:v copy` from a 3 Mb/s encoder, so on-disk size is set by the
bitrate, not the container:

| | |
|---|---|
| Video | **1.38 GB/h** (measured: 1.01 GB / 2642 s) |
| `.txt` sidecar | ~7 MB/h (120 fps × 17 B/line) |
| 2 h 19 m session | 3.19 GB + 17 MB |
| 3 h session | 4.14 GB + 22 MB |

`MIN_FREE_BYTES` is 5 GB, reclaimed at session start by deleting the oldest
recordings (never the current session's). That covers a ~3 h run with little to
spare — check the Pi's free space before anything longer.

Reclaim runs only at session start and after stop, so **nothing frees space
mid-run**. Three guards bound what a bad session can consume:

- `MAX_SEGMENTS` (10) caps watchdog restarts. Past it the watchdog stops
  restarting and the session continues without video, rather than producing
  hundreds of files the desktop must fetch one at a time.
- `FFMPEG_LOG_MAX_BYTES` (2 MB) caps each `.ffmpeg.log`. A muxer stuck
  rejecting packets writes ~240 lines/s (~170 MB over a full session); past the
  cap the log is truncated and the truncation is reported.
- Below 1 GB free while recording, the Pi logs a warning every 60 s and reports
  `low_disk_during_run` in the STOP_SESSION reply; the GUI surfaces it. It does
  not delete anything mid-run — the desktop may be fetching an old file.

A `.ffmpeg.log` is copied back to the desktop only if it is non-empty, so a
healthy run still transfers exactly the two files it always did.

## Desktop usage

1. In the Recording GUI, open the **Video Capture (Pi Camera)** card.
2. Toggle **Enable concurrent video**.
3. Enter the Pi host/IP and port (default 8770).
4. Choose the **camera sensor** — the sensor whose Start button bookmarks video.
5. Click **Test connection** (expects "✓ Connected").
6. Start the session, then click that sensor's Start when the sipper goes in.
7. On Stop, the MP4 + `.txt` are copied into the session output directory.

## Output

- `<session>.mp4` — H.264 video (plus `<session>_part2.mp4` … if the watchdog
  restarted recording mid-session).
- `<session>.ffmpeg.log` — muxer stderr for that segment; normally empty.
  Deleted along with its `.mp4` by the Pi's disk reclaim.
- `<session>.txt` — **Pending format change:** Currently writes raw absolute `SensorTimestamp` values (nanoseconds, one per line, no header). For compatibility with `false_positive_analysis.load_frame_offsets()`, this must be changed to: 2-line header + per-frame offsets in relative nanoseconds (offset = `SensorTimestamp - first_frame_SensorTimestamp`). This change is pending validation on the Pi and will be implemented as part of on-device backend improvements.
- HDF5 datasets per camera sensor: `video_frame_index`, `video_pts`,
  `video_filename` (cycle-suffixed like `start_time`). **Note:** `video_pts` is currently reported as absolute seconds; it must be changed to relative seconds (seconds since video start) for consistency with the planned `.txt` format.

## Analysis

Use `false_positive_analysis.alignment_from_bookmark(start_time_abs, video_pts)`
to map video PTS to Unix time directly from the bookmark — no sipper-step
detection needed for recordings made this way. Legacy data (e.g. ACG-26-3)
still uses `detect_sipper_step` / `establish_alignment`.

### Analysis: synced video

`video/sync_video.py` turns a recorded session into review media:

- `make_sync_video_from_hdf5(raw_h5, sensor_id, video_path, frame_offsets_path, output_path, lick_times=...)`
  trims the video to the sensor's `start_time`/`stop_time` window (via the
  bookmark) and renders an MP4 with the video on the left and the capacitive
  trace on the right. The trace is zoomed to ±1 s around the current time, with
  the current time fixed at center and the window sliding; detected licks are
  marked in red. Output frame rate matches the source video.
- `trim_video(...)` alone writes just the trimmed recording-window MP4.

Call it from `DataAnalysis.ipynb` after lick detection, passing the absolute
lick times for the camera sensor.

## Cropping and rendering a sync video

Two steps. Crop once per recording, then render as many clips as you like.

1. **Crop** — trims the video to approximately the capacitance-recording window
   (keeping a few seconds of lead-in; frames are timed by PTS downstream, not by
   the trim) and crops it to a square you position by hand:

       python crop_video.py --h5 "Lickometry Data/<animal>/raw_data_<stamp>.h5"

   A window opens on a frame from the middle of the recording. Drag the green
   box over the sipper and press **Crop**. Writes `raw_data_<stamp>_cropped.mp4`
   next to the recording. `--size` changes the box (default 360x360); `--force`
   overwrites an existing crop.

2. **Render** — builds the side-by-side video + trace clip:

       python make_sync_video.py --h5 "Lickometry Data/<animal>/raw_data_<stamp>.h5" \
         --layout "Lickometry Data/<animal>/layout.csv" \
         --start 120 --end 130 --out clip.mp4

   It picks up `_cropped.mp4` automatically. If you skip step 1 it falls back to
   the uncropped recording, prints a note, and renders a full-frame video panel.

The cropped file keeps the original video's presentation timestamps, so the
sync anchor is identical either way — cropping changes framing only, never
alignment.

## Known limitations / pending work

- **Frame-offset `.txt` format and `video_pts` base:** The current backend writes absolute `SensorTimestamp` nanoseconds (no header) and reports absolute `video_pts` seconds. These must be changed to relative values (relative nanoseconds and relative seconds, respectively) to establish consistency with `false_positive_analysis.load_frame_offsets()` and `alignment_from_bookmark()`. This change is deferred pending on-device testing and validation on the Pi.
- **picamera2 backend:** Not yet hardware-validated on production Pi systems. Initial testing on development hardware only.

## Troubleshooting

- **"✗ Not reachable":** verify the Pi server is running, host/IP is correct,
  and the LAN allows the port. The session still records capacitively.
- **No video files copied:** they remain on the Pi under `--output-dir`; copy
  manually (`scp`).
