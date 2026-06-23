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
python -m pi.pi_camera_server --port 8770 --output-dir ~/cliqr_clips
```

The server pre-rolls on `START_SESSION`, bookmarks on `BOOKMARK`, finalizes the
MP4 + `.txt` on `STOP_SESSION`, and serves files on `GET_FILE`.

## Desktop usage

1. In the Recording GUI, open the **Video Capture (Pi Camera)** card.
2. Toggle **Enable concurrent video**.
3. Enter the Pi host/IP and port (default 8770).
4. Choose the **camera sensor** — the sensor whose Start button bookmarks video.
5. Click **Test connection** (expects "✓ Connected").
6. Start the session, then click that sensor's Start when the sipper goes in.
7. On Stop, the MP4 + `.txt` are copied into the session output directory.

## Output

- `<session>.mp4` — H.264 video.
- `<session>.txt` — per-frame `SensorTimestamp` (numpy.loadtxt-compatible),
  consumed by `false_positive_analysis.load_frame_offsets()`.
- HDF5 datasets per camera sensor: `video_frame_index`, `video_pts`,
  `video_filename` (cycle-suffixed like `start_time`).

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

## Troubleshooting

- **"✗ Not reachable":** verify the Pi server is running, host/IP is correct,
  and the LAN allows the port. The session still records capacitively.
- **No video files copied:** they remain on the Pi under `--output-dir`; copy
  manually (`scp`).
