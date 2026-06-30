# Concurrent Video Capture — Design Spec

**Date:** 2026-06-23
**Branch:** `concurrent-video`
**Status:** Approved design, pending implementation plan

## Goal

Add concurrent video capture to the CLiQR capacitive lickometry system. A Raspberry Pi 5 with a Pi Camera 3 films one cage. When the operator clicks a sensor's **Start** button (sipper inserted in cage), the desktop GUI both timestamps the start of the capacitive recording and bookmarks the corresponding video frame, so capacitive data and video share a frame-accurate common time reference.

This replaces the current post-hoc sipper-step alignment used by the false-positive pipeline with a direct, tightly-coupled bookmark established at trigger time.

## Decisions (locked)

| Topic | Decision |
|---|---|
| Camera-to-cage topology | One camera, one cage |
| Pi hardware | Raspberry Pi 5 + Pi Camera 3, picamera2 H.264 encoder |
| Trigger source | Per-sensor **Start** button (`sensor_card.start_sensor`) |
| Camera-sensor mapping | Operator selects the "camera sensor" in the GUI; only that sensor's Start bookmarks |
| Control transport | TCP socket over LAN (one link for trigger + ack + file copy) |
| Camera timing model | **Pre-roll + bookmark** — camera rolls from global session Start; per-sensor Start bookmarks current frame index + PTS |
| Sync method | Ack-based bookmark; **sipper-step alignment dropped** |
| Video file transport | Auto-copy MP4 + frame-offset `.txt` to desktop after Stop, over LAN |
| Video format | MP4 + frame-offset `.txt` (same format `false_positive_analysis.load_frame_offsets()` already parses) |
| Camera dependency | Optional / non-blocking — capacitive recording never blocked by camera failure |

### Why Pi 5 + TCP (not USB gadget/serial)

Pi 5's USB-C port is power-only and its USB ports route through the RP1 southbridge as host-only. USB device/gadget mode (`dwc2` → `ttyGS0` serial gadget or CDC-ECM ethernet gadget) is unsupported on Pi 5. Therefore neither USB-ethernet-gadget nor USB-serial-gadget is available. A LAN TCP socket is Pi-5-native and carries control + file copy on one link. The ack mechanism (Pi reports the exact frame index/PTS at trigger receipt) makes transit latency irrelevant to sync, removing the only reason to prefer serial.

### Why pre-roll + bookmark (not start-on-click)

picamera2 has ~hundreds-of-ms warmup to first frame. Starting the camera on the click would fold that variable warmup into the sync error. Pre-rolling (encoder running from global session Start, per-frame PTS logged continuously) means the per-sensor click only *bookmarks* an already-rolling frame — sync error bounded by one frame interval (~33 ms @ 30 fps), with no warmup slop. This is what justifies dropping the post-hoc sipper-step alignment.

## Architecture

### Component 1 — Pi-side server: `pi_camera_server.py` (new, runs on the Pi)

TCP server with a line-delimited JSON protocol. Wraps picamera2 with an H.264 encoder; uses `start_recording(encoder, output, pts="<name>.txt")` to write the MP4 and the frame-offset `.txt` simultaneously.

Protocol commands:

| Command | Action | Ack payload |
|---|---|---|
| `PING` | health / connection test | `{ok: true, status}` |
| `START_SESSION {name}` | start encoder + PTS logging (begin pre-roll) | `{ok, video_filename}` |
| `BOOKMARK {sensor_id}` | snapshot current frame index + PTS at receipt | `{ok, frame_index, pts, pi_monotonic}` |
| `STOP_SESSION` | finalize MP4 + `.txt` | `{ok, files: [{name, size}]}` |
| `GET_FILE {name}` | stream a file's bytes to the desktop | file bytes |

Notes:
- Frame-offset `.txt` written via picamera2's `pts=` argument; format must match what `false_positive_analysis.load_frame_offsets()` expects (numpy-loadable frame timestamps).
- Server holds at most one active session. `BOOKMARK` before `START_SESSION` returns an error ack.
- `GET_FILE` may stream over the same TCP protocol, or implementation may shell out to `scp`/`rsync`; the client interface (`fetch_files`) hides this.

### Component 2 — Desktop client: `hardware/pi_camera.py` (new)

`PiCameraClient` class. Methods:
- `ping() -> bool`
- `start_session(name) -> dict` (returns video filename / ack)
- `bookmark(sensor_id) -> dict` (returns `frame_index`, `pts`, `pi_monotonic`)
- `stop_session() -> dict` (returns file list)
- `fetch_files(dest_dir) -> list[Path]`

All methods use short TCP timeouts and **never raise into the recording loop** — failures are caught, logged via the GUI log callback, and return a sentinel/None. The camera is auxiliary.

### Component 3 — Mock: `hardware/pi_camera_mock.py` (new)

Loopback fake server / fake client implementing the same interface, so `recording_gui_mock.py` and the desktop test suite run with no physical Pi. Returns deterministic frame indices/PTS for assertions.

### Component 4 — State + UI

`utils/state.py` — new reactive values:
- `camera_enabled: bool`
- `camera_host: str`
- `camera_port: int`
- `camera_sensor_id: int | None`

`components/camera_controls.py` (new) — a card with:
- Host / port input fields
- Enable toggle
- "Camera sensor" dropdown (1–24)
- "Test connection" button (issues `PING`, shows live status)

### Integration hook points (all guarded by `camera_enabled`)

1. `components/session_controls.py::start_recording()` — after creating the recorder, call `client.start_session()` to begin pre-roll. Log ack; on failure, log and continue.
2. `components/sensor_card.py::start_sensor()` — **if `sensor_id == state.camera_sensor_id`**, call `client.bookmark(sensor_id)`; stash returned `frame_index` / `pts` for the HDF5 write.
3. `components/session_controls.py::stop_recording()` — call `client.stop_session()` then `client.fetch_files(output_dir)`; write video metadata to the HDF5 file. Do not delete remote files until copy confirmed.

### HDF5 additions

In the camera-sensor's group, parallel to the existing `start_time` cycle numbering (`start_time`, `start_time1`, …):
- `video_frame_index` — frame index current at bookmark
- `video_pts` — PTS (seconds) at bookmark
- `video_filename` — name of the copied MP4

These provide the capacitive↔video alignment directly, removing the need for `detect_sipper_step()` / step-based `establish_alignment()`.

### Downstream: false-positive pipeline

`false_positive_analysis.establish_alignment()` gains a **direct-bookmark path**: when `video_frame_index` / `video_pts` exist in the HDF5, anchor the video clock to the HDF5 clock using the bookmark instead of sipper-step detection. Sipper-step code remains available for legacy data (e.g. ACG-26-3) but is no longer the primary path for new recordings.

## Error Handling

- Camera fully optional: Pi unreachable / any command failure → log and continue capacitive recording uninterrupted.
- TCP timeouts on every client call.
- File-copy failure → MP4 stays on the Pi; log its remote path for manual retrieval.
- Connection verified via `PING` before a session (and on demand via the Test button).
- `BOOKMARK` without an active session → logged error, no crash.

## Testing (TDD)

- Protocol encode/decode unit tests.
- `PiCameraClient` exercised against `pi_camera_mock` (start → bookmark → stop → fetch), asserting frame index/PTS plumbing and HDF5 writes.
- Mock-mode GUI smoke test: enable camera, run a mock session, verify video metadata datasets land in the HDF5.
- No physical Pi required for the desktop test suite.

## Documentation

- Commit the already-modernized `docs/CLAUDE.md` (uncommitted working-tree edit).
- New `docs/VIDEO_CAPTURE.md`: Pi OS + picamera2 setup, network configuration, running `pi_camera_server.py`, camera wiring, troubleshooting.
- Add the video-capture subsystem (4th subsystem) to `docs/CLAUDE.md` and `CURRENT_STATE.md`.

## Out of Scope (YAGNI)

- Multi-camera / multi-cage simultaneous capture.
- Live video preview/streaming in the GUI (serial/LAN bandwidth + scope).
- Re-recording or re-analysis of legacy ACG-26-3 data (keeps sipper-step path).
- Pi 4 / Zero 2 W USB-gadget single-cable variant (rejected in favor of Pi 5 + TCP).
