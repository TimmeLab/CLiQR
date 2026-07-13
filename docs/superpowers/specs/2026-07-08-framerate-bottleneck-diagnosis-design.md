# Frame-Rate Bottleneck Diagnosis & Fix — Design

**Date:** 2026-07-08
**Status:** Resolved — see Results below

## Problem

Pi camera pipeline requests **120 fps** (`pi/camera_backend.py`: `TARGET_FPS=120`,
`FrameDurationLimits` pinned to 8333 µs), but a real 64.45 s recording
(`Lickometry Data/raw_data_2026-07-08_13-50-38`) delivered only **99.9 fps
effective** — 6440 frames where ~7735 were expected, ~1295 dropped (~20%).

### Evidence it's dropped frames, not exposure extension

Per-frame SensorTimestamp intervals from the `.txt`:

| Metric | Value |
|---|---|
| Median interval | 8.325 ms → 120.1 fps |
| Mean interval | 10.01 ms → 99.9 fps |
| Gaps at ~2×/3×/4×/5× target | 524 / 212 / 78 / 24 |
| Max interval | 49.96 ms (4-frame stall) |

Gaps cluster at **clean integer multiples** of 8.33 ms. Low-light exposure
extension would smear intervals to non-integer values; integer multiples mean
the sensor captures on cadence and individual frames are dropped downstream.

### Leading hypothesis

Raspberry Pi 5 (BCM2712) has **no hardware H.264 encoder**. `H264Encoder()`
therefore runs **software x264** — CPU-bound. Encoding 1536×864 @ 120 fps in
software is the suspected wall. Secondary suspect: the timestamp `.txt` write
happens inside `pre_callback` (per frame, on the camera thread).

## Goal

Maximize **stable, zero-drop** frame rate. Higher fps is better, but a rock-solid
lower rate beats an inconsistent 120. Whatever sustainable rate we find becomes
the new pinned `TARGET_FPS`.

Lick science needs ≥ ~100 fps (10 ms bins) at most — licks are ~5–8 Hz, contact
30–70 ms — so smooth-and-lossless is the real target, not 120 specifically.

## Approach: Isolation A/B testing + CPU snapshot

Run short (~30 s) test recordings, changing **one variable at a time**. After
each, compute drop rate from the `.txt` (median interval → instantaneous fps;
frames/duration → effective fps; count gaps > 1.5× target). Stop at the first
change that sustains a clean high rate. Confirm mechanism with one per-thread
CPU capture during a dropping run.

Rejected alternatives:
- **Pure live instrumentation** — correlational only; shows CPU pegged but not
  that removing it fixes drops.
- **Combine everything upfront** — more steps than needed; ladder stops early.

## Experiment ladder

Claude has no Pi access; the user runs each command and returns the `.txt` (and
any console output) for analysis.

0. **Confirm environment** — Pi model, encoder module, core count:
   ```bash
   cat /proc/device-tree/model; echo
   vcgencmd version 2>/dev/null | head -1
   python3 -c "from picamera2.encoders import H264Encoder; print(H264Encoder.__module__)"
   nproc
   ```
1. **Baseline** — current config; confirm ~100 fps / ~20% drops is repeatable.
2. **Encoder off** — capture + timestamps only, no `H264Encoder`.
   - Clean 120 → **encoder is the bottleneck** (expected on Pi 5).
   - Still dropping → bottleneck is upstream (CSI / callback / memory bandwidth).
3. **CPU snapshot** — during a dropping baseline run, capture per-thread load
   (`top -H -b -n 3`) to confirm x264 saturation.
4. **If encoder-bound, walk down cost** until highest fps sustains zero drops:
   lower H.264 bitrate/preset → lower resolution (1536×864 → 1280×720) → MJPEG.
5. **Callback check** — move `.txt` writes out of `pre_callback` (buffer in RAM,
   flush on stop); re-measure to rule the callback in or out.

## Candidate fixes (selected by findings)

- Lower H.264 bitrate / faster x264 preset.
- Drop main-stream resolution.
- Record raw/YUV to fast storage, encode offline post-session.
- Buffer timestamp writes, flush on `stop_session`.
- Re-pin `TARGET_FPS` to the measured sustainable rate.

## Success criteria

A recording configuration with **zero (or <1%) dropped frames** at the highest
fps the Pi 5 pipeline sustains, with `TARGET_FPS` set to match and the mechanism
understood (not just empirically tuned).

## Results

Ran the ladder as isolated 30–60 s recordings on the Pi (Claude analyzed each
`.txt`):

- **Step 0** — Confirmed Pi 5, 4 cores, `libav_h264_encoder` (software H.264, no
  hardware encode).
- **Step 2 (encoder off)** — 120.12 fps, **0% drops**, max interval 8.33 ms.
  Capture/CSI/callback sustain 120 fps perfectly. Every drop dies at the encoder.
- **Step 3 (CPU)** — Encoder already multithreaded across ~3.4 of 4 cores, one
  thread pegged at 100%. Machine is CPU-saturated by encode; "add threads" is not
  a lever. Not memory-bound (214 MB RES, 14 GB free).
- **Step 4 (resolution sweep, encoder on)** — 1536×864: 9.1% drops; 1280×720:
  0.2%; 1024×576: 0%. Drops scale with pixel count, as expected for x264.
- **Bitrate sweep at 1280×720, 60 s** — 3 Mbps: 0%; 6 Mbps: 0.29%; libav default:
  0.79% with a 383 ms stall and an ffmpeg `Thread message queue blocking
  (thread_queue_size 64)` warning — the mux input queue overflowing when encode
  can't keep up. Lower bitrate drains the queue.

**Root cause:** software H.264 (Pi 5 has no hardware encoder) at full
1536×864/120 saturates the CPU; the ffmpeg mux queue backs up and drops frames
in bursts.

**Fix applied** (`pi/camera_backend.py`): encode a downscaled **1280×720** main
stream at a **3 Mbps** cap, keeping the 1536×864 raw stream to pin the sensor's
fast 120 fps mode. Capture stays native 1536×864/120; only the recorded video is
downscaled.

**Validation on the real pipeline** (full concurrent load — MPR121 racks,
network, server — 70 s, `raw_data_2026-07-09_09-20-33`):

| Metric | Before (1536×864, default br) | After (1280×720, 3 Mbps) |
|---|---|---|
| Effective fps | 104.4 | **119.7** |
| Dropped frames | 13–20% | **0.24%** (28 / 8364) |
| Worst stall | 1432 ms | **25 ms** (3 frames) |

Success criterion met: <1% drops at a stable 120 fps, worst-case gap 25 ms.
The camera server must be **restarted** after deploying config changes — a
running server holds the old `camera_backend.py` in memory (this bit us once
during validation).

### Remaining levers (unused; only if drops reappear under harsher conditions)

- Drop bitrate to 2 Mbps.
- Go 1024×576 (was rock-solid 0% in the sweep).
- Set the encoder-native `YUV420` main format to skip XBGR8888→YUV420 conversion.
