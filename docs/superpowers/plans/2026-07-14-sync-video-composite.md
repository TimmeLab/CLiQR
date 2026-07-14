# Sync Video Composite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `make_sync_video.py`, a CLI that renders a side-by-side demo video — mouse video on the left, the sensor's capacitance trace with a sliding window, centered dot, and lick markers on the right — for a configurable clip window of a concurrent video+capacitance recording.

**Architecture:** A single module of small pure functions (sync math, timing, marker selection) plus three stateful pieces (recording loader, video frame grabber, matplotlib animator) and a CLI. Lick detection is reused wholesale from `data_analysis.filter_data` (run into a temp h5, read back). Video↔capacitance alignment uses the per-frame PTS sidecar; one `FuncAnimation` timeline drives both panels so sync is automatic.

**Tech Stack:** Python 3, `h5py`, `numpy`, `pandas`, `matplotlib` (Agg + FFMpegWriter), `imageio` + `imageio-ffmpeg` (frame reading), `data_analysis.py`, system `ffmpeg`. Tests: `pytest`.

## Global Constraints

- Python venv: pyenv `cliqr-gui` (already active in the environment). All commands run with the project's `python`/`pytest`.
- No new third-party dependencies. Use only libraries already in `requirements.txt` (`imageio==2.36.1`, `imageio-ffmpeg==0.5.1`, `matplotlib`, `h5py`, `numpy`, `pandas`) — do **not** add `opencv-python`/`cv2` or `av`.
- `--start`/`--end` are **session-relative seconds** (0 = the sensor's `start_time` bookmark). This matches the zero-based `time_data` returned by `filter_data`.
- Lick markers use the primary `lick_indices`/`lick_times` from the `basic_threshold` algorithm, not the `optimal_*` variants.
- Detection is reused via `data_analysis.filter_data`; do not reimplement it.
- Reference recording for integration/smoke tests (guard every test that touches it with skip-if-missing):
  - h5: `Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.h5`
  - video: `Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.mp4`
  - pts txt: `Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.txt`
  - layout: `Lickometry Data/ACG-26-3/layout_w_controls.csv`
  - Known facts: video sensor = `board_FT232H0/sensor_1` → animal `ACG-26-3-1`; `video_frame_index=3839`; `video_pts≈443455.239288`; source video ~240 fps.

## File Structure

- Create: `make_sync_video.py` — the whole tool (pure functions, `Recording` dataclass + `load_recording`, `FrameGrabber`, `render_clip`, `main`/argparse).
- Create: `tests/test_make_sync_video.py` — unit tests (pure functions, always run) + integration/smoke tests (guarded by skip-if-reference-files-missing).

Single module is intentional: the pieces share the sync/timing vocabulary and are small. Tests live in the existing `tests/` dir (alongside `tests/test_disk_cleanup.py`).

---

### Task 1: Sync + timing pure functions

**Files:**
- Create: `make_sync_video.py`
- Test: `tests/test_make_sync_video.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `compute_video_base(pts_ns: np.ndarray, frame_index: int) -> float` — seconds; `(pts_ns[frame_index] - pts_ns[0]) / 1e9`.
  - `video_sec(tau, video_base, start_time, t0_raw, sync_offset=0.0) -> float` — video-file second for a session-relative time `tau`. Vectorizes over array `tau`.
  - `n_output_frames(start: float, end: float, fps: float) -> int` — `int(round((end - start) * fps))`.
  - `frame_times(start: float, end: float, fps: float) -> np.ndarray` — `start + np.arange(n_output_frames(...)) / fps`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_make_sync_video.py`:

```python
import numpy as np
import pytest

import make_sync_video as msv


def test_compute_video_base():
    pts_ns = np.array([1_000_000_000, 1_100_000_000, 1_250_000_000], dtype=np.int64)
    # (1_250_000_000 - 1_000_000_000) / 1e9 = 0.25
    assert msv.compute_video_base(pts_ns, 2) == pytest.approx(0.25)


def test_video_sec_at_session_start():
    # tau=0 -> video_base + (start_time - t0_raw)
    got = msv.video_sec(0.0, video_base=32.0, start_time=127.0, t0_raw=100.0)
    assert got == pytest.approx(32.0 + 27.0)


def test_video_sec_linear_and_offset():
    base = dict(video_base=32.0, start_time=127.0, t0_raw=100.0)
    assert msv.video_sec(5.0, **base) - msv.video_sec(0.0, **base) == pytest.approx(5.0)
    assert msv.video_sec(0.0, sync_offset=2.0, **base) - msv.video_sec(0.0, **base) == pytest.approx(2.0)


def test_video_sec_vectorized():
    taus = np.array([0.0, 1.0, 2.0])
    got = msv.video_sec(taus, video_base=32.0, start_time=127.0, t0_raw=100.0)
    assert np.allclose(got, np.array([59.0, 60.0, 61.0]))


def test_n_output_frames():
    assert msv.n_output_frames(10.0, 20.0, 30.0) == 300
    assert msv.n_output_frames(0.0, 5.0, 30.0) == 150


def test_frame_times():
    ft = msv.frame_times(10.0, 12.0, 30.0)
    assert len(ft) == 60
    assert ft[0] == pytest.approx(10.0)
    assert ft[1] == pytest.approx(10.0 + 1 / 30.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_make_sync_video.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'make_sync_video'`.

- [ ] **Step 3: Write minimal implementation**

Create `make_sync_video.py`:

```python
"""Render a side-by-side mouse-video + capacitance-trace demo clip.

Left panel: the mouse video. Right panel: the sensor's capacitance trace in a
sliding window with a centered dot marking the current time and markers on
detected licks. See docs/superpowers/specs/2026-07-14-sync-video-composite-design.md.
"""
import numpy as np


def compute_video_base(pts_ns, frame_index):
    """Seconds between the sync frame's PTS and the first frame's PTS."""
    pts_ns = np.asarray(pts_ns)
    return float(pts_ns[frame_index] - pts_ns[0]) / 1e9


def video_sec(tau, video_base, start_time, t0_raw, sync_offset=0.0):
    """Video-file second for session-relative time ``tau`` (0 = start_time)."""
    return video_base + (start_time + tau - t0_raw) + sync_offset


def n_output_frames(start, end, fps):
    return int(round((end - start) * fps))


def frame_times(start, end, fps):
    return start + np.arange(n_output_frames(start, end, fps)) / fps
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_make_sync_video.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add make_sync_video.py tests/test_make_sync_video.py
git commit -m "feat: sync and timing math for sync video composite"
```

---

### Task 2: Lick-marker window selection

**Files:**
- Modify: `make_sync_video.py`
- Test: `tests/test_make_sync_video.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `window_mask(times: np.ndarray, lo: float, hi: float) -> np.ndarray` — boolean mask for `lo <= times <= hi`.
  - `nearest_index(times: np.ndarray, tau: float) -> int` — index of the sample nearest `tau` (clamped to valid range). Used to place the dot at the current cap value.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_make_sync_video.py`:

```python
def test_window_mask():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mask = msv.window_mask(times, 1.0, 3.0)
    assert list(mask) == [False, True, True, True, False]


def test_nearest_index_interior_and_clamp():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    assert msv.nearest_index(times, 1.4) == 1
    assert msv.nearest_index(times, 1.6) == 2
    assert msv.nearest_index(times, -5.0) == 0
    assert msv.nearest_index(times, 99.0) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_make_sync_video.py -k "window_mask or nearest_index" -v`
Expected: FAIL — `AttributeError: module 'make_sync_video' has no attribute 'window_mask'`.

- [ ] **Step 3: Write minimal implementation**

Add to `make_sync_video.py`:

```python
def window_mask(times, lo, hi):
    times = np.asarray(times)
    return (times >= lo) & (times <= hi)


def nearest_index(times, tau):
    times = np.asarray(times)
    if times.size == 0:
        raise ValueError("times is empty")
    i = int(np.searchsorted(times, tau))
    if i <= 0:
        return 0
    if i >= times.size:
        return times.size - 1
    # searchsorted lands on the right neighbor; pick the closer of i-1, i
    return i if abs(times[i] - tau) < abs(times[i - 1] - tau) else i - 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_make_sync_video.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add make_sync_video.py tests/test_make_sync_video.py
git commit -m "feat: window mask and nearest-index helpers"
```

---

### Task 3: Recording loader (reuses filter_data)

**Files:**
- Modify: `make_sync_video.py`
- Test: `tests/test_make_sync_video.py`

**Interfaces:**
- Consumes: `compute_video_base`.
- Produces:
  - `Recording` dataclass with fields: `animal: str`, `sensor: int`, `cap: np.ndarray` (session, 0-based), `time: np.ndarray` (session-relative seconds), `lick_times: np.ndarray`, `lick_indices: np.ndarray`, `lick_vals: np.ndarray` (`cap[lick_indices]`), `start_time: float`, `t0_raw: float`, `video_base: float`, `video_path: str`, `session_duration: float`.
  - `find_video_sensor(raw_h5) -> tuple[str, str, int]` — returns `(board_id, sensor_group_name, sensor_number)` for the group containing `video_filename`; raises `ValueError` if none.
  - `load_recording(h5_path: str, layout_path: str, pts_txt_path: str, video_path: str) -> Recording`.

**Implementation notes (read before coding):**
- `find_video_sensor` iterates boards→sensor groups, returns the one with a `video_filename` dataset. Sensor number parsed from group name `sensor_<n>`.
- Animal id: `layout = pd.read_csv(layout_path, header=None, index_col=0)`; `animal = str(layout.loc[sensor_number].iloc[0])`.
- `t0_raw = raw[board][sensor]['time_data'][0]` (untrimmed) read **before** `filter_data`.
- `start_time`: mirror `filter_data`'s rule — highest-numbered `start_time{N}` key (or `start_time`); this is what the session zero-bases against. Read it directly from the video sensor group with a small helper `_resolve_start_time(group)`.
- Run `filter_data` into a temp h5 with `recording_length` set to `session_duration + 1.0` so the session is **not** trimmed below its full length (clips can live anywhere in the session). `session_duration = stop_time - start_time` using the matching stop key, falling back to `time_data[-1]` when absent (same fallback as `filter_data`).
- After `filter_data`, read the animal group's `cap_data`, `time_data`, `lick_indices`, `lick_times` from the temp h5. Compute `lick_vals = cap[lick_indices]`.
- `pts_ns = np.loadtxt(pts_txt_path, dtype=np.int64)`; `video_base = compute_video_base(pts_ns, int(video_frame_index))`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_make_sync_video.py`:

```python
import os

REC_DIR = "Lickometry Data/ACG-26-3"
H5 = os.path.join(REC_DIR, "raw_data_2026-07-13_11-59-47.h5")
VIDEO = os.path.join(REC_DIR, "raw_data_2026-07-13_11-59-47.mp4")
PTS = os.path.join(REC_DIR, "raw_data_2026-07-13_11-59-47.txt")
LAYOUT = os.path.join(REC_DIR, "layout_w_controls.csv")

needs_reference = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (H5, PTS, LAYOUT)),
    reason="reference recording files not present",
)


@needs_reference
def test_load_recording_reference():
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO)
    assert rec.animal == "ACG-26-3-1"
    assert rec.sensor == 1
    assert rec.cap.shape == rec.time.shape
    assert rec.cap.size > 1000
    # session-relative time starts at ~0 and increases
    assert rec.time[0] == pytest.approx(0.0, abs=1.0)
    assert rec.time[-1] > rec.time[0]
    # licks detected, indices valid, vals consistent
    assert rec.lick_indices.size == rec.lick_times.size
    assert rec.lick_indices.max() < rec.cap.size
    assert np.allclose(rec.lick_vals, rec.cap[rec.lick_indices])
    # sync fields populated; video_base ~ 32 s for this recording
    assert rec.video_base == pytest.approx(31.97, abs=0.1)
    assert rec.session_duration > 3600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_make_sync_video.py::test_load_recording_reference -v`
Expected: FAIL — `AttributeError: module 'make_sync_video' has no attribute 'load_recording'` (or SKIP if reference files absent — if skipped, note it and proceed; the unit tests still gate the module).

- [ ] **Step 3: Write minimal implementation**

Add to `make_sync_video.py` (add imports at top of file):

```python
import os
import re
import tempfile
from dataclasses import dataclass

import h5py
import pandas as pd

from data_analysis import filter_data

_BOARD_FOR_SENSOR = {
    **{s: "board_FT232H0" for s in (1, 2, 3, 7, 8, 9)},
    **{s: "board_FT232H1" for s in (4, 5, 6, 10, 11, 12)},
    **{s: "board_FT232H2" for s in (13, 14, 15, 19, 20, 21)},
    **{s: "board_FT232H3" for s in (16, 17, 18, 22, 23, 24)},
}


@dataclass
class Recording:
    animal: str
    sensor: int
    cap: np.ndarray
    time: np.ndarray
    lick_times: np.ndarray
    lick_indices: np.ndarray
    lick_vals: np.ndarray
    start_time: float
    t0_raw: float
    video_base: float
    video_path: str
    session_duration: float


def find_video_sensor(raw_h5):
    """Return (board_id, sensor_group_name, sensor_number) for the group that
    carries video sync metadata. Raises ValueError if none do."""
    for board_id, board in raw_h5.items():
        if not isinstance(board, h5py.Group):
            continue
        for sensor_name, group in board.items():
            if isinstance(group, h5py.Group) and "video_filename" in group:
                m = re.match(r"sensor_(\d+)$", sensor_name)
                if not m:
                    continue
                return board_id, sensor_name, int(m.group(1))
    raise ValueError("no sensor group with 'video_filename' found in h5")


def _resolve_start_stop(group):
    """Mirror filter_data: pick the highest-numbered start_time{N}/stop_time{N}
    pair (or the unnumbered pair), falling back to time_data ends."""
    pattern = re.compile(r"^start_time(\d+)?$")
    matches = {}
    for k in group.keys():
        m = pattern.match(k)
        if m:
            num = int(m.group(1)) if m.group(1) else -1
            matches[num] = k
    time_data = group["time_data"]
    if not matches:
        return float(time_data[0]), float(time_data[-1])
    num = max(matches)
    last_start = matches[num]
    start_time = float(group[last_start][()])
    stop_key = "stop" + last_start[5:]
    if stop_key in group:
        stop_time = float(group[stop_key][()])
    else:
        stop_time = float(time_data[-1])
    return start_time, stop_time


def load_recording(h5_path, layout_path, pts_txt_path, video_path):
    layout = pd.read_csv(layout_path, header=None, index_col=0)
    with h5py.File(h5_path, "r") as raw:
        board_id, sensor_name, sensor_number = find_video_sensor(raw)
        group = raw[board_id][sensor_name]
        t0_raw = float(group["time_data"][0])
        start_time, stop_time = _resolve_start_stop(group)
        video_frame_index = int(group["video_frame_index"][()])
    session_duration = stop_time - start_time

    animal = str(layout.loc[sensor_number].iloc[0])

    with tempfile.TemporaryDirectory() as td:
        filt_path = os.path.join(td, "filtered.h5")
        log_path = os.path.join(td, "filter.log")
        with h5py.File(h5_path, "r") as raw, h5py.File(filt_path, "w") as filt:
            filter_data(
                raw, filt, layout, log_path,
                algorithm="basic_threshold",
                recording_length=session_duration + 1.0,
            )
        with h5py.File(filt_path, "r") as filt:
            if animal not in filt:
                raise ValueError(f"filter_data produced no group for animal {animal!r}")
            g = filt[animal]
            cap = g["cap_data"][:]
            time = g["time_data"][:]
            lick_times = g["lick_times"][:] if "lick_times" in g else np.array([])
            lick_indices = (
                g["lick_indices"][:] if "lick_indices" in g else np.array([], dtype=int)
            )

    lick_indices = np.asarray(lick_indices, dtype=int)
    lick_vals = cap[lick_indices] if lick_indices.size else np.array([])

    pts_ns = np.loadtxt(pts_txt_path, dtype=np.int64)
    video_base = compute_video_base(pts_ns, video_frame_index)

    return Recording(
        animal=animal, sensor=sensor_number, cap=cap, time=time,
        lick_times=np.asarray(lick_times), lick_indices=lick_indices,
        lick_vals=lick_vals, start_time=start_time, t0_raw=t0_raw,
        video_base=video_base, video_path=video_path,
        session_duration=session_duration,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_make_sync_video.py::test_load_recording_reference -v`
Expected: PASS (or SKIP if reference files absent). If it runs, it may take ~10-60 s (filter_data processes all sensors).

- [ ] **Step 5: Commit**

```bash
git add make_sync_video.py tests/test_make_sync_video.py
git commit -m "feat: recording loader reusing filter_data detection"
```

---

### Task 4: Video frame grabber

**Files:**
- Modify: `make_sync_video.py`
- Test: `tests/test_make_sync_video.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `FrameGrabber(video_path: str, clip_start_sec: float)` with:
    - `.src_fps: float`
    - `.get(video_sec: float) -> np.ndarray | None` — RGB HxWx3 frame nearest `video_sec`; targets must be monotonically non-decreasing across calls. Returns the last decoded frame (or `None` before any frame).
    - `.close()`.

**Implementation notes:** open with `imageio.get_reader(video_path, "ffmpeg", input_params=["-ss", f"{clip_start_sec:.6f}"])` so decode fast-seeks to the clip; frame `k` after the seek is at ≈ `clip_start_sec + k / src_fps`. `get` advances an internal iterator to `target_k = round((video_sec - clip_start_sec) * src_fps)` (never backward) and returns the current frame.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_make_sync_video.py`:

```python
needs_video = pytest.mark.skipif(
    not os.path.exists(VIDEO), reason="reference video not present"
)


@needs_video
def test_frame_grabber_reads_rgb_and_advances():
    start = 60.0
    g = msv.FrameGrabber(VIDEO, clip_start_sec=start)
    try:
        assert g.src_fps > 1
        f0 = g.get(start)
        assert f0 is not None
        assert f0.ndim == 3 and f0.shape[2] == 3
        # advancing ~1s forward returns a frame of the same shape
        f1 = g.get(start + 1.0)
        assert f1.shape == f0.shape
    finally:
        g.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_make_sync_video.py::test_frame_grabber_reads_rgb_and_advances -v`
Expected: FAIL — `AttributeError: module 'make_sync_video' has no attribute 'FrameGrabber'` (or SKIP if video absent).

- [ ] **Step 3: Write minimal implementation**

Add to `make_sync_video.py` (add `import imageio` at top):

```python
class FrameGrabber:
    """Sequential RGB frame reader; fast-seeks to the clip start, then serves
    the frame nearest each requested (monotonically increasing) video second."""

    def __init__(self, video_path, clip_start_sec):
        self.clip_start = max(0.0, clip_start_sec)
        self._reader = imageio.get_reader(
            video_path, "ffmpeg",
            input_params=["-ss", f"{self.clip_start:.6f}"],
        )
        self.src_fps = float(self._reader.get_meta_data()["fps"])
        self._k = -1
        self._frame = None

    def get(self, video_sec):
        target_k = int(round((video_sec - self.clip_start) * self.src_fps))
        if target_k < 0:
            target_k = 0
        while self._k < target_k:
            try:
                self._frame = self._reader.get_next_data()
                self._k += 1
            except (IndexError, StopIteration):
                break
        return self._frame

    def close(self):
        self._reader.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_make_sync_video.py::test_frame_grabber_reads_rgb_and_advances -v`
Expected: PASS (or SKIP if video absent).

- [ ] **Step 5: Commit**

```bash
git add make_sync_video.py tests/test_make_sync_video.py
git commit -m "feat: imageio-based video frame grabber"
```

---

### Task 5: Composite animation renderer

**Files:**
- Modify: `make_sync_video.py`
- Test: `tests/test_make_sync_video.py`

**Interfaces:**
- Consumes: `Recording`, `frame_times`, `video_sec`, `window_mask`, `nearest_index`, `FrameGrabber`.
- Produces:
  - `render_clip(rec: Recording, start: float, end: float, out_path: str, fps: float = 30.0, window: float = 2.5, sync_offset: float = 0.0) -> None` — writes the composite mp4.

**Implementation notes:** force `matplotlib.use("Agg")` before importing pyplot (module-level, headless). Left axis: `imshow` the current frame, `axis("off")`. Right axis: line for the windowed trace, a red dot at the current cap value, a scatter for in-window lick markers; fixed y-limits from the whole-session cap range (padded); x-limits slide to `[tau - window, tau + window]` so the dot stays centered. `blit=False` (imshow + moving xlim make blitting unreliable; clip lengths are small). Save with `FFMpegWriter(fps=fps)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_make_sync_video.py`:

```python
import subprocess


def _video_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", path,
    ])
    return float(out.strip())


@needs_reference
@needs_video
def test_render_clip_smoke(tmp_path):
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO)
    out = str(tmp_path / "clip.mp4")
    start, end, fps = 120.0, 124.0, 30.0
    msv.render_clip(rec, start, end, out, fps=fps)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    # duration ~ (end - start), within a couple frames
    assert _video_duration(out) == pytest.approx(end - start, abs=2.0 / fps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_make_sync_video.py::test_render_clip_smoke -v`
Expected: FAIL — `AttributeError: module 'make_sync_video' has no attribute 'render_clip'` (or SKIP if reference files absent).

- [ ] **Step 3: Write minimal implementation**

At the **top** of `make_sync_video.py`, before other matplotlib use, add:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
```

Add the function:

```python
def render_clip(rec, start, end, out_path, fps=30.0, window=2.5, sync_offset=0.0):
    taus = frame_times(start, end, fps)
    if taus.size == 0:
        raise ValueError("empty clip: check --start/--end/--fps")

    clip_start_video_sec = video_sec(
        start, rec.video_base, rec.start_time, rec.t0_raw, sync_offset
    )
    grabber = FrameGrabber(rec.video_path, clip_start_video_sec)

    cap_min, cap_max = float(rec.cap.min()), float(rec.cap.max())
    pad = 0.05 * (cap_max - cap_min + 1.0)

    fig, (axv, axt) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(left=0.02, right=0.97, wspace=0.08)

    first_frame = grabber.get(clip_start_video_sec)
    im = axv.imshow(first_frame if first_frame is not None
                    else np.zeros((2, 2, 3), dtype=np.uint8))
    axv.axis("off")

    (line,) = axt.plot([], [], lw=0.8, color="tab:blue")
    (dot,) = axt.plot([], [], "o", color="red", markersize=6, zorder=5)
    markers = axt.scatter([], [], s=40, facecolors="none",
                          edgecolors="tab:orange", linewidths=1.5, zorder=4)
    axt.set_ylim(cap_min - pad, cap_max + pad)
    axt.set_xlabel("Time (s, session)")
    axt.set_ylabel("Capacitance")

    def update(i):
        tau = float(taus[i])
        frame = grabber.get(video_sec(
            tau, rec.video_base, rec.start_time, rec.t0_raw, sync_offset))
        if frame is not None:
            im.set_data(frame)

        lo, hi = tau - window, tau + window
        m = window_mask(rec.time, lo, hi)
        line.set_data(rec.time[m], rec.cap[m])
        axt.set_xlim(lo, hi)

        ci = nearest_index(rec.time, tau)
        dot.set_data([tau], [rec.cap[ci]])

        if rec.lick_times.size:
            lm = window_mask(rec.lick_times, lo, hi)
            markers.set_offsets(np.c_[rec.lick_times[lm], rec.lick_vals[lm]]
                                if np.any(lm) else np.empty((0, 2)))
        return im, line, dot, markers

    anim = FuncAnimation(fig, update, frames=len(taus), blit=False)
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=fps))
    finally:
        plt.close(fig)
        grabber.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_make_sync_video.py::test_render_clip_smoke -v`
Expected: PASS (or SKIP if reference files absent). Renders ~120 frames; a few seconds.

- [ ] **Step 5: Commit**

```bash
git add make_sync_video.py tests/test_make_sync_video.py
git commit -m "feat: composite trace+video animation renderer"
```

---

### Task 6: CLI (argparse, defaults, validation)

**Files:**
- Modify: `make_sync_video.py`
- Test: `tests/test_make_sync_video.py`

**Interfaces:**
- Consumes: `load_recording`, `render_clip`.
- Produces:
  - `resolve_paths(h5_path, video, pts_txt) -> tuple[str, str, str]` — fills `video`/`pts_txt` defaults from the h5's `video_filename` and directory when not given. Returns `(video_path, pts_txt_path, ...)`; see signature below.
  - `validate_window(start, end, session_duration) -> None` — raises `ValueError` on `start >= end`, `start < 0`, or `end > session_duration`.
  - `build_arg_parser() -> argparse.ArgumentParser`.
  - `main(argv=None) -> int`.

**Implementation notes:**
- `resolve_paths(h5_path, video, pts_txt)`: if `video` is None, open the h5, read the video sensor's `video_filename` (bytes → str), resolve relative to `os.path.dirname(h5_path)`. If `pts_txt` is None, default to the video path with its extension replaced by `.txt`. Return `(video_path, pts_txt_path)`.
- CLI arguments: `--h5` (required), `--layout` (required), `--start` (float, required), `--end` (float, required), `--out` (required), `--video` (optional), `--pts-txt` (optional), `--fps` (default 30.0), `--window` (default 2.5), `--sync-offset` (default 0.0).
- `main`: resolve paths → `load_recording` → `validate_window(start, end, rec.session_duration)` → `render_clip`. Print a one-line summary (animal, sensor, clip window, out path). Check `ffmpeg` availability early via `shutil.which("ffmpeg")`; if missing, print an error and return 1.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_make_sync_video.py`:

```python
def test_validate_window_ok():
    msv.validate_window(10.0, 20.0, 100.0)  # no raise


@pytest.mark.parametrize("start,end,dur", [
    (20.0, 10.0, 100.0),   # inverted
    (-1.0, 10.0, 100.0),   # negative start
    (10.0, 200.0, 100.0),  # past session end
])
def test_validate_window_rejects(start, end, dur):
    with pytest.raises(ValueError):
        msv.validate_window(start, end, dur)


def test_build_arg_parser_parses_required():
    p = msv.build_arg_parser()
    args = p.parse_args([
        "--h5", "r.h5", "--layout", "l.csv",
        "--start", "5", "--end", "9", "--out", "o.mp4",
    ])
    assert args.h5 == "r.h5" and args.start == 5.0 and args.end == 9.0
    assert args.fps == 30.0 and args.window == 2.5 and args.sync_offset == 0.0


@needs_reference
def test_resolve_paths_defaults_from_h5():
    video, pts = msv.resolve_paths(H5, None, None)
    assert video.endswith("raw_data_2026-07-13_11-59-47.mp4")
    assert pts.endswith("raw_data_2026-07-13_11-59-47.txt")
    assert os.path.dirname(video) == os.path.dirname(H5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_make_sync_video.py -k "validate_window or arg_parser or resolve_paths" -v`
Expected: FAIL — missing `validate_window`/`build_arg_parser`/`resolve_paths`.

- [ ] **Step 3: Write minimal implementation**

Add to `make_sync_video.py` (add `import argparse`, `import shutil`, `import sys` at top):

```python
def resolve_paths(h5_path, video, pts_txt):
    if video is None:
        with h5py.File(h5_path, "r") as raw:
            _, sensor_name, _ = find_video_sensor(raw)
            board_id, _, _ = find_video_sensor(raw)
            fname = raw[board_id][sensor_name]["video_filename"][()]
        fname = fname.decode() if isinstance(fname, bytes) else str(fname)
        video = os.path.join(os.path.dirname(h5_path), fname)
    if pts_txt is None:
        pts_txt = os.path.splitext(video)[0] + ".txt"
    return video, pts_txt


def validate_window(start, end, session_duration):
    if start < 0:
        raise ValueError(f"--start must be >= 0 (got {start})")
    if end <= start:
        raise ValueError(f"--end ({end}) must be greater than --start ({start})")
    if end > session_duration:
        raise ValueError(
            f"--end ({end}) exceeds session duration ({session_duration:.1f} s)")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Render a side-by-side mouse-video + capacitance-trace clip.")
    p.add_argument("--h5", required=True, help="raw recording .h5")
    p.add_argument("--layout", required=True, help="sensor->animal layout csv")
    p.add_argument("--start", type=float, required=True,
                   help="clip start, seconds since the Start bookmark")
    p.add_argument("--end", type=float, required=True,
                   help="clip end, seconds since the Start bookmark")
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--video", default=None,
                   help="mouse video (default: from h5 video_filename)")
    p.add_argument("--pts-txt", dest="pts_txt", default=None,
                   help="per-frame PTS sidecar (default: video path with .txt)")
    p.add_argument("--fps", type=float, default=30.0, help="output fps (default 30)")
    p.add_argument("--window", type=float, default=2.5,
                   help="trace half-window seconds (default 2.5)")
    p.add_argument("--sync-offset", dest="sync_offset", type=float, default=0.0,
                   help="manual video/cap alignment nudge, seconds (default 0)")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH (needed to write the video)",
              file=sys.stderr)
        return 1
    video, pts_txt = resolve_paths(args.h5, args.video, args.pts_txt)
    rec = load_recording(args.h5, args.layout, pts_txt, video)
    validate_window(args.start, args.end, rec.session_duration)
    print(f"animal {rec.animal} (sensor {rec.sensor}); clip "
          f"[{args.start:.1f}, {args.end:.1f}] s -> {args.out}")
    render_clip(rec, args.start, args.end, args.out,
                fps=args.fps, window=args.window, sync_offset=args.sync_offset)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Then simplify the duplicated `find_video_sensor` call in `resolve_paths` (it was written twice above for clarity of intent — collapse to one call):

```python
def resolve_paths(h5_path, video, pts_txt):
    if video is None:
        with h5py.File(h5_path, "r") as raw:
            board_id, sensor_name, _ = find_video_sensor(raw)
            fname = raw[board_id][sensor_name]["video_filename"][()]
        fname = fname.decode() if isinstance(fname, bytes) else str(fname)
        video = os.path.join(os.path.dirname(h5_path), fname)
    if pts_txt is None:
        pts_txt = os.path.splitext(video)[0] + ".txt"
    return video, pts_txt
```

(Use this single-call version as the final `resolve_paths`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_make_sync_video.py -v`
Expected: PASS for all unit tests; reference-dependent tests PASS or SKIP.

- [ ] **Step 5: End-to-end CLI check + commit**

If the reference files are present, run the CLI on a short window and confirm an mp4 is produced:

```bash
python make_sync_video.py \
  --h5 "Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.h5" \
  --layout "Lickometry Data/ACG-26-3/layout_w_controls.csv" \
  --start 120 --end 126 --out /tmp/sync_clip_demo.mp4
```

Expected: prints the summary line and `done`; `/tmp/sync_clip_demo.mp4` exists. (If reference files are absent, skip this manual check.)

```bash
git add make_sync_video.py tests/test_make_sync_video.py
git commit -m "feat: CLI for sync video composite"
```

---

## Self-Review

**Spec coverage:**
- Configurable clip window, required `--start`/`--end` → Task 6 (args, `validate_window`). ✓
- Auto-detect video sensor → Task 3 (`find_video_sensor`). ✓
- Reuse `filter_data` detection, read back session arrays → Task 3. ✓
- Session-relative time reference; `t0_raw`, `start_time` → Tasks 1, 3. ✓
- Sync map with `video_base` + `--sync-offset` → Tasks 1 (`video_sec`), 6 (arg). ✓
- Left video panel via imageio sequential decode → Task 4 (`FrameGrabber`). ✓
- Right panel: sliding window, centered dot, lick markers → Task 5 (`render_clip`). ✓
- Error handling: no video sensor (Task 3 raises), missing ffmpeg (Task 6), bad window (Task 6), animal not found (Task 3 raises). ✓
- Tests: sync-map unit, frame-count (`n_output_frames`), marker-window (`window_mask`), smoke render → Tasks 1, 2, 5. ✓
- Default video/pts path resolution → Task 6 (`resolve_paths`). ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The one duplicated-call note in Task 6 Step 3 is resolved explicitly with a final single-call version. ✓

**Type consistency:** `Recording` fields defined in Task 3 are the exact names consumed in Task 5 (`rec.cap`, `rec.time`, `rec.lick_times`, `rec.lick_vals`, `rec.video_base`, `rec.start_time`, `rec.t0_raw`, `rec.video_path`, `rec.session_duration`). `video_sec` signature identical across Tasks 1, 5, 6. `FrameGrabber.get`/`.close`/`.src_fps` consistent Tasks 4, 5. `window_mask`/`nearest_index` signatures consistent Tasks 2, 5. ✓
