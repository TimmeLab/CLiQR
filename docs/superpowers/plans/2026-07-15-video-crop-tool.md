# Video Crop Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `crop_video.py`, an interactive tool that trims a Pi recording to its capacitance window and crops it to a hand-positioned 360x360 square, and make `make_sync_video.py` consume its output instead of trimming and cropping on every render.

**Architecture:** Shared pure functions and ffmpeg wrappers move from `make_sync_video.py` into a new `video/trimcrop.py`. `crop_video.py` (root script) adds a matplotlib GUI over them and does the one-time, re-encoding trim+crop. `make_sync_video.py` keeps only a near-instant stream-copy subclip for its `--start`/`--end` window.

**Tech Stack:** Python 3.13, numpy, h5py, imageio, matplotlib (Agg for render, default backend for GUI), ffmpeg/ffprobe CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-07-15-video-crop-tool-design.md`

## Global Constraints

- Every ffmpeg trim uses `-copyts`. Frames are timed by their real PTS read back with ffprobe, never by where a seek landed.
- **Input `-ss` is file-relative; `-to` under `-copyts` is absolute.** Any seek on a file whose PTS may not start at 0 must subtract `probe_start_pts(path)`. Verified experimentally: see spec.
- `start_sec`/`end_sec` are always seconds on the **original video's** timeline.
- The corrected sync anchor is always `video_base_eff = compute_video_base(pts, frame_index) - bookmark_latency(...)`. Never use the raw `video_base` for trimming.
- The PTS sidecar is always `<original-video-base>.txt`. It is never derived from the resolved video path.
- yuv420p requires even crop offsets.
- Error style: raise `ValueError`/`RuntimeError` with a specific message; `main()` catches `(ValueError, FileNotFoundError, KeyError, OSError, RuntimeError)`, prints `error: {e}` to stderr, returns 1.
- Reference recording (tests skip if absent): `Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.{h5,mp4,txt}`, layout `layout_w_controls.csv`. 1280x720, ~120 fps real.
- Baseline before starting: `python -m pytest tests/test_make_sync_video.py -q` → 22 passed.

## File Structure

- **Create** `video/trimcrop.py` — h5 anchor reading, PTS/session-time math, ffmpeg wrappers, path resolution. No GUI, no argparse.
- **Create** `crop_video.py` — CLI + matplotlib GUI. Root script, matching `make_sync_video.py`'s placement.
- **Create** `tests/test_trimcrop.py` — pure functions and ffmpeg argv construction.
- **Create** `tests/test_crop_video.py` — CLI helpers.
- **Modify** `make_sync_video.py` — import from `video.trimcrop`; `render_clip` uses `subclip_copy`; drop `--crop-w`/`--crop-h`.
- **Modify** `tests/test_make_sync_video.py` — follow the moved functions and dropped flags.

`video/__init__.py` already exists, so `from video.trimcrop import ...` works from the repo root.

---

### Task 1: Extract shared math into `video/trimcrop.py`

Pure move. `make_sync_video.py` re-imports the names, so `msv.compute_video_base` etc. keep resolving and the existing 22 tests keep passing unchanged.

**Files:**
- Create: `video/trimcrop.py`
- Create: `tests/test_trimcrop.py`
- Modify: `make_sync_video.py:1-27` (imports), delete `:45-90` (`find_video_sensor`, `_resolve_start_stop`, `read_session_duration`), `:145-148` (`compute_video_base`), `:156-163` (`bookmark_latency`), `:192-206` (`frame_session_times`, `compute_trim_frames`), `:234-249` (`probe_frame_session_times`)

**Interfaces:**
- Consumes: nothing.
- Produces: `video.trimcrop` module exporting `find_video_sensor(raw_h5) -> (board_id, sensor_name, sensor_number)`, `_resolve_start_stop(group) -> (start_time, stop_time)`, `read_session_window(h5_path) -> (start_time, stop_time)`, `compute_video_base(pts_ns, frame_index) -> float`, `bookmark_latency(host_before, host_after, start_time) -> float`, `frame_session_times(pts_ns, video_base) -> ndarray`, `compute_trim_frames(pts_ns, video_base, start, end) -> (int, int)`, `probe_frame_session_times(path, video_base) -> ndarray`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trimcrop.py`:

```python
import numpy as np
import pytest
import h5py

from video import trimcrop as tc


def test_compute_video_base():
    pts_ns = np.array([1_000_000_000, 1_100_000_000, 1_250_000_000], dtype=np.int64)
    assert tc.compute_video_base(pts_ns, 2) == pytest.approx(0.25)


def test_bookmark_latency_from_bracket():
    assert tc.bookmark_latency(1000.0, 1000.2, 998.0) == pytest.approx(2.1)
    assert tc.bookmark_latency(999.9, 1000.1, 1000.0) == pytest.approx(0.0)


def test_bookmark_latency_missing_is_zero():
    assert tc.bookmark_latency(None, None, 1000.0) == 0.0
    assert tc.bookmark_latency(1000.0, None, 998.0) == 0.0


def test_frame_session_times_and_trim_frames():
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)  # 0.0 .. 1.0 s
    vb = tc.compute_video_base(pts_ns, 2)  # 0.2 s
    sess = tc.frame_session_times(pts_ns, vb)
    assert sess[2] == pytest.approx(0.0)
    assert sess[0] == pytest.approx(-0.2)
    sf, ef = tc.compute_trim_frames(pts_ns, vb, 0.0, 0.5)
    assert sf == 2 and ef == 7


def test_compute_trim_frames_window_edges_inclusive():
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    # video_base 0 -> session time == file time; [0.2, 0.5] covers frames 2..5
    sf, ef = tc.compute_trim_frames(pts_ns, 0.0, 0.2, 0.5)
    assert sf == 2 and ef == 5


def test_compute_trim_frames_empty_window_raises():
    pts_ns = (np.arange(0, 5) * 100_000_000).astype(np.int64)
    with pytest.raises(ValueError):
        tc.compute_trim_frames(pts_ns, 0.0, 100.0, 200.0)


def _write_sensor(path, groups):
    """groups: {sensor_name: {dataset: value}} under one board."""
    with h5py.File(path, "w") as f:
        for name, datasets in groups.items():
            g = f.create_group(f"board_FT232H0/{name}")
            for k, v in datasets.items():
                g[k] = v


def test_resolve_start_stop_picks_highest_numbered(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "start_time": 110.0, "stop_time": 120.0,
        "start_time1": 130.0, "stop_time1": 140.0,
    }})
    with h5py.File(p, "r") as f:
        assert tc._resolve_start_stop(f["board_FT232H0/sensor_1"]) == (130.0, 140.0)


def test_resolve_start_stop_unnumbered_pair(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "start_time": 110.0, "stop_time": 120.0,
    }})
    with h5py.File(p, "r") as f:
        assert tc._resolve_start_stop(f["board_FT232H0/sensor_1"]) == (110.0, 120.0)


def test_resolve_start_stop_no_pair_falls_back_to_time_data(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {"time_data": np.array([100.0, 150.0, 200.0])}})
    with h5py.File(p, "r") as f:
        assert tc._resolve_start_stop(f["board_FT232H0/sensor_1"]) == (100.0, 200.0)


def test_find_video_sensor(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {
        "sensor_0": {"time_data": np.array([1.0])},
        "sensor_2": {"video_filename": b"vid.mp4", "video_frame_index": 7},
    })
    with h5py.File(p, "r") as f:
        board_id, name, num = tc.find_video_sensor(f)
    assert board_id == "board_FT232H0"
    assert name == "sensor_2"
    assert num == 2


def test_find_video_sensor_none_raises(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_0": {"time_data": np.array([1.0])}})
    with h5py.File(p, "r") as f:
        with pytest.raises(ValueError):
            tc.find_video_sensor(f)


def test_read_session_window(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "video_filename": b"vid.mp4", "video_frame_index": 3,
        "start_time": 110.0, "stop_time": 175.0,
    }})
    assert tc.read_session_window(str(p)) == (110.0, 175.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trimcrop.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'video.trimcrop'`

- [ ] **Step 3: Create the module**

Create `video/trimcrop.py`. The bodies are moved verbatim from `make_sync_video.py`; only `read_session_window` is renamed (from `read_session_duration`, which returned a difference).

```python
"""Shared trim/crop primitives for the Pi video recordings.

Anchor math, PTS/session-time conversion, and the ffmpeg wrappers used by both
crop_video.py (one-time interactive trim+crop) and make_sync_video.py (per-clip
stream-copy subclip). See docs/superpowers/specs/2026-07-15-video-crop-tool-design.md.
"""
import re
import subprocess

import h5py
import numpy as np


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


def read_session_window(h5_path):
    """Return (start_time, stop_time) of the video sensor's last cycle."""
    with h5py.File(h5_path, "r") as raw:
        board_id, sensor_name, _ = find_video_sensor(raw)
        return _resolve_start_stop(raw[board_id][sensor_name])


def compute_video_base(pts_ns, frame_index):
    """Seconds between the sync frame's PTS and the first frame's PTS."""
    pts_ns = np.asarray(pts_ns)
    return float(pts_ns[frame_index] - pts_ns[0]) / 1e9


def bookmark_latency(host_before, host_after, start_time):
    """Seconds the bookmark round-trip lagged ``start_time``: the bookmarked frame
    was captured mid-round-trip (~midpoint of the host bracket), so the video
    leads the trace by this much and every frame's session time must be shifted
    later by it. Returns 0.0 when the bracket wasn't recorded (older recordings)."""
    if host_before is None or host_after is None:
        return 0.0
    return (float(host_before) + float(host_after)) / 2.0 - float(start_time)


def frame_session_times(pts_ns, video_base):
    """Session-relative time (0 = start_time bookmark) of every video frame."""
    pts_ns = np.asarray(pts_ns)
    return (pts_ns - pts_ns[0]) / 1e9 - video_base


def compute_trim_frames(pts_ns, video_base, start, end):
    """Inclusive (start_frame, stop_frame) of the video frames whose session
    time falls in [start, end]. Raises ValueError if the window has no frames."""
    sess = frame_session_times(pts_ns, video_base)
    idx = np.flatnonzero((sess >= start) & (sess <= end))
    if idx.size == 0:
        raise ValueError(
            f"no video frames fall in the requested window [{start}, {end}] s")
    return int(idx[0]), int(idx[-1])


def probe_frame_session_times(path, video_base):
    """Session-relative time of every frame in ``path``, read from the file's
    real presentation timestamps via ffprobe. Because trimmed clips keep their
    original PTS (``-copyts``), each frame self-reports its true video-second, so
    ``pts - video_base`` is its session time (0 = start_time bookmark)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{r.stderr[-800:]}")
    pts = np.array([float(tok.strip().rstrip(","))
                    for tok in r.stdout.split() if tok.strip().rstrip(",")])
    if pts.size == 0:
        raise RuntimeError(f"ffprobe found no frames in {path}")
    return pts - video_base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trimcrop.py -q`
Expected: PASS, 12 passed

- [ ] **Step 5: Delete the moved functions from `make_sync_video.py` and import them**

Delete from `make_sync_video.py`: `find_video_sensor`, `_resolve_start_stop`, `read_session_duration`, `compute_video_base`, `bookmark_latency`, `frame_session_times`, `compute_trim_frames`, `probe_frame_session_times`. Also delete the now-unused `import re`.

Add to the import block, after `from data_analysis import filter_data`:

```python
from video.trimcrop import (
    _resolve_start_stop,
    bookmark_latency,
    compute_trim_frames,
    compute_video_base,
    find_video_sensor,
    frame_session_times,
    probe_frame_session_times,
    read_session_window,
)
```

Re-add `read_session_duration` as a thin caller (it is still `make_sync_video`'s CLI validation entry point):

```python
def read_session_duration(h5_path):
    """Session duration in seconds, without running filter_data, so the CLI can
    validate --start/--end before the expensive load."""
    start_time, stop_time = read_session_window(h5_path)
    return stop_time - start_time
```

- [ ] **Step 6: Run the full suite to verify nothing regressed**

Run: `python -m pytest tests/test_trimcrop.py tests/test_make_sync_video.py -q`
Expected: PASS, 34 passed (12 new + 22 unchanged). The existing tests still call `msv.compute_video_base` etc. and resolve through the re-imported names.

- [ ] **Step 7: Commit**

```bash
git add video/trimcrop.py tests/test_trimcrop.py make_sync_video.py
git commit -m "refactor: extract shared trim/PTS math into video/trimcrop.py"
```

---

### Task 2: `read_video_anchor` and `clamp_origin`

**Files:**
- Modify: `video/trimcrop.py` (add `VideoAnchor`, `read_video_anchor`, `clamp_origin`)
- Modify: `tests/test_trimcrop.py` (append)

**Interfaces:**
- Consumes: `find_video_sensor`, `_resolve_start_stop`, `bookmark_latency` from Task 1.
- Produces:
  - `VideoAnchor` dataclass, fields: `sensor_number: int`, `video_filename: str`, `video_frame_index: int`, `start_time: float`, `stop_time: float`, `host_before: float | None`, `host_after: float | None`. Properties: `session_duration -> float`, `latency -> float`.
  - `read_video_anchor(h5_path) -> VideoAnchor`
  - `clamp_origin(x, y, frame_w, frame_h, size) -> (int, int)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trimcrop.py`:

```python
def test_clamp_origin_interior_unchanged():
    assert tc.clamp_origin(452, 180, 1280, 720, 360) == (452, 180)


def test_clamp_origin_rounds_down_to_even():
    # yuv420p needs even offsets
    assert tc.clamp_origin(451, 181, 1280, 720, 360) == (450, 180)


def test_clamp_origin_clamps_each_edge():
    assert tc.clamp_origin(-50, -50, 1280, 720, 360) == (0, 0)
    # right/bottom clamp to frame - size, which stays inside
    assert tc.clamp_origin(9999, 9999, 1280, 720, 360) == (920, 360)


def test_clamp_origin_clamped_edge_stays_even():
    # frame_h - size = 365 is odd -> must round down, not out of frame
    assert tc.clamp_origin(9999, 9999, 1280, 725, 360) == (920, 364)


def test_clamp_origin_size_exceeding_frame_raises():
    with pytest.raises(ValueError):
        tc.clamp_origin(0, 0, 320, 720, 360)
    with pytest.raises(ValueError):
        tc.clamp_origin(0, 0, 1280, 200, 360)


def test_read_video_anchor(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {
        "sensor_0": {"time_data": np.array([1.0])},
        "sensor_1": {
            "time_data": np.array([100.0, 200.0]),
            "video_filename": b"vid.mp4",
            "video_frame_index": 42,
            "start_time": 110.0,
            "stop_time": 175.0,
            "video_bookmark_host_before": 111.0,
            "video_bookmark_host_after": 111.4,
        },
    })
    a = tc.read_video_anchor(str(p))
    assert a.sensor_number == 1
    assert a.video_filename == "vid.mp4"   # decoded, not bytes
    assert a.video_frame_index == 42
    assert a.session_duration == pytest.approx(65.0)
    assert a.latency == pytest.approx(1.2)  # (111.0 + 111.4)/2 - 110.0


def test_read_video_anchor_without_host_bracket(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "video_filename": b"vid.mp4",
        "video_frame_index": 3,
        "start_time": 110.0, "stop_time": 175.0,
    }})
    a = tc.read_video_anchor(str(p))
    assert a.host_before is None and a.host_after is None
    assert a.latency == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trimcrop.py -q -k "clamp_origin or video_anchor"`
Expected: FAIL with `AttributeError: module 'video.trimcrop' has no attribute 'clamp_origin'`

- [ ] **Step 3: Implement**

Add `from dataclasses import dataclass` to the imports of `video/trimcrop.py`, then append:

```python
@dataclass
class VideoAnchor:
    """Everything needed to place the video against the trace, from one h5 open."""
    sensor_number: int
    video_filename: str
    video_frame_index: int
    start_time: float
    stop_time: float
    host_before: float | None
    host_after: float | None

    @property
    def session_duration(self):
        return self.stop_time - self.start_time

    @property
    def latency(self):
        return bookmark_latency(self.host_before, self.host_after, self.start_time)


def read_video_anchor(h5_path):
    """Read the video sensor's sync metadata and session window in one open."""
    with h5py.File(h5_path, "r") as raw:
        board_id, sensor_name, sensor_number = find_video_sensor(raw)
        group = raw[board_id][sensor_name]
        start_time, stop_time = _resolve_start_stop(group)
        fname = group["video_filename"][()]
        fname = fname.decode() if isinstance(fname, bytes) else str(fname)
        return VideoAnchor(
            sensor_number=sensor_number,
            video_filename=fname,
            video_frame_index=int(group["video_frame_index"][()]),
            start_time=start_time,
            stop_time=stop_time,
            host_before=(float(group["video_bookmark_host_before"][()])
                         if "video_bookmark_host_before" in group else None),
            host_after=(float(group["video_bookmark_host_after"][()])
                        if "video_bookmark_host_after" in group else None),
        )


def clamp_origin(x, y, frame_w, frame_h, size):
    """Clamp a proposed crop origin so the size x size box stays inside the frame,
    rounding each coordinate down to an even number (yuv420p needs even offsets).
    Rounding down only ever moves the box further inside. Raises ValueError if the
    box cannot fit."""
    if size > frame_w or size > frame_h:
        raise ValueError(
            f"crop size {size} exceeds frame {frame_w}x{frame_h}")
    x = int(min(max(x, 0), frame_w - size))
    y = int(min(max(y, 0), frame_h - size))
    return x - (x % 2), y - (y % 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trimcrop.py -q`
Expected: PASS, 19 passed

- [ ] **Step 5: Verify the anchor against the real recording**

Run:
```bash
python -c "
from video.trimcrop import read_video_anchor
a = read_video_anchor('Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.h5')
print(a.sensor_number, a.video_filename, a.video_frame_index)
print('duration', a.session_duration, 'latency', a.latency)
"
```
Expected: sensor `1`, filename ending `.mp4`, `duration` > 3600, `latency` 0.0 (this recording predates the host bracket — matches `test_load_recording_reference`).

- [ ] **Step 6: Commit**

```bash
git add video/trimcrop.py tests/test_trimcrop.py
git commit -m "feat: video anchor reader and crop-origin clamping"
```

---

### Task 3: ffmpeg wrappers — `probe_start_pts`, `trim_and_crop`, `subclip_copy`

`trim_and_crop` moves out of `make_sync_video.py` **with a changed signature**: explicit square crop origin instead of a centered `(iw-w)/2` expression.

**Files:**
- Modify: `video/trimcrop.py` (add the three functions)
- Modify: `make_sync_video.py` — delete `trim_and_crop` (`:209-231`), import it instead
- Modify: `tests/test_trimcrop.py` (append)
- Modify: `tests/test_make_sync_video.py:132-160` (`test_trim_and_crop_and_frame_source`)

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces:
  - `probe_start_pts(path) -> float`
  - `trim_and_crop(video_path, start_sec, end_sec, out_path, crop_x, crop_y, size, seek_margin=5.0) -> out_path`
  - `subclip_copy(video_path, start_sec, end_sec, out_path, seek_margin=5.0) -> out_path`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trimcrop.py`:

```python
import types


def _fake_run(calls, stdout="", returncode=0):
    def run(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="boom")
    return run


def test_probe_start_pts(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, stdout="30.000000\n"))
    assert tc.probe_start_pts("in.mp4") == pytest.approx(30.0)
    assert "start_time" in " ".join(calls[0])


def test_probe_start_pts_missing_is_zero(monkeypatch):
    # ffprobe prints "N/A" for containers without a start_time
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, stdout="N/A\n"))
    assert tc.probe_start_pts("in.mp4") == 0.0


def test_probe_start_pts_failure_raises(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, returncode=1))
    with pytest.raises(RuntimeError):
        tc.probe_start_pts("in.mp4")


def test_trim_and_crop_builds_argv(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls))
    tc.trim_and_crop("in.mp4", 100.0, 110.0, "out.mp4", 452, 180, 360)
    cmd = calls[0]
    assert "crop=360:360:452:180" in cmd
    assert "-copyts" in cmd
    # trim_and_crop reads the original video (start_time 0) -> plain margin seek
    assert cmd[cmd.index("-ss") + 1] == "95.000000"
    assert cmd[cmd.index("-to") + 1] == "110.000000"
    assert "libx264" in cmd  # re-encode: a filter is applied


def test_trim_and_crop_seek_floors_at_zero(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls))
    tc.trim_and_crop("in.mp4", 2.0, 8.0, "out.mp4", 0, 0, 360)
    assert calls[0][calls[0].index("-ss") + 1] == "0.000000"


def test_trim_and_crop_failure_raises(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, returncode=1))
    with pytest.raises(RuntimeError):
        tc.trim_and_crop("in.mp4", 100.0, 110.0, "out.mp4", 0, 0, 360)


def test_subclip_copy_builds_argv(monkeypatch):
    calls = []
    # first call is probe_start_pts, second is the ffmpeg subclip
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, stdout="0.000000\n"))
    tc.subclip_copy("in.mp4", 100.0, 110.0, "out.mp4")
    cmd = calls[-1]
    assert cmd[cmd.index("-c") + 1] == "copy"   # stream copy, no re-encode
    assert "-copyts" in cmd
    assert "libx264" not in cmd
    assert not any(str(a).startswith("crop=") for a in cmd)
    assert cmd[cmd.index("-ss") + 1] == "95.000000"
    assert cmd[cmd.index("-to") + 1] == "110.000000"


def test_subclip_copy_seek_is_file_relative(monkeypatch):
    """Input -ss is relative to the container start_time, but -to under -copyts is
    absolute. On a cropped file whose PTS start at 30 s, seeking to original-timeline
    second 100 means -ss 65 (=100-30-5), while -to stays at 110."""
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, stdout="30.000000\n"))
    tc.subclip_copy("cropped.mp4", 100.0, 110.0, "out.mp4")
    cmd = calls[-1]
    assert cmd[cmd.index("-ss") + 1] == "65.000000"
    assert cmd[cmd.index("-to") + 1] == "110.000000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trimcrop.py -q -k "probe_start_pts or trim_and_crop or subclip_copy"`
Expected: FAIL with `AttributeError: module 'video.trimcrop' has no attribute 'probe_start_pts'`

- [ ] **Step 3: Implement**

Append to `video/trimcrop.py`:

```python
def probe_start_pts(path):
    """The input's first presentation timestamp, in seconds. Containers without a
    start_time report "N/A"; treat those as 0."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time",
         "-of", "csv=p=0", path],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{r.stderr[-800:]}")
    tok = r.stdout.strip()
    try:
        return float(tok)
    except ValueError:
        return 0.0


def trim_and_crop(video_path, start_sec, end_sec, out_path,
                  crop_x, crop_y, size, seek_margin=5.0):
    """Trim ``video_path`` to video-seconds [start_sec, end_sec] and crop a
    size x size square whose top-left corner is (crop_x, crop_y), writing a new
    file (the original is left untouched). Re-encodes, because a filter applies.

    Uses a coarse fast seek to ``start_sec - seek_margin`` and ``-copyts`` so the
    output frames keep their ORIGINAL presentation timestamps. ffmpeg's input
    ``-ss`` does not land frame-accurately on this footage, so we deliberately
    seek a little early and rely on the preserved PTS (read back with
    ``probe_frame_session_times``) to time each frame — never on where the seek
    landed. This reads the original recording, whose PTS start at 0, so the seek
    needs no start-PTS correction. Returns out_path. Raises RuntimeError if
    ffmpeg fails."""
    coarse = max(0.0, start_sec - seek_margin)
    vf = f"crop={size}:{size}:{crop_x}:{crop_y}"
    cmd = [
        "ffmpeg", "-y", "-ss", f"{coarse:.6f}", "-copyts", "-i", video_path,
        "-to", f"{end_sec:.6f}", "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg trim/crop failed:\n{r.stderr[-800:]}")
    return out_path


def subclip_copy(video_path, start_sec, end_sec, out_path, seek_margin=5.0):
    """Cut video-seconds [start_sec, end_sec] out of ``video_path`` by stream copy
    — no filter, no re-encode, near-instant. Used to seek cheaply into a long file
    without decoding everything before the window.

    ``start_sec``/``end_sec`` are ORIGINAL-timeline seconds. Input ``-ss`` is
    relative to the container's start_time, while ``-to`` under ``-copyts`` is
    absolute, so the seek subtracts the input's start PTS and the end does not.
    That subtraction is a no-op on the original recording (start_time 0) and is
    what makes the seek land on a cropped file (start_time = session start).

    Stream copy cuts at the keyframe at or before the seek target, so the output
    may carry frames earlier than ``start_sec``. That is harmless: consumers time
    frames by PTS and skip past them. Returns out_path."""
    start_pts = probe_start_pts(video_path)
    coarse = max(0.0, start_sec - start_pts - seek_margin)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{coarse:.6f}", "-copyts", "-i", video_path,
        "-to", f"{end_sec:.6f}", "-an", "-c", "copy", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg subclip failed:\n{r.stderr[-800:]}")
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trimcrop.py -q`
Expected: PASS, 27 passed

- [ ] **Step 5: Delete `trim_and_crop` from `make_sync_video.py` and import it**

Delete the `trim_and_crop` definition from `make_sync_video.py`. Add `subclip_copy`, `trim_and_crop`, and `probe_start_pts` to the `from video.trimcrop import (...)` block from Task 1 (keep it alphabetized).

`render_clip` still calls `trim_and_crop(rec.video_path, start_sec, end_sec, intermediate_path, crop_w, crop_h)` — a stale signature. Task 7 rewrites that call; until then the reference-video tests that reach it will fail, which is expected and is why Step 6 scopes the run.

- [ ] **Step 6: Update the moved test in `tests/test_make_sync_video.py`**

Replace `test_trim_and_crop_and_frame_source` (lines 132-160) — it used the old `crop_w`/`crop_h` kwargs and the uncorrected `rec.video_base`:

```python
@needs_reference
@needs_video
def test_trim_and_crop_and_frame_source(tmp_path):
    import imageio
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO)
    sf, ef = msv.compute_trim_frames(rec.pts_ns, rec.video_base, 120.0, 123.0)
    start_sec = float(rec.pts_ns[sf] - rec.pts_ns[0]) / 1e9
    end_sec = float(rec.pts_ns[ef] - rec.pts_ns[0]) / 1e9 + 0.3
    out = str(tmp_path / "trim.mp4")
    msv.trim_and_crop(VIDEO, start_sec, end_sec, out, 452, 180, 360)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    r = imageio.get_reader(out, "ffmpeg")
    size = r.get_meta_data()["size"]
    r.close()
    assert size == (360, 360)  # (width, height)

    frame_sess = msv.probe_frame_session_times(out, rec.video_base)
    assert frame_sess[0] <= 120.0 and frame_sess[-1] >= 123.0
    assert np.all(np.diff(frame_sess) >= 0)  # monotonic

    src = msv.TrimmedFrameSource(out, frame_sess)
    try:
        f0 = src.get(120.0)
        assert f0 is not None and f0.shape[:2] == (360, 360)  # (h, w)
        f1 = src.get(122.0)
        assert f1.shape == f0.shape
    finally:
        src.close()
```

Add a real end-to-end test of the seek asymmetry, right after it — this is the one behavior that argv assertions cannot prove:

```python
@needs_reference
@needs_video
def test_subclip_copy_lands_on_a_cropped_file(tmp_path):
    """A cropped file's PTS start at the session start, not 0. Stream-copying a
    window out of it must still cover that window."""
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO)
    sf, ef = msv.compute_trim_frames(rec.pts_ns, rec.video_base, 120.0, 130.0)
    start_sec = float(rec.pts_ns[sf] - rec.pts_ns[0]) / 1e9
    end_sec = float(rec.pts_ns[ef] - rec.pts_ns[0]) / 1e9 + 0.3
    cropped = str(tmp_path / "cropped.mp4")
    msv.trim_and_crop(VIDEO, start_sec, end_sec, cropped, 452, 180, 360)
    assert msv.probe_start_pts(cropped) > 1.0  # not a zero-based timeline

    sub = str(tmp_path / "sub.mp4")
    msv.subclip_copy(cropped, start_sec + 2.0, start_sec + 5.0, sub)
    sess = msv.probe_frame_session_times(sub, rec.video_base)
    assert sess.size > 0
    assert sess[0] <= 122.0 and sess[-1] >= 124.0
```

- [ ] **Step 7: Run both test files**

Run: `python -m pytest tests/test_trimcrop.py tests/test_make_sync_video.py -q -k "not render_clip_smoke"`
Expected: PASS. `test_render_clip_smoke` is excluded because `render_clip` still passes the old `crop_w`/`crop_h` args; Task 7 fixes it.

- [ ] **Step 8: Commit**

```bash
git add video/trimcrop.py make_sync_video.py tests/test_trimcrop.py tests/test_make_sync_video.py
git commit -m "feat: explicit crop origin, stream-copy subclip, start-PTS-aware seek"
```

---

### Task 4: Path resolution with cropped-file preference

**Files:**
- Modify: `video/trimcrop.py` (add `resolve_paths`)
- Modify: `make_sync_video.py` — delete `resolve_paths` (`:365-374`), import it
- Modify: `tests/test_trimcrop.py` (append)
- Modify: `tests/test_make_sync_video.py:222-227` (`test_resolve_paths_defaults_from_h5`)

**Interfaces:**
- Consumes: `VideoAnchor` from Task 2.
- Produces: `resolve_paths(h5_path, anchor, video=None, pts_txt=None, prefer_cropped=False) -> (video_path, pts_txt_path)`, and `cropped_path_for(video_path) -> str`.

Note the signature change: `resolve_paths` now takes the already-read `VideoAnchor` rather than re-opening the h5.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trimcrop.py`:

```python
import os


def _anchor(video_filename="v.mp4"):
    return tc.VideoAnchor(
        sensor_number=1, video_filename=video_filename, video_frame_index=3,
        start_time=110.0, stop_time=175.0, host_before=None, host_after=None,
    )


def test_cropped_path_for():
    assert tc.cropped_path_for("/d/v.mp4") == "/d/v_cropped.mp4"


def test_resolve_paths_plain(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    video, pts = tc.resolve_paths(str(tmp_path / "r.h5"), _anchor())
    assert video == str(tmp_path / "v.mp4")
    assert pts == str(tmp_path / "v.txt")


def test_resolve_paths_prefers_cropped_when_present(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    video, pts = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), prefer_cropped=True)
    assert video == str(tmp_path / "v_cropped.mp4")
    # THE TRAP: the sidecar belongs to the ORIGINAL video and must not follow
    # the cropped name. The cropped file has no sidecar and needs none.
    assert pts == str(tmp_path / "v.txt")


def test_resolve_paths_falls_back_when_no_cropped(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    video, pts = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), prefer_cropped=True)
    assert video == str(tmp_path / "v.mp4")
    assert pts == str(tmp_path / "v.txt")


def test_resolve_paths_ignores_cropped_when_not_preferred(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    video, _ = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), prefer_cropped=False)
    assert video == str(tmp_path / "v.mp4")


def test_resolve_paths_explicit_overrides(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    video, pts = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), video="/elsewhere/x.mp4",
        prefer_cropped=True)
    assert video == "/elsewhere/x.mp4"
    # sidecar still derives from the h5's video_filename, not from --video
    assert pts == str(tmp_path / "v.txt")


def test_resolve_paths_explicit_pts_overrides(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    _, pts = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), pts_txt="/elsewhere/x.txt")
    assert pts == "/elsewhere/x.txt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trimcrop.py -q -k "resolve_paths or cropped_path_for"`
Expected: FAIL with `AttributeError: module 'video.trimcrop' has no attribute 'cropped_path_for'`

- [ ] **Step 3: Implement**

Add `import os` to the imports of `video/trimcrop.py`, then append:

```python
def cropped_path_for(video_path):
    """The conventional cropped sibling of a recording: <base>_cropped.mp4."""
    return os.path.splitext(video_path)[0] + "_cropped.mp4"


def resolve_paths(h5_path, anchor, video=None, pts_txt=None, prefer_cropped=False):
    """Resolve (video_path, pts_txt_path) for a recording.

    The PTS sidecar is ALWAYS <original-video-base>.txt, derived from the h5's
    video_filename — never from the resolved video path. A cropped file has no
    sidecar of its own and needs none: it carries the original PTS via -copyts,
    and probe_frame_session_times reads them back from the file itself.

    ``prefer_cropped`` picks <base>_cropped.mp4 when it exists, printing a note to
    stderr when it doesn't, so a forgotten crop degrades to a correct (just
    uncropped) render instead of an error.
    """
    base = os.path.join(os.path.dirname(h5_path), anchor.video_filename)
    if pts_txt is None:
        pts_txt = os.path.splitext(base)[0] + ".txt"
    if video is None:
        video = base
        if prefer_cropped:
            cropped = cropped_path_for(base)
            if os.path.exists(cropped):
                video = cropped
            else:
                print(f"note: using uncropped video {base} "
                      f"(no _cropped.mp4; run crop_video.py first)",
                      file=sys.stderr)
    return video, pts_txt
```

Add `import sys` to the module imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trimcrop.py -q`
Expected: PASS, 34 passed

- [ ] **Step 5: Delete `resolve_paths` from `make_sync_video.py` and import it**

Delete the `resolve_paths` definition. Add `cropped_path_for`, `read_video_anchor`, and `resolve_paths` to the `from video.trimcrop import (...)` block.

`main()` currently calls `resolve_paths(args.h5, args.video, args.pts_txt)`; Task 7 updates the call site. Scope the test run accordingly.

- [ ] **Step 6: Update `test_resolve_paths_defaults_from_h5` in `tests/test_make_sync_video.py`**

Replace lines 222-227 — the signature now takes an anchor:

```python
@needs_reference
def test_resolve_paths_defaults_from_h5():
    anchor = msv.read_video_anchor(H5)
    video, pts = msv.resolve_paths(H5, anchor)
    assert video.endswith("raw_data_2026-07-13_11-59-47.mp4")
    assert pts.endswith("raw_data_2026-07-13_11-59-47.txt")
    assert os.path.dirname(video) == os.path.dirname(H5)
```

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/test_trimcrop.py tests/test_make_sync_video.py -q -k "not render_clip_smoke"`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add video/trimcrop.py make_sync_video.py tests/test_trimcrop.py tests/test_make_sync_video.py
git commit -m "feat: prefer _cropped.mp4, keep PTS sidecar on the original name"
```

---

### Task 5: `crop_video.py` CLI core (no GUI)

Everything except the matplotlib window, so it can be tested headlessly.

**Files:**
- Create: `crop_video.py`
- Create: `tests/test_crop_video.py`

**Interfaces:**
- Consumes: `read_video_anchor`, `resolve_paths`, `compute_video_base`, `compute_trim_frames`, `cropped_path_for`, `trim_and_crop` from Tasks 1-4.
- Produces: `build_arg_parser()`, `compute_crop_window(anchor, pts_ns) -> (start_frame, stop_frame, start_sec, end_sec)`, `resolve_out_path(video, out, force) -> str`, `reject_cropped_input(video) -> None`, `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_crop_video.py`:

```python
import os

import numpy as np
import pytest

import crop_video as cv
from video.trimcrop import VideoAnchor


def _anchor(frame_index=2, start=110.0, stop=110.5,
            host_before=None, host_after=None):
    return VideoAnchor(
        sensor_number=1, video_filename="v.mp4", video_frame_index=frame_index,
        start_time=start, stop_time=stop,
        host_before=host_before, host_after=host_after,
    )


def test_compute_crop_window_spans_the_session():
    # frames every 0.1 s over 1.0 s; bookmark frame 2 -> session zero at 0.2 s
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    sf, ef, start_sec, end_sec = cv.compute_crop_window(
        _anchor(frame_index=2, start=110.0, stop=110.5), pts_ns)
    assert (sf, ef) == (2, 7)              # session [0, 0.5] s
    assert start_sec == pytest.approx(0.2)  # video-file seconds
    assert end_sec == pytest.approx(0.7 + 0.3)  # tail margin


def test_compute_crop_window_applies_bookmark_latency():
    """The bookmarked frame was captured mid-round-trip, so raw frame session
    times run early (the video leads the trace). Correcting for it shifts every
    frame's session label LATER, so the same session window resolves to EARLIER
    frames in the file.

    The bracket is chosen so the latency is exactly representable in binary
    (0.25). Values like 0.2 are not: (110.1+110.3)/2 - 110.0 evaluates to
    0.19999999999998863, which leaves video_base_eff a hair ABOVE zero, and the
    `sess >= start` test in compute_trim_frames then drops frame 0 on an
    epsilon. That knife-edge would make this test assert floating-point noise
    rather than the behavior it is here to pin down.
    """
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    plain = cv.compute_crop_window(_anchor(frame_index=2, start=110.0, stop=110.3), pts_ns)
    assert plain[0] == 2 and plain[2] == pytest.approx(0.2)
    # bracket midpoint 0.25 s after start_time -> latency 0.25 (exact)
    shifted = cv.compute_crop_window(
        _anchor(frame_index=2, start=110.0, stop=110.3,
                host_before=110.0, host_after=110.5), pts_ns)
    assert shifted[0] == 0
    assert shifted[0] < plain[0]           # earlier start frame
    assert shifted[2] < plain[2]           # earlier start second


def test_compute_crop_window_empty_raises():
    """A reversed window (stop_time before start_time — corrupt h5) is the only
    way this raises, so it is what we test.

    For any sane recording the window CANNOT be empty: the bookmarked frame sits
    at session time == latency by construction, which is inside [0, duration]
    unless the latency exceeds the whole session. The guard is therefore about
    propagating compute_trim_frames' error on degenerate data, not a case real
    recordings reach.
    """
    pts_ns = (np.arange(0, 3) * 100_000_000).astype(np.int64)
    with pytest.raises(ValueError):
        cv.compute_crop_window(_anchor(frame_index=0, start=900.0, stop=500.0), pts_ns)


def test_resolve_out_path_default(tmp_path):
    v = str(tmp_path / "v.mp4")
    assert cv.resolve_out_path(v, None, False) == str(tmp_path / "v_cropped.mp4")


def test_resolve_out_path_explicit(tmp_path):
    v = str(tmp_path / "v.mp4")
    assert cv.resolve_out_path(v, "/o/x.mp4", False) == "/o/x.mp4"


def test_resolve_out_path_existing_raises(tmp_path):
    v = str(tmp_path / "v.mp4")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    with pytest.raises(ValueError, match="--force"):
        cv.resolve_out_path(v, None, False)


def test_resolve_out_path_existing_with_force(tmp_path):
    v = str(tmp_path / "v.mp4")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    assert cv.resolve_out_path(v, None, True) == str(tmp_path / "v_cropped.mp4")


def test_reject_cropped_input():
    with pytest.raises(ValueError, match="already-cropped"):
        cv.reject_cropped_input("/d/v_cropped.mp4")
    cv.reject_cropped_input("/d/v.mp4")  # no raise


def test_build_arg_parser_defaults():
    args = cv.build_arg_parser().parse_args(["--h5", "r.h5"])
    assert args.h5 == "r.h5"
    assert args.size == 360
    assert args.video is None and args.pts_txt is None
    assert args.out is None and args.force is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_crop_video.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'crop_video'`

- [ ] **Step 3: Implement**

Create `crop_video.py`:

```python
"""Interactively crop a Pi video recording to a square region and trim it to the
capacitance-recording window.

Shows the frame at the middle of the recording window, lets you drag a fixed-size
square over the region of interest, then writes <video>_cropped.mp4 — trimmed to
the session and cropped to the square. make_sync_video.py picks that file up
automatically. See docs/superpowers/specs/2026-07-15-video-crop-tool-design.md.
"""
import argparse
import os
import shutil
import sys

import numpy as np

from video.trimcrop import (
    compute_trim_frames,
    compute_video_base,
    cropped_path_for,
    read_video_anchor,
    resolve_paths,
    trim_and_crop,
)

TAIL_MARGIN = 0.3  # seconds of slack past the last in-window frame


def compute_crop_window(anchor, pts_ns):
    """Return (start_frame, stop_frame, start_sec, end_sec) covering the whole
    session. Frames are the indices into the original video; the seconds are on
    the original video's timeline. Uses the latency-corrected anchor, matching
    make_sync_video's render_clip."""
    pts_ns = np.asarray(pts_ns)
    video_base_eff = compute_video_base(pts_ns, anchor.video_frame_index) - anchor.latency
    sf, ef = compute_trim_frames(pts_ns, video_base_eff, 0.0, anchor.session_duration)
    start_sec = float(pts_ns[sf] - pts_ns[0]) / 1e9
    end_sec = float(pts_ns[ef] - pts_ns[0]) / 1e9 + TAIL_MARGIN
    return sf, ef, start_sec, end_sec


def reject_cropped_input(video):
    """Refuse to crop a file that is already a crop."""
    if os.path.splitext(video)[0].endswith("_cropped"):
        raise ValueError(f"refusing to crop an already-cropped video: {video}")


def resolve_out_path(video, out, force):
    if out is None:
        out = cropped_path_for(video)
    if os.path.exists(out) and not force:
        raise ValueError(f"{out} exists; pass --force to overwrite")
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Trim a recording to its capacitance window and crop it to a "
                    "hand-positioned square.")
    p.add_argument("--h5", required=True, help="raw recording .h5")
    p.add_argument("--video", default=None,
                   help="source video (default: from h5 video_filename)")
    p.add_argument("--pts-txt", dest="pts_txt", default=None,
                   help="per-frame PTS sidecar (default: video path with .txt)")
    p.add_argument("--size", type=int, default=360,
                   help="side length of the square crop (default 360)")
    p.add_argument("--out", default=None,
                   help="output path (default: <video>_cropped.mp4)")
    p.add_argument("--force", action="store_true",
                   help="overwrite the output if it exists")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH (needed to write the video)",
              file=sys.stderr)
        return 1
    try:
        anchor = read_video_anchor(args.h5)
        video, pts_txt = resolve_paths(args.h5, anchor, args.video, args.pts_txt,
                                       prefer_cropped=False)
        reject_cropped_input(video)
        out = resolve_out_path(video, args.out, args.force)
        pts_ns = np.loadtxt(pts_txt, dtype=np.int64)
        sf, ef, start_sec, end_sec = compute_crop_window(anchor, pts_ns)
        print(f"animal sensor {anchor.sensor_number}; session "
              f"{anchor.session_duration:.1f} s -> frames {sf}..{ef} "
              f"({start_sec:.2f}..{end_sec:.2f} s of video)")
        raise NotImplementedError("GUI wired up in Task 6")
    except (ValueError, FileNotFoundError, KeyError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_crop_video.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Verify the window against the real recording**

Run:
```bash
python -c "
import numpy as np
from video.trimcrop import read_video_anchor, resolve_paths
import crop_video as cv
h5 = 'Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.h5'
a = read_video_anchor(h5)
video, pts_txt = resolve_paths(h5, a)
pts = np.loadtxt(pts_txt, dtype=np.int64)
print(cv.compute_crop_window(a, pts))
"
```
Expected: `start_sec` ≈ 31.97 (this recording's `video_base`, latency 0), `stop_frame` near the last frame, `end_sec` > 3600.

- [ ] **Step 6: Commit**

```bash
git add crop_video.py tests/test_crop_video.py
git commit -m "feat: crop_video CLI core — session window, output path, validation"
```

---

### Task 6: The crop GUI

**Files:**
- Modify: `crop_video.py` (add `load_preview_frame`, `CropSelector`, wire `main`)

**Interfaces:**
- Consumes: `clamp_origin` (Task 2), `compute_crop_window`, `resolve_out_path` (Task 5).
- Produces: `load_preview_frame(video, frame_index) -> ndarray`, `CropSelector(frame, size).run() -> (x, y) | None`.

The GUI is deliberately thin — the event handlers delegate all math to the unit-tested `clamp_origin`, so it is verified by running it, not by unit tests.

- [ ] **Step 1: Implement the preview loader and the selector**

Add to the imports of `crop_video.py`:

```python
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

from video.trimcrop import clamp_origin
```

Note: do **not** set the Agg backend here. `make_sync_video.py` forces Agg because it renders headlessly; this tool needs a real window.

Append:

```python
def load_preview_frame(video, frame_index):
    """The frame at ``frame_index`` of the original video, as an RGB array."""
    reader = imageio.get_reader(video, "ffmpeg")
    try:
        return reader.get_data(frame_index)
    finally:
        reader.close()


class CropSelector:
    """Drag a fixed-size square over a still frame. run() returns the chosen
    (x, y) origin, or None if the window was closed without pressing Crop."""

    def __init__(self, frame, size):
        self.frame = frame
        self.size = size
        self.h, self.w = frame.shape[:2]
        self.result = None
        self._grab = None  # (dx, dy) offset from the box origin to the cursor

        x0, y0 = clamp_origin((self.w - size) / 2, (self.h - size) / 2,
                              self.w, self.h, size)
        self.x, self.y = x0, y0

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.subplots_adjust(bottom=0.12)
        self.ax.imshow(frame)
        self.ax.axis("off")
        self.rect = Rectangle((self.x, self.y), size, size, fill=False,
                              lw=2, edgecolor="lime")
        self.ax.add_patch(self.rect)
        self._update_title()

        self.button = Button(self.fig.add_axes([0.82, 0.02, 0.13, 0.06]), "Crop")
        self.button.on_clicked(self._on_crop)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _update_title(self):
        self.ax.set_title(
            f"drag the box over the sipper, then press Crop   "
            f"[{self.size}x{self.size} @ ({self.x}, {self.y})]")

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if (self.x <= event.xdata <= self.x + self.size
                and self.y <= event.ydata <= self.y + self.size):
            self._grab = (event.xdata - self.x, event.ydata - self.y)

    def _on_motion(self, event):
        if self._grab is None or event.inaxes is not self.ax or event.xdata is None:
            return
        dx, dy = self._grab
        self.x, self.y = clamp_origin(event.xdata - dx, event.ydata - dy,
                                      self.w, self.h, self.size)
        self.rect.set_xy((self.x, self.y))
        self._update_title()
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self._grab = None

    def _on_crop(self, event):
        self.result = (self.x, self.y)
        plt.close(self.fig)

    def run(self):
        plt.show()
        return self.result
```

- [ ] **Step 2: Wire it into `main`**

Replace the `raise NotImplementedError("GUI wired up in Task 6")` line with:

```python
        mid_frame = (sf + ef) // 2
        frame = load_preview_frame(video, mid_frame)
        origin = CropSelector(frame, args.size).run()
        if origin is None:
            print("cancelled")
            return 0
        x, y = origin
        print(f"cropping {args.size}x{args.size} @ ({x}, {y}) -> {out}")
        trim_and_crop(video, start_sec, end_sec, out, x, y, args.size)
        print("done")
```

- [ ] **Step 3: Verify the tests still pass**

Run: `python -m pytest tests/test_crop_video.py -q`
Expected: PASS, 9 passed (the CLI-core tests never reach the GUI)

- [ ] **Step 4: Run the tool on the reference recording**

Run:
```bash
python crop_video.py --h5 "Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47.h5" --out /tmp/ref_cropped.mp4
```

Expected: a window opens showing a frame from the middle of the session with a green 360x360 box. Confirm by hand:
- dragging the box moves it and the title updates live
- the box cannot be dragged outside the frame
- pressing Crop closes the window and the trim runs
- `/tmp/ref_cropped.mp4` is 360x360 and its duration ≈ the session duration:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 /tmp/ref_cropped.mp4
ffprobe -v error -show_entries format=duration,start_time -of csv=p=0 /tmp/ref_cropped.mp4
```
Expected: `360,360`; `start_time` ≈ 32 (the session start on the original timeline, preserved by `-copyts`), `duration` ≈ the session duration.

Also confirm the cancel path: rerun, close the window with its X instead of pressing Crop → prints `cancelled`, exit 0, no file written.

- [ ] **Step 5: Commit**

```bash
git add crop_video.py
git commit -m "feat: draggable crop selector GUI"
```

---

### Task 7: Point `make_sync_video.py` at the cropped file

**Files:**
- Modify: `make_sync_video.py` — `load_recording`, `render_clip` (`:277-306` region), `build_arg_parser`, `main`
- Modify: `tests/test_make_sync_video.py:210-218` (`test_build_arg_parser_parses_required`), and every `load_recording` call site

**Interfaces:**
- Consumes: `subclip_copy`, `resolve_paths(prefer_cropped=True)`, `read_video_anchor` from Tasks 3-4.
- Produces: `load_recording(h5_path, layout_path, pts_txt_path, video_path, anchor)` — new trailing `anchor` parameter. `render_clip`'s `crop_w`/`crop_h` parameters are gone.

- [ ] **Step 1: Write the failing test**

Replace `test_build_arg_parser_parses_required` in `tests/test_make_sync_video.py`:

```python
def test_build_arg_parser_parses_required():
    p = msv.build_arg_parser()
    args = p.parse_args([
        "--h5", "r.h5", "--layout", "l.csv",
        "--start", "5", "--end", "9", "--out", "o.mp4",
    ])
    assert args.h5 == "r.h5" and args.start == 5.0 and args.end == 9.0
    assert args.fps == 30.0 and args.window == 2.5 and args.sync_offset == 0.0
    assert args.intermediate is None
    # cropping is crop_video.py's job now
    assert not hasattr(args, "crop_w")
    assert not hasattr(args, "crop_h")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_make_sync_video.py -q -k build_arg_parser`
Expected: FAIL — `assert not hasattr(args, "crop_w")`

- [ ] **Step 3: Make `load_recording` take the anchor instead of re-deriving it**

`load_recording` currently opens the h5 and redoes exactly what `read_video_anchor` does. Take the anchor as a parameter instead. Change the signature:

```python
def load_recording(h5_path, layout_path, pts_txt_path, video_path, anchor):
```

Delete this block from its body (the first `with h5py.File(...)` and the two lines after it):

```python
    with h5py.File(h5_path, "r") as raw:
        board_id, sensor_name, sensor_number = find_video_sensor(raw)
        group = raw[board_id][sensor_name]
        start_time, stop_time = _resolve_start_stop(group)
        video_frame_index = int(group["video_frame_index"][()])
        host_before = (float(group["video_bookmark_host_before"][()])
                       if "video_bookmark_host_before" in group else None)
        host_after = (float(group["video_bookmark_host_after"][()])
                      if "video_bookmark_host_after" in group else None)
    session_duration = stop_time - start_time
    latency = bookmark_latency(host_before, host_after, start_time)

    animal = str(layout.loc[sensor_number].iloc[0])
```

replacing it with:

```python
    session_duration = anchor.session_duration
    latency = anchor.latency
    video_frame_index = anchor.video_frame_index
    animal = str(layout.loc[anchor.sensor_number].iloc[0])
```

and change the `Recording(...)` construction's `sensor=sensor_number` to `sensor=anchor.sensor_number`. The rest of the body (the `filter_data` call, the pts load, `compute_video_base`) is unchanged.

`find_video_sensor`, `_resolve_start_stop`, and `bookmark_latency` are now unused in `make_sync_video.py` — drop them from the `from video.trimcrop import (...)` block. Importing the private `_resolve_start_stop` across modules goes away with them.

Update the four `load_recording` call sites in `tests/test_make_sync_video.py` (in `test_load_recording_reference`, `test_trim_and_crop_and_frame_source`, `test_subclip_copy_lands_on_a_cropped_file`, `test_render_clip_smoke`) to pass an anchor:

```python
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO, msv.read_video_anchor(H5))
```

- [ ] **Step 4: Update `render_clip`**

Change the signature — drop `crop_w`/`crop_h`:

```python
def render_clip(rec, start, end, out_path, fps=30.0, window=2.5, sync_offset=0.0,
                intermediate_path=None):
```

Update its docstring's first paragraph:

```python
    """Render the side-by-side clip. First stream-copies the mouse video down to
    the clip window (intermediate file, kept) so we don't decode the whole
    recording, then composites from it: the left panel is the video frame, the
    right panel the sliding capacitance trace with a centered dot and lick
    markers. Each video frame is placed by its true session time (from its
    preserved PTS) — no seeking assumptions, no fps assumption — so video and
    trace stay aligned.

    The video panel shows whatever region ``rec.video_path`` already carries:
    crop it once with crop_video.py, or pass the uncropped recording for a
    full-frame panel.

    ``sync_offset`` is a residual manual nudge in seconds: increase it if the
    video still runs ahead of the trace.
    """
```

Replace the `trim_and_crop` call:

```python
    trim_and_crop(rec.video_path, start_sec, end_sec, intermediate_path,
                  crop_w, crop_h)
```

with:

```python
    subclip_copy(rec.video_path, start_sec, end_sec, intermediate_path)
```

Everything after it is unchanged: `probe_frame_session_times(intermediate_path, video_base_eff)` reads the subclip's preserved PTS, and `TrimmedFrameSource` matches frames by session time.

- [ ] **Step 5: Update `build_arg_parser` and `main`**

Delete the `--crop-w` and `--crop-h` arguments. Update `--video`'s help:

```python
    p.add_argument("--video", default=None,
                   help="mouse video (default: <base>_cropped.mp4 from "
                        "crop_video.py, else the uncropped recording)")
```

In `main`, replace the resolve + render calls:

```python
        anchor = read_video_anchor(args.h5)
        video, pts_txt = resolve_paths(args.h5, anchor, args.video, args.pts_txt,
                                       prefer_cropped=True)
        validate_window(args.start, args.end, anchor.session_duration)
```

(`read_session_duration` is no longer needed in `main` — the anchor already carries the window. Keep the function; `tests/test_make_sync_video.py::test_read_session_duration_reference` covers it.)

Pass the anchor through to `load_recording`:

```python
        rec = load_recording(args.h5, args.layout, pts_txt, video, anchor)
```

and drop the crop args from the render call:

```python
        render_clip(rec, args.start, args.end, args.out,
                    fps=args.fps, window=args.window, sync_offset=args.sync_offset,
                    intermediate_path=intermediate)
```

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/test_trimcrop.py tests/test_crop_video.py tests/test_make_sync_video.py -q`
Expected: PASS, all tests including `test_render_clip_smoke` (excluded since Task 3, now unblocked).

- [ ] **Step 7: End-to-end verification against the reference recording**

This is the spec's verification section. `/tmp/ref_cropped.mp4` from Task 6 is the crop; put it where the fallback logic looks for it:

```bash
REC="Lickometry Data/ACG-26-3/raw_data_2026-07-13_11-59-47"
cp /tmp/ref_cropped.mp4 "${REC}_cropped.mp4"
python make_sync_video.py --h5 "${REC}.h5" --layout "Lickometry Data/ACG-26-3/layout_w_controls.csv" \
  --start 120 --end 130 --out /tmp/clip_cropped.mp4
```
Expected: no `note:` line (the cropped file was found), the video panel shows the region you chose, licks line up with the trace. Confirm the panel is the crop:
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 /tmp/clip_cropped_trimcrop.mp4
```
Expected: `360,360`

Then the fallback path:
```bash
mv "${REC}_cropped.mp4" /tmp/stashed_cropped.mp4
python make_sync_video.py --h5 "${REC}.h5" --layout "Lickometry Data/ACG-26-3/layout_w_controls.csv" \
  --start 120 --end 130 --out /tmp/clip_uncropped.mp4
```
Expected: prints `note: using uncropped video ... (no _cropped.mp4; run crop_video.py first)` to stderr, and still renders — with a full 1280x720 panel:
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 /tmp/clip_uncropped_trimcrop.mp4
```
Expected: `1280,720`

Compare `/tmp/clip_cropped.mp4` and `/tmp/clip_uncropped.mp4` by eye: the trace panel and lick alignment must be identical between them; only the video panel's framing differs. If they disagree, the anchor math diverged between the two paths — stop and debug before committing.

Clean up: `rm -f /tmp/stashed_cropped.mp4 /tmp/clip_*.mp4 /tmp/ref_cropped.mp4`

- [ ] **Step 8: Commit**

```bash
git add make_sync_video.py tests/test_make_sync_video.py
git commit -m "refactor: make_sync_video consumes the cropped video, drops trim+crop"
```

---

### Task 8: Document the two-step workflow

**Files:**
- Modify: `docs/VIDEO_CAPTURE.md`

**Interfaces:**
- Consumes: the finished CLI from Tasks 5-7.
- Produces: nothing.

- [ ] **Step 1: Read the existing doc to match its structure and tone**

Run: `cat docs/VIDEO_CAPTURE.md`

- [ ] **Step 2: Append the workflow section**

Add, adapting the heading level to the surrounding document:

```markdown
## Cropping and rendering a sync video

Two steps. Crop once per recording, then render as many clips as you like.

1. **Crop** — trims the video to the capacitance-recording window and crops it
   to a square you position by hand:

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
```

- [ ] **Step 3: Commit**

```bash
git add docs/VIDEO_CAPTURE.md
git commit -m "docs: crop + render workflow"
```
