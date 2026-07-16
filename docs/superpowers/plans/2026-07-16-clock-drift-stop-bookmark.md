# Clock-Drift Correction via Stop Bookmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second (Stop) video bookmark and consume both bookmarks as a two-point linear clock-rate fit, so cap↔video alignment stays tight across a multi-hour session instead of drifting.

**Architecture:** Acquisition records a Stop bookmark mirroring the existing Start bookmark. `video/trimcrop.py` replaces its scalar `video_base` timing anchor with a `SessionClock(pts_start_sec, latency, slope)` that maps a video PTS to session time τ; `slope` comes from the two bookmarks and defaults to 1.0, making every existing recording behave identically. `make_sync_video.py` and `crop_video.py` build one clock and thread it through.

**Tech Stack:** Python 3.13, numpy, h5py, pytest, ffmpeg/ffprobe (via subprocess + imageio), Solara (GUI event loop + daemon threads).

## Global Constraints

- Session time τ = host-seconds since `start_time` (τ = 0 ≡ Start click); the capacitance trace uses this axis.
- Timing formula (single source of truth): `τ(pts_sec) = latency + slope · (pts_sec − pts_start_sec)`.
- `slope` defaults to **1.0** and `latency` to **0.0** when their inputs are absent → existing recordings unchanged (byte-identical numbers at slope=1, latency=0).
- New HDF5 datasets are per-cycle with the existing suffix convention (`""` for cycle 0, `str(cycle)` otherwise) and are **omitted when the value is None** (older callers unaffected).
- Wireless bookmark round-trips run on daemon threads and never block or raise into acquisition/stop (mirror `docs/video-sync-alignment-bugs.md` fix 0).
- Reuse `trimcrop.bookmark_latency`; do not re-derive the latency or slope formula anywhere else.

---

### Task 1: Recorder writes the Stop bookmark datasets

**Files:**
- Modify: `recording/recorder.py` — `write_video_metadata` (currently at lines 250-288)
- Test: `tests/test_video_metadata.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `write_video_metadata(..., stop_frame_index=None, stop_pts=None, stop_host_before=None, stop_host_after=None)` writing per-cycle datasets `video_stop_frame_index{suffix}`, `video_stop_pts{suffix}`, `video_stop_bookmark_host_before{suffix}`, `video_stop_bookmark_host_after{suffix}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_video_metadata.py` (follow the existing fixture style in that file for building a `SensorRecorder` over a temp h5; reuse its existing helper if present):

```python
def test_write_video_metadata_writes_stop_bookmark_datasets(tmp_path):
    rec = _make_recorder(tmp_path)  # existing helper in this test module
    sensor_id = 1
    rec.write_video_metadata(
        sensor_id=sensor_id, frame_index=10, pts=1.0, video_filename="v.mp4",
        cycle=0,
        stop_frame_index=200, stop_pts=5.0,
        stop_host_before=1000.0, stop_host_after=1000.4)

    sn = rec._serial_for_sensor(sensor_id)
    import h5py
    with h5py.File(rec.filename, "r") as f:
        g = f[f"board_{sn}/sensor_{sensor_id}"]
        assert int(g["video_stop_frame_index"][()]) == 200
        assert float(g["video_stop_pts"][()]) == 5.0
        assert float(g["video_stop_bookmark_host_before"][()]) == 1000.0
        assert float(g["video_stop_bookmark_host_after"][()]) == 1000.4


def test_write_video_metadata_omits_stop_datasets_when_none(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.write_video_metadata(sensor_id=1, frame_index=10, pts=1.0,
                             video_filename="v.mp4", cycle=0)
    sn = rec._serial_for_sensor(1)
    import h5py
    with h5py.File(rec.filename, "r") as f:
        g = f[f"board_{sn}/sensor_1"]
        assert "video_stop_frame_index" not in g
        assert "video_stop_pts" not in g
```

If `_make_recorder` does not exist in the file, build the recorder the same way the file's other tests do (they already create a `SensorRecorder` + call `initialize_hdf5_file()`); do not invent a new pattern.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_video_metadata.py -k stop -v`
Expected: FAIL — `write_video_metadata() got an unexpected keyword argument 'stop_frame_index'`.

- [ ] **Step 3: Add the stop params and datasets**

In `recording/recorder.py`, change the `write_video_metadata` signature to add the four stop params, and extend the write loop. New signature:

```python
    def write_video_metadata(self, sensor_id: int, frame_index=None, pts=None,
                             video_filename=None, cycle=0,
                             pi_monotonic=None, host_time_before=None,
                             host_time_after=None,
                             stop_frame_index=None, stop_pts=None,
                             stop_host_before=None, stop_host_after=None):
```

Extend the existing `for base, value in (...)` tuple with the four stop datasets:

```python
            for base, value in (
                (f"video_frame_index{suffix}", frame_index),
                (f"video_pts{suffix}", pts),
                (f"video_filename{suffix}", video_filename),
                (f"video_pi_monotonic{suffix}", pi_monotonic),
                (f"video_bookmark_host_before{suffix}", host_time_before),
                (f"video_bookmark_host_after{suffix}", host_time_after),
                (f"video_stop_frame_index{suffix}", stop_frame_index),
                (f"video_stop_pts{suffix}", stop_pts),
                (f"video_stop_bookmark_host_before{suffix}", stop_host_before),
                (f"video_stop_bookmark_host_after{suffix}", stop_host_after),
            ):
```

Update the docstring to mention the stop bookmark records the second clock anchor for drift correction (one sentence).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_video_metadata.py -v`
Expected: PASS (new + all existing).

- [ ] **Step 5: Commit**

```bash
git add recording/recorder.py tests/test_video_metadata.py
git commit -m "feat: recorder writes stop-bookmark clock-anchor datasets"
```

---

### Task 2: trimcrop learns about drift (SessionClock, anchor fields, reader)

**Files:**
- Modify: `video/trimcrop.py` — `VideoAnchor` (147-164), `read_video_anchor` (167-194); add `SessionClock` + `session_clock`
- Test: `tests/test_trimcrop.py`

**Interfaces:**
- Consumes: `bookmark_latency` (existing).
- Produces:
  - `SessionClock` dataclass with fields `pts_start_sec: float, latency: float, slope: float` and method `session_time(pts_sec) -> np.ndarray` implementing `latency + slope*(pts_sec - pts_start_sec)`.
  - `session_clock(anchor, pts_ns) -> SessionClock`.
  - `VideoAnchor` gains `stop_frame_index: int | None = None`, `stop_host_before: float | None = None`, `stop_host_after: float | None = None`, and method `drift_slope(pts_ns) -> float` (1.0 when any needed field is absent or degenerate).
  - `read_video_anchor` populates the new fields from `video_stop_frame_index{suffix}` / `video_stop_bookmark_host_before{suffix}` / `video_stop_bookmark_host_after{suffix}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trimcrop.py`:

```python
def test_session_clock_slope1_matches_old_formula():
    import numpy as np
    pts_ns = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64) * 1_000_000
    # old video_base for frame_index 2 == 0.2 s; latency 0
    clock = tc.SessionClock(pts_start_sec=0.2, latency=0.0, slope=1.0)
    sess = clock.session_time(pts_ns / 1e9)
    # frame 2 sits at session 0.0, frame 5 at 0.3
    assert sess[2] == pytest.approx(0.0)
    assert sess[5] == pytest.approx(0.3)


def test_drift_slope_recovers_known_skew():
    import numpy as np
    # Video clock runs 1000 ppm fast vs host over the window.
    slope_true = 1.0 / 1.001  # host-seconds per video-second
    pts_ns = (np.arange(0, 1000) * 1_000_000).astype(np.int64)  # 0..0.999 s
    anchor = tc.VideoAnchor(
        sensor_number=1, video_filename="v.mp4", video_frame_index=0,
        start_time=1000.0, stop_time=1001.0,
        host_before=1000.0, host_after=1000.0,           # mid_start = 1000.0
        stop_frame_index=999,
        stop_host_before=1000.0 + 999e-3 * slope_true,
        stop_host_after=1000.0 + 999e-3 * slope_true)     # mid_stop
    assert anchor.drift_slope(pts_ns) == pytest.approx(slope_true, rel=1e-6)


def test_drift_slope_defaults_to_one_without_stop():
    import numpy as np
    pts_ns = (np.arange(0, 10) * 1_000_000).astype(np.int64)
    anchor = tc.VideoAnchor(
        sensor_number=1, video_filename="v.mp4", video_frame_index=0,
        start_time=1000.0, stop_time=1001.0,
        host_before=1000.0, host_after=1000.2)
    assert anchor.drift_slope(pts_ns) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trimcrop.py -k "session_clock or drift_slope" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'SessionClock'` / `VideoAnchor.__init__() got an unexpected keyword argument 'stop_frame_index'`.

- [ ] **Step 3: Add SessionClock, the builder, and the anchor fields**

In `video/trimcrop.py`, add after `bookmark_latency` (near line 88):

```python
@dataclass
class SessionClock:
    """Maps a video-clock PTS (seconds) to session time τ (host-seconds since
    start_time). τ(pts) = latency + slope·(pts − pts_start_sec).

    pts_start_sec : PTS of the Start-bookmark frame.
    latency       : host seconds the Start bookmark lagged start_time (0 if the
                    host bracket wasn't recorded); the Start frame sits at τ=latency.
    slope         : host-seconds per video-second from the two bookmarks (1.0 when
                    no Stop bookmark) — corrects clock drift across the session.
    """
    pts_start_sec: float
    latency: float
    slope: float

    def session_time(self, pts_sec):
        pts_sec = np.asarray(pts_sec, dtype=float)
        return self.latency + self.slope * (pts_sec - self.pts_start_sec)


def session_clock(anchor, pts_ns):
    """Build the SessionClock for a recording from its anchor and PTS sidecar."""
    pts_ns = np.asarray(pts_ns)
    pts_start_sec = float(pts_ns[anchor.video_frame_index]) / 1e9
    return SessionClock(pts_start_sec=pts_start_sec,
                        latency=anchor.latency,
                        slope=anchor.drift_slope(pts_ns))
```

Extend the `VideoAnchor` dataclass with the three optional stop fields (after `host_after`) and the `drift_slope` method (after the `latency` property):

```python
    stop_frame_index: int | None = None
    stop_host_before: float | None = None
    stop_host_after: float | None = None

    def drift_slope(self, pts_ns) -> float:
        """host-seconds per video-second from the two bookmarks; 1.0 when the
        Stop bookmark (or the Start host bracket) is absent, or degenerate."""
        if (self.stop_frame_index is None or self.stop_host_before is None
                or self.stop_host_after is None or self.host_before is None
                or self.host_after is None):
            return 1.0
        pts_ns = np.asarray(pts_ns)
        pts_start = float(pts_ns[self.video_frame_index]) / 1e9
        pts_stop = float(pts_ns[self.stop_frame_index]) / 1e9
        if pts_stop == pts_start:
            return 1.0
        mid_start = (self.host_before + self.host_after) / 2.0
        mid_stop = (self.stop_host_before + self.stop_host_after) / 2.0
        return (mid_stop - mid_start) / (pts_stop - pts_start)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trimcrop.py -k "session_clock or drift_slope" -v`
Expected: PASS.

- [ ] **Step 5: Write the failing reader test**

Add to `tests/test_trimcrop.py` (reuse the module's existing h5-building helper that `read_video_anchor` tests already use; the snippet below writes the datasets directly):

```python
def test_read_video_anchor_reads_stop_bookmark(tmp_path):
    import h5py
    p = tmp_path / "raw.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("board_FT0/sensor_1")
        g.create_dataset("time_data", data=[100.0, 200.0])
        g.create_dataset("start_time", data=150.0)
        g.create_dataset("stop_time", data=190.0)
        g.create_dataset("video_filename", data="v.mp4")
        g.create_dataset("video_frame_index", data=3)
        g.create_dataset("video_bookmark_host_before", data=150.0)
        g.create_dataset("video_bookmark_host_after", data=150.4)
        g.create_dataset("video_stop_frame_index", data=900)
        g.create_dataset("video_stop_bookmark_host_before", data=190.0)
        g.create_dataset("video_stop_bookmark_host_after", data=190.4)
    a = tc.read_video_anchor(str(p))
    assert a.stop_frame_index == 900
    assert a.stop_host_before == 190.0
    assert a.stop_host_after == 190.4


def test_read_video_anchor_stop_fields_none_when_absent(tmp_path):
    import h5py
    p = tmp_path / "raw.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("board_FT0/sensor_1")
        g.create_dataset("time_data", data=[100.0, 200.0])
        g.create_dataset("start_time", data=150.0)
        g.create_dataset("stop_time", data=190.0)
        g.create_dataset("video_filename", data="v.mp4")
        g.create_dataset("video_frame_index", data=3)
    a = tc.read_video_anchor(str(p))
    assert a.stop_frame_index is None
    assert a.stop_host_before is None
```

- [ ] **Step 6: Run to verify it fails**

Run: `python -m pytest tests/test_trimcrop.py -k read_video_anchor_reads_stop -v`
Expected: FAIL — `stop_frame_index` is None (reader doesn't populate it yet).

- [ ] **Step 7: Populate the stop fields in read_video_anchor**

In `read_video_anchor`, before the `return VideoAnchor(...)`, add suffix-scoped keys and pass them:

```python
        stop_idx_key = f"video_stop_frame_index{suffix}"
        stop_before_key = f"video_stop_bookmark_host_before{suffix}"
        stop_after_key = f"video_stop_bookmark_host_after{suffix}"
        return VideoAnchor(
            sensor_number=sensor_number,
            video_filename=fname,
            video_frame_index=int(group[f"video_frame_index{suffix}"][()]),
            start_time=start_time,
            stop_time=stop_time,
            host_before=(float(group[before_key][()])
                         if before_key in group else None),
            host_after=(float(group[after_key][()])
                        if after_key in group else None),
            stop_frame_index=(int(group[stop_idx_key][()])
                              if stop_idx_key in group else None),
            stop_host_before=(float(group[stop_before_key][()])
                              if stop_before_key in group else None),
            stop_host_after=(float(group[stop_after_key][()])
                             if stop_after_key in group else None),
        )
```

- [ ] **Step 8: Run to verify it passes**

Run: `python -m pytest tests/test_trimcrop.py -k read_video_anchor -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add video/trimcrop.py tests/test_trimcrop.py
git commit -m "feat: trimcrop SessionClock + drift slope from stop bookmark"
```

---

### Task 3: Refactor timing functions onto SessionClock; update consumers

**Files:**
- Modify: `video/trimcrop.py` — `frame_session_times` (90-93), `compute_trim_frames` (96-104), `trim_window_seconds` (110-126), `probe_frame_session_times` (129-144)
- Modify: `make_sync_video.py` — imports, `Recording`, `load_recording`, `clip_trim_window`, `render_clip`
- Modify: `crop_video.py` — imports, `compute_crop_window`
- Test: `tests/test_trimcrop.py`, `tests/test_make_sync_video.py`, `tests/test_crop_video.py`

**Interfaces:**
- Consumes: `SessionClock`, `session_clock` (Task 2).
- Produces (new signatures — every caller must switch):
  - `frame_session_times(clock, pts_ns)`
  - `compute_trim_frames(clock, pts_ns, start, end)`
  - `trim_window_seconds(clock, pts_ns, start, end, tail_margin=TAIL_MARGIN)`
  - `probe_frame_session_times(path, clock)`
  - `make_sync_video.Recording.clock: SessionClock` (replaces `video_base` + `bookmark_latency`)
  - `make_sync_video.clip_trim_window(rec, start, end) -> (sf, ef, start_sec, end_sec)` (drops the 5th `video_base_eff` return)

- [ ] **Step 1: Rewrite the trimcrop timing functions**

Replace the four functions in `video/trimcrop.py`:

```python
def frame_session_times(clock, pts_ns):
    """Session time τ (0 = start_time) of every sidecar frame, drift-corrected."""
    pts_ns = np.asarray(pts_ns)
    return clock.session_time(pts_ns / 1e9)


def compute_trim_frames(clock, pts_ns, start, end):
    """Inclusive (start_frame, stop_frame) of frames whose session time falls in
    [start, end]. Raises ValueError if the window has no frames."""
    sess = frame_session_times(clock, pts_ns)
    idx = np.flatnonzero((sess >= start) & (sess <= end))
    if idx.size == 0:
        raise ValueError(
            f"no video frames fall in the requested window [{start}, {end}] s")
    return int(idx[0]), int(idx[-1])


TAIL_MARGIN = 0.3  # seconds of slack past the last in-window frame


def trim_window_seconds(clock, pts_ns, start, end, tail_margin=TAIL_MARGIN):
    """Resolve session window [start, end] to (start_frame, stop_frame, start_sec,
    end_sec). The seconds are REAL original-video-timeline seconds (from raw PTS,
    NOT drift-scaled) — that is what ffmpeg seeking needs; only frame *selection*
    is drift-aware. crop_video and make_sync_video share this so their windows
    cannot drift apart."""
    pts_ns = np.asarray(pts_ns)
    sf, ef = compute_trim_frames(clock, pts_ns, start, end)
    start_sec = float(pts_ns[sf] - pts_ns[0]) / 1e9
    end_sec = float(pts_ns[ef] - pts_ns[0]) / 1e9 + tail_margin
    return sf, ef, start_sec, end_sec


def probe_frame_session_times(path, clock):
    """Session time of every frame in ``path`` from its real presentation
    timestamps (ffprobe). Trimmed clips keep original PTS (-copyts), so each
    frame self-reports its video-second; ``clock.session_time`` maps it to τ."""
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
    return clock.session_time(pts)
```

Note: the `TAIL_MARGIN` constant already exists above `trim_window_seconds`; keep a single definition (do not duplicate it — leave the original at its current location and don't re-add it if it's already there).

- [ ] **Step 2: Update the trimcrop timing tests to the new signature**

In `tests/test_trimcrop.py`, replace the old `(pts_ns, video_base)` calls. Build a clock from a base via a small local helper so the identity is explicit:

```python
def _clock(pts_start_sec, latency=0.0, slope=1.0):
    return tc.SessionClock(pts_start_sec=pts_start_sec, latency=latency, slope=slope)
```

Then convert the existing assertions. `video_base = compute_video_base(pts_ns, k)` corresponds to `pts_start_sec = pts_ns[k]/1e9` with `latency = 0`; a "shifted" `video_base - L` corresponds to `latency = L`. For example the existing frame-session test becomes:

```python
def test_frame_session_times_zeroes_at_bookmark():
    import numpy as np
    pts_ns = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64) * 1_000_000
    clock = _clock(pts_start_sec=float(pts_ns[2]) / 1e9)  # frame 2 is the bookmark
    sess = tc.frame_session_times(clock, pts_ns)
    assert sess[2] == pytest.approx(0.0)
    sf, ef = tc.compute_trim_frames(clock, pts_ns, 0.0, 0.3)
    assert (sf, ef) == (2, 5)
```

And the `trim_window_seconds` drift/latency tests (currently passing `vb` and `vb - 0.25`) become two clocks with `latency=0.0` and `latency=0.25`:

```python
def test_smaller_base_labels_frames_later():
    import numpy as np
    pts_ns = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64) * 1_000_000
    ps = float(pts_ns[2]) / 1e9
    plain = tc.trim_window_seconds(_clock(ps, latency=0.0), pts_ns, 0.0, 0.3)
    shifted = tc.trim_window_seconds(_clock(ps, latency=0.25), pts_ns, 0.0, 0.3)
    assert shifted[0] <= plain[0]  # later labels pull earlier frames into [0,0.3]
```

Update every remaining `tc.frame_session_times(pts_ns, ...)`, `tc.compute_trim_frames(pts_ns, ...)`, `tc.trim_window_seconds(pts_ns, ...)` call in the file to pass a `_clock(...)` first arg. The degenerate-window test (`tc.trim_window_seconds(pts_ns, 0.0, 0.0, -400.0)`) becomes `tc.trim_window_seconds(_clock(0.0), pts_ns, 0.0, -400.0)` and still expects `ValueError`.

- [ ] **Step 3: Add the drift identity + non-unit-slope test**

```python
def test_probe_and_frame_times_agree_under_slope(monkeypatch):
    import numpy as np
    pts_ns = (np.arange(0, 600) * 1_000_000).astype(np.int64)  # 0..0.599 s
    clock = tc.SessionClock(pts_start_sec=0.1, latency=0.05, slope=1.002)
    # frame at index 300 -> pts 0.3 s -> τ = 0.05 + 1.002*(0.3-0.1)
    sess = tc.frame_session_times(clock, pts_ns)
    assert sess[300] == pytest.approx(0.05 + 1.002 * (0.3 - 0.1))
    # probe uses the same transform on ffprobe seconds
    monkeypatch.setattr(tc.subprocess, "run", lambda *a, **k: _fake_ffprobe([0.3]))
    got = tc.probe_frame_session_times("clip.mp4", clock)
    assert got[0] == pytest.approx(0.05 + 1.002 * (0.3 - 0.1))
```

Add a `_fake_ffprobe(seconds)` helper returning an object with `returncode=0` and `stdout="\n".join(str(s) for s in seconds)`, or reuse an equivalent helper if `tests/test_trimcrop.py` already fakes ffprobe.

- [ ] **Step 4: Run trimcrop tests**

Run: `python -m pytest tests/test_trimcrop.py -v`
Expected: PASS (all, including the converted ones).

- [ ] **Step 5: Update make_sync_video imports and Recording**

In `make_sync_video.py`, change the trimcrop import block: drop `compute_video_base`, `frame_session_times`, `bookmark_latency`; add `SessionClock`, `session_clock`. Keep `compute_trim_frames`, `probe_frame_session_times`, `trim_window_seconds`, `read_session_window`, `read_video_anchor`, `resolve_paths`, `subclip_copy`, `trim_and_crop`, `find_video_sensor`, `probe_start_pts`.

Replace the `Recording` dataclass fields `video_base: float` and `bookmark_latency: float` with:

```python
    clock: SessionClock
```

(remove both `video_base` and `bookmark_latency` fields; keep `pts_ns`).

- [ ] **Step 6: Update load_recording**

Replace the tail of `load_recording` (the `video_base = compute_video_base(...)` line and the `Recording(...)` construction):

```python
    pts_ns = np.loadtxt(pts_txt_path, dtype=np.int64)
    clock = session_clock(anchor, pts_ns)

    return Recording(
        animal=animal, sensor=anchor.sensor_number, cap=cap, time=time,
        lick_times=np.asarray(lick_times), lick_indices=lick_indices,
        lick_vals=lick_vals,
        clock=clock, video_path=video_path,
        session_duration=session_duration, pts_ns=pts_ns,
    )
```

(remove `video_frame_index`/`latency` locals that were only feeding `video_base`/`bookmark_latency`; `anchor` is already in scope.)

- [ ] **Step 7: Update clip_trim_window and render_clip**

Replace `clip_trim_window`:

```python
def clip_trim_window(rec, start, end):
    """Video-file window (start_frame, stop_frame, start_sec, end_sec) for the
    clip's session window [start, end]. Drift + latency live in rec.clock, shared
    with crop_video's compute_crop_window so the two windows cannot diverge."""
    return trim_window_seconds(rec.clock, rec.pts_ns, start, end)
```

In `render_clip`, update the unpack and the probe call:

```python
    _, _, start_sec, end_sec = clip_trim_window(rec, start, end)
    ...
    subclip_copy(rec.video_path, start_sec, end_sec, intermediate_path)
    frame_sess = probe_frame_session_times(intermediate_path, rec.clock)
```

- [ ] **Step 8: Update crop_video.compute_crop_window**

In `crop_video.py`, change imports (drop `compute_video_base`; add `session_clock`) and replace:

```python
def compute_crop_window(anchor, pts_ns):
    """(start_frame, stop_frame, start_sec, end_sec) covering the whole session.
    Uses the same SessionClock (latency + drift) as make_sync_video's
    clip_trim_window, so crop and render select identical frames."""
    clock = session_clock(anchor, pts_ns)
    return trim_window_seconds(clock, pts_ns, 0.0, anchor.session_duration)
```

- [ ] **Step 9: Update make_sync_video + crop_video tests to the new signatures**

In `tests/test_make_sync_video.py` and `tests/test_crop_video.py`, every `compute_trim_frames(pts_ns, vb, ...)` / `probe_frame_session_times(out, vb)` / `trim_window_seconds(pts_ns, vb, ...)` call becomes clock-based. Where a test built `vb = msv.compute_video_base(pts_ns, k)`, replace with `clock = tc.SessionClock(pts_start_sec=float(pts_ns[k])/1e9, latency=0.0, slope=1.0)` (import trimcrop as `tc`) and pass `clock` as the first arg. Where a test asserted on `rec.video_base` (e.g. `rec.video_base == pytest.approx(31.97, ...)`), assert on `rec.clock.pts_start_sec` instead (same numeric value, since latency defaults to 0 on those fixtures). For the latency test that checked `video_base_eff` as the 5th return of `clip_trim_window`, drop the 5th-element assertion and instead assert the selected `start_frame`/`start_sec` shifts as expected, or assert on `rec.clock.latency`. Keep `compute_video_base` as-is in trimcrop (still unit-tested directly); it is simply no longer on the render path.

- [ ] **Step 10: Run the full render/crop suites**

Run: `python -m pytest tests/test_trimcrop.py tests/test_make_sync_video.py tests/test_crop_video.py -v`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add video/trimcrop.py make_sync_video.py crop_video.py \
  tests/test_trimcrop.py tests/test_make_sync_video.py tests/test_crop_video.py
git commit -m "refactor: SessionClock drives trim/render timing (drift-aware)"
```

---

### Task 4: Acquisition fires the Stop bookmark

**Files:**
- Modify: `components/sensor_card.py` — add `bookmark_stop`; call it in `stop_sensor` (112-152)
- Modify: `components/session_controls.py` — `stop_recording` (178-262): fire stop bookmark for the camera sensor and join it before `STOP_SESSION`
- Test: `tests/test_camera_session_lifecycle.py` (and/or `tests/test_camera_integration.py`)

**Interfaces:**
- Consumes: `recorder.write_video_metadata(..., stop_frame_index, stop_pts, stop_host_before, stop_host_after)` (Task 1).
- Produces: `sensor_card.bookmark_stop(sensor_id, cycle) -> threading.Thread | None` — fires a daemon bookmark round-trip and writes the stop datasets; returns the thread (join-able) or None when this sensor isn't the camera driver / no client.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_camera_session_lifecycle.py` (reuse the module's existing fakes for `state`, `camera_client`, and `current_recorder`; the existing start-bookmark test in `tests/test_camera_integration.py` shows the pattern for a fake client returning a bookmark dict and a recorder capturing kwargs — mirror it):

```python
def test_bookmark_stop_writes_stop_datasets(monkeypatch, fake_camera_env):
    # fake_camera_env: camera_enabled=True, camera_sensor_id=1, a fake client
    # whose bookmark() returns {"ok": True, "frame_index": 900, "pts": 5.0},
    # and a recorder that records write_video_metadata kwargs.
    from components import sensor_card
    t = sensor_card.bookmark_stop(sensor_id=1, cycle=0)
    assert t is not None
    t.join(timeout=2.0)
    kw = fake_camera_env.recorder.last_video_metadata_kwargs
    assert kw["stop_frame_index"] == 900
    assert kw["stop_pts"] == 5.0
    assert kw["cycle"] == 0
    assert "stop_host_before" in kw and "stop_host_after" in kw


def test_bookmark_stop_returns_none_for_non_camera_sensor(fake_camera_env):
    from components import sensor_card
    assert sensor_card.bookmark_stop(sensor_id=2, cycle=0) is None
```

If `fake_camera_env` doesn't exist, build the fakes inline the way the existing camera tests do (set `state.camera_enabled`, `state.camera_sensor_id`, monkeypatch `components.session_controls.camera_client` and `.current_recorder`).

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_camera_session_lifecycle.py -k bookmark_stop -v`
Expected: FAIL — `AttributeError: module 'components.sensor_card' has no attribute 'bookmark_stop'`.

- [ ] **Step 3: Implement bookmark_stop**

Add to `components/sensor_card.py`:

```python
def bookmark_stop(sensor_id: int, cycle: int):
    """Bookmark the video at the stop of the camera sensor's cycle (second clock
    anchor for drift correction). Mirrors start_sensor: runs the wireless
    round-trip on a daemon thread so stop never blocks, brackets it with host
    wall-clock, and writes the stop datasets. Returns the thread (join-able), or
    None when this sensor isn't the camera driver or no client is connected."""
    if not (state.camera_enabled.value and sensor_id == state.camera_sensor_id.value):
        return None
    from components import session_controls
    client = session_controls.camera_client
    recorder = session_controls.current_recorder
    if client is None or recorder is None:
        return None

    def _bookmark():
        try:
            host_before = time_module.time()
            resp = client.bookmark(sensor_id)
            host_after = time_module.time()
            if resp.get("ok"):
                recorder.write_video_metadata(
                    sensor_id=sensor_id, cycle=cycle,
                    stop_frame_index=resp.get("frame_index"),
                    stop_pts=resp.get("pts"),
                    stop_host_before=host_before,
                    stop_host_after=host_after)
                state.add_log_message(
                    f"Sensor {sensor_id}: video stop bookmark "
                    f"frame={resp.get('frame_index')}")
            else:
                state.add_log_message(
                    f"WARNING: Sensor {sensor_id}: stop bookmark failed: "
                    f"{resp.get('error')}")
        except Exception as exc:
            state.add_log_message(
                f"WARNING: Sensor {sensor_id}: stop bookmark error: {exc}")

    t = threading.Thread(target=_bookmark, daemon=True)
    t.start()
    return t
```

In `stop_sensor`, after the `write_sensor_metadata(stop_time=...)` call (around line 142) and before `state.sensor_states.set(sensors)`, fire it (fire-and-forget; the camera is still active on an individual stop):

```python
    bookmark_stop(sensor_id, current_cycle)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_camera_session_lifecycle.py -k bookmark_stop -v`
Expected: PASS.

- [ ] **Step 5: Write the failing global-stop ordering test**

The global stop path must capture the stop bookmark BEFORE `STOP_SESSION` (else the Pi session is gone). Add to `tests/test_camera_session_lifecycle.py`:

```python
def test_global_stop_bookmarks_before_stop_session(monkeypatch, fake_camera_env):
    """stop_recording must obtain the stop bookmark before the camera receives
    STOP_SESSION, else there is no active session to bookmark."""
    order = []
    fake_camera_env.client.on_bookmark = lambda: order.append("bookmark")
    fake_camera_env.client.on_stop_session = lambda: order.append("stop_session")
    fake_camera_env.recording_sensor(1)  # sensor 1 (camera) still recording

    from components import session_controls
    session_controls.stop_recording()
    # let the daemon stop/fetch thread run
    session_controls._join_camera_threads_for_test(timeout=2.0)

    assert order.index("bookmark") < order.index("stop_session")
```

If the test module lacks `recording_sensor`/`_join_camera_threads_for_test` helpers, implement the check with whatever join hook the existing stop tests use; the essential assertion is bookmark-before-stop_session ordering.

- [ ] **Step 6: Run to verify it fails**

Run: `python -m pytest tests/test_camera_session_lifecycle.py -k global_stop_bookmarks_before -v`
Expected: FAIL — bookmark not fired, or fired after stop_session.

- [ ] **Step 7: Wire the global stop path**

In `components/session_controls.py` `stop_recording`, inside the loop that stops still-recording sensors (around lines 190-198), capture the camera-sensor stop-bookmark thread before incrementing/clearing state:

```python
    stop_bookmark_thread = None
    sensors = state.sensor_states.value.copy()
    for sensor_id, sensor in sensors.items():
        if sensor.is_recording:
            if current_recorder:
                current_recorder.write_sensor_metadata(
                    sensor_id=sensor_id, stop_time=time.time(),
                    cycle=sensor.recording_cycle)
            from components.sensor_card import bookmark_stop
            t = bookmark_stop(sensor_id, sensor.recording_cycle)
            if t is not None:
                stop_bookmark_thread = t
            sensors[sensor_id] = replace(
                sensor, is_recording=False, status="idle",
                recording_cycle=sensor.recording_cycle + 1)
```

Then pass the thread into `_camera_stop_and_fetch` so it joins before `STOP_SESSION`:

```python
        def _camera_stop_and_fetch(client, out_dir, bookmark_thread):
            try:
                if bookmark_thread is not None:
                    bookmark_thread.join(timeout=5.0)  # bookmark the Pi BEFORE it stops
                resp = client.stop_session()
                ...
        threading.Thread(target=_camera_stop_and_fetch,
                         args=(_client, _out_dir, stop_bookmark_thread),
                         daemon=True).start()
```

Add a test-only join hook if the test needs one (e.g. store the last spawned thread on a module global the test can join), matching how the module already exposes threads for tests.

- [ ] **Step 8: Run to verify it passes**

Run: `python -m pytest tests/test_camera_session_lifecycle.py -v`
Expected: PASS.

- [ ] **Step 9: Full suite**

Run: `python -m pytest -q`
Expected: PASS (no regressions).

- [ ] **Step 10: Commit**

```bash
git add components/sensor_card.py components/session_controls.py \
  tests/test_camera_session_lifecycle.py
git commit -m "feat: fire stop bookmark on sensor/global stop for drift fit"
```

---

## Self-Review

**Spec coverage:**
- Acquisition stop bookmark (sensor_card + session_controls + recorder) → Tasks 1, 4. ✓
- Timing formula / SessionClock / slope default 1.0 → Task 2, 3. ✓
- VideoAnchor stop fields + drift_slope + read_video_anchor → Task 2. ✓
- Consumers (make_sync_video, crop_video) → Task 3. ✓
- Backward compat (slope=1 identity, absent datasets) → Task 2 (drift_slope default), Task 3 (identity test), Task 1 (omit-when-None). ✓
- Degenerate fit guard → Task 2 `drift_slope` (`pts_stop == pts_start`). ✓
- Global-stop ordering (bookmark before STOP_SESSION) → Task 4 Steps 5-7. ✓
- Testing items 1-6 from spec → covered across Tasks 1-4. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. Test-helper reuse notes point to concrete existing patterns rather than inventing.

**Type consistency:** `SessionClock(pts_start_sec, latency, slope)` and `session_clock(anchor, pts_ns)` used identically in Tasks 2/3. `bookmark_stop(sensor_id, cycle) -> Thread | None` defined in Task 4 Step 3, consumed in `stop_sensor` and `stop_recording` with matching args. `write_video_metadata` stop kwargs named identically in Tasks 1, 4. Timing functions' new signatures `(clock, pts_ns, ...)` consistent across trimcrop and both consumers.
