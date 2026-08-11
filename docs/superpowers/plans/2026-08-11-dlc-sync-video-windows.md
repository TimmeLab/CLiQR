# DLC Windows -> Synced Videos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--dlc-windows CSV` option to `scripts/find_interesting_windows.py` that turns DeepLabCut window-selection rows into `make_sync_video.py` commands, rendering from the ORIGINAL recording (never the DLC labeled video).

**Architecture:** A new "pure DLC mode" short-circuits the existing capacitance-trace search. DLC rows carry frame indices into the original video container; they are converted to session seconds with the exact `SessionClock` + PTS sidecar that `make_sync_video.py` uses to time frames, then mapped to a combined-results cycle by matching the video's basename against each cycle's `raw_h5` attribute. Output flows through the existing CSV and shell-script writers unchanged.

**Tech Stack:** Python 3, numpy, h5py, pytest. Reuses `video/trimcrop.py` (`read_video_anchor`, `resolve_paths`, `session_clock`, `frame_session_times`) and `make_sync_video.load_container_pts`.

**Spec:** `docs/superpowers/specs/2026-08-11-dlc-sync-video-windows-design.md`

## Global Constraints

- All new logic lives in `scripts/find_interesting_windows.py`. Do NOT modify `dlc_integration/find_dlc_windows.py`, its CSV schema, `make_sync_video.py`, or `video/trimcrop.py`.
- `CSV_COLUMNS` in `scripts/find_interesting_windows.py` must NOT change.
- Category string for DLC regions is exactly `"dlc"`. Only `"climb"` gets `--no-crop`, so DLC clips keep the crop with no extra code.
- DLC CSV `end_frame` is EXCLUSIVE (`Frames2plot=range(start_frame, end_frame)`); the last in-window frame is `end_frame - 1`.
- Never derive session time from the DLC CSV's `start_sec`/`end_sec` columns (they are `frame / fps`, video-file time, and drift against session time by seconds over a session). Always go through `frame_session_times`.
- Windows are used as-is: no padding, no minimum width, no cap on how many are emitted.
- Every skipped row/video must be counted by reason and reported; nothing is silently dropped.
- Tests are pure/self-contained: no real recordings, no ffmpeg, no network. Existing test style: `tests/test_find_interesting_windows.py`, plain `assert`, numpy for arrays, `sys.path.insert` header already present at the top of that file.
- Run tests with: `python -m pytest tests/test_find_interesting_windows.py -v`

## File Structure

| File | Responsibility |
|---|---|
| `scripts/find_interesting_windows.py` (modify) | All new functions + CLI flag. Keep the new code in one clearly-marked "DLC windows" section placed AFTER the existing provenance section and BEFORE "Output writers". |
| `tests/test_find_interesting_windows.py` (modify) | New unit tests appended, following the existing file's style. |
| `docs/superpowers/specs/2026-08-11-dlc-sync-video-windows-design.md` (read only) | The approved design. |

New functions, in dependency order:

| Function | Purpose |
|---|---|
| `parse_dlc_video_stem(video_path)` | `-> (raw_stem, is_cfr)`; basename minus extension, minus one trailing `_cfr`. |
| `read_dlc_windows(path)` | Read the DLC CSV into a list of plain dicts with parsed numbers. |
| `frame_window_to_session(sess, start_frame, end_frame)` | `-> (start_s, end_s)` or `None` when `start_frame` is past the sidecar. |
| `clamp_to_trace(start_s, end_s, first_s, span_s)` | `-> (start_s, end_s)` clipped, or `None` when disjoint from the trace. |
| `load_frame_session_times(raw_h5_path)` | I/O: anchor + PTS sidecar -> per-frame session times, or `None` if anything is missing. |
| `build_dlc_rois(dlc_rows, cycles, sess_loader=...)` | `-> (rows, skips)`; the whole selection, injectable loader so it unit-tests with no files. |
| `build_cycles_for_dlc(combined_h5_path, raw_map, animals)` | I/O: `{raw_stem: cycle info dict}` from the combined results file. |
| `write_outputs(all_rows, args)` | Extracted tail of `main` (sort + write CSV + write .sh + print), shared by both modes. |

---

### Task 1: Parse the DLC CSV and its video stems

**Files:**
- Modify: `scripts/find_interesting_windows.py` (new "DLC windows" section, after `is_restart_recording`, before the `# Output writers` banner)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `parse_dlc_video_stem(video_path: str) -> tuple[str, bool]` — `(raw_stem, is_cfr)`.
  - `read_dlc_windows(path: str) -> list[dict]` — each dict has keys `label` (str), `video` (str), `start_frame` (int), `end_frame` (int), `tongue_rate` (float).

- [ ] **Step 1: Write the failing tests**

Add to the import block at the top of `tests/test_find_interesting_windows.py` (extend the existing `from scripts.find_interesting_windows import (...)` list):

```python
    parse_dlc_video_stem, read_dlc_windows,
```

Append to the end of the file:

```python
# ---------------------------------------------------------------------------
# DLC window mode
# ---------------------------------------------------------------------------
def test_parse_dlc_video_stem_strips_cfr():
    # DLC ran on a CFR re-encode: the stem the raw .h5 is named after is the one WITHOUT _cfr,
    # and the caller needs to know it was a re-encode (those rows are not renderable).
    stem, is_cfr = parse_dlc_video_stem(
        "/N/lustre/project/proj-530/videos/raw_data_2026-07-13_11-59-47_cfr.mp4")
    assert stem == "raw_data_2026-07-13_11-59-47"
    assert is_cfr is True


def test_parse_dlc_video_stem_leaves_original_alone():
    stem, is_cfr = parse_dlc_video_stem("videos/raw_data_2026-07-24_12-02-14.mp4")
    assert stem == "raw_data_2026-07-24_12-02-14"
    assert is_cfr is False


def test_parse_dlc_video_stem_only_strips_a_trailing_cfr():
    # "_cfr" in the middle of a name is part of the name, not the re-encode marker.
    stem, is_cfr = parse_dlc_video_stem("raw_data_cfr_test_2026-07-24.mp4")
    assert stem == "raw_data_cfr_test_2026-07-24"
    assert is_cfr is False


def test_read_dlc_windows_parses_numbers(tmp_path):
    csv_path = tmp_path / "windows.csv"
    csv_path.write_text(
        "task_id,label,video,h5,scorer,bodypart,start_frame,end_frame,n_frames,"
        "start_sec,end_sec,duration_sec,frac_above,mean_likelihood,mean_nose_dist,"
        "min_nose_dist,tongue_rate\n"
        "1,w000,/videos/raw_data_2026-07-24_12-02-14.mp4,/p.h5,SCORER,nose,"
        "2606,2799,193,21.717,23.325,1.608,0.1347,0.4471,45.37,20.29,7.119\n"
    )
    rows = read_dlc_windows(str(csv_path))
    assert len(rows) == 1
    assert rows[0]["label"] == "w000"
    assert rows[0]["video"] == "/videos/raw_data_2026-07-24_12-02-14.mp4"
    assert rows[0]["start_frame"] == 2606
    assert rows[0]["end_frame"] == 2799
    assert rows[0]["tongue_rate"] == 7.119


def test_read_dlc_windows_empty_tongue_rate_is_zero(tmp_path):
    # tongue_rate is written on every run, but a hand-edited or older CSV may leave it blank.
    csv_path = tmp_path / "windows.csv"
    csv_path.write_text(
        "task_id,label,video,start_frame,end_frame,tongue_rate\n"
        "1,w000,/videos/v.mp4,10,20,\n"
    )
    rows = read_dlc_windows(str(csv_path))
    assert rows[0]["tongue_rate"] == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k dlc -v`
Expected: FAIL at import — `ImportError: cannot import name 'parse_dlc_video_stem'`

- [ ] **Step 3: Write the implementation**

Insert into `scripts/find_interesting_windows.py`, immediately after `is_restart_recording` and before the `# Output writers` banner comment:

```python
# ---------------------------------------------------------------------------
# DLC window mode
# ---------------------------------------------------------------------------
# `dlc_integration/find_dlc_windows.py` writes one row per stretch of video where the animal was
# confidently at the sipper. Those rows are frame ranges into the ORIGINAL recording's container
# (half-open, `[start_frame, end_frame)`), which is a different time reference from the one
# make_sync_video takes -- see `frame_window_to_session` for the conversion.


def parse_dlc_video_stem(video_path):
    """(raw_stem, is_cfr) for a DLC CSV `video` path.

    DLC was run on a CFR re-encode for the older sessions (`<stem>_cfr.mp4`) and on the original
    recording for the newer ones. The raw `.h5` is always named after the stem WITHOUT `_cfr`, so
    strip it -- but report that we did, because a re-encode's frame indices are NOT guaranteed to be
    the original container's ordinals (`-fps_mode cfr` drops frames to hit a flat rate), which makes
    those rows unrenderable rather than merely inconvenient.
    """
    stem = os.path.splitext(os.path.basename(str(video_path)))[0]
    if stem.endswith("_cfr"):
        return stem[: -len("_cfr")], True
    return stem, False


def read_dlc_windows(path):
    """Read a find_dlc_windows.py CSV into plain dicts with the fields this script needs.

    Only `label`, `video`, `start_frame`, `end_frame` and `tongue_rate` are used; the rest of the
    columns (likelihoods, distances, the derived seconds) are diagnostics for a human reading that
    CSV. `tongue_rate` is blank on a row written before the column existed, which reads as 0.0.
    """
    rows = []
    with open(path, "r", newline="") as f:
        for record in csv.DictReader(f):
            rate = (record.get("tongue_rate") or "").strip()
            rows.append({
                "label": record.get("label", ""),
                "video": record.get("video", ""),
                "start_frame": int(record["start_frame"]),
                "end_frame": int(record["end_frame"]),
                "tongue_rate": float(rate) if rate else 0.0,
            })
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS (all pre-existing tests still pass too)

- [ ] **Step 5: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): read find_dlc_windows CSVs and parse their video stems"
```

---

### Task 2: Convert a frame range to a session-time window

**Files:**
- Modify: `scripts/find_interesting_windows.py` (same DLC section, after `read_dlc_windows`)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `frame_window_to_session(sess, start_frame, end_frame) -> tuple[float, float] | None`
  - `clamp_to_trace(start_s, end_s, first_s, span_s) -> tuple[float, float] | None`

- [ ] **Step 1: Write the failing tests**

Extend the import list in `tests/test_find_interesting_windows.py` with:

```python
    frame_window_to_session, clamp_to_trace,
```

Add this import near the top of the test file, below the existing `sys.path.insert` line:

```python
from video.trimcrop import SessionClock, frame_session_times  # noqa: E402
```

Append to the end of the file:

```python
def _fake_session_times(n_frames=100, latency=1.5, slope=1.002, fps=120.0):
    """Per-frame session times built the way the renderer builds them: absolute SensorTimestamps
    through a SessionClock that carries a bookmark latency and a drift slope."""
    pts_ns = (np.arange(n_frames) / fps * 1e9).astype(np.int64) + 123_456_789
    clock = SessionClock(pts_start_sec=0.0, latency=latency, slope=slope)
    return frame_session_times(clock, pts_ns)


def test_frame_window_to_session_matches_frame_session_times():
    # The window must be the session time of the frames DLC scored -- latency and drift included,
    # not frame/fps. sess[k] is the reference the renderer itself uses to place frame k.
    sess = _fake_session_times()
    start, end = frame_window_to_session(sess, 10, 20)
    assert start == sess[10]
    # end_frame is EXCLUSIVE in the DLC CSV: the last frame in the window is 19.
    assert end == sess[19]
    # And that is emphatically not frame/fps: the clock's latency alone shifts it by 1.5 s.
    assert start > 10 / 120.0 + 1.0


def test_frame_window_to_session_clamps_end_past_sidecar():
    sess = _fake_session_times(n_frames=50)
    start, end = frame_window_to_session(sess, 40, 999)
    assert start == sess[40]
    assert end == sess[-1]


def test_frame_window_to_session_returns_none_when_start_past_sidecar():
    sess = _fake_session_times(n_frames=50)
    assert frame_window_to_session(sess, 50, 60) is None


def test_clamp_to_trace_clips_partial_overlap():
    assert clamp_to_trace(95.0, 110.0, 0.0, 100.0) == (95.0, 100.0)
    assert clamp_to_trace(-5.0, 10.0, 0.0, 100.0) == (0.0, 10.0)


def test_clamp_to_trace_returns_none_when_disjoint():
    assert clamp_to_trace(120.0, 130.0, 0.0, 100.0) is None
    assert clamp_to_trace(-20.0, -10.0, 0.0, 100.0) is None


def test_clamp_to_trace_returns_none_when_clamped_window_is_empty():
    # Touching the edge leaves no time to render.
    assert clamp_to_trace(100.0, 110.0, 0.0, 100.0) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k "session or clamp" -v`
Expected: FAIL at import — `ImportError: cannot import name 'frame_window_to_session'`

- [ ] **Step 3: Write the implementation**

Append to the DLC section of `scripts/find_interesting_windows.py`:

```python
def frame_window_to_session(sess, start_frame, end_frame):
    """Session-time window (start_s, end_s) for a DLC frame range, or None if it starts past the
    end of the recording's per-frame time array.

    `sess` is `frame_session_times(clock, container_pts_ns)`: the session time of every container
    frame, latency- and drift-corrected. Indexing it is the whole conversion -- do NOT compute
    `start_frame / fps`, which is video-file time and drifts against session time by seconds over a
    long recording (see docs/video-sync-alignment-bugs.md).

    `end_frame` is EXCLUSIVE (the CSV mirrors `Frames2plot=range(start, end)`), so the window ends
    at frame `end_frame - 1`, clamped to the last frame we have a time for.
    """
    sess = np.asarray(sess, dtype=np.float64)
    if sess.size == 0 or start_frame >= sess.size:
        return None
    last = min(int(end_frame) - 1, sess.size - 1)
    if last < start_frame:
        return None
    return float(sess[int(start_frame)]), float(sess[last])


def clamp_to_trace(start_s, end_s, first_s, span_s):
    """Clip a window to the trace's own [first_s, span_s], or None if nothing is left.

    A window can fall outside the trace because the video and the capacitance recording do not
    start and stop at exactly the same instant. Clipping keeps a partially-overlapping window
    renderable (make_sync_video rejects an --end past the session), and dropping a disjoint one
    keeps a clip with no trace to draw out of the shell script.
    """
    start_s = max(float(start_s), float(first_s))
    end_s = min(float(end_s), float(span_s))
    if end_s <= start_s:
        return None
    return start_s, end_s
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): convert DLC frame ranges to session-time windows"
```

---

### Task 3: Load a recording's per-frame session times

**Files:**
- Modify: `scripts/find_interesting_windows.py` (DLC section, after `clamp_to_trace`; also the import block at the top of the file)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `load_frame_session_times(raw_h5_path) -> np.ndarray | None` — per-container-frame session times, or `None` when the raw `.h5`, the video, or the PTS sidecar is missing/unreadable.

- [ ] **Step 1: Write the failing tests**

Extend the import list in `tests/test_find_interesting_windows.py` with:

```python
    load_frame_session_times,
```

Append to the end of the file:

```python
def test_load_frame_session_times_missing_h5_returns_none():
    # Provenance pointing at a file that isn't on this machine is the normal case for an older
    # combined file, so it must degrade to "no commands for this video", never to a traceback.
    assert load_frame_session_times("/nonexistent/raw_data_2026-01-01_00-00-00.h5") is None


def test_load_frame_session_times_none_path_returns_none():
    assert load_frame_session_times(None) is None


def test_load_frame_session_times_missing_pts_sidecar_returns_none(tmp_path):
    # A raw .h5 with no video sensor at all: read_video_anchor raises, and we swallow it.
    import h5py
    h5_path = tmp_path / "raw_data_2026-01-01_00-00-00.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_group("board0")
    assert load_frame_session_times(str(h5_path)) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k load_frame_session -v`
Expected: FAIL at import — `ImportError: cannot import name 'load_frame_session_times'`

- [ ] **Step 3: Write the implementation**

First extend the existing trimcrop import near the top of `scripts/find_interesting_windows.py`. Replace:

```python
from video.trimcrop import find_video_sensor, read_video_anchor, _resolve_cycle  # noqa: E402
```

with:

```python
from video.trimcrop import (  # noqa: E402
    find_video_sensor, read_video_anchor, _resolve_cycle,
    frame_session_times, resolve_paths, session_clock,
)
```

`load_container_pts` comes from `make_sync_video`, but it is imported LAZILY inside
`load_frame_session_times` (see below), not at module scope: importing `make_sync_video` pulls in
matplotlib, imageio and pandas, and the trace-search mode has no use for any of them.

Then append to the DLC section:

```python
def load_frame_session_times(raw_h5_path):
    """Session time of every container frame of a recording, or None if we cannot get it.

    This is the same three-step the renderer does: read the video anchor out of the raw .h5, build
    the SessionClock (bookmark latency + two-bookmark drift slope), and time every frame of the
    PTS sidecar with it. Returning None -- rather than raising -- for a missing raw file, missing
    video sensor, or missing sidecar is deliberate: a combined file routinely names recordings that
    are not on this machine, and those videos simply produce no clips.
    """
    if not raw_h5_path or not os.path.exists(raw_h5_path):
        return None
    # Lazy: make_sync_video drags in matplotlib/imageio/pandas, which the trace-search mode has no
    # use for. It owns the "which sidecar times container frames" rule (the Pi's encoded-frame
    # sidecar when present, else the capture sidecar); reuse it so a window computed here always
    # selects the frames the renderer will place.
    from make_sync_video import load_container_pts
    try:
        anchor = read_video_anchor(raw_h5_path)
        _, pts_txt = resolve_paths(raw_h5_path, anchor)
        if not os.path.exists(pts_txt):
            return None
        pts_ns = np.loadtxt(pts_txt, dtype=np.int64)
        if np.asarray(pts_ns).size < 2:
            return None
        clock = session_clock(anchor, pts_ns)
        return frame_session_times(clock, load_container_pts(pts_txt, pts_ns))
    except (ValueError, KeyError, OSError, IndexError):
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): load per-frame session times for a raw recording"
```

---

### Task 4: Build DLC regions of interest with counted skips

**Files:**
- Modify: `scripts/find_interesting_windows.py` (DLC section, after `load_frame_session_times`)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: `parse_dlc_video_stem`, `frame_window_to_session`, `clamp_to_trace`, `load_frame_session_times`, and the existing `count_licks_in_window`.
- Produces: `build_dlc_rois(dlc_rows, cycles, sess_loader=None) -> tuple[list[dict], dict[str, int]]` (a `None` loader resolves to `load_frame_session_times` at CALL time, so a test that monkeypatches the module attribute is honoured)
  - `cycles` maps `raw_stem -> {"cycle", "raw_h5", "layout", "animal", "restart", "first_s", "span_s", "lick_times"}`. `animal` is the filmed animal or `None`.
  - Each returned row has the keys the existing `write_csv` / `write_shell_script` expect: `animal, cycle, category, rank, start, end, center, score, n_licks_in_window, filmed, restart, raw_h5, layout`.
  - `skips` counts: `cfr_video`, `no_cycle`, `no_video_times`, `frame_out_of_range`, `outside_trace`.

- [ ] **Step 1: Write the failing tests**

Extend the import list in `tests/test_find_interesting_windows.py` with:

```python
    build_dlc_rois,
```

Append to the end of the file:

```python
def _cycle_info(**overrides):
    info = {
        "cycle": 2,
        "raw_h5": "/data/raw_data_2026-07-24_12-02-14.h5",
        "layout": "/data/layout.csv",
        "animal": "ACG-26-3-1",
        "restart": False,
        "first_s": 0.0,
        "span_s": 1000.0,
        "lick_times": np.array([]),
    }
    info.update(overrides)
    return info


def _dlc_row(video="/videos/raw_data_2026-07-24_12-02-14.mp4", start_frame=10, end_frame=20,
             tongue_rate=5.0, label="w000"):
    return {"label": label, "video": video, "start_frame": start_frame,
            "end_frame": end_frame, "tongue_rate": tongue_rate, }


def test_build_dlc_rois_emits_one_row_per_window():
    sess = np.arange(1000, dtype=np.float64) / 10.0  # frame k at k/10 s
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info(lick_times=np.array([1.5]))}
    rows, skips = build_dlc_rois(
        [_dlc_row(start_frame=10, end_frame=20), _dlc_row(start_frame=30, end_frame=41)],
        cycles, sess_loader=lambda path: sess)

    assert [r["category"] for r in rows] == ["dlc", "dlc"]
    assert [r["rank"] for r in rows] == [0, 1]
    assert rows[0]["start"] == 1.0 and rows[0]["end"] == 1.9
    assert rows[1]["start"] == 3.0 and rows[1]["end"] == 4.0
    assert rows[0]["center"] == (1.0 + 1.9) / 2.0
    assert rows[0]["score"] == 5.0
    # the lick at 1.5 s falls inside the first window only
    assert rows[0]["n_licks_in_window"] == 1
    assert rows[1]["n_licks_in_window"] == 0
    assert rows[0]["animal"] == "ACG-26-3-1"
    assert rows[0]["cycle"] == 2
    assert rows[0]["filmed"] is True
    assert rows[0]["raw_h5"] == "/data/raw_data_2026-07-24_12-02-14.h5"
    assert rows[0]["layout"] == "/data/layout.csv"
    assert sum(skips.values()) == 0


def test_build_dlc_rois_skips_cfr_videos():
    # The _cfr files are re-encodes; their frame indices are not the original container's ordinals,
    # so a window built from them could be silently misaligned. Never emit one.
    cycles = {"raw_data_2026-07-13_11-59-47": _cycle_info()}
    rows, skips = build_dlc_rois(
        [_dlc_row(video="/videos/raw_data_2026-07-13_11-59-47_cfr.mp4")],
        cycles, sess_loader=lambda path: np.arange(1000, dtype=np.float64) / 10.0)
    assert rows == []
    assert skips["cfr_video"] == 1


def test_build_dlc_rois_skips_video_with_no_matching_cycle():
    rows, skips = build_dlc_rois(
        [_dlc_row(video="/videos/raw_data_1999-01-01_00-00-00.mp4")],
        {}, sess_loader=lambda path: np.arange(100, dtype=np.float64))
    assert rows == []
    assert skips["no_cycle"] == 1


def test_build_dlc_rois_skips_video_without_frame_times():
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    rows, skips = build_dlc_rois([_dlc_row(), _dlc_row(label="w001")], cycles,
                                 sess_loader=lambda path: None)
    assert rows == []
    # counted per ROW, so the report says how much output was lost, not how many files were missing
    assert skips["no_video_times"] == 2


def test_build_dlc_rois_loads_frame_times_once_per_video():
    calls = []

    def loader(path):
        calls.append(path)
        return np.arange(1000, dtype=np.float64) / 10.0

    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    rows, _ = build_dlc_rois([_dlc_row(), _dlc_row(label="w001", start_frame=30, end_frame=40)],
                             cycles, sess_loader=loader)
    assert len(rows) == 2
    assert calls == ["/data/raw_data_2026-07-24_12-02-14.h5"]


def test_build_dlc_rois_skips_row_past_end_of_sidecar():
    sess = np.arange(100, dtype=np.float64) / 10.0
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    rows, skips = build_dlc_rois([_dlc_row(start_frame=500, end_frame=510)], cycles,
                                 sess_loader=lambda path: sess)
    assert rows == []
    assert skips["frame_out_of_range"] == 1


def test_build_dlc_rois_clamps_and_drops_against_the_trace():
    sess = np.arange(1000, dtype=np.float64) / 10.0  # 0 .. 99.9 s
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info(first_s=0.0, span_s=50.0)}
    rows, skips = build_dlc_rois(
        [_dlc_row(start_frame=495, end_frame=520),   # 49.5 .. 51.9 -> clamped to 50.0
         _dlc_row(start_frame=700, end_frame=710, label="w001")],  # 70 .. 70.9 -> gone
        cycles, sess_loader=lambda path: sess)
    assert len(rows) == 1
    assert rows[0]["start"] == 49.5
    assert rows[0]["end"] == 50.0
    assert skips["outside_trace"] == 1


def test_build_dlc_rois_unfilmed_cycle_produces_unfilmed_rows():
    # No filmed animal (layout doesn't name the video sensor): the row is still reported in the
    # CSV, but write_shell_script will not emit a command for it.
    sess = np.arange(1000, dtype=np.float64) / 10.0
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info(animal=None)}
    rows, _ = build_dlc_rois([_dlc_row()], cycles, sess_loader=lambda path: sess)
    assert rows[0]["filmed"] is False
    assert rows[0]["animal"] == ""


def test_build_dlc_rois_carries_the_restart_flag():
    sess = np.arange(1000, dtype=np.float64) / 10.0
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info(restart=True)}
    rows, _ = build_dlc_rois([_dlc_row()], cycles, sess_loader=lambda path: sess)
    assert rows[0]["restart"] is True


def test_build_dlc_rois_ranks_within_each_video():
    sess = np.arange(1000, dtype=np.float64) / 10.0
    cycles = {
        "raw_data_2026-07-24_12-02-14": _cycle_info(cycle=2),
        "raw_data_2026-07-27_11-56-15": _cycle_info(
            cycle=3, raw_h5="/data/raw_data_2026-07-27_11-56-15.h5"),
    }
    rows, _ = build_dlc_rois(
        [_dlc_row(video="/videos/raw_data_2026-07-24_12-02-14.mp4"),
         _dlc_row(video="/videos/raw_data_2026-07-27_11-56-15.mp4"),
         _dlc_row(video="/videos/raw_data_2026-07-27_11-56-15.mp4", start_frame=30, end_frame=40)],
        cycles, sess_loader=lambda path: sess)
    ranks = {(r["cycle"], r["rank"]) for r in rows}
    assert ranks == {(2, 0), (3, 0), (3, 1)}


def test_build_dlc_rois_emits_no_lick_or_climb_rows():
    sess = np.arange(1000, dtype=np.float64) / 10.0
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    rows, _ = build_dlc_rois([_dlc_row()], cycles, sess_loader=lambda path: sess)
    assert {r["category"] for r in rows} == {"dlc"}


def test_dlc_row_builds_a_cropped_command_with_speed():
    # Only "climb" gets --no-crop; a DLC clip is nose-at-sipper footage, where the crop is what
    # makes the tongue visible.
    row = {"animal": "ACG-26-3-1", "cycle": 2, "category": "dlc", "rank": 0,
           "start": 10.0, "end": 12.0, "raw_h5": "/data/raw.h5", "layout": "/data/layout.csv",
           "restart": False}
    lines = build_command(row, "clips", {}, "/data/combined.h5", speed=0.25)
    assert len(lines) == 1
    assert "--no-crop" not in lines[0]
    assert "--speed 0.25" in lines[0]
    assert "clips/ACG-26-3-1_c2_dlc0.mp4" in lines[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k dlc -v`
Expected: FAIL at import — `ImportError: cannot import name 'build_dlc_rois'`

- [ ] **Step 3: Write the implementation**

Append to the DLC section of `scripts/find_interesting_windows.py`:

```python
# Reasons a DLC row can fail to become a clip. Counted rather than printed per-row: a real CSV has
# hundreds of rows, and the useful summary is "how many, and why", not a wall of warnings.
DLC_SKIP_REASONS = ("cfr_video", "no_cycle", "no_video_times", "frame_out_of_range",
                    "outside_trace")


def build_dlc_rois(dlc_rows, cycles, sess_loader=None):
    """Turn DLC window rows into regions of interest, plus a count of what was skipped and why.

    `cycles` maps a recording's stem (`raw_data_YYYY-MM-DD_HH-MM-SS`) to everything we know about
    that cycle: {cycle, raw_h5, layout, animal (the filmed one, or None), restart, first_s, span_s,
    lick_times}. See `build_cycles_for_dlc`.

    Rows are grouped by video so the (relatively expensive) per-frame session times are read once
    per recording rather than once per window. `sess_loader` is injected so this is testable
    without a recording on disk; it is resolved here rather than in the signature so that patching
    the module-level `load_frame_session_times` also takes effect.

    Ranks are assigned in CSV order within each video, which is the order find_dlc_windows.py
    emitted them (ascending start frame).
    """
    sess_loader = sess_loader or load_frame_session_times
    skips = {reason: 0 for reason in DLC_SKIP_REASONS}
    rows = []

    # Preserve CSV order within each video (dicts keep insertion order).
    by_video = {}
    for row in dlc_rows:
        by_video.setdefault(str(row["video"]), []).append(row)

    for video, video_rows in by_video.items():
        stem, is_cfr = parse_dlc_video_stem(video)
        if is_cfr:
            # A CFR re-encode's frame k is not guaranteed to be the original container's frame k,
            # so we cannot honestly place it on the session clock.
            skips["cfr_video"] += len(video_rows)
            continue
        cycle = cycles.get(stem)
        if cycle is None:
            skips["no_cycle"] += len(video_rows)
            continue
        sess = sess_loader(cycle["raw_h5"])
        if sess is None:
            skips["no_video_times"] += len(video_rows)
            continue

        rank = 0
        for row in video_rows:
            window = frame_window_to_session(sess, row["start_frame"], row["end_frame"])
            if window is None:
                skips["frame_out_of_range"] += 1
                continue
            window = clamp_to_trace(window[0], window[1], cycle["first_s"], cycle["span_s"])
            if window is None:
                skips["outside_trace"] += 1
                continue
            start_s, end_s = window
            animal = cycle["animal"]
            rows.append({
                "animal": animal or "",
                "cycle": cycle["cycle"],
                "category": "dlc",
                "rank": rank,
                "start": start_s,
                "end": end_s,
                "center": (start_s + end_s) / 2.0,
                "score": float(row["tongue_rate"]),
                "n_licks_in_window": count_licks_in_window(cycle["lick_times"], start_s, end_s),
                "filmed": animal is not None,
                "restart": cycle["restart"],
                "raw_h5": cycle["raw_h5"] or "",
                "layout": cycle["layout"] or "",
            })
            rank += 1

    return rows, skips
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): build DLC regions of interest with counted skips"
```

---

### Task 5: Read cycle context out of the combined results file

**Files:**
- Modify: `scripts/find_interesting_windows.py` (DLC section, after `build_dlc_rois`)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: existing `cycle_provenance`, `resolve_filmed_animal`, `is_restart_recording`, `is_control`.
- Produces: `build_cycles_for_dlc(combined_h5_path, raw_map=None, animals=None) -> dict[str, dict]` — the `cycles` mapping `build_dlc_rois` takes, keyed by raw-recording stem.

- [ ] **Step 1: Write the failing test**

Extend the import list in `tests/test_find_interesting_windows.py` with:

```python
    build_cycles_for_dlc,
```

Append to the end of the file:

```python
def _write_combined(tmp_path, raw_h5_path):
    """A minimal results_combined_*.h5: one animal, one cycle, provenance attrs, and the arrays
    build_cycles_for_dlc reads."""
    import h5py
    combined = tmp_path / "combined.h5"
    with h5py.File(combined, "w") as f:
        g = f.create_group("ACG-26-3-1").create_group("0")
        g.attrs["raw_h5"] = str(raw_h5_path)
        g.attrs["layout"] = "/data/layout.csv"
        g.create_dataset("time_data", data=np.linspace(0.0, 100.0, 1001))
        g.create_dataset("lick_times", data=np.array([5.0, 6.0]))
    return str(combined)


def test_build_cycles_for_dlc_keys_on_raw_stem(tmp_path, monkeypatch):
    import scripts.find_interesting_windows as fiw
    raw_h5 = tmp_path / "raw_data_2026-07-24_12-02-14.h5"
    raw_h5.write_text("")  # never opened: both readers below are stubbed
    combined = _write_combined(tmp_path, raw_h5)
    monkeypatch.setattr(fiw, "resolve_filmed_animal", lambda h5, layout: "ACG-26-3-1")
    monkeypatch.setattr(fiw, "is_restart_recording", lambda h5: False)

    cycles = build_cycles_for_dlc(combined)

    assert list(cycles) == ["raw_data_2026-07-24_12-02-14"]
    info = cycles["raw_data_2026-07-24_12-02-14"]
    assert info["cycle"] == 0
    assert info["animal"] == "ACG-26-3-1"
    assert info["layout"] == "/data/layout.csv"
    assert info["first_s"] == 0.0
    assert info["span_s"] == 100.0
    assert info["restart"] is False
    np.testing.assert_allclose(info["lick_times"], [5.0, 6.0])


def test_build_cycles_for_dlc_unfilmed_cycle_has_no_animal(tmp_path, monkeypatch):
    import scripts.find_interesting_windows as fiw
    raw_h5 = tmp_path / "raw_data_2026-07-24_12-02-14.h5"
    raw_h5.write_text("")
    combined = _write_combined(tmp_path, raw_h5)
    monkeypatch.setattr(fiw, "resolve_filmed_animal", lambda h5, layout: None)
    monkeypatch.setattr(fiw, "is_restart_recording", lambda h5: False)

    info = build_cycles_for_dlc(combined)["raw_data_2026-07-24_12-02-14"]
    assert info["animal"] is None
    # No filmed animal means no trace to draw either, so the bounds stay empty rather than
    # borrowing some other animal's.
    assert info["lick_times"].size == 0


def test_build_cycles_for_dlc_animals_filter_drops_other_cycles(tmp_path, monkeypatch):
    import scripts.find_interesting_windows as fiw
    raw_h5 = tmp_path / "raw_data_2026-07-24_12-02-14.h5"
    raw_h5.write_text("")
    combined = _write_combined(tmp_path, raw_h5)
    monkeypatch.setattr(fiw, "resolve_filmed_animal", lambda h5, layout: "ACG-26-3-1")
    monkeypatch.setattr(fiw, "is_restart_recording", lambda h5: False)

    assert build_cycles_for_dlc(combined, animals=["SOMEONE-ELSE"]) == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_find_interesting_windows.py -k build_cycles -v`
Expected: FAIL at import — `ImportError: cannot import name 'build_cycles_for_dlc'`

- [ ] **Step 3: Write the implementation**

Append to the DLC section of `scripts/find_interesting_windows.py`:

```python
def build_cycles_for_dlc(combined_h5_path, raw_map=None, animals=None):
    """Everything `build_dlc_rois` needs about each cycle, keyed by the raw recording's stem.

    The DLC CSV names a video, never a raw `.h5`, so the stem is the join key: a cycle's `raw_h5`
    provenance attribute (or the `--raw-map` fallback) is named after the same recording the video
    is. Only the FILMED animal matters here -- it is the one with a camera, so it is the only one a
    sync clip can exist for -- so the trace bounds and lick times come from that animal's group.

    `animals` (from --animals) filters on the resolved filmed animal; a cycle whose filmed animal is
    not in the list is dropped entirely.
    """
    raw_map = raw_map or {}
    cycles = {}
    with h5py.File(combined_h5_path, "r") as combined:
        # Provenance is per-cycle, so visit each cycle once, via whichever animal group carries it.
        seen_cycles = set()
        for animal_id in combined.keys():
            for cycle_key in combined[animal_id].keys():
                if cycle_key in seen_cycles:
                    continue
                raw_h5_path, layout_path = cycle_provenance(combined, animal_id, cycle_key, raw_map)
                if not raw_h5_path:
                    continue
                seen_cycles.add(cycle_key)

                filmed_animal = resolve_filmed_animal(raw_h5_path, layout_path)
                if animals and filmed_animal not in set(animals):
                    continue

                first_s, span_s = 0.0, 0.0
                lick_times = np.array([])
                if filmed_animal is not None and filmed_animal in combined:
                    group = combined[filmed_animal].get(cycle_key)
                    if group is not None and "time_data" in group:
                        time_data = group["time_data"][:]
                        if len(time_data) >= 2:
                            first_s = float(time_data[0])
                            span_s = float(time_data[-1])
                        if "lick_times" in group:
                            lick_times = group["lick_times"][:]

                try:
                    cycle_value = int(cycle_key)
                except ValueError:
                    cycle_value = cycle_key

                stem = os.path.splitext(os.path.basename(raw_h5_path))[0]
                cycles[stem] = {
                    "cycle": cycle_value,
                    "raw_h5": raw_h5_path,
                    "layout": layout_path,
                    "animal": filmed_animal,
                    "restart": is_restart_recording(raw_h5_path),
                    "first_s": first_s,
                    "span_s": span_s,
                    "lick_times": lick_times,
                }
    return cycles
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): map combined-file cycles to raw recording stems for DLC mode"
```

---

### Task 6: Wire up the `--dlc-windows` CLI flag

**Files:**
- Modify: `scripts/find_interesting_windows.py` (module docstring, `main`, new `write_outputs` and `print_dlc_skips`)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: `read_dlc_windows`, `build_cycles_for_dlc`, `build_dlc_rois`, existing `write_csv`, `write_shell_script`, `load_raw_map`, `parse_offsets`.
- Produces:
  - `write_outputs(all_rows, args) -> int` — sorts, writes the CSV and the shell script, prints the summary, returns the number of commands written.
  - `print_dlc_skips(skips, n_rows) -> None`
  - `main(argv)` accepts `--dlc-windows PATH`.

- [ ] **Step 1: Write the failing test**

Append to the end of `tests/test_find_interesting_windows.py`:

```python
def test_main_dlc_mode_writes_csv_and_commands(tmp_path, monkeypatch):
    # End-to-end through main(): a DLC CSV plus a combined file becomes a .sh of make_sync_video
    # calls, with no lick/climb search anywhere in it.
    import scripts.find_interesting_windows as fiw

    raw_h5 = tmp_path / "raw_data_2026-07-24_12-02-14.h5"
    raw_h5.write_text("")
    combined = _write_combined(tmp_path, raw_h5)
    windows = tmp_path / "windows.csv"
    windows.write_text(
        "task_id,label,video,start_frame,end_frame,tongue_rate\n"
        "1,w000,/videos/raw_data_2026-07-24_12-02-14.mp4,100,200,7.1\n"
        "2,w001,/videos/raw_data_2026-07-13_11-59-47_cfr.mp4,100,200,7.1\n"
    )
    monkeypatch.setattr(fiw, "resolve_filmed_animal", lambda h5, layout: "ACG-26-3-1")
    monkeypatch.setattr(fiw, "is_restart_recording", lambda h5: False)
    # frame k at k/100 s, so window [100, 200) is 1.0 .. 1.99 s
    monkeypatch.setattr(fiw, "load_frame_session_times",
                        lambda path: np.arange(5000, dtype=np.float64) / 100.0)

    out_csv = tmp_path / "dlc_rois.csv"
    out_sh = tmp_path / "make_dlc_clips.sh"
    rc = fiw.main([combined, "--dlc-windows", str(windows),
                   "--csv", str(out_csv), "--sh", str(out_sh),
                   "--out-dir", str(tmp_path / "clips"), "--speed", "0.25"])
    assert rc == 0

    csv_text = out_csv.read_text()
    assert "dlc" in csv_text
    assert "climb" not in csv_text
    assert ",lick," not in csv_text

    sh_text = out_sh.read_text()
    assert sh_text.count("make_sync_video.py") == 1     # the _cfr row produced nothing
    assert "--start 1.000 --end 1.990" in sh_text
    assert "--speed 0.25" in sh_text
    assert "--no-crop" not in sh_text
    assert "--cycle 0" in sh_text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_find_interesting_windows.py -k main_dlc -v`
Expected: FAIL with `SystemExit: 2` / `unrecognized arguments: --dlc-windows`

- [ ] **Step 3: Write the implementation**

3a. Add the flag in `build_arg_parser`'s equivalent — this script builds its parser inline in `main`. Add after the `--speed` argument:

```python
    parser.add_argument("--dlc-windows", dest="dlc_windows", default=None, metavar="CSV",
                        help="CSV written by dlc_integration/find_dlc_windows.py. Given it, the "
                             "script runs in DLC mode: the capacitance search is skipped entirely "
                             "and every emitted window comes from a DLC row. The trace-search "
                             "options (--n-lick/--n-climb/--roi-seconds/--lick-pad/"
                             "--climb-skip-edges/--var-window/--min-var/--include-controls) are "
                             "unused in this mode")
```

3b. Add these two functions just above `main`:

```python
def write_outputs(all_rows, args):
    """Sort, write the CSV and the shell script, and print the summary. Shared by both modes."""
    # Sort for a tidy, deterministic CSV: by animal, then cycle, then category, then rank.
    all_rows.sort(key=lambda r: (str(r["animal"]), str(r["cycle"]), r["category"], r["rank"]))

    offsets = parse_offsets(args.offset)
    write_csv(all_rows, args.csv)
    n_commands = write_shell_script(all_rows, args.sh, args.out_dir, offsets, args.combined_h5,
                                    args.speed)

    n_filmed_rows = sum(1 for r in all_rows if r["filmed"])
    print(f"Wrote {len(all_rows)} regions of interest to {args.csv}.")
    print(f"Wrote {n_commands} make_sync_video command(s) to {args.sh} "
          f"(filmed-animal regions only; {n_filmed_rows} filmed rows found).")
    if n_commands == 0:
        print("No commands were emitted. This usually means the combined file has no stored "
              "raw_h5/layout provenance and no --raw-map was given, or the raw files aren't at the "
              "stored paths. The CSV is still complete.")
    return n_commands


def print_dlc_skips(skips, n_rows):
    """Report every DLC row that did not become a region, by reason."""
    explanations = {
        "cfr_video": "video is a _cfr re-encode (frame indices are not the original container's; "
                     "re-run DLC on the original recording)",
        "no_cycle": "no cycle in the combined file matches the video's recording",
        "no_video_times": "the recording's video or PTS sidecar is missing on this machine",
        "frame_out_of_range": "start_frame is past the end of the PTS sidecar",
        "outside_trace": "the window falls outside the cycle's trace",
    }
    total = sum(skips.values())
    print(f"DLC mode: read {n_rows} window(s); {n_rows - total} became regions of interest.")
    for reason, count in skips.items():
        if count:
            print(f"  skipped {count}: {explanations[reason]}")
```

3c. Replace the body of `main` from `raw_map = load_raw_map(args.raw_map)` onward. The DLC branch returns early; the existing trace-search path keeps its loop verbatim and ends by calling `write_outputs`:

```python
    raw_map = load_raw_map(args.raw_map)

    if args.dlc_windows:
        cycles = build_cycles_for_dlc(args.combined_h5, raw_map, args.animals)
        dlc_rows = read_dlc_windows(args.dlc_windows)
        all_rows, skips = build_dlc_rois(dlc_rows, cycles)
        write_outputs(all_rows, args)
        print_dlc_skips(skips, len(dlc_rows))
        return 0

    params = {
        ...unchanged...
    }
    ...the existing per-animal/per-cycle loop, unchanged...

    write_outputs(all_rows, args)
    return 0
```

Concretely, in the existing code:
- delete the `offsets = parse_offsets(args.offset)` line near the top of `main` (it moved into `write_outputs`),
- delete the trailing block from `# Sort for a tidy, deterministic CSV:` through `return 0` and replace it with `write_outputs(all_rows, args)` then `return 0`.

3d. Update the module docstring: add a third bullet under "What it produces" and a DLC usage example.

Add after the numbered list in "What it produces":

```
3. With `--dlc-windows <csv>`, the script runs in DLC mode instead: it does not search the trace at
   all. Every window comes from a row of the CSV written by dlc_integration/find_dlc_windows.py --
   the stretches where DeepLabCut saw the animal confidently at the sipper. The frame ranges in
   that CSV are converted to session seconds with the same SessionClock and PTS sidecar
   make_sync_video uses to time frames, so the clip shows exactly the frames DLC scored. The clips
   are rendered from the ORIGINAL recording, never from a DLC labeled video. Rows whose video is a
   `_cfr` re-encode are skipped: a re-encode's frame indices are not the original container's
   ordinals, so they cannot be honestly placed on the session clock.
```

And add to the Usage block:

```
    # DLC-selected windows instead of the capacitance search:
    python scripts/find_interesting_windows.py \
        "Lickometry Data/results_combined_ACG-26-3_2026-07-22_23_24_27_28_29_basic-algorithm.h5" \
        --dlc-windows second_iteration_windows_with_tongue.csv \
        --csv dlc_rois.csv --sh make_dlc_clips.sh --speed 0.25
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS, including every pre-existing test (the `write_outputs` extraction must not change trace-mode behavior)

- [ ] **Step 5: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): add --dlc-windows mode to find_interesting_windows"
```

---

### Task 7: Verify against the real data

**Files:**
- No source changes expected. If a defect turns up, fix it with a test first, in the task it belongs to.

**Interfaces:**
- Consumes: everything above.
- Produces: evidence the feature works on a real combined file and a real DLC CSV.

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest tests/test_find_interesting_windows.py tests/test_dlc_windows.py tests/test_make_sync_video.py -v`
Expected: PASS. (The last two guard the modules this feature imports from.)

- [ ] **Step 2: Run DLC mode on the real files**

```bash
python scripts/find_interesting_windows.py \
    "Lickometry Data/results_combined_ACG-26-3_2026-07-22_23_24_27_28_29_basic-algorithm.h5" \
    --dlc-windows second_iteration_windows_with_tongue.csv \
    --csv /tmp/dlc_rois.csv --sh /tmp/make_dlc_clips.sh --out-dir clips --speed 0.25
```

Expected, from what is on disk today:
- The four `_cfr` videos (07-13, 07-14, 07-16, 07-21) are skipped as `cfr_video` — 35 of the 80 rows in `second_iteration_windows_with_tongue.csv`.
- The six non-`_cfr` videos (07-22 .. 07-29) map to cycles 0-5 and produce commands.
- `/tmp/dlc_rois.csv` has only `dlc` rows.

- [ ] **Step 3: Sanity-check a window against the DLC CSV**

Pick one emitted command and confirm its `--start` is close to, but not equal to, the CSV's `start_sec` for the same window: they differ by the bookmark latency plus drift, which is exactly the correction this feature exists to apply. A difference of 0.000 s means the conversion silently fell back to video time — that is a bug.

```bash
head -20 /tmp/make_dlc_clips.sh
grep "raw_data_2026-07-24" second_iteration_windows_with_tongue.csv | head -3
```

- [ ] **Step 4: Render one clip and eyeball it**

```bash
bash -c "$(grep -m1 'make_sync_video.py' /tmp/make_dlc_clips.sh)"
```

Expected: an `.mp4` in `clips/` where the mouse is at the sipper for the whole clip (that is what DLC selected), with the trace panel moving in step.

- [ ] **Step 5: Commit any fixes, then report**

If steps 2-4 were clean there is nothing to commit. Report the skip counts and the command count actually observed — do not restate the expectations above as if they were results.

---

## Notes for the implementer

- `scripts/find_interesting_windows.py` builds its `argparse` parser inline inside `main`; there is no `build_arg_parser` function in this file (unlike `make_sync_video.py`).
- `write_shell_script` already skips rows where `filmed` is false or provenance is missing, and `build_command` already handles `--no-crop` (climb only), `--speed`, `--offset` and the restart WARNING. DLC mode needs no changes there.
- `count_licks_in_window` takes `(lick_times, start_s, end_s)` and is inclusive at both ends.
- Do not add a `--dlc-pad`, an `--n-dlc` cap, or a ranking heuristic. Selection belongs upstream in `find_dlc_windows.py` (`--require-tongue`, `--min-frames`, `--pad`).
