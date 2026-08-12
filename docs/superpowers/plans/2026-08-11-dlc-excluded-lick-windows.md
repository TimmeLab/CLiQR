# `--dlc-exclude` (licks DLC never saw) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--dlc-exclude CSV` mode to `scripts/find_interesting_windows.py` that emits synced-video clips for the busiest detected licking bouts that fall in NO DeepLabCut at-sipper window — the capacitance detector's false-positive candidates.

**Architecture:** Bout-seeded like the default trace search, DLC-filtered. A new `dlc_exclusion_spans()` converts every row of a `find_dlc_windows.py` CSV into session-second spans using the renderer's own frame clock (`load_frame_session_times` + `frame_window_to_session`, both already in the file). A new `bouts_outside_dlc()` produces a keep-mask over a cycle's bouts. The surviving bouts are handed to the EXISTING `build_rois_for_cycle()` with `n_climb=0`, and its rows are relabelled `category="no_dlc"`. All output goes through the existing `write_csv` / `write_shell_script` / `build_command`.

**Tech Stack:** Python 3, numpy, h5py, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-11-dlc-excluded-lick-windows-design.md`

## Global Constraints

- Everything lives in `scripts/find_interesting_windows.py`; tests in `tests/test_find_interesting_windows.py`. No new modules.
- Frame → session time is NEVER computed as `frame / fps`. Only `load_frame_session_times()` + `frame_window_to_session()`, which use the same `SessionClock` and PTS sidecar `make_sync_video` uses to place frames. See `docs/video-sync-alignment-bugs.md`.
- DLC exclusion spans are NOT clamped to the trace. A window outside the trace still proves the animal was at the sipper.
- A cycle with no rows in the DLC CSV emits NOTHING (skip reason `no_dlc_video`). "DLC found no windows" and "DLC never ran" are indistinguishable from the CSV, and treating the second as the first would dump a whole session's bouts into the false-positive pile.
- Only the FILMED animal of each cycle is considered. Other cages have no camera.
- Every skip is counted by reason and printed; nothing is silently dropped.
- Default `--dlc-guard` is `1.0` seconds.
- New `category` value is exactly `"no_dlc"`.
- Run tests with: `python -m pytest tests/test_find_interesting_windows.py -v`
- Test style: follow the existing file — helpers `_cycle_info(**overrides)` and `_dlc_row(...)` already exist at `tests/test_find_interesting_windows.py:570` and `:585`; reuse them, do not redefine them.

---

### Task 1: Convert DLC CSV rows into per-recording exclusion spans

**Files:**
- Modify: `scripts/find_interesting_windows.py` (add after `build_dlc_rois`, which ends around line 683)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes (all already exist in the module): `parse_dlc_video_stem(video_path) -> (stem, is_cfr)`, `frame_window_to_session(sess, start_frame, end_frame) -> (start_s, end_s) | None`, `load_frame_session_times(raw_h5_path) -> np.ndarray | None`, and the `cycles` dict shape from `build_cycles_for_dlc` (`{stem: {cycle, raw_h5, layout, animal, restart, first_s, span_s, lick_times}}`).
- Produces:
  - `DLC_EXCLUDE_SKIP_REASONS` — tuple of reason strings.
  - `dlc_exclusion_spans(dlc_rows, cycles, sess_loader=None) -> (by_stem, skips)` where
    `by_stem` is `{stem: {"spans": [(start_s, end_s), ...], "coverage": (first_s, last_s)}}`
    and `skips` is `{reason: count}` over `DLC_EXCLUDE_SKIP_REASONS`.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_find_interesting_windows.py`:

```python
def test_dlc_exclusion_spans_converts_frames_and_reports_coverage():
    sess = np.arange(1000, dtype=np.float64) / 10.0  # frame k at k/10 s
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    by_stem, skips = dlc_exclusion_spans(
        [_dlc_row(start_frame=10, end_frame=20),
         _dlc_row(start_frame=30, end_frame=41, label="w001")],
        cycles, sess_loader=lambda path: sess)

    entry = by_stem["raw_data_2026-07-24_12-02-14"]
    assert entry["spans"] == [(1.0, 1.9), (3.0, 4.0)]
    assert entry["coverage"] == (0.0, 99.9)
    assert sum(skips.values()) == 0


def test_dlc_exclusion_spans_does_not_clamp_to_the_trace():
    # A window past the end of the capacitance trace is still proof the animal was at the sipper,
    # so it must stay in the exclusion set at full width. Clamping it (as the RENDERING path does)
    # would let a bout in that stretch through as a false-positive candidate.
    sess = np.arange(1000, dtype=np.float64) / 10.0  # 0 .. 99.9 s
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info(first_s=0.0, span_s=50.0)}
    by_stem, skips = dlc_exclusion_spans(
        [_dlc_row(start_frame=700, end_frame=710)], cycles, sess_loader=lambda path: sess)

    assert by_stem["raw_data_2026-07-24_12-02-14"]["spans"] == [(70.0, 70.9)]
    assert sum(skips.values()) == 0


def test_dlc_exclusion_spans_counts_whole_video_skips_once_each():
    sess = np.arange(1000, dtype=np.float64) / 10.0
    cycles = {
        "raw_data_2026-07-13_11-59-47": _cycle_info(),
        "raw_data_2026-07-24_12-02-14": _cycle_info(first_s=0.0, span_s=0.0),
    }
    by_stem, skips = dlc_exclusion_spans(
        [_dlc_row(video="/videos/raw_data_2026-07-13_11-59-47_cfr.mp4"),
         _dlc_row(video="/videos/raw_data_2026-07-13_11-59-47_cfr.mp4", label="w001"),
         _dlc_row(video="/videos/raw_data_1999-01-01_00-00-00.mp4"),
         _dlc_row()],
        cycles, sess_loader=lambda path: sess)

    assert by_stem == {}
    # counted per VIDEO, not per row: the unit that is unusable here is the whole recording
    assert skips["cfr_video"] == 1
    assert skips["no_cycle"] == 1
    assert skips["no_trace"] == 1


def test_dlc_exclusion_spans_counts_missing_frame_times():
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    by_stem, skips = dlc_exclusion_spans([_dlc_row()], cycles, sess_loader=lambda path: None)
    assert by_stem == {}
    assert skips["no_video_times"] == 1


def test_dlc_exclusion_spans_counts_out_of_range_rows_per_row():
    sess = np.arange(100, dtype=np.float64) / 10.0
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    by_stem, skips = dlc_exclusion_spans(
        [_dlc_row(start_frame=500, end_frame=510),
         _dlc_row(start_frame=10, end_frame=20, label="w001")],
        cycles, sess_loader=lambda path: sess)

    assert by_stem["raw_data_2026-07-24_12-02-14"]["spans"] == [(1.0, 1.9)]
    assert skips["frame_out_of_range"] == 1


def test_dlc_exclusion_spans_loads_frame_times_once_per_video():
    calls = []

    def loader(path):
        calls.append(path)
        return np.arange(1000, dtype=np.float64) / 10.0

    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    dlc_exclusion_spans([_dlc_row(), _dlc_row(label="w001", start_frame=30, end_frame=40)],
                        cycles, sess_loader=loader)
    assert calls == ["/data/raw_data_2026-07-24_12-02-14.h5"]
```

Add `dlc_exclusion_spans` to the existing import block at the top of the test file (the one that already imports `build_dlc_rois`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k dlc_exclusion_spans -v`
Expected: FAIL at import — `ImportError: cannot import name 'dlc_exclusion_spans'`

- [ ] **Step 3: Write the implementation**

Insert into `scripts/find_interesting_windows.py`, immediately after `build_dlc_rois` and before `build_cycles_for_dlc`:

```python
# Reasons a cycle or a bout produces no clip in --dlc-exclude mode. The first five mirror
# DLC_SKIP_REASONS (a video we cannot place on the session clock is unusable in either mode);
# the last three are specific to hunting licks that DLC never saw.
DLC_EXCLUDE_SKIP_REASONS = ("cfr_video", "no_cycle", "no_video_times", "no_trace",
                            "frame_out_of_range", "no_dlc_video", "outside_video",
                            "in_dlc_window")


def dlc_exclusion_spans(dlc_rows, cycles, sess_loader=None):
    """Session-second spans of every DLC at-sipper window, keyed by recording stem.

    Returns ({stem: {"spans": [(start_s, end_s), ...], "coverage": (first_s, last_s)}}, skips).

    `coverage` is the session time of the FIRST and LAST frame of the video. Outside it DLC
    observed nothing, so the absence of a span there is not evidence that the animal was away --
    the caller must drop bouts that fall outside it rather than reporting them.

    Unlike `build_dlc_rois`, spans are NOT clamped to the cycle's trace. Clamping is a rendering
    concern (make_sync_video cannot draw a trace that isn't there); a window past the end of the
    trace is still proof that the animal was at the sipper, and dropping it here would let the
    bouts under it through as false-positive candidates.

    Whole-video failures are counted ONCE per video (the unusable unit is the recording), not once
    per row as in `build_dlc_rois`, because here a video's rows are not clips-in-waiting -- they
    are evidence about one cycle, and the cycle either has usable evidence or it does not.

    `sess_loader` is injected for testing; it defaults to the module-level
    `load_frame_session_times` (resolved here so patching the module attribute takes effect).
    """
    sess_loader = sess_loader or load_frame_session_times
    skips = {reason: 0 for reason in DLC_EXCLUDE_SKIP_REASONS}
    by_stem = {}

    by_video = {}
    for row in dlc_rows:
        by_video.setdefault(str(row["video"]), []).append(row)

    for video, video_rows in by_video.items():
        stem, is_cfr = parse_dlc_video_stem(video)
        if is_cfr:
            skips["cfr_video"] += 1
            continue
        cycle = cycles.get(stem)
        if cycle is None:
            skips["no_cycle"] += 1
            continue
        if cycle["span_s"] <= cycle["first_s"]:
            skips["no_trace"] += 1
            continue
        sess = sess_loader(cycle["raw_h5"])
        if sess is None:
            skips["no_video_times"] += 1
            continue

        spans = []
        for row in video_rows:
            window = frame_window_to_session(sess, row["start_frame"], row["end_frame"])
            if window is None:
                skips["frame_out_of_range"] += 1
                continue
            spans.append(window)

        sess = np.asarray(sess, dtype=np.float64)
        by_stem[stem] = {"spans": spans,
                         "coverage": (float(sess[0]), float(sess[-1]))}

    return by_stem, skips
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -k dlc_exclusion_spans -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the whole file to check nothing regressed**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): convert DLC windows into unclamped exclusion spans"
```

---

### Task 2: Decide which bouts DLC never saw

**Files:**
- Modify: `scripts/find_interesting_windows.py` (add immediately after `dlc_exclusion_spans`)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: the `{"spans": [...], "coverage": (...)}` entry shape from Task 1.
- Produces: `bouts_outside_dlc(bout_start_times, bout_durations, spans, coverage, guard_s) -> (keep_mask, skips)` where `keep_mask` is a boolean `np.ndarray` the same length as `bout_start_times`, and `skips` is `{"outside_video": int, "in_dlc_window": int}`.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_find_interesting_windows.py`:

```python
def test_bouts_outside_dlc_keeps_bouts_no_window_touches():
    starts = np.array([10.0, 100.0])
    durations = np.array([5.0, 5.0])
    keep, skips = bouts_outside_dlc(starts, durations, spans=[(95.0, 110.0)],
                                    coverage=(0.0, 500.0), guard_s=1.0)
    assert list(keep) == [True, False]
    assert skips["in_dlc_window"] == 1
    assert skips["outside_video"] == 0


def test_bouts_outside_dlc_guard_band_disqualifies_near_misses():
    # Bout ends at 20.0; the DLC window opens at 20.5. Half a second apart is inside the ragged
    # edge a merge-gap/pad window has, so the guard must reject it -- and accept it when off.
    starts = np.array([15.0])
    durations = np.array([5.0])
    guarded, _ = bouts_outside_dlc(starts, durations, spans=[(20.5, 25.0)],
                                   coverage=(0.0, 500.0), guard_s=1.0)
    unguarded, _ = bouts_outside_dlc(starts, durations, spans=[(20.5, 25.0)],
                                     coverage=(0.0, 500.0), guard_s=0.0)
    assert list(guarded) == [False]
    assert list(unguarded) == [True]


def test_bouts_outside_dlc_drops_bouts_outside_video_coverage():
    # Before the camera started and after it stopped, DLC saw nothing, so "no window here" is not
    # evidence the animal was away.
    starts = np.array([5.0, 50.0, 900.0])
    durations = np.array([2.0, 2.0, 2.0])
    keep, skips = bouts_outside_dlc(starts, durations, spans=[], coverage=(10.0, 800.0),
                                    guard_s=1.0)
    assert list(keep) == [False, True, False]
    assert skips["outside_video"] == 2
    assert skips["in_dlc_window"] == 0


def test_bouts_outside_dlc_requires_the_whole_bout_inside_coverage():
    # Bout straddles the end of the video: DLC could not have scored its tail.
    keep, skips = bouts_outside_dlc(np.array([790.0]), np.array([20.0]), spans=[],
                                    coverage=(10.0, 800.0), guard_s=1.0)
    assert list(keep) == [False]
    assert skips["outside_video"] == 1


def test_bouts_outside_dlc_handles_no_bouts():
    keep, skips = bouts_outside_dlc(np.array([]), np.array([]), spans=[(1.0, 2.0)],
                                    coverage=(0.0, 10.0), guard_s=1.0)
    assert keep.size == 0
    assert sum(skips.values()) == 0
```

Add `bouts_outside_dlc` to the test file's import block.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k bouts_outside_dlc -v`
Expected: FAIL at import — `ImportError: cannot import name 'bouts_outside_dlc'`

- [ ] **Step 3: Write the implementation**

```python
def bouts_outside_dlc(bout_start_times, bout_durations, spans, coverage, guard_s):
    """Which detected bouts DLC never saw: (keep_mask, skips).

    A bout is kept only if BOTH hold:

    * it lies entirely within `coverage`, the video's session-time extent. A bout that starts
      before the camera or runs past it was partly unobserved, and "DLC reported no window" says
      nothing about an interval DLC never looked at;
    * it intersects no span in `spans` after each span is grown by `guard_s` on both sides.

    The guard band matters because a DLC window's edges are artefacts of `--merge-gap`,
    `--min-frames` and `--pad` in find_dlc_windows.py, not the moment the animal arrived or left;
    without it, bouts that start a few hundred milliseconds before a window opens survive and are
    almost all real licks.

    The test is applied to the BOUT, not to the padded clip window: the clip is deliberately wider
    (`--lick-pad` of context), and showing the edge of a DLC window is useful context, not grounds
    for disqualification.
    """
    starts = np.asarray(bout_start_times, dtype=np.float64)
    durations = np.asarray(bout_durations, dtype=np.float64)
    skips = {"outside_video": 0, "in_dlc_window": 0}
    if starts.size == 0:
        return np.zeros(0, dtype=bool), skips

    ends = starts + durations
    first_s, last_s = float(coverage[0]), float(coverage[1])

    keep = np.ones(starts.size, dtype=bool)
    outside = (starts < first_s) | (ends > last_s)
    keep &= ~outside
    skips["outside_video"] = int(np.count_nonzero(outside))

    inside_window = np.zeros(starts.size, dtype=bool)
    for span_start, span_end in spans:
        guarded_start = float(span_start) - guard_s
        guarded_end = float(span_end) + guard_s
        inside_window |= (starts <= guarded_end) & (ends >= guarded_start)
    # Count only bouts disqualified by DLC alone, so the two reasons never double-count one bout.
    skips["in_dlc_window"] = int(np.count_nonzero(inside_window & ~outside))
    keep &= ~inside_window

    return keep, skips
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -k bouts_outside_dlc -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): mask bouts that DLC saw, or never observed"
```

---

### Task 3: Build the `no_dlc` regions for one cycle

**Files:**
- Modify: `scripts/find_interesting_windows.py` — `build_rois_for_cycle` (climbing block starts at line 351: `variance_window_samples = ...`), plus a new function after `bouts_outside_dlc`
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: `bouts_outside_dlc` (Task 2), existing `build_rois_for_cycle(cap_data, time_data, bout_start_times, bout_durations, bout_lick_counts, lick_times, params)`.
- Produces: `build_no_dlc_rois_for_cycle(cap_data, time_data, bout_start_times, bout_durations, bout_lick_counts, lick_times, entry, params) -> (rois, skips)` where `entry` is one `dlc_exclusion_spans` value, `rois` is a list of the same region dicts `build_rois_for_cycle` returns but with `category == "no_dlc"`, and `skips` is `{"outside_video": int, "in_dlc_window": int}`.
- `params` keys read: `n_lick`, `roi_seconds`, `lick_pad`, `dlc_guard`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_rois_for_cycle_emits_no_climbing_when_n_climb_is_zero():
    # select_climbing_centers appends its first candidate before testing the count, so a caller
    # asking for zero climbing windows used to get one anyway. Licking-only callers depend on this.
    time_data = np.arange(0.0, 200.0, 0.01)
    cap_data = np.random.RandomState(0).normal(size=time_data.size)
    rois = build_rois_for_cycle(
        cap_data, time_data,
        bout_start_times=np.array([50.0]), bout_durations=np.array([3.0]),
        bout_lick_counts=np.array([30]), lick_times=np.array([50.5]),
        params={"n_lick": 3, "n_climb": 0, "roi_seconds": 12.0, "lick_pad": 2.0,
                "var_window": 1.0, "min_var": 0.0, "climb_skip_edges": 0.0},
    )
    assert [r["category"] for r in rois] == ["lick"]


def test_build_no_dlc_rois_for_cycle_keeps_only_unseen_bouts():
    time_data = np.arange(0.0, 400.0, 0.01)
    cap_data = np.zeros(time_data.size)
    # Bout 0 (busiest) sits inside a DLC window; bout 1 does not.
    starts = np.array([100.0, 300.0])
    durations = np.array([4.0, 4.0])
    counts = np.array([90, 20])
    licks = np.concatenate([np.linspace(100.0, 104.0, 90), np.linspace(300.0, 304.0, 20)])
    entry = {"spans": [(99.0, 106.0)], "coverage": (0.0, 400.0)}
    rois, skips = build_no_dlc_rois_for_cycle(
        cap_data, time_data, starts, durations, counts, licks, entry,
        params={"n_lick": 3, "roi_seconds": 12.0, "lick_pad": 2.0, "dlc_guard": 1.0})

    assert [r["category"] for r in rois] == ["no_dlc"]
    assert [r["rank"] for r in rois] == [0]
    assert rois[0]["score"] == 20.0
    assert rois[0]["start"] == 298.0 and rois[0]["end"] == 306.0
    assert rois[0]["n_licks_in_window"] == 20
    assert skips["in_dlc_window"] == 1


def test_build_no_dlc_rois_for_cycle_ranks_busiest_first():
    time_data = np.arange(0.0, 500.0, 0.01)
    cap_data = np.zeros(time_data.size)
    starts = np.array([100.0, 300.0])
    durations = np.array([4.0, 4.0])
    counts = np.array([10, 40])
    entry = {"spans": [], "coverage": (0.0, 500.0)}
    rois, _ = build_no_dlc_rois_for_cycle(
        cap_data, time_data, starts, durations, counts, np.array([]), entry,
        params={"n_lick": 3, "roi_seconds": 12.0, "lick_pad": 2.0, "dlc_guard": 1.0})

    assert [r["score"] for r in rois] == [40.0, 10.0]
    assert [r["rank"] for r in rois] == [0, 1]


def test_build_no_dlc_rois_for_cycle_emits_nothing_when_every_bout_is_seen():
    time_data = np.arange(0.0, 200.0, 0.01)
    cap_data = np.zeros(time_data.size)
    entry = {"spans": [(90.0, 120.0)], "coverage": (0.0, 200.0)}
    rois, skips = build_no_dlc_rois_for_cycle(
        cap_data, time_data, np.array([100.0]), np.array([4.0]), np.array([30]),
        np.array([]), entry,
        params={"n_lick": 3, "roi_seconds": 12.0, "lick_pad": 2.0, "dlc_guard": 1.0})

    assert rois == []
    assert skips["in_dlc_window"] == 1
```

Add `build_no_dlc_rois_for_cycle` to the test file's import block (`build_rois_for_cycle` is already imported).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k "no_dlc_rois or n_climb_is_zero" -v`
Expected: FAIL at import — `ImportError: cannot import name 'build_no_dlc_rois_for_cycle'`

- [ ] **Step 3: Guard the climbing block on `n_climb > 0`**

In `build_rois_for_cycle`, the climbing section currently begins:

```python
    variance_window_samples = (max(1, int(round(params["var_window"] / median_dt_s)))
                               if median_dt_s > 0 else 1)
    window_variance = sliding_variance(cap_data, variance_window_samples)
    if window_variance.size > 0 and median_dt_s > 0:
```

Change the guard so the whole block is skipped when no climbing windows were asked for (this also avoids an 800k-sample variance pass for a licking-only caller):

```python
    if params["n_climb"] > 0:
        variance_window_samples = (max(1, int(round(params["var_window"] / median_dt_s)))
                                   if median_dt_s > 0 else 1)
        window_variance = sliding_variance(cap_data, variance_window_samples)
        if window_variance.size > 0 and median_dt_s > 0:
            ...
```

Indent the rest of the climbing block (through the `for rank, (center_time, variance) in enumerate(picks):` loop) one level to match. Everything up to and including the licking section is unchanged, as is the final `return rois`.

- [ ] **Step 4: Write `build_no_dlc_rois_for_cycle`**

Add after `bouts_outside_dlc`:

```python
def build_no_dlc_rois_for_cycle(cap_data, time_data, bout_start_times, bout_durations,
                                bout_lick_counts, lick_times, entry, params):
    """Regions for the busiest bouts of one cycle that DLC never saw: (rois, skips).

    `entry` is one value from `dlc_exclusion_spans`. The surviving bouts go through the SAME
    window construction as the trace search (`build_rois_for_cycle` with `n_climb=0`), so a clip
    here is framed exactly like a licking clip there -- whole bout plus `--lick-pad` of context --
    and only the selection differs. The category is renamed so the CSV, the clip filenames and the
    crop decision can tell the two apart.
    """
    keep, skips = bouts_outside_dlc(bout_start_times, bout_durations, entry["spans"],
                                    entry["coverage"], params["dlc_guard"])
    if not np.any(keep):
        return [], skips

    rois = build_rois_for_cycle(
        cap_data, time_data,
        np.asarray(bout_start_times, dtype=np.float64)[keep],
        np.asarray(bout_durations, dtype=np.float64)[keep],
        np.asarray(bout_lick_counts)[keep],
        lick_times,
        {"n_lick": params["n_lick"], "n_climb": 0,
         "roi_seconds": params["roi_seconds"], "lick_pad": params["lick_pad"]},
    )
    for roi in rois:
        roi["category"] = "no_dlc"
    return rois, skips
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS (all tests in the file, including the pre-existing `build_rois_for_cycle` ones)

- [ ] **Step 6: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): build no_dlc regions from the bouts DLC never saw"
```

---

### Task 4: Name `no_dlc` clips and render them full-frame

**Files:**
- Modify: `scripts/find_interesting_windows.py` — `build_command` (crop decision at line 809)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: `build_command(row, out_dir, offsets, combined_h5, speed=1.0) -> list[str]` (unchanged signature).
- Produces: nothing new; behaviour change only.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_command_no_dlc_renders_full_frame():
    # The crop box is framed on the sipper tip. A no_dlc clip exists precisely to show where the
    # animal WAS instead, which is outside that box -- cropping would hide the evidence.
    row = {"animal": "A1", "cycle": 3, "category": "no_dlc", "rank": 0,
           "start": 10.0, "end": 22.0, "restart": False,
           "raw_h5": "/data/raw.h5", "layout": "/data/layout.csv"}
    lines = build_command(row, "clips", {}, "/data/combined.h5")
    assert "--no-crop" in lines[-1]
    assert "clips/A1_c3_no_dlc0.mp4" in lines[-1]


def test_build_command_no_dlc_restart_warning_recommends_offset():
    # A no_dlc window comes from the TRACE (bout times), not from DLC frames, so --offset is the
    # right remedy for a restart recording -- the opposite of the advice a "dlc" row gets.
    row = {"animal": "A1", "cycle": 0, "category": "no_dlc", "rank": 1,
           "start": 10.0, "end": 22.0, "restart": True,
           "raw_h5": "/data/raw.h5", "layout": "/data/layout.csv"}
    lines = build_command(row, "clips", {}, "/data/combined.h5")
    warning = "\n".join(lines[:-1])
    assert "--offset 0=<seconds>" in warning
    assert "Do NOT use --offset" not in warning
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k "no_dlc_renders or no_dlc_restart" -v`
Expected: the first FAILS (`assert "--no-crop" in ...`); the second already PASSES, because `build_command`'s restart branch keys on `category == "dlc"` and `no_dlc` correctly falls to the `else`. Keep it — it pins that behaviour against a future `startswith("dlc")` refactor.

- [ ] **Step 3: Extend the crop decision**

Replace:

```python
    crop_flag = " --no-crop" if row["category"] == "climb" else ""
```

with:

```python
    # Both of these show the animal AWAY from the sipper tip the crop box is framed on: a climbing
    # clip by definition, and a no_dlc clip because DeepLabCut placed the animal nowhere near the
    # sipper. Cropping would hide the very thing the clip exists to show.
    crop_flag = " --no-crop" if row["category"] in ("climb", "no_dlc") else ""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): render no_dlc clips full-frame"
```

---

### Task 5: Wire the mode — cycle keys, driver, and skip report

**Files:**
- Modify: `scripts/find_interesting_windows.py` — `build_cycles_for_dlc` (the `cycles[stem] = {...}` literal, line 738), plus new functions after `print_dlc_skips`
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: `dlc_exclusion_spans` (Task 1), `build_no_dlc_rois_for_cycle` (Task 3), existing `build_cycles_for_dlc`, `read_dlc_windows`, `write_outputs`.
- Produces:
  - `build_cycles_for_dlc` entries gain a `"cycle_key"` field: the HDF5 group key verbatim (a `str`), alongside the existing numeric `"cycle"`.
  - `run_dlc_exclude_mode(args, raw_map) -> (all_rows, skips, n_cycles_used)`.
  - `print_dlc_exclude_skips(skips, n_rows, n_cycles_used)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_cycles_for_dlc_keeps_the_raw_group_key(tmp_path):
    # run_dlc_exclude_mode has to re-open the filmed animal's group, so the literal HDF5 key must
    # survive: int("07") == 7 would not find a group named "07".
    import h5py
    path = tmp_path / "combined.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("A1/0")
        group.create_dataset("time_data", data=np.arange(10, dtype=np.float64))
        group.attrs["raw_h5"] = "/data/raw_data_2026-07-24_12-02-14.h5"
        group.attrs["layout"] = "/data/layout.csv"

    cycles = build_cycles_for_dlc(str(path))
    entry = cycles["raw_data_2026-07-24_12-02-14"]
    assert entry["cycle_key"] == "0"
    assert entry["cycle"] == 0


def test_print_dlc_exclude_skips_reports_counts(capsys):
    skips = {reason: 0 for reason in DLC_EXCLUDE_SKIP_REASONS}
    skips["no_dlc_video"] = 2
    skips["in_dlc_window"] = 7
    print_dlc_exclude_skips(skips, n_rows=12, n_cycles_used=3)
    out = capsys.readouterr().out
    assert "12 DLC window(s)" in out
    assert "3 cycle(s)" in out
    assert "no DLC windows" in out          # the no_dlc_video explanation
    assert "7" in out                        # bouts DLC agrees about
```

Add `DLC_EXCLUDE_SKIP_REASONS` and `print_dlc_exclude_skips` to the test file's import block (`build_cycles_for_dlc` is already imported).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k "raw_group_key or exclude_skips" -v`
Expected: FAIL — `ImportError: cannot import name 'print_dlc_exclude_skips'`

- [ ] **Step 3: Add `cycle_key` to `build_cycles_for_dlc`**

In the `cycles[stem] = {...}` literal, add one entry next to `"cycle": cycle_value,`:

```python
                cycles[stem] = {
                    "cycle": cycle_value,
                    # The HDF5 key verbatim: --dlc-exclude re-opens this group to read the bouts,
                    # and int("07") would not find a group named "07".
                    "cycle_key": cycle_key,
                    "raw_h5": raw_h5_path,
                    ...
```

(leave every other field as it is).

- [ ] **Step 4: Write the driver and the report**

Add after `print_dlc_skips`:

```python
def run_dlc_exclude_mode(args, raw_map):
    """--dlc-exclude: regions for the licking bouts that fall in no DLC at-sipper window.

    Returns (rows, skips, n_cycles_used). Only cycles with usable DLC evidence contribute: a cycle
    the CSV never mentions is counted `no_dlc_video` and emits nothing, because "DLC found no
    windows here" and "DLC never ran on this video" look identical from the CSV, and treating the
    second as the first would report a whole session's bouts as false positives.
    """
    cycles = build_cycles_for_dlc(args.combined_h5, raw_map, args.animals)
    dlc_rows = read_dlc_windows(args.dlc_exclude)
    spans_by_stem, skips = dlc_exclusion_spans(dlc_rows, cycles)

    params = {
        "n_lick": args.n_lick,
        "roi_seconds": args.roi_seconds,
        "lick_pad": args.lick_pad,
        "dlc_guard": args.dlc_guard,
    }

    all_rows = []
    n_cycles_used = 0
    with h5py.File(args.combined_h5, "r") as combined:
        for stem, cycle in cycles.items():
            entry = spans_by_stem.get(stem)
            if entry is None:
                # Either the video was unusable (already counted by reason) or the CSV never
                # mentions this recording at all.
                if stem not in {parse_dlc_video_stem(r["video"])[0] for r in dlc_rows}:
                    skips["no_dlc_video"] += 1
                continue

            animal = cycle["animal"]
            group = combined.get(animal, {}).get(cycle["cycle_key"]) if animal else None
            required = ("cap_data", "time_data", "bout_start_times", "bout_durations",
                        "bout_lick_counts", "lick_times")
            if group is None or not all(name in group for name in required):
                skips["no_trace"] += 1
                continue

            time_data = group["time_data"][:]
            if len(time_data) < 2:
                skips["no_trace"] += 1
                continue

            rois, bout_skips = build_no_dlc_rois_for_cycle(
                group["cap_data"][:], time_data,
                group["bout_start_times"][:], group["bout_durations"][:],
                group["bout_lick_counts"][:], group["lick_times"][:],
                entry, params,
            )
            for reason, count in bout_skips.items():
                skips[reason] += count
            n_cycles_used += 1

            for roi in rois:
                roi.update({
                    "animal": animal,
                    "cycle": cycle["cycle"],
                    "filmed": True,          # by construction: only the filmed animal is read
                    "restart": cycle["restart"],
                    "raw_h5": cycle["raw_h5"] or "",
                    "layout": cycle["layout"] or "",
                })
                all_rows.append(roi)

    return all_rows, skips, n_cycles_used


def print_dlc_exclude_skips(skips, n_rows, n_cycles_used):
    """Report what --dlc-exclude read and what it refused to draw conclusions from."""
    explanations = {
        "cfr_video": "video(s) are _cfr re-encodes (frame indices are not the original "
                     "container's; re-run DLC on the original recording)",
        "no_cycle": "video(s) match no cycle in the combined file (or were filtered out by "
                    "--animals)",
        "no_video_times": "video(s) whose recording or PTS sidecar is missing on this machine",
        "no_trace": "cycle(s) with no usable trace for the filmed animal",
        "frame_out_of_range": "DLC window(s) whose start_frame is negative or past the end of "
                              "the PTS sidecar",
        "no_dlc_video": "cycle(s) with no DLC windows in the CSV at all -- SKIPPED, not treated "
                        "as 'the animal was never at the sipper'. Run find_dlc_windows.py on "
                        "these recordings before trusting their absence here",
        "outside_video": "bout(s) outside the video's coverage, so DLC never observed them",
        "in_dlc_window": "bout(s) inside a DLC window (with the guard band) -- DLC agrees the "
                         "animal was at the sipper",
    }
    print(f"DLC-exclude mode: read {n_rows} DLC window(s); searched {n_cycles_used} cycle(s).")
    for reason in DLC_EXCLUDE_SKIP_REASONS:
        count = skips.get(reason, 0)
        if count:
            print(f"  skipped {count}: {explanations.get(reason, reason)}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): drive --dlc-exclude over the combined file's cycles"
```

---

### Task 6: CLI flags and documentation

**Files:**
- Modify: `scripts/find_interesting_windows.py` — module docstring (top of file), `main()` argument parser and dispatch (lines 915-1000)
- Test: `tests/test_find_interesting_windows.py`

**Interfaces:**
- Consumes: `run_dlc_exclude_mode`, `print_dlc_exclude_skips` (Task 5), existing `write_outputs`.
- Produces: `--dlc-exclude CSV` and `--dlc-guard SECONDS` command-line options.

- [ ] **Step 1: Write the failing tests**

```python
def test_main_rejects_dlc_exclude_together_with_dlc_windows(tmp_path):
    from scripts.find_interesting_windows import main
    combined = tmp_path / "combined.h5"
    combined.write_bytes(b"")
    with pytest.raises(SystemExit) as excinfo:
        main([str(combined), "--dlc-windows", "a.csv", "--dlc-exclude", "b.csv"])
    assert excinfo.value.code != 0


def test_main_dlc_exclude_writes_outputs(tmp_path, monkeypatch):
    from scripts import find_interesting_windows as fiw

    captured = {}

    def fake_run(args, raw_map):
        captured["dlc_exclude"] = args.dlc_exclude
        captured["dlc_guard"] = args.dlc_guard
        row = {"animal": "A1", "cycle": 0, "category": "no_dlc", "rank": 0,
               "start": 10.0, "end": 22.0, "center": 16.0, "score": 30.0,
               "n_licks_in_window": 30, "filmed": True, "restart": False,
               "raw_h5": "/data/raw.h5", "layout": "/data/layout.csv"}
        return [row], {reason: 0 for reason in fiw.DLC_EXCLUDE_SKIP_REASONS}, 1

    monkeypatch.setattr(fiw, "run_dlc_exclude_mode", fake_run)

    csv_path = tmp_path / "fp_rois.csv"
    sh_path = tmp_path / "make_fp_clips.sh"
    rc = fiw.main([str(tmp_path / "combined.h5"), "--dlc-exclude", "windows.csv",
                   "--csv", str(csv_path), "--sh", str(sh_path)])

    assert rc == 0
    assert captured["dlc_exclude"] == "windows.csv"
    assert captured["dlc_guard"] == 1.0          # default
    assert "no_dlc" in csv_path.read_text()
    assert "--no-crop" in sh_path.read_text()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_find_interesting_windows.py -k "dlc_exclude_together or dlc_exclude_writes" -v`
Expected: FAIL — `argparse` errors on the unknown `--dlc-exclude` option (SystemExit code 2 from the second test's `main` call, and the first test passing for the wrong reason until the flag exists)

- [ ] **Step 3: Add the flags and the dispatch**

After the existing `--dlc-windows` argument in `main()`, add:

```python
    parser.add_argument("--dlc-exclude", dest="dlc_exclude", default=None, metavar="CSV",
                        help="CSV written by dlc_integration/find_dlc_windows.py. Given it, the "
                             "script emits the busiest detected licking bouts that fall in NO DLC "
                             "window -- the capacitance detector's false-positive candidates. A "
                             "cycle absent from the CSV is skipped, never treated as 'the animal "
                             "was never at the sipper'. Cannot be combined with --dlc-windows. "
                             "The climbing options (--n-climb/--var-window/--min-var/"
                             "--climb-skip-edges) and --include-controls are unused in this mode")
    parser.add_argument("--dlc-guard", type=float, default=1.0, metavar="SECONDS",
                        help="with --dlc-exclude, grow every DLC window by this many seconds on "
                             "each side before testing a bout against it (default 1.0; 0 "
                             "disables). DLC window edges are merge-gap/pad artefacts, so a bout "
                             "that starts just before a window opens is usually a real lick")
```

Then, immediately after `args = parser.parse_args(argv)`:

```python
    if args.dlc_windows and args.dlc_exclude:
        parser.error("--dlc-windows and --dlc-exclude select windows by opposite criteria; "
                     "pass only one")
```

And after the existing `if args.dlc_windows:` block, add the new dispatch:

```python
    if args.dlc_exclude:
        all_rows, skips, n_cycles_used = run_dlc_exclude_mode(args, raw_map)
        write_outputs(all_rows, args)
        print_dlc_exclude_skips(skips, len(read_dlc_windows(args.dlc_exclude)), n_cycles_used)
        return 0
```

Note: `read_dlc_windows` is called a second time only to report the row count; that is a few
hundred rows of CSV and keeps `run_dlc_exclude_mode`'s return value to what the caller needs.

- [ ] **Step 4: Update the module docstring**

The module docstring at the top of `scripts/find_interesting_windows.py` describes the modes. Add a third bullet/paragraph in the same voice as the existing DLC-mode text:

```
    # the licks DeepLabCut never saw: false-positive candidates
    python scripts/find_interesting_windows.py results_combined.h5 \
        --dlc-exclude second_iteration_windows_with_tongue.csv \
        --csv fp_rois.csv --sh make_fp_clips.sh --speed 0.25
```

and this paragraph immediately above it:

```
`--dlc-exclude` runs the opposite selection to `--dlc-windows`: it keeps the busiest detected
licking bouts that touch NO DLC window (each grown by `--dlc-guard` seconds, because a window's
edges are merge-gap/pad artefacts) and that lie entirely inside the video's coverage. Those are the
stretches where the capacitance detector reported licking and DeepLabCut says the animal was not at
the sipper -- the false-positive candidates. A cycle with no rows in the DLC CSV is SKIPPED and
reported, never reported wholesale as licks-without-the-animal: "DLC found nothing here" and "DLC
never ran on this video" are indistinguishable from the CSV.
```

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/test_find_interesting_windows.py -v`
Expected: PASS

- [ ] **Step 6: Smoke-test against the real data**

Run:

```bash
python scripts/find_interesting_windows.py \
    "Lickometry Data/results_combined_ACG-26-3_2026-07-22_23_24_27_28_29_basic-algorithm.h5" \
    --dlc-exclude second_iteration_windows_with_tongue.csv \
    --n-lick 5 --speed 0.25 \
    --csv fp_rois.csv --sh make_fp_clips.sh
```

Expected: a summary naming the cycles searched and every skip count; `fp_rois.csv` containing
`no_dlc` rows; `make_fp_clips.sh` holding `make_sync_video.py ... --no-crop --speed 0.25` commands.
Sanity-check two rows by hand: their `start`/`end` must not fall inside any window of the DLC CSV
for the same recording. Report the counts in the commit message; do NOT commit `fp_rois.csv` or
`make_fp_clips.sh` (both are generated artefacts, like the existing `rois.csv` / `make_clips.sh`).

- [ ] **Step 7: Commit**

```bash
git add scripts/find_interesting_windows.py tests/test_find_interesting_windows.py
git commit -m "feat(rois): add --dlc-exclude mode for licks DLC never saw"
```

---

## Spec amendment applied during planning

Task 4 (`--no-crop` for `no_dlc` clips) is not in the spec. It follows from the same reasoning the
spec's sibling design gives for climbing clips: the crop box is framed on the sipper tip, and a
`no_dlc` clip exists to show that the animal was somewhere else. Add a short "Output" note to
`docs/superpowers/specs/2026-08-11-dlc-excluded-lick-windows-design.md` recording this, in the same
commit as Task 4.
