# DLC Sipper-Proximity Window Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate DeepLabCut review windows on the nose being confidently detected *and* close to the sipper tip, with an opt-in filter for rhythmic tongue appearance, instead of nose likelihood alone.

**Architecture:** `dlc_integration/find_dlc_windows.py` currently thresholds one bodypart's likelihood to decide "the mouse is here". We add three pure-numpy helpers to that file — a per-session static sipper anchor built from the four `sipper_*` keypoint medians, a point-to-polyline distance, and a tongue-likelihood upward-crossing rate — then feed a combined boolean mask into the existing (unchanged) merge/pad/split window pipeline. `find_windows` starts taking a boolean mask rather than `(likelihood, pcutoff)`, which requires a one-line update at its only other call site in `extract_outliers.py`.

**Tech Stack:** Python 3.13, numpy, h5py (pandas/pytables optional — the module has an h5py fallback because the laptop env has no pytables), pytest 8.3.4.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-10-dlc-sipper-proximity-windows-design.md`. Read it before starting.
- **No new dependencies.** numpy only for the new maths. Do not add scipy; do not add pandas as a hard requirement.
- **The laptop environment has no pytables.** `python -c "import tables"` fails. Every test must run without it, and the h5py fallback path in `load_dlc_h5` must keep working.
- **Sipper keypoint names, in anatomical order:** `sipper_top`, `sipper_midtop`, `sipper_midbottom`, `sipper_bottom`. Never re-sort them by coordinate.
- **Defaults, exactly:** `--sipper-pcutoff 0.6`, `--max-nose-dist 0.6` (fraction of arc length), `--max-nose-dist-px None`, `--tongue-pcutoff 0.6`, `--tongue-min-rate 3.0`, `--require-tongue` off. Existing defaults (`--pcutoff 0.8`, `--merge-gap 120`, `--min-frames 30`, `--min-confident 15`, `--pad 60`, `--max-frames 3600`) are unchanged.
- **`--max-nose-dist 0` with no `--max-nose-dist-px` must reproduce today's behavior exactly.**
- **Distances are always reported in pixels**, even when the threshold was given as a fraction.
- Run tests from the repository root: `python -m pytest tests/test_dlc_windows.py -v`.
- Commit after every task. Do not push.

## File Structure

- `dlc_integration/find_dlc_windows.py` (modify) — all production changes. Currently 474 lines and organized in banner-comment sections (`reading DLC predictions`, `window construction`, `driver`); keep that structure. New geometry/rhythm helpers go in a new `sipper geometry and tongue rhythm` section between `reading DLC predictions` and `window construction`.
- `dlc_integration/extract_outliers.py` (modify, one line at `gate_mask`, ~line 241) — adapt to the new `find_windows` signature.
- `tests/test_dlc_windows.py` (create) — pure-numpy unit tests plus one skipif-guarded test against a real prediction file.

`dlc_integration/` is not a package (no `__init__.py`), and `extract_outliers.py` imports its sibling via `sys.path.insert`. The test file does the same:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dlc_integration"))

import find_dlc_windows  # noqa: E402
```

---

### Task 1: Point-to-polyline distance

**Files:**
- Modify: `dlc_integration/find_dlc_windows.py` (new section after `probe_fps`, before the `window construction` banner)
- Test: `tests/test_dlc_windows.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `point_to_polyline_distance(px, py, points) -> np.ndarray`, where `px`/`py` are float arrays of length n and `points` is a list of at least two `(x, y)` float tuples in anatomical order. Returns a length-n float array: the minimum distance from each point to any segment of the polyline.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dlc_windows.py`:

```python
"""Unit tests for the DLC window-selection helpers in dlc_integration/find_dlc_windows.py.

`dlc_integration/` is not a package, so we put it on sys.path the same way
`extract_outliers.py` does rather than inventing an import mechanism for tests only.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dlc_integration"))

import find_dlc_windows as fdw  # noqa: E402


# ------------------------------------------------------------------ geometry
def test_distance_perpendicular_to_segment_interior():
    """A point off the middle of a horizontal segment: distance is the perpendicular drop."""
    points = [(0.0, 0.0), (10.0, 0.0)]
    d = fdw.point_to_polyline_distance(np.array([5.0]), np.array([3.0]), points)
    assert d == pytest.approx([3.0])


def test_distance_beyond_endpoint_clamps_to_endpoint():
    """Past the end of the segment, the nearest point is the endpoint itself, not the
    infinite line: (14, 3) is sqrt(4^2 + 3^2) = 5 from the endpoint (10, 0)."""
    points = [(0.0, 0.0), (10.0, 0.0)]
    d = fdw.point_to_polyline_distance(np.array([14.0]), np.array([3.0]), points)
    assert d == pytest.approx([5.0])


def test_distance_on_the_polyline_is_zero():
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    d = fdw.point_to_polyline_distance(np.array([10.0]), np.array([4.0]), points)
    assert d == pytest.approx([0.0])


def test_distance_takes_the_minimum_over_segments():
    """An L-shaped polyline: this point is ~8.06 from the first segment, 1.0 from the second."""
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    d = fdw.point_to_polyline_distance(np.array([11.0]), np.array([8.0]), points)
    assert d == pytest.approx([1.0])


def test_distance_degenerate_segment_uses_the_shared_endpoint():
    """Two identical consecutive points must not divide by zero."""
    points = [(3.0, 4.0), (3.0, 4.0)]
    d = fdw.point_to_polyline_distance(np.array([0.0]), np.array([0.0]), points)
    assert d == pytest.approx([5.0])


def test_distance_is_vectorized():
    points = [(0.0, 0.0), (10.0, 0.0)]
    d = fdw.point_to_polyline_distance(
        np.array([5.0, 14.0, 5.0]), np.array([3.0, 3.0, 0.0]), points
    )
    assert d == pytest.approx([3.0, 5.0, 0.0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dlc_windows.py -v`
Expected: 6 failures, `AttributeError: module 'find_dlc_windows' has no attribute 'point_to_polyline_distance'`.

- [ ] **Step 3: Write the implementation**

In `dlc_integration/find_dlc_windows.py`, after `probe_fps` and before the `window construction` banner comment, add:

```python
# --------------------------------------------------------------------------------------------
# sipper geometry and tongue rhythm
# --------------------------------------------------------------------------------------------
# The four sipper_* keypoints trace the curve of the sipper tip. Anatomical order matters: the
# polyline models the physical tube, so the points must be joined top -> bottom, never re-sorted
# by coordinate (the sipper sits diagonally in most recordings).
SIPPER_BODYPARTS = ("sipper_top", "sipper_midtop", "sipper_midbottom", "sipper_bottom")


def point_to_polyline_distance(px, py, points):
    """Distance from each (px, py) to the nearest point ON the polyline through `points`.

    Segment distance, not nearest-vertex distance: adjacent sipper keypoints are ~40-50 px apart,
    so a nose resting midway between two of them reads as up to ~25 px farther from the sipper
    than it really is if you only measure to the vertices.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    if len(points) < 2:
        raise ValueError("need at least two points to form a polyline")

    best = None
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
        vx, vy = bx - ax, by - ay
        length_sq = vx * vx + vy * vy
        if length_sq == 0:
            # Degenerate segment (two identical keypoint medians): fall back to the endpoint.
            d = np.hypot(px - ax, py - ay)
        else:
            t = np.clip(((px - ax) * vx + (py - ay) * vy) / length_sq, 0.0, 1.0)
            d = np.hypot(px - (ax + t * vx), py - (ay + t * vy))
        best = d if best is None else np.minimum(best, d)
    return best
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dlc_windows.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_dlc_windows.py dlc_integration/find_dlc_windows.py
git commit -m "feat(dlc): point-to-polyline distance for sipper proximity"
```

---

### Task 2: `load_dlc_h5` returns coordinates

**Files:**
- Modify: `dlc_integration/find_dlc_windows.py:95-144` (`load_dlc_h5`), and its caller at `dlc_integration/find_dlc_windows.py:307`
- Test: `tests/test_dlc_windows.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `load_dlc_h5(path) -> (scorer: str, bodyparts: list[str], coords: dict[str, dict[str, np.ndarray]])`. `coords[bodypart]` has keys `"x"`, `"y"`, `"likelihood"`, each a float array of length n_frames. Replaces the old third return value (a `{bodypart: likelihood_array}` dict). Both the pandas path and the h5py fallback populate all three keys.

- [ ] **Step 1: Write the failing test**

Real prediction files are ~1 GB of untracked session data, so this test is guarded — it runs on the lab laptop and cluster where the data lives, and skips in a bare checkout. Append to `tests/test_dlc_windows.py`:

```python
# ------------------------------------------------------------------ real-file smoke test
PRED_H5 = (
    REPO_ROOT / "Lickometry Data" / "ACG-26-3" / "dlc_analysis_results"
    / "raw_data_2026-07-21_12-59-50_cfrDLC_Resnet50_CLiQR_ValidationJul27shuffle1"
      "_snapshot_best-140.h5"
)
needs_predictions = pytest.mark.skipif(
    not PRED_H5.exists(), reason=f"no analyzed session data at {PRED_H5}"
)


@needs_predictions
def test_load_dlc_h5_returns_xy_and_likelihood():
    scorer, bodyparts, coords = fdw.load_dlc_h5(PRED_H5)
    assert scorer
    assert set(fdw.SIPPER_BODYPARTS) <= set(bodyparts)
    assert {"nose", "tongue"} <= set(bodyparts)
    nose = coords["nose"]
    assert set(nose) == {"x", "y", "likelihood"}
    n = nose["likelihood"].size
    assert n > 0
    assert nose["x"].size == n and nose["y"].size == n
    # Likelihoods are probabilities; coordinates are pixels well outside [0, 1].
    assert nose["likelihood"].min() >= 0.0 and nose["likelihood"].max() <= 1.0
    confident = nose["likelihood"] >= 0.8
    assert confident.any()
    assert nose["x"][confident].max() > 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_dlc_windows.py -v -k load_dlc_h5`
Expected: FAIL with `TypeError`/`KeyError` — the current function returns bare likelihood arrays, so `set(nose)` is not `{"x", "y", "likelihood"}`. (If it reports SKIPPED, the session data is missing: fetch it or run this task on a machine that has it — do not weaken the test.)

- [ ] **Step 3: Rewrite `load_dlc_h5`**

Replace the body of `load_dlc_h5` (`dlc_integration/find_dlc_windows.py:95-144`) with:

```python
def load_dlc_h5(path):
    """Return (scorer, bodyparts, coords) from a DLC prediction file.

    `coords[bodypart]` is a dict of `x`, `y`, `likelihood` float arrays, one entry per frame.

    Uses pandas/pytables when available; otherwise falls back to reading the pytables `table`
    dataset directly with h5py and unpickling the column MultiIndex out of the group attributes.
    The fallback exists so this script runs in the plain `cliqr-gui` environment (no pytables)
    as well as inside the DLC conda env on the cluster.
    """
    try:
        import pandas as pd

        df = pd.read_hdf(path)
        scorer = df.columns.get_level_values(0)[0]
        bodyparts = list(dict.fromkeys(df.columns.get_level_values(1)))
        coords = {
            bp: {
                coord: np.asarray(df[(scorer, bp, coord)].values, dtype=float)
                for coord in ("x", "y", "likelihood")
            }
            for bp in bodyparts
        }
        return scorer, bodyparts, coords
    except ImportError:
        pass  # no pytables -> h5py fallback below

    import h5py

    with h5py.File(path, "r") as fh:
        # DLC always writes a single group; find it rather than hard-coding "df_with_missing".
        group_names = [k for k in fh.keys() if isinstance(fh[k], h5py.Group) and "table" in fh[k]]
        if not group_names:
            raise ValueError(f"{path}: no pytables frame group found")
        grp = fh[group_names[0]]
        table = grp["table"]
        blocks = sorted(n for n in table.dtype.names if n.startswith("values_block_"))
        if len(blocks) != 1:
            raise ValueError(
                f"{path}: {len(blocks)} value blocks; install pytables "
                "(`pip install tables`) so pandas can read this file"
            )
        values = np.asarray(table[blocks[0]], dtype=float)
        columns = pickle.loads(bytes(grp.attrs["non_index_axes"]))[0][1]

    if values.shape[1] != len(columns):
        raise ValueError(f"{path}: {values.shape[1]} columns of data vs {len(columns)} names")

    scorer = columns[0][0]
    bodyparts = list(dict.fromkeys(col[1] for col in columns))
    coords = {}
    for i, (_scorer, bp, coord) in enumerate(columns):
        if coord in ("x", "y", "likelihood"):
            coords.setdefault(bp, {})[coord] = values[:, i]
    return scorer, bodyparts, coords
```

- [ ] **Step 4: Keep the existing caller compiling**

At `dlc_integration/find_dlc_windows.py:307`, inside `rows_for_file`, change:

```python
    scorer, bodyparts, likelihood = load_dlc_h5(h5_path)
    if args.bodypart not in likelihood:
        raise ValueError(
            f"bodypart '{args.bodypart}' not in this file. Available: {bodyparts}"
        )
    like = likelihood[args.bodypart]
```

to:

```python
    scorer, bodyparts, coords = load_dlc_h5(h5_path)
    if args.bodypart not in coords:
        raise ValueError(
            f"bodypart '{args.bodypart}' not in this file. Available: {bodyparts}"
        )
    like = coords[args.bodypart]["likelihood"]
```

Nothing else in `rows_for_file` changes yet — Task 6 rewrites it.

- [ ] **Step 5: Run the tests and a real end-to-end invocation**

Run: `python -m pytest tests/test_dlc_windows.py -v`
Expected: 7 passed (or the new one SKIPPED only in a data-less checkout).

Run: `python dlc_integration/find_dlc_windows.py "Lickometry Data/ACG-26-3/dlc_analysis_results/raw_data_2026-07-21_12-59-50_cfrDLC_Resnet50_CLiQR_ValidationJul27shuffle1_snapshot_best-140.h5" --csv /tmp/w_before.csv`
Expected: stderr reports `30 windows` for this file (the pre-change baseline; keep `/tmp/w_before.csv` — Task 6 compares against it).

- [ ] **Step 6: Commit**

```bash
git add tests/test_dlc_windows.py dlc_integration/find_dlc_windows.py
git commit -m "refactor(dlc): load_dlc_h5 returns x/y as well as likelihood"
```

---

### Task 3: Sipper anchor

**Files:**
- Modify: `dlc_integration/find_dlc_windows.py` (the `sipper geometry and tongue rhythm` section added in Task 1)
- Test: `tests/test_dlc_windows.py`

**Interfaces:**
- Consumes: `SIPPER_BODYPARTS`, `load_dlc_h5`'s `coords` dict shape from Tasks 1-2.
- Produces: `sipper_anchor(coords, pcutoff=0.6, min_frames=100) -> (points, arc_length)`. `points` is a list of `(x, y)` float tuples — the per-keypoint medians over confident frames, in `SIPPER_BODYPARTS` order, skipping keypoints absent from the file or with fewer than `min_frames` confident frames. `arc_length` is the summed distance between consecutive surviving points. Raises `ValueError` when fewer than two points survive.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dlc_windows.py`, after the geometry tests:

```python
# ------------------------------------------------------------------ sipper anchor
def _sipper_coords(positions, n=500, confident=None, jitter=0.0):
    """Synthetic `coords` dict: each sipper keypoint parked at a fixed position.

    `positions` maps bodypart -> (x, y). `confident` maps bodypart -> how many of the n frames
    clear likelihood 0.9 (the rest are 0.1, with their coordinates thrown far away so a median
    that fails to mask them is obviously wrong).
    """
    coords = {}
    for bp, (x, y) in positions.items():
        k = n if confident is None else confident[bp]
        like = np.concatenate([np.full(k, 0.95), np.full(n - k, 0.1)])
        xs = np.concatenate([np.full(k, x), np.full(n - k, x + 5000.0)])
        ys = np.concatenate([np.full(k, y), np.full(n - k, y + 5000.0)])
        if jitter:
            rng = np.random.default_rng(0)
            xs[:k] += rng.uniform(-jitter, jitter, k)
            ys[:k] += rng.uniform(-jitter, jitter, k)
        coords[bp] = {"x": xs, "y": ys, "likelihood": like}
    return coords


def test_sipper_anchor_medians_ignore_low_likelihood_frames():
    coords = _sipper_coords(
        {"sipper_top": (100.0, 100.0), "sipper_midtop": (100.0, 130.0),
         "sipper_midbottom": (100.0, 160.0), "sipper_bottom": (100.0, 190.0)},
        n=500, confident={"sipper_top": 400, "sipper_midtop": 400,
                          "sipper_midbottom": 400, "sipper_bottom": 400},
    )
    points, arc = fdw.sipper_anchor(coords, pcutoff=0.6, min_frames=100)
    np.testing.assert_allclose(points, [(100.0, 100.0), (100.0, 130.0),
                                        (100.0, 160.0), (100.0, 190.0)])
    assert arc == pytest.approx(90.0)


def test_sipper_anchor_drops_keypoints_with_too_few_confident_frames():
    coords = _sipper_coords(
        {"sipper_top": (100.0, 100.0), "sipper_midtop": (100.0, 130.0),
         "sipper_midbottom": (100.0, 160.0), "sipper_bottom": (100.0, 190.0)},
        n=500, confident={"sipper_top": 5, "sipper_midtop": 400,
                          "sipper_midbottom": 400, "sipper_bottom": 400},
    )
    points, arc = fdw.sipper_anchor(coords, pcutoff=0.6, min_frames=100)
    np.testing.assert_allclose(points, [(100.0, 130.0), (100.0, 160.0), (100.0, 190.0)])
    assert arc == pytest.approx(60.0)


def test_sipper_anchor_keeps_anatomical_order_for_a_diagonal_sipper():
    """Points are joined top -> bottom even though sorting by x or y would reorder them."""
    coords = _sipper_coords(
        {"sipper_top": (200.0, 100.0), "sipper_midtop": (230.0, 140.0),
         "sipper_midbottom": (180.0, 180.0), "sipper_bottom": (120.0, 190.0)},
    )
    points, arc = fdw.sipper_anchor(coords)
    np.testing.assert_allclose(points, [(200.0, 100.0), (230.0, 140.0),
                                        (180.0, 180.0), (120.0, 190.0)])
    expected = 50.0 + np.hypot(50.0, 40.0) + np.hypot(60.0, 10.0)
    assert arc == pytest.approx(expected)


def test_sipper_anchor_raises_when_fewer_than_two_keypoints_survive():
    coords = _sipper_coords({"sipper_top": (100.0, 100.0)})
    with pytest.raises(ValueError, match="usable sipper keypoints"):
        fdw.sipper_anchor(coords)


@needs_predictions
def test_sipper_anchor_on_a_real_session():
    _scorer, _bodyparts, coords = fdw.load_dlc_h5(PRED_H5)
    points, arc = fdw.sipper_anchor(coords)
    assert len(points) == 4
    # Measured across all ten analyzed ACG-26-3 sessions: 140-165 px.
    assert 100.0 < arc < 250.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dlc_windows.py -v -k sipper_anchor`
Expected: 5 failures, `AttributeError: module 'find_dlc_windows' has no attribute 'sipper_anchor'`.

- [ ] **Step 3: Write the implementation**

Add to the `sipper geometry and tongue rhythm` section, after `point_to_polyline_distance`:

```python
def sipper_anchor(coords, pcutoff=0.6, min_frames=100):
    """Static per-session sipper position: (points, arc_length_px).

    The sipper does not move within a recording -- across the ten analyzed ACG-26-3 sessions each
    keypoint's position IQR over confident frames is 0.5-3.5 px -- so one median per keypoint is
    both more robust and cheaper than tracking it frame by frame, and it survives the 1-23% of
    frames where a keypoint drops below `pcutoff`.

    `arc_length` is the length of the polyline through the surviving keypoints. It is the natural
    scale of the sipper in this recording (140-165 px across our sessions, varying with camera
    distance), which is what makes a proximity threshold expressed as a fraction of it portable
    between sessions.
    """
    points = []
    for bp in SIPPER_BODYPARTS:
        if bp not in coords:
            continue
        confident = coords[bp]["likelihood"] >= pcutoff
        if confident.sum() < min_frames:
            continue
        points.append(
            (float(np.median(coords[bp]["x"][confident])),
             float(np.median(coords[bp]["y"][confident])))
        )
    if len(points) < 2:
        raise ValueError(
            f"no usable sipper keypoints: fewer than two of {list(SIPPER_BODYPARTS)} have "
            f"{min_frames}+ frames at likelihood >= {pcutoff}"
        )
    arc_length = sum(
        float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(points[:-1], points[1:])
    )
    return points, arc_length
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dlc_windows.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_dlc_windows.py dlc_integration/find_dlc_windows.py
git commit -m "feat(dlc): static per-session sipper anchor from keypoint medians"
```

---

### Task 4: `find_windows` takes a boolean mask

**Files:**
- Modify: `dlc_integration/find_dlc_windows.py:241-253` (`find_windows`) and its caller in `rows_for_file` (~line 321)
- Modify: `dlc_integration/extract_outliers.py:241-249` (`gate_mask`)
- Test: `tests/test_dlc_windows.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `find_windows(mask, merge_gap, min_frames, min_confident, pad, max_frames) -> list[tuple[int, int]]` — half-open `[start, end)` frame ranges. The `pcutoff` parameter is gone; callers threshold first. Argument order is otherwise unchanged, and all callers pass by keyword except the first positional.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dlc_windows.py`:

```python
# ------------------------------------------------------------------ find_windows
def test_find_windows_takes_a_boolean_mask():
    mask = np.zeros(1000, dtype=bool)
    mask[100:400] = True
    windows = fdw.find_windows(
        mask, merge_gap=0, min_frames=30, min_confident=15, pad=0, max_frames=0
    )
    assert windows == [(100, 400)]


def test_find_windows_merges_across_gaps_and_drops_thin_windows():
    """Two runs 50 frames apart merge at merge_gap=120; a lone 5-frame flicker does not survive
    min_confident even though the merge gap would happily absorb it."""
    mask = np.zeros(2000, dtype=bool)
    mask[100:200] = True
    mask[250:350] = True   # 50-frame gap -> merges with the run above
    mask[1500:1505] = True  # isolated flicker -> dropped by min_confident=15
    windows = fdw.find_windows(
        mask, merge_gap=120, min_frames=30, min_confident=15, pad=0, max_frames=0
    )
    assert windows == [(100, 350)]


def test_find_windows_pads_and_splits():
    mask = np.zeros(1000, dtype=bool)
    mask[400:700] = True
    padded = fdw.find_windows(
        mask, merge_gap=0, min_frames=30, min_confident=15, pad=60, max_frames=0
    )
    assert padded == [(340, 760)]
    split = fdw.find_windows(
        mask, merge_gap=0, min_frames=30, min_confident=15, pad=0, max_frames=100
    )
    assert split == [(400, 500), (500, 600), (600, 700)]


def test_find_windows_clamps_padding_at_the_session_edges():
    mask = np.zeros(500, dtype=bool)
    mask[10:480] = True
    windows = fdw.find_windows(
        mask, merge_gap=0, min_frames=30, min_confident=15, pad=60, max_frames=0
    )
    assert windows == [(0, 500)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dlc_windows.py -v -k find_windows`
Expected: 4 failures, `TypeError: find_windows() missing 1 required positional argument: 'pcutoff'`.

- [ ] **Step 3: Change the signature**

Replace `find_windows` at `dlc_integration/find_dlc_windows.py:241-253` with:

```python
def find_windows(mask, merge_gap, min_frames, min_confident, pad, max_frames):
    """Full window pipeline over a per-frame "the animal is here" mask.

    Takes the mask rather than a likelihood array because the gate is no longer a single
    threshold: a frame counts when the bodypart is confidently detected AND close to the sipper.
    Callers build the mask; everything below is unchanged.

    Returns half-open [start, end) frame ranges.
    """
    mask = np.asarray(mask, dtype=bool)
    windows = merge_close(runs_of_true(mask), merge_gap)
    windows = [
        (s, e) for s, e in windows
        if e - s >= min_frames and mask[s:e].sum() >= min_confident
    ]
    if pad:
        n = mask.size
        windows = [(max(0, s - pad), min(n, e + pad)) for s, e in windows]
        windows = merge_close(windows, 0)
    return split_long(windows, max_frames)
```

- [ ] **Step 4: Update the in-file caller**

In `rows_for_file` (~line 321), change:

```python
    windows = find_windows(
        like,
        pcutoff=args.pcutoff,
        merge_gap=args.merge_gap,
```

to:

```python
    windows = find_windows(
        like >= args.pcutoff,
        merge_gap=args.merge_gap,
```

(leave the remaining keyword arguments as they are; Task 6 rewrites this block properly).

- [ ] **Step 5: Update the `extract_outliers.py` caller**

In `dlc_integration/extract_outliers.py`, inside `gate_mask` (~line 241), change:

```python
    windows = find_dlc_windows.find_windows(
        arrays[args.gate_bodypart][:, 2],
        pcutoff=args.gate_pcutoff,
        merge_gap=args.gate_merge_gap,
```

to:

```python
    windows = find_dlc_windows.find_windows(
        arrays[args.gate_bodypart][:, 2] >= args.gate_pcutoff,
        merge_gap=args.gate_merge_gap,
```

Its behavior is unchanged: it gated on `likelihood >= gate_pcutoff` before and does so now. It does not gain proximity gating in this work — its `--windows-csv` path already lets it replay the proximity-gated windows this script writes.

- [ ] **Step 6: Run the tests and check both entry points still start**

Run: `python -m pytest tests/test_dlc_windows.py -v`
Expected: 16 passed.

Run: `python dlc_integration/extract_outliers.py --help > /dev/null && python dlc_integration/find_dlc_windows.py --help > /dev/null && echo OK`
Expected: `OK` (imports resolve, no signature errors at module load).

Run: `git grep -n "find_windows(" -- '*.py' | grep -v tests`
Expected: only the definition and the two call sites you just edited — confirm no third caller was missed.

- [ ] **Step 7: Commit**

```bash
git add tests/test_dlc_windows.py dlc_integration/find_dlc_windows.py dlc_integration/extract_outliers.py
git commit -m "refactor(dlc): find_windows gates on a boolean mask"
```

---

### Task 5: Tongue upward-crossing rate

**Files:**
- Modify: `dlc_integration/find_dlc_windows.py` (the `sipper geometry and tongue rhythm` section)
- Test: `tests/test_dlc_windows.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `tongue_upcross_rate(likelihood, start, end, pcutoff, fps) -> float` — upward crossings of `pcutoff` within `[start, end)` divided by the window duration in seconds. Returns `0.0` for an empty or zero-duration window.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dlc_windows.py`:

```python
# ------------------------------------------------------------------ tongue rhythm
def _square_wave(n, period, duty=0.5, high=0.9, low=0.1):
    """Likelihood trace that rises above 0.6 once per `period` frames."""
    phase = np.arange(n) % period
    return np.where(phase < period * duty, high, low)


def test_tongue_rate_counts_one_crossing_per_cycle():
    """120 fps, a 15-frame period is 8 Hz -- squarely in the 7-9 Hz mouse licking band."""
    like = _square_wave(1200, period=15)
    rate = fdw.tongue_upcross_rate(like, 0, 1200, pcutoff=0.6, fps=120.0)
    # 1200 frames / 15 = 80 cycles, minus the leading run that starts already high, over 10 s.
    assert rate == pytest.approx(7.9, abs=0.15)


def test_tongue_rate_is_zero_when_never_confident():
    like = np.full(600, 0.05)
    assert fdw.tongue_upcross_rate(like, 0, 600, pcutoff=0.6, fps=120.0) == 0.0


def test_tongue_rate_is_zero_when_continuously_confident():
    """A tongue that never disappears is not licking -- it is a stuck detection."""
    like = np.full(600, 0.95)
    assert fdw.tongue_upcross_rate(like, 0, 600, pcutoff=0.6, fps=120.0) == 0.0


def test_tongue_rate_respects_the_window_bounds():
    like = np.full(1200, 0.05)
    like[600:] = _square_wave(600, period=15)
    quiet = fdw.tongue_upcross_rate(like, 0, 600, pcutoff=0.6, fps=120.0)
    busy = fdw.tongue_upcross_rate(like, 600, 1200, pcutoff=0.6, fps=120.0)
    assert quiet == 0.0
    assert busy > 7.0


def test_tongue_rate_of_an_empty_window_is_zero():
    like = _square_wave(600, period=15)
    assert fdw.tongue_upcross_rate(like, 300, 300, pcutoff=0.6, fps=120.0) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dlc_windows.py -v -k tongue_rate`
Expected: 5 failures, `AttributeError: module 'find_dlc_windows' has no attribute 'tongue_upcross_rate'`.

- [ ] **Step 3: Write the implementation**

Add to the `sipper geometry and tongue rhythm` section, after `sipper_anchor`:

```python
def tongue_upcross_rate(likelihood, start, end, pcutoff, fps):
    """Upward crossings of `pcutoff` per second within [start, end).

    The tongue is only visible at the top of each lick, so during drinking its likelihood PULSES
    rather than staying high: in the analyzed sessions, drinking stretches cross 0.6 upward 3.4-7.8
    times per second (a 7.9-9.1 Hz rhythm, part of it below the cutoff) while non-drinking
    stretches near the sipper cross it 0-0.4 times per second. Counting crossings separates the two
    without an FFT.

    A window that opens already above the cutoff contributes no crossing for that leading run. That
    undercounts by at most one crossing, which is immaterial against a 3+/s threshold.
    """
    duration = (end - start) / fps
    if duration <= 0:
        return 0.0
    confident = np.asarray(likelihood[start:end]) >= pcutoff
    crossings = int(np.sum(np.diff(confident.astype(np.int8)) == 1))
    return crossings / duration
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dlc_windows.py -v`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_dlc_windows.py dlc_integration/find_dlc_windows.py
git commit -m "feat(dlc): tongue-likelihood upcross rate for rhythmic licking"
```

---

### Task 6: Wire the gate into the driver — CLI, CSV, summary

**Files:**
- Modify: `dlc_integration/find_dlc_windows.py` — `CSV_FIELDS` (lines 74-89), `rows_for_file` (lines 306-355), `main`'s argparse block (lines 358-399), and the module docstring (lines 1-54)
- Test: `tests/test_dlc_windows.py`

**Interfaces:**
- Consumes: `point_to_polyline_distance`, `sipper_anchor`, `SIPPER_BODYPARTS` (Tasks 1, 3), `load_dlc_h5`'s `coords` (Task 2), mask-taking `find_windows` (Task 4), `tongue_upcross_rate` (Task 5).
- Produces: `build_near_mask(coords, bodypart, pcutoff, points, threshold_px) -> (mask, dist)` — `mask` is the per-frame gate, `dist` the per-frame distance in px (or `None` when proximity is disabled, i.e. `threshold_px` is `None`). Plus the finished CLI.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dlc_windows.py`:

```python
# ------------------------------------------------------------------ the gate
def _nose_coords(x, y, likelihood):
    return {"x": np.asarray(x, float), "y": np.asarray(y, float),
            "likelihood": np.asarray(likelihood, float)}


def test_build_near_mask_excludes_a_confident_but_distant_nose():
    """Frames 0-99: nose right at the sipper. Frames 100-199: nose confident but 400 px away."""
    points = [(100.0, 100.0), (100.0, 190.0)]
    n = 200
    x = np.concatenate([np.full(100, 105.0), np.full(100, 505.0)])
    y = np.full(n, 140.0)
    like = np.full(n, 0.95)
    coords = {"nose": _nose_coords(x, y, like)}
    mask, dist = fdw.build_near_mask(coords, "nose", 0.8, points, threshold_px=90.0)
    assert mask[:100].all()
    assert not mask[100:].any()
    assert dist[0] == pytest.approx(5.0)
    assert dist[100] == pytest.approx(405.0)


def test_build_near_mask_still_requires_likelihood():
    points = [(100.0, 100.0), (100.0, 190.0)]
    coords = {"nose": _nose_coords(np.full(50, 105.0), np.full(50, 140.0), np.full(50, 0.5))}
    mask, _dist = fdw.build_near_mask(coords, "nose", 0.8, points, threshold_px=90.0)
    assert not mask.any()


def test_build_near_mask_without_a_threshold_is_likelihood_only():
    """--max-nose-dist 0 must reproduce the old behavior exactly, distant nose included."""
    points = [(100.0, 100.0), (100.0, 190.0)]
    x = np.concatenate([np.full(100, 105.0), np.full(100, 5005.0)])
    coords = {"nose": _nose_coords(x, np.full(200, 140.0), np.full(200, 0.95))}
    mask, dist = fdw.build_near_mask(coords, "nose", 0.8, points, threshold_px=None)
    assert mask.all()
    assert dist is None


def test_build_near_mask_rejects_an_unknown_bodypart():
    points = [(100.0, 100.0), (100.0, 190.0)]
    coords = {"nose": _nose_coords([1.0], [1.0], [1.0])}
    with pytest.raises(ValueError, match="jaw"):
        fdw.build_near_mask(coords, "jaw", 0.8, points, threshold_px=90.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dlc_windows.py -v -k build_near_mask`
Expected: 4 failures, `AttributeError: module 'find_dlc_windows' has no attribute 'build_near_mask'`.

- [ ] **Step 3: Add `build_near_mask`**

Add to the `sipper geometry and tongue rhythm` section, after `tongue_upcross_rate`:

```python
def build_near_mask(coords, bodypart, pcutoff, points, threshold_px):
    """(mask, distance_px) for "the animal's `bodypart` is confidently AT the sipper".

    `threshold_px=None` disables the proximity term, reproducing the original likelihood-only gate;
    `distance_px` is then None as well, and the distance columns are left empty in the CSV.
    """
    if bodypart not in coords:
        raise ValueError(
            f"bodypart '{bodypart}' not in this file. Available: {sorted(coords)}"
        )
    part = coords[bodypart]
    confident = part["likelihood"] >= pcutoff
    if threshold_px is None:
        return confident, None
    dist = point_to_polyline_distance(part["x"], part["y"], points)
    return confident & (dist <= threshold_px), dist
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_dlc_windows.py -v`
Expected: 25 passed.

- [ ] **Step 5: Add the three CSV columns**

Replace `CSV_FIELDS` (lines 74-89) with:

```python
CSV_FIELDS = [
    "task_id",
    "label",
    "video",
    "h5",
    "scorer",
    "bodypart",
    "start_frame",
    "end_frame",
    "n_frames",
    "start_sec",
    "end_sec",
    "duration_sec",
    "frac_above",
    "mean_likelihood",
    # Appended, so the original column order is untouched. The distance columns are empty when
    # proximity gating is disabled (--max-nose-dist 0 and no --max-nose-dist-px).
    "mean_nose_dist",
    "min_nose_dist",
    "tongue_rate",
]
```

- [ ] **Step 6: Rewrite `rows_for_file`**

Replace `rows_for_file` (lines 306-355) with:

```python
def rows_for_file(h5_path, args, task_id_start):
    scorer, bodyparts, coords = load_dlc_h5(h5_path)

    video = Path(args.video) if args.video else guess_video(h5_path)
    if args.video_dir:
        # The CSV is usually written on the laptop but consumed on the cluster, where the videos
        # live in the DLC project's videos/ directory rather than next to the recordings.
        video = Path(args.video_dir) / video.name
    fps = args.fps or probe_fps(video) or 120.0

    # Proximity is off only when the user asked for it: --max-nose-dist 0 and no pixel override.
    if args.max_nose_dist_px is not None:
        points, arc_length = sipper_anchor(coords, pcutoff=args.sipper_pcutoff)
        threshold_px = float(args.max_nose_dist_px)
    elif args.max_nose_dist > 0:
        points, arc_length = sipper_anchor(coords, pcutoff=args.sipper_pcutoff)
        threshold_px = args.max_nose_dist * arc_length
    else:
        points, arc_length, threshold_px = None, None, None

    mask, dist = build_near_mask(coords, args.bodypart, args.pcutoff, points, threshold_px)
    like = coords[args.bodypart]["likelihood"]

    if args.require_tongue and "tongue" not in coords:
        raise ValueError(
            f"--require-tongue needs a 'tongue' bodypart; this file has {bodyparts}"
        )
    tongue = coords["tongue"]["likelihood"] if "tongue" in coords else None

    # Rhythm is judged on the merged window BEFORE padding and splitting, so the rate describes
    # the detected behavior rather than the context padding, and one long bout is judged once
    # rather than chunk by chunk.
    windows = find_windows(
        mask,
        merge_gap=args.merge_gap,
        min_frames=args.min_frames,
        min_confident=args.min_confident,
        pad=0,
        max_frames=0,
    )
    rates = {
        (s, e): (
            tongue_upcross_rate(tongue, s, e, args.tongue_pcutoff, fps)
            if tongue is not None else None
        )
        for s, e in windows
    }
    if args.require_tongue:
        windows = [w for w in windows if rates[w] >= args.tongue_min_rate]

    # Pad and split only what survived, reusing the same pipeline on an already-gated mask so the
    # edge clamping and re-merging rules stay in one place.
    kept = np.zeros(mask.size, dtype=bool)
    for s, e in windows:
        kept[s:e] = True
    final = find_windows(
        kept,
        merge_gap=0,
        min_frames=0,
        min_confident=0,
        pad=args.pad,
        max_frames=args.max_frames,
    )

    rows = []
    for i, (start, end) in enumerate(final):
        chunk = like[start:end]
        in_window = mask[start:end]
        # Report the rate of the pre-pad window(s) this row came from, NOT a rate recomputed over
        # the padded row: padding is quiet by construction and would dilute the number, so a user
        # tuning --tongue-min-rate from an unfiltered CSV would pick a threshold that is too low.
        overlapping = [r for (ws, we), r in rates.items() if ws < end and we > start]
        rate = max((r for r in overlapping if r is not None), default="")
        if dist is not None and in_window.any():
            near_dist = dist[start:end][in_window]
            mean_dist = round(float(np.mean(near_dist)), 2)
            min_dist = round(float(np.min(near_dist)), 2)
        else:
            mean_dist = min_dist = ""
        rows.append(
            {
                "task_id": task_id_start + i,
                "label": f"{video.stem}_w{i:03d}_f{start}-{end}",
                "video": str(video),
                "h5": str(h5_path),
                "scorer": scorer,
                "bodypart": args.bodypart,
                "start_frame": start,
                "end_frame": end,
                "n_frames": end - start,
                "start_sec": round(start / fps, 3),
                "end_sec": round(end / fps, 3),
                "duration_sec": round((end - start) / fps, 3),
                "frac_above": round(float(np.mean(chunk >= args.pcutoff)), 4),
                "mean_likelihood": round(float(np.mean(chunk)), 4),
                "mean_nose_dist": mean_dist,
                "min_nose_dist": min_dist,
                "tongue_rate": round(rate, 3) if rate != "" else "",
            }
        )
    return rows, dict(
        h5=str(h5_path), video=str(video), fps=fps, n_frames=int(like.size),
        frames_above=int(np.sum(like >= args.pcutoff)), frames_near=int(np.sum(mask)),
        sipper_scale=round(arc_length, 2) if arc_length is not None else None,
        nose_dist_thresh_px=round(threshold_px, 2) if threshold_px is not None else None,
        n_windows=len(final),
    )
```

- [ ] **Step 7: Add the CLI options**

In `main`, immediately after the existing `--pcutoff` argument (line 374), insert:

```python
    parser.add_argument("--sipper-pcutoff", type=float, default=0.6,
                        help="minimum likelihood for a sipper_* keypoint to contribute to the "
                             "session's static sipper position")
    parser.add_argument("--max-nose-dist", type=float, default=0.6,
                        help="how close the bodypart must be to the sipper, as a fraction of the "
                             "sipper tip's arc length (140-165 px in our recordings, so 0.6 is "
                             "~85-100 px); 0 disables proximity gating entirely")
    parser.add_argument("--max-nose-dist-px", type=float, default=None,
                        help="proximity threshold in raw pixels; overrides --max-nose-dist")
    parser.add_argument("--require-tongue", action="store_true",
                        help="keep only windows where the tongue appears rhythmically, i.e. the "
                             "animal was actually licking rather than just present")
    parser.add_argument("--tongue-pcutoff", type=float, default=0.6,
                        help="likelihood the tongue must cross for a crossing to count")
    parser.add_argument("--tongue-min-rate", type=float, default=3.0,
                        help="minimum upward tongue-likelihood crossings per second for "
                             "--require-tongue (drinking measured 3.4-7.8/s, non-drinking 0-0.4/s)")
```

- [ ] **Step 8: Update the per-file progress line**

In `main`, replace the `print(...)` that reports each file's result (lines 435-439) with:

```python
        scale = summary["sipper_scale"]
        gate = (f"within {summary['nose_dist_thresh_px']:.0f} px of the sipper "
                f"(arc {scale:.0f} px)" if scale is not None else "anywhere in frame")
        print(
            f"{Path(h5_path).name}: {summary['frames_near']}/{summary['n_frames']} frames "
            f">= {args.pcutoff} on '{args.bodypart}' and {gate} -> "
            f"{summary['n_windows']} windows",
            file=sys.stderr,
        )
```

- [ ] **Step 9: Update the module docstring**

In the `How a window is built (in order)` section of the module docstring (lines 20-33), replace step 1 and add a new step 4, renumbering the rest:

```
1. `mask = likelihood >= --pcutoff` over the chosen `--bodypart`, AND that bodypart within
   `--max-nose-dist` (a fraction of the sipper tip's arc length) of the sipper. The sipper's
   position is the per-session median of the four `sipper_*` keypoints, which move by only 0.5-3.5
   px within a recording. `--max-nose-dist 0` drops the proximity test and restores the original
   likelihood-only gate. Proximity matters because a confident nose ANYWHERE in frame -- the animal
   crossing the cage, grooming in a corner -- used to be enough to spend a render job on.
```

and after the min-frames/min-confident step:

```
4. With `--require-tongue`, windows whose tongue likelihood does not pulse at least
   `--tongue-min-rate` times per second are dropped. The tongue is only visible at the top of each
   lick, so drinking shows up as 3.4-7.8 upward crossings of `--tongue-pcutoff` per second against
   0-0.4/s for an animal that is merely present. The rate is written to the CSV on every run, so
   one unfiltered pass tells you where to put the threshold.
```

Also extend the "What it produces"/usage notes with an example invocation:

```
    # only clips where the animal was actually drinking
    python dlc_integration/find_dlc_windows.py "Lickometry Data/ACG-26-3/dlc_analysis_results" \
        --csv dlc_windows.csv --require-tongue
```

- [ ] **Step 10: Run the tests**

Run: `python -m pytest tests/test_dlc_windows.py -v`
Expected: 25 passed.

- [ ] **Step 11: Verify against the real sessions**

Run (proximity disabled must reproduce the Task 2 baseline byte-for-byte apart from the three new empty columns):

```bash
python dlc_integration/find_dlc_windows.py \
  "Lickometry Data/ACG-26-3/dlc_analysis_results/raw_data_2026-07-21_12-59-50_cfrDLC_Resnet50_CLiQR_ValidationJul27shuffle1_snapshot_best-140.h5" \
  --csv /tmp/w_off.csv --max-nose-dist 0
diff <(cut -d, -f1-14 /tmp/w_before.csv) <(cut -d, -f1-14 /tmp/w_off.csv) && echo "IDENTICAL"
```

Expected: `IDENTICAL`, and the run reports 30 windows.

Run the whole directory at the defaults and with the tongue filter:

```bash
python dlc_integration/find_dlc_windows.py "Lickometry Data/ACG-26-3/dlc_analysis_results" \
  --csv /tmp/w_prox.csv --summary-json /tmp/w_prox.json
python dlc_integration/find_dlc_windows.py "Lickometry Data/ACG-26-3/dlc_analysis_results" \
  --csv /tmp/w_tongue.csv --require-tongue
```

Expected, matching the prototype table in the spec (window counts, ±1 for padding interactions):

| session | `--max-nose-dist 0` | default | `--require-tongue` |
|---|---|---|---|
| 07-13 | 95 | ~26 | ~14 |
| 07-21 | 30 | ~9 | ~6 |
| 07-22 | 56 | ~27 | ~9 |
| 07-23 | 99 | ~35 | ~1 |
| 07-29 | 22 | ~9 | ~6 |

If a count is wildly off (2x or more), stop and investigate before committing — the likely causes are the anchor picking up a bad keypoint or the rhythm being measured after padding instead of before.

Also confirm the new columns are populated: `head -3 /tmp/w_prox.csv` should show non-empty `mean_nose_dist`, `min_nose_dist`, `tongue_rate`, and `/tmp/w_prox.json` should carry `sipper_scale` between 140 and 165 for each file.

Finally, check the parent-rate mapping did what it says — every row in the filtered CSV must carry a rate at or above the threshold, and none of them should be diluted by padding:

```bash
awk -F, 'NR>1 && $17+0 < 3.0 {print; bad=1} END {exit bad}' /tmp/w_tongue.csv && echo "ALL >= 3.0/s"
```

Expected: `ALL >= 3.0/s`. A row below 3.0 means the CSV rate was recomputed over the padded window instead of inherited from its pre-pad parent.

- [ ] **Step 12: Commit**

```bash
git add tests/test_dlc_windows.py dlc_integration/find_dlc_windows.py
git commit -m "feat(dlc): gate review windows on sipper proximity and tongue rhythm"
```

---

### Task 7: Regression check on `extract_outliers.py`

**Files:**
- Test: `tests/test_dlc_windows.py`
- Read-only: `dlc_integration/extract_outliers.py`

**Interfaces:**
- Consumes: `find_windows` (Task 4), `extract_outliers.gate_mask`.
- Produces: nothing; this task only proves Task 4 did not change the outlier-extraction gate.

- [ ] **Step 1: Write the test**

Append to `tests/test_dlc_windows.py`:

```python
# ------------------------------------------------------------------ extract_outliers regression
import types  # noqa: E402

import extract_outliers  # noqa: E402  (same sys.path entry as find_dlc_windows)


def test_extract_outliers_gate_mask_is_unchanged_by_the_signature_change():
    """gate_mask must still mean 'likelihood >= gate_pcutoff, run through the window pipeline'."""
    n = 2000
    like = np.full(n, 0.1)
    like[100:400] = 0.95
    like[1500:1505] = 0.95  # flicker, below gate_min_confident
    arrays = {"nose": np.column_stack([np.zeros(n), np.zeros(n), like])}
    args = types.SimpleNamespace(
        windows_csv=None, gate_bodypart="nose", gate_pcutoff=0.8, gate_merge_gap=120,
        gate_min_frames=30, gate_min_confident=15, gate_pad=0,
    )
    mask = extract_outliers.gate_mask(arrays, args, video="whatever.mp4", n_frames=n)
    assert mask[100:400].all()
    assert not mask[:100].any()
    assert not mask[400:].any()


def test_extract_outliers_gate_mask_rejects_an_unknown_bodypart():
    arrays = {"nose": np.zeros((10, 3))}
    args = types.SimpleNamespace(
        windows_csv=None, gate_bodypart="whisker", gate_pcutoff=0.8, gate_merge_gap=120,
        gate_min_frames=30, gate_min_confident=15, gate_pad=0,
    )
    with pytest.raises(ValueError, match="whisker"):
        extract_outliers.gate_mask(arrays, args, video="whatever.mp4", n_frames=10)
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_dlc_windows.py -v`
Expected: 27 passed. If the import of `extract_outliers` fails for a missing third-party module, note which one and stop — it must import with numpy alone for these tests to be worth having.

- [ ] **Step 3: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: no new failures compared to `git stash && python -m pytest tests/ -q && git stash pop` on the pre-change tree. Record both counts in the commit message if any pre-existing failures are present.

- [ ] **Step 4: Commit**

```bash
git add tests/test_dlc_windows.py
git commit -m "test(dlc): pin extract_outliers gate behavior after the mask refactor"
```

---

## Done when

- `python -m pytest tests/test_dlc_windows.py -v` → 27 passed.
- `--max-nose-dist 0` reproduces the pre-change CSV on the first 14 columns.
- Default run over `Lickometry Data/ACG-26-3/dlc_analysis_results` produces roughly the window counts in the spec's table, with `mean_nose_dist`, `min_nose_dist`, `tongue_rate` populated.
- `--require-tongue` cuts the window list further, again roughly matching the table.
- `python dlc_integration/extract_outliers.py --help` still works.
