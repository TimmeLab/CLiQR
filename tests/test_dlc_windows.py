"""Unit tests for the DLC window-selection helpers in dlc_integration/find_dlc_windows.py.

`dlc_integration/` is not a package, so we put it on sys.path the same way
`extract_outliers.py` does rather than inventing an import mechanism for tests only.
"""
import sys
import types
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


# ------------------------------------------------------------------ sipper anchor
def _sipper_coords(positions, n=500, confident=None):
    """Synthetic `coords` dict: each sipper keypoint parked at a fixed position.

    `positions` maps bodypart -> (x, y). `confident` maps bodypart -> how many of the n frames
    clear likelihood 0.95 (the rest are 0.1, with their coordinates thrown far away so a median
    that fails to mask them is obviously wrong).
    """
    coords = {}
    for bp, (x, y) in positions.items():
        k = n if confident is None else confident[bp]
        like = np.concatenate([np.full(k, 0.95), np.full(n - k, 0.1)])
        xs = np.concatenate([np.full(k, x), np.full(n - k, x + 5000.0)])
        ys = np.concatenate([np.full(k, y), np.full(n - k, y + 5000.0)])
        coords[bp] = {"x": xs, "y": ys, "likelihood": like}
    return coords


def test_sipper_anchor_medians_ignore_low_likelihood_frames():
    coords = _sipper_coords(
        {"sipper_top": (100.0, 100.0), "sipper_midtop": (100.0, 130.0),
         "sipper_midbottom": (100.0, 160.0), "sipper_bottom": (100.0, 190.0)},
        n=500, confident={"sipper_top": 200, "sipper_midtop": 200,
                          "sipper_midbottom": 200, "sipper_bottom": 200},
    )
    points, arc, iqr = fdw.sipper_anchor(coords, pcutoff=0.6, min_frames=100)
    np.testing.assert_allclose(points, [(100.0, 100.0), (100.0, 130.0),
                                        (100.0, 160.0), (100.0, 190.0)])
    assert arc == pytest.approx(90.0)
    # Every keypoint is parked at a fixed position (no spread) over the confident frames.
    assert set(iqr) == {"sipper_top", "sipper_midtop", "sipper_midbottom", "sipper_bottom"}
    assert all(v == (0.0, 0.0) for v in iqr.values())


def test_sipper_anchor_drops_keypoints_with_too_few_confident_frames():
    coords = _sipper_coords(
        {"sipper_top": (100.0, 100.0), "sipper_midtop": (100.0, 130.0),
         "sipper_midbottom": (100.0, 160.0), "sipper_bottom": (100.0, 190.0)},
        n=500, confident={"sipper_top": 5, "sipper_midtop": 200,
                          "sipper_midbottom": 200, "sipper_bottom": 200},
    )
    points, arc, iqr = fdw.sipper_anchor(coords, pcutoff=0.6, min_frames=100)
    np.testing.assert_allclose(points, [(100.0, 130.0), (100.0, 160.0), (100.0, 190.0)])
    assert arc == pytest.approx(60.0)
    # The dropped keypoint contributes no IQR entry at all.
    assert set(iqr) == {"sipper_midtop", "sipper_midbottom", "sipper_bottom"}


def test_sipper_anchor_keeps_anatomical_order_for_a_diagonal_sipper():
    """Points are joined top -> bottom even though sorting by x or y would reorder them."""
    coords = _sipper_coords(
        {"sipper_top": (200.0, 100.0), "sipper_midtop": (230.0, 140.0),
         "sipper_midbottom": (180.0, 180.0), "sipper_bottom": (120.0, 190.0)},
    )
    points, arc, _iqr = fdw.sipper_anchor(coords)
    np.testing.assert_allclose(points, [(200.0, 100.0), (230.0, 140.0),
                                        (180.0, 180.0), (120.0, 190.0)])
    expected = 50.0 + np.hypot(50.0, 40.0) + np.hypot(60.0, 10.0)
    assert arc == pytest.approx(expected)


def test_sipper_anchor_raises_when_fewer_than_two_keypoints_survive():
    coords = _sipper_coords({"sipper_top": (100.0, 100.0)})
    with pytest.raises(ValueError, match="usable sipper keypoints"):
        fdw.sipper_anchor(coords)


def test_sipper_anchor_error_names_available_bodyparts_and_the_escape_hatch():
    """The error should say what bodyparts the file DOES have (this model may just lack sipper
    keypoints) and that --max-nose-dist 0 disables proximity gating, not just fault the data."""
    coords = _sipper_coords({"sipper_top": (100.0, 100.0)})
    coords["jaw"] = coords.pop("sipper_top")
    with pytest.raises(ValueError, match=r"jaw") as excinfo:
        fdw.sipper_anchor(coords)
    assert "--max-nose-dist 0" in str(excinfo.value)


def test_sipper_anchor_ignores_nan_coordinates_in_otherwise_confident_frames():
    """A few NaN x/y values in confident frames must not NaN-poison the median: M1 regression.

    5 of the 200 "confident" frames for sipper_top carry a NaN coordinate. Excluding them leaves
    195 clean frames all parked at (100, 100), so the median is still exactly (100, 100) and the
    arc length is still exactly 90.0 -- not NaN.
    """
    coords = _sipper_coords(
        {"sipper_top": (100.0, 100.0), "sipper_midtop": (100.0, 130.0),
         "sipper_midbottom": (100.0, 160.0), "sipper_bottom": (100.0, 190.0)},
        n=500, confident={"sipper_top": 200, "sipper_midtop": 200,
                          "sipper_midbottom": 200, "sipper_bottom": 200},
    )
    nan_idx = [3, 17, 42, 99, 150]  # all within the first 200 (confident) frames
    coords["sipper_top"]["x"][nan_idx] = np.nan
    coords["sipper_top"]["y"][1] = np.nan
    points, arc, iqr = fdw.sipper_anchor(coords, pcutoff=0.6, min_frames=100)
    np.testing.assert_allclose(points, [(100.0, 100.0), (100.0, 130.0),
                                        (100.0, 160.0), (100.0, 190.0)])
    assert arc == pytest.approx(90.0)
    assert not any(np.isnan(v) for pair in iqr.values() for v in pair)


def test_sipper_anchor_nan_coordinates_count_against_min_frames():
    """A keypoint whose confident frames are mostly NaN must be dropped, not silently kept with a
    median computed over too few real points."""
    coords = _sipper_coords(
        {"sipper_top": (100.0, 100.0), "sipper_midtop": (100.0, 130.0),
         "sipper_midbottom": (100.0, 160.0), "sipper_bottom": (100.0, 190.0)},
        n=500, confident={"sipper_top": 200, "sipper_midtop": 200,
                          "sipper_midbottom": 200, "sipper_bottom": 200},
    )
    # Only 50 of sipper_top's 200 "confident" frames have finite coordinates -- below min_frames.
    coords["sipper_top"]["x"][50:200] = np.nan
    points, arc, iqr = fdw.sipper_anchor(coords, pcutoff=0.6, min_frames=100)
    np.testing.assert_allclose(points, [(100.0, 130.0), (100.0, 160.0), (100.0, 190.0)])
    assert arc == pytest.approx(60.0)
    assert "sipper_top" not in iqr


@needs_predictions
def test_sipper_anchor_on_a_real_session():
    _scorer, _bodyparts, coords = fdw.load_dlc_h5(PRED_H5)
    points, arc, _iqr = fdw.sipper_anchor(coords)
    assert len(points) == 4
    # Measured across all ten analyzed ACG-26-3 sessions: 140-165 px.
    assert 100.0 < arc < 250.0


# ------------------------------------------------------------------ find_windows
def test_find_windows_takes_a_boolean_mask():
    mask = np.zeros(1000, dtype=bool)
    mask[100:400] = True
    windows = fdw.find_windows(
        mask, merge_gap=0, min_frames=30, min_confident=15, pad=0, max_frames=0
    )
    assert windows == [(100, 400)]


def test_find_windows_merges_across_gaps_and_drops_thin_windows():
    """Two runs 50 frames apart merge at merge_gap=120; two separate 5-frame flickers that
    merge together still do not survive min_confident even though the merge gap absorbs them:
    [1500:1505] and [1600:1605] merge to (1500, 1605) with length 105 >= min_frames 30,
    but only 10 confident frames < min_confident 15, rejected by min_confident only."""
    mask = np.zeros(2000, dtype=bool)
    mask[100:200] = True
    mask[250:350] = True   # 50-frame gap -> merges with the run above
    mask[1500:1505] = True
    mask[1600:1605] = True  # 95-frame gap <= merge_gap 120, so these two flickers merge
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


def test_tongue_rate_does_not_count_a_leading_run_already_above_cutoff():
    """Opens HIGH: the leading run is not a crossing, so 3 pulses yield 2 crossings, not 3."""
    like = np.array([0.9] * 5 + [0.1] * 5 + [0.9] * 5 + [0.1] * 5 + [0.9] * 5)
    # Rising edges at indices 10 and 20 only -- index 0 has no predecessor to rise from.
    rate = fdw.tongue_upcross_rate(like, 0, 25, pcutoff=0.6, fps=120.0)
    assert rate == pytest.approx(2 / (25 / 120.0))


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


# ------------------------------------------------------------------ rows_for_file (assembly)
def _sipper_polyline_coords(n=6000):
    """Sipper polyline at x=300, y=100..190 (four keypoints, arc length 90 px), all confident."""
    like = np.full(n, 0.95)
    return {
        "sipper_top": {"x": np.full(n, 300.0), "y": np.full(n, 100.0), "likelihood": like},
        "sipper_midtop": {"x": np.full(n, 300.0), "y": np.full(n, 130.0), "likelihood": like},
        "sipper_midbottom": {"x": np.full(n, 300.0), "y": np.full(n, 160.0), "likelihood": like},
        "sipper_bottom": {"x": np.full(n, 300.0), "y": np.full(n, 190.0), "likelihood": like},
    }


def _tongue_pulse(n, start, end, period=15):
    """8 Hz (at 120 fps) square wave over [start, end) that starts LOW, unlike `_square_wave`
    above (which starts high and loses its leading edge): every one of the (end-start)/period
    cycles then contributes a counted rising edge, giving an exact, hand-checkable rate rather
    than one diluted by the "opens already above cutoff" undercount."""
    like = np.full(n, 0.1)
    phase = np.arange(end - start) % period
    like[start:end] = np.where(phase < period // 2, 0.1, 0.9)
    return like


def _session_coords(with_sipper=True, third_bout=False):
    """6000-frame synthetic session built to the final reviewer's spec:

    - sipper polyline at x=300, y=100..190 (arc length 90 px), all confident.
    - a confident nose bout at frames 1000-1600, 5 px off the polyline, tongue pulsing at 8 Hz.
    - a second, equally confident nose bout at frames 3000-3600, at x=5000 (across the cage),
      no tongue activity.
    - with `third_bout`: a third confident, near-sipper bout at 4500-5100 with no tongue rhythm,
      for exercising --require-tongue.
    """
    n = 6000
    nose_like = np.full(n, 0.1)
    nose_x = np.zeros(n)
    nose_y = np.full(n, 140.0)  # within the polyline's y range for every bout
    bouts = [(1000, 1600), (3000, 3600)] + ([(4500, 5100)] if third_bout else [])
    for s, e in bouts:
        nose_like[s:e] = 0.95
    nose_x[1000:1600] = 305.0   # 5 px off the (x=300) polyline
    nose_x[3000:3600] = 5000.0  # across the cage
    if third_bout:
        nose_x[4500:5100] = 305.0  # near, but silent tongue

    coords = {
        "nose": {"x": nose_x, "y": nose_y, "likelihood": nose_like},
        "tongue": {"x": np.zeros(n), "y": np.zeros(n), "likelihood": _tongue_pulse(n, 1000, 1600)},
    }
    if with_sipper:
        coords.update(_sipper_polyline_coords(n))
    return coords


def _rows_args(**overrides):
    """`args` namespace with exactly the attributes `rows_for_file` reads, at CLI defaults."""
    defaults = dict(
        video="dummy.mp4", video_dir=None, fps=120.0,
        bodypart="nose", pcutoff=0.8, sipper_pcutoff=0.6,
        max_nose_dist=0.6, max_nose_dist_px=None,
        require_tongue=False, tongue_pcutoff=0.6, tongue_min_rate=3.0,
        merge_gap=120, min_frames=30, min_confident=15,
        pad=60, max_frames=3600,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_rows_for_file_default_gates_on_proximity(monkeypatch):
    """Hand-verified arithmetic (see final-fix-report.md for the full derivation):
    - mask is True only for frames 1000-1600 (bout 2 is 5000 px away, threshold is 54.0 px).
    - find_windows pre-pad keeps (1000, 1600) whole (600 >= min_frames 30, all confident).
    - padding by 60 each side -> (940, 1660); nothing else to merge with.
    - the near-mask is True on exactly [1000, 1600) within that range, all at distance 5.0 px,
      so mean/min over "mask-True frames only" is 5.0/5.0, not diluted by the quiet padding.
    - the tongue pulse is a clean 40 cycles over 600 frames / 5.0 s = 8.0 crossings/s exactly
      (see `_tongue_pulse`), inherited unchanged from the one overlapping pre-pad window.
    """
    coords = _session_coords()
    monkeypatch.setattr(fdw, "load_dlc_h5", lambda path: ("scorer", list(coords), coords))
    args = _rows_args()

    rows, summary = fdw.rows_for_file(Path("dummy.h5"), args, task_id_start=1)

    assert len(rows) == 1
    row = rows[0]
    assert (row["start_frame"], row["end_frame"]) == (940, 1660)
    assert row["mean_nose_dist"] == 5.0
    assert row["min_nose_dist"] == 5.0
    assert row["tongue_rate"] == 8.0
    assert summary["sipper_scale"] == 90.0
    assert summary["nose_dist_thresh_px"] == 54.0


def test_rows_for_file_max_nose_dist_zero_disables_proximity(monkeypatch):
    """--max-nose-dist 0 must let the far bout back in, with the distance columns left empty
    and no sipper scale reported -- proximity is off, not just failing to exclude anything."""
    coords = _session_coords()
    monkeypatch.setattr(fdw, "load_dlc_h5", lambda path: ("scorer", list(coords), coords))
    args = _rows_args(max_nose_dist=0)

    rows, summary = fdw.rows_for_file(Path("dummy.h5"), args, task_id_start=1)

    assert len(rows) == 2
    far = rows[1]
    assert (far["start_frame"], far["end_frame"]) == (2940, 3660)
    assert far["mean_nose_dist"] == ""
    assert far["min_nose_dist"] == ""
    assert summary["sipper_scale"] is None


def test_rows_for_file_max_nose_dist_zero_never_calls_sipper_anchor(monkeypatch):
    """With proximity disabled, sipper_anchor must never run at all -- so a file with no
    sipper_* keypoints whatsoever still works, and a call would blow up this stub anyway."""
    coords = _session_coords(with_sipper=False)
    monkeypatch.setattr(fdw, "load_dlc_h5", lambda path: ("scorer", list(coords), coords))

    def _boom(*args, **kwargs):
        raise AssertionError("sipper_anchor must not be called when proximity is disabled")
    monkeypatch.setattr(fdw, "sipper_anchor", _boom)

    args = _rows_args(max_nose_dist=0)
    rows, summary = fdw.rows_for_file(Path("dummy.h5"), args, task_id_start=1)

    assert len(rows) == 2
    assert summary["sipper_scale"] is None


def test_rows_for_file_require_tongue_drops_silent_near_bouts(monkeypatch):
    """A third, near-sipper bout with no tongue rhythm must be dropped by --require-tongue, and
    the surviving drinking window's tongue_rate must still be its own parent's rate (8.0), not
    something recomputed over the padded row or contaminated by the dropped bout."""
    coords = _session_coords(third_bout=True)
    monkeypatch.setattr(fdw, "load_dlc_h5", lambda path: ("scorer", list(coords), coords))
    args = _rows_args(require_tongue=True)

    rows, _summary = fdw.rows_for_file(Path("dummy.h5"), args, task_id_start=1)

    assert len(rows) == 1
    row = rows[0]
    assert (row["start_frame"], row["end_frame"]) == (940, 1660)
    assert row["tongue_rate"] == 8.0


# ------------------------------------------------------------------ extract_outliers regression
import extract_outliers  # noqa: E402  (same sys.path entry as find_dlc_windows)


def test_extract_outliers_gate_mask_is_unchanged_by_the_signature_change():
    """gate_mask must still mean 'likelihood >= gate_pcutoff, run through the window pipeline'."""
    n = 2000
    like = np.full(n, 0.1)
    like[100:400] = 0.95
    like[1500:1505] = 0.95
    like[1600:1605] = 0.95  # two 5-frame runs, 95-frame gap <= merge_gap 120; merged window
                             # (1500, 1605) has length 105 >= min_frames 30, but only 10 confident
                             # frames < min_confident 15, so rejected by min_confident only
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
