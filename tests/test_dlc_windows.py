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
    points, arc = fdw.sipper_anchor(coords, pcutoff=0.6, min_frames=100)
    np.testing.assert_allclose(points, [(100.0, 100.0), (100.0, 130.0),
                                        (100.0, 160.0), (100.0, 190.0)])
    assert arc == pytest.approx(90.0)


def test_sipper_anchor_drops_keypoints_with_too_few_confident_frames():
    coords = _sipper_coords(
        {"sipper_top": (100.0, 100.0), "sipper_midtop": (100.0, 130.0),
         "sipper_midbottom": (100.0, 160.0), "sipper_bottom": (100.0, 190.0)},
        n=500, confident={"sipper_top": 5, "sipper_midtop": 200,
                          "sipper_midbottom": 200, "sipper_bottom": 200},
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
