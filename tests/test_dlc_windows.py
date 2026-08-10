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
