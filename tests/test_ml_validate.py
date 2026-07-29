"""Unit tests for the lick-time comparison logic in ml_detection.validate.

Only the pure `compare_lick_times` matcher is unit-tested here; the cascade runners
(`run_python_cascade`, `run_matlab_cascade`, `validate_recording`) require real weights, MATLAB,
and recordings and are exercised by hand during validation gate #2.
"""
import numpy as np

from ml_detection.validate import compare_lick_times


def test_identical_lists_match_perfectly():
    # Two identical lick-time lists should match one-to-one with perfect precision/recall.
    times = np.array([1.0, 2.0, 3.5, 10.0])
    report = compare_lick_times(times, times, tol_s=0.02)
    assert report["n_matched"] == 4
    assert report["n_missed"] == 0
    assert report["n_extra"] == 0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["max_offset_s"] == 0.0


def test_within_tolerance_counts_as_match_and_reports_offset():
    # A 10 ms difference is within the default 20 ms tolerance -> matched, offset reported.
    matlab = np.array([1.0, 2.0])
    python = np.array([1.01, 1.99])  # 10 ms late, 10 ms early
    report = compare_lick_times(python, matlab, tol_s=0.02)
    assert report["n_matched"] == 2
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["max_offset_s"] <= 0.0100001
    assert report["mean_offset_s"] <= 0.0100001


def test_missed_and_extra_are_counted():
    # MATLAB has a lick at 5.0 that Python misses (false negative); Python has an extra at 9.0
    # (false positive). The lick at 1.0 matches.
    matlab = np.array([1.0, 5.0])
    python = np.array([1.0, 9.0])
    report = compare_lick_times(python, matlab, tol_s=0.02)
    assert report["n_matched"] == 1
    assert report["n_missed"] == 1      # MATLAB's 5.0 unmatched
    assert report["n_extra"] == 1       # Python's 9.0 unmatched
    assert report["precision"] == 0.5   # 1 of 2 Python licks real
    assert report["recall"] == 0.5      # 1 of 2 MATLAB licks recovered


def test_empty_python_yields_all_missed():
    matlab = np.array([1.0, 2.0, 3.0])
    python = np.array([])
    report = compare_lick_times(python, matlab, tol_s=0.02)
    assert report["n_matched"] == 0
    assert report["n_missed"] == 3
    assert report["recall"] == 0.0
    # precision is defined as 0.0 when there are no Python licks (no false-positive rate to report)
    assert report["precision"] == 0.0


def test_greedy_matching_is_one_to_one():
    # Two MATLAB licks close together, only one Python lick between them: exactly one should match,
    # the other counts as missed (no double-claiming of the single Python lick).
    matlab = np.array([1.00, 1.01])
    python = np.array([1.005])
    report = compare_lick_times(python, matlab, tol_s=0.02)
    assert report["n_matched"] == 1
    assert report["n_missed"] == 1
    assert report["n_extra"] == 0
