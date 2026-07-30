"""Unit tests for the pure logic in scripts/find_interesting_windows.py.

The HDF5 reading, provenance resolution, and shell-script writing are exercised by hand on real
recordings; only the deterministic, self-contained helpers (variance, masking, selection, window
construction, command formatting) are unit-tested here.
"""
import os
import sys

import numpy as np

# Make the repository root importable so `scripts.find_interesting_windows` resolves regardless of
# where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.find_interesting_windows import (  # noqa: E402
    sliding_variance, center_sample_indices, mask_bout_windows, select_climbing_centers,
    clip_window, count_licks_in_window, build_rois_for_cycle, parse_offsets, build_command,
    is_control,
)


def test_sliding_variance_matches_numpy_var():
    # For a small signal, every sliding window's variance must equal numpy's population variance
    # of that same slice. This checks the prefix-sum formula against a trusted reference.
    signal = np.array([1.0, 3.0, 2.0, 8.0, 4.0, 4.0, 4.0])
    window = 3
    result = sliding_variance(signal, window)
    expected = np.array([np.var(signal[i:i + window]) for i in range(len(signal) - window + 1)])
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_sliding_variance_constant_signal_is_zero():
    # A flat signal has zero variance in every window; round-off must not make it negative.
    signal = np.full(50, 5.0)
    result = sliding_variance(signal, 10)
    assert result.shape[0] == 50 - 10 + 1
    assert np.all(result >= 0.0)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_sliding_variance_signal_shorter_than_window_is_empty():
    assert sliding_variance(np.array([1.0, 2.0]), 5).size == 0


def test_center_sample_indices():
    # Window width 3 -> each window's center is offset by 1 (== 3 // 2) from its start index.
    np.testing.assert_array_equal(center_sample_indices(4, 3), np.array([1, 2, 3, 4]))


def test_mask_bout_windows_sets_neg_inf_near_bouts():
    # Centers every 1 s from 0..9 s. A single bout at t = 5 s (duration 0) with a 2 s guard must
    # disqualify the windows centered at 3, 4, 5, 6, 7 s (within 2 s of the bout), leaving the rest.
    center_times = np.arange(10, dtype=float)
    variance = np.ones(10)
    mask_bout_windows(variance, center_times, [5.0], [0.0], guard_seconds=2.0)
    disqualified = ~np.isfinite(variance)
    np.testing.assert_array_equal(np.where(disqualified)[0], np.array([3, 4, 5, 6, 7]))
    # Windows outside the guard keep their original value.
    assert variance[0] == 1.0 and variance[9] == 1.0


def test_select_climbing_centers_respects_separation_and_order():
    # Variance peaks at indices 2 (highest) and 3 (second), which are only 1 s apart. With a 5 s
    # minimum separation, the greedy picker must take index 2 and then SKIP index 3, taking the
    # next far-enough, lower peak (index 8).
    center_times = np.arange(10, dtype=float)
    variance = np.array([0, 0, 10, 9, 0, 0, 0, 0, 5, 0], dtype=float)
    picks = select_climbing_centers(variance, center_times, n_wanted=2, min_separation_s=5.0)
    picked_centers = [c for c, _ in picks]
    assert picked_centers == [2.0, 8.0]
    # Scores are reported alongside, strongest first.
    assert picks[0][1] == 10.0 and picks[1][1] == 5.0


def test_select_climbing_centers_stops_at_masked():
    # Only two finite windows exist; the rest are -inf (masked). Even asking for 4, we get 2.
    center_times = np.arange(5, dtype=float)
    variance = np.array([3.0, -np.inf, -np.inf, 1.0, -np.inf])
    picks = select_climbing_centers(variance, center_times, n_wanted=4, min_separation_s=1.0)
    assert [c for c, _ in picks] == [0.0, 3.0]


def test_clip_window_clamps_to_recording_bounds():
    # A centered window is kept full width in the interior...
    assert clip_window(50.0, 12.0, span_s=100.0) == (44.0, 56.0)
    # ...but is clipped (not shifted) at the start and end edges.
    assert clip_window(2.0, 12.0, span_s=100.0) == (0.0, 8.0)
    assert clip_window(99.0, 12.0, span_s=100.0) == (93.0, 100.0)


def test_count_licks_in_window_is_inclusive():
    licks = np.array([1.0, 5.0, 5.5, 9.0])
    assert count_licks_in_window(licks, 5.0, 6.0) == 2   # 5.0 and 5.5
    assert count_licks_in_window(np.array([]), 0.0, 10.0) == 0


def test_build_rois_separates_licking_and_climbing():
    # Synthetic 60 s / 100 Hz trace. A slow, large "climbing" excursion sits at ~10 s (a smooth
    # bump, no licks). A detected bout sits at ~40 s. The builder must return the bump as a
    # lick-free 'climb' window and the bout as a 'lick' window, with no overlap between them.
    fs = 100.0
    t = np.arange(0, 60, 1.0 / fs)
    cap = np.full(t.shape, 676.0)                     # flat baseline (median capacitance)
    # Climbing bump: a wide Gaussian centered at 10 s -> high local variance, slow, no licks.
    cap += 120.0 * np.exp(-((t - 10.0) ** 2) / (2 * 1.5 ** 2))

    bout_start_times = np.array([40.0])
    bout_durations = np.array([0.5])
    bout_lick_counts = np.array([7])
    lick_times = np.linspace(40.0, 40.5, 7)           # the 7 licks of that bout

    params = {"n_lick": 3, "n_climb": 3, "roi_seconds": 6.0, "var_window": 1.0, "min_var": 0.0}
    rois = build_rois_for_cycle(cap, t, bout_start_times, bout_durations,
                                bout_lick_counts, lick_times, params)

    licks = [r for r in rois if r["category"] == "lick"]
    climbs = [r for r in rois if r["category"] == "climb"]
    assert len(licks) == 1                            # only one bout exists
    assert licks[0]["center"] == 40.25               # bout_start + duration/2
    assert len(climbs) >= 1
    # The strongest climbing window sits on the 10 s bump and contains no licks.
    top_climb = climbs[0]
    assert 7.0 <= top_climb["center"] <= 13.0
    assert top_climb["n_licks_in_window"] == 0
    # Licking and climbing windows do not overlap.
    for climb in climbs:
        assert climb["end"] <= licks[0]["start"] or climb["start"] >= licks[0]["end"]


def test_parse_offsets():
    assert parse_offsets(["0=280", "3=-1.5"]) == {0: 280.0, 3: -1.5}
    assert parse_offsets(None) == {}


def test_build_command_restart_warns_without_offset():
    # A restart cycle with no offset must emit a WARNING block and the command with raw start/end.
    row = {"animal": "A1", "cycle": 0, "category": "lick", "rank": 0,
           "start": 100.0, "end": 112.0, "restart": True,
           "raw_h5": "data/raw_07-22.h5", "layout": "data/layout.csv"}
    lines = build_command(row, out_dir="clips", offsets={}, combined_h5="results_combined.h5")
    text = "\n".join(lines)
    assert "WARNING" in text
    assert "--start 100.000 --end 112.000" in text
    assert "clips/A1_c0_lick0.mp4" in text
    # The command reads the trace from the combined file (fast path), not by re-running filter_data.
    assert "--combined-h5 results_combined.h5 --cycle 0" in text


def test_build_command_offset_shifts_and_silences_warning():
    # Supplying an offset for the restart cycle shifts start/end and suppresses the warning
    # (the user has taken responsibility for the alignment).
    row = {"animal": "A1", "cycle": 0, "category": "lick", "rank": 0,
           "start": 100.0, "end": 112.0, "restart": True,
           "raw_h5": "data/raw_07-22.h5", "layout": "data/layout.csv"}
    lines = build_command(row, out_dir="clips", offsets={0: 280.0}, combined_h5="results_combined.h5")
    text = "\n".join(lines)
    assert "WARNING" not in text
    assert "--start 380.000 --end 392.000" in text


def test_is_control():
    assert is_control("Control1") and is_control("Control12")
    assert not is_control("ACG-26-3-5")
