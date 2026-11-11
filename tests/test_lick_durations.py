"""Per-lick duration (half-depth width) measurement.

A lick pulls the sipper capacitance down into a roughly square dip. Its
duration is the time the trace spends below half the dip depth, measured
from linearly interpolated crossings and converted to seconds through
time_data (sampling is non-uniform). Licks whose span contains a sampling
gap, or that are longer than a lick can be, get NaN so the output stays
aligned with lick_indices.
"""
import h5py
import numpy as np
import pytest

from data_analysis import compute_bout_structure, compute_lick_durations, save_filtered_data


def _square_dips(n=2000, fs=100.0, base=700, depth=120, width=4, period=14, start=100, n_licks=10):
    """Uniformly sampled trace of `n_licks` square dips `width` samples wide."""
    trace = np.full(n, base, dtype=np.int64)
    centers = []
    for k in range(n_licks):
        s = start + k * period
        trace[s:s + width] = base - depth
        centers.append(s + width // 2)
    return trace, np.arange(n) / fs, np.array(centers)


def test_square_dip_duration_and_depth():
    trace, t, li = _square_dips(width=4, depth=120)
    out = compute_lick_durations(trace, t, li)
    # Square dip 4 samples wide: half-depth crossings sit half a sample outside
    # the flat bottom on each side -> width = 4 samples = 40 ms at 100 Hz.
    np.testing.assert_allclose(out['lick_durations'], 0.04, atol=1e-9)
    np.testing.assert_allclose(out['lick_depths'], 120)


def test_duration_independent_of_depth():
    a = compute_lick_durations(*_square_dips(depth=30))['lick_durations']
    b = compute_lick_durations(*_square_dips(depth=300))['lick_durations']
    np.testing.assert_allclose(a, b)


def test_uses_time_data_not_sample_count():
    trace, t, li = _square_dips(width=4)
    t = t * 2  # same samples, half the rate -> twice as long
    np.testing.assert_allclose(compute_lick_durations(trace, t, li)['lick_durations'], 0.08)


def test_far_outlier_does_not_set_baseline():
    trace, t, li = _square_dips()
    trace[1500] = 100_000  # huge spike far from any lick
    np.testing.assert_allclose(compute_lick_durations(trace, t, li)['lick_durations'], 0.04)
    np.testing.assert_allclose(compute_lick_durations(trace, t, li)['lick_depths'], 120)


def test_gap_inside_dip_is_nan():
    trace, t, li = _square_dips()
    t = t.copy()
    t[li[3]:] += 0.2  # 200 ms stall inside lick 3
    d = compute_lick_durations(trace, t, li)['lick_durations']
    assert np.isnan(d[3])
    assert np.isfinite(np.delete(d, 3)).all()


def test_too_long_dip_is_nan():
    trace, t, li = _square_dips(width=20, period=40)  # 200 ms > 1/6 s
    assert np.isnan(compute_lick_durations(trace, t, li)['lick_durations']).all()


def test_empty_and_order_preserved():
    trace, t, li = _square_dips()
    out = compute_lick_durations(trace, t, np.array([], dtype=int))
    assert out['lick_durations'].shape == (0,) and out['lick_depths'].shape == (0,)
    rev = compute_lick_durations(trace, t, li[::-1])
    np.testing.assert_allclose(rev['lick_durations'], 0.04)
    assert rev['lick_durations'].shape == li.shape


def test_bout_median_lick_durations():
    lick_times = np.array([0.0, 0.15, 0.3, 5.0, 5.15, 5.3, 5.45, 20.0])
    durs = np.array([0.03, 0.04, np.nan, 0.05, 0.05, 0.06, 0.07, 0.01])
    m = compute_bout_structure(lick_times, ibi_threshold=1., min_licks=2, lick_durations=durs)
    np.testing.assert_allclose(m['bout_median_lick_durations'], [0.035, 0.055])
    # Without durations the key is absent and nothing else changes.
    assert 'bout_median_lick_durations' not in compute_bout_structure(lick_times, 1., 2)


def test_save_filtered_data_writes_durations(tmp_path):
    trace, t, li = _square_dips()
    data = {'cap_data': trace, 'time_data': t, 'lick_indices': li, 'lick_times': t[li],
            'used_start_idx': 0, 'used_stop_idx': len(t) - 1, 'consumed_vol': 0.3,
            'weight': 25.0}
    with h5py.File(tmp_path / "f.h5", 'w') as f:
        assert save_filtered_data(data, "A-1", f, str(tmp_path / "log.txt")) is False
        g = f["A-1"]
        np.testing.assert_allclose(g['lick_durations'][()], 0.04)
        np.testing.assert_allclose(g['lick_depths'][()], 120)
        np.testing.assert_allclose(g['bout_median_lick_durations'][()], [0.04])


def test_onsets_offsets_bracket_detected_time():
    trace, t, li = _square_dips(width=4)
    out = compute_lick_durations(trace, t, li)
    # Square dip starting at sample s: half-depth crossing is half a sample earlier.
    starts = np.array([100 + k * 14 for k in range(10)])
    np.testing.assert_allclose(out['lick_onsets'], (starts - 0.5) / 100)
    np.testing.assert_allclose(out['lick_offsets'] - out['lick_onsets'], out['lick_durations'])
    assert (out['lick_onsets'] < t[li]).all() and (out['lick_offsets'] > t[li]).all()


def test_onsets_kept_when_duration_rejected():
    trace, t, li = _square_dips(width=20, period=40)  # too long -> NaN duration
    out = compute_lick_durations(trace, t, li)
    assert np.isnan(out['lick_durations']).all()
    assert np.isfinite(out['lick_onsets']).all() and np.isfinite(out['lick_offsets']).all()
