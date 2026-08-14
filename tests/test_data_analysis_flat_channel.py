"""Channels with too few discrete capacitance levels must still be saved.

A control cage has a sipper but no animal, so its trace is nearly flat and can
end up with only a handful of unique capacitance values. The threshold scan in
basic_algorithm cannot run on such a channel (it needs more levels than its
depth requirement), but "no licks detectable" is a zero-lick result, not a
missing channel -- the evaporation volume still has to reach the combined file.
"""

import numpy as np
import h5py
import pytest

from data_analysis import basic_algorithm


def _channel(unique_levels, n_samples=5000, fs=250.0):
    """Build one animal's data dict whose trace has exactly `unique_levels` values."""
    rng = np.random.default_rng(0)
    base = 600
    trace = base + rng.integers(0, unique_levels, size=n_samples)
    # Guarantee every level actually appears
    trace[:unique_levels] = base + np.arange(unique_levels)
    return {
        'cap_data': trace.astype(np.int64),
        'time_data': np.arange(n_samples) / fs,
        'fs': fs,
        'consumed_vol': 0.35,
        'used_start_idx': 0,
        'used_stop_idx': n_samples - 1,
    }


@pytest.fixture
def h5f():
    f = h5py.File('mem', driver='core', backing_store=False, mode='w')
    yield f
    f.close()


# 4 levels clears the old `len(unique_vals) > 3` gate but leaves 3 thresholds,
# far fewer than the 20-level depth requirement; 2 levels fails that gate too.
@pytest.mark.parametrize('unique_levels', [2, 4, 8, 21])
def test_flat_control_channel_is_saved_with_zero_licks(h5f, tmp_path, unique_levels):
    logfile = tmp_path / 'run.log'
    data_by_animal = {'Control1': _channel(unique_levels)}

    missing = basic_algorithm(data_by_animal, h5f, str(logfile))

    assert missing is False
    assert 'Control1' in h5f, (
        f"channel with {unique_levels} capacitance levels was dropped entirely"
    )
    grp = h5f['Control1']
    assert grp['lick_indices'][()].size == 0
    assert grp['lick_times'][()].size == 0
    assert grp['consumed_vol'][()] == pytest.approx(0.35)


def test_normal_channel_still_detects_licks(h5f, tmp_path):
    """The fix must not change channels that have enough levels to scan."""
    fs = 250.0
    n = 5000
    rng = np.random.default_rng(0)
    trace = (760 - rng.integers(0, 40, size=n)).astype(np.int64)  # noisy baseline
    # 6 Hz dips, each 40 ms long, well below the baseline spread
    for start in range(500, 3000, int(fs / 6)):
        trace[start:start + 10] = 660

    data_by_animal = {'ACG-1': dict(_channel(2), cap_data=trace,
                                    time_data=np.arange(n) / fs, weight=25.0)}

    missing = basic_algorithm(data_by_animal, h5f, str(tmp_path / 'run.log'))

    assert missing is False
    assert h5f['ACG-1']['lick_indices'][()].size > 0
