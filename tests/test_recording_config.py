"""Sampling roughly doubled (~112 Hz vs ~46 Hz), so the flush buffer grows to
keep flush cadence near the old ~4.5 s and HDF5 chunks a healthy size. The
measurement-persist cadence is wall-clock seconds, independent of buffer size."""
from utils import state


def test_history_size_bumped_for_doubled_rate():
    assert state.HISTORY_SIZE == 500


def test_measurement_persist_seconds_default():
    assert state.MEASUREMENT_PERSIST_SECONDS == 300
