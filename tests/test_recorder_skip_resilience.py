"""A transient FTDI read failure must never take down a recording session.

The rig occasionally raises "No answer from FTDI" on a USB read. Two layers guard
against losing a run:

  Layer 1 (hardware/mpr121.py): the per-board read retries a few times before
  giving up, so a single hiccup is absorbed silently.

  Layer 2 (recording/recorder.py): if a board's read fails even after retries,
  that board is skipped for that iteration only -- the loop logs and continues.
  Because a skipping board then produces fewer samples than its peers, the HDF5
  write path is driven by per-sensor produced/written counters (not loop_counter),
  so each sensor's dataset stays correctly sized and aligned.

These tests exercise both layers and assert the on-disk data is intact after a
board suffers periodic skips: sensors within a board stay length-matched, the
healthy board is unaffected, and no stale/duplicate samples are injected.
"""
import asyncio
from collections import defaultdict

import h5py
import numpy as np
import pytest

import utils.state
import recording.recorder
from hardware.mpr121 import MPR121Manager
from recording.recorder import SensorRecorder


# ---------------------------------------------------------------------------
# Layer 1: the read itself retries transient failures.
# ---------------------------------------------------------------------------

class _FakePort:
    """I2C port whose first `fail_times` reads raise, then succeed."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.reads = 0

    def read_from(self, register, num_bytes):
        self.reads += 1
        if self.reads <= self.fail_times:
            raise RuntimeError("No answer from FTDI")
        return bytearray(num_bytes)


def test_read_retries_transient_failure(monkeypatch):
    # No real sleep between retries -- keep the test fast.
    monkeypatch.setattr(utils.state, "READ_RETRY_DELAY", 0)
    port = _FakePort(fail_times=utils.state.READ_RETRIES - 1)  # fails, then succeeds
    mgr = MPR121Manager({"BOARD": {"port": port}})

    time_data, cap_data, sn = mgr.read_sensor_data("BOARD")

    assert sn == "BOARD"
    assert port.reads == utils.state.READ_RETRIES  # retried up to the last attempt
    assert len(cap_data) == utils.state.NUM_CHANNELS


def test_read_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(utils.state, "READ_RETRY_DELAY", 0)
    port = _FakePort(fail_times=utils.state.READ_RETRIES + 5)  # always fails
    mgr = MPR121Manager({"BOARD": {"port": port}})

    with pytest.raises(Exception):
        mgr.read_sensor_data("BOARD")

    # Gave up after exactly READ_RETRIES attempts -- no infinite loop.
    assert port.reads == utils.state.READ_RETRIES


# ---------------------------------------------------------------------------
# Layer 2: a board that keeps failing is skipped, the run survives, and the
# HDF5 file stays correct.
# ---------------------------------------------------------------------------

_TEST_MAP = {"BOARD_OK": [1, 2, 3], "BOARD_FLAKY": [4, 5, 6]}


class _FlakyManager:
    """Stand-in for MPR121Manager (post-retry outcome).

    BOARD_FLAKY raises on every `skip_period`-th call -- i.e. it simulates a read
    that already exhausted Layer-1 retries. cap_data is a per-board monotonic
    counter that increments ONLY on a successful read, so any duplicate or
    mis-ordered sample injected by a buggy write path is detectable on disk.
    """

    def __init__(self, flaky_board, skip_period):
        self.flaky_board = flaky_board
        self.skip_period = skip_period
        self.calls = defaultdict(int)
        self.successes = defaultdict(int)

    def read_sensor_data(self, sn):
        self.calls[sn] += 1
        if sn == self.flaky_board and self.calls[sn] % self.skip_period == 0:
            raise RuntimeError("No answer from FTDI")
        self.successes[sn] += 1
        v = self.successes[sn]
        n = len(_TEST_MAP[sn])
        return [1000.0 + v] * n, [v] * n, sn


def test_recording_survives_periodic_board_skips(tmp_path, monkeypatch):
    # Deterministic 2-board layout; small flush window so we exercise several
    # flushes; fast pacing so the run finishes quickly.
    monkeypatch.setattr(utils.state, "SERIAL_NUMBER_SENSOR_MAP", _TEST_MAP)
    monkeypatch.setattr(recording.recorder, "HISTORY_SIZE", 10)
    monkeypatch.setattr(recording.recorder, "MAX_SAMPLE_HZ", 500)

    mgr = _FlakyManager(flaky_board="BOARD_FLAKY", skip_period=5)
    controllers = {"BOARD_OK": object(), "BOARD_FLAKY": object()}
    rec = SensorRecorder(mpr121_manager=mgr,
                         filename=str(tmp_path / "raw.h5"),
                         controllers=controllers)

    logs = []

    async def run():
        task = asyncio.create_task(rec.record_sensors(log_callback=logs.append))
        await asyncio.sleep(0.6)   # hundreds of loops -> many flushes + skips
        rec.stop()
        # If a skip had propagated, awaiting the task would re-raise here.
        await task

    asyncio.run(run())

    # The run survived and actually exercised skips.
    assert rec._read_error_count > 0
    assert any("read failed on board BOARD_FLAKY" in m for m in logs)

    # Only the flaky board ever failed, so every skip is one fewer sample it
    # produced relative to the healthy board.
    ok_produced = rec.board_produced["BOARD_OK"][1]
    flaky_produced = rec.board_produced["BOARD_FLAKY"][4]
    assert ok_produced - flaky_produced == rec._read_error_count

    with h5py.File(rec.filename, "r") as h5f:
        def lengths(board, sensors):
            return [h5f[f"board_{board}/sensor_{s}/cap_data"].shape[0] for s in sensors]

        ok_lens = lengths("BOARD_OK", _TEST_MAP["BOARD_OK"])
        flaky_lens = lengths("BOARD_FLAKY", _TEST_MAP["BOARD_FLAKY"])

        # Sensors on the same board are read atomically -> identical lengths.
        assert len(set(ok_lens)) == 1, ok_lens
        assert len(set(flaky_lens)) == 1, flaky_lens

        # The healthy board was not starved by the flaky board's skips: it has
        # strictly more samples on disk. (The gap equals the number of skips
        # that occurred before the last flush -- skips landing in the final
        # unflushed tail aren't on disk yet -- so it is positive and bounded by
        # the total skip count. The exact per-skip accounting is verified above
        # on the produced counters.)
        assert 0 < ok_lens[0] - flaky_lens[0] <= rec._read_error_count

        for board, sensors in _TEST_MAP.items():
            for s in sensors:
                grp = h5f[f"board_{board}/sensor_{s}"]
                cap = grp["cap_data"][:]
                # time and cap stay paired.
                assert grp["time_data"].shape[0] == cap.shape[0]
                # No stale/duplicate injection and no reordering: cap is the
                # success counter, stored contiguously, so it must step by 1.
                assert cap.shape[0] >= 2
                assert np.all(np.diff(cap) == 1), (board, s, cap[:20])
                # Each flushed dataset matches its written counter (allow the
                # one well-known first-flush ring drop of a single sample).
                written = rec.board_written[board][s]
                assert 0 <= written - cap.shape[0] <= 1
