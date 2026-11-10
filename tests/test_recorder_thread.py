"""MPR121 sampling must not depend on the UI event loop or the disk.

The recorder used to run as an asyncio task on Solara's event loop, so any slow
UI work (a render, a button handler) or HDF5 flush delayed the next read and
every board silently lost samples. It also exited without a final flush, so a
session Stop discarded up to HISTORY_SIZE unflushed samples per channel.

These tests pin the threaded recorder (spec:
docs/superpowers/specs/2026-11-10-recorder-thread-design.md): sampling keeps
its cadence while the event loop or the h5 lock is blocked, every sample read
is on disk after stop(), write failures are retried loudly without loss, and
stop() never hangs.
"""
import asyncio
import threading
import time
from collections import defaultdict

import h5py
import numpy as np
import pytest

import recording.recorder
import utils.state
from recording.recorder import SensorRecorder

_TEST_MAP = {"B0": [1, 2, 3], "B1": [4, 5, 6]}
_CONTROLLERS = {"B0": object(), "B1": object()}


class _CounterManager:
    """Fake MPR121Manager: per-board counter values with real wall-clock stamps.

    cap_data counts successful reads per board (1, 2, 3, ...), so any dropped,
    duplicated or reordered sample is visible on disk. ``bad_call`` makes that
    (global) call return a malformed read (wrong channel count), which breaks
    sampling outside the per-board read error handling (a fatal error).
    """

    def __init__(self, bad_call=None):
        self.n = defaultdict(int)
        self.calls = 0
        self.bad_call = bad_call
        self._lock = threading.Lock()

    def read_sensor_data(self, sn):
        with self._lock:
            self.calls += 1
            if self.bad_call is not None and self.calls >= self.bad_call:
                return [time.time()], [0], sn  # malformed: 1 channel, not 3
            self.n[sn] += 1
            v = self.n[sn]
        k = len(_TEST_MAP[sn])
        return [time.time()] * k, [v] * k, sn


@pytest.fixture
def make_rec(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.state, "SERIAL_NUMBER_SENSOR_MAP", _TEST_MAP)

    def _make(history=50, hz=500, manager=None):
        monkeypatch.setattr(recording.recorder, "HISTORY_SIZE", history)
        monkeypatch.setattr(recording.recorder, "MAX_SAMPLE_HZ", hz)
        return SensorRecorder(mpr121_manager=manager or _CounterManager(),
                              filename=str(tmp_path / "raw.h5"),
                              controllers=dict(_CONTROLLERS))
    return _make


def _on_disk(rec):
    out = {}
    with h5py.File(rec.filename, "r") as h5f:
        for sn, sensors in _TEST_MAP.items():
            for s in sensors:
                g = h5f[f"board_{sn}/sensor_{s}"]
                out[(sn, s)] = (g["time_data"][:], g["cap_data"][:])
    return out


def _assert_complete(rec):
    for (sn, s), (t, c) in _on_disk(rec).items():
        assert len(c) == rec.board_produced[sn][s] > 0, (sn, s)
        assert np.array_equal(c, np.arange(1, len(c) + 1)), (sn, s)
        assert len(t) == len(c)


def _max_gap(rec):
    return max(np.diff(t).max() for t, _ in _on_disk(rec).values())


def test_stop_flushes_every_sample(make_rec):
    rec = make_rec(history=50, hz=500)
    rec.start()
    time.sleep(0.25)
    assert rec.stop() is True
    _assert_complete(rec)


def test_stop_before_first_flush_writes_data(make_rec):
    rec = make_rec(history=100000, hz=500)
    rec.start()
    time.sleep(0.1)
    assert rec.stop() is True
    _assert_complete(rec)


def test_ui_loop_stall_does_not_gap_samples(make_rec):
    rec = make_rec(history=50, hz=150)

    async def run():
        rec.start()
        await asyncio.sleep(0.1)
        time.sleep(0.3)  # a slow synchronous UI handler blocks the loop
        await asyncio.sleep(0.1)
        rec.stop()

    asyncio.run(run())
    _assert_complete(rec)
    assert _max_gap(rec) < 4 / 150


def test_h5_lock_held_does_not_gap_samples(make_rec):
    rec = make_rec(history=10, hz=150)
    rec.start()
    time.sleep(0.05)

    def hold():
        with rec._h5_lock:  # e.g. a GUI bookmark/metadata write in progress
            time.sleep(0.5)

    t = threading.Thread(target=hold)
    t.start()
    t.join()
    time.sleep(0.1)
    assert rec.stop() is True
    _assert_complete(rec)
    assert _max_gap(rec) < 4 / 150


def test_write_failure_retries_without_loss(make_rec):
    rec = make_rec(history=10, hz=500)
    real = rec._write_batch
    fails = {"left": 2}

    def flaky(batch):
        if fails["left"] > 0:
            fails["left"] -= 1
            raise OSError("disk")
        real(batch)

    rec._write_batch = flaky
    logs = []
    rec.start(log_callback=logs.append)
    time.sleep(0.2)
    assert rec.stop() is True
    assert fails["left"] == 0
    assert any(m.startswith("ERROR:") for m in logs)
    _assert_complete(rec)


def test_stop_returns_when_writer_stuck(make_rec):
    rec = make_rec(history=10, hz=500)

    def broken(batch):
        raise OSError("disk full")

    rec._write_batch = broken
    logs = []
    rec.start(log_callback=logs.append)
    time.sleep(0.1)
    t0 = time.monotonic()
    ok = rec.stop(timeout=1.0)
    assert ok is False
    assert time.monotonic() - t0 < 3.0
    assert any(m.startswith("ERROR:") for m in logs)


def test_stop_is_idempotent_and_safe_unstarted(make_rec, tmp_path):
    unstarted = SensorRecorder(mpr121_manager=None,
                               filename=str(tmp_path / "never.h5"),
                               controllers=dict(_CONTROLLERS))
    assert unstarted.stop() is True

    rec = make_rec(history=50, hz=500)
    rec.start()
    time.sleep(0.1)
    assert rec.stop() is True
    first = {k: len(c) for k, (_, c) in _on_disk(rec).items()}
    assert rec.stop() is True
    assert {k: len(c) for k, (_, c) in _on_disk(rec).items()} == first


def test_recent_samples_consistent_while_running(make_rec):
    rec = make_rec(history=50, hz=500)
    rec.start()
    results = []
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        results.append(rec.get_recent_samples(1))
    rec.stop()
    assert any(c is not None for c, _ in results)
    assert all(c is None or len(c) == len(t) for c, t in results)


def test_sampler_fatal_error_reports_and_keeps_data(make_rec):
    rec = make_rec(history=50, hz=500, manager=_CounterManager(bad_call=40))
    errors, logs = [], []
    rec.start(log_callback=logs.append, on_error=errors.append)
    time.sleep(0.3)
    assert len(errors) == 1
    assert rec.stop() is True
    _assert_complete(rec)


def test_session_start_stop_uses_recorder_threads(tmp_path, monkeypatch):
    """The GUI's Start/Stop drive the recorder (hardware process + writer
    thread, no asyncio task) and a session Stop leaves every sample read on
    disk."""
    import pandas as pd
    import components.hardware_status as hw
    import components.session_controls as sc
    from utils import state

    monkeypatch.setattr(utils.state, "SERIAL_NUMBER_SENSOR_MAP", _TEST_MAP)
    monkeypatch.setattr(recording.recorder, "HISTORY_SIZE", 50)
    monkeypatch.setattr(hw, "BACKEND_PATH", "hardware.fake_backends:CounterBackend")
    monkeypatch.setattr(hw, "hw_client", None)
    monkeypatch.setattr(utils.state, "ACTIVE_CHANNELS", [1, 6, 11])
    monkeypatch.setattr(sc, "current_recorder", None)
    state.set_session("i2c_controllers", dict(_CONTROLLERS))
    state.layout_df.set(pd.DataFrame({"animal": ["A1"]}, index=[1]))
    state.output_directory.set(str(tmp_path))
    state.camera_enabled.set(False)
    state.set_session("recording_all", False)
    try:
        hw.initialize_hardware()
        sc.start_recording()
        rec = sc.current_recorder
        assert rec is not None and rec._source is not None
        time.sleep(0.2)
        sc.stop_recording()
        assert not rec._source._receiver.is_alive()
        assert not rec._writer.is_alive()
        _assert_complete(rec)
    finally:
        hw.disconnect_hardware()
        state.layout_df.set(pd.DataFrame())
        state.set_session("recording_all", False)


def test_fatal_error_ends_writer_without_stop(make_rec):
    """After a fatal sampler error the GUI clears recording_all and may never
    call stop(). The writer must still exit, or its periodic volume/weight
    persist would later write the NEXT session's values into this old file."""
    rec = make_rec(history=50, hz=500, manager=_CounterManager(bad_call=40))
    errors = []
    rec.start(on_error=errors.append)
    deadline = time.monotonic() + 3.0
    while rec._writer.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(errors) == 1
    assert not rec._writer.is_alive()
    _assert_complete(rec)


def test_session_stop_completes_when_metadata_write_fails(tmp_path, monkeypatch):
    """A failing disk must not make Stop impossible: the metadata write error is
    logged and the recorder is still stopped."""
    import pandas as pd
    from dataclasses import replace
    import components.hardware_status as hw
    import components.session_controls as sc
    from utils import state

    monkeypatch.setattr(utils.state, "SERIAL_NUMBER_SENSOR_MAP", _TEST_MAP)
    monkeypatch.setattr(recording.recorder, "HISTORY_SIZE", 50)
    monkeypatch.setattr(hw, "BACKEND_PATH", "hardware.fake_backends:CounterBackend")
    monkeypatch.setattr(hw, "hw_client", None)
    monkeypatch.setattr(utils.state, "ACTIVE_CHANNELS", [1, 6, 11])
    monkeypatch.setattr(sc, "current_recorder", None)
    state.set_session("i2c_controllers", dict(_CONTROLLERS))
    state.layout_df.set(pd.DataFrame({"animal": ["A1"]}, index=[1]))
    state.output_directory.set(str(tmp_path))
    state.camera_enabled.set(False)
    state.set_session("recording_all", False)
    try:
        hw.initialize_hardware()
        sc.start_recording()
        rec = sc.current_recorder
        sensors = state.session["sensor_states"].copy()
        sensors[1] = replace(sensors[1], is_recording=True, start_time=1.0)
        state.set_session("sensor_states", sensors)

        def broken(*a, **k):
            raise OSError("disk unwritable")

        monkeypatch.setattr(rec, "write_sensor_metadata", broken)
        time.sleep(0.1)
        sc.stop_recording()
        assert not rec._source._receiver.is_alive()
        assert state.session["recording_all"] is False
        assert any("ERROR" in m and "disk unwritable" in m
                   for m in state.log_messages.value)
    finally:
        hw.disconnect_hardware()
        state.layout_df.set(pd.DataFrame())
        state.set_session("recording_all", False)
