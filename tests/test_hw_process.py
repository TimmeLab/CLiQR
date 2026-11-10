"""Sampling in the hardware process (its own GIL).

On the rig, every Solara re-render held the GUI process's GIL long enough to
stretch one pyftdi read (~20 GIL-releasing USB transfers) across the render:
~125 ms gap on every board per UI action. These tests drive the real spawned
hardware process with fake backends (hardware/fake_backends.py) and pin:
complete data across the process boundary, no gaps while the GUI process is
CPU-busy, loud handling of a child crash, and no orphaned child.
"""
import threading
import time

import h5py
import numpy as np
import pytest

import recording.recorder
import utils.state
from hardware.fake_backends import DEFAULT_COUNTER_MAP, PYFTDI_LIKE_MAP
from hardware.hw_client import HardwareClient
from recording.recorder import SensorRecorder

COUNTER = "hardware.fake_backends:CounterBackend"


@pytest.fixture
def clients():
    made = []

    def make(backend=COUNTER):
        c = HardwareClient(backend)
        made.append(c)
        return c
    yield make
    for c in made:
        c.shutdown()


def _rec(tmp_path, monkeypatch, client, board_map=DEFAULT_COUNTER_MAP,
         channels=(1, 6, 11), history=50, hz=150):
    monkeypatch.setattr(utils.state, "SERIAL_NUMBER_SENSOR_MAP", board_map)
    monkeypatch.setattr(recording.recorder, "HISTORY_SIZE", history)
    return SensorRecorder(mpr121_manager=None,
                          filename=str(tmp_path / "raw.h5"),
                          controllers={sn: {} for sn in board_map},
                          source=client.source(list(channels), hz))


def _on_disk(rec, board_map):
    out = {}
    with h5py.File(rec.filename, "r") as h5f:
        for sn, sensors in board_map.items():
            for s in sensors:
                g = h5f[f"board_{sn}/sensor_{s}"]
                out[(sn, s)] = (g["time_data"][:], g["cap_data"][:])
    return out


def _assert_complete(rec, board_map=DEFAULT_COUNTER_MAP):
    for (sn, s), (t, c) in _on_disk(rec, board_map).items():
        assert len(c) == rec.board_produced[sn][s] > 0, (sn, s)
        assert np.array_equal(c, np.arange(1, len(c) + 1)), (sn, s)
        assert len(t) == len(c)


def test_init_reports_boards(clients):
    res = clients().start()
    assert res.board_info == {"B0": 3, "B1": 3}
    assert res.errors == []


def test_child_does_not_import_solara(clients):
    res = clients("hardware.fake_backends:ImportProbeBackend").start()
    assert res.errors == []


def test_process_recording_is_complete(tmp_path, monkeypatch, clients):
    client = clients()
    client.start()
    rec = _rec(tmp_path, monkeypatch, client)
    rec.start()
    time.sleep(0.5)
    assert rec.stop() is True
    _assert_complete(rec)


def test_stop_before_first_flush_is_complete(tmp_path, monkeypatch, clients):
    client = clients()
    client.start()
    rec = _rec(tmp_path, monkeypatch, client, history=100000)
    rec.start()
    time.sleep(0.2)
    assert rec.stop() is True
    _assert_complete(rec)


def _busy(seconds):
    end = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < end:  # pure Python, holds the GIL like a render
        x += 1


def test_busy_gui_thread_does_not_gap_samples(tmp_path, monkeypatch, clients):
    client = clients("hardware.fake_backends:PyftdiLikeBackend")
    client.start()
    rec = _rec(tmp_path, monkeypatch, client, board_map=PYFTDI_LIKE_MAP,
               channels=(1, 3, 5, 7, 9, 11), history=500, hz=150)
    rec.start()
    time.sleep(0.3)
    for _ in range(5):
        _busy(0.120)
        time.sleep(0.4)
    assert rec.stop() is True
    gaps = [np.diff(t).max() for t, _ in _on_disk(rec, PYFTDI_LIKE_MAP).values()]
    assert max(gaps) < 4 / 150


def test_child_crash_reports_fatal_and_keeps_data(tmp_path, monkeypatch, clients):
    client = clients("hardware.fake_backends:CrashAfterBackend")
    client.start()
    rec = _rec(tmp_path, monkeypatch, client)
    errors = []
    rec.start(on_error=errors.append)
    deadline = time.monotonic() + 5.0
    while not errors and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(errors) == 1 and "hardware process died" in str(errors[0])
    t0 = time.monotonic()
    rec.stop(timeout=2.0)
    assert time.monotonic() - t0 < 5.0
    _assert_complete(rec)


def test_child_exits_when_client_pipe_closes(clients):
    client = clients()
    client.start()
    client._cmd.close()
    client._proc.join(5)
    assert not client._proc.is_alive()


def test_shutdown_is_idempotent_and_ends_process(clients):
    client = clients()
    client.start()
    assert client.is_alive()
    client.shutdown()
    client.shutdown()
    assert not client.is_alive()
