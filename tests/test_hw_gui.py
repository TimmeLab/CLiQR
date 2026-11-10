"""GUI wiring of the hardware process: init/reuse, refresh rules, sessions.

The hardware process belongs to the Solara server, not to a browser tab: a
refresh or a second Initialize must reuse it (two processes would fight over
the USB boards), and recording must carry on across a refresh with no gap.
Uses the fake CounterBackend so the real spawned process and IPC path run.
"""
import time

import h5py
import numpy as np
import pandas as pd
import pytest

import components.hardware_status as hw
import components.session_controls as sc
import recording.recorder
import utils.state
from hardware.fake_backends import DEFAULT_COUNTER_MAP
from utils import state


@pytest.fixture
def gui(tmp_path, monkeypatch):
    monkeypatch.setattr(hw, "BACKEND_PATH", "hardware.fake_backends:CounterBackend")
    monkeypatch.setattr(hw, "hw_client", None)
    monkeypatch.setattr(utils.state, "SERIAL_NUMBER_SENSOR_MAP", DEFAULT_COUNTER_MAP)
    monkeypatch.setattr(utils.state, "ACTIVE_CHANNELS", [1, 6, 11])
    monkeypatch.setattr(recording.recorder, "HISTORY_SIZE", 50)
    monkeypatch.setattr(sc, "current_recorder", None)
    state.layout_df.set(pd.DataFrame({"animal": ["A1"]}, index=[1]))
    state.output_directory.set(str(tmp_path))
    state.camera_enabled.set(False)
    state.set_session("recording_all", False)
    yield
    if state.session["recording_all"]:
        sc.stop_recording()
    hw.disconnect_hardware()
    state.layout_df.set(pd.DataFrame())
    state.set_session("recording_all", False)


def _assert_complete(rec):
    with h5py.File(rec.filename, "r") as h5f:
        for sn, sensors in DEFAULT_COUNTER_MAP.items():
            for s in sensors:
                g = h5f[f"board_{sn}/sensor_{s}"]
                t, c = g["time_data"][:], g["cap_data"][:]
                assert len(c) == rec.board_produced[sn][s] > 0
                assert np.all(np.diff(c) == 1)
                assert np.diff(t).max() < 4 / 150


def _record(seconds):
    sc.start_recording()
    rec = sc.current_recorder
    time.sleep(seconds)
    sc.stop_recording()
    return rec


def test_initialize_publishes_plain_board_info(gui):
    hw.initialize_hardware()
    assert hw.hw_client.is_alive()
    assert state.session["boards_connected"] == {"B0": 3, "B1": 3}
    assert state.session["i2c_controllers"] == {"B0": {"num_sensors": 3},
                                                "B1": {"num_sensors": 3}}


def test_initialize_twice_reuses_process(gui):
    hw.initialize_hardware()
    pid = hw.hw_client._proc.pid
    hw.initialize_hardware()
    assert hw.hw_client._proc.pid == pid and hw.hw_client.is_alive()
    assert any("reusing" in m for m in state.log_messages.value)


def test_refresh_refused_while_recording(gui):
    hw.initialize_hardware()
    pid = hw.hw_client._proc.pid
    sc.start_recording()
    hw.refresh_hardware()
    assert hw.hw_client._proc.pid == pid and hw.hw_client.is_alive()
    assert any("cannot refresh hardware while recording" in m
               for m in state.log_messages.value)


def test_refresh_when_idle_restarts_process(gui):
    hw.initialize_hardware()
    old = hw.hw_client
    hw.refresh_hardware()
    assert hw.hw_client is not old and hw.hw_client.is_alive()
    assert not old.is_alive()


def test_two_sessions_one_process(gui):
    hw.initialize_hardware()
    pid = hw.hw_client._proc.pid
    first = _record(0.3)
    time.sleep(1.1)  # filenames are second-resolution timestamps
    second = _record(0.3)
    assert first.filename != second.filename
    _assert_complete(first)
    _assert_complete(second)
    assert hw.hw_client._proc.pid == pid


def test_recording_survives_simulated_refresh(gui):
    hw.initialize_hardware()
    sc.start_recording()
    rec = sc.current_recorder
    time.sleep(0.3)
    # A NEW kernel context: reactives at defaults, server globals intact.
    state.recording_all.set(False)
    state.boards_connected.set({})
    state.i2c_controllers.set({})
    state.rehydrate_reactives_from_session()
    assert state.recording_all.value is True
    time.sleep(0.3)
    sc.stop_recording()
    _assert_complete(rec)
    assert state.session["recording_all"] is False


def test_test_button_reads_live_buffer(gui):
    from components.sensor_card import test_sensor
    hw.initialize_hardware()
    sc.start_recording()
    time.sleep(0.3)
    test_sensor(1)
    traces = state.test_plot_data.value["traces"]
    assert len(traces[0]["cap_data"]) > 0


def test_disconnect_ends_process(gui):
    hw.initialize_hardware()
    client = hw.hw_client
    hw.disconnect_hardware()
    assert hw.hw_client is None and not client.is_alive()
    assert state.session["boards_connected"] == {}


def test_concurrent_initialize_spawns_one_process(gui, monkeypatch):
    """Two tabs (or a double-click) initializing at once must not spawn two
    hardware processes fighting over the boards."""
    import threading
    from hardware.hw_client import HardwareClient
    made = []

    class Counting(HardwareClient):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(hw, "HardwareClient", Counting)
    ts = [threading.Thread(target=hw.initialize_hardware) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    alive = [c for c in made if c.is_alive()]
    try:
        assert len(made) == 1 and len(alive) == 1
    finally:
        for c in made:
            if c is not hw.hw_client:
                c.shutdown()
