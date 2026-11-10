"""2-bottle choice mode: each sensor (cage) records an adjacent channel pair,
(fluid 1, fluid 2), into sensor_<id>/fluid1 and sensor_<id>/fluid2. Start/stop
times and weight stay at the sensor level; volumes go in the fluid subgroups."""
import asyncio
from dataclasses import replace

import h5py
import numpy as np
import pytest

import utils.state
import recording.recorder
from hardware.mpr121 import MPR121Manager
from recording.recorder import SensorRecorder, measurement_warnings
from utils.state import SensorState


@pytest.fixture
def two_bottle_on(monkeypatch):
    monkeypatch.setitem(utils.state.session, "two_bottle", True)


def test_two_bottle_channel_pairs():
    configs = utils.state._RACK_CONFIGS
    assert configs["8board"]["channels_two_bottle"] == [0, 1, 5, 6, 10, 11]
    assert configs["4board"]["channels_two_bottle"] == list(range(12))
    # One (fluid 1, fluid 2) pair per mapped sensor on every rack.
    for config in configs.values():
        for sensors in config["map"].values():
            assert len(config["channels_two_bottle"]) == 2 * len(sensors)


def test_active_channels_follow_mode(monkeypatch):
    monkeypatch.setitem(utils.state.session, "two_bottle", False)
    assert utils.state.active_channels() == utils.state.ACTIVE_CHANNELS
    monkeypatch.setitem(utils.state.session, "two_bottle", True)
    assert utils.state.active_channels() == utils.state.ACTIVE_CHANNELS_TWO_BOTTLE


def test_channel_keys_and_paths():
    assert utils.state.channel_keys([4, 5], False) == [4, 5]
    keys = utils.state.channel_keys([4, 5], True)
    assert keys == [(4, 1), (4, 2), (5, 1), (5, 2)]
    assert utils.state.channel_group_path(4) == "sensor_4"
    assert utils.state.channel_group_path((4, 2)) == "sensor_4/fluid2"


class _ChannelPort:
    """Port whose channel c reads back the value 100 + c."""

    def read_from(self, register, num_bytes):
        buf = bytearray(num_bytes)
        for chan in range(num_bytes // 2):
            buf[2 * chan] = 100 + chan
        return buf


def test_read_returns_channel_pairs(two_bottle_on):
    mgr = MPR121Manager({"BOARD": {"port": _ChannelPort()}})
    _, cap_data, _ = mgr.read_sensor_data("BOARD")
    assert list(cap_data) == [100 + c for c in utils.state.ACTIVE_CHANNELS_TWO_BOTTLE]


_TEST_MAP = {"BOARD": [1, 2, 3]}


class _PairManager:
    """Returns fluid 1 = +v, fluid 2 = -v for each sensor on each read."""

    def __init__(self):
        self.v = 0

    def read_sensor_data(self, sn):
        self.v += 1
        cap = []
        for sensor in _TEST_MAP[sn]:
            cap += [sensor * 1000 + self.v, -(sensor * 1000 + self.v)]
        return [1000.0 + self.v] * len(cap), cap, sn


def test_recorder_writes_fluid_subgroups(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.state, "SERIAL_NUMBER_SENSOR_MAP", _TEST_MAP)
    monkeypatch.setattr(recording.recorder, "HISTORY_SIZE", 10)
    monkeypatch.setattr(recording.recorder, "MAX_SAMPLE_HZ", 500)

    rec = SensorRecorder(mpr121_manager=_PairManager(),
                         filename=str(tmp_path / "raw.h5"),
                         controllers={"BOARD": object()},
                         two_bottle=True)

    async def run():
        task = asyncio.create_task(rec.record_sensors())
        await asyncio.sleep(0.3)
        rec.stop()
        await task

    asyncio.run(run())

    with h5py.File(rec.filename, "r") as h5f:
        assert bool(h5f.attrs["two_bottle"]) is True
        for sensor in _TEST_MAP["BOARD"]:
            f1 = h5f[f"board_BOARD/sensor_{sensor}/fluid1"]
            f2 = h5f[f"board_BOARD/sensor_{sensor}/fluid2"]
            cap1, cap2 = f1["cap_data"][:], f2["cap_data"][:]
            assert cap1.shape[0] >= 2
            assert cap1.shape == cap2.shape == f1["time_data"].shape
            # Each fluid got its own channel of the pair, for the right sensor.
            assert np.all(cap1 // 1000 == sensor)
            assert np.array_equal(cap2, -cap1)


def test_metadata_volumes_go_to_fluid_subgroups(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.state, "SERIAL_NUMBER_SENSOR_MAP", _TEST_MAP)
    rec = SensorRecorder(mpr121_manager=None,
                         filename=str(tmp_path / "raw.h5"),
                         controllers={"BOARD": object()},
                         two_bottle=True)
    rec.initialize_hdf5_file()
    rec.write_sensor_metadata(sensor_id=2, start_time=5.0, cycle=1)
    rec.write_sensor_metadata(sensor_id=2, start_vol=8.6, stop_vol=8.1,
                              weight=21.0, start_vol2=7.5, stop_vol2=6.9, cycle=1)

    with h5py.File(rec.filename, "r") as h5f:
        g = h5f["board_BOARD/sensor_2"]
        assert g["start_time1"][()] == 5.0
        assert g["weight1"][()] == 21.0
        assert g["fluid1/start_vol1"][()] == 8.6
        assert g["fluid1/stop_vol1"][()] == 8.1
        assert g["fluid2/start_vol1"][()] == 7.5
        assert g["fluid2/stop_vol1"][()] == 6.9
        assert "start_vol1" not in g


def test_flush_measurements_two_bottle(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.state, "SERIAL_NUMBER_SENSOR_MAP", _TEST_MAP)
    states = {1: replace(SensorState(sensor_id=1), is_recording=True,
                         start_time=1.0, start_volume=8.0, start_volume2=7.0,
                         weight=20.0)}
    rec = SensorRecorder(mpr121_manager=None,
                         filename=str(tmp_path / "raw.h5"),
                         controllers={"BOARD": object()},
                         measurements_provider=lambda: states,
                         two_bottle=True)
    rec.initialize_hdf5_file()
    rec._flush_measurements()

    with h5py.File(rec.filename, "r") as h5f:
        g = h5f["board_BOARD/sensor_1"]
        assert g["fluid1/start_vol"][()] == 8.0
        assert g["fluid2/start_vol"][()] == 7.0
        assert g["weight"][()] == 20.0
        assert "stop_vol" not in g["fluid1"]


def test_measurement_warnings_two_bottle():
    s = replace(SensorState(sensor_id=1), start_time=1.0, start_volume=5.0,
                stop_volume=4.0, start_volume2=5.0, weight=20.0)
    assert measurement_warnings({1: s}, two_bottle=True) == [
        "Sensor 1: no fluid2 stop_vol recorded"]
    # Fluid 2 volumes are ignored in one-bottle mode.
    assert measurement_warnings({1: s}) == []
