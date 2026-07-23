"""The recorder persists changed, >0 measurements to the h5 during the run so a
reconnect/crash before Stop cannot lose them. The >0 guard means a later reset
(source reads 0.0) never clobbers a saved value."""
from dataclasses import replace

import h5py

from recording.recorder import SensorRecorder, measurement_warnings
from utils.state import SERIAL_NUMBER_SENSOR_MAP, SensorState


def _make_recorder(tmp_path, provider):
    serial = next(iter(SERIAL_NUMBER_SENSOR_MAP))
    rec = SensorRecorder(mpr121_manager=None,
                         filename=str(tmp_path / "raw.h5"),
                         controllers={serial: object()},
                         measurements_provider=provider)
    rec.initialize_hdf5_file()
    sensor_id = SERIAL_NUMBER_SENSOR_MAP[serial][0]
    return rec, serial, sensor_id


def test_flush_writes_started_sensor_measurements(tmp_path):
    box = {}

    def provider():
        return box["states"]

    rec, serial, sid = _make_recorder(tmp_path, provider)
    states = {sid: replace(SensorState(sensor_id=sid),
                           is_recording=True, start_time=1.0,
                           start_volume=8.6, stop_volume=8.4, weight=21.1)}
    box["states"] = states

    rec._flush_measurements()

    with h5py.File(rec.filename, "r") as f:
        g = f[f"board_{serial}/sensor_{sid}"]
        assert g["start_vol"][()] == 8.6
        assert g["stop_vol"][()] == 8.4
        assert g["weight"][()] == 21.1


def test_flush_skips_unstarted_and_zero(tmp_path):
    box = {}
    rec, serial, sid = _make_recorder(tmp_path, lambda: box["states"])
    # Started but measurements still 0.0 -> nothing written.
    box["states"] = {sid: replace(SensorState(sensor_id=sid),
                                  is_recording=True, start_time=1.0)}
    rec._flush_measurements()
    with h5py.File(rec.filename, "r") as f:
        assert "start_vol" not in f[f"board_{serial}/sensor_{sid}"]


def test_later_zero_does_not_clobber_saved_value(tmp_path):
    box = {}
    rec, serial, sid = _make_recorder(tmp_path, lambda: box["states"])
    box["states"] = {sid: replace(SensorState(sensor_id=sid),
                                  is_recording=True, start_time=1.0,
                                  start_volume=6.5, weight=20.0)}
    rec._flush_measurements()
    # Simulate a context reset: source now reads defaults (0.0).
    box["states"] = {sid: replace(SensorState(sensor_id=sid),
                                  is_recording=True, start_time=1.0)}
    rec._flush_measurements()
    with h5py.File(rec.filename, "r") as f:
        g = f[f"board_{serial}/sensor_{sid}"]
        assert g["start_vol"][()] == 6.5   # preserved, not clobbered
        assert g["weight"][()] == 20.0


def test_measurement_warnings_flags_started_missing():
    sensors = {
        1: replace(SensorState(sensor_id=1), start_time=1.0,
                   start_volume=5.0, stop_volume=4.0, weight=20.0),  # complete
        2: replace(SensorState(sensor_id=2), start_time=1.0,
                   start_volume=5.0),                                 # missing stop_vol+weight
        3: SensorState(sensor_id=3),                                 # never started
    }
    msgs = measurement_warnings(sensors)
    assert len(msgs) == 1
    assert "Sensor 2" in msgs[0]
    assert "stop_vol" in msgs[0] and "weight" in msgs[0]
    assert "start_vol" not in msgs[0]
