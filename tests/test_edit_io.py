import h5py
import pytest
import edit_io


def build_h5(path, sensors):
    """sensors: {sensor_id: {cycle: {"start_time": t, "start_vol": v, "stop_vol": v, "weight": w}}}
    Only keys present in the inner dict are written. Creates board groups as needed."""
    with h5py.File(path, "w") as h5f:
        for sensor_id, cycles in sensors.items():
            sn = edit_io.board_for_sensor(sensor_id)
            board = h5f.require_group(f"board_{sn}")
            group = board.require_group(f"sensor_{sensor_id}")
            for cycle, fields in cycles.items():
                suf = "" if cycle == 0 else str(cycle)
                for name, value in fields.items():
                    group.create_dataset(f"{name}{suf}", data=value)


def test_board_for_sensor_maps_known_sensor():
    assert edit_io.board_for_sensor(1) == "FT232H0"
    assert edit_io.board_for_sensor(24) == "FT232H3"


def test_board_for_sensor_raises_on_unmapped():
    with pytest.raises(ValueError):
        edit_io.board_for_sensor(999)


def test_load_reads_single_cycle_values(tmp_path):
    p = tmp_path / "raw.h5"
    build_h5(p, {3: {0: {"start_time": 100.0, "start_vol": 5.0, "stop_vol": 3.2, "weight": 24.1}}})
    data = edit_io.load_file(str(p))
    assert data == {3: {0: {"start_vol": 5.0, "stop_vol": 3.2, "weight": 24.1}}}


def test_load_defaults_missing_fields_to_zero(tmp_path):
    p = tmp_path / "raw.h5"
    build_h5(p, {3: {0: {"start_time": 100.0, "weight": 24.1}}})
    data = edit_io.load_file(str(p))
    assert data[3][0] == {"start_vol": 0.0, "stop_vol": 0.0, "weight": 24.1}


def test_load_reads_multiple_cycles(tmp_path):
    p = tmp_path / "raw.h5"
    build_h5(p, {3: {
        0: {"start_time": 100.0, "weight": 24.1},
        1: {"start_time": 200.0, "weight": 23.8},
    }})
    data = edit_io.load_file(str(p))
    assert sorted(data[3].keys()) == [0, 1]
    assert data[3][1]["weight"] == 23.8


def test_load_skips_sensor_without_start_time(tmp_path):
    p = tmp_path / "raw.h5"
    # sensor group exists but never recorded (no start_time)
    with h5py.File(p, "w") as h5f:
        h5f.require_group("board_FT232H0").require_group("sensor_2")
    data = edit_io.load_file(str(p))
    assert data == {}


def test_save_updates_value_in_place(tmp_path):
    p = tmp_path / "raw.h5"
    build_h5(p, {3: {0: {"start_time": 100.0, "weight": 24.1}}})
    edit_io.save_file(str(p), {3: {0: {"start_vol": 0.0, "stop_vol": 0.0, "weight": 25.0}}})
    with h5py.File(p, "r") as h5f:
        assert h5f["board_FT232H0/sensor_3/weight"][()] == 25.0


def test_save_zero_deletes_dataset(tmp_path):
    p = tmp_path / "raw.h5"
    build_h5(p, {3: {0: {"start_time": 100.0, "weight": 24.1}}})
    edit_io.save_file(str(p), {3: {0: {"start_vol": 0.0, "stop_vol": 0.0, "weight": 0.0}}})
    with h5py.File(p, "r") as h5f:
        assert "weight" not in h5f["board_FT232H0/sensor_3"]


def test_save_writes_cycle_suffix(tmp_path):
    p = tmp_path / "raw.h5"
    build_h5(p, {3: {
        0: {"start_time": 100.0},
        1: {"start_time": 200.0},
    }})
    edit_io.save_file(str(p), {3: {1: {"start_vol": 6.0, "stop_vol": 0.0, "weight": 0.0}}})
    with h5py.File(p, "r") as h5f:
        assert h5f["board_FT232H0/sensor_3/start_vol1"][()] == 6.0


def test_save_preserves_untouched_datasets(tmp_path):
    p = tmp_path / "raw.h5"
    build_h5(p, {3: {0: {"start_time": 100.0, "weight": 24.1}}})
    edit_io.save_file(str(p), {3: {0: {"start_vol": 0.0, "stop_vol": 0.0, "weight": 25.0}}})
    with h5py.File(p, "r") as h5f:
        assert h5f["board_FT232H0/sensor_3/start_time"][()] == 100.0
