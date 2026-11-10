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


# ---------------------------------------------------------------------------
# 2-bottle choice files: volumes live in sensor_<id>/fluid1, fluid2 subgroups;
# start/stop times and weight stay at the sensor level.
# ---------------------------------------------------------------------------
def build_two_bottle_h5(path, board="board_FT232H0", sensor_id=3, cycles=(0,)):
    with h5py.File(path, "w") as h5f:
        h5f.attrs["two_bottle"] = True
        g = h5f.require_group(board).require_group(f"sensor_{sensor_id}")
        for cycle in cycles:
            suf = "" if cycle == 0 else str(cycle)
            g.create_dataset(f"start_time{suf}", data=100.0 + cycle)
            g.create_dataset(f"weight{suf}", data=24.0 + cycle)
            g.create_dataset(f"fluid1/start_vol{suf}", data=10.0 + cycle)
            g.create_dataset(f"fluid1/stop_vol{suf}", data=9.5)
            g.create_dataset(f"fluid2/start_vol{suf}", data=8.0)
            # fluid2 stop_vol deliberately missing
            g.create_dataset("fluid1/cap_data", data=[1, 2, 3]) if cycle == 0 else None


def test_fields_for_mode():
    assert edit_io.fields_for(False) == ("start_vol", "stop_vol", "weight")
    assert edit_io.fields_for(True) == ("fluid1/start_vol", "fluid1/stop_vol",
                                        "fluid2/start_vol", "fluid2/stop_vol", "weight")


def test_is_two_bottle(tmp_path):
    p1, p2 = tmp_path / "one.h5", tmp_path / "two.h5"
    build_h5(p1, {3: {0: {"start_time": 100.0}}})
    build_two_bottle_h5(p2)
    assert edit_io.is_two_bottle(str(p1)) is False
    assert edit_io.is_two_bottle(str(p2)) is True


def test_load_two_bottle_reads_fluid_volumes(tmp_path):
    p = tmp_path / "raw.h5"
    build_two_bottle_h5(p, cycles=(0, 1))
    data = edit_io.load_file(str(p))
    assert data[3][0] == {"fluid1/start_vol": 10.0, "fluid1/stop_vol": 9.5,
                          "fluid2/start_vol": 8.0, "fluid2/stop_vol": 0.0, "weight": 24.0}
    assert data[3][1]["fluid1/start_vol"] == 11.0 and data[3][1]["weight"] == 25.0
    # field order is display order
    assert tuple(data[3][0]) == edit_io.fields_for(True)


def test_save_two_bottle_writes_fluid_subgroups(tmp_path):
    p = tmp_path / "raw.h5"
    build_two_bottle_h5(p)
    edit_io.save_file(str(p), {3: {0: {"fluid1/start_vol": 10.0, "fluid1/stop_vol": 9.4,
                                        "fluid2/start_vol": 0.0, "fluid2/stop_vol": 7.6,
                                        "weight": 24.5}}})
    with h5py.File(p, "r") as h5f:
        g = h5f["board_FT232H0/sensor_3"]
        assert g["fluid1/stop_vol"][()] == 9.4
        assert g["fluid2/stop_vol"][()] == 7.6
        assert "fluid2/start_vol" not in g          # 0 deletes, as in 1-bottle mode
        assert g["weight"][()] == 24.5
        assert "start_vol" not in g and "stop_vol" not in g  # nothing at sensor level
        assert list(g["fluid1/cap_data"][()]) == [1, 2, 3]   # traces untouched


def test_save_finds_sensor_on_any_board(tmp_path):
    # 8-board rack file (sensor 3 on FT232H0 there, but sensor 13 on FT232H4):
    # the group is located in the file, not via the configured rack map.
    p = tmp_path / "raw.h5"
    build_two_bottle_h5(p, board="board_FT232H4", sensor_id=13)
    data = edit_io.load_file(str(p))
    data[13][0]["weight"] = 30.0
    edit_io.save_file(str(p), data)
    with h5py.File(p, "r") as h5f:
        assert h5f["board_FT232H4/sensor_13/weight"][()] == 30.0


def test_save_rejects_fields_of_other_mode(tmp_path):
    p = tmp_path / "raw.h5"
    build_two_bottle_h5(p)
    with pytest.raises(ValueError):
        edit_io.save_file(str(p), {3: {0: {"start_vol": 5.0}}})


def test_edit_gui_renders_and_saves_two_bottle(tmp_path):
    solara = pytest.importorskip("solara")
    import edit_gui

    p = tmp_path / "raw_data_2026-11-09_12-45-00.h5"
    build_two_bottle_h5(p, board="board_FT232H4", sensor_id=13)
    edit_gui.load_selected(str(p))
    assert edit_gui.two_bottle.value is True
    assert edit_gui.field_label("fluid2/stop_vol") == "Stop Vol (mL)"
    assert edit_gui.field_section("fluid2/stop_vol") == "Fluid 2"
    solara.render(edit_gui.Page(), handle_error=False)

    edit_gui.set_field(13, 0, "fluid2/stop_vol", 7.5)
    edit_gui.save_selected()
    with h5py.File(p, "r") as h5f:
        assert h5f["board_FT232H4/sensor_13/fluid2/stop_vol"][()] == 7.5
    assert not any("ERROR" in m for m in edit_gui.log_messages.value)

    # a 1-bottle file afterwards switches the mode back
    p1 = tmp_path / "raw_data_one.h5"
    build_h5(p1, {3: {0: {"start_time": 100.0, "weight": 24.1}}})
    edit_gui.load_selected(str(p1))
    assert edit_gui.two_bottle.value is False
    solara.render(edit_gui.Page(), handle_error=False)
