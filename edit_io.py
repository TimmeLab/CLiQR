"""Pure HDF5 read/write for the weight/volume post-hoc editor.

No Solara / hardware imports so it is unit-testable against fixtures.
Mirrors the dataset naming used by recording/recorder.py.

2-bottle choice files (file attr two_bottle=True) keep start/stop times and
weight at the sensor level but put each bottle's volumes in its fluid subgroup,
so their editable fields are paths relative to the sensor group:
fluid1/start_vol, fluid1/stop_vol, fluid2/start_vol, fluid2/stop_vol, weight.
"""
import re
import h5py
from utils.state import SERIAL_NUMBER_SENSOR_MAP

FIELDS = ("start_vol", "stop_vol", "weight")
FIELDS_TWO_BOTTLE = ("fluid1/start_vol", "fluid1/stop_vol",
                     "fluid2/start_vol", "fluid2/stop_vol", "weight")


def fields_for(two_bottle: bool) -> tuple:
    """Editable fields (dataset paths relative to the sensor group), in display order."""
    return FIELDS_TWO_BOTTLE if two_bottle else FIELDS


def _file_is_two_bottle(h5f) -> bool:
    return bool(h5f.attrs.get("two_bottle", False))


def is_two_bottle(path: str) -> bool:
    """True if the raw file was recorded in 2-bottle choice mode."""
    with h5py.File(path, "r") as h5f:
        return _file_is_two_bottle(h5f)


def board_for_sensor(sensor_id: int) -> str:
    """Return the FT232H serial string that owns this sensor id."""
    for serial, sensors in SERIAL_NUMBER_SENSOR_MAP.items():
        if sensor_id in sensors:
            return serial
    raise ValueError(f"Sensor {sensor_id} is not mapped to any board")


def _sensor_group(h5f, sensor_id: int):
    """The board_*/sensor_<id> group, found in the file itself.

    Sensor IDs are unique across boards, so this works for files from either
    rack design regardless of which one CLIQR_RACK currently selects.
    """
    name = f"sensor_{sensor_id}"
    for board_name in h5f:
        if board_name.startswith("board_") and name in h5f[board_name]:
            return h5f[board_name][name]
    return h5f[f"board_{board_for_sensor(sensor_id)}/{name}"]


def _suffix(cycle: int) -> str:
    return "" if cycle == 0 else str(cycle)


def detect_cycles(group) -> list:
    """Cycle numbers present in a sensor group, found via start_time* datasets."""
    cycles = set()
    for key in group:
        m = re.fullmatch(r"start_time(\d*)", key)
        if m:
            cycles.add(int(m.group(1)) if m.group(1) else 0)
    return sorted(cycles)


def load_file(path: str) -> dict:
    """Read editable volume/weight values for every recorded sensor/cycle.

    Returns {sensor_id: {cycle: {field: float}}} with fields in display order
    (see fields_for); a missing dataset reads as 0.0.
    """
    data = {}
    with h5py.File(path, "r") as h5f:
        fields = fields_for(_file_is_two_bottle(h5f))
        for board_name in h5f:
            if not board_name.startswith("board_"):
                continue
            board = h5f[board_name]
            for sensor_name in board:
                if not sensor_name.startswith("sensor_"):
                    continue
                sensor_id = int(sensor_name.split("_")[1])
                group = board[sensor_name]
                cycles = detect_cycles(group)
                if not cycles:
                    continue
                data[sensor_id] = {}
                for cycle in cycles:
                    suf = _suffix(cycle)
                    data[sensor_id][cycle] = {
                        field: float(group[f"{field}{suf}"][()])
                        if f"{field}{suf}" in group else 0.0
                        for field in fields
                    }
    return data


def save_file(path: str, data: dict) -> None:
    """Write edited values back in place; >0 creates (delete-first), else deletes.

    Only the fields of the file's own mode are accepted (ValueError otherwise),
    so a 1-bottle volume can never be written into a 2-bottle file or vice versa.
    """
    with h5py.File(path, "r+") as h5f:
        allowed = fields_for(_file_is_two_bottle(h5f))
        for sensor_id, cycles in data.items():
            for fields in cycles.values():
                unknown = set(fields) - set(allowed)
                if unknown:
                    raise ValueError(f"Sensor {sensor_id}: fields {sorted(unknown)} do not "
                                     f"belong to this file (expected {list(allowed)})")
        for sensor_id, cycles in data.items():
            group = _sensor_group(h5f, sensor_id)
            for cycle, fields in cycles.items():
                suf = _suffix(cycle)
                for field in allowed:
                    name = f"{field}{suf}"
                    value = fields.get(field, 0.0)
                    if name in group:
                        del group[name]
                    if value and value > 0:
                        group.create_dataset(name, data=value)
