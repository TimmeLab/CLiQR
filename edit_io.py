"""Pure HDF5 read/write for the weight/volume post-hoc editor.

No Solara / hardware imports so it is unit-testable against fixtures.
Mirrors the dataset naming used by recording/recorder.py.
"""
import re
import h5py
from utils.state import SERIAL_NUMBER_SENSOR_MAP

FIELDS = ("start_vol", "stop_vol", "weight")


def board_for_sensor(sensor_id: int) -> str:
    """Return the FT232H serial string that owns this sensor id."""
    for serial, sensors in SERIAL_NUMBER_SENSOR_MAP.items():
        if sensor_id in sensors:
            return serial
    raise ValueError(f"Sensor {sensor_id} is not mapped to any board")


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
    """Read editable volume/weight values for every recorded sensor/cycle."""
    data = {}
    with h5py.File(path, "r") as h5f:
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
                        for field in FIELDS
                    }
    return data


def save_file(path: str, data: dict) -> None:
    """Write edited values back in place; >0 creates (delete-first), else deletes."""
    with h5py.File(path, "r+") as h5f:
        for sensor_id, cycles in data.items():
            serial = board_for_sensor(sensor_id)
            group = h5f[f"board_{serial}/sensor_{sensor_id}"]
            for cycle, fields in cycles.items():
                suf = _suffix(cycle)
                for field in FIELDS:
                    name = f"{field}{suf}"
                    value = fields.get(field, 0.0)
                    if name in group:
                        del group[name]
                    if value and value > 0:
                        group.create_dataset(name, data=value)
