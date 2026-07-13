"""
Post-hoc editor for per-sensor volume and weight measurements.

Start volume, stop volume, and animal weight are typed in by hand at recording
time and are easy to fat-finger. This little Solara app opens a raw_data HDF5
file after the fact, shows the 24 sensor cards with the recorded values, and
writes corrections back in place.

Run it the same way as the main recorder:

    solara run edit_measurements.py

File layout is board_<serial>/sensor_<id>/{start_vol, stop_vol, weight}.
Recording cycles past the first append a numeric suffix (start_vol1, ...); every
cycle present in the file gets its own set of entry boxes.
"""
import re
from pathlib import Path

import h5py
import solara

# Editable fields, in display order, with human labels.
FIELDS = [
    ("start_vol", "Start Vol (mL)"),
    ("stop_vol", "Stop Vol (mL)"),
    ("weight", "Weight (g)"),
]

NUM_SENSORS = 24
COLS = 6  # matches the physical rack: 4 shelves x 6 positions

# --- reactive app state ----------------------------------------------------
selected_file = solara.reactive(None)  # Path | None
# {sensor_id: {"group": "board_x/sensor_y", "cycles": [0, 1, ...]}}
sensor_index = solara.reactive({})
# {(sensor_id, cycle, field): value or None}, the live editable buffer
edits = solara.reactive({})
status_message = solara.reactive("")


def dataset_name(field: str, cycle: int) -> str:
    return field if cycle == 0 else f"{field}{cycle}"


def scan_file(path: Path):
    """Build the sensor index and load current values into the edit buffer."""
    index = {}
    values = {}
    with h5py.File(path, "r") as h5f:
        for board_key in h5f:
            board = h5f[board_key]
            if not isinstance(board, h5py.Group):
                continue
            for sensor_key in board:
                m = re.fullmatch(r"sensor_(\d+)", sensor_key)
                if not m:
                    continue
                sid = int(m.group(1))
                group = board[sensor_key]
                cycles = set()
                for name in group:
                    fm = re.fullmatch(r"(?:start_vol|stop_vol|weight)(\d*)", name)
                    if fm:
                        cycles.add(int(fm.group(1)) if fm.group(1) else 0)
                cycles = sorted(cycles) if cycles else [0]
                index[sid] = {"group": f"{board_key}/{sensor_key}", "cycles": cycles}
                for cycle in cycles:
                    for field, _ in FIELDS:
                        name = dataset_name(field, cycle)
                        val = float(group[name][()]) if name in group else None
                        values[(sid, cycle, field)] = val
    return index, values


def load_file(path):
    if path is None or Path(path).is_dir():
        return
    try:
        index, values = scan_file(path)
    except (OSError, KeyError) as exc:
        status_message.value = f"ERROR: cannot open file: {exc}"
        return
    if not index:
        status_message.value = "ERROR: no sensor groups found (expected board_*/sensor_*)"
        return
    selected_file.value = Path(path)
    sensor_index.value = index
    edits.value = values
    status_message.value = f"Loaded {len(index)} sensor(s) from {Path(path).name}"


def save_file():
    path = selected_file.value
    if path is None:
        return
    index = sensor_index.value
    written = cleared = 0
    try:
        with h5py.File(path, "r+") as h5f:
            for (sid, cycle, field), value in edits.value.items():
                group = h5f[index[sid]["group"]]
                name = dataset_name(field, cycle)
                if value is None:
                    if name in group:
                        del group[name]
                        cleared += 1
                    continue
                if name in group:
                    del group[name]
                group.create_dataset(name, data=float(value))
                written += 1
    except OSError as exc:
        status_message.value = f"ERROR: write failed: {exc}"
        return
    msg = f"Saved: {written} value(s) written"
    if cleared:
        msg += f", {cleared} blank field(s) cleared"
    status_message.value = msg


@solara.component
def CycleFields(sid: int, cycle: int, show_label: bool):
    if show_label:
        solara.Text(
            f"Cycle {cycle + 1}",
            style={"font-weight": "bold", "font-size": "11px", "color": "#336"},
        )
    for field, label in FIELDS:
        key = (sid, cycle, field)

        def set_value(value, key=key):
            new = dict(edits.value)
            new[key] = value
            edits.value = new

        solara.InputFloat(
            label=label,
            value=edits.value.get(key),
            on_value=set_value,
            clearable=True,
            style={"width": "100%"},
        )


@solara.component
def SensorCard(sid: int):
    info = sensor_index.value.get(sid)
    present = info is not None
    border = "#4CAF50" if present else "#DDD"

    with solara.Card(
        style={
            "width": "200px",
            "border": f"2px solid {border}",
            "padding": "8px",
            "opacity": "1" if present else "0.5",
        }
    ):
        title = f"Sensor {sid}" if present else f"Sensor {sid} (not recorded)"
        solara.Text(title, style={"font-weight": "bold", "font-size": "13px"})
        if not present:
            solara.Text("—", style={"color": "#bbb"})
            return
        with solara.Column(style={"gap": "4px", "margin-top": "6px"}):
            cycles = info["cycles"]
            for cycle in cycles:
                CycleFields(sid, cycle, show_label=len(cycles) > 1)


@solara.component
def Page():
    solara.Title("CLiQR — Edit Volume / Weight Measurements")

    with solara.Column(style={"padding": "16px", "gap": "12px"}):
        solara.Markdown("### Edit recorded volume / weight measurements")

        with solara.Card(title="1. Pick a raw_data HDF5 file"):
            solara.FileBrowser(
                directory=str(Path.cwd()),
                on_file_open=load_file,
                filter=lambda p: p.is_dir() or p.suffix in (".h5", ".hdf5"),
                directory_first=True,
            )
            solara.Markdown("*Double-click an `.h5` file to load it.*")

        if status_message.value:
            color = "error" if status_message.value.startswith("ERROR") else "success"
            solara.Info(status_message.value, color=color)

        if selected_file.value is not None:
            with solara.Row(style={"align-items": "center", "gap": "12px"}):
                solara.Button(
                    "Save changes to file",
                    on_click=save_file,
                    color="primary",
                )
                solara.Text(
                    str(selected_file.value),
                    style={"font-family": "monospace", "font-size": "12px", "color": "#555"},
                )

            with solara.Card(title="2. Edit sensor cards"):
                for shelf in range(NUM_SENSORS // COLS):
                    with solara.Row(style={"gap": "10px", "flex-wrap": "wrap",
                                           "margin-bottom": "10px"}):
                        for offset in range(COLS):
                            SensorCard(sid=shelf * COLS + offset + 1)
