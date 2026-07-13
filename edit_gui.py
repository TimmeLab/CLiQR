"""Standalone Solara GUI for post-hoc editing of weight/volume measurements.

Run with:  solara run edit_gui.py

Pick a raw_data*.h5 recording file, correct any mistyped start/stop volume or
weight values (all recording cycles shown), and save the corrections in place.
"""
import datetime
from pathlib import Path

import solara

import edit_io

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
selected_file = solara.reactive("")
edit_data = solara.reactive({})   # {sensor_id: {cycle: {field: float}}}
log_messages = solara.reactive([])

FIELD_LABELS = {
    "start_vol": "Start Vol (mL)",
    "stop_vol": "Stop Vol (mL)",
    "weight": "Weight (g)",
}


def add_log(message: str):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_messages.set(log_messages.value + [f"[{stamp}] {message}"])


def load_selected(path: str):
    try:
        data = edit_io.load_file(path)
    except Exception as exc:  # bad/corrupt/non-h5 file
        add_log(f"ERROR loading {Path(path).name}: {exc}")
        edit_data.set({})
        selected_file.set("")
        return
    edit_data.set(data)
    selected_file.set(path)
    n_cycles = sum(len(c) for c in data.values())
    add_log(f"Loaded {len(data)} sensors, {n_cycles} cycles from {Path(path).name}")


def save_selected():
    if not selected_file.value:
        return
    try:
        edit_io.save_file(selected_file.value, edit_data.value)
    except Exception as exc:
        add_log(f"ERROR saving: {exc}")
        return
    add_log(f"Saved corrections to {Path(selected_file.value).name}")


def set_field(sensor_id: int, cycle: int, field: str, value):
    data = {sid: {c: dict(fields) for c, fields in cycles.items()}
            for sid, cycles in edit_data.value.items()}
    data[sensor_id][cycle][field] = value if value is not None else 0.0
    edit_data.set(data)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
@solara.component
def EditSensorCard(sensor_id: int):
    cycles = edit_data.value.get(sensor_id)
    border = "#4CAF50" if cycles else "#9E9E9E"
    with solara.Card(style={
        "width": "250px", "min-height": "160px",
        "border": f"2px solid {border}", "padding": "10px",
    }):
        solara.Text(f"Sensor {sensor_id}",
                    style={"font-weight": "bold", "font-size": "14px"})
        if not cycles:
            solara.Text("no data", style={"color": "#9E9E9E", "font-size": "11px"})
            return
        multi = len(cycles) > 1
        for cycle in sorted(cycles):
            with solara.Column(style={"margin-top": "8px", "gap": "4px"}):
                if multi:
                    solara.Text(f"Cycle {cycle}",
                                style={"font-size": "11px", "font-weight": "bold"})
                for field in edit_io.FIELDS:
                    solara.InputFloat(
                        label=FIELD_LABELS[field],
                        value=cycles[cycle][field],
                        on_value=lambda v, s=sensor_id, c=cycle, f=field: set_field(s, c, f, v),
                        continuous_update=True,
                        style={"width": "100%"},
                    )


@solara.component
def EditSensorGrid():
    with solara.Card(title="Sensor Grid",
                     style={"margin-bottom": "20px"}):
        solara.Markdown("*Sensors arranged to match physical rack layout "
                        "(4 shelves × 6 positions)*")
        for shelf in range(4):
            label = ["Top", "Second", "Third", "Bottom"][shelf]
            with solara.Column(style={"margin-bottom": "20px"}):
                solara.Text(f"Shelf {shelf + 1} ({label})",
                            style={"font-weight": "bold", "margin-bottom": "10px"})
                with solara.Row(style={"gap": "10px", "flex-wrap": "wrap"}):
                    for offset in range(6):
                        EditSensorCard(sensor_id=shelf * 6 + offset + 1)


@solara.component
def Page():
    solara.Title("CLiQR — Weight/Volume Editor")
    with solara.Column(style={"padding": "20px", "gap": "16px"}):
        solara.Markdown("# CLiQR — Weight/Volume Editor")
        solara.Markdown("Pick a `raw_data*.h5` file, correct values, then Save.")

        directory = solara.use_reactive(Path("Lickometry Data").resolve())

        def on_file(path):
            if path is not None and path.name.startswith("raw_data") \
                    and path.suffix == ".h5":
                load_selected(str(path))

        solara.FileBrowser(
            directory=directory,
            on_file_open=on_file,
            filter=lambda p: p.is_dir() or (
                p.name.startswith("raw_data") and p.suffix == ".h5"),
        )

        if selected_file.value:
            solara.Info(f"Editing: {Path(selected_file.value).name}")
            solara.Button("SAVE", color="primary", on_click=save_selected)
            EditSensorGrid()
        else:
            solara.Warning("No file loaded — select a raw_data*.h5 file above.")

        with solara.Card(title="Activity Log"):
            with solara.Column(style={"max-height": "160px", "overflow-y": "auto"}):
                for line in reversed(log_messages.value[-50:]):
                    solara.Text(line, style={"font-family": "monospace",
                                             "font-size": "12px"})
