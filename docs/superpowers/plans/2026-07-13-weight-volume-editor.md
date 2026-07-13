# Weight/Volume Post-Hoc Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Solara GUI to correct weight/volume data-entry errors in existing `raw_data*.h5` recording files.

**Architecture:** Split into a pure-I/O module (`edit_io.py`, no Solara — unit-testable against h5 fixtures) and a Solara UI (`edit_gui.py`) that reuses the recording GUI's 24-card grid layout with recording controls stripped. Load reads existing volume/weight datasets (all cycles) into a nested dict; Save writes edits back in place using the recorder's delete-then-create idiom.

**Tech Stack:** Python, `h5py`, `solara`, `pytest`.

## Global Constraints

- HDF5 layout is fixed by `recording/recorder.py`: `board_{serial}/sensor_{id}/` groups; cycle 0 datasets have no suffix, cycle `c ≥ 1` uses integer suffix `c`.
- Datasets edited: `start_vol`, `stop_vol`, `weight` (+ `{c}` suffix). Never touch `time_data`, `cap_data`, `start_time*`, `stop_time*`, `comments`.
- Real measurements are never exactly 0; a value `> 0` means "has value", `0`/blank means "delete the dataset".
- Sensor→board resolution via `SERIAL_NUMBER_SENSOR_MAP` in `utils/state.py`.
- Save is in place (overwrite selected file). No hardware imports anywhere in the new files.

---

## File Structure

- Create: `edit_io.py` — pure HDF5 read/write. Functions: `board_for_sensor`, `detect_cycles`, `load_file`, `save_file`. No Solara.
- Create: `edit_gui.py` — Solara app. Reactive state, FileBrowser, `EditSensorCard`, `EditSensorGrid`, save button, activity log. Imports `edit_io`.
- Create: `tests/test_edit_io.py` — unit tests + h5 fixture builder for `edit_io`.

---

### Task 1: Pure HDF5 I/O module (`edit_io.py`)

**Files:**
- Create: `edit_io.py`
- Test: `tests/test_edit_io.py`

**Interfaces:**
- Consumes: `SERIAL_NUMBER_SENSOR_MAP` from `utils.state`.
- Produces:
  - `board_for_sensor(sensor_id: int) -> str` — returns serial string (e.g. `"FT232H0"`); raises `ValueError` if unmapped.
  - `detect_cycles(group) -> list[int]` — sorted cycle numbers present in an h5 sensor group (by scanning `start_time*` keys).
  - `load_file(path: str) -> dict` — `{sensor_id: {cycle: {"start_vol": float, "stop_vol": float, "weight": float}}}`. Only sensors with ≥1 cycle.
  - `save_file(path: str, data: dict) -> None` — writes `data` back in place; `>0` creates (delete-first), `0`/falsy deletes.
  - `FIELDS = ("start_vol", "stop_vol", "weight")`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_edit_io.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_edit_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'edit_io'`

- [ ] **Step 3: Write the implementation**

```python
# edit_io.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edit_io.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Commit**

```bash
git add edit_io.py tests/test_edit_io.py
git commit -m "feat: add HDF5 I/O for weight/volume editor"
```

---

### Task 2: Solara editor UI (`edit_gui.py`)

**Files:**
- Create: `edit_gui.py`
- Reference (read for layout/styling to copy): `components/sensor_card.py:155-313`, `recording_gui.py`

**Interfaces:**
- Consumes: `edit_io.load_file`, `edit_io.save_file`, `edit_io.FIELDS`.
- Produces: a Solara `Page` component (module attribute `Page`) launched via `solara run edit_gui.py`.

Note: this task is verified by manual launch, not pytest — Solara UI rendering is not unit-tested here (the I/O it depends on is fully covered by Task 1).

- [ ] **Step 1: Write the module**

```python
# edit_gui.py
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
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "import edit_gui; print('ok')"`
Expected: prints `ok` (no import errors)

- [ ] **Step 3: Launch and manually verify**

Run: `solara run edit_gui.py`
Then in the browser:
- FileBrowser lists `Lickometry Data/`; navigate into `AEW2/`, open `raw_data_2025-05-12_14-10-36.h5`.
- Activity log shows "Loaded N sensors …"; grid renders cards for recorded sensors, "no data" for the rest.
- Change one Weight value, click SAVE, log shows "Saved corrections to …".

Expected: all of the above succeed.

- [ ] **Step 4: Confirm the save round-trips and leaves samples intact**

Run (replace `<file>` with the file edited in Step 3, `<sensor>`/`<board>` accordingly):
```bash
python -c "import edit_io; print(edit_io.load_file('<file>'))"
```
Expected: printed dict shows the new weight value.

- [ ] **Step 5: Commit**

```bash
git add edit_gui.py
git commit -m "feat: add Solara GUI for weight/volume post-hoc editing"
```

---

## Self-Review

**Spec coverage:**
- Standalone app, `solara run edit_gui.py`, no hardware deps → Task 2. ✓
- FileBrowser filtered to `raw_data*.h5` rooted at `Lickometry Data/` → Task 2 `Page`. ✓
- 24-card grid reusing SensorCard look, controls stripped → Task 2 `EditSensorCard`/`EditSensorGrid`. ✓
- All cycles editable → `load_file`/`detect_cycles` (Task 1) + per-cycle blocks (Task 2). ✓
- Save in place, delete-then-create, recorder naming, 0/blank deletes → `save_file` (Task 1). ✓
- Never touch cap_data/time_data/start_time → covered by `test_save_preserves_untouched_datasets`. ✓
- Activity log → Task 2 `Page`. ✓

**Placeholder scan:** none — all code and commands concrete.

**Type consistency:** `edit_data` shape `{sensor_id: {cycle: {field: float}}}` consistent across `load_file`, `save_file`, `set_field`, `EditSensorCard`. `FIELDS`/`FIELD_LABELS` keys match. `board_for_sensor` returns serial string used in both save and fixture builder.
