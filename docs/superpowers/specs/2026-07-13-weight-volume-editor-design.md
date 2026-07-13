# Weight/Volume Post-Hoc Editor — Design

**Date:** 2026-07-13
**Status:** Approved (design), pending spec review

## Purpose

Fix data-entry errors in weight/volume measurements after a recording session.
During recording, operators type start volume, stop volume, and animal weight into
the Solara recording GUI; these are saved into the session's `raw_data*.h5`. Photos of
the scale/graduated cylinders are taken for later comparison. When a typo is found, this
standalone GUI lets the operator reopen the file, correct the values against the photos,
and save the corrections back into the file.

## Scope

- One new file: `edit_gui.py` at repo root.
- Run with `solara run edit_gui.py`.
- No hardware dependencies (no `ft232h`, `mpr121`, `recorder` imports).
- No changes to existing recording app.

Out of scope (YAGNI): editing animal_id (not stored in h5), editing start/stop
timestamps, editing comments, writing to a new copy, any hardware interaction,
editing `cap_data`/`time_data`.

## Data Model (existing HDF5 layout)

File structure written by `recording/recorder.py`:

```
raw_data_YYYY-MM-DD_HH-MM-SS.h5
├── comments                       (dataset, optional)
└── board_{serial}/                (one per connected FT232H board)
    └── sensor_{id}/               (id 1–24)
        ├── time_data              (recording samples — untouched)
        ├── cap_data               (recording samples — untouched)
        ├── start_time             (cycle 0)
        ├── stop_time
        ├── start_vol              (mL, only present if > 0)
        ├── stop_vol
        ├── weight                 (g)
        ├── start_time1 / stop_time1 / start_vol1 / stop_vol1 / weight1   (cycle 1)
        └── ... {c} suffix for cycle c ...
```

- Cycle 0 datasets have **no** numeric suffix; cycle `c ≥ 1` uses suffix `c`.
- Volume/weight datasets exist only when the operator entered a value `> 0`.
  Real measurements are never exactly 0, so absence of a dataset == "not recorded".
- Board for a given sensor comes from `SERIAL_NUMBER_SENSOR_MAP` in `utils/state.py`
  (a hardware-free import).

## Architecture

Single-file Solara app, `edit_gui.py`. Three concerns:

### 1. State

Module-level reactives (mirroring the `utils/state.py` pattern, but local to this file):

- `selected_file: reactive[str]` — absolute path of the chosen h5, `""` when none.
- `edit_data: reactive[dict]` — loaded/edited values, shape:
  `{ sensor_id: { cycle: {"start_vol": float, "stop_vol": float, "weight": float} } }`
  Only sensors present in the file with ≥1 detected cycle appear.
- `log_messages: reactive[list[str]]` — timestamped status/confirmation lines.

### 2. Load (`load_file(path)`)

1. Open `h5py.File(path, "r")`.
2. Walk every `board_*` group, then every `sensor_*` subgroup → parse `sensor_id`.
3. Detect cycles: cycle 0 present iff `start_time` in group; cycle `c` present iff
   `start_time{c}` in group. Collect the sorted set of present cycles.
4. For each present cycle read `start_vol{c}`, `stop_vol{c}`, `weight{c}`; missing → `0.0`.
5. Build the nested dict and assign to `edit_data`; set `selected_file`; log
   "Loaded N sensors, M cycles from <name>".
6. On failure (bad file, unreadable): log an error, leave `edit_data` empty.

Sensors with a group but no `start_time*` (never recorded) are skipped — no card.

### 3. Save (`save_file()`)

Open `h5py.File(selected_file, "r+")`. For each sensor/cycle/field in `edit_data`:

- Resolve dataset name: field name for cycle 0 (`start_vol`), field+suffix for
  cycle ≥ 1 (`start_vol1`).
- Resolve board path via `SERIAL_NUMBER_SENSOR_MAP` → `board_{sn}/sensor_{id}`.
- If value `> 0`: delete existing dataset if present, then `create_dataset(name, data=value)`.
- If value `== 0` (blank/cleared): delete existing dataset if present, else no-op.

This matches the recorder's delete-then-write idiom exactly, so files stay
format-compatible with `data_analysis.py`. Log "Saved corrections to <name>" on success;
log an error and abort (no partial guarantee needed — single-user, single-file) on failure.

## UI

Reuses the visual language of `components/sensor_card.py` (`SensorGrid` 4×6 shelf layout,
card sizing/border styling) with recording controls removed.

- **Header row:** title "CLiQR — Weight/Volume Editor".
- **File picker:** `solara.FileBrowser` rooted at `Lickometry Data/`, filtered to
  files matching `raw_data*.h5`. Selecting a file calls `load_file`.
- **Grid** (shown only when `edit_data` non-empty): 4 shelves × 6 positions, matching the
  physical rack, iterating sensor ids 1–24. A card renders only if its sensor id is in
  `edit_data`; empty positions render a muted placeholder ("Sensor N — no data").
- **Card** (`EditSensorCard(sensor_id)`):
  - Header: `Sensor {id}`.
  - One block per detected cycle, labeled `Cycle {c}` (only shown when the sensor has
    >1 cycle; single-cycle cards omit the label to stay clean).
  - Each block: three `solara.InputFloat` — "Start Vol (mL)", "Stop Vol (mL)", "Weight (g)"
    — bound to `edit_data[sensor_id][cycle][field]` via an on_value handler that copies
    the dict, updates the leaf, and re-sets the reactive (dataclass-free nested-dict copy).
- **Save button:** disabled until a file is loaded; on click calls `save_file`.
- **Activity log:** scrollable list of `log_messages` (same style as recording GUI).

## Error Handling

- No file selected → Save disabled; grid hidden.
- Load of a non-h5 / corrupt file → caught, logged as error, state left empty.
- Save I/O error → caught, logged as error; the on-disk file keeps whatever h5py already
  flushed (acceptable: single operator, they can reload and retry against the photo).

## Testing

- Manual: launch `solara run edit_gui.py`, load a known single-cycle file
  (e.g. `Lickometry Data/AEW2/raw_data_2025-05-12_14-10-36.h5`), change a weight, save,
  reopen in a Python shell / `data_analysis.py`, confirm the new value and that
  `cap_data`/`time_data` are untouched.
- Multi-cycle: load a file with a `weight1` present, confirm both cycles render and edit.
- Round-trip: set a field to 0, save, confirm the dataset is deleted; set it back, confirm
  recreated.
- Automated (if feasible without a display): unit-test `load_file`/`save_file` against a
  synthetically built h5 fixture, asserting the read dict and the written datasets. UI
  rendering left to manual check.
