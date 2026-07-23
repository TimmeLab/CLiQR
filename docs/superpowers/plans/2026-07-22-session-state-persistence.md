# Session-State Persistence & Measurement Durability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make recording-session state survive a Solara kernel-context reset (reconnect/refresh), and persist volume/weight measurements to the HDF5 file during the run so they cannot be silently lost.

**Architecture:** Approach A from the spec. Authoritative session-critical state lives in a plain module-global `session` dict in `utils/state.py` (shared across kernel contexts, like `current_recorder` already is). All session-critical writes funnel through a `set_session` helper that updates `session` and mirrors into the existing per-context reactives. `Page` rehydrates its reactives from `session` on mount, so a reconnected context shows the live run. The recorder periodically flushes changed, `>0` measurements to the h5 (piggybacked on the cap-data flush), and the existing `>0` write guard makes a later reset non-clobbering.

**Tech Stack:** Python 3.13, Solara 1.57, h5py, pytest.

## Global Constraints

- Only `>0` measurement values are ever written; a `0`/`None` value must never overwrite an existing on-disk value. (Verbatim invariant from spec §4.)
- Session-critical state = `recording_all`, `filename`, `comments`, and the full per-sensor `SensorState` (`is_recording`, `start_time`, `elapsed_seconds`, `start_volume`, `stop_volume`, `weight`, `status`, `animal_id`, `recording_cycle`). Ephemeral UI reactives stay per-context.
- Measurement persist cadence is expressed in **elapsed seconds (300)**, checked at existing flush points — never a flush count, never a separate timer/thread.
- `SensorRecorder` must not import UI/component modules; it reads session state only through the injected `measurements_provider` callback.
- Commit messages end with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.

---

### Task 1: Authoritative `session` + `set_session` + rehydrate helper

**Files:**
- Modify: `utils/state.py` (after the `sensor_states` reactive, ~line 76)
- Test: `tests/test_session_persistence.py`

**Interfaces:**
- Produces:
  - `state.session: dict` with keys `"recording_all"`, `"filename"`, `"comments"`, `"sensor_states"`.
  - `state.set_session(key: str, value) -> None` — sets `session[key]` and mirrors into the matching reactive.
  - `state.rehydrate_reactives_from_session() -> None` — sets every mirrored reactive from `session`.
  - `state._REACTIVE_FOR: dict[str, solara.Reactive]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_session_persistence.py`:

```python
"""Session-critical state must live in a context-immune plain global and stay
mirrored into the per-context reactives, so a kernel-context reset (reconnect)
can be recovered by rehydrating from the global."""
from dataclasses import replace

from utils import state


def test_set_session_updates_both_global_and_reactive():
    state.set_session("recording_all", True)
    assert state.session["recording_all"] is True
    assert state.recording_all.value is True

    sensors = {i: state.SensorState(sensor_id=i) for i in range(1, 25)}
    sensors[3] = replace(sensors[3], start_volume=5.0, weight=22.0)
    state.set_session("sensor_states", sensors)
    assert state.session["sensor_states"][3].start_volume == 5.0
    assert state.sensor_states.value[3].start_volume == 5.0

    # Cleanup for other tests.
    state.set_session("recording_all", False)
    state.set_session("sensor_states",
                      {i: state.SensorState(sensor_id=i) for i in range(1, 25)})


def test_rehydrate_restores_reactives_from_session():
    # Simulate a run's live state in the authoritative global.
    sensors = {i: state.SensorState(sensor_id=i) for i in range(1, 25)}
    sensors[7] = replace(sensors[7], is_recording=True, start_time=123.0,
                         start_volume=6.5, weight=20.0, recording_cycle=0)
    state.set_session("sensor_states", sensors)
    state.set_session("recording_all", True)

    # Simulate a NEW kernel context: reactives back at defaults, global intact.
    state.recording_all.set(False)
    state.sensor_states.set({i: state.SensorState(sensor_id=i) for i in range(1, 25)})
    assert state.sensor_states.value[7].start_volume == 0.0  # reset

    state.rehydrate_reactives_from_session()

    assert state.recording_all.value is True
    assert state.sensor_states.value[7].is_recording is True
    assert state.sensor_states.value[7].start_volume == 6.5

    # Cleanup.
    state.set_session("recording_all", False)
    state.set_session("sensor_states",
                      {i: state.SensorState(sensor_id=i) for i in range(1, 25)})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session_persistence.py -v`
Expected: FAIL — `AttributeError: module 'utils.state' has no attribute 'session'`.

- [ ] **Step 3: Write minimal implementation**

In `utils/state.py`, immediately after the `sensor_states` reactive block (after line 76, the closing docstring of `sensor_states`), add:

```python
# ============================================================================
# Authoritative session state (context-immune)
# ============================================================================
#
# Solara stores each module-level reactive value PER kernel context
# (toestand.py KernelStore._get_dict): a reconnect/refresh gives the browser a
# fresh context whose reactives all read their defaults, while plain module
# globals (like current_recorder) are shared. So session-critical values are
# held here in a plain dict and MIRRORED into the reactives, and Page rehydrates
# the reactives from here on mount. This is what survives a mid-run reconnect.
session = {
    "recording_all": False,
    "filename": "",
    "comments": "",
    "sensor_states": {i: SensorState(sensor_id=i) for i in range(1, 25)},
}

_REACTIVE_FOR = {
    "recording_all": recording_all,
    "filename": filename,
    "comments": comments,
    "sensor_states": sensor_states,
}


def set_session(key, value):
    """Update authoritative session state and mirror it into the reactive.

    All session-critical writes must go through here so the durable global and
    the per-context reactive never diverge.
    """
    session[key] = value
    _REACTIVE_FOR[key].set(value)


def rehydrate_reactives_from_session():
    """Restore this kernel context's reactives from the authoritative global.

    Called once per Page mount. On a fresh first mount this is a no-op (global
    == defaults); after a mid-run reconnect it repopulates the new context's
    reactives with the live run.
    """
    for key, rx in _REACTIVE_FOR.items():
        rx.set(session[key])
```

Note: `comments` is defined later in the file (line ~100). Move the `comments = solara.reactive("")` definition up so it exists before `_REACTIVE_FOR`, OR reference it lazily. Simplest: relocate the `comments` reactive definition to just above this new block. Verify with `grep -n "comments = solara.reactive" utils/state.py` that only one definition remains.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_session_persistence.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add utils/state.py tests/test_session_persistence.py
git commit -m "feat: add context-immune session state + set_session/rehydrate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Route session-critical writes (and Stop's read) through `session`

**Files:**
- Modify: `components/session_controls.py` (`_reset_sensor_lifecycle` ~36-51, `start_recording` ~122-133/148, `stop_recording` ~224-318)
- Modify: `components/sensor_card.py` (`start_sensor` ~136, `stop_sensor` ~275, `update_sensor_timer` ~302, three InputFloat handlers ~432/446/460)
- Modify: `components/hardware_status.py` (`update_animal_ids_from_layout` ~78)
- Modify: `recording_gui.py` (comments `on_value` ~63)
- Test: `tests/test_session_lifecycle_reset.py` (extend)

**Interfaces:**
- Consumes: `state.set_session` (Task 1), `state.session` (Task 1).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_session_lifecycle_reset.py`:

```python
def test_reset_keeps_session_global_in_sync():
    """_reset_sensor_lifecycle must update the authoritative global, not just
    the reactive, so a reconnect during the run rehydrates correct lifecycle."""
    sensors = {i: state.SensorState(sensor_id=i) for i in range(1, 25)}
    sensors[4] = replace(sensors[4], recording_cycle=3, is_recording=True,
                         start_volume=7.0, weight=21.0)
    state.set_session("sensor_states", sensors)

    sc._reset_sensor_lifecycle()

    s_global = state.session["sensor_states"][4]
    assert s_global.recording_cycle == 0        # lifecycle reset...
    assert s_global.is_recording is False
    assert s_global.start_volume == 7.0         # ...inputs preserved...
    # ...and the global matches the reactive.
    assert state.sensor_states.value[4].recording_cycle == 0
    assert state.sensor_states.value[4].start_volume == 7.0

    state.set_session("sensor_states",
                      {i: state.SensorState(sensor_id=i) for i in range(1, 25)})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session_lifecycle_reset.py::test_reset_keeps_session_global_in_sync -v`
Expected: FAIL — `_reset_sensor_lifecycle` writes only the reactive, so `state.session["sensor_states"][4].recording_cycle == 3`.

- [ ] **Step 3: Write minimal implementation**

In `components/session_controls.py`, `_reset_sensor_lifecycle` — change the final write and the read source:

```python
def _reset_sensor_lifecycle():
    """... (unchanged docstring) ..."""
    sensors = state.session["sensor_states"].copy()
    for sid, s in sensors.items():
        sensors[sid] = replace(
            s, is_recording=False, status="idle",
            recording_cycle=0, elapsed_seconds=0, start_time=0.0,
        )
    state.set_session("sensor_states", sensors)
```

In `start_recording`, replace:
- `state.filename.set(full_path)` → `state.set_session("filename", full_path)`
- `state.recording_all.set(True)` → `state.set_session("recording_all", True)`
- `state.recording_all.set(False)` (the `mpr121_manager is None` guard) → `state.set_session("recording_all", False)`
- inside `run_recording` error path: `state.recording_all.set(False)` → `state.set_session("recording_all", False)`

In `stop_recording`:
- Change the read to the authoritative global: `sensors = state.session["sensor_states"].copy()` (was `state.sensor_states.value.copy()`, ~line 232).
- `state.sensor_states.set(sensors)` (~line 257) → `state.set_session("sensor_states", sensors)`
- `state.recording_all.set(False)` (~line 318) → `state.set_session("recording_all", False)`

In `components/sensor_card.py`, replace each of these `state.sensor_states.set(sensors)` calls with `state.set_session("sensor_states", sensors)`:
- `start_sensor` (~line 136)
- `stop_sensor` (~line 275)
- `update_sensor_timer` (~line 302)
- `set_start_vol` (~line 432)
- `set_stop_vol` (~line 446)
- `set_weight` (~line 460)

In `components/hardware_status.py`, `update_animal_ids_from_layout` (~line 78): `state.sensor_states.set(sensors)` → `state.set_session("sensor_states", sensors)`.

In `recording_gui.py`, the comments input (~line 63): change `on_value=state.comments.set` → `on_value=lambda v: state.set_session("comments", v)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_session_lifecycle_reset.py -v`
Expected: PASS (both existing and new test). The existing `test_reset_clears_cycle_but_keeps_inputs` still passes because `set_session` also updates the reactive it asserts on.

- [ ] **Step 5: Commit**

```bash
git add components/session_controls.py components/sensor_card.py components/hardware_status.py recording_gui.py tests/test_session_lifecycle_reset.py
git commit -m "feat: route session-critical writes and Stop's read through session

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Rehydrate reactives on Page mount

**Files:**
- Modify: `recording_gui.py` (`Page` component, ~line 20)

**Interfaces:**
- Consumes: `state.rehydrate_reactives_from_session` (Task 1).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_session_persistence.py`:

```python
def test_page_registers_rehydrate_and_imports_clean():
    """Page must call the rehydrate helper once on mount. We can't easily mount
    Solara here, so assert the helper is referenced in the Page module and that
    importing it does not error."""
    import inspect
    import recording_gui

    src = inspect.getsource(recording_gui.Page)
    assert "rehydrate_reactives_from_session" in src
    assert "use_effect" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session_persistence.py::test_page_registers_rehydrate_and_imports_clean -v`
Expected: FAIL — `Page` source contains no `rehydrate_reactives_from_session`.

- [ ] **Step 3: Write minimal implementation**

In `recording_gui.py`, inside `Page`, as the first statement in the function body (before the `with solara.Column(...)`):

```python
@solara.component
def Page():
    """Main application page."""

    # Restore this kernel context's reactives from the authoritative session
    # global on mount. On a fresh start this is a no-op; after a mid-run
    # reconnect it repopulates the new context so the UI shows the live run.
    solara.use_effect(state.rehydrate_reactives_from_session, [])

    with solara.Column(style={"padding": "20px", "max-width": "1800px", "margin": "0 auto"}):
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_session_persistence.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add recording_gui.py tests/test_session_persistence.py
git commit -m "feat: rehydrate reactives from session on Page mount

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Bump flush buffer and add persist-cadence constant

**Files:**
- Modify: `utils/state.py` (`HISTORY_SIZE` line 197; add `MEASUREMENT_PERSIST_SECONDS` nearby)
- Test: `tests/test_recording_config.py`

**Interfaces:**
- Produces: `state.HISTORY_SIZE == 500`, `state.MEASUREMENT_PERSIST_SECONDS == 300`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_recording_config.py`:

```python
"""Sampling roughly doubled (~112 Hz vs ~46 Hz), so the flush buffer grows to
keep flush cadence near the old ~4.5 s and HDF5 chunks a healthy size. The
measurement-persist cadence is wall-clock seconds, independent of buffer size."""
from utils import state


def test_history_size_bumped_for_doubled_rate():
    assert state.HISTORY_SIZE == 500


def test_measurement_persist_seconds_default():
    assert state.MEASUREMENT_PERSIST_SECONDS == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recording_config.py -v`
Expected: FAIL — `HISTORY_SIZE == 100`; `MEASUREMENT_PERSIST_SECONDS` undefined.

- [ ] **Step 3: Write minimal implementation**

In `utils/state.py`, line 197, change:

```python
HISTORY_SIZE = 500  # Buffer size before HDF5 write. Sized for ~112 Hz sampling
# so flushes fire ~every 4.5 s (near the old cadence) and HDF5 chunks are ~4 KB.
# Crash exposure = up to HISTORY_SIZE unflushed samples (~4.5 s), acceptable.
```

Immediately after the `MAX_SAMPLE_HZ = 150` line (line 203), add:

```python
# How often (wall-clock seconds) the recorder persists changed volume/weight
# values to the h5 during a run, checked at existing flush points. Elapsed-time
# based so it stays ~5 min regardless of HISTORY_SIZE or sample-rate changes.
MEASUREMENT_PERSIST_SECONDS = 300
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_recording_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/state.py tests/test_recording_config.py
git commit -m "feat: HISTORY_SIZE 100->500 and add MEASUREMENT_PERSIST_SECONDS

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Recorder periodic measurement persistence

**Files:**
- Modify: `recording/recorder.py` (`__init__` ~22-70, import line 16, `record_sensors` flush block ~159-167; add `_flush_measurements`)
- Modify: `components/session_controls.py` (`SensorRecorder(...)` construction ~136-140)
- Test: `tests/test_recorder_measurement_persist.py`

**Interfaces:**
- Consumes: `state.session["sensor_states"]` (Task 1), `state.MEASUREMENT_PERSIST_SECONDS` (Task 4), `state.SensorState`.
- Produces:
  - `SensorRecorder(__init__)` gains keyword `measurements_provider: Callable[[], dict] | None = None`.
  - `SensorRecorder._flush_measurements() -> None`.
  - Instance attrs `self.measurements_provider`, `self._last_persist: float`, `self._persisted: dict[int, dict[str, float]]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_recorder_measurement_persist.py`:

```python
"""The recorder persists changed, >0 measurements to the h5 during the run so a
reconnect/crash before Stop cannot lose them. The >0 guard means a later reset
(source reads 0.0) never clobbers a saved value."""
from dataclasses import replace

import h5py

from recording.recorder import SensorRecorder
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recorder_measurement_persist.py -v`
Expected: FAIL — `SensorRecorder.__init__` got an unexpected keyword `measurements_provider`.

- [ ] **Step 3: Write minimal implementation**

In `recording/recorder.py`, change the import (line 16):

```python
from utils.state import HISTORY_SIZE, NUM_CHANNELS, MAX_SAMPLE_HZ, MEASUREMENT_PERSIST_SECONDS
```

Change `__init__` signature (line 22) and add attributes (after line 37, `self._read_error_count = 0`):

```python
    def __init__(self, mpr121_manager: MPR121Manager, filename: str,
                 controllers: Dict, measurements_provider=None):
```

```python
        # Injected callback returning {sensor_id: SensorState}, read from the
        # context-immune session global. None disables measurement persistence
        # (e.g. in tests that don't exercise it). Kept as a callback so the
        # recorder never imports UI/state-write code.
        self.measurements_provider = measurements_provider
        # Wall-clock of the last measurement persist; 0.0 makes the first flush
        # persist ASAP so pre-entered volumes reach disk before any reset.
        self._last_persist = 0.0
        # Last value written per sensor per dataset name, to skip unchanged writes.
        self._persisted = {}
```

Add the elapsed-time check in `record_sensors`, right after the flush block (after `self.loop_counter += 1` at line 167 — place it just before that increment or right after the flush `elif`; put it after the flush block, before `self.loop_counter += 1`):

```python
                    # Write to HDF5 file every HISTORY_SIZE loops
                    if self.loop_counter == HISTORY_SIZE:
                        # First write - create datasets
                        self._write_initial_data()
                    elif self.loop_counter > 0 and self.loop_counter % HISTORY_SIZE == 0:
                        # Subsequent writes - append data
                        self._append_data()

                    # Periodically persist volume/weight (piggybacked on the
                    # flush cadence, gated by wall-clock so it stays ~5 min).
                    if (self.measurements_provider is not None
                            and time.monotonic() - self._last_persist
                            >= MEASUREMENT_PERSIST_SECONDS):
                        self._flush_measurements()
                        self._last_persist = time.monotonic()

                    self.loop_counter += 1
```

Add the method (place after `write_sensor_metadata`, ~after line 296):

```python
    def _flush_measurements(self):
        """Persist changed, >0 volume/weight values for started sensors.

        Reads the injected session snapshot, opens the h5 once under the lock,
        and for each sensor that has started a recording writes start_vol /
        stop_vol / weight (at the sensor's current cycle) only when the value is
        > 0 and differs from the last persisted value. The >0 guard means a
        later reset (source reads 0.0) never overwrites a saved value.
        """
        if self.measurements_provider is None:
            return
        sensor_states = self.measurements_provider()
        from utils.state import SERIAL_NUMBER_SENSOR_MAP

        with self._h5_lock, h5py.File(self.filename, "r+") as h5f:
            for sn in self.controllers.keys():
                for sensor_id in SERIAL_NUMBER_SENSOR_MAP.get(sn, []):
                    s = sensor_states.get(sensor_id)
                    if s is None:
                        continue
                    # Only sensors that have actually started recording.
                    if not (s.is_recording or s.start_time > 0):
                        continue

                    cycle = s.recording_cycle
                    suffix = "" if cycle == 0 else str(cycle)
                    fields = {
                        f"start_vol{suffix}": s.start_volume,
                        f"stop_vol{suffix}": s.stop_volume,
                        f"weight{suffix}": s.weight,
                    }
                    grp = h5f[f"board_{sn}/sensor_{sensor_id}"]
                    cache = self._persisted.setdefault(sensor_id, {})
                    for name, value in fields.items():
                        if value is None or value <= 0:
                            continue
                        if cache.get(name) == value:
                            continue
                        if name in grp:
                            del grp[name]
                        grp.create_dataset(name, data=value)
                        cache[name] = value
```

In `components/session_controls.py`, wire the provider at construction (~line 136):

```python
    current_recorder = SensorRecorder(
        mpr121_manager=mpr121_manager,
        filename=full_path,
        controllers=state.i2c_controllers.value,
        measurements_provider=lambda: state.session["sensor_states"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recorder_measurement_persist.py -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add recording/recorder.py components/session_controls.py tests/test_recorder_measurement_persist.py
git commit -m "feat: recorder persists changed >0 measurements every ~5 min

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Loud WARNING for missing measurements at stop

**Files:**
- Modify: `recording/recorder.py` (add module-level `measurement_warnings`)
- Modify: `components/session_controls.py` (`stop_recording`, after the vol/weight write loop ~270)
- Test: `tests/test_recorder_measurement_persist.py` (extend)

**Interfaces:**
- Produces: `recording.recorder.measurement_warnings(sensor_states: dict) -> list[str]`.
- Consumes: `state.session["sensor_states"]`, `state.add_log_message`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recorder_measurement_persist.py`:

```python
from recording.recorder import measurement_warnings


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recorder_measurement_persist.py::test_measurement_warnings_flags_started_missing -v`
Expected: FAIL — `cannot import name 'measurement_warnings'`.

- [ ] **Step 3: Write minimal implementation**

In `recording/recorder.py`, at module level (after the imports, before `class SensorRecorder`):

```python
def measurement_warnings(sensor_states):
    """Return one warning string per started sensor missing any measurement.

    A sensor is "started" if it is recording or has a start_time. A field is
    missing when its value is None or <= 0. Used at stop to make silent
    measurement loss visible in the activity log.
    """
    warnings = []
    for sid, s in sensor_states.items():
        if not (s.is_recording or s.start_time > 0):
            continue
        missing = [name for name, value in (
            ("start_vol", s.start_volume),
            ("stop_vol", s.stop_volume),
            ("weight", s.weight),
        ) if value is None or value <= 0]
        if missing:
            warnings.append(f"Sensor {sid}: no {', '.join(missing)} recorded")
    return warnings
```

In `components/session_controls.py`, add the import near the top (with the other `from recording...` imports):

```python
from recording.recorder import measurement_warnings
```

In `stop_recording`, immediately after the vol/weight write loop (after the `for cycle in range(...)` block, ~line 270, before "Write comments to file"):

```python
    # Surface any started sensor whose measurements never made it to disk, so a
    # silent loss (e.g. a mid-run reset that zeroed the store) is visible.
    for msg in measurement_warnings(state.session["sensor_states"]):
        state.add_log_message(f"WARNING: {msg}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recorder_measurement_persist.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Commit**

```bash
git add recording/recorder.py components/session_controls.py tests/test_recorder_measurement_persist.py
git commit -m "feat: warn at stop for started sensors missing measurements

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Full-suite regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `pytest -q`
Expected: all pass, including `tests/test_session_lifecycle_reset.py`, `tests/test_recorder_concurrency.py`, `tests/test_edit_io.py`.

- [ ] **Step 2: If any pre-existing test fails**

Investigate before proceeding — a session-critical write that still uses `state.<x>.set(...)` instead of `set_session` would leave the global stale. `grep -n "sensor_states.set\|recording_all.set\|\.comments.set\|filename.set" components/ recording_gui.py` and confirm every session-critical write goes through `set_session` (UI-only reactives are fine).

- [ ] **Step 3: Commit (only if step 2 required a fix)**

```bash
git add -A
git commit -m "fix: route remaining session-critical write through set_session

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §1 persistent session state → Tasks 1, 2, 3. ✓
- §2 measurements reach h5 during run → Tasks 4, 5. ✓
- §3 loud skip → Task 6. ✓
- §4 `>0` guard as non-clobber protection → Task 5 (`_flush_measurements` + `test_later_zero_does_not_clobber_saved_value`). ✓
- §6 HISTORY_SIZE→500 + elapsed-time cadence → Task 4, Task 5. ✓
- Testing bullets → covered across Tasks 1–6; full-suite gate in Task 7. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `set_session(key, value)`, `rehydrate_reactives_from_session()`, `_REACTIVE_FOR`, `session`, `measurements_provider`, `_flush_measurements`, `_last_persist`, `_persisted`, `measurement_warnings(sensor_states)`, `MEASUREMENT_PERSIST_SECONDS`, `HISTORY_SIZE` used consistently across tasks. ✓

**Note for implementer:** Task 1 Step 3 requires the `comments` reactive to be defined before `_REACTIVE_FOR`. Confirm `comments` is relocated above the new block and only one definition remains (`grep -n "comments = solara.reactive" utils/state.py`).
