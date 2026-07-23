# Hardware & Camera State Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make initialized hardware, the concurrent-video controls, and Start/Stop bookmark delivery to the Pi survive a browser refresh by putting their reactive state on the existing session-persistence mechanism.

**Architecture:** Reuse PR #5's pattern — an authoritative plain-global `session` dict mirrored into per-context `solara.reactive`s via `set_session()`, rehydrated on `Page` mount by `rehydrate_reactives_from_session()`. This plan only adds keys to `session`/`_REACTIVE_FOR` and reroutes existing bare `.set(...)` writes through `set_session(...)`. The rehydrate path and `recording_gui.py` are unchanged; the bookmark fix falls out for free because the bookmark gates read `camera_enabled`/`camera_sensor_id`, which now survive refresh.

**Tech Stack:** Python 3.13, Solara (reactive web GUI), pytest.

## Global Constraints

- Persistence mechanism is fixed: authoritative writes go through
  `state.set_session(key, value)`; never add a new persistence path.
- Only these keys are added to persistence — hardware: `boards_connected`,
  `i2c_controllers`; camera: `camera_enabled`, `camera_sensor_id`,
  `camera_host`, `camera_port`, `camera_video_filename`, `camera_disk_warning`,
  `camera_stall_warning`, `camera_status`.
- Do NOT persist transient UI reactives: `snapshot_image`, `snapshot_error`,
  `snapshot_pending`, `show_snapshot_dialog`, `show_test_dialog`,
  `test_plot_data`, `log_messages`.
- Default values in `session` must exactly match each reactive's current default
  in `utils/state.py`.
- Read sites (`state.<x>.value`) stay unchanged — only writes are rerouted.
- Run tests with `pytest` from the repo root (`/Users/christopher/TimmeLab/CLiQR`).

---

### Task 1: Add hardware + camera keys to the persistence mechanism

**Files:**
- Modify: `utils/state.py` (`session` dict ~lines 97-102, `_REACTIVE_FOR` map ~lines 104-109)
- Test: `tests/test_hardware_camera_persistence.py` (create)

**Interfaces:**
- Consumes: existing `state.set_session(key, value)`,
  `state.rehydrate_reactives_from_session()`, `state.session`,
  `state._REACTIVE_FOR`, and the reactives `boards_connected`,
  `i2c_controllers`, `camera_enabled`, `camera_sensor_id`, `camera_host`,
  `camera_port`, `camera_video_filename`, `camera_disk_warning`,
  `camera_stall_warning`, `camera_status` (all already defined in `state.py`).
- Produces: those ten keys become valid arguments to `set_session` and are
  restored by `rehydrate_reactives_from_session`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hardware_camera_persistence.py`:

```python
"""Hardware and camera reactives must ride the same session-persistence
mechanism as the recording lifecycle, so a browser refresh (a fresh Solara
kernel context) restores initialized hardware, the video controls, and the
bookmark-gate values instead of showing defaults."""
from utils import state


# The full set of keys this feature adds to the persistence mechanism.
NEW_PERSISTED_KEYS = [
    "boards_connected",
    "i2c_controllers",
    "camera_enabled",
    "camera_sensor_id",
    "camera_host",
    "camera_port",
    "camera_video_filename",
    "camera_disk_warning",
    "camera_stall_warning",
    "camera_status",
]


def test_new_keys_are_registered_for_persistence():
    # Every new key must exist in BOTH the authoritative global (session) and
    # the reactive-mirror map, or rehydrate would skip it on refresh.
    for key in NEW_PERSISTED_KEYS:
        assert key in state.session, f"{key} missing from session dict"
        assert key in state._REACTIVE_FOR, f"{key} missing from _REACTIVE_FOR"


def test_session_defaults_match_reactive_defaults():
    # The durable twin must start equal to the reactive's own default, so a
    # first-ever mount rehydrates to a no-op rather than changing state.
    assert state.session["boards_connected"] == {}
    assert state.session["i2c_controllers"] == {}
    assert state.session["camera_enabled"] is False
    assert state.session["camera_sensor_id"] is None
    assert state.session["camera_host"] == "picamera0.local"
    assert state.session["camera_port"] == 8770
    assert state.session["camera_video_filename"] == ""
    assert state.session["camera_disk_warning"] == ""
    assert state.session["camera_stall_warning"] == ""
    assert state.session["camera_status"] == "unknown"


def test_hardware_survives_a_simulated_refresh():
    # Simulate an initialized run: two boards connected, live controller dict.
    fake_controllers = {"FT232H0": object(), "FT232H1": object()}
    state.set_session("boards_connected", {"FT232H0": 6, "FT232H1": 6})
    state.set_session("i2c_controllers", fake_controllers)

    # Simulate a NEW kernel context: reactives reset to defaults, global intact.
    state.boards_connected.set({})
    state.i2c_controllers.set({})

    state.rehydrate_reactives_from_session()

    # The Hardware card branches on boards_connected being truthy.
    assert state.boards_connected.value == {"FT232H0": 6, "FT232H1": 6}
    # Same live controller objects re-pointed, not copies.
    assert state.i2c_controllers.value is fake_controllers

    # Cleanup so other tests see defaults.
    state.set_session("boards_connected", {})
    state.set_session("i2c_controllers", {})


def test_camera_enable_and_bookmark_sensor_survive_a_refresh():
    # The bookmark gate is: camera_enabled AND sensor_id == camera_sensor_id.
    state.set_session("camera_enabled", True)
    state.set_session("camera_sensor_id", 5)

    # New context: reactives back at defaults.
    state.camera_enabled.set(False)
    state.camera_sensor_id.set(None)

    state.rehydrate_reactives_from_session()

    assert state.camera_enabled.value is True
    assert state.camera_sensor_id.value == 5
    # Gate for sensor 5 now evaluates true again -> Start/Stop send bookmarks.
    assert state.camera_enabled.value and 5 == state.camera_sensor_id.value

    # Cleanup.
    state.set_session("camera_enabled", False)
    state.set_session("camera_sensor_id", None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hardware_camera_persistence.py -v`
Expected: FAIL — `test_new_keys_are_registered_for_persistence` asserts the
keys are missing from `state.session` / `state._REACTIVE_FOR`.

- [ ] **Step 3: Add the keys to `session` and `_REACTIVE_FOR`**

In `utils/state.py`, extend the `session` dict (currently ends at the
`sensor_states` entry) so it reads:

```python
session = {
    "recording_all": False,
    "filename": "",
    "comments": "",
    "sensor_states": {i: SensorState(sensor_id=i) for i in range(1, 25)},
    # Hardware — live handles stay open in the server process across a browser
    # refresh; these hold the same references so a fresh context can be
    # re-pointed at them without a USB re-scan.
    "boards_connected": {},
    "i2c_controllers": {},
    # Camera run-state + config (concurrent Pi video capture).
    "camera_enabled": False,
    "camera_sensor_id": None,
    "camera_host": "picamera0.local",
    "camera_port": 8770,
    "camera_video_filename": "",
    "camera_disk_warning": "",
    "camera_stall_warning": "",
    "camera_status": "unknown",
}
```

And extend `_REACTIVE_FOR` to:

```python
_REACTIVE_FOR = {
    "recording_all": recording_all,
    "filename": filename,
    "comments": comments,
    "sensor_states": sensor_states,
    # Hardware
    "boards_connected": boards_connected,
    "i2c_controllers": i2c_controllers,
    # Camera
    "camera_enabled": camera_enabled,
    "camera_sensor_id": camera_sensor_id,
    "camera_host": camera_host,
    "camera_port": camera_port,
    "camera_video_filename": camera_video_filename,
    "camera_disk_warning": camera_disk_warning,
    "camera_stall_warning": camera_stall_warning,
    "camera_status": camera_status,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hardware_camera_persistence.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `pytest -q`
Expected: PASS (existing session-persistence tests still green).

- [ ] **Step 6: Commit**

```bash
git add utils/state.py tests/test_hardware_camera_persistence.py
git commit -m "feat: persist hardware & camera reactives across refresh

Add boards_connected, i2c_controllers, and the camera run-state/config
reactives to the session dict and _REACTIVE_FOR so rehydrate restores
them on a fresh kernel context.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Route hardware writes through `set_session`

**Files:**
- Modify: `components/hardware_status.py` (`initialize_hardware` lines 26, 40, 44, 55; `disconnect_hardware` lines 116-117)
- Modify: `hardware/mock_hardware.py` (`mock_initialize` lines 247, 258)
- Test: `tests/test_hardware_camera_persistence.py` (append)

**Interfaces:**
- Consumes: `state.set_session` (Task 1), the `boards_connected` /
  `i2c_controllers` keys registered in Task 1.
- Produces: `initialize_hardware`, `disconnect_hardware`, and mock
  `mock_initialize` write hardware state through `set_session`, so a real or
  mock init survives refresh.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hardware_camera_persistence.py`:

```python
import inspect


def test_hardware_writes_go_through_set_session():
    # Guards against a future edit re-introducing a bare .set() that would
    # silently drop hardware state on refresh again. We inspect source because
    # exercising real FT232H init needs physical USB boards.
    from components import hardware_status

    for func in (hardware_status.initialize_hardware,
                 hardware_status.disconnect_hardware):
        src = inspect.getsource(func)
        assert "boards_connected.set(" not in src, (
            f"{func.__name__} still writes boards_connected with bare .set()")
        assert "i2c_controllers.set(" not in src, (
            f"{func.__name__} still writes i2c_controllers with bare .set()")


def test_mock_initialize_goes_through_set_session():
    from hardware import mock_hardware

    src = inspect.getsource(mock_hardware)
    # mock_initialize is defined inside a setup function; check the module text.
    assert "i2c_controllers.set(" not in src, (
        "mock_initialize still writes i2c_controllers with bare .set()")
    assert "boards_connected.set(" not in src, (
        "mock_initialize still writes boards_connected with bare .set()")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hardware_camera_persistence.py -k hardware_writes -v`
Expected: FAIL — bare `boards_connected.set(` / `i2c_controllers.set(` still
present.

- [ ] **Step 3: Reroute writes in `components/hardware_status.py`**

Replace each bare hardware write:

- Line 26: `state.boards_connected.set({})` →
  `state.set_session("boards_connected", {})`
- Line 40: `state.boards_connected.set({})` →
  `state.set_session("boards_connected", {})`
- Line 44: `state.i2c_controllers.set(controllers)` →
  `state.set_session("i2c_controllers", controllers)`
- Line 55: `state.boards_connected.set(board_info)` →
  `state.set_session("boards_connected", board_info)`
- Line 116 (inside `disconnect_hardware`): `state.boards_connected.set({})` →
  `state.set_session("boards_connected", {})`
- Line 117: `state.i2c_controllers.set({})` →
  `state.set_session("i2c_controllers", {})`

- [ ] **Step 4: Reroute writes in `hardware/mock_hardware.py`**

- Line 247: `state.i2c_controllers.set(controllers)` →
  `state.set_session("i2c_controllers", controllers)`
- Line 258: `state.boards_connected.set(board_info)` →
  `state.set_session("boards_connected", board_info)`

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hardware_camera_persistence.py -k "hardware_writes or mock_initialize" -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add components/hardware_status.py hardware/mock_hardware.py tests/test_hardware_camera_persistence.py
git commit -m "feat: route hardware writes through set_session

initialize_hardware, disconnect_hardware, and the mock initializer now
persist boards_connected/i2c_controllers so initialized hardware shows
after a browser refresh instead of the Initialize button.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Route camera writes through `set_session`

**Files:**
- Modify: `components/camera_controls.py` (`test_connection` line 53; `CameraControlsCard` `on_value` handlers lines 64, 77, 81, 88)
- Modify: `components/session_controls.py` (camera writes lines 64, 65, 66, 77, 180, 214)
- Test: `tests/test_hardware_camera_persistence.py` (append)

**Interfaces:**
- Consumes: `state.set_session` (Task 1), the camera keys registered in Task 1.
- Produces: every camera state write goes through `set_session`, so the video
  controls' enabled/sensor/host/port/status and the disk/stall/video-filename
  values survive refresh; the bookmark gate reads restored values.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hardware_camera_persistence.py`:

```python
def test_camera_writes_go_through_set_session():
    # Same source-inspection guard as the hardware writes: these on_value
    # handlers and status writes must persist, or the video button and bookmark
    # gate reset on refresh.
    from components import camera_controls
    from components import session_controls

    camera_src = inspect.getsource(camera_controls)
    # The Switch/Select/InputText handlers must not bind the bare reactive setter.
    assert "on_value=state.camera_enabled.set" not in camera_src
    assert "on_value=state.camera_sensor_id.set" not in camera_src
    assert "on_value=state.camera_host.set" not in camera_src
    assert "on_value=state.camera_port.set" not in camera_src
    assert "state.camera_status.set(" not in camera_src

    session_src = inspect.getsource(session_controls)
    assert "camera_video_filename.set(" not in session_src
    assert "camera_disk_warning.set(" not in session_src
    assert "camera_stall_warning.set(" not in session_src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hardware_camera_persistence.py -k camera_writes -v`
Expected: FAIL — bare camera setters still present.

- [ ] **Step 3: Reroute writes in `components/camera_controls.py`**

- Line 53: `state.camera_status.set("connected" if ok else "disconnected")` →
  `state.set_session("camera_status", "connected" if ok else "disconnected")`
- Line 64: `on_value=state.camera_enabled.set` →
  `on_value=lambda v: state.set_session("camera_enabled", v)`
- Line 77: `on_value=state.camera_host.set` →
  `on_value=lambda v: state.set_session("camera_host", v)`
- Line 81: `on_value=state.camera_port.set` →
  `on_value=lambda v: state.set_session("camera_port", v)`
- Line 88: `on_value=state.camera_sensor_id.set` →
  `on_value=lambda v: state.set_session("camera_sensor_id", v)`

- [ ] **Step 4: Reroute writes in `components/session_controls.py`**

- Line 64: `state.camera_video_filename.set("")` →
  `state.set_session("camera_video_filename", "")`
- Line 65: `state.camera_disk_warning.set("")` →
  `state.set_session("camera_disk_warning", "")`
- Line 66: `state.camera_stall_warning.set("")` →
  `state.set_session("camera_stall_warning", "")`
- Line 77: `state.camera_video_filename.set(resp.get("video_filename", ""))` →
  `state.set_session("camera_video_filename", resp.get("video_filename", ""))`
- Line 180: `state.camera_disk_warning.set(message)` →
  `state.set_session("camera_disk_warning", message)`
- Line 214: the `state.camera_stall_warning.set(` call (spanning to its closing
  paren) → change the call to
  `state.set_session("camera_stall_warning", <the same string argument>)`.
  Preserve the existing argument expression exactly; only the call form changes.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hardware_camera_persistence.py -k camera_writes -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: PASS — including `tests/test_camera_state.py` and
`tests/test_camera_session_lifecycle.py` (reads unchanged; writes now go
through `set_session`, which still calls the reactive setter).

- [ ] **Step 7: Commit**

```bash
git add components/camera_controls.py components/session_controls.py tests/test_hardware_camera_persistence.py
git commit -m "feat: route camera writes through set_session

Camera enable/sensor/host/port/status and disk/stall/video-filename
writes now persist, so the concurrent-video controls and Start/Stop
bookmark delivery to the Pi survive a browser refresh.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Manual verification (after all tasks)

Not automatable in pytest (needs a running GUI); do once by hand in mock mode:

1. `solara run recording_gui_mock.py`, open `http://localhost:8765`.
2. Click **Initialize Hardware** → boards appear.
3. Enable **concurrent video**, pick a bookmark sensor, start a session, Start a
   sensor.
4. Refresh the browser.
5. Confirm: Hardware card still shows connected boards (not the Initialize
   button); the video controls are still enabled with the same sensor selected;
   pressing Stop still logs a bookmark send to the (mock) Pi.
