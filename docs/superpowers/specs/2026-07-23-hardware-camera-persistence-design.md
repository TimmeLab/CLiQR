# Hardware & Camera State Persistence Across Refresh

**Date:** 2026-07-23
**Status:** Approved design

## Problem

The 2026-07-22 session-state-persistence work (PR #5) made the recording
lifecycle and measurements survive a browser refresh / websocket reconnect. But
three other pieces of state still reset to their defaults on refresh:

1. **Initialized hardware disappears.** After a refresh the Hardware Status card
   shows the "Initialize Hardware" button again, as if no boards were connected.
2. **The concurrent-video button is no longer enabled.** The camera controls
   reset, so the video feature looks disabled even though the Pi is still
   recording.
3. **Start/Stop bookmark requests stop reaching the Pi.** After a refresh,
   pressing a sensor's Start/Stop no longer sends a bookmark to the Pi camera
   server, so those events are missing from the video.

All three are the same class of bug as the one PR #5 fixed, in state that was
not migrated onto the persistence mechanism.

### Root cause (verified)

Solara stores each module-level `solara.reactive` value **per kernel context**
(`solara/toestand.py`, `KernelStore._get_dict`). A refresh/reconnect hands the
browser a fresh context whose reactives all read their defaults. PR #5 solved
this for a chosen set of keys by holding an authoritative copy in a plain
module-global `session` dict (context-immune), mirroring writes into the
reactives via `set_session()`, and rehydrating the reactives from `session` on
every `Page` mount (`rehydrate_reactives_from_session`, called from a
`use_effect` in `recording_gui.py`).

The three broken pieces were never added to that mechanism:

- **Hardware** — `boards_connected` and `i2c_controllers`
  (`utils/state.py:44,47`) are written with bare `state.<x>.set(...)` in
  `components/hardware_status.py` (`initialize_hardware`, `disconnect_hardware`).
  Not in `session`, not in `_REACTIVE_FOR`, so `rehydrate` never restores them →
  a fresh context reads `{}` → the card falls into its "no boards" branch
  (`hardware_status.py:86`) and shows Initialize.
- **Camera** — `camera_enabled`, `camera_sensor_id`, `camera_host`,
  `camera_port`, `camera_video_filename`, `camera_disk_warning`,
  `camera_stall_warning`, `camera_status` are all written with bare `.set(...)`
  and are absent from the persistence mechanism → a fresh context reads defaults
  (`camera_enabled=False`, `camera_sensor_id=None`).
- **Bookmarks** — the Start/Stop bookmark calls gate on
  `state.camera_enabled.value and sensor_id == state.camera_sensor_id.value`
  (`components/sensor_card.py:149,203`) and `state.camera_enabled.value`
  (`components/session_controls.py:154,293`). After a refresh those read
  `False` / `None`, so the gate is never satisfied and no bookmark is sent.
  **Fixing camera persistence fixes bookmarks** — no separate change needed.

**Why the live hardware is still usable after a refresh:** the actual FT232H
I2C controller/port objects and `mpr121_manager` are held in **plain module
globals** (`ft232h_manager`, `mpr121_manager` in `hardware_status.py`, and the
same controller objects referenced inside the `i2c_controllers` dict). A browser
refresh does not touch the server process, so those USB handles stay open. Only
the per-context *reactive* pointing at them was lost. Re-mirroring the same live
references is therefore valid and requires **no USB re-scan** — which is also
what makes it safe to do during an active recording (a re-scan would re-open the
USB devices out from under the running acquisition loop).

## Goals

1. After a refresh, the Hardware Status card shows the connected boards, not the
   Initialize button.
2. After a refresh, the concurrent-video controls show their pre-refresh state
   (enabled/disabled, selected bookmark sensor, host/port).
3. After a refresh, Start/Stop still send bookmark requests to the Pi.

**Out of scope (stays per-context, transient UI — YAGNI):** `snapshot_image`,
`snapshot_error`, `snapshot_pending`, `show_snapshot_dialog`,
`show_test_dialog`, `test_plot_data`, `log_messages`.

## Approach

Reuse PR #5's mechanism exactly — add the missing keys to the authoritative
`session` dict and the `_REACTIVE_FOR` map, then route their writes through
`set_session()`. `rehydrate_reactives_from_session()` already loops over every
`_REACTIVE_FOR` key, so the new keys are restored on mount with no change to the
rehydrate path or to `recording_gui.py`.

This was chosen over any alternative (e.g. a separate hardware-only persistence
path) because a second mechanism would duplicate logic and diverge; the existing
one already covers exactly this failure mode.

### 1. Extend authoritative state (`utils/state.py`)

Add to the `session` dict (defaults match today's reactive defaults):

```python
session = {
    # ... existing keys ...
    # Hardware
    "boards_connected": {},
    "i2c_controllers": {},
    # Camera (run-state + config)
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

Add the matching entries to `_REACTIVE_FOR`:

```python
_REACTIVE_FOR = {
    # ... existing keys ...
    "boards_connected": boards_connected,
    "i2c_controllers": i2c_controllers,
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

`set_session` and `rehydrate_reactives_from_session` need **no** change — they
already iterate `_REACTIVE_FOR`.

**Note on `i2c_controllers`:** storing the live controller dict in `session` is
storing a reference, not serializing hardware. The same objects are already
process-global via `mpr121_manager`; `session` just holds another reference so a
fresh context can be re-pointed at them.

### 2. Route hardware writes through `set_session`
(`components/hardware_status.py`)

- `initialize_hardware`: the two error-path `state.boards_connected.set({})`
  (lines 26, 40), the success `state.i2c_controllers.set(controllers)` (44), and
  `state.boards_connected.set(board_info)` (55) → `state.set_session(...)`.
- `disconnect_hardware`: `state.boards_connected.set({})` and
  `state.i2c_controllers.set({})` (116, 117) → `state.set_session(...)`.

Read sites (`boards_connected.value`, `i2c_controllers.value`) are unchanged —
subscriptions still work.

`hardware/mock_hardware.py`'s `mock_initialize` (which overrides
`initialize_hardware`) sets `state.i2c_controllers` and `state.boards_connected`
(lines 247, 258) — route those through `set_session` too, so refresh
persistence can be exercised in mock mode (`recording_gui_mock.py`).

### 3. Route camera writes through `set_session`

- `components/camera_controls.py`:
  - `on_value=state.camera_enabled.set` (64) →
    `on_value=lambda v: state.set_session("camera_enabled", v)`.
  - `on_value=state.camera_host.set` (77) → `set_session("camera_host", v)`.
  - `on_value=state.camera_port.set` (81) → `set_session("camera_port", v)`.
  - `on_value=state.camera_sensor_id.set` (88) →
    `set_session("camera_sensor_id", v)`.
  - `state.camera_status.set(...)` (53) → `set_session("camera_status", ...)`.
- `components/session_controls.py`:
  - `camera_video_filename.set("")` / `.set(resp.get(...))` (64, 77) →
    `set_session`.
  - `camera_disk_warning.set(...)` (65, 180) → `set_session`.
  - `camera_stall_warning.set(...)` (66, 214) → `set_session`.

### 4. Rehydrate — no change

`rehydrate_reactives_from_session()` already restores every `_REACTIVE_FOR`
key; the `use_effect(state.rehydrate_reactives_from_session, [])` in
`recording_gui.py` is untouched. On a refresh the fresh context now repopulates
hardware + camera reactives from `session`, so the Hardware card shows boards,
the video controls show their state, and the bookmark gates read live values.

## Testing

Follow PR #5's test pattern (`tests/` session-state coverage). For each newly
persisted key:

- `set_session(key, value)` updates both `session[key]` and the mirrored
  reactive.
- Rehydrate: set the reactive to a wrong/default value (simulating a fresh
  context), call `rehydrate_reactives_from_session()`, assert the reactive is
  restored from `session`.
- Specifically assert the user-visible outcomes hold after rehydrate:
  - `boards_connected` non-empty → Hardware card is in its "connected" branch
    (test at the state level: `boards_connected.value` truthy).
  - `camera_enabled=True` and `camera_sensor_id=N` survive rehydrate, so the
    bookmark gate `camera_enabled.value and sensor_id == camera_sensor_id.value`
    evaluates true again.
- Existing suite stays green (writes now route through `set_session`).

## Files touched

- `utils/state.py` — add hardware + camera keys to `session` and
  `_REACTIVE_FOR`.
- `components/hardware_status.py` — route `boards_connected` / `i2c_controllers`
  writes through `set_session`.
- `hardware/mock_hardware.py` — route `mock_initialize`'s `boards_connected` /
  `i2c_controllers` writes through `set_session`.
- `components/camera_controls.py` — route `camera_enabled` / `camera_sensor_id`
  / `camera_host` / `camera_port` / `camera_status` writes through `set_session`.
- `components/session_controls.py` — route `camera_video_filename` /
  `camera_disk_warning` / `camera_stall_warning` writes through `set_session`.
- `tests/` — new coverage per above.
