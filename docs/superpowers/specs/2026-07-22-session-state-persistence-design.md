# Session-State Persistence & Measurement Durability

**Date:** 2026-07-22
**Status:** Approved design

## Problem

The `raw_data_2026-07-22_11-43-05.h5` run recorded capacitance for all 24
sensors but wrote **no** `stop_time`, `start_vol`, `stop_vol`, or `weight` for
any sensor. Volumes/weights had been entered before the run. Opening the file in
`edit_gui.py` shows every measurement as `0.0`.

### Root cause (verified)

Solara stores each module-level `solara.reactive` value **per kernel context**.
Confirmed in `solara/toestand.py`:

- `KernelStore._get_dict()` (lines 283–296) stores a reactive's value in
  `context.user_dicts` keyed by `context.id` when a kernel context is current.
- `KernelStore.get()` (301–308) initializes a missing key to `initial_value()`
  — the default.

So when Solara hands the browser session a **new kernel context** mid-run (a
websocket reconnect / kernel refresh, which can happen without a user-visible
disconnect — the vivarium networking is flaky), every reactive reads its default
again. Meanwhile `current_recorder` and the recording task are **plain module
globals**, shared across contexts, so acquisition keeps sampling in the old
context. Result: continuous capacitance data, but a Stop handler that reads a
freshly-defaulted `sensor_states` — `is_recording=False`, `recording_cycle=0`,
volumes `0.0`.

Proof from the file: `stop_recording`'s first loop writes `stop_time` only for
sensors with `is_recording=True` (read server-side, browser not in the path).
No sensor got a `stop_time`, so the store it read had `is_recording=False` for
all started sensors. `recording_cycle=0` then makes the vol/weight loop's
`range(recording_cycle)` empty; `start_volume=0.0` also fails the `>0` write
guard. Both independently skip the write.

This is a latent bug, not a regression from the recent sampling-rate config
changes. Earlier runs worked because no context refresh happened during them.

## Goals

1. **Session-critical state survives a refresh/reconnect.** Recording lifecycle
   (`recording_all`, `filename`, per-sensor `is_recording` / `start_time` /
   `recording_cycle` / `elapsed_seconds` / `status`), measurements
   (`start_volume` / `stop_volume` / `weight`), `animal_id`, and `comments`.
2. **Measurements reach the h5 during the run** (defense-in-depth), so a mid-run
   crash before Stop does not lose them.
3. **Silent measurement-skip becomes loud.**

Out of scope: ephemeral UI reactives (log messages, camera status, dialog
toggles, snapshot state) stay per-context. Multi-browser concurrent use is not
supported and not a requirement.

## Approach A — plain-global authoritative state + reactive mirror + rehydrate

Chosen over (B) replacing reactives with a `session_rev` counter — touches every
component, fights Solara's model — and (C) forcing reactives into Solara's global
storage scope — relies on `toestand.py` internals and breaks per-context render
notification. A is the least magical and, given single-browser use, rehydrate on
mount fully covers reconnect.

### 1. Authoritative `session` (utils/state.py)

Plain module-global, shared across kernel contexts, never reset by a reconnect:

```python
session = {
    "recording_all": False,
    "filename": "",
    "comments": "",
    "sensor_states": {i: SensorState(sensor_id=i) for i in range(1, 25)},
}
```

Same shape and values as today's reactives — the durable twin.
`recording_task` / `current_recorder` are unchanged (a live asyncio task cannot
be rehydrated).

### 2. Write boundary — one helper

```python
_REACTIVE_FOR = {
    "recording_all": recording_all,
    "filename": filename,
    "comments": comments,
    "sensor_states": sensor_states,
}

def set_session(key, value):
    session[key] = value            # authoritative, context-immune
    _REACTIVE_FOR[key].set(value)   # mirror into current context -> re-render
```

Replace the session-critical `state.<x>.set(...)` call sites with
`state.set_session(...)`:

- `components/session_controls.py`: `_reset_sensor_lifecycle` (still preserves
  volumes via `replace`), `start_recording` (`recording_all`, `filename`),
  `run_recording` / error path (`recording_all`), `stop_recording`
  (`sensor_states`, `recording_all`).
- `components/sensor_card.py`: `start_sensor`, `stop_sensor`,
  `update_sensor_timer`, the three InputFloat `on_value` handlers
  (`set_start_vol` / `set_stop_vol` / `set_weight`).
- `components/hardware_status.py`: `update_animal_ids_from_layout`.
- `recording_gui.py`: comments `on_value`.

Components keep reading `state.<x>.value` — subscriptions unchanged.

### 3. Rehydrate on (re)connect

In `Page`, run once per context mount:

```python
def _rehydrate():
    for key, rx in state._REACTIVE_FOR.items():
        rx.set(state.session[key])
solara.use_effect(_rehydrate, [])
```

First-ever mount: `session` == defaults → no-op. Mount after a mid-run
reconnect: `session` holds live values → this context's reactives are restored
and the UI shows the running session. One brief render with defaults before the
effect fires — acceptable flash.

### 4. Recorder — periodic measurement persistence

- `SensorRecorder` gains a `measurements_provider` callback, wired at
  construction to `lambda: state.session["sensor_states"]` (context-immune
  source; keeps the recorder decoupled from `state`).
- Piggyback the existing cap-data flush. At each flush point, check elapsed
  wall-clock; every ~5 min call `_flush_measurements()`:

  ```python
  if time.monotonic() - self._last_persist >= MEASUREMENT_PERSIST_SECONDS:  # 300
      self._flush_measurements()
      self._last_persist = time.monotonic()
  ```

  Not a timer/thread — a cheap compare at the flush already performed. Stays
  ~5 min regardless of future `HISTORY_SIZE` or sample-rate changes.
- `_flush_measurements()` opens the h5 **once** under `_h5_lock` and, for each
  started sensor, writes `start_vol` / `stop_vol` / `weight` at the sensor's
  current `recording_cycle`, reusing the existing `>0` guard and del+recreate.
  Tracks last-persisted values per sensor; writes only **changed, >0** fields.

**The `>0` guard is the safety net.** Once a value is on disk, a later reset
reads `0.0` → skipped → saved data is never clobbered. As long as one persist
lands before a reset, the measurement is safe.

### 5. Loud failure at stop

In `stop_recording`, for each started sensor whose `start_volume` /
`stop_volume` / `weight` is `<= 0` at stop, log a WARNING naming the sensor and
the missing fields. Turns silent loss into a visible message.

### 6. Flush-size tuning (related, in scope)

Sampling roughly doubled (≈112 Hz vs the old ≈46 Hz), so flushes doubled in
frequency (`HISTORY_SIZE=100` → every ~0.9 s), increasing event-loop stalls and
leaving HDF5 chunks at 800 B (inefficient below ~8 KB).

- **`HISTORY_SIZE`: 100 → 500.** Flush every ~4.5 s (near the old cadence);
  chunk size 4 KB. Crash exposure rises to ~4.5 s of unflushed capacitance —
  negligible.
- Persist cadence expressed in **elapsed seconds (300)**, not a flush count, so
  it is unaffected by the `HISTORY_SIZE` change (a plain count would have made
  "5 min" drift to ~37 min at 500).

## Testing

- `set_session` updates both `session` and the mirrored reactive.
- Rehydrate: reset reactives to defaults (simulate a new context), call
  `_rehydrate`, assert each reactive restored from `session`.
- `_flush_measurements`: writes `>0` changed values; skips `0`; does **not**
  clobber an existing on-disk value when the source later reads `0`
  (reset-protection invariant); writes to the sensor's current cycle.
- Stop-time WARNING fires when a started sensor's measurements are `<= 0`.
- Existing suite stays green — especially `tests/test_session_lifecycle_reset.py`
  (reset now routes through `set_session`).

## Files touched

- `utils/state.py` — `session`, `set_session`, `_REACTIVE_FOR`, `HISTORY_SIZE`,
  `MEASUREMENT_PERSIST_SECONDS`.
- `recording/recorder.py` — `measurements_provider`, `_flush_measurements`,
  `_last_persist`, elapsed-time check in the flush path.
- `components/session_controls.py`, `components/sensor_card.py`,
  `components/hardware_status.py`, `recording_gui.py` — route session-critical
  writes through `set_session`; add `_rehydrate` effect in `Page`; stop-time
  WARNING; wire `measurements_provider`.
- `tests/` — new coverage per above.
