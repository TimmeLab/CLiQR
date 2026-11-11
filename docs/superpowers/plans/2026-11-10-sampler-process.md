# Sampler Process Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** All FT232H/MPR121 USB I/O (init + sampling) runs in a separate hardware process with its own GIL, so Solara UI work can no longer stall sampling; the process survives browser refresh.

**Architecture:** A Solara-free hardware layer (`utils/hw_constants.py`, backends, a shared `run_sampling_loop`) runs inside a `spawn`ed child (`hardware/hw_process.py`). The GUI-side `HardwareClient` (`hardware/hw_client.py`) owns the process and exposes a `ProcessSource`; `SensorRecorder` consumes any *source* (in-process `ThreadSource` for unit tests, `ProcessSource` in the GUI) and keeps its existing buffering, writer thread and full flush on stop. The activity log moves into the context-free session store.

**Tech Stack:** Python 3.13, `multiprocessing` (spawn context, Pipe + Queue), `threading`, pyftdi, h5py, Solara 1.57, pytest.

**Spec:** `docs/superpowers/specs/2026-11-10-sampler-process-design.md`

## Global Constraints

- No change to h5 layout, dataset dtypes/chunking, timestamps (`time.time()` per channel at the read), `MAX_SAMPLE_HZ` (150), `HISTORY_SIZE` (500), channel/board mapping, MPR121 register values, camera/bookmark code, analysis code.
- The child's import graph must not include `solara`, `pandas`, `utils.state`, or any `components.*` module.
- Start method is always `multiprocessing.get_context("spawn")` (Windows rig; macOS default too).
- Never silently drop or duplicate a sample; failures logged with `ERROR:` prefix.
- The hardware process never outlives the GUI server and is never started twice.
- `SensorRecorder.start(log_callback=None, on_error=None)`, `stop(timeout=10.0) -> bool`, `get_recent_samples(...)` keep their current signatures and semantics.

## Review Focus

1. GUI server killed hard (Task Manager / Ctrl-C twice) → child notices its command pipe closed and exits, releasing USB. (Task 2 `test_child_exits_when_client_pipe_closes`)
2. Initialize clicked again, or a refresh, while the process is alive → same process reused, no second USB open. (Task 3 `test_initialize_twice_reuses_process`)
3. Start → Stop → Start again in one GUI launch (second session) → second recording complete and in a new file; process still alive. (Task 3 `test_two_sessions_one_process`)
4. Hardware process crashes mid-run → ERROR logged, data received so far on disk, `recording_all` cleared, writer ended. (Task 2 `test_child_crash_reports_fatal_and_keeps_data`)
5. Background-thread log line while the original browser context is gone → visible after refresh, and never raises in the logging thread. (Task 4 `test_log_written_from_thread_survives_refresh`)

---

### Task 1: Solara-free hardware layer + source-based recorder

**Files:**
- Create: `utils/hw_constants.py` — move the block `utils/state.py:220-311` (rack configs, `RACK_DESIGN`, `SERIAL_NUMBER_SENSOR_MAP`, MPR121 registers/values, `HISTORY_SIZE`, `MAX_SAMPLE_HZ`, `MEASUREMENT_PERSIST_SECONDS`, `READ_RETRIES`, `READ_RETRY_DELAY`, `ACTIVE_CHANNELS*`, `NUM_CHANNELS`, `FLUIDS`) here verbatim, comments included.
- Modify: `utils/state.py` — replace that block with `from utils.hw_constants import *  # noqa` plus an explicit `__all__`-free re-export so `utils.state.X` keeps working (tests monkeypatch `utils.state.SERIAL_NUMBER_SENSOR_MAP`).
- Modify: `hardware/mpr121.py`, `hardware/mock_hardware.py` — import constants from `utils.hw_constants`; `read_sensor_data(serial_number, channels=None)`: when `channels is None`, lazily `from utils.state import active_channels` (legacy path). Remove `get_last_reading` from both (moved in Task 3). `mock_hardware.use_mock_hardware` untouched until Task 3.
- Create: `hardware/sampling.py`, `hardware/backends.py`, `hardware/fake_backends.py`, `recording/sources.py`
- Modify: `recording/recorder.py`
- Test: `tests/test_sampling_loop.py` (new); existing recorder tests must pass unchanged.

**Interfaces:**
- Produces:
  - `hardware.sampling.run_sampling_loop(read: Callable[[str], tuple[list[float], list[int]]], serials: list[str], hz: float, stop: threading.Event, on_loop: Callable[[list[tuple[str, list[float], list[int]]]], None], on_read_error: Callable[[str, str], None]) -> None` — the existing pacing (running deadline, resync when behind), parallel reads on a `ThreadPoolExecutor(max_workers=len(serials))`, a failed board read → `on_read_error(sn, str(exc))` and that board omitted from this loop; `on_loop(results)` once per loop with `(sn, times, caps)` for the boards that succeeded. Returns when `stop` is set; exceptions from `on_loop` propagate.
  - `hardware.backends.Backend` protocol: `init() -> tuple[dict[str, int], list[str], list[str]]` (board_info sn→num_sensors, log messages, errors), `read(sn: str, channels: list[int]) -> tuple[list[float], list[int]]`, `close() -> None`. Classes `RealBackend` (FT232HManager + MPR121Manager, same steps/log text as today's `initialize_hardware`) and `MockBackend` (Mock managers). Constructed with no args.
  - `hardware.fake_backends`: `CounterBackend` (boards from env `CLIQR_FAKE_MAP` JSON or default `{"B0":[1,2,3],"B1":[4,5,6]}`; caps = per-board successful-read counter, times = `time.time()`), `PyftdiLikeBackend` (boards `P0`..`P3`, board b → sensors `6b+1..6b+6`, exposed as `PYFTDI_LIKE_MAP` for tests/script to patch `SERIAL_NUMBER_SENSOR_MAP`; each read = 20 × `time.sleep(0.0001)`, returning `len(channels)` values), `CrashAfterBackend` (like Counter, `os._exit(3)` on the 50th read), `ImportProbeBackend` (`init` returns errors list containing `"solara imported"` iff `"solara" in sys.modules`).
  - `recording.sources.SampleSink` protocol: `on_loops(loops: list[list[tuple[str, list[float], list[int]]]]) -> None`, `on_read_error(sn: str, message: str) -> None`, `on_fatal(message: str) -> None`.
  - `recording.sources.ThreadSource(manager, serials: list[str], hz: float)`: `manager.read_sensor_data(sn) -> (times, caps, sn)`; `start(sink) -> None` runs `run_sampling_loop` on thread `cliqr-sampler`, delivering each loop as `sink.on_loops([loop])`; unexpected exception → `sink.on_fatal(str(exc))`. `stop(timeout: float) -> bool`.
  - `SensorRecorder(mpr121_manager=None, filename, controllers, measurements_provider=None, two_bottle=False, source=None)`: if `source is None`, `start()` builds `ThreadSource(mpr121_manager, list(controllers), MAX_SAMPLE_HZ)` (read `MAX_SAMPLE_HZ` at start time; tests monkeypatch it). Recorder implements `SampleSink`: `on_loops` = today's buffer/pending/produced/`loop_counter`/batch-every-`HISTORY_SIZE` logic; `on_read_error` increments `_read_error_count` and logs `WARN: read failed on board {sn} after retries (total skips={n}): {message}` with today's rate limit; `on_fatal` = today's fatal path (log `ERROR in recording loop: {msg}`, `on_error(RuntimeError(msg))` once, hand over pending, end writer). `stop()` calls `source.stop(timeout)` instead of joining the sampler.

- [ ] **Step 1: Write failing tests in `tests/test_sampling_loop.py`**

```python
def test_loop_paces_and_reports_each_board():   # 2 fake boards, hz=200, run 0.25 s
    assert 40 <= len(loops) <= 60
    assert all({sn for sn, _, _ in l} == {"B0", "B1"} for l in loops)

def test_failed_read_skips_only_that_board():   # B1 raises every 3rd call
    assert errors and all(sn == "B1" for sn, _ in errors)
    assert len(b0_loops) - len(b1_loops) == len(errors)

def test_on_loop_exception_propagates():         # on_loop raises KeyError on loop 5
    with pytest.raises(KeyError): run_sampling_loop(...)

def test_backends_import_without_solara():       # subprocess: python -I -c "import hardware.backends, hardware.fake_backends, hardware.sampling, utils.hw_constants, sys; print('solara' in sys.modules, 'utils.state' in sys.modules)"
    assert out.strip() == "False False"
```

- [ ] **Step 2: Run** `python -m pytest tests/test_sampling_loop.py -v` — Expected: FAIL (`ModuleNotFoundError: hardware.sampling`)

- [ ] **Step 3: Implement** the files above per Interfaces. `recorder.py` loses `_sample_loop`/`_sample_once`; `_take_pending`, `_write_loop`, `_drain`, `_write_batch` unchanged.

- [ ] **Step 4: Run** `python -m pytest tests/test_sampling_loop.py tests/test_recorder_thread.py tests/test_recorder_skip_resilience.py tests/test_two_bottle.py tests/test_recorder_measurement_persist.py -v` — Expected: all PASS, recorder tests unmodified.

- [ ] **Step 5: Run full suite** `python -m pytest tests -q --ignore=tests/test_ml_train.py` — Expected: same pass count as before + new tests. (Tests touching `hw.mpr121_manager` in `test_recorder_thread.py` still pass because Task 1 doesn't touch the GUI.)

- [ ] **Step 6: Commit** — `git commit -m "Split sampling into a Solara-free loop, backends and recorder sources"`

### Task 2: Hardware process and client

**Files:**
- Create: `hardware/hw_process.py`, `hardware/hw_client.py`
- Test: `tests/test_hw_process.py`

**Interfaces:**
- Consumes: Task 1 `run_sampling_loop`, `Backend`, fake backends, `SampleSink`, `SensorRecorder(source=...)`.
- Produces:
  - `hardware.hw_process.hardware_main(cmd_conn, data_q, backend_path: str) -> None` (child entry; imports backend via `"module:Class"`). Commands on `cmd_conn` → replies: `("init",)` → `("init", board_info, messages, errors)`; `("start", channels: list[int], hz: float)` → `("started",)` and sampling runs on a child thread; `("stop",)` → loop stopped, buffered loops flushed, `("end",)` put on `data_q`, reply `("stopped",)`; `("shutdown",)` → `backend.close()`, reply `("bye",)`, return. `EOFError`/`OSError` on `cmd_conn` (parent gone) → stop, close backend, return. Data on `data_q`: `("loops", loops)` every `BATCH_SECONDS = 0.025` (only if non-empty), `("read_error", sn, msg)`, `("fatal", msg)`, `("end",)`.
  - `hardware.hw_client.InitResult` dataclass: `board_info: dict[str, int]`, `messages: list[str]`, `errors: list[str]`.
  - `hardware.hw_client.HardwareClient(backend_path: str)`: `start(timeout: float = 60.0) -> InitResult` (spawn daemon `Process(name="cliqr-hardware")`, send init, wait reply); `is_alive() -> bool`; `source(channels: list[int], hz: float) -> ProcessSource`; `shutdown(timeout: float = 5.0) -> None` (idempotent; `("shutdown",)`, join, `terminate()` if still alive). A lock serializes command send/recv.
  - `hardware.hw_client.ProcessSource`: `start(sink)` sends start, waits `("started",)`, runs receiver thread `cliqr-hw-receiver` dispatching `data_q` messages to `sink.on_loops/on_read_error/on_fatal`; if `data_q.get(timeout=0.5)` times out and the process is dead → `sink.on_fatal(f"hardware process died (exit code {code})")` and exit. `stop(timeout) -> bool`: send stop, wait `("stopped",)`, join receiver (exits on `("end",)`); True iff both within timeout. Safe if the process already died (returns False after logging nothing itself).

- [ ] **Step 1: Write failing tests in `tests/test_hw_process.py`** (helper `_rec(tmp_path, client, history=50, hz=150)` builds `SensorRecorder(source=client.source([1,6,11], hz), ...)` with `SERIAL_NUMBER_SENSOR_MAP` monkeypatched to the Counter map; every test calls `client.shutdown()` in a finally)

```python
def test_init_reports_boards():                       # CounterBackend
    assert res.board_info == {"B0": 3, "B1": 3} and res.errors == []

def test_child_does_not_import_solara():              # ImportProbeBackend
    assert "solara imported" not in client.start().errors

def test_process_recording_is_complete(tmp_path):     # start, 0.5 s, stop
    assert rec.stop() is True
    # on disk: per channel cap_data == arange(1, n+1), len == board_produced, n > 0

def test_stop_before_first_flush_is_complete(tmp_path):   # history=100000, 0.2 s

def test_busy_gui_thread_does_not_gap_samples(tmp_path):  # PyftdiLikeBackend, hz=150
    # 5 × (120 ms pure-Python busy loop in this process, 0.4 s sleep)
    assert max_gap < 4 / 150

def test_child_crash_reports_fatal_and_keeps_data(tmp_path):  # CrashAfterBackend
    assert len(errors) == 1 and "hardware process died" in str(errors[0])
    t0 = time.monotonic(); rec.stop(timeout=2.0)
    assert time.monotonic() - t0 < 5.0                 # returns, never hangs
    # every loop received before the crash is on disk, in order

def test_child_exits_when_client_pipe_closes():
    client._cmd.close(); client._proc.join(5); assert not client._proc.is_alive()

def test_shutdown_is_idempotent_and_ends_process():
    client.shutdown(); client.shutdown(); assert not client.is_alive()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_hw_process.py -v` — Expected: FAIL (`ModuleNotFoundError: hardware.hw_client`)

- [ ] **Step 3: Implement** `hw_process.py` and `hw_client.py` per Interfaces. Child: keep the cmd loop on the child main thread (`cmd_conn.poll(0.1)`), sampling on a child thread, a batcher accumulating loops under a lock and flushing every `BATCH_SECONDS`.

- [ ] **Step 4: Run** `python -m pytest tests/test_hw_process.py -v` — Expected: all PASS. Then full suite — Expected: green.

- [ ] **Step 5: Commit** — `git commit -m "Run hardware I/O and sampling in a spawned hardware process"`

### Task 3: GUI wiring (init, refresh/disconnect, start/stop, TEST, mock)

**Files:**
- Modify: `components/hardware_status.py` (replace `ft232h_manager`/`mpr121_manager` globals with `hw_client: HardwareClient | None`, `BACKEND_PATH = "hardware.backends:RealBackend"`; `initialize_hardware()`, new `refresh_hardware()`, `disconnect_hardware()` module functions; `atexit.register(_shutdown_hw)`)
- Modify: `components/session_controls.py` (start: require `hw_client` alive; `source=hw_client.source(state.active_channels(), MAX_SAMPLE_HZ)`)
- Create: `recording/recent.py` — `get_last_reading(sensor_id: int, num_samples: int = 250, fluid=None) -> tuple[list | None, list | None]`: live `current_recorder.get_recent_samples(...)`, else today's HDF5 fallback (moved from `MPR121Manager.get_last_reading`).
- Modify: `components/sensor_card.py` (`test_sensor` uses `recording.recent.get_last_reading`; "Hardware not initialized" check uses `hw_client`)
- Modify: `hardware/mock_hardware.py` (`use_mock_hardware()` sets `hardware_status.BACKEND_PATH = "hardware.backends:MockBackend"`)
- Modify: `tests/test_recorder_thread.py` session tests (`test_session_start_stop_uses_recorder_threads`, `test_session_stop_completes_when_metadata_write_fails`) to use `hardware_status.BACKEND_PATH = "hardware.fake_backends:CounterBackend"` + `initialize_hardware()` instead of patching `mpr121_manager`; their assertions stay.
- Test: `tests/test_hw_gui.py`

**Interfaces:**
- Consumes: Task 2 `HardwareClient`, `InitResult`, `ProcessSource`.
- Produces: `hardware_status.initialize_hardware() -> None`, `refresh_hardware() -> None`, `disconnect_hardware() -> None`, `hardware_status.hw_client`. Session values: `boards_connected = board_info`; `i2c_controllers = {sn: {"num_sensors": n}}` (plain data).
- Rules: `initialize_hardware` with a live `hw_client` → log `Hardware process already running; reusing it`, republish session values, no new process. `refresh_hardware` while `recording_all` → log `ERROR: cannot refresh hardware while recording`, no-op; else shutdown + initialize. Refresh Hardware button calls `refresh_hardware`; Disconnect calls `disconnect_hardware`.

- [ ] **Step 1: Write failing tests in `tests/test_hw_gui.py`** (fixture sets `BACKEND_PATH` to CounterBackend, monkeypatches `SERIAL_NUMBER_SENSOR_MAP`, output dir to tmp_path, layout one row; teardown `disconnect_hardware()` and resets `recording_all`)

```python
def test_initialize_publishes_plain_board_info():
    assert state.session["boards_connected"] == {"B0": 3, "B1": 3}
    assert state.session["i2c_controllers"] == {"B0": {"num_sensors": 3}, "B1": {"num_sensors": 3}}

def test_initialize_twice_reuses_process():
    pid = hw.hw_client._proc.pid; hw.initialize_hardware()
    assert hw.hw_client._proc.pid == pid and hw.hw_client.is_alive()

def test_refresh_refused_while_recording(): ...      # same pid, ERROR logged

def test_two_sessions_one_process(tmp_path):
    # start/sleep/stop twice; two distinct files, both complete; same pid

def test_recording_survives_simulated_refresh(tmp_path):
    # start; sleep 0.3; reset reactives to defaults; rehydrate_reactives_from_session();
    # sleep 0.3; stop -> complete on disk, max gap < 4/150, recording_all False

def test_test_button_reads_live_buffer():
    # during recording: test_sensor(1) -> state.test_plot_data.value["traces"][0]["cap_data"] non-empty

def test_disconnect_ends_process(): ...
```

- [ ] **Step 2: Run** `python -m pytest tests/test_hw_gui.py -v` — Expected: FAIL (`AttributeError: ... 'hw_client'` / `BACKEND_PATH`)

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Run** `python -m pytest tests/test_hw_gui.py tests/test_recorder_thread.py tests/test_camera_integration.py tests/test_hardware_camera_persistence.py tests/test_session_persistence.py -v` — Expected: PASS. Full suite — Expected: green.

- [ ] **Step 5: Mock GUI smoke** — `solara run recording_gui_mock.py --port 8799 --no-open`; HTTP 200; plus headless `use_mock_hardware(); initialize_hardware(); start_recording(); sleep 1; stop_recording()` → all 24 channels on disk with counts == produced. Confirm `ps` shows exactly one `cliqr-hardware` child and none after server exit.

- [ ] **Step 6: Commit** — `git commit -m "Drive hardware process from the GUI; refresh-safe reuse"`

### Task 4: Activity log in the session store

**Files:**
- Modify: `utils/state.py` (`session["log_messages"] = []`, `_REACTIVE_FOR["log_messages"] = log_messages`, `LOG_MAX = 5000`, `_log_lock`, `add_log_message` appends under the lock to the session list (trim to `LOG_MAX`) then mirrors to the reactive inside `try/except Exception: pass`; new `clear_log()`)
- Modify: `recording_gui.py:57` (Clear button → `state.clear_log`)
- Test: `tests/test_activity_log.py`

**Interfaces:**
- Produces: `state.add_log_message(message: str) -> None` (unchanged signature, thread-safe), `state.clear_log() -> None`.

- [ ] **Step 1: Write failing tests**

```python
def test_log_written_from_thread_survives_refresh():
    t = threading.Thread(target=state.add_log_message, args=("ERROR: from thread",)); t.start(); t.join()
    state.log_messages.set([]); state.rehydrate_reactives_from_session()
    assert any("ERROR: from thread" in m for m in state.log_messages.value)

def test_concurrent_log_writes_lose_nothing():   # 8 threads × 200 messages
    assert len(state.session["log_messages"]) == 1600

def test_log_is_capped():                         # LOG_MAX + 10 messages
    assert len(state.session["log_messages"]) == state.LOG_MAX

def test_clear_log_clears_both(): ...
```

- [ ] **Step 2: Run** `python -m pytest tests/test_activity_log.py -v` — Expected: FAIL (`KeyError: 'log_messages'` / no `clear_log`)

- [ ] **Step 3: Implement**; add `"log_messages"` to the persisted-keys expectations only if `tests/test_hardware_camera_persistence.py` enumerates all keys (it lists only its own; leave it).

- [ ] **Step 4: Run** activity-log tests + full suite — Expected: green.

- [ ] **Step 5: Commit** — `git commit -m "Keep the activity log in the session store so it survives refresh"`

### Task 5: Measurement script + docs

**Files:**
- Modify: `scripts/measure_sample_gaps.py` — add `--source {thread,process}` (default `process`), `--busy-gui` (pure-Python busy loop 120 ms every 2 s in the GUI process) and `--backend` (default `hardware.fake_backends:PyftdiLikeBackend`); keep `--stall-ui`, `--channels`, `--seconds`.
- Modify: `docs/CLAUDE.md` — recorder row + new rows for `hardware/hw_process.py`, `hardware/hw_client.py`, `hardware/sampling.py`; note in Important Notes that USB I/O lives in the hardware process.

- [ ] **Step 1: Run** `python scripts/measure_sample_gaps.py --source thread --busy-gui --seconds 20` and `--source process --busy-gui --seconds 20`. Expected: thread ≈ 125 ms max gap (reproduces rig), process ≲ 20 ms.
- [ ] **Step 2: Commit** — `git commit -m "Measure sample gaps for thread vs process sources; document hardware process"`
