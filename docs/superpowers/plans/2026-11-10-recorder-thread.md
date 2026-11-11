# Recorder Thread Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MPR121 polling and HDF5 writes run on their own threads so UI work and disk stalls can't drop samples, and a session Stop flushes every sample read.

**Architecture:** `SensorRecorder` gets a sampler thread (pace + parallel reads + buffer appends) and a writer thread (all recorder HDF5 writes) joined by a `queue.Queue` of per-channel batches. `start()`/`stop()` replace the asyncio task; `stop()` joins both threads after a final batch so everything read is on disk.

**Tech Stack:** Python 3.13, `threading`, `queue`, `concurrent.futures`, h5py 3.13, Solara 1.57, pytest.

**Spec:** `docs/superpowers/specs/2026-11-10-recorder-thread-design.md`

## Global Constraints

- No change to h5 layout, dataset names, dtypes (`time_data` float64, `cap_data` int64), chunking `(HISTORY_SIZE,)`, `maxshape=(None,)`.
- No change to `MAX_SAMPLE_HZ` (150), `HISTORY_SIZE` (500), `MEASUREMENT_PERSIST_SECONDS`, read retries, or per-board skip behaviour.
- `get_recent_samples(sensor_id, num_samples=250, fluid=None)` and `get_last_reading` signatures unchanged.
- Never silently drop or duplicate a sample; write failures logged with `ERROR:` prefix.
- Threads are daemon threads.
- All recorder HDF5 access still goes through `self._h5_lock`.

## Review Focus

1. Disk keeps failing at Stop → `stop()` must return within its timeout, return `False`, and log an ERROR — not hang the GUI. (Task 1 test `test_stop_returns_when_writer_stuck`)
2. `stop()` called twice, or after a fatal sampler error, or on a never-started recorder (camera tests do this) → no exception, no duplicate data. (Task 1 `test_stop_is_idempotent_and_safe_unstarted`)
3. TEST plot read while sampling → `get_recent_samples` returns equal-length cap/time lists. (Task 1 `test_recent_samples_consistent_while_running`)
4. Unexpected exception inside the sampler (not a board read failure) → logged, `on_error` called, samples read so far still on disk. (Task 1 `test_sampler_fatal_error_reports_and_keeps_data`)
5. 48-channel 2-bottle layout → fluid subgroups still written correctly and completely. (existing `tests/test_two_bottle.py::test_recorder_writes_fluid_subgroups`, adapted in Task 1, plus Task 3 measurement)

---

### Task 1: Threaded recorder core

**Files:**
- Modify: `recording/recorder.py` (module docstring; `__init__`; replace `record_sensors`, `_write_initial_data`, `_append_data`, `stop`; `get_recent_samples` buffer copy)
- Modify: `tests/test_recorder_skip_resilience.py:120-130`, `tests/test_two_bottle.py:89-95` (drive via `start()`/`stop()` instead of asyncio)
- Create: `tests/test_recorder_thread.py`

**Interfaces:**
- Produces:
  - `SensorRecorder.start(log_callback=None, on_error=None) -> None` — calls `initialize_hdf5_file()`, starts writer then sampler thread. `on_error(exc)` is called once from the sampler thread if it dies on an unexpected exception.
  - `SensorRecorder.stop(timeout: float = 10.0) -> bool` — idempotent; safe if never started (returns `True`). Signals sampler, joins it, enqueues final partial batch + sentinel, joins writer. Returns `True` iff both threads finished and no batch is left unwritten; otherwise logs `ERROR:` and returns `False`.
  - `self.recording: bool` stays (True between start and stop).
  - `self.board_produced[sn][key]`, `self.board_written[sn][key]`, `self._read_error_count` keep their meaning (existing tests assert on them).
  - `SensorRecorder._write_batch(batch: dict) -> None` — `batch[sn][key] = (times: list[float], caps: list[int])`; creates `time_data`/`cap_data` if absent (same create_dataset args as today), else resizes and appends; updates `board_written`. Single h5 open under `_h5_lock`.
  - `record_sensors`, `_write_initial_data`, `_append_data` are removed.

- [ ] **Step 1: Write failing tests in `tests/test_recorder_thread.py`**

Shared helpers: monkeypatch `utils.state.SERIAL_NUMBER_SENSOR_MAP` to `{"B0": [1, 2, 3], "B1": [4, 5, 6]}`, controllers `{"B0": object(), "B1": object()}`; `_CounterManager.read_sensor_data(sn)` returns `([time.time()]*3, [n]*3, sn)` with per-board `n` incrementing per successful read. Helper `_on_disk(rec)` → `{(sn, sensor): (time_data, cap_data)}` arrays.

```python
def test_stop_flushes_every_sample(tmp_path, monkeypatch):
    # HISTORY_SIZE=50, MAX_SAMPLE_HZ=500, run 0.25 s -> stops mid-batch
    ...
    assert rec.stop() is True
    for (sn, s), (t, c) in _on_disk(rec).items():
        assert len(c) == rec.board_produced[sn][s]
        assert np.array_equal(c, np.arange(1, len(c) + 1))   # in order, no dups
        assert len(t) == len(c)

def test_stop_before_first_flush_writes_data(tmp_path, monkeypatch):
    # HISTORY_SIZE=100000, run 0.1 s
    ...
    assert all(len(c) == rec.board_produced[sn][s] > 0 for (sn, s), (_, c) in d.items())

def test_ui_loop_stall_does_not_gap_samples(tmp_path, monkeypatch):
    # MAX_SAMPLE_HZ=150, HISTORY_SIZE=50. Inside asyncio.run: rec.start();
    # await asyncio.sleep(0.1); time.sleep(0.3)  # blocks the loop
    # await asyncio.sleep(0.1); rec.stop()
    assert max(np.diff(t).max() for t, _ in _on_disk(rec).values()) < 4 / 150

def test_h5_lock_held_does_not_gap_samples(tmp_path, monkeypatch):
    # MAX_SAMPLE_HZ=150, HISTORY_SIZE=10; another thread holds rec._h5_lock 0.5 s mid-run
    assert max(np.diff(t).max() for t, _ in _on_disk(rec).values()) < 4 / 150
    # and still complete: len(c) == board_produced

def test_write_failure_retries_without_loss(tmp_path, monkeypatch):
    # wrap rec._write_batch: first 2 calls raise OSError("disk"), then delegate
    assert any(m.startswith("ERROR:") for m in logs)
    # data complete & in order as in test_stop_flushes_every_sample

def test_stop_returns_when_writer_stuck(tmp_path, monkeypatch):
    # rec._write_batch always raises
    t0 = time.monotonic(); ok = rec.stop(timeout=1.0)
    assert ok is False and time.monotonic() - t0 < 3.0
    assert any(m.startswith("ERROR:") for m in logs)

def test_stop_is_idempotent_and_safe_unstarted(tmp_path, monkeypatch):
    assert SensorRecorder(..., mpr121_manager=None, ...).stop() is True
    # started recorder: rec.stop() twice -> both True, on-disk length unchanged after 2nd

def test_recent_samples_consistent_while_running(tmp_path, monkeypatch):
    # 200 calls to rec.get_recent_samples(1) during a 0.3 s run
    assert all(c is None or len(c) == len(t) for c, t in results)

def test_sampler_fatal_error_reports_and_keeps_data(tmp_path, monkeypatch):
    # manager returns serial "BOGUS" on its 20th call (KeyError on append)
    assert len(errors) == 1             # on_error called once
    assert rec.stop() is True
    # samples from calls before the bad one are on disk
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_recorder_thread.py -v`
Expected: FAIL — `AttributeError: 'SensorRecorder' object has no attribute 'start'`

- [ ] **Step 3: Implement in `recording/recorder.py`**

- `__init__`: add `self._buf_lock = threading.Lock()`, `self._pending = {sn: {key: ([], []) for key in keys}}`, `self._queue = queue.Queue()`, `self._stop_event = threading.Event()`, `self._sampler = self._writer = None`, `self._unwritten = []` (failed batches awaiting retry), `self._log = None`, `self._on_error = None`.
- `_sample_loop()`: today's loop body, synchronous: `time.sleep` instead of `asyncio.sleep`, same running-deadline / resync logic; reads via the `ThreadPoolExecutor`; appends to ring buffers **and** `_pending` under `_buf_lock`; every `HISTORY_SIZE` loops swap `_pending` for fresh lists and `self._queue.put(batch)`. Loop until `_stop_event` set. On unexpected exception: log `ERROR in recording loop: ...`, call `on_error(exc)`, exit (pending samples are still enqueued by `stop()`).
- `_write_loop()`: `queue.get(timeout=1.0)`; on batch → append to `_unwritten`, then try writing each in order via `_write_batch`, dropping each only after success; on exception log `ERROR: HDF5 write failed (N batches pending): ...` and keep them for the next cycle. Also run `_flush_measurements()` when `MEASUREMENT_PERSIST_SECONDS` elapsed (same gating as today, moved here). Exit on sentinel `None` after a last write attempt.
- `stop(timeout)`: set `recording=False` and `_stop_event`; join sampler (`timeout`); under `_buf_lock` enqueue remaining `_pending` + `None`; join writer (`timeout`); return per Interfaces. Second call is a no-op returning the first result.
- `get_recent_samples`: copy both deques under `_buf_lock`.
- Update module docstring (no longer "async").

- [ ] **Step 4: Adapt existing recorder-loop tests**

`tests/test_recorder_skip_resilience.py` and `tests/test_two_bottle.py`: replace the `asyncio.run(run())` block with `rec.start(log_callback=logs.append)` / `time.sleep(...)` / `rec.stop()`. Assertions unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_recorder_thread.py tests/test_recorder_skip_resilience.py tests/test_two_bottle.py tests/test_recorder_measurement_persist.py tests/test_recorder_concurrency.py tests/test_video_metadata.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add recording/recorder.py tests/test_recorder_thread.py tests/test_recorder_skip_resilience.py tests/test_two_bottle.py
git commit -m "Run MPR121 sampling and HDF5 writes on dedicated threads; flush all samples on stop"
```

### Task 2: GUI wiring + docs

**Files:**
- Modify: `components/session_controls.py:17-18,89,145-153` (start), `:227,289-295` (stop)
- Modify: `docs/CLAUDE.md` architecture table row for `recording/recorder.py`
- Test: `tests/test_camera_integration.py` (existing; must still pass), new test in `tests/test_recorder_thread.py`

**Interfaces:**
- Consumes: `SensorRecorder.start(log_callback, on_error)`, `SensorRecorder.stop(timeout) -> bool` from Task 1.

- [ ] **Step 1: Write failing test**

```python
def test_session_start_stop_uses_recorder_threads(tmp_path, monkeypatch):
    # monkeypatch components.hardware_status.mpr121_manager = _CounterManager(),
    # state.i2c_controllers / layout_df / output_directory=tmp_path as needed,
    # sc.start_recording(); time.sleep(0.2); sc.stop_recording()
    assert sc.current_recorder._sampler is not None
    assert not sc.current_recorder._sampler.is_alive()
    # h5 on disk: len(cap_data) == board_produced for every channel
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_recorder_thread.py::test_session_start_stop_uses_recorder_threads -v`
Expected: FAIL (start_recording still calls removed `record_sensors`)

- [ ] **Step 3: Implement**

- `start_recording`: replace `run_recording`/`asyncio.create_task` with `current_recorder.start(log_callback=state.add_log_message, on_error=_on_recorder_error)`. `_on_recorder_error(exc)`: log `Recording error: {exc}` and `state.set_session("recording_all", False)` (today's behaviour).
- `stop_recording`: replace `current_recorder.stop()` + task cancel with `if not current_recorder.stop(): state.add_log_message("ERROR: recorder did not finish writing all data to disk")`. Keep metadata/comments writes before it (spec: keep order). Remove `recording_task` global and its uses.
- `docs/CLAUDE.md`: recorder row → "Sampler + writer threads, ThreadPoolExecutor reads, batched HDF5 writes".

- [ ] **Step 4: Run full suite**

Run: `python -m pytest tests -q --ignore=tests/test_ml_train.py`
Expected: all pass (skips as before)

- [ ] **Step 5: Smoke-test mock GUI**

Run `solara run recording_gui_mock.py`, start session, open TEST on a sensor, stop. Expected: log shows "Recording started"/"Recording stopped", no errors, h5 written.

- [ ] **Step 6: Commit**

```bash
git add components/session_controls.py docs/CLAUDE.md tests/test_recorder_thread.py
git commit -m "Start/stop recorder threads from session controls"
```

### Task 3: Flush-gap measurement

**Files:**
- Create: `scripts/measure_sample_gaps.py`

**Interfaces:**
- Consumes: `SensorRecorder.start/stop` from Task 1.

- [ ] **Step 1: Write the script**

CLI: `--channels {24,48}` `--seconds` (default 30) `--out-dir` (default a temp dir). Fake manager: 4 boards (24) or 4 boards × 12 (48, `two_bottle=True`), returns `time.time()` stamps; real `HISTORY_SIZE` (500) and `MAX_SAMPLE_HZ` (150). Prints per-run: samples/s achieved, max gap, 99.9th-percentile gap, count of gaps > 2 periods. Optional `--stall-ui` runs an asyncio loop that blocks 0.3 s every 2 s to reproduce the original failure.

- [ ] **Step 2: Run and record**

Run: `python scripts/measure_sample_gaps.py --channels 24` and `--channels 48`, each with and without `--stall-ui`.
Expected: max gap ≲ 2–3 periods (≤ ~20 ms). If larger, record numbers and the h5py call responsible for the PR description (spec "Known limit") — do not change design without asking.

- [ ] **Step 3: Commit**

```bash
git add scripts/measure_sample_gaps.py
git commit -m "Add sample-gap measurement script for recorder threads"
```
