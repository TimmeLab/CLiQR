# Recorder off the UI event loop — design

Date: 2026-11-10 · Branch: `feature/recorder-thread`

## Intent

MPR121 polling (`SensorRecorder.record_sensors`) runs as an asyncio task on
Solara's event loop — the same loop that renders the UI and runs every button
handler. Anything slow on that loop (a render, a Start/Stop handler, the HDF5
flush every `HISTORY_SIZE` loops) delays the next read; the pacing logic then
resyncs its deadline rather than bursting, so **every board silently loses
samples** for the length of the stall. Seen most recently as gaps on all other
sensors while the TEST plot dialog was open (dialog itself fixed in a5c9560).

Acquisition must not depend on what the UI or the disk is doing.

## Success

On the rig: open TEST on one sensor (and click Start/Stop, wait through flushes)
during a recording; the other sensors' `time_data` shows **no unexpectedly long
gaps** — inter-sample intervals stay near the 6.7 ms period (150 Hz).

## Goals

1. Polling runs on a dedicated thread, independent of the Solara event loop.
2. HDF5 writes run on a separate writer thread, so a slow write — or waiting on
   `_h5_lock` while a GUI bookmark/metadata write holds it — never delays reads.
3. **Fix tail loss on Stop** (newly found, in scope): today the loop exits with
   no final flush, so a session Stop discards up to `HISTORY_SIZE` (500)
   unflushed samples per channel, ~3.3 s at 150 Hz — data from *before* the
   stop time written to the file. After this change `stop()` returns only when
   every sample read is on disk. If Stop comes before the first flush, the
   datasets must still be created and written.

## Non-goals

- No change to the h5 layout, dataset names/dtypes, chunking, or timestamps
  (`time.time()` per channel, taken in `read_sensor_data`).
- No change to `MAX_SAMPLE_HZ`, `HISTORY_SIZE`, retry behaviour, or the
  per-board skip resilience (one failed board read skips one sample on that
  board only).
- No change to `get_recent_samples` / `get_last_reading` signatures.
- Not touching the camera/bookmark threads or the time-base/sync-video logic.

## Anti-goals (failure even if it "works")

- Any silently dropped or duplicated sample, including across flush boundaries
  and at Stop. A write failure must be loud in the activity log.
- A recording thread that dies without the GUI knowing (today an exception in
  the loop logs "Recording error" and clears `recording_all`; keep that).
- Stop hanging the GUI indefinitely.

## Decided

- **Sampler thread**: pacing (running monotonic deadline, resync-on-behind as
  today), parallel board reads on the existing `ThreadPoolExecutor`, append to
  buffers. Never touches HDF5. Uses `time.sleep`, not asyncio.
- **Writer thread**: owns all recorder-initiated HDF5 work — dataset creation,
  appends, and the periodic measurement persist (`_flush_measurements`). Fed
  through a queue by the sampler.
- **Hand-off by batches, not ring-buffer tails**: the sampler accumulates new
  samples per channel in pending lists and, every `HISTORY_SIZE` loops, swaps
  them out and enqueues the batch. The writer appends batches in order. This
  decouples disk writes from the 500-sample ring buffers (which today must
  never overflow between flushes) and keeps per-channel counts independent, so
  a skipping board stays correctly sized.
- **Ring buffers** (`board_cap_data` / `board_time_data`) stay, for the TEST
  plot only. A short lock guards appends and the copy in `get_recent_samples`
  so the plot gets a consistent snapshot (cap and time the same length).
- **Stop**: `stop()` signals the sampler, joins it, enqueues the final partial
  batch, signals and joins the writer, then returns. Join with a timeout
  (builder picks; a few seconds) and log loudly if it expires.
- **GUI wiring**: `start_recording` starts the recorder's threads instead of
  `asyncio.create_task`; `stop_recording` calls `stop()` (which blocks until
  data is flushed) instead of cancelling a task. `stop_recording` currently
  writes per-sensor stop metadata *before* `recorder.stop()`; keep that order
  (a few extra samples after stop_time is fine; missing samples before it is
  not).
- **Logging** from the threads uses the existing `log_callback`
  (`state.add_log_message`); background threads already log this way (bookmark
  threads in `sensor_card.py`).

## Known limit (measure, don't assume)

h5py holds the GIL inside each HDF5 call, so while the writer is mid-call the
sampler can't run Python. With batching, the worst stall shrinks from a whole
flush to the longest single HDF5 call (file open/close, a dataset resize or
write). Measure the max sampler gap during flushes with a realistic 24-channel
(and 48-channel 2-bottle) layout; if it's more than a couple of sample periods,
report it — options then (not pre-decided) include keeping the file open
across flushes.

## Verification

Automated (fake manager + real h5py, as in `test_recorder_skip_resilience.py`):
- **UI stall**: block the asyncio loop (e.g. synchronous `time.sleep(0.3)` in a
  coroutine) during a recording; max `time_data` gap stays ≲ 2–3 periods.
- **Disk stall**: hold `_h5_lock` from another thread for ~0.5 s; no gap.
- **Stop completeness**: for every channel, on-disk sample count ==
  `board_produced`, values strictly in read order, no duplicates — including a
  Stop before the first flush and a Stop mid-batch.
- Existing skip-resilience, 2-bottle, measurement-persist, concurrency and
  video-metadata tests keep passing (adapt them to the new start/stop API where
  they drive `record_sensors` directly).
- Flush-gap measurement above, reported in the PR.

Rig (user): the TEST-dialog gap check under "Success".

## Left to the builder

- Exact API shape (`start()`/`stop()`, whether `record_sensors` survives as a
  thin awaitable wrapper for existing callers/tests).
- Queue type, join timeouts, thread names, daemon flags.
- How a writer-side HDF5 exception is handled beyond "log loudly, never drop
  silently" — my call: keep the batch and retry it on the next flush cycle.
- How the sampler reports a fatal error back to the GUI (keep current
  behaviour: log + clear `recording_all`).

## Calls made during review (my calls, not user decisions)

- Threads are daemon threads: a server kill (Ctrl-C) loses unflushed data
  exactly as today; a non-daemon thread would hang shutdown instead.
- A browser refresh mid-run does not touch the threads (recorder is a plain
  module global, like today's task on the server loop).
- Per-sensor Start/Stop is unchanged; only the session Stop gains the final
  flush.
- Retrying a failed batch must not grow memory without bound: after a few
  consecutive failures, log an ERROR every cycle and keep retrying, but keep
  sampling (the operator can then stop the session).
- Mock mode (`MockMPR121Manager`) needs no change; it exposes the same
  `read_sensor_data`.
- Update the `recording/recorder.py` row in `docs/CLAUDE.md`'s architecture
  table ("Async loop" → sampler + writer threads).
