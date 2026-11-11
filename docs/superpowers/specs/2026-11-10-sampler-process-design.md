# MPR121 sampling in a separate process — design

Date: 2026-11-10 · Branch: `feature/sampler-process` (from `feature/recorder-thread`)

## Intent

On the Windows rig every UI action — TEST, Refresh, typing a volume, clicking
Start — stalls sampling on **all boards** for ~110–260 ms (≈125 ms per TEST
press). Moving sampling to a thread (`feature/recorder-thread`) fixed the HDF5
flush gaps and the tail lost at Stop, but not this.

Cause (profiled on the rig with py-spy, reproduced on macOS): each reactive
`.set()` triggers a synchronous full-page Solara re-render (pure Python, all 24
sensor cards). One pyftdi board read is ~20 USB transfers, each releasing and
re-acquiring the GIL; while the UI thread is CPU-bound each re-acquire waits up
to the 5 ms switch interval, so one read stretches across the whole render (GIL
"convoy effect"). Simulation: 125 ms gap at the default switch interval, ~50 ms
at 0.5 ms. Only a separate process (its own GIL) removes it.

## Success

On the rig, during a recording: pressing TEST/Refresh and other UI actions
leaves **no unexpectedly long gaps** in other sensors' `time_data` — intervals
stay near the sample period. Checked as before (open TEST on one sensor, look at
the others), plus a mid-run browser refresh.

## How the rig is used (user)

Hardware is initialized once per GUI launch, used for the experiment, then the
GUI is fully closed before the next run. Boards are never re-initialized,
unplugged or replugged mid-run.

## Goals

1. **Hardware process** owns all FT232H/MPR121 USB I/O for the life of the GUI
   server: scan, I2C controller setup, MPR121 address detection and
   configuration, and sampling. The GUI process never touches USB.
2. Sampling loop unchanged in behaviour: paced at `MAX_SAMPLE_HZ` with a running
   deadline, all boards read in parallel, `time.time()` stamped per channel at
   the read, per-board skip on read failure after retries.
3. Samples reach the GUI process in small batches (my call: every ~20–50 ms) and
   flow into the existing pieces from `feature/recorder-thread`: TEST-plot ring
   buffers, pending batches, the HDF5 writer thread (still under `_h5_lock`,
   shared with bookmark/metadata writes), and Stop's full flush.
4. **Refresh/reconnect safe (user requirement)**: the hardware process and the
   GUI-side receiver/writer threads belong to the server process, held in plain
   module globals / `state.session` (like `current_recorder` today). A browser
   refresh or reconnect never stops, restarts or duplicates them; recording
   continues with no gap.
5. **Activity log survives refresh** (my suggestion, agreed): the log moves into
   the context-free session store so a refreshed tab shows the full log,
   including ERRORs logged by background threads while it was disconnected.

## Non-goals

- No change to the h5 layout, timestamps, `MAX_SAMPLE_HZ` (150),
  `HISTORY_SIZE` (500), channel/board mapping, MPR121 register config, camera
  and bookmark code, or analysis code.
- Not reducing Solara render cost (option 3) and not lowering the switch
  interval (option 2); revisit only if rig testing shows residual gaps.
- No support for hot-plugging or re-initializing boards mid-run.

## Anti-goals (failure even if it "works")

- Any dropped or duplicated sample, including at Stop and across a refresh.
- Two processes opening the same boards (e.g. Initialize clicked again, or a
  refresh re-running init).
- An orphaned hardware process holding the USB boards after the GUI server
  exits (next launch couldn't open them).
- A hardware-process crash the operator can't see.

## Decided

- **Lifecycle**: Initialize Hardware starts the hardware process (if one isn't
  already alive — then it is reused), which scans/configures and replies with
  the board info and log lines shown today. It lives until the server exits or
  the operator clicks Disconnect. Start/Stop are commands to it.
- **Refresh Hardware / Disconnect buttons** (exist today): not allowed while
  recording (Disconnect is already hidden then; Refresh Hardware gets the same
  rule). When idle, Refresh Hardware shuts the process down and starts a fresh
  one; Disconnect shuts it down. (My call; the user never uses them mid-run.)
- **Server exit**: the process must not outlive the GUI server (daemon process
  plus explicit shutdown; builder ensures USB handles are released).
- **Crash**: if the hardware process dies mid-run, the GUI logs an ERROR,
  everything received is saved, and the existing fatal path applies (log +
  clear `recording_all`), with the writer ended as on `feature/recorder-thread`.
- **Mock mode**: the same process runs with the mock boards (`use_mock_hardware`
  / `recording_gui_mock.py` keep working), so tests exercise the real IPC path.
- **Child start method**: Windows uses `spawn`; the child must not import the
  Solara app. Config it needs (channel list, retries, sample rate, rack map) is
  passed explicitly.
- **TEST plot** keeps reading the GUI-side ring buffers (`get_recent_samples`),
  never the hardware process.

## Known costs

- Initialize Hardware takes ~1–3 s longer (fresh Python process on Windows).
- TEST plot data lags real time by the batch interval (tens of ms).

## Verification

Automated:
- Sampling under a busy GUI thread: with a pyftdi-like fake board (many short
  GIL-releasing calls per read), 120 ms of pure-Python work in the GUI process
  leaves max gap ≲ 2–3 periods (the thread version shows ~125 ms here).
- Stop completeness across the process boundary: on-disk count == samples read
  by the child, in order, no duplicates, incl. Stop before the first flush.
- Refresh mid-recording (`rehydrate_reactives_from_session` on a fresh context):
  same process, no gap, Stop still complete; log visible in the new context.
- Initialize twice → one process. Disconnect / server shutdown → process gone.
- Child crash mid-run → ERROR logged, received data on disk, GUI state cleared.
- Existing recorder, 2-bottle, skip-resilience, measurement-persist and
  camera tests keep passing.
- `scripts/measure_sample_gaps.py` extended to drive the process path with a
  busy GUI thread.

Rig (user): TEST-press gap check and a mid-run browser refresh.

## Left to the builder

- IPC mechanism (`multiprocessing` Pipe/Queue vs shared memory), batch format
  and interval, command protocol.
- Where the per-board thread pool lives inside the child (keep parallel reads).
- How `SensorRecorder` is split between "source of samples" (child) and the
  GUI-side buffering/writing; keep the `start()`/`stop()`/`get_recent_samples`
  API for callers where possible.
- Exact shutdown hooks (atexit, server shutdown event) and timeouts.

## Calls made during builder check (my calls, not user decisions)

- `state.i2c_controllers` / `boards_connected` stay as the GUI's view of the
  hardware but hold plain data (serial → info) instead of live pyftdi handles;
  `start_recording`'s "hardware initialized" check keeps working off them.
- `spawn` re-imports the parent's `__main__`. Under `solara run` that is
  solara's CLI module, whose `main()` is behind `if __name__ == "__main__"`, so
  the child won't start a second server — but the builder must verify it on the
  real launch path (`solara run recording_gui.py` and the Windows
  `solara.exe` launcher) and keep the child's entry module free of Solara/app
  imports so spawn stays fast. macOS also defaults to `spawn`, so tests on the
  dev Mac exercise the same start path as Windows.
- The child only samples between Start and Stop (idle otherwise), as today.
- `get_last_reading`'s HDF5 fallback (no live recorder) stays GUI-side; it
  never needed USB.
