"""
Threaded recording for continuous sensor data acquisition.

Sampling and disk writes each run on their own thread, independent of the
Solara event loop. Previously the loop was an asyncio task on the UI's event
loop, so any slow UI work or HDF5 flush delayed the next read and every board
silently lost samples. Now:

  sample source   paces reads at MAX_SAMPLE_HZ and reads all boards in
                  parallel -- in the hardware process (GUI; own GIL, see
                  hardware/hw_process.py) or on an in-process thread (tests).
                  It pushes loops into on_loops(), which appends to the plot
                  ring buffers and per-channel pending batches.
  writer thread   owns all recorder-initiated HDF5 writes (sample batches and
                  the periodic volume/weight persist), fed through a queue.

stop() joins both after handing over the final partial batch, so every sample
read is on disk when it returns. See
docs/superpowers/specs/2026-11-10-recorder-thread-design.md.
"""
import queue
import threading
import time
from collections import deque
from typing import Dict
import h5py

from utils.state import (HISTORY_SIZE, MAX_SAMPLE_HZ, MEASUREMENT_PERSIST_SECONDS,
                         channel_keys, channel_group_path)


def measurement_fields(s, two_bottle, suffix=""):
    """Map HDF5 dataset path (relative to the sensor group) -> measured value.

    In 2-bottle mode the volumes live in the fluid1/fluid2 subgroups beside each
    fluid's cap_data; the shared weight stays at the sensor level.
    """
    if two_bottle:
        return {
            f"fluid1/start_vol{suffix}": s.start_volume,
            f"fluid1/stop_vol{suffix}": s.stop_volume,
            f"fluid2/start_vol{suffix}": s.start_volume2,
            f"fluid2/stop_vol{suffix}": s.stop_volume2,
            f"weight{suffix}": s.weight,
        }
    return {
        f"start_vol{suffix}": s.start_volume,
        f"stop_vol{suffix}": s.stop_volume,
        f"weight{suffix}": s.weight,
    }


def measurement_warnings(sensor_states, two_bottle=False):
    """Return one warning string per started sensor missing any measurement.

    A sensor is "started" if it is recording or has a start_time. A field is
    missing when its value is None or <= 0. Used at stop to make silent
    measurement loss visible in the activity log.
    """
    warnings = []
    for sid, s in sensor_states.items():
        if not (s.is_recording or s.start_time > 0):
            continue
        missing = [name.replace("/", " ") for name, value
                   in measurement_fields(s, two_bottle).items()
                   if value is None or value <= 0]
        if missing:
            warnings.append(f"Sensor {sid}: no {', '.join(missing)} recorded")
    return warnings


class SensorRecorder:
    """Manages asynchronous sensor data recording."""

    def __init__(self, mpr121_manager, filename: str,
                 controllers: Dict, measurements_provider=None,
                 two_bottle: bool = False, source=None):
        """
        Initialize the sensor recorder.

        Args:
            mpr121_manager: MPR121Manager instance for reading sensors
            filename: Path to HDF5 file for saving data
            controllers: Dictionary of I2C controllers
            measurements_provider: Optional callable returning
                {sensor_id: SensorState}, read from the context-immune session
                global, used for periodic measurement persistence.
            source: Sample source (see recording/sources.py). None samples
                in-process from mpr121_manager (ThreadSource); the GUI passes
                the hardware process's ProcessSource.
            two_bottle: 2-bottle choice mode. Each sensor then records two
                channels (fluid 1, fluid 2) into sensor_<id>/fluid1 and
                sensor_<id>/fluid2. Must match the channels the manager reads.
        """
        self.mpr121_manager = mpr121_manager
        self.filename = filename
        self.controllers = controllers
        self.two_bottle = two_bottle
        self._source = source
        self.recording = False
        self.loop_counter = 0
        # Count of read failures that survived retries, for rate-limited logging.
        self._read_error_count = 0

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

        # Thread machinery (see module docstring). _buf_lock guards the ring
        # buffers, pending batches and produced counters, which the sampler
        # writes and the GUI (get_recent_samples) / stop() read.
        self._buf_lock = threading.Lock()
        self._queue = queue.Queue()
        self._started = False
        self._fatal = False
        self._writer = None
        self._stop_result = None
        # Batches handed to the writer but not yet on disk (failed writes are
        # kept here and retried, never dropped).
        self._unwritten = []
        self._log = None
        self._on_error = None

        # HDF5 does not allow two concurrent open-for-write handles in one
        # process. The periodic buffer flush runs on the recording task while
        # metadata/bookmark writes are triggered from GUI event handlers on a
        # different thread, so every file access is serialized through this lock
        # to avoid "unable to lock file" errors and lost writes.
        self._h5_lock = threading.Lock()

        # Initialize data buffers for each board channel. Keys are sensor IDs,
        # or (sensor_id, fluid) in 2-bottle mode (see utils.state.channel_keys),
        # in the same order as the channels a board read returns.
        self.board_time_data = {}
        self.board_cap_data = {}

        for sn in controllers.keys():
            keys = self._channel_keys(sn)

            self.board_time_data[sn] = {
                key: deque(maxlen=HISTORY_SIZE) for key in keys
            }
            self.board_cap_data[sn] = {
                key: deque(maxlen=HISTORY_SIZE) for key in keys
            }

        # Per-sensor counters that drive HDF5 writes independently of the plot
        # ring buffers above. board_produced counts every sample appended;
        # board_written counts those already flushed. Sizing writes off these
        # (not loop_counter) keeps each sensor's dataset correct even when a
        # board skips a read and produces fewer samples than its peers.
        self.board_produced = {}
        self.board_written = {}
        for sn in controllers.keys():
            self.board_produced[sn] = {s: 0 for s in self.board_cap_data[sn]}
            self.board_written[sn] = {s: 0 for s in self.board_cap_data[sn]}

        # Samples read but not yet handed to the writer, per channel:
        # [start_index, times, caps]. start_index is the channel's sample
        # number of times[0], which makes batch writes idempotent (a retried or
        # duplicated batch rewrites the same slice instead of appending twice).
        self._pending = {
            sn: {key: [0, [], []] for key in self.board_cap_data[sn]}
            for sn in controllers.keys()
        }

    def _channel_keys(self, sn):
        """Buffer keys for board sn, in the order its read returns channels."""
        from utils.state import SERIAL_NUMBER_SENSOR_MAP
        return channel_keys(SERIAL_NUMBER_SENSOR_MAP.get(sn, []), self.two_bottle)

    def initialize_hdf5_file(self):
        """Create the HDF5 file and initialize the group structure."""
        with self._h5_lock, h5py.File(self.filename, "w") as h5f:
            h5f.attrs["two_bottle"] = self.two_bottle
            # Create groups for each board channel (require_group creates the
            # parent sensor group of a 2-bottle fluid subgroup on the way).
            for sn in self.controllers.keys():
                board_group = h5f.create_group(f"board_{sn}")
                for key in self._channel_keys(sn):
                    board_group.require_group(channel_group_path(key))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, log_callback=None, on_error=None):
        """Create the HDF5 file, start the writer thread and the sample source.

        Args:
            log_callback: Optional callable(str) for activity-log messages.
            on_error: Optional callable(exc), called once if sampling dies
                (board read failures are not fatal; they skip one sample on
                that board).
        """
        self._log = log_callback
        self._on_error = on_error
        self.initialize_hdf5_file()
        self.recording = True
        if self._source is None:
            # In-process sampling (unit tests, scripts). The GUI passes a
            # hardware-process source instead.
            from recording.sources import ThreadSource
            self._source = ThreadSource(self.mpr121_manager,
                                        list(self.controllers), MAX_SAMPLE_HZ)
        self._writer = threading.Thread(
            target=self._write_loop, name="cliqr-h5-writer", daemon=True)
        self._writer.start()
        self._started = True
        self._source.start(self)
        self._emit("Recording started - capturing data from all sensors")

    def stop(self, timeout: float = 10.0) -> bool:
        """Stop sampling and block until every sample read is on disk.

        Idempotent, and safe on a recorder that was never started. Returns True
        when sampling ended, the writer finished and nothing is left unwritten;
        otherwise logs an ERROR and returns False (never hangs longer than
        ~2*timeout).
        """
        if self._stop_result is not None:
            return self._stop_result
        self.recording = False
        if not self._started:
            self._stop_result = True
            return True

        ok = True
        if not self._source.stop(timeout):
            ok = False
            self._emit("ERROR: sampling did not stop in time; "
                       "flushing the samples received so far")
        # Hand over the final partial batch. Batch writes are idempotent, so
        # repeating a hand-off (e.g. after a fatal error) cannot duplicate data.
        self._queue.put(self._take_pending())
        self._queue.put(None)  # writer: write what's left, then exit
        self._writer.join(timeout)
        if self._writer.is_alive():
            ok = False
            self._emit("ERROR: HDF5 writer did not finish in time; "
                       "the last samples may not be on disk")
        elif self._unwritten:
            ok = False
            self._emit(f"ERROR: {len(self._unwritten)} batch(es) of samples "
                       "could not be written to disk")
        self._emit("Recording stopped")
        self._stop_result = ok
        return ok

    def _emit(self, message):
        if self._log is not None:
            try:
                self._log(message)
            except Exception:  # logging must never take down acquisition
                pass

    # ------------------------------------------------------------------
    # SampleSink (called by the source: sampler thread or receiver thread)
    # ------------------------------------------------------------------

    def on_loops(self, loops):
        """Buffer sampling loops; each loop is [(sn, times, caps)]."""
        for results in loops:
            # Validate the whole loop before touching any buffer, so a
            # malformed read can never leave a board's channels uneven.
            for sn, times, caps in results:
                n = len(self.board_cap_data[sn])  # KeyError: unknown board
                if len(times) != n or len(caps) != n:
                    raise ValueError(
                        f"board {sn} returned {len(caps)} channels, expected {n}")
            with self._buf_lock:
                for sn, times, caps in results:
                    pending = self._pending[sn]
                    for idx, key in enumerate(self.board_cap_data[sn]):
                        t, c = times[idx], caps[idx]
                        self.board_time_data[sn][key].append(t)
                        self.board_cap_data[sn][key].append(c)
                        pending[key][1].append(t)
                        pending[key][2].append(c)
                        self.board_produced[sn][key] += 1
            self.loop_counter += 1
            if self.loop_counter % HISTORY_SIZE == 0:
                self._queue.put(self._take_pending())

    def on_read_error(self, sn, message):
        self._read_error_count += 1
        # Log the first few, then only occasionally, to avoid flooding the log
        # if a board goes fully offline.
        if self._read_error_count <= 5 or self._read_error_count % 100 == 0:
            self._emit(f"WARN: read failed on board {sn} after retries "
                       f"(total skips={self._read_error_count}): {message}")

    def on_fatal(self, message):
        """Sampling died. Save what was received and end the writer."""
        with self._buf_lock:
            if self._fatal:
                return
            self._fatal = True
        self._emit(f"ERROR in recording loop: {message}")
        if self._on_error is not None:
            try:
                self._on_error(RuntimeError(message))
            except Exception:
                pass
        self._queue.put(self._take_pending())
        # The GUI clears recording_all on a fatal error, so stop() may never be
        # called. End the writer now: left running, its periodic volume/weight
        # persist would write a LATER session's values into this file.
        self._queue.put(None)

    def _take_pending(self):
        """Swap out the pending samples as a batch for the writer.

        Returns {sn: {key: (start_index, times, caps)}}.
        """
        with self._buf_lock:
            batch = {}
            for sn, chans in self._pending.items():
                batch[sn] = {}
                for key, (start, times, caps) in chans.items():
                    batch[sn][key] = (start, times, caps)
                    chans[key] = [start + len(times), [], []]
            return batch

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

    def _write_loop(self):
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                item = False
            done = item is None
            if item:
                self._unwritten.append(item)
            self._drain()

            # Periodically persist volume/weight (wall-clock gated, ~5 min).
            if (self.measurements_provider is not None
                    and time.monotonic() - self._last_persist
                    >= MEASUREMENT_PERSIST_SECONDS):
                try:
                    self._flush_measurements()
                except Exception as e:
                    self._emit(f"ERROR: volume/weight persist failed: {e}")
                self._last_persist = time.monotonic()

            if done:
                return

    def _drain(self):
        """Write queued batches in order; keep any that fail for a retry."""
        while self._unwritten:
            try:
                self._write_batch(self._unwritten[0])
            except Exception as e:
                self._emit(f"ERROR: HDF5 write failed ({len(self._unwritten)} "
                           f"batch(es) pending, will retry): {e}")
                return
            self._unwritten.pop(0)

    def _write_batch(self, batch):
        """Write one batch to HDF5 (creating datasets on first use).

        Each channel's samples land at [start_index, start_index + n), so
        rewriting a batch (retry after a partial failure) is idempotent.
        Channels grow independently, so a board that skipped reads stays
        correctly sized.
        """
        with self._h5_lock, h5py.File(self.filename, "r+") as h5f:
            for sn, chans in batch.items():
                for key, (start, times, caps) in chans.items():
                    group = h5f[f"board_{sn}"][channel_group_path(key)]
                    if "time_data" not in group:
                        # dtypes pinned so an empty first batch from an
                        # offline board doesn't drift to float.
                        group.create_dataset(
                            "time_data", shape=(0,), dtype="float64",
                            chunks=(HISTORY_SIZE,), maxshape=(None,))
                        group.create_dataset(
                            "cap_data", shape=(0,), dtype="int64",
                            chunks=(HISTORY_SIZE,), maxshape=(None,))
                    n = len(times)
                    if n:
                        end = start + n
                        if group["time_data"].shape[0] < end:
                            group["time_data"].resize((end,))
                            group["cap_data"].resize((end,))
                        group["time_data"][start:end] = times
                        group["cap_data"][start:end] = caps
                    self.board_written[sn][key] = max(
                        self.board_written[sn][key], start + n)

    def write_sensor_metadata(self, sensor_id: int, start_time=None, stop_time=None,
                              start_vol=None, stop_vol=None, weight=None, cycle=0,
                              start_vol2=None, stop_vol2=None):
        """
        Write metadata for a specific sensor to the HDF5 file.

        Args:
            sensor_id: Sensor ID (1-24)
            start_time: Recording start timestamp
            stop_time: Recording stop timestamp
            start_vol: Starting volume (mL)
            stop_vol: Stopping volume (mL)
            weight: Animal weight (g)
            cycle: Recording cycle number (0 for first, 1 for second, etc.)
            start_vol2: Fluid 2 starting volume (mL), 2-bottle mode only
            stop_vol2: Fluid 2 stopping volume (mL), 2-bottle mode only

        In 2-bottle mode start_vol/stop_vol are fluid 1 and land, with the fluid
        2 volumes, in the fluid subgroups; times and weight stay sensor-level.
        """
        # Find which board this sensor is on
        sn = self._serial_for_sensor(sensor_id)

        with self._h5_lock, h5py.File(self.filename, "r+") as h5f:
            group_path = f"board_{sn}/sensor_{sensor_id}"

            # Create the sensor group if it doesn't exist
            if group_path not in h5f:
                board_group = h5f[f"board_{sn}"]
                board_group.create_group(f"sensor_{sensor_id}")

            group = h5f[group_path]

            # Determine dataset names based on cycle number
            start_name = "start_time" if cycle == 0 else f"start_time{cycle}"
            stop_name = "stop_time" if cycle == 0 else f"stop_time{cycle}"
            suffix = "" if cycle == 0 else str(cycle)
            fluid1 = "fluid1/" if self.two_bottle else ""
            start_vol_name = f"{fluid1}start_vol{suffix}"
            stop_vol_name = f"{fluid1}stop_vol{suffix}"
            weight_name = f"weight{suffix}"

            # Write metadata (only if provided and not already exists)
            if start_time is not None and start_name not in group:
                group.create_dataset(start_name, data=start_time)
            if stop_time is not None and stop_name not in group:
                group.create_dataset(stop_name, data=stop_time)

            # Only write volume/weight if they're greater than 0
            if start_vol is not None and start_vol > 0:
                if start_vol_name in group:
                    del group[start_vol_name]
                group.create_dataset(start_vol_name, data=start_vol)
            if stop_vol is not None and stop_vol > 0:
                if stop_vol_name in group:
                    del group[stop_vol_name]
                group.create_dataset(stop_vol_name, data=stop_vol)
            if weight is not None and weight > 0:
                if weight_name in group:
                    del group[weight_name]
                group.create_dataset(weight_name, data=weight)
            if self.two_bottle:
                for name, value in ((f"fluid2/start_vol{suffix}", start_vol2),
                                    (f"fluid2/stop_vol{suffix}", stop_vol2)):
                    if value is not None and value > 0:
                        if name in group:
                            del group[name]
                        group.create_dataset(name, data=value)

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
                    fields = measurement_fields(s, self.two_bottle, suffix)
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

    def _serial_for_sensor(self, sensor_id: int) -> str:
        """Return the board serial number that owns the given sensor.

        SERIAL_NUMBER_SENSOR_MAP maps each board serial to the list of sensor
        numbers wired to it, so we simply walk the boards and return the first
        one whose sensor list contains this sensor.
        """
        from utils.state import SERIAL_NUMBER_SENSOR_MAP

        for serial_number, sensor_numbers in SERIAL_NUMBER_SENSOR_MAP.items():
            if sensor_id in sensor_numbers:
                return serial_number
        raise ValueError(f"Sensor {sensor_id} is not mapped to any board.")

    def write_video_metadata(self, sensor_id: int, frame_index=None, pts=None,
                             video_filename=None, cycle=0,
                             pi_monotonic=None, host_time_before=None,
                             host_time_after=None,
                             stop_frame_index=None, stop_pts=None,
                             stop_host_before=None, stop_host_after=None,
                             stop_pi_monotonic=None):
        """Write video bookmark metadata for a sensor's recording cycle.

        Datasets mirror the start_time cycle-suffix convention:
        cycle 0 -> "video_frame_index", cycle 1 -> "video_frame_index1", etc.

        ``pi_monotonic`` (Pi clock at bookmark) and ``host_time_before`` /
        ``host_time_after`` (host wall-clock bracketing the bookmark round-trip)
        record the bookmark latency so the video<->trace anchor can be corrected
        post-hoc without a manual offset: the bookmarked frame's true host time is
        ~``host_after`` (bookmark() runs at the END of the Pi round-trip), backed
        off the Pi capture->exec gap ``pi_monotonic - pts``; its offset from
        ``start_time`` is the latency the video panel would otherwise lead the
        trace by. Any of these left None (older callers) is simply not written.

        The ``stop_*`` fields record the Stop bookmark — a second clock anchor at
        the end of the cycle — so the video<->cap clock-rate drift can be fit as a
        line across the session (see docs/superpowers/specs/
        2026-07-16-clock-drift-stop-bookmark-design.md). Same omit-when-None rule.
        """
        sn = self._serial_for_sensor(sensor_id)
        suffix = "" if cycle == 0 else str(cycle)

        with self._h5_lock, h5py.File(self.filename, "r+") as h5f:
            group_path = f"board_{sn}/sensor_{sensor_id}"
            if group_path not in h5f:
                h5f[f"board_{sn}"].create_group(f"sensor_{sensor_id}")
            group = h5f[group_path]

            for base, value in (
                (f"video_frame_index{suffix}", frame_index),
                (f"video_pts{suffix}", pts),
                (f"video_filename{suffix}", video_filename),
                (f"video_pi_monotonic{suffix}", pi_monotonic),
                (f"video_bookmark_host_before{suffix}", host_time_before),
                (f"video_bookmark_host_after{suffix}", host_time_after),
                (f"video_stop_frame_index{suffix}", stop_frame_index),
                (f"video_stop_pts{suffix}", stop_pts),
                (f"video_stop_pi_monotonic{suffix}", stop_pi_monotonic),
                (f"video_stop_bookmark_host_before{suffix}", stop_host_before),
                (f"video_stop_bookmark_host_after{suffix}", stop_host_after),
            ):
                if value is None:
                    continue
                if base in group:
                    del group[base]
                group.create_dataset(base, data=value)

    def write_comments(self, comments: str):
        """
        Write session comments to the HDF5 file.

        Args:
            comments: Comment text to save
        """
        with self._h5_lock, h5py.File(self.filename, "r+") as h5f:
            if "comments" in h5f:
                del h5f["comments"]
            h5f.create_dataset("comments", data=comments)

    def get_recent_samples(self, sensor_id: int, num_samples: int = 250, fluid=None):
        """
        Get recent capacitance samples for a specific sensor.

        Args:
            sensor_id: Sensor ID (1-24)
            num_samples: Number of recent samples to retrieve
            fluid: 1 or 2 for one bottle of a 2-bottle sensor; None otherwise

        Returns:
            Tuple of (cap_data, time_data) lists, or (None, None) if no data available
        """
        from utils.state import SERIAL_NUMBER_SENSOR_MAP

        # Find which board this sensor is on
        if not any(sensor_id in sensors for sensors in SERIAL_NUMBER_SENSOR_MAP.values()):
            return None, None

        sn = self._serial_for_sensor(sensor_id)

        # Check if this sensor has data in the buffer
        key = (sensor_id, fluid) if fluid is not None else sensor_id
        if sn not in self.board_cap_data or key not in self.board_cap_data[sn]:
            return None, None

        # Get data from the live buffer
        # Snapshot under the buffer lock so cap and time stay paired while the
        # sampler thread appends.
        with self._buf_lock:
            cap_buffer = list(self.board_cap_data[sn][key])
            time_buffer = list(self.board_time_data[sn][key])

        if len(cap_buffer) == 0:
            return None, None

        # Return the requested number of samples (or all if less than requested)
        n = min(num_samples, len(cap_buffer))
        cap_data = cap_buffer[-n:]
        time_data = time_buffer[-n:]

        return cap_data, time_data
