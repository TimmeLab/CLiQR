"""
Async recording functionality for continuous sensor data acquisition.

This module implements the main recording loop that reads from all sensors
and manages the buffered writes to HDF5 files.
"""
import asyncio
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import h5py

from hardware.mpr121 import MPR121Manager
from utils.state import HISTORY_SIZE, NUM_CHANNELS, MAX_SAMPLE_HZ, MEASUREMENT_PERSIST_SECONDS


class SensorRecorder:
    """Manages asynchronous sensor data recording."""

    def __init__(self, mpr121_manager: MPR121Manager, filename: str,
                 controllers: Dict, measurements_provider=None):
        """
        Initialize the sensor recorder.

        Args:
            mpr121_manager: MPR121Manager instance for reading sensors
            filename: Path to HDF5 file for saving data
            controllers: Dictionary of I2C controllers
            measurements_provider: Optional callable returning
                {sensor_id: SensorState}, read from the context-immune session
                global, used for periodic measurement persistence.
        """
        self.mpr121_manager = mpr121_manager
        self.filename = filename
        self.controllers = controllers
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

        # HDF5 does not allow two concurrent open-for-write handles in one
        # process. The periodic buffer flush runs on the recording task while
        # metadata/bookmark writes are triggered from GUI event handlers on a
        # different thread, so every file access is serialized through this lock
        # to avoid "unable to lock file" errors and lost writes.
        self._h5_lock = threading.Lock()

        # Initialize data buffers for each board and sensor
        self.board_time_data = {}
        self.board_cap_data = {}

        for sn in controllers.keys():
            from utils.state import SERIAL_NUMBER_SENSOR_MAP
            sensors = SERIAL_NUMBER_SENSOR_MAP.get(sn, [])

            self.board_time_data[sn] = {
                sensor: deque(maxlen=HISTORY_SIZE) for sensor in sensors
            }
            self.board_cap_data[sn] = {
                sensor: deque(maxlen=HISTORY_SIZE) for sensor in sensors
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

    def initialize_hdf5_file(self):
        """Create the HDF5 file and initialize the group structure."""
        with self._h5_lock, h5py.File(self.filename, "w") as h5f:
            # Create groups for each board and sensor
            for sn in self.controllers.keys():
                from utils.state import SERIAL_NUMBER_SENSOR_MAP
                sensors = SERIAL_NUMBER_SENSOR_MAP.get(sn, [])

                board_group = h5f.create_group(f"board_{sn}")
                for sensor in sensors:
                    board_group.create_group(f"sensor_{sensor}")

    async def record_sensors(self, log_callback=None):
        """
        Main async recording loop.

        Args:
            log_callback: Optional callback function for logging messages
        """
        self.recording = True
        self.loop_counter = 0

        # Initialize the HDF5 file structure
        self.initialize_hdf5_file()

        if log_callback:
            log_callback("Recording started - capturing data from all sensors")

        # Use ThreadPoolExecutor for parallel sensor reads
        num_boards = len(self.controllers)
        with ThreadPoolExecutor(max_workers=num_boards) as executor:
            try:
                # Pace the loop to MAX_SAMPLE_HZ. Use a running deadline (not a
                # fixed per-iteration sleep) so read/flush jitter doesn't
                # accumulate into drift.
                sample_period = 1.0 / MAX_SAMPLE_HZ
                next_deadline = time.monotonic() + sample_period
                while self.recording:
                    # Sleep until the next scheduled sample (this also yields to
                    # the event loop for UI updates).
                    sleep_for = next_deadline - time.monotonic()
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                        next_deadline += sample_period
                    else:
                        # Fell behind (e.g. an HDF5 flush stall). Yield and resync
                        # the deadline instead of bursting to catch up.
                        await asyncio.sleep(0)
                        next_deadline = time.monotonic() + sample_period

                    # Launch parallel sensor reads on all boards
                    futures = []
                    for sn in self.controllers.keys():
                        future = executor.submit(
                            self.mpr121_manager.read_sensor_data,
                            sn
                        )
                        futures.append((sn, future))

                    # Collect results. A board whose read failed even after
                    # retries is skipped for THIS iteration only -- one transient
                    # USB error costs a sample on one board, never the whole run.
                    results = []
                    for sn, future in futures:
                        try:
                            results.append(future.result())
                        except Exception as e:
                            self._read_error_count += 1
                            # Log the first few, then only occasionally, to avoid
                            # flooding the log if a board goes fully offline.
                            if log_callback and (self._read_error_count <= 5
                                                 or self._read_error_count % 100 == 0):
                                log_callback(
                                    f"WARN: read failed on board {sn} after retries "
                                    f"(total skips={self._read_error_count}): {e}"
                                )

                    # Append data to buffers
                    for local_time, local_cap, serial_number in results:
                        from utils.state import SERIAL_NUMBER_SENSOR_MAP
                        sensors = SERIAL_NUMBER_SENSOR_MAP.get(serial_number, [])

                        for idx, sensor in enumerate(sensors):
                            self.board_time_data[serial_number][sensor].append(local_time[idx])
                            self.board_cap_data[serial_number][sensor].append(local_cap[idx])
                            self.board_produced[serial_number][sensor] += 1

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

            except Exception as e:
                if log_callback:
                    log_callback(f"ERROR in recording loop: {str(e)}")
                raise

        if log_callback:
            log_callback("Recording stopped")

    def _write_initial_data(self):
        """Write the first batch of data to HDF5 (creates datasets).

        Writes the un-flushed tail per sensor, sized from produced-vs-written
        counters rather than a fixed HISTORY_SIZE, so a board that skipped reads
        is handled correctly.
        """
        with self._h5_lock, h5py.File(self.filename, "r+") as h5f:
            for sn in self.controllers.keys():
                from utils.state import SERIAL_NUMBER_SENSOR_MAP
                sensors = SERIAL_NUMBER_SENSOR_MAP.get(sn, [])

                for sensor in sensors:
                    group = h5f[f"board_{sn}"][f"sensor_{sensor}"]

                    produced = self.board_produced[sn][sensor]
                    n = produced - self.board_written[sn][sensor]
                    time_tail = list(self.board_time_data[sn][sensor])[-n:] if n > 0 else []
                    cap_tail = list(self.board_cap_data[sn][sensor])[-n:] if n > 0 else []

                    # Create resizable datasets (dtypes pinned so an empty first
                    # batch from an offline board doesn't drift to float).
                    group.create_dataset(
                        "time_data", data=time_tail, dtype="float64",
                        chunks=(HISTORY_SIZE,), maxshape=(None,)
                    )
                    group.create_dataset(
                        "cap_data", data=cap_tail, dtype="int64",
                        chunks=(HISTORY_SIZE,), maxshape=(None,)
                    )
                    self.board_written[sn][sensor] = produced

    def _append_data(self):
        """Append buffered data to existing HDF5 datasets.

        The number of samples written per sensor is driven by produced-vs-written
        counters, not loop_counter, so a board that skipped reads (and thus
        produced fewer samples) stays correctly sized and aligned. Datasets grow
        independently per sensor.
        """
        with self._h5_lock, h5py.File(self.filename, "r+") as h5f:
            for sn in self.controllers.keys():
                from utils.state import SERIAL_NUMBER_SENSOR_MAP
                sensors = SERIAL_NUMBER_SENSOR_MAP.get(sn, [])

                for sensor in sensors:
                    produced = self.board_produced[sn][sensor]
                    n = produced - self.board_written[sn][sensor]
                    if n <= 0:
                        continue

                    group = h5f[f"board_{sn}"][f"sensor_{sensor}"]
                    time_tail = list(self.board_time_data[sn][sensor])[-n:]
                    cap_tail = list(self.board_cap_data[sn][sensor])[-n:]

                    old_size = group["time_data"].shape[0]
                    new_size = old_size + n
                    group["time_data"].resize((new_size,))
                    group["cap_data"].resize((new_size,))
                    group["time_data"][old_size:new_size] = time_tail
                    group["cap_data"][old_size:new_size] = cap_tail
                    self.board_written[sn][sensor] = produced

    def stop(self):
        """Signal the recording loop to stop."""
        self.recording = False

    def write_sensor_metadata(self, sensor_id: int, start_time=None, stop_time=None,
                              start_vol=None, stop_vol=None, weight=None, cycle=0):
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
            start_vol_name = "start_vol" if cycle == 0 else f"start_vol{cycle}"
            stop_vol_name = "stop_vol" if cycle == 0 else f"stop_vol{cycle}"
            weight_name = "weight" if cycle == 0 else f"weight{cycle}"

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

    def _serial_for_sensor(self, sensor_id: int) -> str:
        """Return the board serial number that owns the given sensor."""
        from utils.state import SERIAL_NUMBER_SENSOR_MAP
        import numpy as np

        sn_idx = [sensor_id in sensors for sensors in SERIAL_NUMBER_SENSOR_MAP.values()]
        return str(np.array(list(SERIAL_NUMBER_SENSOR_MAP.keys()))[sn_idx].item())

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

    def get_recent_samples(self, sensor_id: int, num_samples: int = 250):
        """
        Get recent capacitance samples for a specific sensor.

        Args:
            sensor_id: Sensor ID (1-24)
            num_samples: Number of recent samples to retrieve

        Returns:
            Tuple of (cap_data, time_data) lists, or (None, None) if no data available
        """
        from utils.state import SERIAL_NUMBER_SENSOR_MAP

        # Find which board this sensor is on
        if not any(sensor_id in sensors for sensors in SERIAL_NUMBER_SENSOR_MAP.values()):
            return None, None

        sn = self._serial_for_sensor(sensor_id)

        # Check if this sensor has data in the buffer
        if sn not in self.board_cap_data or sensor_id not in self.board_cap_data[sn]:
            return None, None

        # Get data from the live buffer
        cap_buffer = self.board_cap_data[sn][sensor_id]
        time_buffer = self.board_time_data[sn][sensor_id]

        if len(cap_buffer) == 0:
            return None, None

        # Return the requested number of samples (or all if less than requested)
        n = min(num_samples, len(cap_buffer))
        cap_data = list(cap_buffer)[-n:]
        time_data = list(time_buffer)[-n:]

        return cap_data, time_data
