"""
Async recording functionality for continuous sensor data acquisition.

This module implements the main recording loop that reads from all sensors
and manages the buffered writes to HDF5 files.
"""
import asyncio
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import h5py

from hardware.mpr121 import MPR121Manager
from utils.state import HISTORY_SIZE, NUM_CHANNELS


class SensorRecorder:
    """Manages asynchronous sensor data recording."""

    def __init__(self, mpr121_manager: MPR121Manager, filename: str, controllers: Dict):
        """
        Initialize the sensor recorder.

        Args:
            mpr121_manager: MPR121Manager instance for reading sensors
            filename: Path to HDF5 file for saving data
            controllers: Dictionary of I2C controllers
        """
        self.mpr121_manager = mpr121_manager
        self.filename = filename
        self.controllers = controllers
        self.recording = False
        self.loop_counter = 0

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

    def initialize_hdf5_file(self):
        """Create the HDF5 file and initialize the group structure."""
        with h5py.File(self.filename, "w") as h5f:
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
                while self.recording:
                    # Yield control to allow UI updates
                    await asyncio.sleep(0)

                    # Launch parallel sensor reads on all boards
                    futures = []
                    for sn in self.controllers.keys():
                        future = executor.submit(
                            self.mpr121_manager.read_sensor_data,
                            sn
                        )
                        futures.append(future)

                    # Wait for all reads to complete
                    results = [future.result() for future in futures]

                    # Append data to buffers
                    for local_time, local_cap, serial_number in results:
                        from utils.state import SERIAL_NUMBER_SENSOR_MAP
                        sensors = SERIAL_NUMBER_SENSOR_MAP.get(serial_number, [])

                        for idx, sensor in enumerate(sensors):
                            self.board_time_data[serial_number][sensor].append(local_time[idx])
                            self.board_cap_data[serial_number][sensor].append(local_cap[idx])

                    # Write to HDF5 file every HISTORY_SIZE loops
                    if self.loop_counter == HISTORY_SIZE:
                        # First write - create datasets
                        self._write_initial_data()
                    elif self.loop_counter > 0 and self.loop_counter % HISTORY_SIZE == 0:
                        # Subsequent writes - append data
                        self._append_data()

                    self.loop_counter += 1

            except Exception as e:
                if log_callback:
                    log_callback(f"ERROR in recording loop: {str(e)}")
                raise

        if log_callback:
            log_callback("Recording stopped")

    def _write_initial_data(self):
        """Write the first batch of data to HDF5 (creates datasets)."""
        with h5py.File(self.filename, "r+") as h5f:
            for sn in self.controllers.keys():
                from utils.state import SERIAL_NUMBER_SENSOR_MAP
                sensors = SERIAL_NUMBER_SENSOR_MAP.get(sn, [])

                for sensor in sensors:
                    group = h5f[f"board_{sn}"][f"sensor_{sensor}"]

                    # Create resizable datasets
                    group.create_dataset(
                        "time_data",
                        data=list(self.board_time_data[sn][sensor]),
                        chunks=(HISTORY_SIZE,),
                        maxshape=(None,)
                    )
                    group.create_dataset(
                        "cap_data",
                        data=list(self.board_cap_data[sn][sensor]),
                        chunks=(HISTORY_SIZE,),
                        maxshape=(None,)
                    )

    def _append_data(self):
        """Append buffered data to existing HDF5 datasets."""
        # Calculate the current size before this batch
        tmp_ctr = self.loop_counter - HISTORY_SIZE

        with h5py.File(self.filename, "r+") as h5f:
            for sn in self.controllers.keys():
                from utils.state import SERIAL_NUMBER_SENSOR_MAP
                sensors = SERIAL_NUMBER_SENSOR_MAP.get(sn, [])

                for sensor in sensors:
                    group = h5f[f"board_{sn}"][f"sensor_{sensor}"]

                    # Resize datasets to accommodate new data
                    new_size = tmp_ctr + HISTORY_SIZE
                    group["time_data"].resize((new_size,))
                    group["cap_data"].resize((new_size,))

                    # Write the new data
                    group["time_data"][tmp_ctr:new_size] = list(
                        self.board_time_data[sn][sensor]
                    )
                    group["cap_data"][tmp_ctr:new_size] = list(
                        self.board_cap_data[sn][sensor]
                    )

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
        from utils.state import SERIAL_NUMBER_SENSOR_MAP
        import numpy as np

        sn_idx = [sensor_id in sensors for sensors in SERIAL_NUMBER_SENSOR_MAP.values()]
        sn = str(np.array(list(SERIAL_NUMBER_SENSOR_MAP.keys()))[sn_idx].item())

        with h5py.File(self.filename, "r+") as h5f:
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

    def write_comments(self, comments: str):
        """
        Write session comments to the HDF5 file.

        Args:
            comments: Comment text to save
        """
        with h5py.File(self.filename, "r+") as h5f:
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
        import numpy as np

        # Find which board this sensor is on
        sn_idx = [sensor_id in sensors for sensors in SERIAL_NUMBER_SENSOR_MAP.values()]
        if not any(sn_idx):
            return None, None

        sn = str(np.array(list(SERIAL_NUMBER_SENSOR_MAP.keys()))[sn_idx].item())

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
