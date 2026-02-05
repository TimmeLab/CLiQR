"""
MPR121 capacitive touch sensor configuration and management.

This module handles the configuration and operation of MPR121 sensors
connected via FT232H I2C interfaces.
"""
import time
from typing import Dict, List
from utils.state import SOFT_RESET, CONFIG, DATA


class MPR121Manager:
    """Manages MPR121 capacitive touch sensors."""

    def __init__(self, controllers: Dict[str, Dict]):
        """
        Initialize the MPR121 manager.

        Args:
            controllers: Dictionary of I2C controllers from FT232HManager
        """
        self.controllers = controllers

    def configure_all_sensors(self) -> List[str]:
        """
        Configure all MPR121 sensors by performing soft reset and applying configuration.

        Returns:
            List of success messages for each configured board
        """
        messages = []

        for sn, controller_data in self.controllers.items():
            try:
                success = self._configure_single_sensor(controller_data["port"], sn)
                if success:
                    messages.append(f"Board {sn} configured successfully")
                else:
                    messages.append(f"Board {sn} configuration failed")
            except Exception as e:
                messages.append(f"Board {sn} error: {str(e)}")

        return messages

    def _configure_single_sensor(self, port, serial_number: str) -> bool:
        """
        Configure a single MPR121 sensor.

        Args:
            port: I2C port object
            serial_number: Serial number of the board for logging

        Returns:
            True if configuration successful, False otherwise
        """
        try:
            # Perform soft reset
            port.write_to(SOFT_RESET, b'\x63')

            # Configure the MPR121 (0x8F starts with Adafruit library default config)
            # This sets up baseline filtering, touch/release thresholds, etc.
            port.write_to(CONFIG, b'\x8F')

            # Give it time to initialize
            time.sleep(0.1)

            # Verify configuration by reading data
            cap = port.read_from(DATA, 24)
            if cap != bytearray(24):  # Empty bytearray means sensor not responding
                return True
            else:
                return False

        except Exception as e:
            print(f"Error configuring sensor on {serial_number}: {e}")
            return False

    def read_sensor_data(self, serial_number: str) -> tuple:
        """
        Read capacitance data from all channels of a sensor.

        Args:
            serial_number: Serial number of the board to read from

        Returns:
            Tuple of (time_data, cap_data, serial_number)
        """
        from collections import deque
        from utils.state import NUM_CHANNELS

        if serial_number not in self.controllers:
            raise ValueError(f"No controller found for {serial_number}")

        port = self.controllers[serial_number]["port"]
        local_time_data = deque(maxlen=NUM_CHANNELS)
        local_cap_data = deque(maxlen=NUM_CHANNELS)

        # Read 24 bytes (2 bytes for each of the 12 channels)
        raw_buffer = port.read_from(DATA, 24)

        # We only record every other channel starting with channel 1
        # (channels 1, 3, 5, 7, 9, 11 = 6 channels total)
        if NUM_CHANNELS == 6:
            for chan in range(12):
                # Skip even channels
                if chan % 2 == 0:
                    continue

                # Combine two bytes (little-endian)
                value = raw_buffer[2 * chan] | (raw_buffer[2 * chan + 1] << 8)
                local_cap_data.append(value)
                local_time_data.append(time.time())
        else:
            # Record all 12 channels
            for chan in range(12):
                value = raw_buffer[2 * chan] | (raw_buffer[2 * chan + 1] << 8)
                local_cap_data.append(value)
                local_time_data.append(time.time())

        return local_time_data, local_cap_data, serial_number

    def get_last_reading(self, sensor_id: int, num_samples: int = 250) -> tuple:
        """
        Get the last N samples from a sensor for testing/plotting.

        Args:
            sensor_id: Sensor ID (1-24)
            num_samples: Number of samples to retrieve

        Returns:
            Tuple of (cap_data, time_data) lists, or (None, None) if no data available
        """
        # Try to get data from the live recorder buffer first
        try:
            from components.session_controls import current_recorder
            if current_recorder is not None:
                cap_data, time_data = current_recorder.get_recent_samples(sensor_id, num_samples)
                if cap_data is not None:
                    return cap_data, time_data
        except Exception:
            pass  # Fall through to HDF5 reading

        # If no live data, try reading from the most recent HDF5 file
        try:
            from utils import state
            import h5py
            from utils.state import SERIAL_NUMBER_SENSOR_MAP
            import numpy as np

            filename = state.filename.value
            if not filename:
                return None, None

            # Find which board this sensor is on
            sn_idx = [sensor_id in sensors for sensors in SERIAL_NUMBER_SENSOR_MAP.values()]
            if not any(sn_idx):
                return None, None

            sn = str(np.array(list(SERIAL_NUMBER_SENSOR_MAP.keys()))[sn_idx].item())

            with h5py.File(filename, "r") as h5f:
                group_path = f"board_{sn}/sensor_{sensor_id}"
                if group_path not in h5f:
                    return None, None

                group = h5f[group_path]
                if "cap_data" not in group or "time_data" not in group:
                    return None, None

                # Read the last N samples
                cap_data = group["cap_data"][:]
                time_data = group["time_data"][:]

                n = min(num_samples, len(cap_data))
                return list(cap_data[-n:]), list(time_data[-n:])

        except Exception:
            return None, None
