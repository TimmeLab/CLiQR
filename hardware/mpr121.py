"""
MPR121 capacitive touch sensor configuration and management.

This module handles the configuration and operation of MPR121 sensors
connected via FT232H I2C interfaces.
"""
import time
from typing import Dict, List
from utils.hw_constants import (
    SOFT_RESET, CONFIG, CONFIG1, CONFIG2, CONFIG1_VALUE, CONFIG2_VALUE, DATA
)


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
            # Perform soft reset (leaves ECR=0 => electrodes stopped, so the AFE
            # config registers below can be written before entering run mode).
            port.write_to(SOFT_RESET, b'\x63')

            # AFE config tuned for signal quality (see utils.state). Must be written
            # while electrodes are stopped, i.e. before the ECR write. Soft reset
            # zeros these registers, which disables the charge current/time (CDC=0,
            # CDT=000), so the signal is noisy garbage unless we set them explicitly.
            port.write_to(CONFIG1, bytes([CONFIG1_VALUE]))  # 0x5C: FFI=18, CDC=16uA
            port.write_to(CONFIG2, bytes([CONFIG2_VALUE]))  # 0x5D: CDT=2us, SFI=10, ESI=1ms -> 100Hz

            # Enter run mode via ECR (0x8F: baseline tracking on, electrodes enabled)
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

    def read_sensor_data(self, serial_number: str, channels=None) -> tuple:
        """
        Read capacitance data from all channels of a sensor.

        Args:
            serial_number: Serial number of the board to read from

        Returns:
            Tuple of (time_data, cap_data, serial_number)
        """
        from collections import deque
        from utils.hw_constants import READ_RETRIES, READ_RETRY_DELAY

        if serial_number not in self.controllers:
            raise ValueError(f"No controller found for {serial_number}")

        port = self.controllers[serial_number]["port"]
        if channels is None:
            # Legacy in-process path (GUI / unit tests): current mode's channels.
            from utils.state import active_channels
            channels = active_channels()
        local_time_data = deque(maxlen=len(channels))
        local_cap_data = deque(maxlen=len(channels))

        # Read 24 bytes (2 bytes for each of the 12 channels). Retry transient
        # USB/FTDI read failures ("No answer from FTDI") so a single hiccup
        # doesn't propagate up and kill the recording session.
        raw_buffer = None
        for attempt in range(READ_RETRIES):
            try:
                raw_buffer = port.read_from(DATA, 24)
                break
            except Exception:
                if attempt == READ_RETRIES - 1:
                    raise
                time.sleep(READ_RETRY_DELAY)

        # Only the physically wired MPR121 channels are recorded. The channel
        # list is set by the rack design and 2-bottle mode (e.g. [1, 6, 11] on
        # the 8-board rack, [0, 1, 5, 6, 10, 11] there in 2-bottle mode); its
        # order matches the board's sensor IDs in SERIAL_NUMBER_SENSOR_MAP, with
        # each sensor's (fluid 1, fluid 2) pair adjacent in 2-bottle mode.
        for chan in channels:
            # Combine two bytes (little-endian)
            value = raw_buffer[2 * chan] | (raw_buffer[2 * chan + 1] << 8)
            local_cap_data.append(value)
            local_time_data.append(time.time())

        return local_time_data, local_cap_data, serial_number
