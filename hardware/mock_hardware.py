"""
Mock hardware for testing the GUI without physical FT232H/MPR121 devices.

This module provides simulated hardware that mimics the behavior of real sensors,
allowing for GUI development and testing without requiring physical hardware.
"""
import random
import time
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
from utils.state import SERIAL_NUMBER_SENSOR_MAP, NUM_CHANNELS


class MockFT232HManager:
    """Mock FT232H manager that simulates device detection and initialization."""

    def __init__(self):
        """Initialize the mock FT232H manager."""
        self.controllers = {}
        self.devices = []

    def scan_devices(self) -> int:
        """
        Simulate scanning for FT232H devices.

        Returns:
            Number of mock devices (always 4 for testing)
        """
        # Simulate all 4 expected boards
        self.devices = [
            ("FT232H0", None),
            ("FT232H1", None),
            ("FT232H2", None),
            ("FT232H3", None),
        ]
        return len(self.devices)

    def initialize_controllers(self) -> Tuple[Dict[str, Dict], list]:
        """
        Simulate initialization of I2C controllers.

        Returns:
            Tuple of (controllers dict, empty error list)
        """
        self.controllers = {}
        serial_numbers = [dev[0] for dev in self.devices]

        for sn in serial_numbers:
            self.controllers[sn] = {
                "controller": f"MockController_{sn}",
                "port": MockI2CPort(sn),
                "serial_number": sn
            }

        return self.controllers, []

    def get_controller_info(self) -> Dict[str, int]:
        """
        Get mock controller information.

        Returns:
            Dictionary mapping serial numbers to sensor counts
        """
        info = {}
        for sn in self.controllers.keys():
            sensor_count = len(SERIAL_NUMBER_SENSOR_MAP.get(sn, []))
            info[sn] = sensor_count
        return info

    def close_all(self):
        """Close all mock connections."""
        self.controllers = {}


class MockI2CPort:
    """Mock I2C port that simulates MPR121 sensor readings."""

    def __init__(self, serial_number: str):
        """
        Initialize mock I2C port.

        Args:
            serial_number: Serial number of the board
        """
        self.serial_number = serial_number
        # Generate a baseline capacitance value for this sensor
        self.baseline = random.randint(400, 600)
        # Track time for generating realistic patterns
        self.start_time = time.time()

    def write_to(self, register: int, data: bytes):
        """
        Simulate writing to MPR121 register.

        Args:
            register: Register address
            data: Data to write
        """
        # Mock write - do nothing
        pass

    def read_from(self, register: int, num_bytes: int) -> bytearray:
        """
        Simulate reading from MPR121 data register.

        Args:
            register: Register address
            num_bytes: Number of bytes to read

        Returns:
            Simulated capacitance data
        """
        if num_bytes == 1:
            # Single byte read (used for address detection)
            return bytearray([0x5A])

        elif num_bytes == 24:
            # Reading 24 bytes = 12 channels × 2 bytes each
            # Simulate realistic capacitance patterns with some noise
            data = bytearray()

            elapsed = time.time() - self.start_time

            for chan in range(12):
                # Generate value between baseline ± 50 with some periodic variation
                # Add a slow sine wave to simulate licking patterns
                variation = 30 * np.sin(elapsed * 0.5 + chan * 0.1)
                noise = random.randint(-10, 10)
                value = int(self.baseline + variation + noise)

                # Ensure value is in valid range (0-1023 for 10-bit ADC)
                value = max(0, min(1023, value))

                # Convert to little-endian 2-byte format
                low_byte = value & 0xFF
                high_byte = (value >> 8) & 0xFF
                data.append(low_byte)
                data.append(high_byte)

            return data

        else:
            return bytearray(num_bytes)


class MockMPR121Manager:
    """Mock MPR121 manager for testing."""

    def __init__(self, controllers: Dict[str, Dict]):
        """
        Initialize mock MPR121 manager.

        Args:
            controllers: Dictionary of mock I2C controllers
        """
        self.controllers = controllers

    def configure_all_sensors(self) -> List[str]:
        """
        Simulate configuring all sensors.

        Returns:
            List of success messages
        """
        messages = []
        for sn in self.controllers.keys():
            messages.append(f"Board {sn} configured successfully (MOCK)")
        return messages

    def read_sensor_data(self, serial_number: str) -> Tuple:
        """
        Simulate reading sensor data.

        Args:
            serial_number: Serial number of the board

        Returns:
            Tuple of (time_data, cap_data, serial_number)
        """
        if serial_number not in self.controllers:
            raise ValueError(f"No controller found for {serial_number}")

        port = self.controllers[serial_number]["port"]
        local_time_data = deque(maxlen=NUM_CHANNELS)
        local_cap_data = deque(maxlen=NUM_CHANNELS)

        # Read simulated data
        raw_buffer = port.read_from(0x04, 24)

        # Extract every other channel (same as real implementation)
        if NUM_CHANNELS == 6:
            for chan in range(12):
                if chan % 2 == 0:
                    continue

                value = raw_buffer[2 * chan] | (raw_buffer[2 * chan + 1] << 8)
                local_cap_data.append(value)
                local_time_data.append(time.time())
        else:
            for chan in range(12):
                value = raw_buffer[2 * chan] | (raw_buffer[2 * chan + 1] << 8)
                local_cap_data.append(value)
                local_time_data.append(time.time())

        return local_time_data, local_cap_data, serial_number

    def get_last_reading(self, sensor_id: int, num_samples: int = 250) -> Tuple:
        """
        Simulate getting last N samples.

        Args:
            sensor_id: Sensor ID (1-24)
            num_samples: Number of samples to retrieve

        Returns:
            Tuple of (cap_data, time_data) lists
        """
        # Generate some random data for plotting
        baseline = 500
        cap_data = [baseline + random.randint(-30, 30) for _ in range(num_samples)]
        time_data = [time.time() + i * 0.02 for i in range(num_samples)]  # ~50Hz
        return cap_data, time_data


def use_mock_hardware():
    """
    Enable mock hardware mode for testing.

    This function should be called at the start of the app to replace
    real hardware with mock implementations.
    """
    import components.hardware_status as hw_status

    # Replace the real managers with mock versions
    hw_status.ft232h_manager = MockFT232HManager()

    def mock_initialize():
        """Mock initialization function."""
        from utils import state

        state.add_log_message("Using MOCK HARDWARE for testing")

        # Scan for mock devices
        num_devices = hw_status.ft232h_manager.scan_devices()
        state.add_log_message(f"Found {num_devices} mock FT232H board(s)")

        # Initialize mock controllers
        controllers, errors = hw_status.ft232h_manager.initialize_controllers()

        if errors:
            for error in errors:
                state.add_log_message(f"ERROR: {error}")

        # Store controllers
        state.i2c_controllers.set(controllers)

        # Create mock MPR121 manager
        hw_status.mpr121_manager = MockMPR121Manager(controllers)
        config_messages = hw_status.mpr121_manager.configure_all_sensors()

        for msg in config_messages:
            state.add_log_message(msg)

        # Update boards_connected
        board_info = hw_status.ft232h_manager.get_controller_info()
        state.boards_connected.set(board_info)

        state.add_log_message(f"Mock hardware ready: {len(board_info)} boards")

    # Replace the initialize function
    hw_status.initialize_hardware = mock_initialize
