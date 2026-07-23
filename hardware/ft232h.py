"""
FT232H USB-to-I2C board initialization and management.

This module handles detection, configuration, and management of FT232H boards
used to interface with MPR121 capacitive touch sensors.
"""
import os
from typing import Dict, Tuple, Optional
from pyftdi.i2c import I2cController
from pyftdi.usbtools import UsbTools

# Set environment variables for Blinka compatibility
os.environ['BLINKA_MPR121'] = '1'
os.environ['BLINKA_FT232H'] = '1'


class FT232HManager:
    """Manages FT232H boards and their I2C connections."""

    # USB Vendor and Product IDs for FT232H
    VENDOR_ID = 0x0403
    PRODUCT_ID = 0x6014

    def __init__(self):
        """Initialize the FT232H manager."""
        self.controllers: Dict[str, Dict] = {}
        self.devices = []

    def scan_devices(self) -> int:
        """
        Scan for connected FT232H devices.

        Returns:
            Number of devices found.
        """
        self.devices = UsbTools.find_all([(self.VENDOR_ID, self.PRODUCT_ID)])
        return len(self.devices)

    def initialize_controllers(self) -> Tuple[Dict[str, Dict], list]:
        """
        Initialize I2C controllers for all detected FT232H boards.

        Returns:
            Tuple of (controllers dict, error messages list)
        """
        errors = []
        self.controllers = {}

        if not self.devices:
            errors.append("No FT232H devices found. Please check USB connections.")
            return self.controllers, errors

        serial_numbers = [dev.sn for (dev, _) in self.devices]

        for sn in serial_numbers:
            try:
                controller_data = self._initialize_single_controller(sn)
                if controller_data:
                    self.controllers[sn] = controller_data
                else:
                    errors.append(f"Failed to initialize controller for {sn}")
            except Exception as e:
                errors.append(f"Error initializing {sn}: {str(e)}")

        return self.controllers, errors

    def _initialize_single_controller(self, serial_number: str) -> Optional[Dict]:
        """
        Initialize I2C controller for a single FT232H board.

        Args:
            serial_number: Serial number of the FT232H board

        Returns:
            Dictionary with controller and port, or None if initialization failed
        """
        try:
            url = f"ftdi://ftdi:232h:{serial_number}/1"
            controller = I2cController()
            # 400 kHz (I2C Fast mode) is the MPR121's max SCL. pyftdi defaults to
            # 100 kHz; the per-poll 24-byte read dominates the loop, so raising the
            # bus clock is what lets us actually poll at the chip's 250 Hz ceiling.
            controller.configure(url, frequency=400000)

            # FT232H defaults to a 16 ms USB latency timer, which caps polling at
            # ~50 Hz regardless of bus/chip speed (every read blocks up to 16 ms
            # before the USB buffer flushes). Drop it so the USB round-trip stops
            # being the bottleneck. 2 ms (not the 1 ms minimum) still allows a
            # ~250 Hz ceiling -- well above our 150 Hz need -- while giving a bit
            # more USB-transaction headroom to reduce transient read timeouts.
            controller.ftdi.set_latency_timer(2)

            # Try to find the correct MPR121 I2C address
            port = self._find_mpr121_address(controller)

            if port is None:
                print(f"Warning: Could not find MPR121 on {serial_number}. "
                      "Check that FT232H is in I2C mode (not UART).")
                return None

            return {
                "controller": controller,
                "port": port,
                "serial_number": serial_number
            }

        except Exception as e:
            print(f"Error configuring {serial_number}: {e}")
            return None

    def _find_mpr121_address(self, controller: I2cController) -> Optional[object]:
        """
        Auto-detect the MPR121 I2C address by trying common addresses.

        Args:
            controller: I2C controller to use

        Returns:
            I2C port object if found, None otherwise
        """
        # MPR121 can use addresses 0x5A, 0x5B, 0x5C, or 0x5D
        # depending on the ADDR pin configuration
        possible_addresses = [0x5A, 0x5B, 0x5C, 0x5D]

        for address in possible_addresses:
            try:
                port = controller.get_port(address)
                # Try to read one byte from the data register to verify connection
                port.read_from(0x04, 1)
                return port
            except:
                continue

        return None

    def get_controller_info(self) -> Dict[str, int]:
        """
        Get information about connected controllers.

        Returns:
            Dictionary mapping serial numbers to sensor counts
        """
        from utils.state import SERIAL_NUMBER_SENSOR_MAP

        info = {}
        for sn in self.controllers.keys():
            # Sensors per board depends on the rack design (6 on the 4-board
            # rack, 3 on the 8-board rack); read it from the mapping directly.
            sensor_count = len(SERIAL_NUMBER_SENSOR_MAP.get(sn, []))
            info[sn] = sensor_count

        return info

    def close_all(self):
        """Close all I2C controller connections."""
        for sn, controller_data in self.controllers.items():
            try:
                controller_data["controller"].terminate()
            except:
                pass
        self.controllers = {}
