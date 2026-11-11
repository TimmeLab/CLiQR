"""
Hardware backends run inside the hardware process (hardware/hw_process.py).

A backend owns the boards: init() scans/configures them, read() does one board
read, close() releases USB. Solara-free so the spawned process imports fast.
Selected by a "module:Class" path (see components/hardware_status.BACKEND_PATH).
"""


class RealBackend:
    """FT232H boards over USB with MPR121 sensors."""

    def __init__(self):
        from hardware.ft232h import FT232HManager
        self._ft = FT232HManager()
        self._mpr = None

    def init(self):
        """Scan, open and configure all boards.

        Returns (board_info {sn: num_sensors}, log messages, errors), with the
        same steps and log text the GUI used when it did this itself.
        """
        from hardware.mpr121 import MPR121Manager
        messages, errors = ["Scanning for FT232H boards..."], []
        n = self._ft.scan_devices()
        if n == 0:
            return {}, messages, ["No FT232H boards found"]
        messages.append(f"Found {n} FT232H board(s)")
        controllers, init_errors = self._ft.initialize_controllers()
        errors.extend(init_errors)
        if not controllers:
            errors.append("No controllers initialized")
            return {}, messages, errors
        self._mpr = MPR121Manager(controllers)
        messages.extend(self._mpr.configure_all_sensors())
        return self._ft.get_controller_info(), messages, errors

    def read(self, sn, channels):
        times, caps, _ = self._mpr.read_sensor_data(sn, channels)
        return list(times), list(caps)

    def close(self):
        self._ft.close_all()


class MockBackend(RealBackend):
    """Simulated boards (recording_gui_mock.py)."""

    def __init__(self):
        from hardware.mock_hardware import MockFT232HManager
        self._ft = MockFT232HManager()
        self._mpr = None

    def init(self):
        from hardware.mock_hardware import MockMPR121Manager
        messages = ["Using MOCK HARDWARE for testing"]
        n = self._ft.scan_devices()
        messages.append(f"Found {n} mock FT232H board(s)")
        controllers, errors = self._ft.initialize_controllers()
        self._mpr = MockMPR121Manager(controllers)
        messages.extend(self._mpr.configure_all_sensors())
        return self._ft.get_controller_info(), messages, errors
