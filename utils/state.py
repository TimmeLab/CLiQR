"""
Reactive state management for CLiQR Solara GUI.

This module defines all reactive state variables used throughout the application.
"""
import solara
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass, field


# ============================================================================
# Session State
# ============================================================================

recording_all = solara.reactive(False)
"""Whether a recording session is currently active."""

filename = solara.reactive("")
"""Current HDF5 filename for the recording session."""

output_directory = solara.reactive("Lickometry Data")
"""Directory where HDF5 files will be saved."""

recording_task = solara.reactive(None)
"""Reference to the async recording task."""


# ============================================================================
# Layout and Animal IDs
# ============================================================================

layout_df = solara.reactive(pd.DataFrame())
"""DataFrame mapping sensor IDs to animal IDs from uploaded layout CSV."""

session_error = solara.reactive("")
"""Error message to display prominently in session controls."""


# ============================================================================
# Hardware State
# ============================================================================

boards_connected = solara.reactive({})
"""Dictionary of connected FT232H boards: {serial_number: num_sensors}"""

i2c_controllers = solara.reactive({})
"""Dictionary storing I2C controller and port objects for each board."""


# ============================================================================
# Per-Sensor State
# ============================================================================

@dataclass
class SensorState:
    """State for an individual sensor."""
    sensor_id: int
    is_recording: bool = False
    start_time: float = 0.0
    elapsed_seconds: int = 0
    start_volume: float = 0.0
    stop_volume: float = 0.0
    weight: float = 0.0
    status: str = "idle"  # idle, recording, error
    error_message: str = ""
    animal_id: str = ""
    recording_cycle: int = 0  # Track number of times sensor has been started


# Create reactive state for all 24 sensors
sensor_states = solara.reactive({
    i: SensorState(sensor_id=i)
    for i in range(1, 25)
})
"""Dictionary of sensor states indexed by sensor ID (1-24)."""


# ============================================================================
# Activity Log
# ============================================================================

log_messages = solara.reactive([])
"""List of timestamped log messages for the activity log."""


def add_log_message(message: str):
    """Add a timestamped message to the activity log."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    current = log_messages.value.copy()
    current.append(f"[{timestamp}] {message}")
    log_messages.set(current)


# ============================================================================
# Comments
# ============================================================================

comments = solara.reactive("")
"""User comments to be saved to HDF5 file on session stop."""


# ============================================================================
# Test Plot State
# ============================================================================

test_plot_data = solara.reactive(None)
"""Data for sensor test plot: dict with 'sensor_id', 'cap_data', 'time_data'"""

show_test_dialog = solara.reactive(False)
"""Whether to show the test plot dialog."""


# ============================================================================
# Constants
# ============================================================================

# Sensor mapping (same as in DataRecording.ipynb)
# SERIAL_NUMBER_SENSOR_MAP = {
#     "FT232H0": [1, 2, 3, 7, 8, 9],
#     "FT232H1": [4, 5, 6, 10, 11, 12],
#     "FT232H2": [13, 14, 15, 19, 20, 21],
#     "FT232H3": [16, 17, 18, 22, 23, 24],
# }
# For the new DID rack, we now have 8 FT232H boards so
# that cables don't have to be run so far and things are a bit
# more organized
SERIAL_NUMBER_SENSOR_MAP = {
    "FT232H0": [1, 2, 3],
    "FT232H1": [7, 8, 9],
    "FT232H2": [4, 5, 6],
    "FT232H3": [10, 11, 12],
    "FT232H4": [13, 14, 15],
    "FT232H5": [19, 20, 21],
    "FT232H6": [16, 17, 18],
    "FT232H7": [22, 23, 24],
}


# MPR121 register addresses
SOFT_RESET = 0x80
CONFIG = 0x5E
DATA = 0x04

# Recording parameters
HISTORY_SIZE = 100  # Buffer size before HDF5 write
# On the new 8-board rack each MPR121 has only 3 sensors wired, on
# channels 1, 6 and 11. read_sensor_data reads these in this order and
# maps them to the 3 sensor IDs for the board (see SERIAL_NUMBER_SENSOR_MAP).
ACTIVE_CHANNELS = [1, 6, 11]  # MPR121 channels actually wired
NUM_CHANNELS = len(ACTIVE_CHANNELS)  # Channels recorded per MPR121


# ============================================================================
# Video Capture State (Pi camera)
# ============================================================================

camera_enabled = solara.reactive(False)
"""Whether concurrent Pi video capture is active for this session."""

camera_host = solara.reactive("picamera0.local")
"""Hostname or IP of the Raspberry Pi camera server."""

camera_port = solara.reactive(8770)
"""TCP port of the Pi camera server."""

camera_sensor_id = solara.reactive(None)
"""Sensor ID (1-24) whose Start button bookmarks the video, or None."""

camera_status = solara.reactive("unknown")
"""Last known camera connection status string for the UI."""

camera_mock = solara.reactive(False)
"""Use the in-memory mock camera client (set by recording_gui_mock.py)."""

camera_video_filename = solara.reactive("")
"""Video filename reported by the Pi at session start."""

camera_disk_warning = solara.reactive("")
"""Warning shown when the Pi is low on disk even after deleting old videos
(the user must free up space manually before the next session)."""

show_snapshot_dialog = solara.reactive(False)
"""Whether the camera test-snapshot dialog is open."""

snapshot_image = solara.reactive("")
"""Base64 JPEG of the most recent camera test snapshot (empty if none)."""

snapshot_error = solara.reactive("")
"""Error message from the most recent snapshot attempt (empty if none)."""

snapshot_pending = solara.reactive(False)
"""True while a test snapshot is being captured (camera opens ~1-2 s)."""


def make_camera_client(timeout=None):
    """Build the appropriate camera client based on mock/real state.

    Args:
        timeout: Optional timeout in seconds for connection (ignored by mock).
    """
    if camera_mock.value:
        from hardware.pi_camera_mock import MockPiCameraClient
        return MockPiCameraClient()
    from hardware.pi_camera import PiCameraClient
    if timeout is None:
        return PiCameraClient(camera_host.value, camera_port.value)
    return PiCameraClient(camera_host.value, camera_port.value, timeout=timeout)
