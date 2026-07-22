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

# ---------------------------------------------------------------------------
# Rack design toggle
# ---------------------------------------------------------------------------
# Two rack generations are in use. Each pairs a board->sensor map with the
# MPR121 channels actually wired on that rack; the two must stay consistent
# because read_sensor_data zips ACTIVE_CHANNELS (in order) against a board's
# sensor list (in order).
#
#   "4board" - original DID rack: 4 FT232H boards, 6 sensors per MPR121 wired
#              on channels [1, 3, 5, 7, 9, 11].
#   "8board" - new DID rack: 8 FT232H boards (shorter cable runs, tidier),
#              3 sensors per MPR121 wired on channels [1, 6, 11].
#
# Select with the RACK_DESIGN constant below, or override at launch with the
# CLIQR_RACK environment variable (e.g. CLIQR_RACK=8board).
import os

RACK_DESIGN = os.environ.get("CLIQR_RACK", "4board").strip().lower()

_RACK_CONFIGS = {
    "4board": {
        "map": {
            "FT232H0": [1, 2, 3, 7, 8, 9],
            "FT232H1": [4, 5, 6, 10, 11, 12],
            "FT232H2": [13, 14, 15, 19, 20, 21],
            "FT232H3": [16, 17, 18, 22, 23, 24],
        },
        "channels": [1, 3, 5, 7, 9, 11],
    },
    "8board": {
        "map": {
            "FT232H0": [1, 2, 3],
            "FT232H1": [4, 5, 6],
            "FT232H2": [7, 8, 9],
            "FT232H3": [10, 11, 12],
            "FT232H4": [13, 14, 15],
            "FT232H5": [16, 17, 18],
            "FT232H6": [19, 20, 21],
            "FT232H7": [22, 23, 24],
        },
        "channels": [1, 6, 11],
    },
}

if RACK_DESIGN not in _RACK_CONFIGS:
    raise ValueError(
        f"Unknown RACK_DESIGN {RACK_DESIGN!r}; expected one of "
        f"{sorted(_RACK_CONFIGS)}"
    )

# Sensor mapping (same as in DataRecording.ipynb) selected by RACK_DESIGN.
SERIAL_NUMBER_SENSOR_MAP = _RACK_CONFIGS[RACK_DESIGN]["map"]


# MPR121 register addresses
SOFT_RESET = 0x80
CONFIG1 = 0x5C  # AFE Config 1: FFI (first filter iterations) + CDC (charge current)
CONFIG2 = 0x5D  # AFE Config 2: CDT (charge time) + SFI (second filter iter) + ESI (sample interval)
CONFIG = 0x5E  # ECR (Electrode Configuration): calibration lock + electrode enable
DATA = 0x04

# AFE config values tuned for signal quality, not peak rate. We only poll ~113 Hz
# over USB (latency-timer + flush-stall limited) yet the chip's old 4ms/250 Hz
# output meant we discarded ~half its samples. Slowing the distinct-output period
# to ~10 Hz-headroom and spending that budget on averaging (more filter
# iterations) trades unused rate for lower noise (~sqrt(N) on the extra samples).
#
# CONFIG1 = 0x90: FFI=10 (18 first-filter samples, was 6), CDC=16uA charge current.
# CONFIG2 = 0x70: CDT=011 (2us charge time, was 0.5us), SFI=10 (10 second-filter
#                 samples, was 4), ESI=000 (1ms electrode sample interval).
# Distinct-output period = SFI * ESI = 10 * 1ms = 10ms -> 100 Hz, just under our
# ~113 Hz poll so we catch ~every distinct sample (near 1:1, less duplication).
# CDT raised from the 0.5us minimum to 2us: longer charge time per measurement
# gives a larger electrode voltage swing (more signal amplitude) at the same CDC.
CONFIG1_VALUE = 0x90
CONFIG2_VALUE = 0x70

# Recording parameters
HISTORY_SIZE = 100  # Buffer size before HDF5 write
# Poll-rate cap (Hz). The USB/chip path can sustain ~330 Hz, but we only need
# ~150 Hz for lick detection. Capping here gives evenly-spaced samples and
# smaller files. Note: this caps how often we READ the chip over USB; it does
# not change the MPR121's on-chip charge/measure timing (that's the AFE config
# in CONFIG1/CONFIG2). Keep below the chip's distinct-output rate set by SFI*ESI.
MAX_SAMPLE_HZ = 150
# Transient FTDI/USB reads occasionally time out ("No answer from FTDI"). Retry
# the read a few times before giving up so a single hiccup doesn't drop a sample
# (and, at the loop level, doesn't crash the whole session).
READ_RETRIES = 3
READ_RETRY_DELAY = 0.001  # seconds between read retries
# The wired MPR121 channels for this rack. read_sensor_data reads these in
# order and maps them to the board's sensor IDs (see SERIAL_NUMBER_SENSOR_MAP).
ACTIVE_CHANNELS = _RACK_CONFIGS[RACK_DESIGN]["channels"]
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
