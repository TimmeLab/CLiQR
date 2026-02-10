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
SERIAL_NUMBER_SENSOR_MAP = {
    "FT232H0": [1, 2, 3, 7, 8, 9],
    "FT232H1": [4, 5, 6, 10, 11, 12],
    "FT232H2": [13, 14, 15, 19, 20, 21],
    "FT232H3": [16, 17, 18, 22, 23, 24],
}

# MPR121 register addresses
SOFT_RESET = 0x80
CONFIG = 0x5E
DATA = 0x04

# Recording parameters
HISTORY_SIZE = 100  # Buffer size before HDF5 write
NUM_CHANNELS = 6    # Channels per MPR121 (every other channel)
