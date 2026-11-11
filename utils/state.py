"""
Reactive state management for CLiQR Solara GUI.

This module defines all reactive state variables used throughout the application.
"""
import threading

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
    # Fluid 2 volumes, used only in 2-bottle choice mode (start_volume /
    # stop_volume then hold fluid 1). Weight is shared by both bottles.
    start_volume2: float = 0.0
    stop_volume2: float = 0.0
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
# Comments
# ============================================================================

comments = solara.reactive("")
"""User comments to be saved to HDF5 file on session stop."""


# ============================================================================
# 2-bottle choice
# ============================================================================

two_bottle = solara.reactive(False)
"""2-bottle choice mode: each sensor card records two sipper channels (fluid 1
and fluid 2) that share one start/stop and one weight but have separate
volumes. Fixed for the whole session (the toggle is locked while recording)."""


# ============================================================================
# Authoritative session state (context-immune)
# ============================================================================
#
# Solara stores each module-level reactive value PER kernel context
# (toestand.py KernelStore._get_dict): a reconnect/refresh gives the browser a
# fresh context whose reactives all read their defaults, while plain module
# globals (like current_recorder) are shared. So session-critical values are
# held here in a plain dict and MIRRORED into the reactives, and Page rehydrates
# the reactives from here on mount. This is what survives a mid-run reconnect.
session = {
    # Activity log (see add_log_message): kept here so lines logged by
    # background threads while a tab was disconnected show after a refresh.
    "log_messages": [],
    "recording_all": False,
    "filename": "",
    "comments": "",
    "two_bottle": False,
    "sensor_states": {i: SensorState(sensor_id=i) for i in range(1, 25)},
    # Hardware — live handles stay open in the server process across a browser
    # refresh; these hold the same references so a fresh context can be
    # re-pointed at them without a USB re-scan.
    "boards_connected": {},
    "i2c_controllers": {},
    # Camera run-state + config (concurrent Pi video capture).
    "camera_enabled": False,
    "camera_sensor_id": None,
    "camera_host": "picamera0.local",
    "camera_port": 8770,
    "camera_video_filename": "",
    "camera_disk_warning": "",
    "camera_stall_warning": "",
    "camera_status": "unknown",
}

_REACTIVE_FOR = {
    "recording_all": recording_all,
    "filename": filename,
    "comments": comments,
    "two_bottle": two_bottle,
    "sensor_states": sensor_states,
    # Hardware (these reactives are defined above, near the top of the module).
    "boards_connected": boards_connected,
    "i2c_controllers": i2c_controllers,
    # Camera keys are registered further down, right after the camera reactives
    # are defined (they live near the end of this module, below this point).
}


def set_session(key, value):
    """Update authoritative session state and mirror it into the reactive.

    All session-critical writes must go through here so the durable global and
    the per-context reactive never diverge.
    """
    session[key] = value
    _REACTIVE_FOR[key].set(value)


def rehydrate_reactives_from_session():
    """Restore this kernel context's reactives from the authoritative global.

    Called once per Page mount. On a fresh first mount this is a no-op (global
    == defaults); after a mid-run reconnect it repopulates the new context's
    reactives with the live run.
    """
    for key, rx in _REACTIVE_FOR.items():
        rx.set(session[key])


# ============================================================================
# Activity Log
# ============================================================================

log_messages = solara.reactive([])
"""List of timestamped log messages for the activity log (mirror of
session["log_messages"])."""
_REACTIVE_FOR["log_messages"] = log_messages

# Lines kept in memory; the GUI shows the last 20. Bounded so a long outage
# that logs every second can't grow the per-message copy without limit.
LOG_MAX = 5000
_log_lock = threading.Lock()


def _mirror_log():
    # Mirror into this context's reactive. From a background thread whose
    # browser context is gone this may fail; the session copy is what counts.
    try:
        log_messages.set(session["log_messages"])
    except Exception:
        pass


def add_log_message(message: str):
    """Add a timestamped message to the activity log. Thread-safe."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with _log_lock:
        log = session["log_messages"] + [f"[{timestamp}] {message}"]
        session["log_messages"] = log[-LOG_MAX:]
    _mirror_log()


def clear_log():
    """Clear the activity log."""
    with _log_lock:
        session["log_messages"] = []
    _mirror_log()


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

# Hardware constants live in a Solara-free module (imported by the hardware
# process); re-exported here so `utils.state.X` keeps working.
import os
from utils.hw_constants import (  # noqa: F401
    RACK_DESIGN,
    _RACK_CONFIGS,
    SERIAL_NUMBER_SENSOR_MAP,
    SOFT_RESET,
    CONFIG1,
    CONFIG2,
    CONFIG,
    DATA,
    CONFIG1_VALUE,
    CONFIG2_VALUE,
    HISTORY_SIZE,
    MAX_SAMPLE_HZ,
    MEASUREMENT_PERSIST_SECONDS,
    READ_RETRIES,
    READ_RETRY_DELAY,
    ACTIVE_CHANNELS,
    NUM_CHANNELS,
    ACTIVE_CHANNELS_TWO_BOTTLE,
    FLUIDS,
)

def active_channels():
    """MPR121 channels to read for the current mode, in board sensor order.

    Reads the context-immune session dict (not the reactive) because it is
    called from the recorder's worker threads.
    """
    if session["two_bottle"]:
        return ACTIVE_CHANNELS_TWO_BOTTLE
    return ACTIVE_CHANNELS


def channel_keys(sensors, two_bottle):
    """Recorder buffer keys for a board's sensors, in channel read order.

    One-bottle: the sensor ID. 2-bottle: (sensor_id, fluid) for each fluid, so
    keys line up one-to-one with active_channels().
    """
    if two_bottle:
        return [(sensor, fluid) for sensor in sensors for fluid in FLUIDS]
    return list(sensors)


def channel_group_path(key):
    """HDF5 group path (relative to the board group) for a recorder key."""
    if isinstance(key, tuple):
        sensor, fluid = key
        return f"sensor_{sensor}/fluid{fluid}"
    return f"sensor_{key}"


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

camera_stall_warning = solara.reactive("")
"""Warning shown when the Pi's frame watchdog restarted recording mid-session,
so the video is split across `_partN` segments (cleared at the next start)."""

show_snapshot_dialog = solara.reactive(False)
"""Whether the camera test-snapshot dialog is open."""

snapshot_image = solara.reactive("")
"""Base64 JPEG of the most recent camera test snapshot (empty if none)."""

snapshot_error = solara.reactive("")
"""Error message from the most recent snapshot attempt (empty if none)."""

snapshot_pending = solara.reactive(False)
"""True while a test snapshot is being captured (camera opens ~1-2 s)."""


# Register the camera run-state/config reactives with the persistence mechanism.
# This must happen here, AFTER the reactives above are defined, because
# _REACTIVE_FOR (defined earlier in the module) cannot reference names that do
# not yet exist at that point. The snapshot_* and show_*_dialog reactives are
# deliberately left out: they are transient UI state that should NOT survive a
# refresh.
_REACTIVE_FOR.update({
    "camera_enabled": camera_enabled,
    "camera_sensor_id": camera_sensor_id,
    "camera_host": camera_host,
    "camera_port": camera_port,
    "camera_video_filename": camera_video_filename,
    "camera_disk_warning": camera_disk_warning,
    "camera_stall_warning": camera_stall_warning,
    "camera_status": camera_status,
})


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
