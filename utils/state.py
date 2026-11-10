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
"""List of timestamped log messages for the activity log."""


def add_log_message(message: str):
    """Add a timestamped message to the activity log."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    current = log_messages.value.copy()
    current.append(f"[{timestamp}] {message}")
    log_messages.set(current)


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
# In 2-bottle choice mode each sensor (cage) owns a PAIR of adjacent channels,
# (fluid 1, fluid 2): (0, 1), (2, 3), ... (10, 11) on the 4-board rack and
# (0, 1), (5, 6), (10, 11) on the 8-board rack. "channels_two_bottle" lists the
# pairs flattened in sensor order, fluid 1 first.
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
        "channels_two_bottle": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
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
        "channels_two_bottle": [0, 1, 5, 6, 10, 11],
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
HISTORY_SIZE = 500  # Buffer size before HDF5 write. Sized for ~112 Hz sampling
# so flushes fire ~every 4.5 s (near the old cadence) and HDF5 chunks are ~4 KB.
# Crash exposure = up to HISTORY_SIZE unflushed samples (~4.5 s), acceptable.
# Poll-rate cap (Hz). The USB/chip path can sustain ~330 Hz, but we only need
# ~150 Hz for lick detection. Capping here gives evenly-spaced samples and
# smaller files. Note: this caps how often we READ the chip over USB; it does
# not change the MPR121's on-chip charge/measure timing (that's the AFE config
# in CONFIG1/CONFIG2). Keep below the chip's distinct-output rate set by SFI*ESI.
MAX_SAMPLE_HZ = 150
# How often (wall-clock seconds) the recorder persists changed volume/weight
# values to the h5 during a run, checked at existing flush points. Elapsed-time
# based so it stays ~5 min regardless of HISTORY_SIZE or sample-rate changes.
MEASUREMENT_PERSIST_SECONDS = 300
# Transient FTDI/USB reads occasionally time out ("No answer from FTDI"). Retry
# the read a few times before giving up so a single hiccup doesn't drop a sample
# (and, at the loop level, doesn't crash the whole session).
READ_RETRIES = 3
READ_RETRY_DELAY = 0.001  # seconds between read retries
# The wired MPR121 channels for this rack. read_sensor_data reads these in
# order and maps them to the board's sensor IDs (see SERIAL_NUMBER_SENSOR_MAP).
ACTIVE_CHANNELS = _RACK_CONFIGS[RACK_DESIGN]["channels"]
NUM_CHANNELS = len(ACTIVE_CHANNELS)  # Channels recorded per MPR121 (1 bottle)
ACTIVE_CHANNELS_TWO_BOTTLE = _RACK_CONFIGS[RACK_DESIGN]["channels_two_bottle"]

# Fluids per sensor in 2-bottle mode. HDF5 subgroup names are fluid1, fluid2.
FLUIDS = (1, 2)


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
