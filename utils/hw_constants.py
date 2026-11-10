"""
Hardware constants (rack layout, MPR121 registers, sampling parameters).

Kept free of Solara/pandas imports so the hardware process (spawned; see
hardware/hw_process.py) can import it cheaply. utils.state re-exports every
name here, so `from utils.state import X` keeps working in the GUI.
"""
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

