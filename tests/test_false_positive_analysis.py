"""Regression tests for the false-positive-analysis workflow.

Guards the two correctness bugs found in review:
  1. SENSOR_BOARD_MAP had drifted to the retired 4-board rack layout, so
     lookups for most sensors on the 8-board rack raised KeyError.
  2. load_frame_offsets used skiprows=2, silently dropping the first two
     frames of the headerless .txt the Pi backend actually writes.
"""
import numpy as np

import false_positive_analysis as fp
from utils.state import SERIAL_NUMBER_SENSOR_MAP


def test_board_map_matches_recorder_layout():
    # Every sensor must resolve to exactly the board the recorder writes it
    # under: HDF5 groups are named board_{serial}/sensor_{id}.
    expected = {
        sensor: f"board_{serial}"
        for serial, sensors in SERIAL_NUMBER_SENSOR_MAP.items()
        for sensor in sensors
    }
    assert fp.SENSOR_BOARD_MAP == expected


def test_board_map_covers_every_mapped_sensor():
    all_sensors = {s for sensors in SERIAL_NUMBER_SENSOR_MAP.values() for s in sensors}
    assert set(fp.SENSOR_BOARD_MAP) == all_sensors


def test_load_frame_offsets_reads_headerless_file(tmp_path):
    # The Pi backend writes one absolute-ns timestamp per line, no header.
    stamps = [18074913446000, 18074921771000, 18074930096000, 18074938425000]
    txt = tmp_path / "frames.txt"
    txt.write_text("\n".join(str(s) for s in stamps) + "\n")

    offsets = fp.load_frame_offsets(str(txt))

    # No rows skipped: frame 0 is the first timestamp, all frames preserved.
    assert list(offsets) == stamps
    assert offsets.dtype == np.int64
