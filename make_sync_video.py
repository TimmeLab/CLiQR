"""Render a side-by-side mouse-video + capacitance-trace demo clip.

Left panel: the mouse video. Right panel: the sensor's capacitance trace in a
sliding window with a centered dot marking the current time and markers on
detected licks. See docs/superpowers/specs/2026-07-14-sync-video-composite-design.md.
"""
import os
import re
import tempfile
from dataclasses import dataclass

import h5py
import imageio
import numpy as np
import pandas as pd

from data_analysis import filter_data

_BOARD_FOR_SENSOR = {
    **{s: "board_FT232H0" for s in (1, 2, 3, 7, 8, 9)},
    **{s: "board_FT232H1" for s in (4, 5, 6, 10, 11, 12)},
    **{s: "board_FT232H2" for s in (13, 14, 15, 19, 20, 21)},
    **{s: "board_FT232H3" for s in (16, 17, 18, 22, 23, 24)},
}


@dataclass
class Recording:
    animal: str
    sensor: int
    cap: np.ndarray
    time: np.ndarray
    lick_times: np.ndarray
    lick_indices: np.ndarray
    lick_vals: np.ndarray
    start_time: float
    t0_raw: float
    video_base: float
    video_path: str
    session_duration: float


def find_video_sensor(raw_h5):
    """Return (board_id, sensor_group_name, sensor_number) for the group that
    carries video sync metadata. Raises ValueError if none do."""
    for board_id, board in raw_h5.items():
        if not isinstance(board, h5py.Group):
            continue
        for sensor_name, group in board.items():
            if isinstance(group, h5py.Group) and "video_filename" in group:
                m = re.match(r"sensor_(\d+)$", sensor_name)
                if not m:
                    continue
                return board_id, sensor_name, int(m.group(1))
    raise ValueError("no sensor group with 'video_filename' found in h5")


def _resolve_start_stop(group):
    """Mirror filter_data: pick the highest-numbered start_time{N}/stop_time{N}
    pair (or the unnumbered pair), falling back to time_data ends."""
    pattern = re.compile(r"^start_time(\d+)?$")
    matches = {}
    for k in group.keys():
        m = pattern.match(k)
        if m:
            num = int(m.group(1)) if m.group(1) else -1
            matches[num] = k
    time_data = group["time_data"]
    if not matches:
        return float(time_data[0]), float(time_data[-1])
    num = max(matches)
    last_start = matches[num]
    start_time = float(group[last_start][()])
    stop_key = "stop" + last_start[5:]
    if stop_key in group:
        stop_time = float(group[stop_key][()])
    else:
        stop_time = float(time_data[-1])
    return start_time, stop_time


def load_recording(h5_path, layout_path, pts_txt_path, video_path):
    layout = pd.read_csv(layout_path, header=None, index_col=0)
    with h5py.File(h5_path, "r") as raw:
        board_id, sensor_name, sensor_number = find_video_sensor(raw)
        group = raw[board_id][sensor_name]
        t0_raw = float(group["time_data"][0])
        start_time, stop_time = _resolve_start_stop(group)
        video_frame_index = int(group["video_frame_index"][()])
    session_duration = stop_time - start_time

    animal = str(layout.loc[sensor_number].iloc[0])

    with tempfile.TemporaryDirectory() as td:
        filt_path = os.path.join(td, "filtered.h5")
        log_path = os.path.join(td, "filter.log")
        with h5py.File(h5_path, "r") as raw, h5py.File(filt_path, "w") as filt:
            filter_data(
                raw, filt, layout, log_path,
                algorithm="basic_threshold",
                recording_length=session_duration + 1.0,
            )
        with h5py.File(filt_path, "r") as filt:
            if animal not in filt:
                raise ValueError(f"filter_data produced no group for animal {animal!r}")
            g = filt[animal]
            cap = g["cap_data"][:]
            time = g["time_data"][:]
            lick_times = g["lick_times"][:] if "lick_times" in g else np.array([])
            lick_indices = (
                g["lick_indices"][:] if "lick_indices" in g else np.array([], dtype=int)
            )

    lick_indices = np.asarray(lick_indices, dtype=int)
    lick_vals = cap[lick_indices] if lick_indices.size else np.array([])

    pts_ns = np.loadtxt(pts_txt_path, dtype=np.int64)
    video_base = compute_video_base(pts_ns, video_frame_index)

    return Recording(
        animal=animal, sensor=sensor_number, cap=cap, time=time,
        lick_times=np.asarray(lick_times), lick_indices=lick_indices,
        lick_vals=lick_vals, start_time=start_time, t0_raw=t0_raw,
        video_base=video_base, video_path=video_path,
        session_duration=session_duration,
    )


def compute_video_base(pts_ns, frame_index):
    """Seconds between the sync frame's PTS and the first frame's PTS."""
    pts_ns = np.asarray(pts_ns)
    return float(pts_ns[frame_index] - pts_ns[0]) / 1e9


def video_sec(tau, video_base, start_time, t0_raw, sync_offset=0.0):
    """Video-file second for session-relative time ``tau`` (0 = start_time)."""
    return video_base + (start_time + tau - t0_raw) + sync_offset


def n_output_frames(start, end, fps):
    return int(round((end - start) * fps))


def frame_times(start, end, fps):
    return start + np.arange(n_output_frames(start, end, fps)) / fps


def window_mask(times, lo, hi):
    times = np.asarray(times)
    return (times >= lo) & (times <= hi)


def nearest_index(times, tau):
    times = np.asarray(times)
    if times.size == 0:
        raise ValueError("times is empty")
    i = int(np.searchsorted(times, tau))
    if i <= 0:
        return 0
    if i >= times.size:
        return times.size - 1
    # searchsorted lands on the right neighbor; pick the closer of i-1, i
    return i if abs(times[i] - tau) < abs(times[i - 1] - tau) else i - 1


class FrameGrabber:
    """Sequential RGB frame reader; fast-seeks to the clip start, then serves
    the frame nearest each requested (monotonically increasing) video second."""

    def __init__(self, video_path, clip_start_sec):
        self.clip_start = max(0.0, clip_start_sec)
        self._reader = imageio.get_reader(
            video_path, "ffmpeg",
            input_params=["-ss", f"{self.clip_start:.6f}"],
        )
        self.src_fps = float(self._reader.get_meta_data()["fps"])
        self._k = -1
        self._frame = None

    def get(self, video_sec):
        target_k = int(round((video_sec - self.clip_start) * self.src_fps))
        if target_k < 0:
            target_k = 0
        while self._k < target_k:
            try:
                self._frame = self._reader.get_next_data()
                self._k += 1
            except (IndexError, StopIteration):
                break
        return self._frame

    def close(self):
        self._reader.close()
