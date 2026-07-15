"""Shared trim/crop primitives for the Pi video recordings.

Anchor math, PTS/session-time conversion, and the ffmpeg wrappers used by both
crop_video.py (one-time interactive trim+crop) and make_sync_video.py (per-clip
stream-copy subclip). See docs/superpowers/specs/2026-07-15-video-crop-tool-design.md.
"""
import re
import subprocess
from dataclasses import dataclass

import h5py
import numpy as np


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


def read_session_window(h5_path):
    """Return (start_time, stop_time) of the video sensor's last cycle."""
    with h5py.File(h5_path, "r") as raw:
        board_id, sensor_name, _ = find_video_sensor(raw)
        return _resolve_start_stop(raw[board_id][sensor_name])


def compute_video_base(pts_ns, frame_index):
    """Seconds between the sync frame's PTS and the first frame's PTS."""
    pts_ns = np.asarray(pts_ns)
    return float(pts_ns[frame_index] - pts_ns[0]) / 1e9


def bookmark_latency(host_before, host_after, start_time):
    """Seconds the bookmark round-trip lagged ``start_time``: the bookmarked frame
    was captured mid-round-trip (~midpoint of the host bracket), so the video
    leads the trace by this much and every frame's session time must be shifted
    later by it. Returns 0.0 when the bracket wasn't recorded (older recordings)."""
    if host_before is None or host_after is None:
        return 0.0
    return (float(host_before) + float(host_after)) / 2.0 - float(start_time)


def frame_session_times(pts_ns, video_base):
    """Session-relative time (0 = start_time bookmark) of every video frame."""
    pts_ns = np.asarray(pts_ns)
    return (pts_ns - pts_ns[0]) / 1e9 - video_base


def compute_trim_frames(pts_ns, video_base, start, end):
    """Inclusive (start_frame, stop_frame) of the video frames whose session
    time falls in [start, end]. Raises ValueError if the window has no frames."""
    sess = frame_session_times(pts_ns, video_base)
    idx = np.flatnonzero((sess >= start) & (sess <= end))
    if idx.size == 0:
        raise ValueError(
            f"no video frames fall in the requested window [{start}, {end}] s")
    return int(idx[0]), int(idx[-1])


def probe_frame_session_times(path, video_base):
    """Session-relative time of every frame in ``path``, read from the file's
    real presentation timestamps via ffprobe. Because trimmed clips keep their
    original PTS (``-copyts``), each frame self-reports its true video-second, so
    ``pts - video_base`` is its session time (0 = start_time bookmark)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{r.stderr[-800:]}")
    pts = np.array([float(tok.strip().rstrip(","))
                    for tok in r.stdout.split() if tok.strip().rstrip(",")])
    if pts.size == 0:
        raise RuntimeError(f"ffprobe found no frames in {path}")
    return pts - video_base


@dataclass
class VideoAnchor:
    """Everything needed to place the video against the trace, from one h5 open."""
    sensor_number: int
    video_filename: str
    video_frame_index: int
    start_time: float
    stop_time: float
    host_before: float | None
    host_after: float | None

    @property
    def session_duration(self):
        return self.stop_time - self.start_time

    @property
    def latency(self):
        return bookmark_latency(self.host_before, self.host_after, self.start_time)


def read_video_anchor(h5_path):
    """Read the video sensor's sync metadata and session window in one open."""
    with h5py.File(h5_path, "r") as raw:
        board_id, sensor_name, sensor_number = find_video_sensor(raw)
        group = raw[board_id][sensor_name]
        start_time, stop_time = _resolve_start_stop(group)
        fname = group["video_filename"][()]
        fname = fname.decode() if isinstance(fname, bytes) else str(fname)
        return VideoAnchor(
            sensor_number=sensor_number,
            video_filename=fname,
            video_frame_index=int(group["video_frame_index"][()]),
            start_time=start_time,
            stop_time=stop_time,
            host_before=(float(group["video_bookmark_host_before"][()])
                         if "video_bookmark_host_before" in group else None),
            host_after=(float(group["video_bookmark_host_after"][()])
                        if "video_bookmark_host_after" in group else None),
        )


def clamp_origin(x, y, frame_w, frame_h, size):
    """Clamp a proposed crop origin so the size x size box stays inside the frame,
    rounding each coordinate down to an even number (yuv420p needs even offsets).
    Rounding down only ever moves the box further inside. Raises ValueError if the
    box cannot fit."""
    if size > frame_w or size > frame_h:
        raise ValueError(
            f"crop size {size} exceeds frame {frame_w}x{frame_h}")
    x = int(min(max(x, 0), frame_w - size))
    y = int(min(max(y, 0), frame_h - size))
    return x - (x % 2), y - (y % 2)
