"""Render a side-by-side mouse-video + capacitance-trace demo clip.

Left panel: the mouse video. Right panel: the sensor's capacitance trace in a
sliding window with a centered dot marking the current time and markers on
detected licks. See docs/superpowers/specs/2026-07-14-sync-video-composite-design.md.
"""
import numpy as np


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
