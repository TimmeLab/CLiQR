"""Render a side-by-side mouse-video + capacitance-trace demo clip.

Left panel: the mouse video. Right panel: the sensor's capacitance trace in a
sliding window with a centered dot marking the current time and markers on
detected licks. See docs/superpowers/specs/2026-07-14-sync-video-composite-design.md.
"""
import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import h5py
import imageio
import numpy as np
import pandas as pd

from data_analysis import filter_data
# Several trimcrop primitives are imported here purely to re-export them on the
# make_sync_video namespace (CLI + test convenience); this module is the public
# surface the sync-video tests poke at.
from video.trimcrop import (  # noqa: F401
    SessionClock,
    bookmark_latency,
    compute_trim_frames,
    compute_video_base,
    find_video_sensor,
    frame_session_times,
    probe_frame_session_times,
    probe_start_pts,
    read_session_window,
    read_video_anchor,
    resolve_paths,
    session_clock,
    subclip_copy,
    trim_and_crop,
    trim_window_seconds,
)


@dataclass
class Recording:
    animal: str
    sensor: int
    cap: np.ndarray
    time: np.ndarray
    lick_times: np.ndarray
    lick_indices: np.ndarray
    lick_vals: np.ndarray
    clock: SessionClock
    video_path: str
    session_duration: float
    pts_ns: np.ndarray


def read_session_duration(h5_path):
    """Session duration in seconds, without running filter_data, so the CLI can
    validate --start/--end before the expensive load."""
    start_time, stop_time = read_session_window(h5_path)
    return stop_time - start_time


def load_recording(h5_path, layout_path, pts_txt_path, video_path, anchor):
    layout = pd.read_csv(layout_path, header=None, index_col=0)
    session_duration = anchor.session_duration
    animal = str(layout.loc[anchor.sensor_number].iloc[0])

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
    clock = session_clock(anchor, pts_ns)

    return Recording(
        animal=animal, sensor=anchor.sensor_number, cap=cap, time=time,
        lick_times=np.asarray(lick_times), lick_indices=lick_indices,
        lick_vals=lick_vals,
        clock=clock, video_path=video_path,
        session_duration=session_duration, pts_ns=pts_ns,
    )


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


class TrimmedFrameSource:
    """Sequential RGB frame reader over a trimmed clip. Each source frame carries
    its true session time (from the PTS sidecar); ``get(target)`` returns the
    frame nearest ``target`` session-seconds. Targets must be non-decreasing."""

    def __init__(self, path, frame_sess):
        # Decode PASSTHROUGH ("-vsync 0"): this footage is VFR (coded 240 fps,
        # real ~120), and imageio's default reader forces CFR, duplicating frames
        # so its decode count exceeds the ffprobe pts list. Since frames are timed
        # by that pts list (frame_sess) but counted by this decode, any duplicate
        # slips the frame<->session mapping (~1 s per ~300 s, worse the longer the
        # clip). Passthrough yields exactly one decoded frame per pts entry.
        self._reader = imageio.get_reader(path, "ffmpeg",
                                          output_params=["-vsync", "0"])
        self._sess = np.asarray(frame_sess, dtype=float)
        self._j = -1
        self._frame = None

    def get(self, target_session):
        target_k = nearest_index(self._sess, target_session)
        while self._j < target_k:
            try:
                self._frame = self._reader.get_next_data()
                self._j += 1
            except (IndexError, StopIteration):
                break
        return self._frame

    def close(self):
        self._reader.close()


def clip_trim_window(rec, start, end):
    """Video-file window (start_frame, stop_frame, start_sec, end_sec) for the
    clip's session window [start, end].

    Latency and clock drift both live in rec.clock (the bookmark frame sits at
    τ=latency; frame times are scaled by the two-bookmark slope), shared with
    crop_video's compute_crop_window through trim_window_seconds so the crop and
    render windows cannot drift apart.
    """
    return trim_window_seconds(rec.clock, rec.pts_ns, start, end)


def render_clip(rec, start, end, out_path, fps=30.0, window=2.5, sync_offset=0.0,
                intermediate_path=None):
    """Render the side-by-side clip. First stream-copies the mouse video down to
    the clip window (intermediate file, kept) so we don't decode the whole
    recording, then composites from it: the left panel is the video frame, the
    right panel the sliding capacitance trace with a centered dot and lick
    markers. Each video frame is placed by its true session time (from its
    preserved PTS) — no seeking assumptions, no fps assumption — so video and
    trace stay aligned.

    The video panel shows whatever region ``rec.video_path`` already carries:
    crop it once with crop_video.py, or pass the uncropped recording for a
    full-frame panel.

    ``sync_offset`` is a residual manual nudge in seconds: increase it if the
    video still runs ahead of the trace.
    """
    taus = frame_times(start, end, fps)
    if taus.size == 0:
        raise ValueError("empty clip: check --start/--end/--fps")

    _, _, start_sec, end_sec = clip_trim_window(rec, start, end)

    if intermediate_path is None:
        intermediate_path = os.path.splitext(out_path)[0] + "_trimcrop.mp4"
    subclip_copy(rec.video_path, start_sec, end_sec, intermediate_path)

    # Time each trimmed frame by its real (preserved) PTS, not by seek position.
    frame_sess = probe_frame_session_times(intermediate_path, rec.clock)

    src = TrimmedFrameSource(intermediate_path, frame_sess)

    cap_min, cap_max = float(rec.cap.min()), float(rec.cap.max())
    pad = 0.05 * (cap_max - cap_min + 1.0)

    fig, (axv, axt) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(left=0.02, right=0.97, wspace=0.08)

    first_frame = src.get(start - sync_offset)
    im = axv.imshow(first_frame if first_frame is not None
                    else np.zeros((2, 2, 3), dtype=np.uint8))
    axv.axis("off")
    im_sized = first_frame is not None

    (line,) = axt.plot([], [], lw=0.8, color="tab:blue")
    (dot,) = axt.plot([], [], "o", color="red", markersize=6, zorder=5)
    markers = axt.scatter([], [], s=40, facecolors="none",
                          edgecolors="tab:orange", linewidths=1.5, zorder=4)
    axt.set_ylim(cap_min - pad, cap_max + pad)
    axt.set_xlabel("Time (s, session)")
    axt.set_ylabel("Capacitance")

    def update(i):
        nonlocal im_sized
        tau = float(taus[i])
        frame = src.get(tau - sync_offset)
        if frame is not None:
            im.set_data(frame)
            if not im_sized:
                im.set_extent((-0.5, frame.shape[1] - 0.5, frame.shape[0] - 0.5, -0.5))
                im_sized = True

        lo, hi = tau - window, tau + window
        m = window_mask(rec.time, lo, hi)
        line.set_data(rec.time[m], rec.cap[m])
        axt.set_xlim(lo, hi)

        ci = nearest_index(rec.time, tau)
        dot.set_data([tau], [rec.cap[ci]])

        if rec.lick_times.size:
            lm = window_mask(rec.lick_times, lo, hi)
            markers.set_offsets(np.c_[rec.lick_times[lm], rec.lick_vals[lm]]
                                if np.any(lm) else np.empty((0, 2)))
        return im, line, dot, markers

    anim = FuncAnimation(fig, update, frames=len(taus), blit=False)
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=fps))
    finally:
        plt.close(fig)
        src.close()


def validate_window(start, end, session_duration):
    if start < 0:
        raise ValueError(f"--start must be >= 0 (got {start})")
    if end <= start:
        raise ValueError(f"--end ({end}) must be greater than --start ({start})")
    if end > session_duration:
        raise ValueError(
            f"--end ({end}) exceeds session duration ({session_duration:.1f} s)")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Render a side-by-side mouse-video + capacitance-trace clip.")
    p.add_argument("--h5", required=True, help="raw recording .h5")
    p.add_argument("--layout", required=True, help="sensor->animal layout csv")
    p.add_argument("--start", type=float, required=True,
                   help="clip start, seconds since the Start bookmark")
    p.add_argument("--end", type=float, required=True,
                   help="clip end, seconds since the Start bookmark")
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--video", default=None,
                   help="mouse video (default: <base>_cropped.mp4 from "
                        "crop_video.py, else the uncropped recording)")
    p.add_argument("--pts-txt", dest="pts_txt", default=None,
                   help="per-frame PTS sidecar (default: from the h5's "
                        "video_filename, with .txt)")
    p.add_argument("--fps", type=float, default=30.0, help="output fps (default 30)")
    p.add_argument("--window", type=float, default=2.5,
                   help="trace half-window seconds (default 2.5)")
    p.add_argument("--sync-offset", dest="sync_offset", type=float, default=0.0,
                   help="manual nudge, seconds; increase if video runs ahead of "
                        "the trace (default 0)")
    p.add_argument("--intermediate", default=None,
                   help="path for the trimmed subclip (kept); "
                        "default: <out>_trimcrop.mp4")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH (needed to write the video)",
              file=sys.stderr)
        return 1
    try:
        anchor = read_video_anchor(args.h5)
        video, pts_txt = resolve_paths(args.h5, anchor, args.video, args.pts_txt,
                                       prefer_cropped=True)
        validate_window(args.start, args.end, anchor.session_duration)
        rec = load_recording(args.h5, args.layout, pts_txt, video, anchor)
        intermediate = args.intermediate or (os.path.splitext(args.out)[0] + "_trimcrop.mp4")
        print(f"animal {rec.animal} (sensor {rec.sensor}); clip "
              f"[{args.start:.1f}, {args.end:.1f}] s")
        print(f"  trimmed video -> {intermediate}")
        print(f"  composite -> {args.out}")
        render_clip(rec, args.start, args.end, args.out,
                    fps=args.fps, window=args.window, sync_offset=args.sync_offset,
                    intermediate_path=intermediate)
        print("done")
    except (ValueError, FileNotFoundError, KeyError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
