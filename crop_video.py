"""Interactively crop a Pi video recording to a square region and trim it to the
capacitance-recording window.

Shows the frame at the middle of the recording window, lets you drag a fixed-size
square over the region of interest, then writes <video>_cropped.mp4 — trimmed to
the session and cropped to the square. make_sync_video.py picks that file up
automatically. See docs/superpowers/specs/2026-07-15-video-crop-tool-design.md.
"""
import argparse
import os
import shutil
import sys

import numpy as np

from video.trimcrop import (
    compute_trim_frames,
    compute_video_base,
    cropped_path_for,
    read_video_anchor,
    resolve_paths,
    trim_and_crop,
)

TAIL_MARGIN = 0.3  # seconds of slack past the last in-window frame


def compute_crop_window(anchor, pts_ns):
    """Return (start_frame, stop_frame, start_sec, end_sec) covering the whole
    session. Frames are the indices into the original video; the seconds are on
    the original video's timeline. Uses the latency-corrected anchor, matching
    make_sync_video's render_clip."""
    pts_ns = np.asarray(pts_ns)
    video_base_eff = compute_video_base(pts_ns, anchor.video_frame_index) - anchor.latency
    sf, ef = compute_trim_frames(pts_ns, video_base_eff, 0.0, anchor.session_duration)
    start_sec = float(pts_ns[sf] - pts_ns[0]) / 1e9
    end_sec = float(pts_ns[ef] - pts_ns[0]) / 1e9 + TAIL_MARGIN
    return sf, ef, start_sec, end_sec


def reject_cropped_input(video):
    """Refuse to crop a file that is already a crop."""
    if os.path.splitext(video)[0].endswith("_cropped"):
        raise ValueError(f"refusing to crop an already-cropped video: {video}")


def resolve_out_path(video, out, force):
    if out is None:
        out = cropped_path_for(video)
    if os.path.exists(out) and not force:
        raise ValueError(f"{out} exists; pass --force to overwrite")
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Trim a recording to its capacitance window and crop it to a "
                    "hand-positioned square.")
    p.add_argument("--h5", required=True, help="raw recording .h5")
    p.add_argument("--video", default=None,
                   help="source video (default: from h5 video_filename)")
    p.add_argument("--pts-txt", dest="pts_txt", default=None,
                   help="per-frame PTS sidecar (default: video path with .txt)")
    p.add_argument("--size", type=int, default=360,
                   help="side length of the square crop (default 360)")
    p.add_argument("--out", default=None,
                   help="output path (default: <video>_cropped.mp4)")
    p.add_argument("--force", action="store_true",
                   help="overwrite the output if it exists")
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
                                       prefer_cropped=False)
        reject_cropped_input(video)
        out = resolve_out_path(video, args.out, args.force)
        pts_ns = np.loadtxt(pts_txt, dtype=np.int64)
        sf, ef, start_sec, end_sec = compute_crop_window(anchor, pts_ns)
        print(f"animal sensor {anchor.sensor_number}; session "
              f"{anchor.session_duration:.1f} s -> frames {sf}..{ef} "
              f"({start_sec:.2f}..{end_sec:.2f} s of video)")
        raise NotImplementedError("GUI wired up in Task 6")
    except (ValueError, FileNotFoundError, KeyError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
