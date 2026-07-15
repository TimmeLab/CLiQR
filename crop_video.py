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

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

from video.trimcrop import (
    TAIL_MARGIN,
    clamp_origin,
    compute_video_base,
    cropped_path_for,
    read_video_anchor,
    resolve_paths,
    trim_and_crop,
    trim_window_seconds,
)


def compute_crop_window(anchor, pts_ns):
    """Return (start_frame, stop_frame, start_sec, end_sec) covering the whole
    session. Frames are the indices into the original video; the seconds are on
    the original video's timeline. Uses the latency-corrected anchor, matching
    make_sync_video's clip_trim_window."""
    pts_ns = np.asarray(pts_ns)
    video_base_eff = compute_video_base(pts_ns, anchor.video_frame_index) - anchor.latency
    return trim_window_seconds(pts_ns, video_base_eff, 0.0, anchor.session_duration)


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


def load_preview_frame(video, frame_index):
    """The frame at ``frame_index`` of the original video, as an RGB array."""
    reader = imageio.get_reader(video, "ffmpeg")
    try:
        return reader.get_data(frame_index)
    finally:
        reader.close()


class CropSelector:
    """Drag a fixed-size square over a still frame. run() returns the chosen
    (x, y) origin, or None if the window was closed without pressing Crop."""

    def __init__(self, frame, size):
        self.frame = frame
        self.size = size
        self.h, self.w = frame.shape[:2]
        self.result = None
        self._grab = None  # (dx, dy) offset from the box origin to the cursor

        x0, y0 = clamp_origin((self.w - size) / 2, (self.h - size) / 2,
                              self.w, self.h, size)
        self.x, self.y = x0, y0

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.subplots_adjust(bottom=0.12)
        self.ax.imshow(frame)
        self.ax.axis("off")
        self.rect = Rectangle((self.x, self.y), size, size, fill=False,
                              lw=2, edgecolor="lime")
        self.ax.add_patch(self.rect)
        self._update_title()

        self.button = Button(self.fig.add_axes([0.82, 0.02, 0.13, 0.06]), "Crop")
        self.button.on_clicked(self._on_crop)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _update_title(self):
        self.ax.set_title(
            f"drag the box over the sipper, then press Crop   "
            f"[{self.size}x{self.size} @ ({self.x}, {self.y})]")

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if (self.x <= event.xdata <= self.x + self.size
                and self.y <= event.ydata <= self.y + self.size):
            self._grab = (event.xdata - self.x, event.ydata - self.y)

    def _on_motion(self, event):
        if self._grab is None or event.inaxes is not self.ax or event.xdata is None:
            return
        dx, dy = self._grab
        self.x, self.y = clamp_origin(event.xdata - dx, event.ydata - dy,
                                      self.w, self.h, self.size)
        self.rect.set_xy((self.x, self.y))
        self._update_title()
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self._grab = None

    def _on_crop(self, event):
        self.result = (self.x, self.y)
        plt.close(self.fig)

    def run(self):
        plt.show()
        return self.result


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
        mid_frame = (sf + ef) // 2
        frame = load_preview_frame(video, mid_frame)
        origin = CropSelector(frame, args.size).run()
        if origin is None:
            print("cancelled")
            return 0
        x, y = origin
        print(f"cropping {args.size}x{args.size} @ ({x}, {y}) -> {out}")
        trim_and_crop(video, start_sec, end_sec, out, x, y, args.size)
        print("done")
    except (ValueError, FileNotFoundError, KeyError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
