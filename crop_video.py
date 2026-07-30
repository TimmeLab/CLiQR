"""Interactively pick the square crop region for a Pi video recording.

Shows the frame at the middle of the recording window, lets you drag a fixed-size
square over the region of interest, then writes the chosen box to
<video>_crop.json. make_sync_video.py reads that sidecar and slices each frame to
the box AT RENDER TIME — the crop is a pure spatial slice, never a re-encode, so
it cannot perturb the frame<->session timing (a crop re-encode regenerates the
video's PTS onto a fresh CFR grid, which slid every crop-first render against the
trace). This tool no longer produces a cropped video. See
docs/superpowers/specs/2026-07-15-video-crop-tool-design.md.
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
    clamp_origin,
    crop_params_path,
    probe_frame_rate,
    read_video_anchor,
    resolve_paths,
    session_clock,
    trim_window_seconds,
    write_crop_params,
)


def compute_crop_window(anchor, pts_ns, framerate):
    """Return (start_frame, stop_frame, start_sec, end_sec) covering the whole
    session. Frames are the indices into the original video; the seconds are
    CONTAINER seconds on the original video's CFR ``framerate``. Uses the same
    SessionClock (latency + drift) and framerate as make_sync_video's
    clip_trim_window, so crop and render select identical frames."""
    pts_ns = np.asarray(pts_ns)
    clock = session_clock(anchor, pts_ns)
    return trim_window_seconds(clock, pts_ns, 0.0, anchor.session_duration,
                               framerate)


def resolve_out_path(video, out, force):
    """Resolve where the crop-box JSON is written (default <base>_crop.json)."""
    if out is None:
        out = crop_params_path(video)
    if os.path.abspath(out) == os.path.abspath(video):
        raise ValueError(f"refusing to overwrite the source recording: {out}")
    if os.path.exists(out) and not force:
        raise ValueError(f"{out} exists; pass --force to overwrite")
    return out


def save_crop_params(out, x, y, size):
    """Persist the hand-positioned crop box to ``out`` (JSON)."""
    return write_crop_params(out, x, y, size)


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
        description="Pick the square crop region for a recording and save it to "
                    "<video>_crop.json (make_sync_video applies it at render time).")
    p.add_argument("--h5", required=True, help="raw recording .h5")
    p.add_argument("--video", default=None,
                   help="source video (default: from h5 video_filename)")
    p.add_argument("--pts-txt", dest="pts_txt", default=None,
                   help="per-frame PTS sidecar (default: from the h5's "
                        "video_filename, with .txt)")
    p.add_argument("--size", type=int, default=360,
                   help="side length of the square crop (default 360)")
    p.add_argument("--out", default=None,
                   help="crop-box JSON path (default: <video>_crop.json)")
    p.add_argument("--force", action="store_true",
                   help="overwrite the output if it exists")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH (needed to read the video)",
              file=sys.stderr)
        return 1
    try:
        anchor = read_video_anchor(args.h5)
        video, pts_txt = resolve_paths(args.h5, anchor, args.video, args.pts_txt)
        out = resolve_out_path(video, args.out, args.force)
        pts_ns = np.loadtxt(pts_txt, dtype=np.int64)
        sf, ef, start_sec, end_sec = compute_crop_window(
            anchor, pts_ns, probe_frame_rate(video))
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
        print(f"crop {args.size}x{args.size} @ ({x}, {y}) -> {out}")
        save_crop_params(out, x, y, args.size)
        print("done")
    except (ValueError, FileNotFoundError, KeyError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
