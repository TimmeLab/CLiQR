"""Render a side-by-side mouse-video + capacitance-trace demo clip.

Left panel: the mouse video. Right panel: the sensor's capacitance trace in a
sliding window with a centered dot marking the current time and markers on
detected licks. See docs/superpowers/specs/2026-07-14-sync-video-composite-design.md.
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace

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
    CropBox,
    SessionClock,
    bookmark_latency,
    compute_trim_frames,
    compute_video_base,
    crop_frame,
    crop_params_path,
    encoded_sidecar_path,
    find_video_sensor,
    frame_session_times,
    probe_frame_rate,
    probe_frame_session_times,
    probe_start_pts,
    read_crop_params,
    read_session_window,
    read_video_anchor,
    resolve_paths,
    RIG_DRIFT_SLOPE,
    session_clock,
    subclip_copy,
    trim_and_crop,
    trim_window_seconds,
)


@dataclass
class Recording:
    animal: str
    date: str  # recording date (YYYY-MM-DD) parsed from the h5 filename
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
    # Per-CONTAINER-frame SensorTimestamps used to time trimmed frames: the Pi's
    # encoded sidecar (drops excluded) when present, else pts_ns (the capture
    # sidecar, which drifts by dropped frames). See load_container_pts.
    container_pts_ns: np.ndarray


# Residual, constant lead of the video over the trace that survives the bookmark
# latency + drift corrections: ~2 frames (2/120 s = 16.667 ms), video ahead. Seen
# to be the SAME across recordings with wildly different bookmark latencies
# (2026-07-24, latency 4.9 s; 2026-07-27, latency 0.18 s) and constant across a
# session, so it is a fixed capture->timestamp systematic, not part of the
# latency/drift model. Applied as the default render sync_offset (positive =
# delay the video), which zeroes it; override with --sync-offset per clip.
#
# The 120 here is the NOMINAL frame rate, used only to turn "2 frames" into a
# fixed time nudge (16.667 ms). It is deliberately NOT the true 120.0048 fps CFR
# rate that probe_frame_rate recovers: this is a constant time offset, and the
# 0.004% difference is 0.4 microseconds, far below any frame it could shift.
DEFAULT_SYNC_OFFSET = 2.0 / 120.0


def load_container_pts(pts_txt_path, capture_pts_ns):
    """Per-container-frame SensorTimestamps for timing trimmed frames.

    The Pi's ``<stem>.encpts.txt`` (one line per ENCODED frame, so encoder drops
    are already excluded) times container frames exactly; when it's absent -- every
    recording made before drop-marking -- fall back to the capture sidecar, which
    drifts by any dropped frames. Kept tiny and separate so the fallback is
    testable without a real recording."""
    enc = encoded_sidecar_path(pts_txt_path)
    if os.path.exists(enc):
        return np.loadtxt(enc, dtype=np.int64)
    return capture_pts_ns


def read_session_duration(h5_path):
    """Session duration in seconds, without running filter_data, so the CLI can
    validate --start/--end before the expensive load."""
    start_time, stop_time = read_session_window(h5_path)
    return stop_time - start_time


def _read_trace_from_combined(combined_h5_path, animal, cycle):
    """Read one animal/cycle's cap, time, and lick arrays straight out of a combined results file.

    The combined file (results_combined_*.h5) was produced by the SAME filter_data pipeline that
    load_recording would otherwise re-run, so the arrays are identical -- this just avoids
    re-analyzing every sensor in the raw recording when the answer is already on disk. Layout is:
        <animal>/<cycle>/{cap_data, time_data, lick_times, lick_indices, ...}
    Returns (cap, time, lick_times, lick_indices)."""
    with h5py.File(combined_h5_path, "r") as combined:
        if animal not in combined:
            raise ValueError(f"animal {animal!r} not found in combined file {combined_h5_path!r}")
        animal_group = combined[animal]
        cycle_key = str(cycle)
        if cycle_key not in animal_group:
            raise ValueError(f"cycle {cycle} not found for animal {animal!r} in combined file "
                             f"(have cycles {sorted(animal_group.keys())})")
        g = animal_group[cycle_key]
        cap = g["cap_data"][:]
        time = g["time_data"][:]
        lick_times = g["lick_times"][:] if "lick_times" in g else np.array([])
        lick_indices = (
            g["lick_indices"][:] if "lick_indices" in g else np.array([], dtype=int)
        )
    return cap, time, lick_times, lick_indices


def load_recording(h5_path, layout_path, pts_txt_path, video_path, anchor,
                   combined_h5=None, cycle=None, sync_slope=None):
    layout = pd.read_csv(layout_path, header=None, index_col=0)
    session_duration = anchor.session_duration
    animal = str(layout.loc[anchor.sensor_number].iloc[0])
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(h5_path))
    date = m.group(1) if m else ""

    if combined_h5 is not None:
        # Fast path: the trace already exists in the combined results file; read it directly
        # instead of re-running filter_data on the whole raw recording.
        cap, time, lick_times, lick_indices = _read_trace_from_combined(
            combined_h5, animal, cycle)
    else:
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
    if sync_slope is not None:
        clock = session_clock(anchor, pts_ns)
        clock = replace(clock, slope=float(sync_slope))
        print(f"  drift slope -> {clock.slope:.9f} (forced by --sync-slope)")
    else:
        clock = session_clock(anchor, pts_ns, fallback_slope=RIG_DRIFT_SLOPE)
        if anchor.stop_frame_index is None or anchor.stop_host_after is None:
            # Silence here would be the dangerous outcome: the clip still renders, and the
            # only symptom of a wrong slope is a lag that grows with session position.
            print(f"  WARNING no Stop bookmark on this recording (restart recording?): the "
                  f"two-bookmark drift fit is impossible, falling back to the rig's measured "
                  f"slope {RIG_DRIFT_SLOPE:.9f}. Pass --sync-slope to override.")
        else:
            print(f"  drift slope {clock.slope:.9f} (two-bookmark fit)")
    container_pts_ns = load_container_pts(pts_txt_path, pts_ns)

    return Recording(
        animal=animal, date=date, sensor=anchor.sensor_number, cap=cap, time=time,
        lick_times=np.asarray(lick_times), lick_indices=lick_indices,
        lick_vals=lick_vals,
        clock=clock, video_path=video_path,
        session_duration=session_duration, pts_ns=pts_ns,
        container_pts_ns=container_pts_ns,
    )


def source_fps(frame_sess):
    """Real capture fps of the trimmed clip, from its per-frame session times.
    The footage is VFR (coded 240, real ~120), so use the median inter-frame
    interval (robust to occasional gaps) rather than count/duration."""
    frame_sess = np.asarray(frame_sess, dtype=float)
    if frame_sess.size < 2:
        raise ValueError("cannot infer fps: clip has < 2 frames")
    dt = np.median(np.diff(frame_sess))
    if dt <= 0:
        raise ValueError("cannot infer fps: non-increasing frame times")
    return 1.0 / dt


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

    def __init__(self, path, frame_sess, crop=None):
        # Decode PASSTHROUGH ("-vsync 0"): this footage is VFR (coded 240 fps,
        # real ~120), and imageio's default reader forces CFR, duplicating frames
        # so its decode count exceeds the ffprobe pts list. Since frames are timed
        # by that pts list (frame_sess) but counted by this decode, any duplicate
        # slips the frame<->session mapping (~1 s per ~300 s, worse the longer the
        # clip). Passthrough yields exactly one decoded frame per pts entry.
        self._reader = imageio.get_reader(path, "ffmpeg",
                                          output_params=["-vsync", "0"])
        self._sess = np.asarray(frame_sess, dtype=float)
        # ``crop`` is a display-only spatial slice (CropBox or None). The FILE we
        # decode and time is the untouched original recording; cropping here, not
        # by re-encoding a _cropped.mp4, keeps the frame<->session mapping exactly
        # the working uncropped path's (a crop re-encode regenerates the PTS onto
        # a fresh CFR grid, which slides that mapping and drifts the render).
        self._crop = crop
        self._j = -1
        self._frame = None

    def get(self, target_session):
        target_k = nearest_index(self._sess, target_session)
        while self._j < target_k:
            try:
                self._frame = crop_frame(self._reader.get_next_data(), self._crop)
                self._j += 1
            except (IndexError, StopIteration):
                break
        return self._frame

    def close(self):
        self._reader.close()


def clip_trim_window(rec, start, end, framerate):
    """Video-file window (start_frame, stop_frame, start_sec, end_sec) for the
    clip's session window [start, end]. ``framerate`` is the original recording's
    true CFR rate (probe_frame_rate); the seconds are CONTAINER seconds on it.

    Latency and clock drift both live in rec.clock (the bookmark frame sits at
    τ=latency; frame times are scaled by the two-bookmark slope), shared with
    crop_video's compute_crop_window through trim_window_seconds so the crop and
    render windows cannot drift apart.
    """
    return trim_window_seconds(rec.clock, rec.pts_ns, start, end, framerate)


def load_crop(video_path, params_path=None, no_crop=False):
    """Resolve the display crop for a render, or None for a full frame.

    ``no_crop`` forces the full frame. Otherwise reads ``params_path`` when given,
    else the recording's conventional ``<base>_crop.json`` sidecar (written by
    crop_video.py); a missing sidecar means no crop was configured."""
    if no_crop:
        return None
    return read_crop_params(params_path or crop_params_path(video_path))


def render_clip(rec, start, end, out_path, fps=None, window=2.5,
                sync_offset=DEFAULT_SYNC_OFFSET, intermediate_path=None,
                crop=None, speed=1.0):
    """Render the side-by-side clip. First stream-copies the mouse video down to
    the clip window (intermediate file) so we don't decode the whole
    recording, then composites from it: the left panel is the video frame, the
    right panel the sliding capacitance trace with a centered dot and lick
    markers. Each video frame is placed by its true session time (from its
    preserved PTS) — no seeking assumptions, no fps assumption — so video and
    trace stay aligned.

    The video panel shows the ``crop`` region (a CropBox, sliced from each frame
    at display time) of the untouched original recording, or the full frame when
    ``crop`` is None. Pick the box once with crop_video.py (it writes the sidecar
    load_crop reads). The crop is a pure spatial slice — it never re-encodes, so
    it cannot perturb the frame<->session timing.

    ``sync_offset`` shifts the video in seconds (positive = delay it): each output
    time tau fetches the source frame at ``tau - sync_offset``. It defaults to
    ``DEFAULT_SYNC_OFFSET``, the measured constant ~2-frame (16.667 ms) residual
    lead the video keeps after the latency/drift corrections; that default zeroes
    the lead. Increase it further only if a given clip still runs ahead.

    ``fps`` None (default) renders at the footage's real capture rate, so no
    source frames are dropped; pass a number to force a different output rate.

    ``speed`` scales PLAYBACK only: the clip is still sampled at ``fps`` in session
    time, so every captured frame appears, but the container is written at
    ``fps * speed``. 0.25 on 120 fps footage gives quarter-speed slow motion at a
    30 fps container. Lowering ``fps`` instead would drop three of every four
    frames and leave the duration unchanged -- not slow motion. The trace panel
    slows with the video for free: it is redrawn per output frame at that frame's
    session time, so the two cannot drift apart.
    """
    # The clip window's seconds are CONTAINER seconds on the ORIGINAL recording's
    # CFR rate, which is what subclip_copy seeks by -- probe that rate here, not
    # the trimmed clip's (identical, but the trim doesn't exist yet).
    src_framerate = probe_frame_rate(rec.video_path)
    _, _, start_sec, end_sec = clip_trim_window(rec, start, end, src_framerate)

    if intermediate_path is None:
        intermediate_path = os.path.splitext(out_path)[0] + "_trimcrop.mp4"
    subclip_copy(rec.video_path, start_sec, end_sec, intermediate_path)

    # Time each trimmed frame by its real capture time: the container is CFR, so
    # its frame index (from the preserved PTS + the mux framerate) selects the
    # matching SensorTimestamp from the per-container-frame array. Not the
    # container PTS, which drift.
    framerate = probe_frame_rate(intermediate_path)
    frame_sess = probe_frame_session_times(intermediate_path, rec.clock,
                                           rec.container_pts_ns, framerate)

    if speed <= 0:
        raise ValueError(f"speed must be > 0 (got {speed})")

    if fps is None:
        fps = source_fps(frame_sess)

    taus = frame_times(start, end, fps)
    if taus.size == 0:
        raise ValueError("empty clip: check --start/--end/--fps")

    src = TrimmedFrameSource(intermediate_path, frame_sess, crop=crop)

    cap_min, cap_max = float(rec.cap.min()), float(rec.cap.max())
    pad = 0.05 * (cap_max - cap_min + 1.0)

    # imshow forces equal aspect, so the video axes keeps side/top gaps that
    # tight_layout can't remove; paint the figure + video axes dark grey so any
    # leftover whitespace reads as intentional background rather than white.
    dark_grey = "#2b2b2b"
    fig, (axv, axt) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.set_facecolor(dark_grey)
    axv.set_facecolor(dark_grey)

    first_frame = src.get(start - sync_offset)
    im = axv.imshow(first_frame if first_frame is not None
                    else np.zeros((2, 2, 3), dtype=np.uint8))
    axv.axis("off")
    # Shape the video axes box to the frame's aspect so imshow (equal aspect)
    # fills it exactly -- no left/right/top gaps to center against.
    if first_frame is not None:
        fh, fw = first_frame.shape[:2]
        axv.set_box_aspect(fh / fw)
    im_sized = first_frame is not None

    (line,) = axt.plot([], [], lw=0.8, color="tab:blue")
    (dot,) = axt.plot([], [], "o", color="red", markersize=6, zorder=5)
    markers = axt.scatter([], [], s=40, facecolors="none",
                          edgecolors="tab:orange", linewidths=1.5, zorder=4)
    axt.set_ylim(cap_min - pad, cap_max + pad)
    axt.set_xlabel("Time (s, session)")
    axt.set_ylabel("Capacitance")
    # trace labels/ticks/spines sit on the dark-grey figure margin -> make white
    axt.tick_params(colors="white")
    axt.xaxis.label.set_color("white")
    axt.yaxis.label.set_color("white")
    for spine in axt.spines.values():
        spine.set_color("white")
    title = " — ".join(p for p in (rec.animal, rec.date) if p)
    fig.suptitle(title, color="white", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

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
        anim.save(out_path, writer=FFMpegWriter(fps=fps * speed),
                  savefig_kwargs={"facecolor": dark_grey})
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
                   help="mouse video (default: the recording from the h5). The "
                        "crop is applied at display time, so pass the ORIGINAL "
                        "recording here, never a pre-cropped file")
    p.add_argument("--crop-params", dest="crop_params", default=None,
                   help="crop-box JSON from crop_video.py (default: the "
                        "recording's <base>_crop.json sidecar, if present)")
    p.add_argument("--no-crop", dest="no_crop", action="store_true",
                   help="render the full frame, ignoring any crop sidecar")
    p.add_argument("--pts-txt", dest="pts_txt", default=None,
                   help="per-frame PTS sidecar (default: from the h5's "
                        "video_filename, with .txt)")
    p.add_argument("--fps", type=float, default=None,
                   help="output fps (default: the footage's real capture rate, "
                        "so no source frames are dropped)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier (default 1.0 = real time). "
                        "0.25 writes the container at a quarter of the capture "
                        "rate -- 120 fps footage becomes 30 fps slow motion with "
                        "every frame kept. Video and trace slow together.")
    p.add_argument("--window", type=float, default=2.5,
                   help="trace half-window seconds (default 2.5)")
    p.add_argument("--sync-offset", dest="sync_offset", type=float,
                   default=DEFAULT_SYNC_OFFSET,
                   help="seconds to delay the video; increase if it runs ahead of "
                        "the trace (default %(default).4f s = 2 frames, the "
                        "measured constant residual lead)")
    p.add_argument("--sync-slope", dest="sync_slope", type=float, default=None,
                   help="force the video<->host drift slope (host-seconds per "
                        "video-second) instead of fitting it from the Start/Stop "
                        "bookmarks. Only needed when the fit is wrong or impossible; "
                        "a recording with no Stop bookmark already falls back to the "
                        f"rig's measured {RIG_DRIFT_SLOPE:.6f}. Unlike --sync-offset "
                        "this scales with session position, which is what a lag that "
                        "grows through the session needs")
    p.add_argument("--intermediate", default=None,
                   help="path for the trimmed subclip (implies "
                        "--keep-intermediate); default: a temp file, deleted "
                        "after rendering")
    p.add_argument("--keep-intermediate", dest="keep_intermediate",
                   action="store_true",
                   help="keep the trimmed subclip after rendering (default: "
                        "delete it); saved to <out>_trimcrop.mp4 unless "
                        "--intermediate gives a path")
    p.add_argument("--combined-h5", dest="combined_h5", default=None,
                   help="results_combined_*.h5 to read the animal's cap/time/lick data from "
                        "directly, instead of re-running filter_data on the raw --h5. Requires "
                        "--cycle. The trace is identical (both come from filter_data), but this "
                        "skips re-analyzing every sensor in the raw file.")
    p.add_argument("--cycle", type=int, default=None,
                   help="cycle (per-file subgroup) index within --combined-h5 for this recording; "
                        "required when --combined-h5 is given")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH (needed to write the video)",
              file=sys.stderr)
        return 1
    if args.combined_h5 is not None and args.cycle is None:
        print("error: --cycle is required when --combined-h5 is given", file=sys.stderr)
        return 1
    try:
        anchor = read_video_anchor(args.h5)
        video, pts_txt = resolve_paths(args.h5, anchor, args.video, args.pts_txt)
        validate_window(args.start, args.end, anchor.session_duration)
        rec = load_recording(args.h5, args.layout, pts_txt, video, anchor,
                             combined_h5=args.combined_h5, cycle=args.cycle,
                             sync_slope=args.sync_slope)
        crop = load_crop(video, args.crop_params, args.no_crop)
        keep = args.keep_intermediate or args.intermediate is not None
        if keep:
            intermediate = (args.intermediate
                            or os.path.splitext(args.out)[0] + "_trimcrop.mp4")
        else:
            fd, intermediate = tempfile.mkstemp(
                suffix="_trimcrop.mp4",
                dir=os.path.dirname(os.path.abspath(args.out)))
            os.close(fd)
        print(f"animal {rec.animal} (sensor {rec.sensor}); clip "
              f"[{args.start:.1f}, {args.end:.1f}] s")
        crop_note = (f"{crop.size}x{crop.size} @ ({crop.x}, {crop.y})"
                     if crop else "full frame")
        print(f"  crop -> {crop_note}")
        print(f"  trimmed video -> {intermediate}"
              f"{'' if keep else ' (temp, will delete)'}")
        print(f"  composite -> {args.out}")
        try:
            render_clip(rec, args.start, args.end, args.out,
                        fps=args.fps, window=args.window,
                        sync_offset=args.sync_offset,
                        intermediate_path=intermediate, crop=crop,
                        speed=args.speed)
        finally:
            if not keep and os.path.exists(intermediate):
                os.remove(intermediate)
        print("done")
    except (ValueError, FileNotFoundError, KeyError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
