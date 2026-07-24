"""Shared trim/crop primitives for the Pi video recordings.

Anchor math, PTS/session-time conversion, and the ffmpeg wrappers used by both
crop_video.py (one-time interactive trim+crop) and make_sync_video.py (per-clip
stream-copy subclip). See docs/superpowers/specs/2026-07-15-video-crop-tool-design.md.
"""
import os
import re
import subprocess
import sys
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
            if not isinstance(group, h5py.Group):
                continue
            # A multi-cycle recording (Start/Stop restart) writes the bookmark
            # datasets suffixed with the cycle number, so match video_filename
            # OR video_filename{N} -- mirroring _resolve_cycle's suffix handling.
            if any(re.match(r"video_filename\d*$", k) for k in group.keys()):
                m = re.match(r"sensor_(\d+)$", sensor_name)
                if not m:
                    continue
                return board_id, sensor_name, int(m.group(1))
    raise ValueError("no sensor group with 'video_filename' found in h5")


def _resolve_cycle(group):
    """Mirror filter_data: pick the highest-numbered start_time{N} (or the
    unnumbered start_time) and return (start_key, cycle_suffix). The suffix is the
    one the video-bookmark datasets use — "" for cycle 0 (unnumbered start_time),
    str(N) for start_time{N} — so callers read a self-consistent cycle. Returns
    (None, "") when the group has no start_time at all."""
    pattern = re.compile(r"^start_time(\d+)?$")
    matches = {}
    for k in group.keys():
        m = pattern.match(k)
        if m:
            num = int(m.group(1)) if m.group(1) else -1
            matches[num] = k
    if not matches:
        return None, ""
    num = max(matches)
    return matches[num], ("" if num < 0 else str(num))


def _resolve_start_stop(group):
    """Mirror filter_data: pick the highest-numbered start_time{N}/stop_time{N}
    pair (or the unnumbered pair), falling back to time_data ends."""
    last_start, _ = _resolve_cycle(group)
    time_data = group["time_data"]
    if last_start is None:
        return float(time_data[0]), float(time_data[-1])
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


def _frame_host_time(host_after, pi_monotonic=None, pts=None):
    """Host wall-clock time the bookmarked frame was captured.

    ``bookmark()`` reads the most-recent frame when it RUNS on the Pi, which is at
    the END of the host round-trip (~``host_after``), not its midpoint: the
    round-trip delay is Pi-side (the call sits blocked/queued while the camera
    keeps capturing), so a midpoint assumption undercounts the lag by ~half the
    round-trip. ``pi_monotonic`` (Pi clock at bookmark exec) and ``pts`` (the
    grabbed frame's SensorTimestamp) share the Pi clock, so their difference is
    how long before exec the frame was captured; back that off ``host_after``.
    The only unmodelled term is the Pi->host response one-way latency (~ms).
    When the Pi clocks aren't recorded the gap is dropped (frame == host_after)."""
    gap = 0.0 if (pi_monotonic is None or pts is None) \
        else float(pi_monotonic) - float(pts)
    return float(host_after) - gap


def bookmark_latency(host_after, start_time, pi_monotonic=None, pts=None):
    """Seconds the bookmarked frame lagged ``start_time`` -- the amount the video
    leads the trace, which every frame's session time must be shifted later by.
    The frame's host time is ``_frame_host_time`` (end of the round-trip, not its
    midpoint). Returns 0.0 when ``host_after`` wasn't recorded (older recordings)."""
    if host_after is None:
        return 0.0
    return _frame_host_time(host_after, pi_monotonic, pts) - float(start_time)


@dataclass
class SessionClock:
    """Maps a VIDEO-FILE second (0-based: elapsed since the recording's first
    frame) to session time τ (host-seconds since start_time).
    τ(pts) = latency + slope·(pts − pts_start_sec).

    This 0-based domain is the one both consumers already live in: the mp4's own
    ffprobe PTS start at 0, and the SensorTimestamp sidecar is normalised by its
    first entry (frame_session_times) before it reaches here. (The absolute
    SensorTimestamp epoch — ~1e5 s — must NOT be fed in raw.)

    pts_start_sec : the Start-bookmark frame's elapsed seconds from the first
                    frame (== compute_video_base).
    latency       : host seconds the Start bookmark lagged start_time (0 when the
                    host bracket wasn't recorded); the Start frame sits at τ=latency.
    slope         : host-seconds per video-second from the two bookmarks (1.0 when
                    there's no Stop bookmark) — corrects video<->cap clock drift
                    across the session.
    """
    pts_start_sec: float
    latency: float
    slope: float

    def session_time(self, pts_sec):
        pts_sec = np.asarray(pts_sec, dtype=float)
        return self.latency + self.slope * (pts_sec - self.pts_start_sec)


def session_clock(anchor, pts_ns):
    """Build the SessionClock for a recording from its anchor and PTS sidecar.

    pts_start_sec is the bookmark frame's offset from the first frame
    (compute_video_base), keeping the clock in the 0-based video-file domain."""
    pts_ns = np.asarray(pts_ns)
    pts_start_sec = compute_video_base(pts_ns, anchor.video_frame_index)
    return SessionClock(pts_start_sec=pts_start_sec,
                        latency=anchor.latency,
                        slope=anchor.drift_slope(pts_ns))


def frame_session_times(clock, pts_ns):
    """Session time τ (0 = start_time) of every sidecar frame, drift-corrected.

    The sidecar holds ABSOLUTE SensorTimestamps; normalise by the first frame so
    the clock sees the 0-based video-file domain it expects (matching the mp4's
    own ffprobe PTS)."""
    pts_ns = np.asarray(pts_ns)
    return clock.session_time((pts_ns - pts_ns[0]) / 1e9)


def compute_trim_frames(clock, pts_ns, start, end):
    """Inclusive (start_frame, stop_frame) of the video frames whose session
    time falls in [start, end]. Raises ValueError if the window has no frames."""
    sess = frame_session_times(clock, pts_ns)
    idx = np.flatnonzero((sess >= start) & (sess <= end))
    if idx.size == 0:
        raise ValueError(
            f"no video frames fall in the requested window [{start}, {end}] s")
    return int(idx[0]), int(idx[-1])


TAIL_MARGIN = 0.3  # seconds of slack past the last in-window frame


def trim_window_seconds(clock, pts_ns, start, end, tail_margin=TAIL_MARGIN):
    """Resolve session window [start, end] to (start_frame, stop_frame, start_sec,
    end_sec). The seconds are REAL original-video-timeline seconds (raw PTS, NOT
    drift-scaled) — that is what trim_and_crop and subclip_copy seek by; only the
    frame *selection* is drift/latency-aware, via ``clock``.

    Both crop_video and make_sync_video route through this function with the SAME
    SessionClock so their windows cannot drift apart: if they did, the crop tool
    would trim to one window while the renderer placed frames using another, and
    every cropped video would silently misalign against its trace.
    """
    pts_ns = np.asarray(pts_ns)
    sf, ef = compute_trim_frames(clock, pts_ns, start, end)
    start_sec = float(pts_ns[sf] - pts_ns[0]) / 1e9
    end_sec = float(pts_ns[ef] - pts_ns[0]) / 1e9 + tail_margin
    return sf, ef, start_sec, end_sec


def encoded_sidecar_path(pts_txt_path):
    """Companion sidecar of per-ENCODED-frame SensorTimestamps -- one line per
    frame the encoder actually emitted, i.e. one per CONTAINER frame, written by
    the Pi output wrapper. ``<stem>.txt`` (every captured frame) -> ``<stem>``
    ``.encpts.txt``.

    When present it times container frames EXACTLY: the software encoder drops
    frames under load (the capture sidecar still logs their SensorTimestamps, so
    it slowly out-counts the CFR container), and this sidecar excludes exactly
    those drops, so container index k maps straight to its real capture time.
    When absent the capture sidecar is the fallback, which drifts by the dropped
    frames (grows with distance from the bookmark)."""
    stem, _ = os.path.splitext(pts_txt_path)
    return stem + ".encpts.txt"


def probe_frame_rate(path):
    """The container's constant frame rate (the ``-framerate`` the mp4 was muxed
    at, see ``pi/ffmpeg_output.py``), from ffprobe's ``r_frame_rate`` ("120/1")."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{r.stderr[-800:]}")
    tok = r.stdout.strip()
    num, _, den = tok.partition("/")
    return float(num) / float(den) if den else float(num)


def probe_frame_session_times(path, clock, container_pts_ns, framerate):
    """Session time τ of every frame in ``path``, timed by real SensorTimestamps
    rather than the mp4 container's own presentation timestamps.

    The container is muxed at a constant ``-framerate`` (``pi/ffmpeg_output.py``),
    so its PTS are evenly spaced and do NOT track the real capture clock: the
    sensor runs at a slightly off-nominal, faintly variable rate, so container
    time drifts from capture time — a fraction of a percent, but seconds over a
    long session, which slides the video against the trace (worse the later in the
    session the clip sits). The container's own docstring says as much: time from
    the sidecar, not the container.

    Because the container is CFR and ``-c:v copy`` keeps frames in emission order,
    each frame's PTS gives its container index (``round(pts * framerate)``), which
    selects that frame's SensorTimestamp from ``container_pts_ns`` -- one entry per
    CONTAINER frame. Prefer the encoded sidecar (``encoded_sidecar_path``): it has
    exactly the frames the encoder emitted, so index k is frame k even when the
    encoder dropped captured frames. The capture sidecar is the fallback and
    drifts by those drops. The resulting elapsed-since-first-frame seconds go
    through the SAME ``clock`` as ``frame_session_times``, so the ffprobe path and
    the sidecar path stay identical (drift/latency included). Trimmed clips keep
    original PTS (``-copyts``), so the index is into the full-recording array; a
    few frames past its end clamp to the last entry."""
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
    container_pts_ns = np.asarray(container_pts_ns)
    idx = np.clip(np.rint(pts * framerate).astype(int), 0, container_pts_ns.size - 1)
    real_sec = (container_pts_ns[idx] - container_pts_ns[0]) / 1e9
    return clock.session_time(real_sec)


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
    pi_monotonic: float | None = None
    video_pts: float | None = None
    stop_frame_index: int | None = None
    stop_host_before: float | None = None
    stop_host_after: float | None = None
    stop_pi_monotonic: float | None = None
    stop_pts: float | None = None

    @property
    def session_duration(self):
        return self.stop_time - self.start_time

    @property
    def latency(self):
        return bookmark_latency(self.host_after, self.start_time,
                                self.pi_monotonic, self.video_pts)

    def drift_slope(self, pts_ns) -> float:
        """host-seconds per video-second from the two bookmarks; 1.0 when the Stop
        bookmark (or the Start host bracket) is absent, or the fit is degenerate.

        Both endpoints use each bookmark's END-of-bracket host time (host_after,
        backed off the Pi capture->exec gap), mirroring ``bookmark_latency``."""
        if (self.stop_frame_index is None or self.stop_host_after is None
                or self.host_after is None):
            return 1.0
        pts_ns = np.asarray(pts_ns)
        pts_start = float(pts_ns[self.video_frame_index]) / 1e9
        pts_stop = float(pts_ns[self.stop_frame_index]) / 1e9
        if pts_stop == pts_start:
            return 1.0
        host_start = _frame_host_time(self.host_after, self.pi_monotonic,
                                      self.video_pts)
        host_stop = _frame_host_time(self.stop_host_after, self.stop_pi_monotonic,
                                     self.stop_pts)
        return (host_stop - host_start) / (pts_stop - pts_start)


def read_video_anchor(h5_path):
    """Read the video sensor's sync metadata and session window in one open.

    Every field is read from the SAME cycle _resolve_start_stop chose: the video
    bookmark (frame_index, host bracket, filename) is re-written each cycle with
    the same suffix as start_time{N}, so pairing a cycle-N window with a cycle-0
    frame index — as reading the unsuffixed keys would for a multi-cycle
    recording — misaligns the video against the trace."""
    with h5py.File(h5_path, "r") as raw:
        board_id, sensor_name, sensor_number = find_video_sensor(raw)
        group = raw[board_id][sensor_name]
        start_time, stop_time = _resolve_start_stop(group)
        _, suffix = _resolve_cycle(group)
        fname = group[f"video_filename{suffix}"][()]
        fname = fname.decode() if isinstance(fname, bytes) else str(fname)
        before_key = f"video_bookmark_host_before{suffix}"
        after_key = f"video_bookmark_host_after{suffix}"
        mono_key = f"video_pi_monotonic{suffix}"
        pts_key = f"video_pts{suffix}"
        stop_idx_key = f"video_stop_frame_index{suffix}"
        stop_before_key = f"video_stop_bookmark_host_before{suffix}"
        stop_after_key = f"video_stop_bookmark_host_after{suffix}"
        stop_mono_key = f"video_stop_pi_monotonic{suffix}"
        stop_pts_key = f"video_stop_pts{suffix}"

        def num(key):
            return float(group[key][()]) if key in group else None

        return VideoAnchor(
            sensor_number=sensor_number,
            video_filename=fname,
            video_frame_index=int(group[f"video_frame_index{suffix}"][()]),
            start_time=start_time,
            stop_time=stop_time,
            host_before=num(before_key),
            host_after=num(after_key),
            pi_monotonic=num(mono_key),
            video_pts=num(pts_key),
            stop_frame_index=(int(group[stop_idx_key][()])
                              if stop_idx_key in group else None),
            stop_host_before=num(stop_before_key),
            stop_host_after=num(stop_after_key),
            stop_pi_monotonic=num(stop_mono_key),
            stop_pts=num(stop_pts_key),
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


def probe_start_pts(path):
    """The input's first presentation timestamp, in seconds. Containers without a
    start_time report "N/A"; treat those as 0."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time",
         "-of", "csv=p=0", path],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{r.stderr[-800:]}")
    tok = r.stdout.strip()
    try:
        return float(tok)
    except ValueError:
        return 0.0


def trim_and_crop(video_path, start_sec, end_sec, out_path,
                  crop_x, crop_y, size, seek_margin=5.0):
    """Trim ``video_path`` to video-seconds [start_sec, end_sec] and crop a
    size x size square whose top-left corner is (crop_x, crop_y), writing a new
    file (the original is left untouched). Re-encodes, because a filter applies.

    Uses a coarse fast seek to ``start_sec - seek_margin`` and ``-copyts`` so the
    output frames keep their ORIGINAL presentation timestamps. ffmpeg's input
    ``-ss`` does not land frame-accurately on this footage, so we deliberately
    seek a little early and rely on the preserved PTS (read back with
    ``probe_frame_session_times``) to time each frame — never on where the seek
    landed. This reads the original recording, whose PTS start at 0, so the seek
    needs no start-PTS correction. Returns out_path. Raises RuntimeError if
    ffmpeg fails."""
    coarse = max(0.0, start_sec - seek_margin)
    vf = f"crop={size}:{size}:{crop_x}:{crop_y}"
    cmd = [
        "ffmpeg", "-y", "-ss", f"{coarse:.6f}", "-copyts", "-i", video_path,
        "-to", f"{end_sec:.6f}", "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg trim/crop failed:\n{r.stderr[-800:]}")
    return out_path


def subclip_copy(video_path, start_sec, end_sec, out_path, seek_margin=5.0):
    """Cut video-seconds [start_sec, end_sec] out of ``video_path`` by stream copy
    — no filter, no re-encode, near-instant. Used to seek cheaply into a long file
    without decoding everything before the window.

    ``start_sec``/``end_sec`` are ORIGINAL-timeline seconds. Input ``-ss`` is
    relative to the container's start_time, while ``-to`` under ``-copyts`` is
    absolute, so the seek subtracts the input's start PTS and the end does not.
    That subtraction is a no-op on the original recording (start_time 0) and is
    what makes the seek land on a cropped file (start_time = session start).

    Stream copy cuts at the keyframe at or before the seek target, so the output
    may carry frames earlier than ``start_sec``. That is harmless: consumers time
    frames by PTS and skip past them. Returns out_path."""
    start_pts = probe_start_pts(video_path)
    coarse = max(0.0, start_sec - start_pts - seek_margin)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{coarse:.6f}", "-copyts", "-i", video_path,
        "-to", f"{end_sec:.6f}", "-an", "-c", "copy", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg subclip failed:\n{r.stderr[-800:]}")
    return out_path


def cropped_path_for(video_path):
    """The conventional cropped sibling of a recording: <base>_cropped.mp4."""
    return os.path.splitext(video_path)[0] + "_cropped.mp4"


def resolve_paths(h5_path, anchor, video=None, pts_txt=None, prefer_cropped=False):
    """Resolve (video_path, pts_txt_path) for a recording.

    The PTS sidecar is ALWAYS <original-video-base>.txt, derived from the h5's
    video_filename — never from the resolved video path. A cropped file has no
    sidecar of its own and needs none: it carries the original PTS via -copyts,
    and probe_frame_session_times reads them back from the file itself.

    ``prefer_cropped`` picks <base>_cropped.mp4 when it exists, printing a note to
    stderr when it doesn't, so a forgotten crop degrades to a correct (just
    uncropped) render instead of an error.
    """
    base = os.path.join(os.path.dirname(h5_path), anchor.video_filename)
    if pts_txt is None:
        pts_txt = os.path.splitext(base)[0] + ".txt"
    if video is None:
        video = base
        if prefer_cropped:
            cropped = cropped_path_for(base)
            if os.path.exists(cropped):
                video = cropped
            else:
                print(f"note: using uncropped video {base} "
                      f"(no _cropped.mp4; run crop_video.py first)",
                      file=sys.stderr)
    return video, pts_txt
