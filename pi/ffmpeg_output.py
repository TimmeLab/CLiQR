"""H.264 -> mp4 muxing for the Pi camera server, with explicit timestamps.

Replaces picamera2's stock ``FfmpegOutput``. That class pipes the encoder's
H.264 elementary stream to ``ffmpeg -f h264 -i pipe:`` *without* an input
framerate, so ffmpeg has to invent presentation timestamps from packet arrival
times. Every session logged ``Timestamps are unset in a packet for stream 0``
at startup and then drifted into ~150 ``Non-monotonous DTS ... changing to``
corrections over a two-hour run. On 2026-07-21 the guesswork failed outright::

    [mp4] Packet duration: 6822877421 / dts: 9993760164 is out of range
    [mp4] pts has no value

One packet arrived with a DTS ~6.8e9 ticks past its predecessor; the implied
duration exceeded the mp4 muxer's range, and from that packet on every frame
was rejected. Video ended 44 min into a 2 h 19 min session while the server
kept answering TCP requests.

Two independent causes, both addressed here:

1. **No input framerate.** ``-framerate`` is passed explicitly, so the h264
   demuxer assigns constant-frame-rate timestamps and never guesses. Nothing
   downstream loses accuracy: per-frame timing is read from the ``.txt``
   sidecar's SensorTimestamps, not the container (see ``video/trimcrop.py``),
   and the container timestamps this replaces were already corrupt.
2. **stderr on a terminal.** ffmpeg inherited the server's tty. At 120 fps the
   rejection flood is ~240 lines/s; a tty/ssh consumer that stops draining
   makes ffmpeg block on the stderr write, stop reading stdin, and back-pressure
   picamera2's request loop until the camera stalls. Here stderr goes to a
   per-session log file and ``-loglevel error`` keeps it near-empty.

The command builder is pure so it is testable off-hardware; only
``spawn_muxer`` touches subprocesses and picamera2.
"""
import subprocess
from pathlib import Path

# How long to wait for ffmpeg to drain its input and write the moov atom after
# stdin closes. Finalizing a multi-GB mp4 is a metadata-only operation, so this
# is generous; on timeout the process is killed rather than hanging the stop.
MUX_FINISH_TIMEOUT_S = 30.0


def ffmpeg_command(video_path, framerate: int) -> list:
    """Build the ffmpeg argv that muxes piped H.264 into ``video_path``.

    ``-framerate`` before ``-i`` applies to the h264 *demuxer*, which is what
    stamps the packets; passing it after would only set output frame rate and
    leave the arrival-time guessing in place.
    """
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "h264", "-framerate", str(framerate), "-i", "pipe:0",
        "-c:v", "copy", "-f", "mp4", str(video_path),
    ]


def log_path_for(video_path) -> Path:
    """Companion ffmpeg stderr log for a video (``clip.mp4`` -> ``clip.ffmpeg.log``)."""
    video_path = Path(video_path)
    return video_path.with_suffix(".ffmpeg.log")


def encoded_sensor_ns(first_sensor_ns: int, timestamp_us) -> int:
    """Absolute SensorTimestamp (ns) of an encoded frame.

    picamera2 hands ``Output.outputframe`` a ``timestamp`` in MICROSECONDS,
    relative to the segment's first encoded frame, on the SensorTimestamp clock
    (probed on the Pi: ``timestamp_us == (SensorTimestamp_ns - first_ns) / 1000``).
    Adding the segment's first SensorTimestamp restores absolute ns, so the
    ``.encpts.txt`` sidecar matches the capture ``.txt`` sidecar and the bookmark
    clock exactly."""
    return int(first_sensor_ns) + int(timestamp_us) * 1000


def make_sidecar_output(fileobj, encpts_fh, first_ns_getter):
    """A picamera2 Output that muxes to ``fileobj`` (ffmpeg's stdin) AND appends
    each ENCODED frame's absolute SensorTimestamp to ``encpts_fh`` -- one line per
    CONTAINER frame, so encoder drops are excluded and the desktop can time
    container frames exactly (see ``video/trimcrop.probe_frame_session_times``).

    ``first_ns_getter`` returns the segment's first SensorTimestamp (ns), set by
    the capture callback on frame 0 -- which always precedes that frame's encode,
    so it is populated before the first ``outputframe``. Defined here so the
    picamera2 import stays Pi-only. Sidecar writes never raise: a fault here must
    not kill the encoder thread (the video and trace matter more than the map)."""
    from picamera2.outputs import FileOutput

    class _SidecarOutput(FileOutput):
        def outputframe(self, frame, keyframe=True, timestamp=None,
                        *args, **kwargs):
            try:
                if (encpts_fh is not None and not encpts_fh.closed
                        and timestamp is not None):
                    base = first_ns_getter()
                    if base is not None:
                        encpts_fh.write(
                            f"{encoded_sensor_ns(base, timestamp)}\n")
            except Exception:  # noqa: BLE001 - never kill the encoder thread
                pass
            return super().outputframe(frame, keyframe, timestamp,
                                       *args, **kwargs)

    return _SidecarOutput(fileobj)


def spawn_muxer(video_path, framerate: int, encpts_fh=None, first_ns_getter=None):
    """Start ffmpeg for ``video_path``; return (picamera2 Output, Popen).

    The returned Output writes the encoder's stream into ffmpeg's stdin. When
    ``encpts_fh`` is given it ALSO logs each encoded frame's absolute
    SensorTimestamp there (one line per container frame; see
    ``make_sidecar_output``), so the desktop can time frames past encoder drops.
    The caller owns shutdown: stop the recording, then call ``finish_muxer``.
    """
    from picamera2.outputs import FileOutput

    log = open(log_path_for(video_path), "wb")
    try:
        proc = subprocess.Popen(
            ffmpeg_command(video_path, framerate),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
    finally:
        # ffmpeg holds its own dup of the fd; ours would otherwise leak per
        # session, and a long run may open several segments.
        log.close()
    if encpts_fh is not None:
        output = make_sidecar_output(proc.stdin, encpts_fh, first_ns_getter)
    else:
        output = FileOutput(proc.stdin)
    return output, proc


def finish_muxer(proc, timeout: float = MUX_FINISH_TIMEOUT_S) -> None:
    """Close ffmpeg's stdin and wait for it to finalize the mp4.

    Closing stdin is what makes ffmpeg write the moov atom; without it the file
    is unplayable. Best-effort: a wedged ffmpeg is killed rather than allowed to
    block STOP_SESSION, since the capacitance data matters more than the video.
    """
    if proc is None:
        return
    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except Exception:
        # Killing can itself fail (already reaped, permissions). Nothing here
        # may raise: this runs from the watchdog's segment restart and from
        # STOP_SESSION, and neither can afford to lose the session over it.
        try:
            proc.kill()
            proc.wait(timeout=5.0)
        except Exception:
            pass
