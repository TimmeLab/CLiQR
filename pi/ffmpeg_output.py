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


def spawn_muxer(video_path, framerate: int):
    """Start ffmpeg for ``video_path``; return (picamera2 Output, Popen).

    The returned Output writes the encoder's stream into ffmpeg's stdin. The
    caller owns shutdown: stop the recording, then call ``finish_muxer``.
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
    return FileOutput(proc.stdin), proc


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
