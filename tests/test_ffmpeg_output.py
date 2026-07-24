"""The ffmpeg mux command must pin the input framerate and stay off the terminal.

Regression for the 2026-07-21 field failure: picamera2's stock FfmpegOutput runs
`ffmpeg -f h264 -i pipe:` with no input framerate, so ffmpeg synthesizes
presentation timestamps from packet arrival times. That produced ~150
"Non-monotonous DTS ... changing to" corrections over the run and then::

    [mp4] Packet duration: 6822877421 / dts: 9993760164 is out of range
    [mp4] pts has no value

after which the muxer rejected every packet. Video stopped 44 min into a
2 h 19 min session while the server kept running.
"""
from pathlib import Path

from pi import ffmpeg_output
from pi.camera_backend import TARGET_FPS


def test_input_framerate_is_pinned_before_the_input():
    cmd = ffmpeg_output.ffmpeg_command("/tmp/clip.mp4", 120)

    # -framerate must precede -i: after it, it would set the OUTPUT rate and
    # leave the h264 demuxer guessing timestamps from arrival times, which is
    # the bug. Before it, the demuxer stamps constant-frame-rate timestamps.
    assert "-framerate" in cmd
    assert cmd.index("-framerate") < cmd.index("-i")
    assert cmd[cmd.index("-framerate") + 1] == "120"


def test_command_uses_the_capture_framerate():
    cmd = ffmpeg_output.ffmpeg_command("/tmp/clip.mp4", TARGET_FPS)
    assert cmd[cmd.index("-framerate") + 1] == str(TARGET_FPS)


def test_input_is_a_raw_h264_stream_on_stdin_copied_to_mp4():
    cmd = ffmpeg_output.ffmpeg_command("/tmp/clip.mp4", 120)
    assert cmd[cmd.index("-f") + 1] == "h264"       # first -f is the input format
    assert cmd[cmd.index("-i") + 1] == "pipe:0"
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[-1] == "/tmp/clip.mp4"


def test_logging_is_quiet_and_non_interactive():
    # The rejection flood was ~240 lines/s at 120 fps. Quiet logging keeps it
    # small; -nostdin stops ffmpeg competing for the server's stdin.
    cmd = ffmpeg_output.ffmpeg_command("/tmp/clip.mp4", 120)
    assert cmd[cmd.index("-loglevel") + 1] == "error"
    assert "-nostdin" in cmd


def test_stderr_log_is_a_companion_file_of_the_video():
    # stderr must go to a file, never an inherited tty: a terminal that stops
    # draining blocks ffmpeg's write, which blocks its stdin reads, which
    # back-pressures picamera2's request loop into a camera stall.
    assert ffmpeg_output.log_path_for("/tmp/clip.mp4") == Path("/tmp/clip.ffmpeg.log")
    assert ffmpeg_output.log_path_for(Path("/tmp/clip_part2.mp4")).name == \
        "clip_part2.ffmpeg.log"


class _FakeProc:
    def __init__(self, wait_raises=False):
        self.stdin = _FakeStdin()
        self.waited = False
        self.killed = False
        self._wait_raises = wait_raises

    def wait(self, timeout=None):
        if self._wait_raises and not self.killed:
            raise TimeoutError("ffmpeg wedged")
        self.waited = True

    def kill(self):
        self.killed = True


class _FakeStdin:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_finish_muxer_closes_stdin_so_the_moov_atom_is_written():
    # ffmpeg only writes the moov atom when its stdin closes; skip this and the
    # segment is an unplayable file.
    proc = _FakeProc()
    ffmpeg_output.finish_muxer(proc)
    assert proc.stdin.closed is True
    assert proc.waited is True


def test_finish_muxer_kills_a_wedged_ffmpeg_instead_of_blocking_stop():
    # The capacitance data matters more than the video: STOP_SESSION must never
    # hang waiting on a stuck muxer.
    proc = _FakeProc(wait_raises=True)
    ffmpeg_output.finish_muxer(proc, timeout=0.01)
    assert proc.killed is True


def test_finish_muxer_tolerates_no_process():
    ffmpeg_output.finish_muxer(None)  # test backends never spawn one


def test_encoded_sensor_ns_restores_absolute_from_relative_us():
    # picamera2 hands outputframe a microsecond time relative to the segment's
    # first encoded frame (Pi-probed). Absolute ns = first_ns + timestamp_us*1000,
    # matching the capture .txt sidecar. Frame 0 (timestamp 0) == first_ns.
    first_ns = 1342773037684000
    assert ffmpeg_output.encoded_sensor_ns(first_ns, 0) == first_ns
    # probe frame 24: 199802 us -> +199802000 ns
    assert ffmpeg_output.encoded_sensor_ns(first_ns, 199802) == 1342773237486000
