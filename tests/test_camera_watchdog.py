"""A stalled camera must be detected and recording restarted, not lost silently.

Regression for the 2026-07-21 field failure: the Pi stopped delivering frames
44 min into a 2 h 19 min session. Nothing noticed. The server kept answering
TCP, STOP_SESSION succeeded, the mp4 finalized cleanly, and the Stop bookmark
reported a frame that was 5414 s stale. 90 min of video was gone and the
session looked clean.

The watchdog polls the frame counter and restarts recording into a `_partN`
segment when it stops advancing, so a stall costs seconds instead of the rest
of the run — and never fails the session, because the capacitance trace is what
must survive.
"""
import time

import pytest

from pi import camera_backend
from pi.camera_backend import Picamera2Backend, segment_stem


@pytest.fixture(autouse=True)
def _fast_watchdog(monkeypatch):
    """Shrink the watchdog timings so tests take milliseconds, not seconds."""
    monkeypatch.setattr(camera_backend, "STALL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(camera_backend, "WATCHDOG_POLL_S", 0.01)


class _FakeCamera:
    _live = 0

    def __init__(self):
        if _FakeCamera._live > 0:
            raise RuntimeError("Camera __init__ sequence did not complete")
        _FakeCamera._live += 1
        self.closed = False
        self.pre_callback = None

    def create_video_configuration(self, **kwargs):
        return {}

    def configure(self, config):
        pass

    def start_recording(self, encoder, output):
        open(output, "wb").close()  # stands in for ffmpeg creating the mp4

    def stop_recording(self):
        pass

    def close(self):
        if not self.closed:
            self.closed = True
            _FakeCamera._live -= 1


class _FakeBackend(Picamera2Backend):
    def _create_camera(self):
        return _FakeCamera()

    def _create_encoder(self):
        return object()

    def _create_output(self, video_path):
        return str(video_path)


class _FakeRequest:
    def __init__(self, ts_ns):
        self._ts_ns = ts_ns

    def get_metadata(self):
        return {"SensorTimestamp": self._ts_ns}


@pytest.fixture(autouse=True)
def _reset_camera_count():
    _FakeCamera._live = 0
    yield
    _FakeCamera._live = 0


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_segment_stem_keeps_the_plain_name_for_the_first_segment():
    # The normal case must produce exactly the filenames it always did.
    assert segment_stem("raw_data_2026-07-21_12-59-50", 1) == \
        "raw_data_2026-07-21_12-59-50"
    assert segment_stem("clip", 2) == "clip_part2"
    assert segment_stem("clip", 3) == "clip_part3"


def test_stalled_camera_restarts_into_a_new_segment(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    backend._on_frame(_FakeRequest(1_000_000))

    # Frames stop arriving. The watchdog must notice and open segment 2.
    assert _wait_for(lambda: backend._segment >= 2), "watchdog never restarted"

    assert backend.is_active, "a stall must never end the session"
    assert (tmp_path / "clip_part2.mp4").exists()
    assert (tmp_path / "clip_part2.txt").exists()
    backend.stop_session()


def test_frames_after_a_restart_land_in_the_new_segment(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    backend._on_frame(_FakeRequest(1_000_000))
    assert _wait_for(lambda: backend._segment >= 2)

    # The new segment's sidecar indexes from 0 again; alignment survives because
    # SensorTimestamps are absolute Pi boot-clock nanoseconds.
    backend._on_frame(_FakeRequest(9_000_000))
    mark = backend.bookmark(sensor_id=5)
    assert mark["frame_index"] == 0
    assert mark["pts"] == 9_000_000 / 1e9
    assert mark["video_filename"] == "clip_part2.mp4"

    backend.stop_session()
    assert (tmp_path / "clip.txt").read_text() == "1000000\n"
    assert (tmp_path / "clip_part2.txt").read_text() == "9000000\n"


def test_stop_session_lists_every_segment(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    backend._on_frame(_FakeRequest(1_000_000))
    assert _wait_for(lambda: backend._segment >= 2)

    names = {f["name"] for f in backend.stop_session()}
    # The desktop fetches exactly what STOP_SESSION lists; a segment missing
    # here is a segment left stranded on the Pi.
    assert {"clip.mp4", "clip.txt", "clip_part2.mp4", "clip_part2.txt"} <= names


def test_stalls_are_reported_so_the_run_does_not_look_clean(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    backend._on_frame(_FakeRequest(1_000_000))
    assert _wait_for(lambda: backend._segment >= 2)
    backend.stop_session()

    assert backend.stalls, "a stall must be reported, not swallowed"
    first = backend.stalls[0]
    assert first["segment"] == 1
    assert first["frames"] == 1
    assert first["idle_seconds"] >= camera_backend.STALL_TIMEOUT_S


def test_no_restart_while_frames_keep_arriving(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    deadline = time.monotonic() + 0.4  # many watchdog polls
    ts = 1_000_000
    while time.monotonic() < deadline:
        backend._on_frame(_FakeRequest(ts))
        ts += 8_333_333
        time.sleep(0.005)

    assert backend._segment == 1, "healthy recording must not be restarted"
    backend.stop_session()


def test_no_restart_during_warmup_before_the_first_frame(tmp_path):
    # Camera warmup legitimately takes a second or two. Restarting during it
    # would loop forever instead of surfacing the real failure.
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    time.sleep(0.3)
    assert backend._segment == 1
    assert backend.stalls == []
    backend.stop_session()


def test_watchdog_stops_with_the_session(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    backend._on_frame(_FakeRequest(1_000_000))
    backend.stop_session()

    time.sleep(0.2)
    assert backend._segment == 1, "watchdog must not restart a stopped session"
    assert _FakeCamera._live == 0


class _ExplodingRequest:
    def get_metadata(self):
        raise OSError("no space left on device")


def test_frame_callback_never_raises_into_picamera2(tmp_path):
    # pre_callback runs inside picamera2's request loop: an exception escaping
    # it kills that thread, stopping frame delivery AND encoding while the
    # server keeps answering TCP. That is the exact silent-death shape of the
    # 2026-07-21 failure, so the callback must swallow everything.
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")

    backend._on_frame(_ExplodingRequest())  # must not raise

    assert backend._frame_count == 0, "a failed frame must not consume an index"
    assert backend._frame_errors == 1
    backend.stop_session()


def test_frames_arriving_mid_restart_are_dropped_not_misfiled(tmp_path):
    # Between closing one segment's sidecar and opening the next, a frame
    # belongs to no file. Dropping it keeps frame_index aligned with sidecar
    # line numbers, which is what the video<->trace anchor depends on.
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    backend._on_frame(_FakeRequest(1_000_000))
    backend._close_segment()

    backend._on_frame(_FakeRequest(2_000_000))
    assert backend._frame_count == 1, "dropped frame must not advance the index"

    backend._open_segment()
    backend._on_frame(_FakeRequest(3_000_000))
    backend.stop_session()
    assert (tmp_path / "clip.txt").read_text() == "1000000\n"
    assert (tmp_path / "clip_part2.txt").read_text() == "3000000\n"


class _DeadProc:
    """An ffmpeg that has already exited (as a full disk would leave it)."""

    def __init__(self):
        self.stdin = None

    def poll(self):
        return 1

    def wait(self, timeout=None):
        return 1

    def kill(self):
        pass


class _DeadMuxerBackend(_FakeBackend):
    def _create_output(self, video_path):
        self._mux_proc = _DeadProc()
        return str(video_path)


def test_dead_muxer_restarts_the_segment_even_while_frames_flow(tmp_path):
    # A full disk kills ffmpeg, not the camera: frames keep reaching the
    # sidecar while the mp4 is silently truncated. Frame staleness alone would
    # never notice, so the watchdog polls the muxer process too.
    backend = _DeadMuxerBackend(output_dir=str(tmp_path))
    backend.start_session("clip")

    assert _wait_for(lambda: backend._segment >= 2), "dead muxer not detected"
    assert backend.stalls[0]["reason"] == "muxer exited"
    backend.stop_session()


def test_segment_cap_stops_the_restart_loop(tmp_path, monkeypatch):
    # Video bytes stay bitrate-bound however often we restart, but each segment
    # costs a .mp4 + .txt + .ffmpeg.log that the desktop fetches one at a time,
    # and every segment is protected from disk reclaim until the session ends.
    monkeypatch.setattr(camera_backend, "MAX_SEGMENTS", 3)
    backend = _DeadMuxerBackend(output_dir=str(tmp_path))
    backend.start_session("clip")

    assert _wait_for(lambda: backend._segment_cap_reached)
    time.sleep(0.2)  # several more polls
    assert backend._segment == 3
    assert backend.is_active, "hitting the cap must not end the session"
    assert not (tmp_path / "clip_part4.mp4").exists()
    backend.stop_session()


def test_runaway_ffmpeg_log_is_truncated(tmp_path):
    # The 2026-07-21 flood was ~240 lines/s (~170 MB over a full session).
    # That used to land on a terminal; now it lands on the disk the video needs.
    backend = _DeadMuxerBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    log_path = tmp_path / "clip.ffmpeg.log"
    log_path.write_bytes(b"x" * (camera_backend.FFMPEG_LOG_MAX_BYTES + 1))

    backend._cap_ffmpeg_log()

    assert log_path.stat().st_size == 0
    assert backend.ffmpeg_log_overflows == 1
    backend.stop_session()


def test_small_ffmpeg_log_is_left_alone(tmp_path):
    backend = _DeadMuxerBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    log_path = tmp_path / "clip.ffmpeg.log"
    log_path.write_bytes(b"one real error\n")

    backend._cap_ffmpeg_log()

    assert log_path.read_bytes() == b"one real error\n"
    assert backend.ffmpeg_log_overflows == 0
    backend.stop_session()


def test_low_disk_while_recording_is_flagged(tmp_path, monkeypatch):
    # Reclaim runs only at session start and after stop, so nothing frees space
    # mid-run; the operator has to be told before the disk fills.
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    monkeypatch.setattr(_FakeBackend, "_disk_free_bytes", lambda self: 100)

    backend._warn_if_disk_low()

    assert backend.low_disk_during_run is True
    backend.stop_session()


def test_empty_ffmpeg_log_is_not_transferred(tmp_path):
    # A healthy run fetches the video plus both sidecars (capture .txt and the
    # per-encoded-frame .encpts.txt), but not an empty ffmpeg log.
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    (tmp_path / "clip.ffmpeg.log").write_bytes(b"")

    names = {f["name"] for f in backend.stop_session()}
    assert names == {"clip.mp4", "clip.txt", "clip.encpts.txt"}


def test_non_empty_ffmpeg_log_is_transferred_for_diagnosis(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    (tmp_path / "clip.ffmpeg.log").write_bytes(b"[mp4] pts has no value\n")

    names = {f["name"] for f in backend.stop_session()}
    assert "clip.ffmpeg.log" in names


def test_restarting_over_a_live_session_leaves_one_watchdog(tmp_path):
    # The desktop abandons a START_SESSION that takes longer than its 10 s
    # timeout and sets its client to None, which also skips STOP_SESSION on the
    # next Stop -- so the operator's next Start arrives with the previous
    # session still recording. Without an explicit teardown the old watchdog
    # thread survives and a second one is started alongside it, both polling
    # and both able to restart segments.
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip1")
    first_watchdog = backend._watchdog

    backend.start_session("clip2")

    assert first_watchdog is not backend._watchdog
    assert not first_watchdog.is_alive(), "previous watchdog thread leaked"
    assert backend._segment == 1
    assert backend._video_path.name == "clip2.mp4"
    assert _FakeCamera._live == 1, "previous camera must have been released"
    backend.stop_session()


def test_reclaim_never_deletes_the_current_sessions_segments(tmp_path, monkeypatch):
    backend = _FakeBackend(output_dir=str(tmp_path))
    (tmp_path / "old.mp4").write_bytes(b"x")
    backend.start_session("clip")
    backend._on_frame(_FakeRequest(1_000_000))
    assert _wait_for(lambda: backend._segment >= 2)

    # Pretend the disk is full so reclaim deletes everything it is willing to.
    monkeypatch.setattr(_FakeBackend, "_disk_free_bytes", lambda self: 0)
    result = backend.reclaim_disk_space()

    assert "old.mp4" in result["deleted"]
    assert (tmp_path / "clip.mp4").exists()
    assert (tmp_path / "clip_part2.mp4").exists()
    backend.stop_session()
