"""After a session stops, the Pi must guarantee >= 5 GB free disk for the
next run by deleting the oldest video recordings — but never the video that
was just recorded. If deleting every *other* video still leaves < 5 GB free,
the backend flags low_disk so the desktop GUI can tell the user to free up
space manually.
"""
import pytest

from pi.camera_backend import Picamera2Backend, MIN_FREE_BYTES
from pi.server_core import CameraServer
from video import protocol


class _FakeCamera:
    def __init__(self):
        self.pre_callback = None

    def create_video_configuration(self, **kwargs):
        return {}

    def configure(self, config):
        pass

    def start_recording(self, encoder, output):
        open(output, "wb").close()

    def stop_recording(self):
        pass

    def close(self):
        pass


class _DiskBackend(Picamera2Backend):
    """Fake disk: free space = capacity - bytes of files in output_dir."""

    def __init__(self, output_dir, capacity_bytes):
        super().__init__(output_dir=output_dir)
        self.capacity_bytes = capacity_bytes

    def _create_camera(self):
        return _FakeCamera()

    def _create_encoder(self):
        return object()

    def _create_output(self, video_path):
        return str(video_path)

    def _disk_free_bytes(self):
        used = sum(p.stat().st_size for p in self.output_dir.iterdir())
        return self.capacity_bytes - used


GB = 1024 ** 3


def _make_video(dir_path, name, size, mtime):
    mp4 = dir_path / f"{name}.mp4"
    with open(mp4, "wb") as fh:
        fh.truncate(size)
    txt = dir_path / f"{name}.txt"
    txt.write_text("0\n")
    import os
    os.utime(mp4, (mtime, mtime))
    return mp4, txt


def _run_session(backend, name="current"):
    backend.start_session(name)
    backend.stop_session()


def test_no_deletion_when_enough_space(tmp_path):
    backend = _DiskBackend(str(tmp_path), capacity_bytes=100 * GB)
    old_mp4, _ = _make_video(tmp_path, "old", size=1 * GB, mtime=100)
    _run_session(backend)

    result = backend.reclaim_disk_space()

    assert result["deleted"] == []
    assert result["low_disk"] is False
    assert old_mp4.exists()


def test_deletes_oldest_videos_until_5gb_free(tmp_path):
    # Capacity 8 GB, two old 2 GB videos -> free = 4 GB < 5 GB. Deleting the
    # single oldest video brings free to 6 GB; the newer old video survives.
    backend = _DiskBackend(str(tmp_path), capacity_bytes=8 * GB)
    oldest_mp4, oldest_txt = _make_video(tmp_path, "oldest", size=2 * GB, mtime=100)
    newer_mp4, _ = _make_video(tmp_path, "newer", size=2 * GB, mtime=200)
    _run_session(backend)

    result = backend.reclaim_disk_space()

    assert result["deleted"] == ["oldest.mp4"]
    assert result["low_disk"] is False
    assert not oldest_mp4.exists()
    assert not oldest_txt.exists()  # companion timestamps go with the video
    assert newer_mp4.exists()


def test_never_deletes_current_session_video(tmp_path):
    # Only the just-recorded video exists and free space is < 5 GB: the
    # current video must survive and low_disk must flag the manual cleanup.
    backend = _DiskBackend(str(tmp_path), capacity_bytes=1 * GB)
    _run_session(backend, name="current")

    result = backend.reclaim_disk_space()

    assert result["deleted"] == []
    assert result["low_disk"] is True
    assert (tmp_path / "current.mp4").exists()
    assert (tmp_path / "current.txt").exists()


def test_low_disk_after_deleting_all_old_videos(tmp_path):
    # Deleting every old video still leaves < 5 GB -> low_disk True, but the
    # current session's files are untouched.
    backend = _DiskBackend(str(tmp_path), capacity_bytes=4 * GB)
    old_mp4, _ = _make_video(tmp_path, "old", size=1 * GB, mtime=100)
    _run_session(backend)

    result = backend.reclaim_disk_space()

    assert result["deleted"] == ["old.mp4"]
    assert result["low_disk"] is True
    assert not old_mp4.exists()
    assert (tmp_path / "current.mp4").exists()


def test_min_free_bytes_is_5gb():
    assert MIN_FREE_BYTES == 5 * GB


def test_fresh_backend_reclaims_any_video(tmp_path):
    # After a crash (e.g. disk filled mid-run) the post-stop cleanup never
    # ran and the server restarts with no session: every leftover video is
    # then fair game, including the crashed run's.
    backend = _DiskBackend(str(tmp_path), capacity_bytes=4 * GB)
    crashed_mp4, crashed_txt = _make_video(tmp_path, "crashed", size=3 * GB, mtime=100)

    result = backend.reclaim_disk_space()

    assert result["deleted"] == ["crashed.mp4"]
    assert not crashed_mp4.exists()
    assert not crashed_txt.exists()


class _FakeServerBackend:
    def __init__(self):
        self._active = False

    @property
    def is_active(self):
        return self._active

    def start_session(self, name):
        self._active = True
        return f"{name}.mp4"

    def stop_session(self):
        self._active = False
        return [{"name": "vid.mp4", "size": 10}]

    def reclaim_disk_space(self):
        return {"deleted": ["old.mp4"], "low_disk": True, "free_bytes": 3 * GB}


def test_stop_session_response_reports_cleanup():
    server = CameraServer(_FakeServerBackend())
    server.handle(protocol.make_request(protocol.START_SESSION, name="vid"))

    resp = server.handle(protocol.make_request(protocol.STOP_SESSION))

    assert resp["ok"] is True
    assert resp["files"] == [{"name": "vid.mp4", "size": 10}]
    assert resp["deleted"] == ["old.mp4"]
    assert resp["low_disk"] is True
    assert resp["free_bytes"] == 3 * GB


def test_start_session_reclaims_disk_before_recording():
    # A crashed run never reaches the post-stop cleanup, so space must also
    # be reclaimed at session start — before the camera begins writing.
    class OrderBackend(_FakeServerBackend):
        def __init__(self):
            super().__init__()
            self.calls = []

        def start_session(self, name):
            self.calls.append("start")
            return super().start_session(name)

        def reclaim_disk_space(self):
            self.calls.append("reclaim")
            return super().reclaim_disk_space()

    backend = OrderBackend()
    server = CameraServer(backend)

    resp = server.handle(protocol.make_request(protocol.START_SESSION, name="vid"))

    assert backend.calls == ["reclaim", "start"]
    assert resp["ok"] is True
    assert resp["video_filename"] == "vid.mp4"
    assert resp["low_disk"] is True  # surfaced so the GUI can warn pre-run


class _CleanupExplodesBackend(_FakeServerBackend):
    def reclaim_disk_space(self):
        raise OSError("disk went away")


def test_cleanup_failure_does_not_break_stop_response():
    # The stop already succeeded and the desktop needs the file list to fetch
    # the recording; a cleanup error must not turn the response into an error.
    server = CameraServer(_CleanupExplodesBackend())
    server.handle(protocol.make_request(protocol.START_SESSION, name="vid"))

    resp = server.handle(protocol.make_request(protocol.STOP_SESSION))

    assert resp["ok"] is True
    assert resp["files"] == [{"name": "vid.mp4", "size": 10}]
