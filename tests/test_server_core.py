import pytest
from pi.server_core import CameraServer
from video import protocol


class FakeBackend:
    def __init__(self):
        self._active = False
        self.bookmarks = 0

    @property
    def is_active(self):
        return self._active

    def start_session(self, name):
        self._active = True
        self.name = name
        return f"{name}.mp4"

    def bookmark(self, sensor_id):
        self.bookmarks += 1
        return {"frame_index": self.bookmarks, "pts": 1.5, "pi_monotonic": 99.0}

    def stop_session(self):
        self._active = False
        return [{"name": f"{self.name}.mp4", "size": 10}]


@pytest.fixture
def server():
    return CameraServer(FakeBackend())


def test_ping(server):
    resp = server.handle(protocol.make_request(protocol.PING))
    assert resp["ok"] is True
    assert resp["status"] == "ready"


def test_start_then_bookmark(server):
    start = server.handle(protocol.make_request(protocol.START_SESSION, name="vid"))
    assert start == {"ok": True, "video_filename": "vid.mp4"}
    mark = server.handle(protocol.make_request(protocol.BOOKMARK, sensor_id=1))
    assert mark["ok"] is True
    assert mark["frame_index"] == 1
    assert mark["pts"] == 1.5


def test_start_requires_name(server):
    resp = server.handle(protocol.make_request(protocol.START_SESSION))
    assert resp["ok"] is False


def test_bookmark_without_session(server):
    resp = server.handle(protocol.make_request(protocol.BOOKMARK, sensor_id=1))
    assert resp == {"ok": False, "error": "no active session"}


def test_stop_without_session(server):
    resp = server.handle(protocol.make_request(protocol.STOP_SESSION))
    assert resp == {"ok": False, "error": "no active session"}


def test_unknown_command(server):
    resp = server.handle({"cmd": "NOPE"})
    assert resp["ok"] is False
    assert "unknown" in resp["error"].lower()
