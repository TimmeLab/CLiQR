"""_start_camera must only retain the client on success.

If the Pi camera fails to start, camera_client must stay None and
camera_video_filename must be cleared, so later per-sensor bookmarks are skipped
and stop_recording does not spawn a stop/fetch against a session that never
started (and no stale video filename is written into the next recording's HDF5).
"""
import pytest

from utils import state
import components.session_controls as sc


class _StubClient:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc

    def start_session(self, name):
        if self._exc is not None:
            raise self._exc
        return self._resp


@pytest.fixture(autouse=True)
def _reset():
    sc.camera_client = None
    state.camera_video_filename.set("")
    yield
    sc.camera_client = None
    state.camera_video_filename.set("")


def test_success_sets_client_and_filename(monkeypatch):
    client = _StubClient(resp={"ok": True, "video_filename": "clip.mp4"})
    monkeypatch.setattr(state, "make_camera_client", lambda timeout=None: client)

    sc._start_camera("clip")

    assert sc.camera_client is client
    assert state.camera_video_filename.value == "clip.mp4"


def test_not_ok_leaves_client_none_and_clears_filename(monkeypatch):
    state.camera_video_filename.set("stale_from_prev_session.mp4")
    client = _StubClient(resp={"ok": False, "error": "camera busy"})
    monkeypatch.setattr(state, "make_camera_client", lambda timeout=None: client)

    sc._start_camera("clip")

    assert sc.camera_client is None
    assert state.camera_video_filename.value == ""


def test_exception_leaves_client_none_and_clears_filename(monkeypatch):
    state.camera_video_filename.set("stale_from_prev_session.mp4")
    client = _StubClient(exc=OSError("connection refused"))
    monkeypatch.setattr(state, "make_camera_client", lambda timeout=None: client)

    sc._start_camera("clip")

    assert sc.camera_client is None
    assert state.camera_video_filename.value == ""
