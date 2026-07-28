"""Every request must leave a timestamped trace on the Pi.

After a 2026-07-22 session where START_SESSION was refused and retried, the Pi
log was empty and there was nothing to diagnose from. Two causes: the server
logged almost nothing, and what it did log sat in a block buffer (Python's
stdout is NOT line-buffered when redirected to a file) so it never reached
disk while the process was alive. This covers the first; pi/run_server.sh's
`-u` / `stdbuf` covers the second.
"""
import logging

import pytest

from pi.server_core import CameraServer
from video import protocol


class _FakeBackend:
    is_active = False

    def __init__(self, fail=False):
        self.fail = fail

    def start_session(self, name):
        if self.fail:
            raise RuntimeError("camera busy")
        return f"{name}.mp4"

    def reclaim_disk_space(self):
        return {}


@pytest.fixture
def captured(caplog):
    caplog.set_level(logging.INFO, logger="pi.server_core")
    return caplog


def test_successful_request_is_logged(captured):
    core = CameraServer(_FakeBackend())
    core.handle(protocol.make_request(protocol.START_SESSION, name="clip"))
    assert "START_SESSION -> ok" in captured.text


def test_failed_request_logs_the_error_at_warning(captured):
    # This is the line that was missing when a refused/failed start left the
    # operator with nothing to look at.
    core = CameraServer(_FakeBackend(fail=True))
    resp = core.handle(protocol.make_request(protocol.START_SESSION, name="clip"))

    assert resp["ok"] is False
    assert "START_SESSION -> error: camera busy" in captured.text
    assert any(r.levelno == logging.WARNING for r in captured.records)


def test_unknown_command_is_logged(captured):
    core = CameraServer(_FakeBackend())
    core.handle(protocol.make_request("NONSENSE"))
    assert "NONSENSE -> error" in captured.text


def test_snapshot_payload_is_not_logged(captured):
    """The reply carries a base64 JPEG; only the outcome belongs in the log."""
    class _SnapBackend(_FakeBackend):
        def snapshot(self):
            return b"\xff\xd8ffffffffffffffffffff"

    core = CameraServer(_SnapBackend())
    resp = core.handle(protocol.make_request(protocol.SNAPSHOT))
    assert resp["ok"] is True
    assert "SNAPSHOT -> ok" in captured.text
    assert resp["image"] not in captured.text
