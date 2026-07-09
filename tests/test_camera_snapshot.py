"""SNAPSHOT: grab one still JPEG from the Pi camera for a pre-recording
alignment check, without starting a recording session.

Idle-only: the backend opens the camera, captures one JPEG, releases it. The
server refuses a snapshot while a session is active (the camera is busy and the
alignment check is a pre-recording step). The JPEG travels back base64-encoded
inside the ordinary one-line JSON reply.
"""
import base64

import pytest

from pi.camera_backend import Picamera2Backend
from pi.server_core import CameraServer
from video import protocol


class _FakeStillCamera:
    """Models a camera opened just long enough to grab one still."""
    def __init__(self):
        self.started = False
        self.closed = False

    def create_still_configuration(self, **kwargs):
        return {}

    def configure(self, config):
        pass

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class _SnapshotBackend(Picamera2Backend):
    """Backend whose camera + capture are faked so snapshot() runs off-Pi."""
    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)
        self.last_camera = None

    def _create_camera(self):
        self.last_camera = _FakeStillCamera()
        return self.last_camera

    def _capture_jpeg(self, cam):
        return b"JPEGBYTES"


# ---- backend ---------------------------------------------------------------

def test_snapshot_returns_jpeg_bytes(tmp_path):
    backend = _SnapshotBackend(str(tmp_path))
    jpeg = backend.snapshot()
    assert jpeg == b"JPEGBYTES"


def test_snapshot_releases_the_camera(tmp_path):
    backend = _SnapshotBackend(str(tmp_path))
    backend.snapshot()
    # The one-shot camera must be closed so a later session can acquire it.
    assert backend.last_camera.closed is True
    # A snapshot must not leave a lingering session/camera behind.
    assert backend.is_active is False
    assert backend._picam2 is None


# ---- server_core dispatch --------------------------------------------------

def test_server_core_snapshot_returns_base64(tmp_path):
    server = CameraServer(_SnapshotBackend(str(tmp_path)))
    resp = server.handle(protocol.make_request(protocol.SNAPSHOT))
    assert resp["ok"] is True
    assert resp["format"] == "jpeg"
    assert base64.b64decode(resp["image"]) == b"JPEGBYTES"


def test_server_core_snapshot_refused_during_active_session(tmp_path):
    class _ActiveBackend(_SnapshotBackend):
        @property
        def is_active(self):
            return True

    server = CameraServer(_ActiveBackend(str(tmp_path)))
    resp = server.handle(protocol.make_request(protocol.SNAPSHOT))
    assert resp["ok"] is False
    assert "recording" in resp["error"].lower()


# ---- mock client -----------------------------------------------------------

def test_mock_client_snapshot_shape():
    from hardware.pi_camera_mock import MockPiCameraClient

    resp = MockPiCameraClient().snapshot()
    assert resp["ok"] is True
    assert resp["format"] == "jpeg"
    # Mock returns a real (tiny) JPEG so the GUI data URI renders.
    assert base64.b64decode(resp["image"])[:2] == b"\xff\xd8"  # JPEG SOI marker
