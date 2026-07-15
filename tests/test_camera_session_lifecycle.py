"""The Pi backend must release the camera on stop so the next session can
acquire it.

Regression for the observed field bug: after a first record/stop cycle without
restarting the server, the second start_session() failed with libcamera
"Camera in Configured state trying acquire() requiring state Available /
Camera __init__ sequence did not complete", leaving no active session, so the
per-sensor video bookmark returned "no active session".

Root cause: stop_session() never called Picamera2.close(), so the previous
instance kept the device acquired.
"""
import pytest

from pi.camera_backend import Picamera2Backend


class _FakeCamera:
    """Models the one physical camera: only one instance may hold it at a time.

    __init__ raises (like libcamera's failed acquire) if a live instance has
    not been close()d, so a leaked camera makes the next start_session fail.
    """
    _live = 0

    def __init__(self):
        if _FakeCamera._live > 0:
            raise RuntimeError("Camera __init__ sequence did not complete")
        _FakeCamera._live += 1
        self.closed = False
        self.recording = False
        self.pre_callback = None

    def create_video_configuration(self, **kwargs):
        return {}

    def configure(self, config):
        pass

    def start_recording(self, encoder, output):
        self.recording = True
        # Stand in for FfmpegOutput creating the mp4 (stop_session stats it).
        open(output, "wb").close()

    def stop_recording(self):
        self.recording = False

    def capture_metadata(self):
        return {"SensorTimestamp": 0}

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
        return str(video_path)  # _FakeCamera.start_recording touches this path


@pytest.fixture(autouse=True)
def _reset_camera_count():
    _FakeCamera._live = 0
    yield
    _FakeCamera._live = 0


def test_second_session_can_acquire_camera_after_stop(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))

    backend.start_session("clip1")
    assert backend.is_active
    backend.stop_session()
    assert not backend.is_active

    # The device must be free now; the second session must acquire it.
    backend.start_session("clip2")
    assert backend.is_active
    backend.stop_session()


def test_stop_session_releases_the_camera(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    cam = backend._picam2
    backend.stop_session()
    assert cam.closed is True
    assert _FakeCamera._live == 0


class _FakeRequest:
    def __init__(self, ts_ns):
        self._ts_ns = ts_ns

    def get_metadata(self):
        return {"SensorTimestamp": self._ts_ns}


def test_bookmark_index_and_pts_describe_same_frame(tmp_path):
    backend = _FakeBackend(output_dir=str(tmp_path))
    backend.start_session("clip")
    for ts_ns in (1_000_000, 2_000_000, 3_000_000):
        backend._on_frame(_FakeRequest(ts_ns))

    mark = backend.bookmark(sensor_id=5)

    # 3 frames logged (0-based indices 0,1,2). _on_frame writes the frame's ts
    # then increments, so the just-logged frame's 0-based sidecar index is
    # _frame_count - 1 = 2, and its ts is the reported pts. frame_index must
    # point at THAT frame, not the not-yet-written next one.
    assert mark["frame_index"] == 2
    assert mark["pts"] == 3_000_000 / 1e9
    backend.stop_session()
