from utils import state
from hardware.pi_camera import PiCameraClient
from hardware.pi_camera_mock import MockPiCameraClient


def test_defaults():
    assert state.camera_enabled.value is False
    assert state.camera_port.value == 8770
    assert state.camera_sensor_id.value is None


def test_make_client_real_vs_mock():
    state.camera_mock.set(False)
    assert isinstance(state.make_camera_client(), PiCameraClient)
    state.camera_mock.set(True)
    assert isinstance(state.make_camera_client(), MockPiCameraClient)
    state.camera_mock.set(False)
