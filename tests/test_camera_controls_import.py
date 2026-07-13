from utils import state


def test_card_is_callable():
    from components.camera_controls import CameraControlsCard, test_connection
    assert callable(CameraControlsCard)
    assert callable(test_connection)


def test_test_connection_sets_status_mock():
    state.camera_mock.set(True)
    test_connection_result = _run_test_connection()
    assert state.camera_status.value in ("connected", "disconnected")
    state.camera_mock.set(False)


def _run_test_connection():
    from components.camera_controls import test_connection
    return test_connection()
