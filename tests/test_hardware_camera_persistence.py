"""Hardware and camera reactives must ride the same session-persistence
mechanism as the recording lifecycle, so a browser refresh (a fresh Solara
kernel context) restores initialized hardware, the video controls, and the
bookmark-gate values instead of showing defaults."""
import inspect

from utils import state


# The full set of keys this feature adds to the persistence mechanism.
NEW_PERSISTED_KEYS = [
    "boards_connected",
    "i2c_controllers",
    "camera_enabled",
    "camera_sensor_id",
    "camera_host",
    "camera_port",
    "camera_video_filename",
    "camera_disk_warning",
    "camera_stall_warning",
    "camera_status",
]


def test_new_keys_are_registered_for_persistence():
    # Every new key must exist in BOTH the authoritative global (session) and
    # the reactive-mirror map, or rehydrate would skip it on refresh.
    for key in NEW_PERSISTED_KEYS:
        assert key in state.session, f"{key} missing from session dict"
        assert key in state._REACTIVE_FOR, f"{key} missing from _REACTIVE_FOR"


def test_session_defaults_match_reactive_defaults():
    # The durable twin must start equal to the reactive's own default, so a
    # first-ever mount rehydrates to a no-op rather than changing state.
    assert state.session["boards_connected"] == {}
    assert state.session["i2c_controllers"] == {}
    assert state.session["camera_enabled"] is False
    assert state.session["camera_sensor_id"] is None
    assert state.session["camera_host"] == "picamera0.local"
    assert state.session["camera_port"] == 8770
    assert state.session["camera_video_filename"] == ""
    assert state.session["camera_disk_warning"] == ""
    assert state.session["camera_stall_warning"] == ""
    assert state.session["camera_status"] == "unknown"


def test_hardware_survives_a_simulated_refresh():
    # Simulate an initialized run: two boards connected, live controller dict.
    fake_controllers = {"FT232H0": object(), "FT232H1": object()}
    state.set_session("boards_connected", {"FT232H0": 6, "FT232H1": 6})
    state.set_session("i2c_controllers", fake_controllers)

    # Simulate a NEW kernel context: reactives reset to defaults, global intact.
    state.boards_connected.set({})
    state.i2c_controllers.set({})

    state.rehydrate_reactives_from_session()

    # The Hardware card branches on boards_connected being truthy.
    assert state.boards_connected.value == {"FT232H0": 6, "FT232H1": 6}
    # Same live controller objects re-pointed, not copies.
    assert state.i2c_controllers.value is fake_controllers

    # Cleanup so other tests see defaults.
    state.set_session("boards_connected", {})
    state.set_session("i2c_controllers", {})


def test_camera_enable_and_bookmark_sensor_survive_a_refresh():
    # The bookmark gate is: camera_enabled AND sensor_id == camera_sensor_id.
    state.set_session("camera_enabled", True)
    state.set_session("camera_sensor_id", 5)

    # New context: reactives back at defaults.
    state.camera_enabled.set(False)
    state.camera_sensor_id.set(None)

    state.rehydrate_reactives_from_session()

    assert state.camera_enabled.value is True
    assert state.camera_sensor_id.value == 5
    # Gate for sensor 5 now evaluates true again -> Start/Stop send bookmarks.
    assert state.camera_enabled.value and 5 == state.camera_sensor_id.value

    # Cleanup.
    state.set_session("camera_enabled", False)
    state.set_session("camera_sensor_id", None)
