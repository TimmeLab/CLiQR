"""A new session must restart every sensor's recording cycle at 0.

Cycles carried over across sessions caused the new session's first recording to
be stored under suffixed datasets (start_time1, video_pts1, ...) the analysis
could not find. Animal IDs and volume/weight inputs must be preserved.
"""
from dataclasses import replace

from utils import state
import components.session_controls as sc


def test_reset_clears_cycle_but_keeps_inputs():
    sensors = state.session["sensor_states"].copy()
    sid = next(iter(sensors))
    sensors[sid] = replace(
        sensors[sid],
        recording_cycle=2,
        is_recording=True,
        status="recording",
        elapsed_seconds=99,
        start_time=1234.0,
        animal_id="ACG-26-3",
        start_volume=5.0,
        weight=22.0,
    )
    state.set_session("sensor_states", sensors)

    sc._reset_sensor_lifecycle()

    s = state.sensor_states.value[sid]
    # Lifecycle reset for the fresh file...
    assert s.recording_cycle == 0
    assert s.is_recording is False
    assert s.status == "idle"
    assert s.elapsed_seconds == 0
    assert s.start_time == 0.0
    # ...but user inputs preserved.
    assert s.animal_id == "ACG-26-3"
    assert s.start_volume == 5.0
    assert s.weight == 22.0

    # Cleanup for other tests.
    state.set_session("sensor_states", {
        i: state.SensorState(sensor_id=i) for i in range(1, 25)
    })


def test_reset_keeps_session_global_in_sync():
    """_reset_sensor_lifecycle must update the authoritative global, not just
    the reactive, so a reconnect during the run rehydrates correct lifecycle."""
    sensors = {i: state.SensorState(sensor_id=i) for i in range(1, 25)}
    sensors[4] = replace(sensors[4], recording_cycle=3, is_recording=True,
                         start_volume=7.0, weight=21.0)
    state.set_session("sensor_states", sensors)

    sc._reset_sensor_lifecycle()

    s_global = state.session["sensor_states"][4]
    assert s_global.recording_cycle == 0        # lifecycle reset...
    assert s_global.is_recording is False
    assert s_global.start_volume == 7.0         # ...inputs preserved...
    # ...and the global matches the reactive.
    assert state.sensor_states.value[4].recording_cycle == 0
    assert state.sensor_states.value[4].start_volume == 7.0

    state.set_session("sensor_states",
                      {i: state.SensorState(sensor_id=i) for i in range(1, 25)})
