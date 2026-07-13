"""A new session must restart every sensor's recording cycle at 0.

Cycles carried over across sessions caused the new session's first recording to
be stored under suffixed datasets (start_time1, video_pts1, ...) the analysis
could not find. Animal IDs and volume/weight inputs must be preserved.
"""
from dataclasses import replace

from utils import state
import components.session_controls as sc


def test_reset_clears_cycle_but_keeps_inputs():
    sensors = state.sensor_states.value.copy()
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
    state.sensor_states.set(sensors)

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
    state.sensor_states.set({
        i: state.SensorState(sensor_id=i) for i in range(1, 25)
    })
