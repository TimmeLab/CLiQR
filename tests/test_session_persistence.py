"""Session-critical state must live in a context-immune plain global and stay
mirrored into the per-context reactives, so a kernel-context reset (reconnect)
can be recovered by rehydrating from the global."""
from dataclasses import replace

from utils import state


def test_set_session_updates_both_global_and_reactive():
    state.set_session("recording_all", True)
    assert state.session["recording_all"] is True
    assert state.recording_all.value is True

    sensors = {i: state.SensorState(sensor_id=i) for i in range(1, 25)}
    sensors[3] = replace(sensors[3], start_volume=5.0, weight=22.0)
    state.set_session("sensor_states", sensors)
    assert state.session["sensor_states"][3].start_volume == 5.0
    assert state.sensor_states.value[3].start_volume == 5.0

    # Cleanup for other tests.
    state.set_session("recording_all", False)
    state.set_session("sensor_states",
                      {i: state.SensorState(sensor_id=i) for i in range(1, 25)})


def test_rehydrate_restores_reactives_from_session():
    # Simulate a run's live state in the authoritative global.
    sensors = {i: state.SensorState(sensor_id=i) for i in range(1, 25)}
    sensors[7] = replace(sensors[7], is_recording=True, start_time=123.0,
                         start_volume=6.5, weight=20.0, recording_cycle=0)
    state.set_session("sensor_states", sensors)
    state.set_session("recording_all", True)

    # Simulate a NEW kernel context: reactives back at defaults, global intact.
    state.recording_all.set(False)
    state.sensor_states.set({i: state.SensorState(sensor_id=i) for i in range(1, 25)})
    assert state.sensor_states.value[7].start_volume == 0.0  # reset

    state.rehydrate_reactives_from_session()

    assert state.recording_all.value is True
    assert state.sensor_states.value[7].is_recording is True
    assert state.sensor_states.value[7].start_volume == 6.5

    # Cleanup.
    state.set_session("recording_all", False)
    state.set_session("sensor_states",
                      {i: state.SensorState(sensor_id=i) for i in range(1, 25)})


def test_page_registers_rehydrate_and_imports_clean():
    """Page must call the rehydrate helper once on mount. We can't easily mount
    Solara here, so assert the helper is referenced in the Page module and that
    importing it does not error."""
    import inspect
    import recording_gui

    src = inspect.getsource(recording_gui.Page)
    assert "rehydrate_reactives_from_session" in src
    assert "use_effect" in src
