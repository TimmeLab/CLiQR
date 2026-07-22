"""
Per-sensor control card component.

Each sensor has its own card with start/stop controls, volume/weight inputs,
timer display, and status indicator.
"""
import solara
import asyncio
import threading
import time as time_module
from dataclasses import replace
from utils import state

# A bookmark reply's `pi_monotonic - pts` is the Pi-side gap between the
# bookmarked frame's capture and the bookmark executing. Healthy runs measure
# ~0.02 s (2026-07-21 Start bookmark: 0.017 s). The same run's Stop bookmark
# measured 5414 s: the camera had frozen 90 min earlier and every layer -- the
# Pi server, the TCP client, the HDF5 writer -- reported success anyway. 5 s is
# ~250x the healthy gap and far below any real stall.
VIDEO_STALL_WARN_S = 5.0

# The video<->trace anchor is single-shot: the Start bookmark is the ONLY thing
# tying frame numbers to session time, so one failed round-trip leaves the whole
# session's video unalignable. On 2026-07-22 a single
# "[WinError 10061] ... actively refused it" did exactly that. Retrying costs a
# few seconds in the worst case and nothing in the normal one.
BOOKMARK_ATTEMPTS = 3
BOOKMARK_RETRY_DELAY_S = 1.0


def _bookmark_with_retry(client, sensor_id: int, label: str):
    """Bookmark the video, retrying failures. Returns (resp, before, after).

    Host wall-clock brackets the attempt that SUCCEEDED, not the first one: the
    latency correction works back from host_after to the bookmarked frame's true
    host time, so carrying a stale bracket across a retry would bias the anchor
    by the whole retry delay. client.bookmark() never raises — it returns an
    error dict — so a failure here is always a value, never an exception.
    """
    resp, host_before, host_after = {}, 0.0, 0.0
    for attempt in range(1, BOOKMARK_ATTEMPTS + 1):
        host_before = time_module.time()
        resp = client.bookmark(sensor_id)
        host_after = time_module.time()
        if resp.get("ok"):
            if attempt > 1:
                state.add_log_message(
                    f"Sensor {sensor_id}: {label} bookmark succeeded on attempt "
                    f"{attempt}")
            return resp, host_before, host_after
        if attempt < BOOKMARK_ATTEMPTS:
            state.add_log_message(
                f"Sensor {sensor_id}: {label} bookmark attempt {attempt} failed "
                f"({resp.get('error')}); retrying")
            time_module.sleep(BOOKMARK_RETRY_DELAY_S)
    return resp, host_before, host_after


def _report_bookmark_failure(sensor_id: int, resp: dict, label: str):
    """Say plainly that the video is unalignable — this is not a minor warning.

    The old message was one quiet WARNING line among many and was easy to scroll
    past, which is how a session ran to completion with no usable anchor.
    """
    state.add_log_message(
        f"ERROR: Sensor {sensor_id}: {label} bookmark FAILED after "
        f"{BOOKMARK_ATTEMPTS} attempts: {resp.get('error')}")
    if label == "start":
        state.add_log_message(
            "ERROR: Without a start bookmark the video CANNOT be aligned to the "
            "capacitance trace. Check that the Pi camera server is running, "
            "then stop and restart the session to get a usable anchor.")


def _warn_if_video_frozen(sensor_id: int, resp: dict, label: str):
    """Log a warning when a bookmark reply describes a long-stale frame.

    The Pi reports `frames_stale_s` directly; `pi_monotonic - pts` is the
    fallback for a server predating that field. Either way this is the check
    that turns a silent 90-minute video loss into a message during the session.
    """
    stale = resp.get("frames_stale_s")
    if stale is None:
        pi_monotonic, pts = resp.get("pi_monotonic"), resp.get("pts")
        if pi_monotonic is None or pts is None:
            return
        stale = float(pi_monotonic) - float(pts)
    if float(stale) < VIDEO_STALL_WARN_S:
        return
    state.add_log_message(
        f"WARNING: Sensor {sensor_id}: {label} bookmark frame is "
        f"{float(stale):.0f}s stale — the Pi camera has stopped delivering "
        f"frames. Video for this period is missing; capacitance data is "
        f"unaffected. Check the Pi server log.")


def start_sensor(sensor_id: int):
    """Start recording for a specific sensor."""
    from components.session_controls import current_recorder

    # Check if global recording is active
    if not state.recording_all.value:
        state.add_log_message(f"ERROR: Sensor {sensor_id}: Please start global recording first")
        return

    if not current_recorder:
        state.add_log_message(f"ERROR: Sensor {sensor_id}: Recorder not initialized")
        return

    # Get current sensor state
    sensors = state.sensor_states.value.copy()
    sensor = sensors[sensor_id]

    if sensor.is_recording:
        state.add_log_message(f"WARNING: Sensor {sensor_id}: Already recording")
        return

    # Create updated sensor state (replace creates a new object)
    start_time = time_module.time()
    current_cycle = sensor.recording_cycle
    sensors[sensor_id] = replace(
        sensor,
        is_recording=True,
        status="recording",
        start_time=start_time,
        elapsed_seconds=0
    )

    # Write start time to HDF5 (volume/weight written later on global stop)
    current_recorder.write_sensor_metadata(
        sensor_id=sensor_id,
        start_time=start_time,
        cycle=current_cycle
    )

    state.sensor_states.set(sensors)
    cycle_text = f" (cycle {current_cycle + 1})" if current_cycle > 0 else ""
    state.add_log_message(f"Sensor {sensor_id}: Recording started{cycle_text}")

    # Bookmark the concurrent video for the designated camera sensor.
    #
    # The bookmark is a blocking wireless round-trip. start_sensor runs on the
    # Solara asyncio event loop, the SAME loop that drives record_sensors, so a
    # synchronous bookmark here stalls all sensor acquisition for the whole
    # round-trip (observed: a ~5 s hole in every sensor's data at session start).
    # Run it on its own thread so acquisition keeps sampling; write_video_metadata
    # is already h5-lock guarded. Return the thread so callers/tests can join it.
    bookmark_thread = None
    if state.camera_enabled.value and sensor_id == state.camera_sensor_id.value:
        from components import session_controls
        client = session_controls.camera_client
        if client is not None:
            recorder = current_recorder
            video_filename = state.camera_video_filename.value

            def _bookmark_video():
                try:
                    # Bracket the round-trip with host wall-clock so the residual
                    # bookmark latency (frame's host time - start_time) stays
                    # recoverable; the frame the Pi returns was captured mid-
                    # round-trip, not at the start_time stamped above.
                    resp, host_before, host_after = _bookmark_with_retry(
                        client, sensor_id, "start")
                    if resp.get("ok"):
                        recorder.write_video_metadata(
                            sensor_id=sensor_id,
                            frame_index=resp.get("frame_index"),
                            pts=resp.get("pts"),
                            video_filename=video_filename,
                            cycle=current_cycle,
                            pi_monotonic=resp.get("pi_monotonic"),
                            host_time_before=host_before,
                            host_time_after=host_after,
                        )
                        state.add_log_message(
                            f"Sensor {sensor_id}: video bookmark "
                            f"frame={resp.get('frame_index')} pts={resp.get('pts'):.3f}")
                        _warn_if_video_frozen(sensor_id, resp, "start")
                    else:
                        _report_bookmark_failure(sensor_id, resp, "start")
                except Exception as exc:
                    state.add_log_message(
                        f"WARNING: Sensor {sensor_id}: bookmark error: {exc}")

            bookmark_thread = threading.Thread(target=_bookmark_video, daemon=True)
            bookmark_thread.start()

    # Start timer update task
    asyncio.create_task(update_sensor_timer(sensor_id))
    return bookmark_thread


def bookmark_stop(sensor_id: int, cycle: int):
    """Bookmark the video at the STOP of the camera sensor's cycle — the second
    clock anchor for drift correction (see docs/superpowers/specs/
    2026-07-16-clock-drift-stop-bookmark-design.md).

    Mirrors start_sensor's bookmark: runs the wireless round-trip on a daemon
    thread so stop never blocks, brackets it with host wall-clock, and writes the
    stop datasets. Returns the thread (join-able), or None when this sensor isn't
    the camera driver or no client/recorder is connected.
    """
    if not (state.camera_enabled.value and sensor_id == state.camera_sensor_id.value):
        return None
    from components import session_controls
    client = session_controls.camera_client
    recorder = session_controls.current_recorder
    if client is None or recorder is None:
        return None

    def _bookmark():
        try:
            resp, host_before, host_after = _bookmark_with_retry(
                client, sensor_id, "stop")
            if resp.get("ok"):
                recorder.write_video_metadata(
                    sensor_id=sensor_id, cycle=cycle,
                    stop_frame_index=resp.get("frame_index"),
                    stop_pts=resp.get("pts"),
                    stop_pi_monotonic=resp.get("pi_monotonic"),
                    stop_host_before=host_before,
                    stop_host_after=host_after,
                )
                state.add_log_message(
                    f"Sensor {sensor_id}: video stop bookmark "
                    f"frame={resp.get('frame_index')}")
                _warn_if_video_frozen(sensor_id, resp, "stop")
            else:
                _report_bookmark_failure(sensor_id, resp, "stop")
        except Exception as exc:
            state.add_log_message(
                f"WARNING: Sensor {sensor_id}: stop bookmark error: {exc}")

    thread = threading.Thread(target=_bookmark, daemon=True)
    thread.start()
    return thread


def stop_sensor(sensor_id: int):
    """Stop recording for a specific sensor."""
    from components.session_controls import current_recorder

    # Get current sensor state
    sensors = state.sensor_states.value.copy()
    sensor = sensors[sensor_id]

    if not sensor.is_recording:
        state.add_log_message(f"WARNING: Sensor {sensor_id}: Not currently recording")
        return

    # Calculate final elapsed time
    elapsed = time_module.time() - sensor.start_time
    current_cycle = sensor.recording_cycle

    # Create updated sensor state (increment cycle for next recording)
    sensors[sensor_id] = replace(
        sensor,
        is_recording=False,
        status="idle",
        recording_cycle=current_cycle + 1
    )

    # Write stop time to HDF5 (volume/weight written later on global stop)
    if current_recorder:
        current_recorder.write_sensor_metadata(
            sensor_id=sensor_id,
            stop_time=time_module.time(),
            cycle=current_cycle
        )

    # Stop bookmark for the camera sensor (drift-fit anchor). Camera is still
    # active on an individual stop, so fire-and-forget like the start bookmark.
    bookmark_stop(sensor_id, current_cycle)

    state.sensor_states.set(sensors)

    # Format elapsed time for log
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    state.add_log_message(f"Sensor {sensor_id}: Recording stopped (duration: {time_str})")


async def update_sensor_timer(sensor_id: int):
    """Update the timer for a sensor every 60 seconds."""
    while True:
        await asyncio.sleep(60)  # Update every minute

        sensors = state.sensor_states.value.copy()
        sensor = sensors[sensor_id]

        if not sensor.is_recording:
            break

        # Create updated sensor state with new elapsed time
        sensors[sensor_id] = replace(
            sensor,
            elapsed_seconds=int(time_module.time() - sensor.start_time)
        )
        state.sensor_states.set(sensors)


def test_sensor(sensor_id: int):
    """Test a sensor by reading recent data and displaying a plot."""
    from components.hardware_status import mpr121_manager

    if mpr121_manager is None:
        state.add_log_message(f"ERROR: Sensor {sensor_id}: Hardware not initialized")
        return

    # Get recent data
    cap_data, time_data = mpr121_manager.get_last_reading(sensor_id, num_samples=250)

    if cap_data is None or len(cap_data) == 0:
        state.add_log_message(f"Sensor {sensor_id}: No data available for testing")
        return

    # Store data in state and show dialog
    state.test_plot_data.set({
        'sensor_id': sensor_id,
        'cap_data': cap_data,
        'time_data': time_data
    })
    state.show_test_dialog.set(True)
    state.add_log_message(f"Sensor {sensor_id}: Displaying {len(cap_data)} samples")


def format_timer(elapsed_seconds: int) -> str:
    """Format elapsed seconds as HH:MM."""
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


@solara.component
def SensorCard(sensor_id: int):
    """
    Display control card for a single sensor.

    Args:
        sensor_id: Sensor ID (1-24)
    """
    # Get sensor state
    sensor = state.sensor_states.value.get(sensor_id)
    if not sensor:
        return solara.Error(f"Invalid sensor ID: {sensor_id}")

    # Determine status color and icon
    if sensor.status == "recording":
        status_color = "#4CAF50"  # Green
        status_icon = "●"
        status_text = "RECORDING"
    elif sensor.status == "error":
        status_color = "#F44336"  # Red
        status_icon = "●"
        status_text = "ERROR"
    else:  # idle
        status_color = "#9E9E9E"  # Gray
        status_icon = "●"
        status_text = "IDLE"

    # Determine if controls should be disabled
    recording_not_started = not state.recording_all.value
    is_recording = sensor.is_recording

    with solara.Card(
        style={
            "width": "250px",
            "min-height": "200px",
            "border": f"2px solid {status_color}",
            "padding": "10px"
        }
    ):
        # Header with sensor ID, animal ID, and status
        with solara.Row(style={"justify-content": "space-between", "align-items": "center"}):
            animal_text = f" [{sensor.animal_id}]" if sensor.animal_id else ""
            solara.Text(
                f"Sensor {sensor_id}{animal_text}",
                style={"font-weight": "bold", "font-size": "14px"}
            )
            solara.Text(
                f"{status_icon} {status_text}",
                style={"color": status_color, "font-size": "10px", "font-weight": "bold"}
            )

        # Controls row: Start/Stop button, Timer, Test button
        with solara.Row(style={"gap": "5px", "margin-top": "10px"}):
            if is_recording:
                solara.Button(
                    label="STOP",
                    on_click=lambda: stop_sensor(sensor_id),
                    color="error",
                    style={"flex": "1"}
                )
            else:
                solara.Button(
                    label="START",
                    on_click=lambda: start_sensor(sensor_id),
                    color="success",
                    disabled=recording_not_started,
                    style={"flex": "1"}
                )

            # Timer display
            timer_text = format_timer(sensor.elapsed_seconds) if is_recording else "00:00"
            solara.Text(
                timer_text,
                style={
                    "font-family": "monospace",
                    "font-size": "14px",
                    "font-weight": "bold",
                    "min-width": "50px"
                }
            )

            solara.Button(
                label="TEST",
                on_click=lambda: test_sensor(sensor_id),
                color="primary",
                disabled=recording_not_started,
                style={"width": "60px"}
            )

        # Volume and weight inputs
        with solara.Column(style={"margin-top": "10px", "gap": "5px"}):
            # Start volume
            def set_start_vol(value):
                sensors = state.sensor_states.value.copy()
                sensors[sensor_id] = replace(sensors[sensor_id], start_volume=value)
                state.sensor_states.set(sensors)

            solara.InputFloat(
                label="Start Vol (mL)",
                value=sensor.start_volume,
                on_value=set_start_vol,
                continuous_update=True,
                style={"width": "100%"}
            )

            # Stop volume
            def set_stop_vol(value):
                sensors = state.sensor_states.value.copy()
                sensors[sensor_id] = replace(sensors[sensor_id], stop_volume=value)
                state.sensor_states.set(sensors)

            solara.InputFloat(
                label="Stop Vol (mL)",
                value=sensor.stop_volume,
                on_value=set_stop_vol,
                continuous_update=True,
                style={"width": "100%"}
            )

            # Weight
            def set_weight(value):
                sensors = state.sensor_states.value.copy()
                sensors[sensor_id] = replace(sensors[sensor_id], weight=value)
                state.sensor_states.set(sensors)

            solara.InputFloat(
                label="Weight (g)",
                value=sensor.weight,
                on_value=set_weight,
                continuous_update=True,
                style={"width": "100%"}
            )


@solara.component
def SensorGrid():
    """Display all 24 sensors in a 4×6 grid matching the physical rack layout."""

    with solara.Card(title="Sensor Grid", style={"margin-bottom": "20px"}):
        solara.Markdown("*Sensors arranged to match physical rack layout (4 shelves × 6 positions)*")

        # Create 4 rows (shelves)
        for shelf in range(4):
            shelf_label = ["Top", "Second", "Third", "Bottom"][shelf]

            with solara.Column(style={"margin-bottom": "20px"}):
                # Shelf label
                solara.Text(
                    f"Shelf {shelf + 1} ({shelf_label})",
                    style={"font-weight": "bold", "margin-bottom": "10px"}
                )

                # 6 sensors per shelf
                with solara.Row(style={"gap": "10px", "flex-wrap": "wrap"}):
                    sensor_start = shelf * 6 + 1
                    for offset in range(6):
                        sensor_id = sensor_start + offset
                        SensorCard(sensor_id=sensor_id)
