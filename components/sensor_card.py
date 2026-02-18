"""
Per-sensor control card component.

Each sensor has its own card with start/stop controls, volume/weight inputs,
timer display, and status indicator.
"""
import solara
import asyncio
import time as time_module
from dataclasses import replace
from utils import state


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

    # Start timer update task
    asyncio.create_task(update_sensor_timer(sensor_id))


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
