"""
Session control components for starting/stopping recordings.
"""
import solara
import asyncio
import datetime
import os
import time
import pandas as pd
from dataclasses import replace
from utils import state
from recording.recorder import SensorRecorder


# Global recorder instance
current_recorder = None
recording_task = None


def start_recording():
    """Start a new recording session."""
    global current_recorder, recording_task

    # Clear any previous error
    state.session_error.set("")

    # Validate prerequisites
    if not state.i2c_controllers.value:
        state.session_error.set("No hardware initialized. Please initialize hardware first.")
        state.add_log_message("ERROR: No hardware initialized. Please initialize hardware first.")
        return

    if state.recording_all.value:
        state.session_error.set("Recording already in progress")
        state.add_log_message("ERROR: Recording already in progress")
        return

    if len(state.layout_df.value) == 0:
        state.session_error.set("Please upload a layout file before starting recording. The layout file maps sensor positions to animal IDs. A default layout file is included at layouts/default_layout.csv")
        state.add_log_message("ERROR: No layout file uploaded. Please upload a layout file first.")
        return

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"raw_data_{timestamp}.h5"

    # Add output directory if specified
    output_dir = state.output_directory.value
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
    else:
        full_path = filename

    # Update state
    state.filename.set(full_path)
    state.recording_all.set(True)
    state.session_error.set("")  # Clear error on successful start

    # Create recorder instance
    from components.hardware_status import mpr121_manager
    if mpr121_manager is None:
        state.add_log_message("ERROR: MPR121 manager not initialized")
        state.recording_all.set(False)
        return

    current_recorder = SensorRecorder(
        mpr121_manager=mpr121_manager,
        filename=full_path,
        controllers=state.i2c_controllers.value
    )

    # Start async recording task
    async def run_recording():
        try:
            await current_recorder.record_sensors(log_callback=state.add_log_message)
        except Exception as e:
            state.add_log_message(f"Recording error: {str(e)}")
            state.recording_all.set(False)

    recording_task = asyncio.create_task(run_recording())

    state.add_log_message(f"Recording session started - saving to: {full_path}")


def stop_recording():
    """Stop the current recording session."""
    global current_recorder, recording_task

    if not state.recording_all.value:
        state.add_log_message("ERROR: No recording in progress")
        return

    state.add_log_message("Stopping recording session...")

    # Stop all sensors that are still recording and write all volume/weight metadata
    sensors = state.sensor_states.value.copy()
    for sensor_id, sensor in sensors.items():
        if sensor.is_recording:
            # Write stop time for sensors still recording
            if current_recorder:
                current_recorder.write_sensor_metadata(
                    sensor_id=sensor_id,
                    stop_time=time.time(),
                    cycle=sensor.recording_cycle
                )
            # Update sensor state (create new object, increment cycle)
            sensors[sensor_id] = replace(
                sensor,
                is_recording=False,
                status="idle",
                recording_cycle=sensor.recording_cycle + 1
            )

    state.sensor_states.set(sensors)

    # Now write volume/weight metadata for ALL sensors that have recording cycles > 0
    if current_recorder:
        for sensor_id, sensor in sensors.items():
            # Write metadata for each recording cycle
            for cycle in range(sensor.recording_cycle):
                current_recorder.write_sensor_metadata(
                    sensor_id=sensor_id,
                    start_vol=sensor.start_volume,
                    stop_vol=sensor.stop_volume,
                    weight=sensor.weight,
                    cycle=cycle
                )

    # Write comments to file
    if current_recorder and state.comments.value:
        current_recorder.write_comments(state.comments.value)

    # Stop the recorder
    if current_recorder:
        current_recorder.stop()

    # Cancel the async task
    if recording_task and not recording_task.done():
        recording_task.cancel()

    # Update state
    state.recording_all.set(False)
    state.add_log_message(f"Recording session stopped - file saved: {state.filename.value}")


def handle_layout_file(file_info):
    """
    Handle uploaded layout file and update sensor states with animal IDs.

    Args:
        file_info: FileInfo object from Solara FileDrop
    """
    try:
        filename = file_info['name']
        file_obj = file_info['file_obj']

        # Reset file pointer to beginning
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)

        # Parse the file based on extension
        # Note: default_layout.csv has no header, just sensor,animal_id pairs
        if filename.endswith('.csv'):
            df = pd.read_csv(file_obj, index_col=0, header=None)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_obj, index_col=0, header=None)
        else:
            state.add_log_message(f"ERROR: Unsupported file format. Use .csv or .xlsx")
            return

        # Validate the layout file
        if df.empty or len(df.columns) == 0:
            state.add_log_message(f"ERROR: Layout file is empty or has no columns")
            return

        # Update state with the layout dataframe
        state.layout_df.set(df)
        state.add_log_message(f"Layout file '{filename}' loaded successfully")

        # Update sensor states with animal IDs
        from components.hardware_status import update_animal_ids_from_layout
        update_animal_ids_from_layout()

        # Count how many sensors were mapped
        sensors = state.sensor_states.value
        mapped_count = sum(1 for s in sensors.values() if s.animal_id)
        state.add_log_message(f"Mapped {mapped_count} sensors to animal IDs")

    except Exception as e:
        state.add_log_message(f"ERROR loading layout file: {str(e)}")


@solara.component
def SessionControlsCard():
    """Session control UI including output directory and start/stop button."""

    with solara.Card(title="Session Controls", style={"margin-bottom": "20px"}):
        # Output directory selection
        with solara.Column(style={"gap": "5px"}):
            solara.Markdown("**Output Directory**", style={"font-size": "16px", "font-weight": "bold", "margin-bottom": "5px"})
            solara.InputText(
                label="",
                value=state.output_directory.value,
                on_value=state.output_directory.set,
                style={"flex": "1", "max-width": "500px"},
                disabled=state.recording_all.value  # Can't change during recording
            )

        # Layout file upload
        with solara.Column(style={"margin-top": "10px", "gap": "5px"}):
            solara.Markdown("**Layout File** (CSV/XLSX mapping sensors to animal IDs)")
            if state.recording_all.value:
                solara.Warning("⚠ Layout file cannot be changed during recording")
            else:
                solara.FileDrop(
                    on_file=handle_layout_file,
                    label="Drag and drop layout file here (CSV or XLSX)",
                    lazy=False
                )
            if len(state.layout_df.value) > 0:
                mapped_count = sum(1 for s in state.sensor_states.value.values() if s.animal_id)
                solara.Success(f"✓ Layout loaded: {mapped_count} sensors mapped")

        # Display session error prominently if present
        if state.session_error.value:
            with solara.Row(style={"margin-top": "10px"}):
                solara.Error(state.session_error.value)

        # Start/Stop button
        with solara.Row(style={"margin-top": "20px"}):
            if state.recording_all.value:
                solara.Button(
                    label="⏹ STOP RECORDING",
                    on_click=stop_recording,
                    color="error",
                    style={
                        "width": "300px",
                        "height": "60px",
                        "font-size": "18px",
                        "font-weight": "bold"
                    }
                )
            else:
                # Only enable if hardware is initialized
                disabled = not bool(state.i2c_controllers.value)

                solara.Button(
                    label="⏺ START RECORDING",
                    on_click=start_recording,
                    color="success",
                    disabled=disabled,
                    style={
                        "width": "300px",
                        "height": "60px",
                        "font-size": "18px",
                        "font-weight": "bold"
                    }
                )

        # Status display
        with solara.Row(style={"margin-top": "10px"}):
            if state.recording_all.value:
                solara.Info(f"📝 Recording to: {state.filename.value}")
            else:
                if state.i2c_controllers.value:
                    solara.Text("Status: Ready to record")
                else:
                    solara.Warning("Status: Please initialize hardware first")
