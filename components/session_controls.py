"""
Session control components for starting/stopping recordings.
"""
import solara
import asyncio
import datetime
import os
import time
import threading
import pandas as pd
from dataclasses import replace
from utils import state
from recording.recorder import SensorRecorder, measurement_warnings


# Global recorder instance
current_recorder = None
recording_task = None

# Active Pi camera client for the current session (None when camera disabled or
# when the camera failed to start).
camera_client = None

# The daemon thread that stops the camera and fetches its files, exposed so tests
# can join it (the GUI never waits on it).
_last_camera_stop_thread = None

# The Pi's START_SESSION does a full Picamera2 warmup (create + configure +
# start_recording) before replying, which takes longer than a normal request;
# use a generous timeout so a slow-but-successful start isn't misreported as a
# failure. (The previous 2.0s override was shorter than the client default and
# could time out mid-warmup.)
CAMERA_START_TIMEOUT = 10.0


def _reset_sensor_lifecycle():
    """Reset each sensor's recording lifecycle for a new session.

    Each session writes a fresh HDF5 file, so recording cycles must restart at 0.
    Without this the cycle counter carried over from the previous session and the
    new session's first recording was written under a suffixed dataset name
    (start_time1, video_pts1, ...) that the analysis, which reads cycle 0, could
    not find. Animal IDs and volume/weight inputs are preserved.
    """
    sensors = state.session["sensor_states"].copy()
    for sid, s in sensors.items():
        sensors[sid] = replace(
            s, is_recording=False, status="idle",
            recording_cycle=0, elapsed_seconds=0, start_time=0.0,
        )
    state.set_session("sensor_states", sensors)


def _start_camera(video_base):
    """Start the Pi camera pre-roll for this session.

    On success, sets the module-level camera_client and camera_video_filename.
    On any failure, leaves camera_client None and clears camera_video_filename
    so later per-sensor bookmarks are skipped cleanly and stop_recording does
    not spawn a doomed stop/fetch against a session that never started.
    """
    global camera_client
    camera_client = None
    # Route through set_session so these clear/set operations survive a refresh.
    state.set_session("camera_video_filename", "")
    state.set_session("camera_disk_warning", "")
    state.set_session("camera_stall_warning", "")

    try:
        client = state.make_camera_client(timeout=CAMERA_START_TIMEOUT)
        resp = client.start_session(video_base)
    except Exception as exc:
        state.add_log_message(f"WARNING: Camera start error: {exc}")
        return

    if resp.get("ok"):
        camera_client = client
        state.set_session("camera_video_filename", resp.get("video_filename", ""))
        state.add_log_message(f"Camera pre-roll started: {resp.get('video_filename')}")
        # The Pi also reclaims disk space at session start (a crashed run
        # never reaches the post-stop cleanup), so surface that here too.
        _report_pi_disk_cleanup(resp)
    else:
        state.add_log_message(f"WARNING: Camera start failed: {resp.get('error')}")


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
    state.set_session("filename", full_path)
    state.set_session("recording_all", True)
    state.session_error.set("")  # Clear error on successful start

    # Fresh file -> restart every sensor's recording cycle at 0.
    _reset_sensor_lifecycle()

    # Create recorder instance
    from components.hardware_status import mpr121_manager
    if mpr121_manager is None:
        state.add_log_message("ERROR: MPR121 manager not initialized")
        state.set_session("recording_all", False)
        return

    current_recorder = SensorRecorder(
        mpr121_manager=mpr121_manager,
        filename=full_path,
        controllers=state.i2c_controllers.value,
        measurements_provider=lambda: state.session["sensor_states"],
    )

    # Start async recording task
    async def run_recording():
        try:
            await current_recorder.record_sensors(log_callback=state.add_log_message)
        except Exception as e:
            state.add_log_message(f"Recording error: {str(e)}")
            state.set_session("recording_all", False)

    recording_task = asyncio.create_task(run_recording())

    # Optionally start concurrent Pi video pre-roll.
    if state.camera_enabled.value:
        video_base = os.path.splitext(os.path.basename(full_path))[0]
        _start_camera(video_base)

    state.add_log_message(f"Recording session started - saving to: {full_path}")


def _report_pi_disk_cleanup(resp):
    """Surface the Pi's post-session disk cleanup (from STOP_SESSION reply).

    The Pi deletes its oldest videos (never the one just recorded) until 5 GB
    is free for the next run. low_disk means even that wasn't enough, so the
    user must free space on the Pi manually — shown as a persistent GUI
    warning, cleared at the next camera start.
    """
    deleted = resp.get("deleted", [])
    if deleted:
        state.add_log_message(
            f"Pi deleted {len(deleted)} old video(s) to free disk space: "
            + ", ".join(deleted))
    if resp.get("low_disk"):
        free_gb = resp.get("free_bytes", 0) / 1024 ** 3
        message = (
            f"Pi disk space low: only {free_gb:.1f} GB free even after "
            "deleting old videos. Free up space on the Pi manually before "
            "the next session, or the video may not fit.")
        state.set_session("camera_disk_warning", message)
        state.add_log_message(f"WARNING: {message}")


def _report_camera_stalls(resp):
    """Surface any watchdog-restarted segments from the STOP_SESSION reply.

    A stall means the Pi camera stopped delivering frames mid-session and the
    watchdog restarted recording into a `_partN` file. The run is intact -- the
    capacitance trace is untouched and the segments cover everything but the
    stall itself -- but the operator must know the video is split, and that the
    camera misbehaved. Silence here is exactly what cost 90 min of video on
    2026-07-21.
    """
    if resp.get("low_disk_during_run"):
        state.add_log_message(
            "WARNING: The Pi dropped below 1 GB free while recording. Video is "
            "written at ~1.4 GB/h, so a long run can fill the disk mid-session "
            "and truncate the video. Free space on the Pi before the next run.")
    if resp.get("ffmpeg_log_overflows"):
        state.add_log_message(
            f"WARNING: The Pi's video muxer logged errors on nearly every frame "
            f"({resp['ffmpeg_log_overflows']} log truncation(s)). The .ffmpeg.log "
            f"file was copied back — check it.")

    stalls = resp.get("stalls") or []
    if not stalls:
        return
    for stall in stalls:
        state.add_log_message(
            f"WARNING: Camera stalled after {stall.get('frames')} frames in "
            f"segment {stall.get('segment')} "
            f"({stall.get('reason', 'no frames')}); recording was restarted "
            f"into a new segment.")
    state.set_session(
        "camera_stall_warning",
        f"Camera stalled {len(stalls)} time(s) this session; the video is split "
        f"across multiple files (_part2, _part3, ...). Each has its own .txt "
        f"sidecar of absolute frame timestamps, so all segments still align to "
        f"the capacitance trace. Check the Pi server log.")


def stop_recording():
    """Stop the current recording session."""
    global current_recorder, recording_task

    if not state.recording_all.value:
        state.add_log_message("ERROR: No recording in progress")
        return

    state.add_log_message("Stopping recording session...")

    # Stop all sensors that are still recording and write all volume/weight metadata
    stop_bookmark_thread = None
    sensors = state.session["sensor_states"].copy()
    for sensor_id, sensor in sensors.items():
        if sensor.is_recording:
            # Write stop time for sensors still recording
            if current_recorder:
                current_recorder.write_sensor_metadata(
                    sensor_id=sensor_id,
                    stop_time=time.time(),
                    cycle=sensor.recording_cycle
                )
            # Stop bookmark (drift-fit anchor) for the camera sensor. Capture the
            # thread so the camera stop below joins it BEFORE STOP_SESSION — once
            # the Pi session ends there is no frame to bookmark.
            from components.sensor_card import bookmark_stop
            t = bookmark_stop(sensor_id, sensor.recording_cycle)
            if t is not None:
                stop_bookmark_thread = t
            # Update sensor state (create new object, increment cycle)
            sensors[sensor_id] = replace(
                sensor,
                is_recording=False,
                status="idle",
                recording_cycle=sensor.recording_cycle + 1
            )

    state.set_session("sensor_states", sensors)

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

    # Surface any started sensor whose measurements never made it to disk, so a
    # silent loss (e.g. a mid-run reset that zeroed the store) is visible.
    for msg in measurement_warnings(state.session["sensor_states"]):
        state.add_log_message(f"WARNING: {msg}")

    # Write comments to file
    if current_recorder and state.comments.value:
        current_recorder.write_comments(state.comments.value)

    # Stop the recorder
    if current_recorder:
        current_recorder.stop()

    # Cancel the async task
    if recording_task and not recording_task.done():
        recording_task.cancel()

    # Stop the camera and copy its files back in a background thread so GUI
    # doesn't block during multi-MB file transfer.
    global camera_client, _last_camera_stop_thread
    if state.camera_enabled.value and camera_client is not None:
        _client = camera_client
        _out_dir = state.output_directory.value

        def _camera_stop_and_fetch(client, out_dir, bookmark_thread):
            try:
                # Bookmark the Pi BEFORE stopping it: STOP_SESSION ends the
                # session, after which BOOKMARK has no frame to return.
                if bookmark_thread is not None:
                    bookmark_thread.join(5.0)
                resp = client.stop_session()
                if resp.get("ok"):
                    names = [f["name"] for f in resp.get("files", [])]
                    fetched = client.fetch_files(names, out_dir)
                    state.add_log_message(
                        f"Camera stopped; copied {len(fetched)} file(s)")
                    _report_camera_stalls(resp)
                    _report_pi_disk_cleanup(resp)
                else:
                    state.add_log_message(
                        f"WARNING: Camera stop failed: {resp.get('error')}")
            except Exception as exc:
                state.add_log_message(f"WARNING: Camera stop error: {exc}")

        _last_camera_stop_thread = threading.Thread(
            target=_camera_stop_and_fetch,
            args=(_client, _out_dir, stop_bookmark_thread), daemon=True)
        _last_camera_stop_thread.start()
    camera_client = None

    # Update state
    state.set_session("recording_all", False)
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
