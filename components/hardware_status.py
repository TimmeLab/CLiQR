"""
Hardware status display and initialization controls.

All USB I/O (board scan, MPR121 config, sampling) runs in the hardware process
(hardware/hw_process.py) so Solara re-renders can't stall sampling. The
HardwareClient handle is a plain module global owned by the server process, so
a browser refresh/reconnect reuses it and never starts a second process.
"""
import atexit

import solara
from dataclasses import replace
from hardware.hw_client import HardwareClient
from utils import state


# Which backend the hardware process runs ("module:Class"); mock mode and tests
# swap this (see hardware/mock_hardware.use_mock_hardware).
BACKEND_PATH = "hardware.backends:RealBackend"

# The live hardware process client (None until initialized).
hw_client = None


def _publish(board_info):
    """Mirror the hardware view into session state (plain data, no handles)."""
    state.set_session("boards_connected", dict(board_info))
    state.set_session("i2c_controllers",
                      {sn: {"num_sensors": n} for sn, n in board_info.items()})


def initialize_hardware():
    """Start the hardware process and initialize boards (or reuse it)."""
    global hw_client

    if hw_client is not None and hw_client.is_alive():
        state.add_log_message("Hardware process already running; reusing it")
        _publish(hw_client.board_info)
        return

    state.add_log_message("Starting hardware process...")
    client = HardwareClient(BACKEND_PATH)
    try:
        result = client.start()
    except Exception as exc:
        client.shutdown()
        state.add_log_message(f"ERROR: hardware process failed to start: {exc}")
        _publish({})
        return

    for msg in result.messages:
        state.add_log_message(msg)
    for err in result.errors:
        state.add_log_message(f"ERROR: {err}")

    if not result.board_info:
        client.shutdown()
        _publish({})
        return

    hw_client = client
    _publish(result.board_info)

    # Update sensor states with animal IDs from layout if available
    if not state.layout_df.value.empty:
        update_animal_ids_from_layout()

    state.add_log_message(
        f"Hardware initialization complete: {len(result.board_info)} boards ready")


def disconnect_hardware():
    """Stop the hardware process and release the boards."""
    global hw_client
    if hw_client is not None:
        hw_client.shutdown()
        hw_client = None
    _publish({})
    state.add_log_message("Hardware disconnected")


def refresh_hardware():
    """Restart the hardware process. Refused while recording."""
    if state.session["recording_all"]:
        state.add_log_message("ERROR: cannot refresh hardware while recording")
        return
    disconnect_hardware()
    initialize_hardware()


@atexit.register
def _shutdown_hw():
    # Release the boards when the GUI server exits (the process is also a
    # daemon and exits by itself if the server dies without running atexit).
    if hw_client is not None:
        hw_client.shutdown()


def update_animal_ids_from_layout():
    """Update sensor states with animal IDs from the uploaded layout."""
    layout = state.layout_df.value
    sensors = state.sensor_states.value.copy()

    for sensor_id in range(1, 25):
        if sensor_id in layout.index:
            animal_id = layout.loc[sensor_id, layout.columns[0]]
            if sensors[sensor_id].animal_id != animal_id:
                sensors[sensor_id] = replace(
                    sensors[sensor_id],
                    animal_id=str(animal_id)
                )

    state.set_session("sensor_states", sensors)


@solara.component
def HardwareStatusCard():
    """Display hardware connection status and initialization controls."""

    with solara.Card(title="Hardware Status", style={"margin-bottom": "20px"}):
        if not state.boards_connected.value:
            solara.Warning("No FT232H boards detected. Click 'Initialize Hardware' to scan.")

            solara.Button(
                label="Initialize Hardware",
                on_click=initialize_hardware,
                color="primary",
                style={"margin-top": "10px"}
            )
        else:
            solara.Success(f"✓ Connected: {len(state.boards_connected.value)} board(s)")

            # Show each board's status
            for sn, num_sensors in sorted(state.boards_connected.value.items()):
                sensor_ids = state.SERIAL_NUMBER_SENSOR_MAP.get(sn, [])
                sensor_list = ",".join(map(str, sensor_ids)) if sensor_ids else "unknown"
                solara.Info(f"● {sn}: {num_sensors} sensors (sensors {sensor_list})")

            # Refresh button
            with solara.Row(style={"margin-top": "10px", "gap": "10px"}):
                solara.Button(
                    label="Refresh Hardware",
                    on_click=refresh_hardware,
                    disabled=state.recording_all.value,
                    color="secondary"
                )

                # Only show disconnect if recording is not active
                if not state.recording_all.value:
                    solara.Button(
                        label="Disconnect",
                        on_click=disconnect_hardware,
                        color="error"
                    )
