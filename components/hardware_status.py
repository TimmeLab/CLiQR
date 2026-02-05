"""
Hardware status display and initialization controls.
"""
import solara
from dataclasses import replace
from hardware.ft232h import FT232HManager
from hardware.mpr121 import MPR121Manager
from utils import state


# Global hardware manager instances
ft232h_manager = FT232HManager()
mpr121_manager = None


def initialize_hardware():
    """Initialize FT232H boards and MPR121 sensors."""
    global mpr121_manager

    state.add_log_message("Scanning for FT232H boards...")

    # Scan for devices
    num_devices = ft232h_manager.scan_devices()
    if num_devices == 0:
        state.add_log_message("ERROR: No FT232H boards found")
        state.boards_connected.set({})
        return

    state.add_log_message(f"Found {num_devices} FT232H board(s)")

    # Initialize I2C controllers
    controllers, errors = ft232h_manager.initialize_controllers()

    if errors:
        for error in errors:
            state.add_log_message(f"ERROR: {error}")

    if not controllers:
        state.add_log_message("ERROR: No controllers initialized")
        state.boards_connected.set({})
        return

    # Store controllers in state
    state.i2c_controllers.set(controllers)

    # Configure MPR121 sensors
    mpr121_manager = MPR121Manager(controllers)
    config_messages = mpr121_manager.configure_all_sensors()

    for msg in config_messages:
        state.add_log_message(msg)

    # Update boards_connected state
    board_info = ft232h_manager.get_controller_info()
    state.boards_connected.set(board_info)

    # Update sensor states with animal IDs from layout if available
    if not state.layout_df.value.empty:
        update_animal_ids_from_layout()

    state.add_log_message(f"Hardware initialization complete: {len(board_info)} boards ready")


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

    state.sensor_states.set(sensors)


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
                    on_click=initialize_hardware,
                    color="secondary"
                )

                # Only show disconnect if recording is not active
                if not state.recording_all.value:
                    def disconnect_hardware():
                        ft232h_manager.close_all()
                        state.boards_connected.set({})
                        state.i2c_controllers.set({})
                        state.add_log_message("Hardware disconnected")

                    solara.Button(
                        label="Disconnect",
                        on_click=disconnect_hardware,
                        color="error"
                    )
