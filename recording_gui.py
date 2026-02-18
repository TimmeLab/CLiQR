"""
CLiQR Recording System - Solara GUI

Standalone GUI application for recording capacitive lickometry data from
MPR121 sensors connected via FT232H USB-to-I2C boards.

Author: Christopher Parker (parkecp@ucmail.uc.edu)
Lab: Timme Lab, University of Cincinnati
"""
import solara
from utils import state
from components.hardware_status import HardwareStatusCard
from components.session_controls import SessionControlsCard
from components.sensor_card import SensorGrid
from components.plot_dialog import TestPlotDialog


@solara.component
def Page():
    """Main application page."""

    with solara.Column(style={"padding": "20px", "max-width": "1800px", "margin": "0 auto"}):
        # Header
        with solara.Card():
            solara.Markdown("# CLiQR Recording System")
            solara.Markdown("*Capacitive Lick Quantification in Rodents*")

        # Hardware Status Section
        HardwareStatusCard()

        # Session Controls Section
        SessionControlsCard()

        # Sensor Grid
        SensorGrid()

        # Activity Log Section
        with solara.Card(title="Activity Log"):
            if state.log_messages.value:
                with solara.Column(style={"max-height": "300px", "overflow-y": "auto"}):
                    for msg in state.log_messages.value[-20:]:  # Show last 20 messages
                        solara.Text(msg, style={"font-family": "monospace", "font-size": "12px"})
            else:
                solara.Text("No activity yet", style={"color": "#757575"})

            solara.Button(
                label="Clear Log",
                on_click=lambda: state.log_messages.set([]),
                color="secondary"
            )

        # Comments Section
        with solara.Card(title="Comments"):
            solara.Markdown("*Comments will be automatically saved to the HDF5 file when recording stops*")
            with solara.Column(style={"width": "600px"}):
                solara.InputTextArea(
                    label="",
                    value=state.comments.value,
                    on_value=state.comments.set,
                    continuous_update=True,
                    rows=6
                )

        # Test plot dialog (shows when test button is clicked)
        TestPlotDialog()


# Solara will automatically detect this component as the main page
