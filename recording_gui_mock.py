"""
CLiQR Recording System - Solara GUI with Mock Hardware

This version of the app uses simulated hardware for testing without
physical FT232H/MPR121 devices.

Run with: solara run recording_gui_mock.py
"""
# Enable mock hardware before importing the main app
from hardware.mock_hardware import use_mock_hardware
use_mock_hardware()

# Now import the main app components
from recording_gui import Page

# Solara will automatically use the Page component
