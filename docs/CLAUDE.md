# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLiQR (Capacitive Lickometry System) is a Python-based system for recording and analyzing rodent licking behavior using MPR121 capacitive touch sensors connected via FT232H USB-to-I2C boards. The system is designed for the Timme Lab at University of Cincinnati and runs primarily on Windows for ease of use by lab personnel.

## Development Goals

**Primary Goal: Solara-based Standalone GUI**

The current development focus is converting `DataRecording.ipynb` into a standalone GUI application using Solara. Key requirements:

- **No functional changes**: The conversion should preserve all existing functionality of the data recording system
- **Improved usability**: Create a user-friendly interface for non-technical lab personnel
- **Standalone deployment**: The GUI should run as a standalone application, not requiring users to interact with JupyterLab
- **DataAnalysis.ipynb remains unchanged**: Analysis workflows will continue using the existing Jupyter notebook

This migration aims to lower the barrier to entry for lab members unfamiliar with Jupyter notebooks while maintaining the same robust data collection capabilities.

## Environment Setup

### Installation Options

**Option 1 - Miniforge (recommended for Windows):**
```bash
conda env create --file environment.yml
conda activate cliqr
solara run recording_gui.py  # For data recording
# OR
jupyter-lab  # For data analysis
```

**Option 2 - pyenv-virtualenv (Unix-based systems):**
```bash
pyenv virtualenv 3.13 cliqr
pyenv activate cliqr
pip install -r requirements.txt
```

### Hardware Setup

Before first use, FT232H boards must be assigned serial numbers (one board at a time):
```bash
python set_ft232h_serial.py FT232H0
```

The system expects serial numbers FT232H0 through FT232H3, mapped to sensors 1-24 in the `SERIAL_NUMBER_SENSOR_MAP` constant in `utils/state.py`.

**Windows-specific requirement:** FT232H drivers must be installed using Zadig before use. See README.md for details.

## Architecture

### Data Recording Pipeline

The system provides two interfaces for data collection:

1. **recording_gui.py** - Standalone Solara GUI (Recommended)
   - Web-based interface running at http://localhost:8765
   - Hardware initialization via "Initialize Hardware" button
   - Session controls for starting/stopping recordings
   - 24 individual sensor cards (arranged in 4 rows × 6 columns matching physical rack)
   - Each sensor card includes:
     - Start/Stop button with status indicator
     - Timer display (updates every 60 seconds)
     - Test button for viewing recent data
     - Volume inputs (start/stop) and weight input
   - Activity log for real-time feedback
   - Comments area (auto-saved to HDF5)
   - Async recording from multiple MPR121 sensors via ThreadPoolExecutor
   - Records capacitance values (~50Hz) to HDF5 files (`raw_data_*.h5`)
   - HDF5 structure: `/board_{serial_number}/sensor_{sensor_id}/[cap_data, time_data, start_time, stop_time, start_vol, stop_vol, weight]`

2. **DataRecording.ipynb** - Legacy Jupyter notebook interface (Deprecated)
   - Original interface, kept for reference
   - Same functionality as Solara GUI but requires JupyterLab
   - Not recommended for new recordings

3. **DataAnalysis.ipynb** - Batch analysis and visualization
   - Processes multiple `raw_data_*.h5` files across multiple cohorts
   - Uses `data_analysis.py:filter_data()` to detect licks via thresholding or Hilbert envelope
   - Generates `filtered_*.h5` files (intermediate), then combines into `results_combined_*.h5` by animal ID
   - Includes correlation analysis (licks vs volume consumed) with outlier detection via MAD

4. **data_analysis.py** - Core lick detection algorithms
   - `basic_algorithm()`: Threshold-based peak detection with depth requirements
   - `hilbert_algorithm()`: Hilbert envelope + high-pass filtering (8-12 Hz)
   - Both handle start/stop time corrections and can use time fix files (`*_time_fix.xlsx`)

### Solara GUI Architecture

The recording GUI is built with Solara (reactive web framework) and organized into modules:

**Core Modules:**
- `recording_gui.py` - Main app entry point, assembles all components
- `utils/state.py` - Reactive state management using `solara.reactive()`
- `hardware/ft232h.py` - FT232H board detection and I2C initialization
- `hardware/mpr121.py` - MPR121 sensor configuration and data reading
- `hardware/mock_hardware.py` - Simulated hardware for testing without physical devices
- `recording/recorder.py` - Async recording loop and HDF5 file operations
- `components/hardware_status.py` - Hardware initialization UI
- `components/session_controls.py` - Session start/stop controls
- `components/sensor_card.py` - Per-sensor control cards (24 instances)

**State Management:**
- Global session state: `recording_all`, `filename`, `output_directory`, `i2c_controllers`
- Per-sensor state: `SensorState` dataclass with recording status, volumes, weight, timer
- Activity log: `log_messages` list with timestamped entries
- Hardware state: `boards_connected` dict of FT232H boards

**Key Design Patterns:**
- **Reactive UI**: Solara components automatically re-render when state changes
- **Asynchronous recording**: `SensorRecorder.record_sensors()` uses `asyncio` to avoid blocking the UI
- **Deferred writes**: Capacitance data is buffered (`HISTORY_SIZE=100`) before writing to HDF5 to reduce I/O overhead
- **Multi-cohort analysis**: DataAnalysis.ipynb can process data from multiple animal cohorts in parallel via the `base_dir` and `animal_id_prefixes` variables
- **Time fix mechanism**: If start/stop times are incorrect, create `<raw_filename>_time_fix.xlsx` with columns `[Sensor, New Start Time, New End Time]` to override
- **Component-based architecture**: Reusable Solara components for sensors, hardware status, etc.
- **Mock hardware support**: `recording_gui_mock.py` uses simulated sensors for testing

### Critical Hardware-Software Mapping

Sensor numbers (1-24) map to FT232H boards based on physical rack layout:
- Sensors 1,2,3,7,8,9 → FT232H0
- Sensors 4,5,6,10,11,12 → FT232H1
- Sensors 13,14,15,19,20,21 → FT232H2
- Sensors 16,17,18,22,23,24 → FT232H3

The system records from 6 channels per MPR121 (every other channel: 1, 3, 5, 7, 9, 11) to match one sensor per cage.

## Common Commands

### Running the Recording GUI

```bash
# Activate environment
conda activate cliqr

# Start the Solara GUI (recommended)
solara run recording_gui.py
# Opens at http://localhost:8765

# For Windows users: double-click start_cliqr.bat

# Test mode with mock hardware (no physical boards needed)
solara run recording_gui_mock.py
```

### Running Data Analysis

```bash
# Start JupyterLab and open DataAnalysis.ipynb
jupyter-lab
```

### Data Analysis

Open `DataAnalysis.ipynb` and:
1. Set `base_dir` and `animal_id_prefixes` for cohorts to analyze
2. Use the file selector widgets to choose raw data files
3. Run the analysis cells to generate combined results

### Modifying Hardware Configuration

If using different FT232H serial numbers or sensor numbering:
1. Edit `SERIAL_NUMBER_SENSOR_MAP` in `utils/state.py`
2. Edit board ID mappings in `data_analysis.py` lines 114-121
3. (Legacy) Edit `serial_number_sensor_map` in DataRecording.ipynb (cell 10) if still using notebook

## Important Notes

### Recording GUI
- **Hardware initialization**: Click "Initialize Hardware" before starting any recording session
- **Sensor controls**: Individual sensors can only be started after the main "START RECORDING" button is clicked
- **Timer updates**: Sensor timers update every 60 seconds (not real-time)
- **Volume/Weight inputs**: These are disabled once a sensor is recording - enter values before starting
- **Activity log**: Check the activity log for real-time status updates and error messages
- **Mock mode**: Use `recording_gui_mock.py` for testing without physical hardware
- **Browser compatibility**: Tested with Chrome, Firefox, and Edge. Safari may have issues.

### Data Analysis
- **Recording length**: Analysis trims recordings to exactly `recording_length` (default 2 hours) after applying start/stop times. Adjust in DataAnalysis.ipynb cell 5.
- **Layout files**: Each cohort directory should contain `layout.csv` mapping sensors to animal IDs (format: sensor number in index, animal ID in first column)
- **Time synchronization**: All timestamps use `time.time()` (Unix epoch). The first timestamp in a recording is subtracted to normalize times to 0.
- **Missing data handling**: If start/stop times or volumes aren't recorded, the analysis logs warnings and may skip animals. Check `logs/*.log` files.

### Troubleshooting
- **"No FT232H boards found"**: Check USB connections, verify Zadig drivers (Windows), try different USB ports
- **Recording not starting**: Ensure hardware is initialized first, check output directory permissions
- **Sensor timers not updating**: Wait at least 60 seconds, check browser console for errors
- **HDF5 file not created**: Check file permissions, verify output directory exists
- **For more help**: See `DEPLOYMENT.md` and `TEST_MOCK_MODE.md`
