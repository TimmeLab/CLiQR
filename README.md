# CLiQR - Capacitive Lick Quantification in Rodents
### Created for the Timme Lab at University of Cincinnati
### Author: Christopher Parker

This repository contains the software for running our capacitive lickometry system, which records rodent licking behavior using MPR121 capacitive touch sensors connected via FT232H USB-to-I2C boards.

## System Overview

**CLiQR** provides a web-based GUI for data recording and Jupyter notebooks for data analysis:
- **recording_gui.py** - Standalone Solara web GUI for recording (primary interface)
- **DataAnalysis.ipynb** - Jupyter notebook for batch analysis and visualization
- **DataRecording.ipynb** - Legacy notebook interface (deprecated, kept for reference)

## Installation

### Windows Driver Setup (Required for Windows)

Before first use on Windows, you must install drivers for the FT232H boards using Zadig:
1. Download Zadig from https://zadig.akeo.ie/
2. Follow the steps outlined here: https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/windows

### Software Installation

Clone this repository (or download and extract the .zip file), then navigate to the directory in your terminal.

**Option 1 - Miniforge (recommended for Windows):**

Download and install Miniforge: https://github.com/conda-forge/miniforge/releases

On Windows, use the "Miniforge Prompt" from the Start menu. On Mac/Linux, use your normal terminal. Then run:
```bash
conda env create --file environment.yml
conda activate cliqr
```

**Option 2 - pyenv-virtualenv (Unix-based systems):**

Install pyenv-virtualenv: https://github.com/pyenv/pyenv-virtualenv

Then run:
```bash
pyenv virtualenv 3.13 cliqr
pyenv activate cliqr
pip install -r requirements.txt
```

### Hardware Setup (One-Time Configuration)

The FT232H boards must be assigned serial numbers so the system can identify them consistently, even if USB connections change. Connect boards **ONE AT A TIME** and run:

```bash
python set_ft232h_serial.py FT232H0
```

Repeat for each board, using serial numbers FT232H0 through FT232H3. These serial numbers map to sensors 1-24 as defined in `utils/state.py`. If you use different serial numbers, update the `SERIAL_NUMBER_SENSOR_MAP` constant.

## Running the System

### Data Recording

**Option 1 - Command Line:**
```bash
conda activate cliqr  # or: pyenv activate cliqr
solara run recording_gui.py
```

The GUI will open in your default web browser at http://localhost:8765

**Option 2 - Windows Desktop Shortcut:**

Double-click `start_cliqr.bat` to launch the GUI without using the command line.

### Recording Workflow

1. **Initialize Hardware** - Click "Initialize Hardware" to detect and connect to FT232H boards
2. **Upload Layout File** - Drag and drop a layout CSV file mapping sensor positions to animal IDs (see `layouts/default_layout.csv` for format)
3. **Enter Animal Weights** - Enter weights in grams for each animal before recording
4. **Start Session** - Click "START RECORDING" to begin the recording session
5. **Start Individual Sensors** - Click START on each sensor card to begin recording for that animal
6. **Test Sensors** - Use the TEST button to view recent data and confirm sensors are working
7. **Enter Start Volumes** - After starting each sensor, enter the initial sipper volume in mL (can be entered while recording)
8. **Stop Individual Sensors** - Click STOP when done recording each animal
9. **Enter Stop Volumes** - After stopping each sensor, enter the final sipper volume in mL
10. **End Session** - Click "STOP RECORDING" to end the session and save all data to HDF5

**Important Notes:**
- Volume and weight inputs can be edited at any time during the session
- All volume/weight data is written to the HDF5 file when you click "STOP RECORDING"
- Data is saved to the "Lickometry Data" directory by default
- Each session creates a timestamped HDF5 file: `raw_data_YYYY-MM-DD_HH-MM-SS.h5`

### Data Analysis

For batch analysis of recorded data:
```bash
conda activate cliqr
jupyter-lab
```

Open `DataAnalysis.ipynb` and follow the instructions to analyze multiple recording sessions.

## Layout File Format

Layout files map sensor numbers (1-24) to animal IDs. The file should be CSV format with no header:

```
1,A1
2,A2
3,A3
...
24,A24
```

A default template is provided at `layouts/default_layout.csv`.

## Troubleshooting

**"No FT232H boards found"**
- Check USB connections
- Verify Zadig drivers are installed (Windows)
- Try different USB ports

**Recording not starting**
- Ensure hardware is initialized first
- Check that layout file has been uploaded
- Verify output directory permissions

**Sensor shows no data**
- Use the TEST button to check if sensor is responding
- Check physical connections to MPR121 boards

For more details, see `DEPLOYMENT.md` and `TEST_MOCK_MODE.md`.

## System Architecture

The system supports 24 sensors arranged in a 4×6 grid (matching the physical rack layout):
- 4 FT232H boards (FT232H0-3)
- 4 MPR121 capacitive touch sensors (one per board)
- 6 sensors per board (using every other channel: 1, 3, 5, 7, 9, 11)

Data is recorded at approximately 50 Hz and saved in HDF5 format with the structure:
```
/board_{serial_number}/sensor_{sensor_id}/
    ├── cap_data       # Capacitance readings
    ├── time_data      # Timestamps
    ├── start_time     # Recording start timestamp
    ├── stop_time      # Recording stop timestamp
    ├── start_vol      # Initial sipper volume (mL)
    ├── stop_vol       # Final sipper volume (mL)
    └── weight         # Animal weight (g)
```

Multiple start/stop cycles per sensor are supported with numbered suffixes: `start_time1`, `start_time2`, etc.
