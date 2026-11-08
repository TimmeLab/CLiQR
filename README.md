# CLiQR - Capacitive Lick Quantification in Rodents
### Created for the Timme Lab at University of Cincinnati
### Author: Christopher Parker

This repository contains the software for running our capacitive lickometry system, which records rodent licking behavior using MPR121 capacitive touch sensors connected via FT232H USB-to-I2C boards. 

To avoid confusion about units, we clarify that the data recorded is not actual capacitance values. By repeatedly charging sensing electrodes with a known current over a known amount of time and measuring the resulting voltage, the MPR121 boards report a proxy for capacitance. This is because the higher the total capacitance in the system, the lower the voltage that results from a fixed current over a fixed time. Because the analog-to-digital converter (ADC) on the MPR121 reads and reports this voltage, the output values read from the boards are inversely related to total system capacitance.

## System Overview

**CLiQR** provides web-based (Solara) GUIs for data recording and data analysis:
- **recording_gui.py** - Standalone Solara web GUI for recording (primary interface)
- **analysis_gui.py** - Standalone Solara web GUI for batch analysis and visualization (handles 24 h recordings)
- **DataAnalysis.ipynb** - Previous Jupyter notebook for analysis (superseded by analysis_gui.py, kept for reference)
- **DataRecording.ipynb** - Legacy notebook interface (deprecated, kept for reference)

## Installation

### Windows Driver Setup

Before first use on Windows, you must install drivers for the FT232H boards using Zadig:
1. Download Zadig from https://zadig.akeo.ie/
2. Follow the steps outlined here: https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/windows

### Software Installation

Clone this repository (or download and extract the .zip file).

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

The FT232H boards must be assigned serial numbers so the system can identify them consistently, even if USB ports end up being shuffled somehow. Connect boards **ONE AT A TIME** and run:

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

Hold Shift and right-click start_cliqr.bat, then select Send To -> Desktop (create shortcut). You can then rename the desktop shortcut as you wish, I have just called it "CLiQR." Double-click the shortcut to launch the GUI without using the command line. Note, this shortcut will only work if you used the miniforge3 installation instructions.

### Recording Workflow

1. **Initialize Hardware** - Click "Initialize Hardware" to detect and connect to FT232H boards
2. **Upload Layout File** - Drag and drop a layout CSV file mapping sensor positions to animal IDs (see `layouts/default_layout.csv` for format)
3. **Enter Animal Weights** - Enter weights in grams for each animal before recording
4. **Start Session** - Click "START RECORDING" to begin the recording session
5. **Start Individual Sensors** - Click START on each sensor card to begin recording for that animal
6. **Test Sensors** - Use the TEST button to view recent data and confirm sensors are working
7. **Enter Start Volumes** - After starting each sensor, enter the initial sipper volume in mL in the Start Vol box
8. **Stop Individual Sensors** - Click STOP when done recording each animal
9. **Enter Stop Volumes** - After stopping each sensor, enter the final sipper volume in mL in the Stop Vol box
10. **End Session** - Click "STOP RECORDING" to end the session and save all data to HDF5

**Important Notes:**
- All volume/weight data and any comments in the Comments box are written to the HDF5 file when you click "STOP RECORDING"
- Data is saved to the "Lickometry Data" directory by default
- Each session creates a timestamped HDF5 file: `raw_data_YYYY-MM-DD_HH-MM-SS.h5`

### Data Analysis

**Option 1 - Command Line:**
```bash
conda activate cliqr  # or: pyenv activate cliqr
solara run analysis_gui.py --port 8766
```

The GUI opens at http://localhost:8766 (port 8766 so it can run next to the recording GUI on 8765). Set `CLIQR_DATA_DIR` to start in a data folder other than `Lickometry Data`.

**Option 2 - Windows Desktop Shortcut:** create a shortcut to `start_analysis.bat` the same way as for `start_cliqr.bat`.

The analysis GUI replaces `DataAnalysis.ipynb`, which became unusably slow with 24 h runs (~5 GB per day). It is organised as tabs:

| Tab | What it does |
|---|---|
| **Run** | Pick cohort folders (sub-folders of the data directory that contain `layout_w_controls.csv` or `layout.csv`) and raw files, choose the algorithm and **recording length** (2 h for the standard protocol, 24 for day-long runs), then *Run analysis*. Detection runs in the background with a progress bar and can be stopped between files. Or open an existing `results_combined_*.h5`. |
| **Start/Stop check** | Plot any sensor of a raw file with its recorded start/stop times, to decide whether a time fix file is needed. |
| **Summary** | Licks per session, lick count vs volume, control-cage false positives and evaporation, and a table of every session. |
| **Behavior** | ILI, licks per bout and bout duration histograms, plus lick/bout time courses (per-bin and cumulative, adjustable bin size). |
| **Traces** | Scroll and zoom through any session's trace with detected licks marked (1 min / 10 min / 1 h / full-day windows). |
| **Correlation** | OLS (HC3) and robust fits of volume vs licks, residual plots, MAD outliers (tick to exclude), cleaned fit and the 0.05 mL correlation ceiling. |
| **Per-animal** | Per-animal volume vs lick fits (mL per lick). |
| **Outliers removed** | Behavioral metrics and time courses recomputed without the excluded sessions. |

Each card has an export button that writes the same CSV files the notebook produced (`cliqr_ILIs.csv`, `cliqr_cumulative_licks.csv`, `nLicksVsVol_outliers_removed.csv`, ...) to the export folder shown in the Activity log card (the repo folder by default).

How it stays fast on long runs: raw files are processed one sensor at a time (memory stays well under 1 GB for a 24-sensor, 24 h file), the threshold scan in `basic_algorithm()` is vectorized (identical results; about 5 s per sensor for a 24 h trace), summaries only read lick times and volumes, and trace plots read only the visible window, reduced to a min/max envelope that keeps every lick dip.

To try the GUI without real data (the example `.h5` files are Git LFS pointers):
```bash
python scripts/make_synthetic_recording.py "Lickometry Data/Synthetic" --days 3 --hours 2
```

#### Time Fix Files

If a recording has incorrect start/stop times (check them on the **Start/Stop check** tab), create `<raw_filename>_time_fix.xlsx` next to the raw file (e.g. `raw_data_2026-01-20_11-10-18_time_fix.xlsx`) with three columns:

| Sensor | New Start Time | New End Time |
|--------|----------------|--------------|
| 20     | 1756825855     | 1756833056   |

Times are Unix epoch seconds. Time fix files are applied automatically on the next run (this can be switched off on the Run tab), and each applied fix is noted in the Activity log.

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
- Verify libusbK drivers are installed (Windows) or libusb is installed (Unix-based)
- Try different USB ports or USB-C cables (not all USB cables are created alike!)

**Recording not starting**
- Ensure hardware is initialized first
- Check that layout file has been uploaded
- Verify output directory permissions

**Sensor shows no data**
- Use the TEST button to check if sensor is responding
- Check physical connections to MPR121 boards
- Ensure FT232H boards are in I2C mode (there is a small switch on the board)

For more details, see `DEPLOYMENT.md` and `TEST_MOCK_MODE.md`.

## System Architecture

The system supports 24 sensors arranged in a 4×6 grid (matching the physical rack layout):
- 4 FT232H boards (FT232H0-3)
- 4 MPR121 capacitive touch sensors (one per FT232H board)
- 6 sensors per board (using every other channel: 1, 3, 5, 7, 9, 11)

Data is recorded at approximately 100 Hz and saved in HDF5 format with the structure:
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
