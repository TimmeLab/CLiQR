# CLiQR GUI Deployment Guide

## For End Users (Windows)

### First-Time Setup

1. **Install Miniforge** (if not already installed)
   - Download from: https://github.com/conda-forge/miniforge
   - Run the installer and follow prompts
   - Restart your computer

2. **Install FT232H Drivers** (Windows only)
   - Download Zadig from: https://zadig.akeo.ie/
   - Connect one FT232H board via USB
   - Run Zadig and select the FT232H device
   - Install the WinUSB driver
   - Repeat for each FT232H board

3. **Set up the CLiQR environment**
   - Open Miniforge Prompt (search for it in Start menu)
   - Navigate to the CLiQR-GUI folder:
     ```
     cd path\to\CLiQR-GUI
     ```
   - Create the environment:
     ```
     conda env create --file environment.yml
     ```

4. **Create a Desktop Shortcut** (Windows)
   - Right-click on `start_cliqr.bat`
   - Select "Create shortcut"
   - Move the shortcut to your Desktop
   - Rename it to "CLiQR Recording System"

### Daily Use

1. **Connect Hardware**
   - Connect all 4 FT232H boards via USB
   - Ensure MPR121 sensors are properly connected via I2C

2. **Start the Application**
   - Double-click the "CLiQR Recording System" shortcut
   - A browser window will open automatically
   - If not, navigate to: http://localhost:8765

3. **Initialize Hardware**
   - Click the "Initialize Hardware" button
   - Wait for all 4 boards to show as connected

4. **Start Recording**
   - (Optional) Upload a cage layout CSV file
   - Select output directory if desired
   - Click "START RECORDING"
   - Individual sensor START buttons will become enabled

5. **Record Data**
   - Click START on each sensor you want to record
   - Enter start volume, stop volume, and weight for each animal
   - Click STOP when done with each sensor

6. **End Session**
   - Add any comments in the comments area
   - Click "STOP RECORDING"
   - Data is automatically saved to HDF5 file

7. **Close Application**
   - Close the browser window
   - Press any key in the command window to exit

## For Developers

### Running from Source

```bash
# Activate environment
conda activate cliqr

# Run with real hardware
solara run recording_gui.py

# Run with mock hardware (for testing)
solara run recording_gui_mock.py

# Run on network (accessible from other computers)
solara run recording_gui.py --host 0.0.0.0 --port 8000
```

### Project Structure

```
CLiQR-GUI/
├── recording_gui.py          # Main app (real hardware)
├── recording_gui_mock.py     # App with mock hardware
├── components/               # UI components
│   ├── hardware_status.py   # Hardware initialization UI
│   ├── session_controls.py  # Start/stop recording controls
│   └── sensor_card.py       # Per-sensor controls (24 cards)
├── hardware/                # Hardware interface
│   ├── ft232h.py           # FT232H board management
│   ├── mpr121.py           # MPR121 sensor interface
│   └── mock_hardware.py    # Simulated hardware for testing
├── recording/              # Recording logic
│   └── recorder.py         # Async recording loop and HDF5 writer
├── utils/
│   └── state.py           # Reactive state management
├── docs/                  # Documentation
│   ├── notebook_analysis.md
│   └── solara_gui_design.md
└── DataAnalysis.ipynb     # Analysis workflow (unchanged)
```

### Making Changes

1. **Modify UI Components**: Edit files in `components/`
2. **Change Hardware Interface**: Edit files in `hardware/`
3. **Modify Recording Logic**: Edit `recording/recorder.py`
4. **Add New State Variables**: Edit `utils/state.py`
5. **Test Changes**: Run `recording_gui_mock.py` to test without hardware

### Code Style

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Use type hints where appropriate
- Keep component functions pure (no side effects)

## Packaging for Distribution

### Option 1: Conda Environment (Recommended)

Users install Miniforge and create the environment from `environment.yml`.

**Pros:**
- Easy updates (git pull + conda env update)
- Works on all platforms
- No build step required

**Cons:**
- Requires Miniforge installation
- Users must use command line initially

### Option 2: PyInstaller (Future Work)

Package the app as a standalone executable.

```bash
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller --onefile --windowed --name CLiQR recording_gui.py

# Executable will be in dist/
```

**Pros:**
- No Python installation required
- Single-file distribution

**Cons:**
- Large file size (~100MB)
- Platform-specific (must build on Windows for Windows)
- Harder to update
- May have issues with Solara

**Status:** Not yet implemented - requires testing

## Troubleshooting

### "No FT232H boards found"

- Check USB connections
- Verify Zadig drivers are installed (Windows)
- Try different USB ports
- Check if FT232H is in I2C mode (not UART)

### "Recording not starting"

- Ensure hardware is initialized first
- Check output directory exists and is writable
- Look for errors in activity log

### Sensor data looks wrong

- Click TEST button to view recent data
- Check sensor connections (I2C wiring)
- Verify MPR121 configuration
- Restart the application

### Browser won't connect

- Check that port 8765 is not in use
- Try http://127.0.0.1:8765 instead of localhost
- Check firewall settings
- Try a different browser

### Performance issues

- Close other applications
- Reduce number of sensors recording simultaneously
- Use a faster USB hub
- Check system resources (CPU, RAM)

## System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, macOS 10.15+, or Linux
- **CPU:** Dual-core 2.0 GHz
- **RAM:** 4 GB
- **Storage:** 1 GB free space
- **USB:** 4 available USB 2.0 ports

### Recommended
- **OS:** Windows 11
- **CPU:** Quad-core 2.5 GHz or better
- **RAM:** 8 GB or more
- **Storage:** 10 GB free space (for data)
- **USB:** Powered USB 3.0 hub with 4+ ports

## Data Management

### File Naming Convention

Files are named: `raw_data_YYYY-MM-DD_HH-MM-SS.h5`

Example: `raw_data_2025-02-05_14-30-00.h5`

### Storage Recommendations

- **Per Session:** ~50-500 MB (depending on duration)
- **Backup:** Copy HDF5 files to network drive daily
- **Long-term Storage:** Keep raw files for at least 1 year

### File Organization

Recommended folder structure:

```
Data/
├── 2025/
│   ├── February/
│   │   ├── Cohort_A/
│   │   │   ├── raw_data_2025-02-05_14-30-00.h5
│   │   │   ├── layout.csv
│   │   │   └── time_fix.xlsx (if needed)
│   │   └── Cohort_B/
│   │       └── ...
│   └── March/
│       └── ...
└── 2024/
    └── ...
```

## Getting Help

- **Bug Reports:** https://github.com/timmelab/CLiQR/issues
- **Email:** parkecp@ucmail.uc.edu
- **Documentation:** See `docs/` folder
- **Analysis Help:** See `DataAnalysis.ipynb`
