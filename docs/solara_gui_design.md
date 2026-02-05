# Solara GUI Design Document

## Overview

This document outlines the design for the Solara-based standalone GUI that replaces DataRecording.ipynb. The design maintains all existing functionality while improving usability for non-technical users.

## Design Principles

1. **Preserve Functionality**: No changes to data recording logic or HDF5 structure
2. **Simplify Access**: Remove JupyterLab dependency
3. **Match Physical Layout**: UI mirrors the physical rack arrangement
4. **Clear Visual Feedback**: Status indicators for all recording operations
5. **Error Prevention**: Disable invalid actions, warn before destructive operations

## Page Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     CLiQR Recording System                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Hardware Status                                     │  │
│  │  ● FT232H0: Connected (6 sensors)                   │  │
│  │  ● FT232H1: Connected (6 sensors)                   │  │
│  │  ● FT232H2: Connected (6 sensors)                   │  │
│  │  ● FT232H3: Connected (6 sensors)                   │  │
│  │  [Refresh Hardware]                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Session Controls                                    │  │
│  │  Output Directory: [____________] [Browse]           │  │
│  │  Layout File: [____________] [Upload CSV]            │  │
│  │  [START RECORDING]                                   │  │
│  │  Status: Idle | Recording since: --:--:--            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Sensor Grid (Rack Layout)                           │  │
│  │  ┌────┬────┬────┬────┬────┬────┐                   │  │
│  │  │ S1 │ S2 │ S3 │ S4 │ S5 │ S6 │  Shelf 1 (Top)    │  │
│  │  ├────┼────┼────┼────┼────┼────┤                   │  │
│  │  │ S7 │ S8 │ S9 │S10 │S11 │S12 │  Shelf 2          │  │
│  │  ├────┼────┼────┼────┼────┼────┤                   │  │
│  │  │S13 │S14 │S15 │S16 │S17 │S18 │  Shelf 3          │  │
│  │  ├────┼────┼────┼────┼────┼────┤                   │  │
│  │  │S19 │S20 │S21 │S22 │S23 │S24 │  Shelf 4 (Bottom) │  │
│  │  └────┴────┴────┴────┴────┴────┘                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Activity Log                       [Clear Log]      │  │
│  │  [Scrollable text area showing operations]           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Comments                                             │  │
│  │  [Text area for session notes]                        │  │
│  │  (Auto-saved to HDF5 file on stop)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Sensor Card Design

Each sensor card contains all controls for one sensor:

```
┌──────────────────────────────────────┐
│ Sensor 1 [Animal: A1]          ●IDLE │  ← Status: IDLE/RECORDING/ERROR
├──────────────────────────────────────┤
│ [START]  Timer: 00:00  [TEST]       │
├──────────────────────────────────────┤
│ Start Vol:  [____] mL               │
│ Stop Vol:   [____] mL               │
│ Weight:     [____] g                │
└──────────────────────────────────────┘
```

### Sensor Card States

1. **IDLE** (Gray):
   - Before session starts or after sensor stopped
   - START button disabled if session not started
   - All inputs editable

2. **RECORDING** (Green):
   - During active recording
   - START button changes to STOP button
   - Timer updates every 60 seconds
   - Volume inputs disabled once recording starts

3. **ERROR** (Red):
   - Sensor disconnected or read failure
   - All controls disabled
   - Shows error message

## Component Hierarchy

```python
App()
├── HeaderCard()
│   └── Title, version info
├── HardwareStatusCard()
│   ├── BoardStatus() × 4 (one per FT232H)
│   └── RefreshButton()
├── SessionControlsCard()
│   ├── OutputDirectoryPicker()
│   ├── LayoutFileUpload()
│   ├── StartStopButton()
│   └── SessionStatus()
├── SensorGrid()
│   └── SensorCard() × 24
│       ├── SensorHeader()
│       ├── ControlButtons()
│       ├── Timer()
│       └── VolumeWeightInputs()
├── ActivityLog()
│   └── LogViewer()
└── CommentsCard()
    └── CommentsTextArea()
```

## Reactive State Structure

```python
# Global session state
@solara.reactive
def recording_all():
    return False

@solara.reactive
def filename():
    return ""

@solara.reactive
def output_directory():
    return os.getcwd()

@solara.reactive
def layout_df():
    return pd.DataFrame()  # Animal ID mapping

# Per-sensor state (24 instances)
class SensorState:
    is_recording: bool = False
    start_time: float = 0.0
    elapsed_seconds: int = 0
    start_volume: float = 0.0
    stop_volume: float = 0.0
    weight: float = 0.0
    status: str = "idle"  # idle, recording, error
    error_message: str = ""
    animal_id: str = ""

# Hardware state
@solara.reactive
def boards_connected():
    return {}  # {serial_number: num_sensors}

# Activity log
@solara.reactive
def log_messages():
    return []  # List of timestamped log entries

# Comments
@solara.reactive
def comments():
    return ""
```

## Key Interactions

### Starting a Recording Session

1. User clicks "START RECORDING"
2. System checks:
   - Output directory writable?
   - At least one board connected?
3. Generate filename with timestamp
4. Create HDF5 file
5. Enable all sensor START buttons
6. Start async recording task
7. Update button to "STOP RECORDING"

### Starting a Sensor

1. User clicks sensor's START button
2. Check session is active
3. Write start_time to HDF5
4. Change button to STOP
5. Start timer
6. Update status indicator to RECORDING (green)
7. Log "Sensor X started"

### Stopping a Sensor

1. User clicks sensor's STOP button
2. Write stop_time to HDF5
3. Write volumes and weight to HDF5
4. Stop timer
5. Update status indicator to IDLE (gray)
6. Log "Sensor X stopped (duration: HH:MM:SS)"

### Testing a Sensor

1. User clicks TEST button
2. Read last 250 samples from HDF5
3. Generate matplotlib plot
4. Display in modal dialog or separate area
5. Log "Sensor X tested"

### Stopping a Recording Session

1. User clicks "STOP RECORDING"
2. For each sensor still recording:
   - Automatically stop sensor
   - Save volumes/weights
3. Save comments to HDF5
4. Cancel async recording task
5. Log "Session ended - file saved: {filename}"
6. Disable all sensor controls
7. Update button to "START RECORDING"

## Color Scheme

```python
COLORS = {
    'primary': '#1976D2',      # Blue for main actions
    'success': '#4CAF50',      # Green for recording
    'warning': '#FF9800',      # Orange for warnings
    'error': '#F44336',        # Red for errors
    'idle': '#9E9E9E',         # Gray for idle
    'background': '#FAFAFA',   # Light gray background
    'card': '#FFFFFF',         # White cards
    'text': '#212121',         # Dark text
    'text_secondary': '#757575' # Gray secondary text
}
```

## Layout Dimensions

- **Sensor Card**: 250px × 200px
- **Sensor Grid**: 6 columns × 4 rows
- **Minimum Window Width**: 1600px
- **Activity Log Height**: 300px
- **Comments Height**: 150px

## Accessibility Features

1. **Keyboard Navigation**: All controls accessible via keyboard
2. **Visual Status**: Color + icon + text for status (not color alone)
3. **Large Click Targets**: Buttons minimum 40px height
4. **Clear Labels**: All inputs have associated labels
5. **Error Messages**: Clear, actionable error messages

## Deployment Considerations

### Running the App

```bash
# Development mode
solara run recording_gui.py

# Production mode (server accessible on network)
solara run recording_gui.py --host 0.0.0.0 --port 8000

# Windows shortcut (start_cliqr.bat)
@echo off
call conda activate cliqr
solara run recording_gui.py
pause
```

### File Structure

```
CLiQR-GUI/
├── recording_gui.py          # Main Solara app
├── components/               # Reusable UI components
│   ├── __init__.py
│   ├── sensor_card.py
│   ├── hardware_status.py
│   └── session_controls.py
├── hardware/                 # Hardware interface
│   ├── __init__.py
│   ├── ft232h.py            # FT232H initialization
│   └── mpr121.py            # MPR121 sensor interface
├── recording/               # Recording logic
│   ├── __init__.py
│   ├── recorder.py          # Async recording loop
│   └── hdf5_writer.py       # HDF5 file operations
└── utils/
    ├── __init__.py
    ├── state.py             # Reactive state definitions
    └── logging.py           # Activity log utilities
```

## Migration Strategy

1. **Phase 1**: Create basic Solara app with UI skeleton (no hardware)
2. **Phase 2**: Integrate hardware initialization and mock recording
3. **Phase 3**: Implement full recording loop with real sensors
4. **Phase 4**: Add testing, error handling, and polish
5. **Phase 5**: User testing and refinement

## Differences from Notebook

### Additions
- Hardware status indicator
- Output directory selection
- Visual status indicators (colored dots)
- Activity log with timestamps
- Modal dialogs for test plots

### Removals
- Graph area for inline plotting (moved to modal)
- Clear output/graph buttons (log auto-scrolls)
- Separate layout display (integrated into session controls)

### Changes
- Timer updates every 60s (same as notebook)
- Comments auto-saved on stop (same as notebook)
- Sensor arrangement matches physical rack (same as notebook)
