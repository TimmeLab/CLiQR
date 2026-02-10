# DataRecording.ipynb Structure Analysis

This document provides a detailed breakdown of the current DataRecording.ipynb notebook to inform the Solara GUI migration.

## Notebook Cell Organization

### Setup and Configuration (Cells 0-12)
- **Cell 2**: Imports
  - Standard: os, time, datetime, io, collections.deque
  - Data: h5py, pandas, numpy
  - Hardware: pyftdi (I2cController, UsbTools)
  - Async: concurrent.futures.ThreadPoolExecutor, asyncio
  - UI: ipywidgets, IPython.display
  - Visualization: matplotlib.pyplot

- **Cells 3-5**: Hardware constants
  - `SOFT_RESET = 0x80`, `CONFIG = 0x5E`, `DATA = 0x04` (MPR121 registers)
  - `HISTORY_SIZE = 100` (buffer size before HDF5 write)
  - `NUM_CHANNELS = 6` (channels per MPR121)
  - `filename` (global for current recording session)

- **Cells 6-9**: FT232H initialization
  - Create logs/ directory
  - Find all FT232H devices (VID:0x0403, PID:0x6014)
  - Create I2C controllers for each board
  - Auto-detect MPR121 address (tries 0x5A-0x5D)
  - Soft reset and configure each MPR121

- **Cells 10-12**: Sensor mapping
  - `serial_number_sensor_map`: Maps FT232H serial numbers to sensor IDs (1-24)
  - `map_from_sensor_id_to_sn()`: Lookup function for sensor→serial number
  - Assigns sensor lists to each I2C controller

### Core Recording Logic (Cells 13-15)
- **Cell 14**: `record(i2c_port, serial_number)`
  - Reads 24 bytes from MPR121 (2 bytes × 12 channels)
  - Extracts every other channel (1, 3, 5, 7, 9, 11) when NUM_CHANNELS=6
  - Returns (time_data, cap_data, serial_number)

- **Cell 15**: `record_sensors()` async function
  - Main recording loop using ThreadPoolExecutor
  - Maintains per-board, per-sensor deques (maxlen=HISTORY_SIZE)
  - Writes to HDF5 every HISTORY_SIZE loops:
    - First write: creates datasets
    - Subsequent writes: resizes and appends
  - Uses `await asyncio.sleep(0)` to yield control to Jupyter kernel

### Widget Callbacks (Cells 16-21)
- **Cell 17**: `start_stop_all(button)`
  - Creates filename with timestamp
  - Starts/cancels `record_sensors()` async task
  - On stop: saves all volumes/weights to HDF5, saves comments

- **Cell 18**: Per-sensor recording callbacks
  - `sensor_recording(sensor_btn, starting, timer_label)`: Writes start_time/stop_time to HDF5
  - `update_timer(sensor_id, timer_label)`: Updates timer display every 60s

- **Cell 19**: `sensor_test(button)`: Plots last 250 samples (~5 seconds)

- **Cell 20**: Clear output/graph callbacks

- **Cell 21**: `display_layout()`: Handles CSV upload for animal ID mapping

### User Interface (Cells 22-26)
- **Cell 24**: File upload widget for cage layout CSV

- **Cell 26**: Main UI construction
  - Global flags: `recording_all`, `recording_task`
  - Output widgets: `output_area` (text), `graph_area` (plots)
  - Main button: "Start/Stop Recording"
  - **24 sensor control groups**, each containing:
    - Toggle button (Start/Stop)
    - Test button
    - Start volume (BoundedFloatText)
    - Stop volume (BoundedFloatText)
    - Weight (BoundedFloatText)
    - Timer label (HH:MM)
  - Clear buttons for output/graph areas
  - Comments textarea (saved to HDF5 on stop)

## HDF5 File Structure

```
/comments (dataset)
/board_{serial_number}/
  sensor_{sensor_id}/
    cap_data (resizable dataset, chunks=100)
    time_data (resizable dataset, chunks=100)
    start_time (scalar dataset)
    stop_time (scalar dataset)
    start_vol (scalar dataset)
    stop_vol (scalar dataset)
    weight (scalar dataset)
```

## Key Design Patterns

1. **Async Recording**: Uses asyncio.create_task() to run record_sensors() without blocking the UI
2. **Parallel I/O**: ThreadPoolExecutor reads from all FT232H boards simultaneously
3. **Deferred Writes**: Buffers 100 samples before writing to HDF5 to reduce I/O overhead
4. **Dynamic HDF5 Resizing**: First write creates datasets, subsequent writes resize and append
5. **Per-Sensor State**: Each sensor independently tracks start/stop times, volumes, weight
6. **Error Logging**: Writes exceptions and edge cases to logs/{filename}.log
7. **Graceful Shutdown**: Main stop button triggers all sensor stops and saves metadata

## Widget Types Used

- `Button`: Main start/stop, clear buttons
- `ToggleButton`: Per-sensor start/stop (24 instances)
- `BoundedFloatText`: Volume and weight inputs (72 instances)
- `Label`: Timer displays, column headers (24+ instances)
- `Output`: Text output and graph display (2 instances)
- `Textarea`: Comments entry
- `FileUpload`: Layout CSV upload
- `HBox/VBox/GridBox`: Layout containers

## Critical Implementation Notes

### For Solara Migration:

1. **State Management**:
   - Need reactive state for recording_all, recording_task, filename
   - Need per-sensor state: is_recording, start_time, stop_time, volumes, weight
   - Need timer state for each sensor

2. **Async Challenges**:
   - Solara uses its own event loop
   - record_sensors() must integrate with Solara's reactivity
   - Timer updates need to work with Solara's reactive system

3. **Hardware Initialization**:
   - FT232H/MPR121 setup should happen once on app startup
   - Need error handling for disconnected boards
   - Consider adding a "Check Hardware" button

4. **File Management**:
   - Filename generation on start
   - Need to prevent overwriting existing files
   - Consider adding output directory selection

5. **Layout**:
   - 24 sensors arranged in 4 rows of 6 (matching physical rack)
   - Each sensor control is a GridBox with 4×3 grid
   - Main UI is approximately 2000px wide

6. **Testing Mode**:
   - Need mock mode for development without hardware
   - Mock should simulate realistic capacitance patterns
