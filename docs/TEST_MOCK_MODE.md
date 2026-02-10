# Testing CLiQR GUI with Mock Hardware

## Quick Start

To test the GUI without physical hardware:

```bash
# Activate the conda environment
conda activate cliqr

# Run the app with mock hardware
solara run recording_gui_mock.py
```

The app will be available at http://localhost:8765

## What Mock Mode Provides

Mock mode simulates:

1. **4 FT232H boards** (FT232H0-3)
2. **24 MPR121 sensors** (6 per board)
3. **Realistic capacitance data** with:
   - Baseline values around 500
   - Periodic variations simulating licking patterns
   - Random noise (±10 counts)
4. **All HDF5 operations** (files are created normally)

## Testing Checklist

Use this checklist to verify all functionality:

### Hardware Initialization
- [ ] Click "Initialize Hardware" button
- [ ] Verify all 4 boards show as connected
- [ ] Check activity log for initialization messages

### Recording Session
- [ ] Select an output directory
- [ ] Click "START RECORDING"
- [ ] Verify filename is generated with timestamp
- [ ] Check that sensor START buttons are now enabled

### Individual Sensors
- [ ] Click START on sensor 1
- [ ] Verify status changes to RECORDING (green)
- [ ] Wait 60+ seconds, verify timer updates
- [ ] Enter start volume, stop volume, and weight
- [ ] Click STOP
- [ ] Verify status returns to IDLE (gray)
- [ ] Check activity log for start/stop messages

### Volume/Weight Inputs
- [ ] Enter values in start volume field
- [ ] Enter values in stop volume field
- [ ] Enter values in weight field
- [ ] Verify values are saved when sensor is stopped

### Session Comments
- [ ] Enter text in comments area
- [ ] Stop recording session
- [ ] Verify comments are saved to HDF5 file

### Complete Recording Session
- [ ] Start recording
- [ ] Start 2-3 sensors
- [ ] Let them record for at least 5 minutes
- [ ] Stop individual sensors
- [ ] Click "STOP RECORDING"
- [ ] Check that HDF5 file was created

### HDF5 File Verification

After a test recording session, verify the HDF5 file structure:

```python
import h5py

# Open the generated file
with h5py.File('raw_data_YYYY-MM-DD_HH-MM-SS.h5', 'r') as f:
    # Check structure
    print("Groups:", list(f.keys()))

    # Check a board
    board = f['board_FT232H0']
    print("Sensors:", list(board.keys()))

    # Check a sensor
    sensor = board['sensor_1']
    print("Datasets:", list(sensor.keys()))

    # Verify data
    if 'cap_data' in sensor:
        print("Samples recorded:", len(sensor['cap_data']))
        print("First 10 samples:", sensor['cap_data'][:10])

    # Check metadata
    if 'start_time' in sensor:
        print("Start time:", sensor['start_time'][()])
    if 'stop_time' in sensor:
        print("Stop time:", sensor['stop_time'][()])
    if 'start_vol' in sensor:
        print("Start volume:", sensor['start_vol'][()])
    if 'stop_vol' in sensor:
        print("Stop volume:", sensor['stop_vol'][()])
    if 'weight' in sensor:
        print("Weight:", sensor['weight'][()])

    # Check comments
    if 'comments' in f:
        print("Comments:", f['comments'][()])
```

## Expected File Structure

```
/comments (dataset)
/board_FT232H0/
  sensor_1/
    cap_data (dataset with samples)
    time_data (dataset with timestamps)
    start_time (scalar)
    stop_time (scalar)
    start_vol (scalar)
    stop_vol (scalar)
    weight (scalar)
  sensor_2/
    ...
/board_FT232H1/
  ...
```

## Known Limitations

Mock mode does NOT test:
- Real USB communication errors
- I2C address detection failures
- MPR121 configuration issues
- FT232H driver problems (Windows Zadig)
- Actual capacitance sensing

For these scenarios, testing with real hardware (Task #11) is required.

## Troubleshooting

**Problem:** App won't start
- Check that solara is installed: `pip install solara`
- Check Python version: `python --version` (should be 3.13)

**Problem:** Import errors
- Ensure you're in the correct directory (CLiQR-GUI)
- Activate conda environment: `conda activate cliqr`

**Problem:** HDF5 file not created
- Check file permissions in output directory
- Verify output directory exists
- Check activity log for error messages

**Problem:** Sensor timers not updating
- Wait at least 60 seconds (timers update every minute)
- Check browser console for JavaScript errors
