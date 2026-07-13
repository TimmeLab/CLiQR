# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

CLiQR (Capacitive Lick Quantification in Rodents) records rodent licking behavior using MPR121 capacitive touch sensors connected via FT232H USB-to-I2C boards. Built for the Timme Lab at University of Cincinnati. Runs on Windows (lab PCs) and macOS (development).

Four subsystems:
1. **Recording GUI** — complete, in production
2. **Data analysis pipeline** — functional, actively maintained for manuscript
3. **False positive analysis** — implemented, validating detection accuracy against CVAT video annotations
4. **Concurrent video capture** — Pi 5 + Camera 3 over TCP; bookmarks video frame at sipper insertion. See `docs/VIDEO_CAPTURE.md`.

## Current Development Focus

Recording GUI is done. Active work is the false positive analysis pipeline (validating CLiQR lick detection accuracy against hand-annotated video). See `CURRENT_STATE.md` for current results.

## Environment Setup

**Option 1 — Miniforge (Windows, recommended for lab use):**
```bash
conda env create --file environment.yml
conda activate cliqr
```

**Option 2 — pyenv-virtualenv (macOS/Linux, development):**
```bash
pyenv virtualenv 3.13 cliqr
pyenv activate cliqr
pip install -r requirements.txt
```

**Hardware setup (one-time):** Assign serial numbers to FT232H boards one at a time:
```bash
python set_ft232h_serial.py FT232H0   # repeat FT232H1, FT232H2, FT232H3
```

**Windows only:** Install libusbK drivers for each FT232H board using Zadig before first use.

## Common Commands

```bash
# Recording GUI (real hardware)
solara run recording_gui.py        # opens at http://localhost:8765

# Recording GUI (mock hardware, no boards needed)
solara run recording_gui_mock.py

# Data analysis
jupyter-lab                        # then open DataAnalysis.ipynb

# False positive analysis
jupyter-lab                        # then open FalsePositive.ipynb
```

## Architecture

### Recording GUI (`recording_gui.py`)

Solara web app at `localhost:8765`. Modular structure:

| Module | Purpose |
|---|---|
| `utils/state.py` | Reactive state for 24 sensors, session, hardware |
| `hardware/ft232h.py` | FT232H USB scan, I2C controller init, auto-detect MPR121 address |
| `hardware/mpr121.py` | MPR121 soft reset, config, channel reads |
| `hardware/mock_hardware.py` | Simulated hardware for testing |
| `recording/recorder.py` | Async loop, ThreadPoolExecutor reads, buffered HDF5 writes |
| `components/hardware_status.py` | Board init UI |
| `components/session_controls.py` | Start/stop session, layout file upload (CSV or XLSX), output dir |
| `components/sensor_card.py` | 24 per-sensor cards |
| `components/plot_dialog.py` | Live test plot of recent sensor data |

HDF5 output structure:
```
raw_data_YYYY-MM-DD_HH-MM-SS.h5
/comments
/board_{serial_number}/sensor_{sensor_id}/
    cap_data, time_data, start_time[N], stop_time[N], start_vol, stop_vol, weight
```
Multiple start/stop cycles per sensor use numbered suffixes: `start_time1`, `start_time2`, etc.

### Data Analysis (`data_analysis.py` + `DataAnalysis.ipynb`)

`data_analysis.py` is the core library. `DataAnalysis.ipynb` is the batch analysis notebook (Panel widgets).

Key functions:

| Function | Description |
|---|---|
| `filter_data()` | Top-level: loads raw HDF5, trims to start/stop, runs algorithm |
| `basic_algorithm()` | Threshold-based peak detection. Scans all inter-value thresholds, picks one maximizing peak count. Requires 2-threshold depth. |
| `hilbert_algorithm()` | Bandpass 8–12 Hz → Hilbert envelope → threshold + neighbor filtering. |
| `_run_optimal_threshold()` | Grid-search threshold fraction maximizing R² vs. volume. Runs as comparison baseline. |
| `compute_bout_structure()` | ILIs, bout lick counts, bout durations. Default params: `ibi_threshold=0.25, min_licks=3`. Call sites in notebook use `ibi_threshold=1.0, min_licks=2`. |
| `save_filtered_data()` | Writes per-animal HDF5 group. |

Notebook pipeline:
1. Set `base_dir`, `animal_id_prefixes`, `recording_length`
2. Load per-cohort `layout.csv` files
3. File selector GUI → pick raw HDF5 files per cohort
4. Run `filter_data()` → `filtered_*.h5` per file
5. Combine into `results_combined_*.h5` by animal ID
6. Algorithm comparison (CLiQR vs. optimal threshold)
7. Behavioral metrics: ILI distribution, licks/bout, bout duration
8. Temporal dynamics: 5-min bins across 2-hour session
9. Correlation: OLS + RLM (HC3-robust), MAD-based outlier detection
10. CSV exports for Prism

**Time fix mechanism:** If start/stop times in raw HDF5 are wrong, create `<raw_filename>_time_fix.xlsx` with columns `[Sensor, New Start Time, New End Time]`.

### False Positive Analysis (`false_positive_analysis.py` + `FalsePositive.ipynb`)

Validates CLiQR lick detection accuracy against CVAT-annotated video. Completely standalone — no imports from `data_analysis.py`.

Pipeline functions:

| Function | Description |
|---|---|
| `parse_job_annotations(xml_path)` | Parse per-job CVAT XML → `{frame_id: [labels]}` |
| `parse_annotations(xml_path)` | Parse project-level CVAT XML with task metadata |
| `load_frame_offsets(txt_path)` | Load picamera `.txt` frame timestamp file as numpy array |
| `frames_to_relative_seconds()` | Convert frame IDs → seconds since video start |
| `build_ground_truth()` | Pair bout starts/ends, inconclusive regions, sipper events → intervals |
| `load_sensor_data()` | Load raw HDF5 cap_data/time_data for a sensor |
| `detect_sipper_step()` | Detect sipper insertion/removal as a step change in cap_data |
| `establish_alignment()` | Anchor video clock to HDF5 clock via sipper insertion. Optional drift correction via sipper removal. |
| `video_relative_to_abs()` | Convert video-relative timestamps to Unix seconds |
| `intervals_to_abs()` | Convert bout/inconclusive intervals to absolute time |
| `load_lick_times_abs()` | Load CLiQR lick_times from filtered HDF5, convert to Unix seconds |
| `classify_licks()` | Label each lick as TP, FP, or excluded (inconclusive window) |
| `plot_session()` | Per-session figure: cap trace + bout regions + colored lick markers |
| `build_results_dataframe()` | Aggregate per-session results into DataFrame |

**Time alignment:** Pi clock unreliable; only relative frame offsets in `.txt` files are trustworthy. Sipper insertion (detected as a step-down in cap_data in the pre-start window) is the primary anchor. Sipper removal (step-up after stop_time) provides optional drift correction.

**Notebook (`FalsePositive.ipynb`):** Manifest-driven. Edit `SESSION_DIR`. Reads `session_manifest.csv` with columns `task_id, xml_path, txt_path, raw_h5, filtered_h5, animal_id, sensor_num`. Produces per-session HTML/PNG figures and `false_positive_results.csv`.

### Concurrent Video Capture (`pi/`, `video/`, `hardware/pi_camera.py`)

Desktop ↔ Pi 5 over TCP. `video/protocol.py` (shared wire format), `pi/server_core.py` (dispatcher), `pi/pi_camera_server.py` (TCP + picamera2 entry), `pi/camera_backend.py` (picamera2), `hardware/pi_camera.py` (desktop client), `hardware/pi_camera_mock.py` (no-Pi mock), `components/camera_controls.py` (UI). Bookmarks stored in HDF5 as `video_frame_index`/`video_pts`/`video_filename`. Full guide: `docs/VIDEO_CAPTURE.md`.

## Hardware: Sensor → Board Mapping

Hardcoded in **two places** — `utils/state.py` (SERIAL_NUMBER_SENSOR_MAP) and `data_analysis.py:114–121`. Change both if layout changes.

| Board | Sensors |
|---|---|
| FT232H0 | 1, 2, 3, 7, 8, 9 |
| FT232H1 | 4, 5, 6, 10, 11, 12 |
| FT232H2 | 13, 14, 15, 19, 20, 21 |
| FT232H3 | 16, 17, 18, 22, 23, 24 |

MPR121 reads every other channel: 1, 3, 5, 7, 9, 11 (6 sensors per board).

## Known Issues / Technical Debt

1. **Sensor-board mapping duplicated** — `utils/state.py` and `data_analysis.py:114–121` both hardcode the same mapping.
2. **`hilbert_algorithm()` filter passes** — `data_analysis.py:276` TODO: currently applies bandpass 7× total, may over-smooth.
3. **`compute_bout_structure()` param inconsistency** — function defaults are `ibi_threshold=0.25, min_licks=3`; notebook call sites use `ibi_threshold=1.0, min_licks=2`.
4. **ML experiments not integrated** — `checkpoints/best.pt` and training data exist but no training or inference code is in the repo.

## Important Notes

- **Recording:** Initialize hardware before starting sessions. Volume/weight inputs disabled once sensor is recording.
- **Layout files:** CSV or XLSX. Format: sensor number in index, animal ID in first column. Template at `layouts/default_layout.csv`.
- **Timestamps:** `time.time()` (Unix epoch) throughout. `time_data` in raw HDF5 is absolute Unix seconds.
- **False positive data:** CVAT annotations, picamera MP4/TXT files, and raw/filtered HDF5s all live under `Lickometry Data/ACG-26-3/`. Pipeline driven by `session_manifest.csv` in that directory.
- **Mock mode:** `solara run recording_gui_mock.py` for UI testing without physical hardware.
- **Browser:** Chrome/Firefox/Edge. Safari may have issues with Solara.
