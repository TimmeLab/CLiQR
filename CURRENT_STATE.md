# CLiQR — Current State (as of 2026-05-28)

## System Summary

CLiQR (Capacitive Lick Quantification in Rodents) records rodent licking behavior via MPR121 capacitive touch sensors on FT232H USB-to-I2C boards. Two primary subsystems: **recording GUI** (complete, in production) and **data analysis pipeline** (functional, actively maintained for manuscript). A third subsystem, **false positive analysis**, has been implemented and run on the ACG-26-3 cohort.

Hardware: 4 FT232H boards × 1 MPR121 each × 6 channels = 24 sensors. Sampling ~56 Hz. Data stored as HDF5.

---

## Component Status

### Recording GUI — COMPLETE

Entry point: `recording_gui.py` (Solara web app, `localhost:8765`)

| Module | Purpose | Status |
|---|---|---|
| `utils/state.py` | Reactive state for all 24 sensors, session, hardware | Complete |
| `hardware/ft232h.py` | FT232H USB scan, I2C controller init, auto-detect MPR121 address | Complete |
| `hardware/mpr121.py` | MPR121 soft reset, config, channel reads (channels 1,3,5,7,9,11) | Complete |
| `hardware/mock_hardware.py` | Simulated hardware for testing | Complete |
| `recording/recorder.py` | Async loop, ThreadPoolExecutor reads, buffered HDF5 writes | Complete |
| `components/hardware_status.py` | Board init UI | Complete |
| `components/session_controls.py` | Start/stop session, layout file upload (CSV/XLSX), output dir | Complete |
| `components/sensor_card.py` | 24 per-sensor cards: start/stop, timer, test button, vol/weight inputs | Complete |
| `components/plot_dialog.py` | Live test plot of recent sensor data | Complete |

---

### Data Analysis Pipeline — FUNCTIONAL

#### Core library: `data_analysis.py`

| Function | Description |
|---|---|
| `filter_data()` | Top-level: loads raw HDF5, trims to start/stop times, calls algorithm |
| `basic_algorithm()` | Threshold-based peak detection. Scans all inter-value thresholds, picks the one maximizing peak count. Requires 2-threshold depth. |
| `hilbert_algorithm()` | Bandpass 8–12 Hz → Hilbert envelope → threshold + neighbor filtering. Known issue: 7 filter passes (see Known Issues). |
| `_run_optimal_threshold()` | Grid-search threshold fraction maximizing R² vs. volume. Comparison baseline. |
| `compute_bout_structure()` | ILIs, bout lick counts, bout durations. |
| `save_filtered_data()` | Writes per-animal HDF5 group. |

#### Notebook: `DataAnalysis.ipynb`

Multi-cohort batch analysis with Panel widgets. Includes:
- File selector GUI for raw HDF5 files per cohort
- Lick detection via `filter_data()`
- Algorithm comparison (CLiQR vs. optimal threshold)
- Behavioral metrics: ILI distribution, licks/bout, bout duration
- Temporal dynamics: 5-min bins across 2-hour session
- Correlation analysis: OLS + RLM (HC3-robust), MAD-based outlier detection
- Outlier exclusion UI → re-fit clean regression
- CSV exports for Prism

Recently updated to include outlier handling and additional analyses requested by reviewers (eNeuro submission).

**Output CSVs** (repo root):
- `cliqr_ILIs.csv`, `cliqr_ILIs_outliers_removed.csv`
- `cliqr_bout_durations.csv`, `cliqr_bout_durations_outliers_removed.csv`
- `cliqr_licks_per_bout.csv`, `cliqr_licks_per_bout_outliers_removed.csv`
- `cliqr_cumulative_licks.csv`, `cliqr_cumulative_licks_outliers_removed.csv`
- `cliqr_cumulative_bouts.csv`, `cliqr_cumulative_bouts_outliers_removed.csv`
- `nLicksVsVol_outliers_removed.csv`
- `optimal_threshold_lick_counts_and_volume.csv`

---

### False Positive Analysis — IMPLEMENTED

Validates CLiQR detection accuracy by comparing detected lick times against CVAT-annotated licking bouts from video.

#### Core library: `false_positive_analysis.py` (777 lines)

Full pipeline — standalone, no imports from `data_analysis.py`:

1. **`parse_job_annotations()` / `parse_annotations()`** — parse CVAT XML exports
2. **`load_frame_offsets()`** — load picamera `.txt` frame timestamps
3. **`frames_to_relative_seconds()`** — frame IDs → seconds since video start
4. **`build_ground_truth()`** — pair bout start/end tags → licking bout intervals; pair inconclusive regions
5. **`load_sensor_data()`** — load raw HDF5 cap_data and time_data
6. **`detect_sipper_step()`** — detect sipper insertion/removal as step change in cap_data
7. **`establish_alignment()`** — anchor video clock to HDF5 Unix clock via sipper insertion (step-down in pre-start window). Optional drift correction via sipper removal (step-up after stop_time).
8. **`load_lick_times_abs()`** — load CLiQR lick times from filtered HDF5, convert to Unix seconds
9. **`classify_licks()`** — label each lick TP / FP / excluded (inconclusive window)
10. **`plot_session()`** — per-session figure: cap trace + shaded bout regions + colored lick markers
11. **`build_results_dataframe()`** — aggregate results across sessions

**Time alignment note:** Pi clock unreliable; only relative frame offsets in `.txt` files are trustworthy. Sipper insertion detected as a step-**down** in cap_data (sipper insertion decreases capacitance baseline). Sipper removal detected as step-**up** after stop_time for optional drift correction.

#### Notebook: `FalsePositive.ipynb`

Manifest-driven pipeline. Edit `SESSION_DIR` to point at a cohort folder. Reads `session_manifest.csv` with columns `task_id, xml_path, txt_path, raw_h5, filtered_h5, animal_id, sensor_num`. Produces per-session HTML/PNG figures and `false_positive_results.csv`.

Sections: Configuration → Load Manifest → Run Pipeline → Alignment QC → Summary Table + Bar Chart → Export CSV.

#### Results — ACG-26-3 Cohort

4 sessions analyzed, saved in `false_positive_results.csv` and plotted in `False Positive Figures/`:

| Session | Animal | Sensor | n_licks | TP | FP | Excluded | FP Rate |
|---|---|---|---|---|---|---|---|
| ACG-26-3-1 Day 1 | ACG-26-3-1 | 1 | 499 | 433 | 66 | 0 | 13.2% |
| ACG-26-3-1 Day 2 | ACG-26-3-1 | 1 | 33 | 10 | 6 | 17 | 37.5% |
| ACG-26-3-1 Day 4 | ACG-26-3-1 | 1 | 510 | 320 | 187 | 3 | 36.9% |
| ACG-26-3-8 Day 5 | ACG-26-3-8 | 15 | 482 | 481 | 1 | 0 | 0.2% |

---

### Concurrent Video Capture — IMPLEMENTED

Pi 5 + Pi Camera 3 records video in sync with capacitive recording, over TCP.

- `video/protocol.py` — shared newline-JSON wire protocol
- `pi/server_core.py` — hardware-independent request dispatcher
- `pi/pi_camera_server.py` — threaded TCP server + `__main__` entry
- `pi/camera_backend.py` — picamera2 backend (pre-roll + PTS log + bookmark)
- `hardware/pi_camera.py` — desktop TCP client (best-effort, non-blocking)
- `hardware/pi_camera_mock.py` — in-memory mock for tests / mock GUI
- `components/camera_controls.py` — GUI card (enable, host/port, camera sensor, test)

Per-sensor Start bookmarks the current video frame; `video_frame_index` /
`video_pts` / `video_filename` written into the session HDF5. Replaces sipper-step
alignment for new recordings. Guide: `docs/VIDEO_CAPTURE.md`.

---

## Data on Disk

**Cohorts analyzed (lickometry):** AEW2, AEW4, AEW5, AEW6, Example

**Cohort analyzed (false positive validation):** ACG-26-3
- Raw HDF5s, filtered HDF5s, CVAT annotation XMLs, picamera MP4s and TXT frame-offset files, session_manifest.csv — all in `Lickometry Data/ACG-26-3/`

**Combined results files** (in `Lickometry Data/`):
- `results_combined_Example_2025-09-01.h5`
- `results_combined_AEW4-AEW5-AEW6_2026-01-19_20_21_22_23.h5`
- `results_combined_AEW4-AEW5-AEW6_2025-09-01_02_03_04_05_10-13_14_15_16_17_2026-01-20_21_22_23.h5` (most complete)
- `results_combined_ACG-26-3_2026-05-19_21_22.h5`

Two raw HDF5 files from 2026-02-18 are in the `Lickometry Data/` root, not organized into a cohort folder.

---

## Manuscript Status

- eNeuro submission decision received April 2026 (`eNeuro_decision_Apr2026.pdf`)
- `Manuscript Supplemental/` contains Hardware Assembly docx, Parts List xlsx, System Operation docx, Prism file
- All CSV exports and figures tied to this submission
- False positive analysis is additional validation work (post-submission or for revision)

---

## Environment & Dependencies

| Tool | Version |
|---|---|
| Python | 3.13 |
| solara | 1.57.3 |
| jupyterlab | 4.5.6 |
| h5py | 3.13.0 |
| numpy | 2.3.5 |
| scipy | 1.16.3 |
| pandas | 2.3.3 |
| statsmodels | 0.14.6 |
| matplotlib | 3.10.8 |
| panel | 1.8.10 |
| bokeh | (panel dependency) |
| pyftdi | 0.56.0 |

Dev: pyenv-virtualenv (`cliqr` env), macOS. Deploy: Miniforge/conda, Windows.

---

## Legacy / Deprecated

| File | Status |
|---|---|
| `DataRecording.ipynb` | Deprecated — replaced by `recording_gui.py` |
| `MPR121_DataAnalysis.ipynb` | Earlier single-file analysis (ipywidgets). Superseded by `DataAnalysis.ipynb`. |
| `DataAnalysis_old.ipynb` | Archive |
| `lickDetector.m`, `lickDetector_modified.m`, `lickDetector_old.m` | Original MATLAB scripts. Reference only. |
| `filtered_data.mat` | MATLAB output artifact |
| `ftconf.py`, `dump_ftdi_eeprom.py` | Hardware utility scripts (one-off use) |

---

## Known Issues / Technical Debt

1. **Sensor-board mapping duplicated** — `utils/state.py:SERIAL_NUMBER_SENSOR_MAP` and `data_analysis.py:114–121` both hardcode the same mapping. Change one → change both.
2. **`hilbert_algorithm()` filter passes** — `data_analysis.py:276` TODO: currently applies bandpass 7× total (1 + 6 extra passes), likely excessive.
3. **`compute_bout_structure()` param inconsistency** — function signature defaults `ibi_threshold=0.25, min_licks=3`; notebook call sites pass `ibi_threshold=1.0, min_licks=2`.
4. **ML experiments not integrated** — `checkpoints/best.pt` and `Training Data/*.pt` exist but no training script or inference path is in the repo.
5. **Two raw HDF5 files unorganized** — `Lickometry Data/raw_data_2026-02-18_*.h5` not in a cohort folder and not analyzed.
