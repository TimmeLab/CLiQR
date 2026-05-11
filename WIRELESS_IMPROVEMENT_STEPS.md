# CLiQR Wireless Upgrade — Project Plan

**Goal:** Replace wired FT232H→MPR121 chain with one ESP32+MPR121 unit per cage that records locally to SD card and transfers data over BLE.

**How to use this document:** Each task has a clear prerequisite list, and a single concrete deliverable. Complete tasks in order within each phase. Tasks marked `[PARALLEL]` can be done simultaneously by different people.

To be clear, I generated this document with Claude by having it review the current GitHub repo and suggest a pathway to make things wireless while utilizing the current GUI. I am not familiar with using the BLE protocol, but I read up on it a little before reviewing this document and everything here seems reasonable. If you have any questions, or anything doesn't make sense, please email or message me on Teams (Christopher Parker, parkecp@ucmail.uc.edu).

---

## Architecture Change Summary

This section includes a description of how we envision the system working after the upgrade. Essentially, we would like to be able to continue using the same GUI, so everything should fit into that framework.

### Current (Wired)
```
MPR121 (shared, 6 channels) ──I2C──> FT232H ──USB──> PC (real-time stream to HDF5)
```
- 4 boards × 6 sensors = 24 sensors
- One MPR121 serves 6 cages via 6 channels
- Real-time data stream, ~56 Hz


### New (Wireless)
```
MPR121 (1 per cage) ──I2C──> ESP32 ──SD card (local buffer)
                                  └──BLE──> PC (batch download after session)
```
- 24 independent units, one per cage
- Each ESP32 records its own sensor to SD card during the session
- PC cycles through all 24 in batches of ≤7 to download data (Windows BLE limit)
- Downloaded data is converted to the same HDF5 format the existing analysis pipeline already expects

### Test Button Requirement

Each sensor card in the recording GUI has a TEST button. In wireless mode this must still work:

1. User clicks TEST on sensor card N
2. GUI connects to `CLiQR-N` over BLE
3. Downloads the last ~5 seconds of data from the ESP32's **in-memory ring buffer** (not from SD — must be fast)
4. GUI displays the data in the existing plot dialog
5. GUI disconnects

This requires the ESP32 to maintain a circular RAM buffer of recent samples at all times, readable over BLE even during active recording.

### Key Constraints
| Constraint | Detail |
|---|---|
| Windows BLE concurrent connections | Max 7 simultaneous |
| Battery life | ≥10 hours required; >24 hours preferred |
| Data format compatibility | Output HDF5 must match existing `raw_data_*.h5` schema |
| Sampling rate | Must match or exceed current ~56 Hz |
| Per-cage footprint | Must fit on/inside redesigned sipper holder |

---

## Phase 0 — Research & Decisions

**Goal:** Choose hardware before writing any code or designing any PCB.

---

### Task 0.1 — Evaluate ESP32 Development Boards

**Prerequisites:** None

**What to do:**
The goal is to find the best **off-the-shelf development board** for the initial prototype. A custom PCB is a long-term goal but is out of scope here. Evaluate at least these two candidates. Look up datasheets, community support, and purchase price. If you find a board other than an ESP32 that fulfills the other requirements and you believe it is a better fit, we are open to that.

| Candidate | Why consider it |
|---|---|
| Seeed XIAO ESP32-S3 | Very small footprint (~21×17mm), built-in LiPo charging, BLE 5.0, MicroSD slot on some variants |
| Adafruit Feather ESP32-S3 | Larger but well-documented, LiPo connector + charging, BLE 5.0, MicroSD FeatherWing available |

Evaluation criteria:
- Physical size (must fit in cage-side enclosure)
- Has LiPo battery connector + onboard charging?
- BLE 5.0 support?
- I2C pins available?
- SPI pins available for SD card? (or is there a companion SD breakout/FeatherWing?)
- Price per unit
- Quality of software library support?

**Deliverable:** A short written comparison table with a recommended board and justification. Share with Christopher for approval before ordering.

---

### Task 0.2 — Select Battery

**Prerequisites:** Task 0.1 (need board dimensions and available connectors to bound battery size and connection)

**What to do:**
Select a single-cell 3.7V LiPo battery that:
- Fits within the cage enclosure alongside the chosen board
- Provides at least 10 hours, but preferably ≥24 hours, runtime given expected current draw

**Estimate current draw:**
You'll need to work out a rough current draw figure to determine what size we need given an ESP32 board with various components attached. It doesn't need to be exact, especially at this early stage. The battery needs to have a compatible connector, so be sure to check what pins are available on the ESP32.

**Deliverable:** Battery part number, capacity, dimensions, supplier link, and runtime calculation in writing.

---

### Task 0.3 — Research Windows BLE in Python

**Prerequisites:** None `[PARALLEL with 0.1]`

**What to do:**
Determine which Python library to use for Bluetooth Low Energy (BLE) communication with the boards. I'd suggest that you start by evaluating the `bleak` Python library (cross-platform BLE client) for use on Windows:

1. Install `bleak` in the `cliqr` environment: `pip install bleak`
2. Confirm it can scan for BLE devices on Windows
3. Confirm it can connect to and read from a BLE peripheral (use any BLE device you have — a phone, a smartwatch, anything)
4. Confirm that connecting to 7 devices simultaneously is possible (test with as many as you have available; document the limit)
5. Note the GATT concepts needed: Service UUID, Characteristic UUID, read/write/notify

Write a short summary (half page) of findings including any troubleshooting you needed to do to get the connections to work on one of the lickometry computers with our Python environment.

**Deliverable:** Written summary + a small Python script (`ble_test.py`) that demonstrates scanning for and connecting to a BLE device using `bleak`. The script must be tested and working on one of the lickometry computers.

---

### Task 0.4 — Design BLE Data Protocol

This was generated by Claude, and I'm not actually familiar with the BLE protocol. This should serve as a good starting reference for what we need. The suggested command codes are reasonable, and everything reads as sensible to me.

**Prerequisites:** Tasks 0.1, 0.3

**What to do:**
Define the BLE GATT service layout that the ESP32 firmware (Phase 1) and PC software (Phase 2) will both implement. This document is a contract between the firmware and software teams.

We will need the service to have the following "Characteristics" (capabilities) at least:


| Characteristic | UUID | Properties | Format | Purpose |
|---|---|---|---|---|
| Control | (generate) | Write | 1 byte command + up to 8 byte payload | Commands: start, stop, sync clock, request data chunk, request recent data |
| Status | (generate) | Read + Notify | 2 bytes state + 4 bytes sample count + 1 byte battery % | Device status |
| Data | (generate) | Read + Indicate | 128-byte packet | Both bulk session transfer and recent-data (Test button) transfer |

3. Command codes for the Control characteristic:
   - `0x01` — Start recording (payload: 8-byte Unix timestamp µs for clock sync)
   - `0x02` — Stop recording (no payload)
   - `0x03` — Sync clock (payload: 8-byte Unix timestamp µs; can be sent mid-session)
   - `0x04` — Request session data chunk (payload: uint32 byte offset + uint16 length)
   - `0x05` — Request recent data (payload: uint16 number of samples requested, max 640)

4. Data packet format on SD card: define byte layout for each sample record (timestamp + capacitance value). Suggested: `[uint64 timestamp_us | uint16 capacitance]` = 10 bytes per record.
5. Session data transfer protocol: how does the PC request chunks (0x04) and how does the ESP32 respond via the Data characteristic?
6. Recent data protocol: on receiving `0x05`, ESP32 reads from in-memory ring buffer and sends samples in one or more Data characteristic indications. Specify max samples per indication based on BLE MTU (expect 20–244 bytes depending on negotiated MTU).

**Deliverable:** A written protocol spec, defines the exact command codes and UUIDs for communication between the desktop and ESP32s and the byte layouts for each payload.

---

## Phase 1 — ESP32 Firmware

**Goal:** Working firmware on a single ESP32 unit that reads MPR121, saves to SD card, and responds to BLE commands.

**Development environment:** Arduino IDE or PlatformIO. Target board: the one selected in Task 0.1.

---

### Task 1.1 — Set Up ESP32 Dev Environment

**Prerequisites:** Task 0.1 (board selected), Task 0.4 (protocol defined)

**What to do:**

1. Install PlatformIO or Arduino IDE

2. Create a new project targeting the selected board

3. Confirm you can compile and upload a "Hello World" sketch that prints to serial

4. Confirm the board shows up in Device Manager on Windows when connected via USB

**Deliverable:** A project folder committed to a new git branch (`wireless-firmware`) in the CLiQR repo. The project compiles and uploads successfully.

---

### Task 1.2 — MPR121 on ESP32 via I2C

**Prerequisites:** Task 1.1, physical MPR121 board wired to ESP32 (SDA, SCL, 3.3V, GND)

**What to do:**

1. Add the Adafruit MPR121 Arduino library to the project

2. Write code that:
   - Initializes the MPR121 at I2C address 0x5A (or auto-scans 0x5A–0x5D). Since we will only have one MPR121 per board, we don't need to worry about bridging the addr pins to set alternative addresses.
   - Reads the raw capacitance value from **channel 0** (just the single sipper contact)
   - Prints the raw value to Serial at ~10 Hz to verify it changes when you touch the sensor

3. Confirm the value drops noticeably when you touch the electrode

**Note:** Unlike the current system which reads 6 channels per MPR121, each new unit only needs **2 channels** (one cage, one sipper).

**Deliverable:** Code that reads and prints MPR121 values to ESP32 serial console. Demonstrate that the value responds to touch.

---

### Task 1.3 — Achieve Target Sample Rate

**Prerequisites:** Task 1.2

**What to do:**
The existing system samples at ~56 Hz. The ESP32 must match or exceed this.

**Deliverable:** Code that demonstrates sampling rate from a single sensor. Create simple script to record 1000 samples with timestamps and compute average sampling rate.

---

### Task 1.4 — SD Card Data Logging

**Prerequisites:** Task 1.3, SD card module wired to ESP32 SPI pins

**What to do:**

1. Add the SD library (built into Arduino) to the project
2. On boot, create a new file on the SD card named `session_NNNNNN.bin` where `NNNNNN` is a zero-padded counter stored in a small config file
3. Write each sample as a binary record: `[uint64 timestamp_us | uint16 capacitance]` (10 bytes per sample)
4. Buffer writes: accumulate 100 records in RAM, then flush to SD in one write call (reduces wear and write latency)
5. On session stop command, close the file properly (flush + close). **Do not delete the file** — SD cards are cleared manually between experiments.
6. Verify: read the file back on a PC and confirm the records are correct

**At 64 Hz × 10 bytes × 7200 seconds (2 hours) = 4.6 MB per session.** A 1 GB SD card holds ~200 sessions.

**Deliverable:** Code that writes binary session files. A Python script (`verify_sd_data.py`) that reads a `.bin` file and prints the first 20 records.

---

### Task 1.5 — BLE GATT Server Setup

**Prerequisites:** Task 1.1, Task 0.4 (protocol spec)

**What to do `[PARALLEL with 1.4]`:**

1. Add the NimBLE-Arduino library (better performance than the default ESP32 BLE library, according to Claude)
2. Set up a GATT server with the UUIDs defined in Task 0.4
3. Implement the **Status** characteristic: returns current state (idle/recording), sample count, battery voltage
4. Implement the **Control** characteristic: accepts write commands; for now just print received bytes to Serial
5. Set BLE device name to `CLiQR-XX` where `XX` is a two-digit sensor number stored in flash (e.g., `CLiQR-01` through `CLiQR-24`)
6. Confirm the device appears when scanning with the `bleak` test script from Task 0.3

**Deliverable:** ESP32 advertises as a BLE peripheral and the Status characteristic can be read from a Python script.

---

### Task 1.6 — BLE Control Commands (Start / Stop / Clock Sync)

**Prerequisites:** Tasks 1.4, 1.5

**What to do:**
Implement the Control characteristic handler to respond to commands defined in Task 0.4:

1. **Start command (0x01) + 8-byte Unix timestamp payload:**
   - Record the received Unix timestamp and current `micros()` value together to establish a wall-clock offset
   - Set internal state to "recording"
   - Begin writing samples to SD card

2. **Stop command (0x02):**
   - Flush and close the SD card file
   - Set state to "idle"

3. **Sync clock command (0x03) + 8-byte timestamp:**
   - Update the stored clock offset (allows re-sync mid-session if needed). I'm not entirely sure this is necessary, but it wouldn't hurt to confirm whether it's helpful to re-sync mid run.

Test by writing a small Python script that sends each command over BLE and reads the Status characteristic back to confirm state changed.

**Deliverable:** Python test script that sends start/stop/sync commands and confirms state transitions.

---

### Task 1.7 — BLE Data Transfer

**Prerequisites:** Tasks 1.4, 1.5

**What to do:**
Implement bulk data download over BLE. The Data characteristic sends the SD card file contents in chunks.

Protocol (as defined in Task 0.4):

1. PC writes a request to the Control characteristic: `0x04` (request data) + `uint32` byte offset + `uint16` length (max 128 bytes)
2. ESP32 reads the requested bytes from SD and writes them to the Data characteristic (indicate, so PC gets a callback)
3. PC requests successive chunks until it has the whole file

Write a Python test script that downloads a complete session file from the ESP32 and saves it to disk, then verify it matches the original file using a checksum.

**Deliverable:** Python script `ble_download_single.py` that downloads a session file from one ESP32. Checksum verification pass.

---

### Task 1.8 — Battery Level Reporting

**Prerequisites:** Task 1.5

**What to do `[PARALLEL with 1.6, 1.7]`:**

1. Read the LiPo voltage via the ESP32 ADC pin connected to a voltage divider on the battery terminal (check your specific board's schematic — many dev boards have this built in)
2. Convert raw ADC reading to voltage, then to approximate state-of-charge percentage (0–100%) using a simple lookup table for LiPo discharge curve
3. Include this percentage in the Status characteristic response
4. Add a low-battery warning: if battery < 15%, blink an LED (if present) and set a flag in the Status characteristic

**Deliverable:** Battery percentage appears correctly in the Status characteristic. Verify by measuring battery voltage with a multimeter and confirming the reported percentage is reasonable.

---

### Task 1.9 — RAM Ring Buffer for Test Button Support

**Prerequisites:** Task 1.3 `[PARALLEL with 1.4 and 1.5]`

**What to do:**
The GUI TEST button requires the ESP32 to serve the last ~5–10 seconds of sensor data over BLE on demand, without reading from the SD card (which is too slow). Implement an in-memory circular buffer that always holds recent samples:

1. Allocate a circular buffer of **1000 samples** (should be ~5 seconds, we need to know the sampling rate to know exactly how many samples this should be) as a global array in RAM:
   
   ```c
   #define RING_BUF_SIZE 1000
   struct Sample { uint64_t timestamp_us; uint16_t capacitance; };
   Sample ring_buf[RING_BUF_SIZE];
   uint32_t ring_head = 0;  // index of next write position
   ```

2. Every time a new sample is taken (in the sampling loop from Task 1.3), write it into `ring_buf[ring_head % RING_BUF_SIZE]` and increment `ring_head`
3. The ring buffer fills continuously regardless of whether a session is recording. This means TEST button works even before recording starts (useful for checking sensor contact)
4. Implement the `0x05` (request recent data) Control command handler:
   - Read the requested number of samples (max 640) from the ring buffer starting from the oldest available
   - Send them in chunks via the Data characteristic (same format as session data: `uint64 timestamp_us | uint16 capacitance` = 10 bytes/sample)
   - 500 samples × 10 bytes = 5 KB — at 20-byte BLE packets this is ~250 indications, but at negotiated 244-byte MTU it's ~20 packets. Expect ~1–3 seconds transfer time.
5. Verify: use a Python script to request recent data mid-recording and confirm you get back the correct sample count with plausible values

**Memory footprint:** 500 × 10 bytes = 5 KB. The ESP32-S3 has 512 KB RAM — this should be fine.

**Deliverable:** Ring buffer implemented, `0x05` command handler working. Python test script demonstrates retrieving 5 seconds of recent data on demand.

---

### Task 1.10 — Integrate All Firmware Components + Power Measurement

**Prerequisites:** Tasks 1.3, 1.4, 1.6, 1.7, 1.8, 1.9

**What to do:**
Combine all firmware components into a single unified sketch and verify the complete flow:

1. Boot → BLE advertise → ring buffer fills continuously
2. Receive start command → also begin writing to SD card
3. While recording: ring buffer continues filling; SD file receives all samples; TEST requests (0x05) served from RAM
4. Receive stop command → flush and close SD file; ring buffer keeps running
5. Respond to full session download requests (0x04)
6. Return to idle/advertise state (ring buffer still running for future TEST requests)

Then measure actual current draw in each state:
- Idle (BLE advertising, not recording)
- Recording (BLE advertising + SD writes)
- Transferring data (BLE connected, reading SD)

Use a bench power supply with current measurement or a USB power meter, this will need to be ordered as the lab doesn't currently own one. Record results and calculate expected battery life with the selected battery (Task 0.2).

**Deliverable:** Unified firmware, current draw measurements documented, calculated battery life vs. requirement (>10 hours).

---

## Phase 2 — PC Software

**Goal:** Python code (integrated with the existing recording GUI) that manages BLE connections to all 24 devices, orchestrates sessions, and produces HDF5 files compatible with the existing analysis pipeline.

All new code goes in a new directory: `wireless/`. This only exists in the `wireless_upgrade` branch of the Git repo, currently.

---

### Task 2.1 — BLE Device Scanner

**Prerequisites:** Task 0.3, Task 1.5 (need a real ESP32 advertising)

**What to do:**
Create `wireless/ble_scanner.py`:

1. Use `bleak` to scan for BLE devices whose name matches `CLiQR-*`
2. Return a list of discovered devices: `{name: "CLiQR-01", address: "XX:XX:XX:XX:XX:XX"}`
3. Accept a configurable scan duration (default 5 seconds)
4. Sort results by sensor number extracted from name

```python
# Target usage:
from wireless.ble_scanner import scan_cliqr_devices
devices = await scan_cliqr_devices(timeout=5.0)
# Returns: [{"name": "CLiQR-01", "address": "..."}, ...]
```

**Deliverable:** `wireless/ble_scanner.py` with a test script that prints discovered CLiQR devices. Works on the lab Windows PC. This should use asynchronous programming so that we aren't blocking the thread for each device in turn (we can concurrently connect to multiple).

---

### Task 2.2 — Single-Device Command Interface

**Prerequisites:** Tasks 2.1, 1.6

**What to do:**
Create `wireless/ble_device.py` with a `CLiQRDevice` class (just a suggestion for included function signatures):

```python
class CLiQRDevice:
    async def connect(self): ...
    async def disconnect(self): ...
    async def send_start(self, unix_timestamp: float): ...
    async def send_stop(self): ...
    async def send_clock_sync(self, unix_timestamp: float): ...
    async def read_status(self) -> dict: ...
    # Returns: {"state": "recording", "sample_count": 12345, "battery_pct": 82}
    async def read_recent_data(self, n_samples: int = 320) -> tuple[np.ndarray, np.ndarray]: ...
    # Returns: (cap_data array, time_data array) — for the Test button plot
```

Write a test script that:

1. Scans for a single CLiQR device
2. Connects to it
3. Sends start, waits 5 seconds, sends stop
4. Reads status before and after to confirm state changes
5. Disconnects

**Deliverable:** `wireless/ble_device.py` + test script.

---

### Task 2.3 — Single-Device Data Download

**Prerequisites:** Tasks 2.2, 1.7

**What to do:**
Add a `download_session` method to `CLiQRDevice`:

```python
async def download_session(self, output_path: str, progress_callback=None) -> int:
    """Download session file from ESP32 SD card. Returns bytes downloaded."""
    ...
```

The method should:

1. Query the Status characteristic to get total session file size
2. Request chunks in a loop until the whole file is received
3. Write received bytes to `output_path` (a `.bin` temp file)
4. Call `progress_callback(bytes_received, total_bytes)` each iteration (for progress display)
5. After full download, send the acknowledge-complete command

Write a test script that downloads a real session file and verifies its checksum against a reference.

**Deliverable:** `CLiQRDevice.download_session()` implemented and tested. Transfer rate documented (expect ~10–50 KB/s over BLE). If the transfer takes too long, we may just go with manually transferring from SD cards. But either way, we need to test this to figure that out.

---

### Task 2.4 — Convert Downloaded Binary to HDF5

**Prerequisites:** Tasks 2.3, Task 0.4 (binary format spec)

**What to do:**
Create `wireless/convert_to_hdf5.py`:

```python
def convert_session_to_hdf5(
    bin_file: str,
    hdf5_file: str,
    sensor_id: int,
    serial_number: str,
    clock_offset_s: float,
):
    """
    Read a CLiQR .bin session file and write it into an HDF5 group that matches
    the existing raw_data_*.h5 schema:
      sensor_{sensor_id}/cap_data
      sensor_{sensor_id}/time_data
      sensor_{sensor_id}/start_time
      sensor_{sensor_id}/stop_time
    clock_offset_s is the offset added to convert ESP32 micros() timestamps
    to Unix wall-clock time (established during the start command sync).
    """
    ...
```

The format of the HDF5 output will actually be slightly more simple than the current format. Instead of saving `board_{serial_number}` as the first group, we can skip that since each sensor is on its own board. Verify by running `DataAnalysis.ipynb` on the converted HDF5 file and confirming lick detection produces sensible results.

**Deliverable:** `wireless/convert_to_hdf5.py`. A test converting a real downloaded session file into HDF5 and verifying it passes through the `filter_data()` function without errors.

---

### Task 2.5 — Batch Session Manager (24 Devices, ≤7 Concurrent)

**Prerequisites:** Tasks 2.2, 2.3

**What to do:**
Create `wireless/session_manager.py` with a `WirelessSessionManager` class that manages the full 24-device workflow in batches of ≤7:

```python
class WirelessSessionManager:
   def start_session(self, device: list[dict], unix_timestamp: float):
        """
        Connect to specified device (no need for simultaneous connections
        just to send start command), send start + clock sync, disconnect.
        This should be called when the user presses the "Start" button for a sensor in the GUI.
        """
        ...

   def stop_session(self, device: list[dict]):
        """Connect to specified device, send stop command, disconnect."""
        ...

   async def download_all(
        self,
        devices: list[dict],
        output_dir: str,
        progress_callback=None,
    ) -> list[str]:
        """
        Download session data from all devices in batches of <=7 simultaneous
        connections. Returns list of downloaded .bin file paths.
        """
        ...
```

Key detail for `download_all`: connect to min(7, remaining) devices simultaneously, await all downloads in parallel using `asyncio.gather()`, then move to the next batch. This could be made more efficient by always connecting to the next device when a device connection is dropped, but doing it in batches will be much more streamlined and shouldn't impact performance unreasonably.

Write a test that simulates the full flow with however many ESP32 units are available (even just 1–2 is enough to validate the logic).

**Deliverable:** `wireless/session_manager.py`. Test script demonstrating batch download with real hardware.

---

### Task 2.6 — Test Button: BLE Peek for Recent Data

**Prerequisites:** Tasks 2.2, 1.9

**What to do:**
Implement the sensor Test button flow for wireless mode. This is the on-demand snapshot that lets lab personnel verify a sensor is working mid-session without downloading the full recording.

Add `read_recent_data()` to `CLiQRDevice` (stub was added in Task 2.2):

```python
async def read_recent_data(self, n_samples: int = 500) -> tuple:
    """
    Connect, send 0x05 command requesting n_samples from ring buffer,
    collect Data characteristic indications until all samples received,
    disconnect. Returns (cap_data, time_data) as numpy arrays.
    n_samples will need to be verified with the actual sampling rate to give ~5 seconds.
    """
    ...
```

Then create `wireless/ble_peek.py` with a convenience wrapper:

```python
async def test_sensor(sensor_id: int, devices: list[dict], n_samples: int = 500):
    """
    Find the CLiQR device for sensor_id, connect, download recent data,
    disconnect. Returns (cap_data, time_data) ready for the plot dialog.
    """
    ...
```

Key requirements:
- Must complete (connect → data → disconnect) in under **5 seconds** so the GUI doesn't feel frozen
- Must work whether or not a session is currently recording
- If the device isn't found or doesn't respond, raise a clear exception that the GUI can display as an error message

Write a test script that calls `test_sensor()` for one device and prints the returned arrays. Confirm the shape is `(n_samples,2)` and the values look like real timestamps and capacitance data.

**Deliverable:** `wireless/ble_peek.py` + test script. Demo showing TEST functionality returns data in under 5 seconds.

---

### Task 2.7 — Integrate Wireless Mode into Recording GUI

**Prerequisites:** Tasks 2.1, 2.4, 2.5.

**What to do:**
Add a "Wireless Mode" to the existing `recording_gui.py`:

1. Add a mode toggle to `utils/state.py`: `wireless_mode = solara.reactive(False)`
2. In `components/hardware_status.py`, add a conditional branch:
   - Wired mode: existing FT232H init (unchanged)
   - Wireless mode: run BLE scan, display discovered CLiQR devices with names and battery levels
3. In `components/session_controls.py`, modify start/stop to:
   - Wired mode: existing behavior (unchanged)
   - Wireless mode: call `WirelessSessionManager.start_session()` / `stop_session()`
4. In `components/sensor_card.py`, modify the TEST button handler:
   - Wired mode: existing behavior, reads from live `SensorRecorder` buffer (unchanged)
   - Wireless mode: call `ble_peek.test_sensor(sensor_id)`, then pass the returned arrays to the existing `TestPlotDialog` — the plot dialog itself does not need to change
5. Add a new "Download Data" button that appears after a wireless session ends, which triggers `WirelessSessionManager.download_all()` followed by `convert_session_to_hdf5()` for each device
6. Show a progress bar during download

The wired mode code path must remain completely unchanged. The `TestPlotDialog` component does not need any modification — only how the data is fetched changes.

**Deliverable:** Updated GUI with wireless mode toggle and working Test button. Demo to supervisor: full wireless session start → TEST button mid-session → stop → download on at least one device.

---

## Phase 3 — Mechanical Design

**Goal:** Redesign the sipper holder to house the ESP32 unit + battery + MPR121 board, while maintaining the same cage-mounting interface.

**Starting point:** Existing `3D Print Files/AllentownSipperHolderRingContact.stl` and `MPR121_mount_with_cutouts.stl`

**Software:** FreeCAD, Fusion 360, or your preferred software

---

### Task 3.1 — Design Electronics Enclosure

**Prerequisites:** Board + battery physically in hand (Tasks 0.1, 0.2)

**What to do:**
Design a housing that:
- Sits on the cage lid using the same interface as the current design
- Contains the ESP32 board + battery stacked (battery underneath or alongside)
- Has a small hole or slot for a USB-C charging cable
- Keeps the MPR121 electrode accessible to the sipper contact ring
- Is printable in PLA or PETG on the lab's 3D printer

**Deliverable:** 3D model file with new enclosure design.

---

### Task 3.2 — 3D Print and Fit-Check Prototype

**Prerequisites:** Task 3.1

**What to do:**

1. Print one prototype of the new enclosure 
2. Test fit with actual ESP32 board, battery, and MPR121
3. Test fit on actual cage

**Deliverable:** Photos of prototype with components installed and mounted on cage.

---

## Phase 4 — Integration Testing

**Goal:** Verify the complete wireless system works reliably before building more units or retiring the wired system.

**Gate:** Task 4.1 (single-unit test) must fully pass — and be approved by PI — before ordering additional hardware or printing more enclosures. Each subsequent task in this phase is a gate for the next. Do not skip ahead.

---

### Task 4.1 — Single-Unit End-to-End Test

**Prerequisites:** Phase 1 complete, Phase 2 Tasks 2.1–2.5, 2.7 complete, Phase 3 complete (at least 1 enclosure printed)

**What to do:**
With one complete wireless unit (ESP32 + MPR121 + SD card + battery in enclosure):

1. Mount on cage with water sipper installed
2. Run a 30-minute session using the GUI wireless mode
3. Mid-session: press the TEST button for this sensor — confirm the plot appears within 5 seconds
4. Simulate licks with finger taps on the sipper tip for a minute or so.
5. Stop session and download data over BLE
6. Run `DataAnalysis.ipynb` on the converted HDF5
7. Confirm lick detection runs

**Deliverable:** Screenshot of lick detection plot. Screenshot of Test button plot mid-session.

---

### Task 4.2 — Battery Life Test

**Prerequisites:** Task 4.1, fully charged battery

**What to do:**

1. Start a recording session on one unit at 8:00 AM
2. Check status periodically (battery percentage via BLE). This can be scripted so that you don't have to manually continue testing until the battery dies (which should hopefully take some hours)
3. Record the time at which the unit stops responding (battery dead)
4. Calculated expected vs. measured runtime

Also test: does the unit resume advertising and remain downloadable after the battery dies and is recharged?

**Deliverable:** Battery life table: time, battery %, sample count at each check. Final runtime.

---

### Task 4.3 — Multi-Unit Batch Download Test

**Prerequisites:** At least 7 wireless units built (**only after Task 4.1 passes and PI approves ordering more units**)

**What to do:**

1. Run a 30-minute session on all available units simultaneously
2. After session: time how long it takes to download all data via the batch manager
3. Verify all downloaded files are complete and pass checksum
4. Convert all to HDF5 and run analysis

Target: download all 24 units in under 15 minutes.

**Deliverable:** Download time log, pass/fail for each device's data integrity check.

---

### Task 4.4 — Full 24-Unit System Test

**Prerequisites:** 24 complete units built (Phase 3 done), Task 4.3 complete

**What to do:**
Run a full experiment-scale test:

1. Install all 24 units on the cage rack
2. Run a 2-hour session with animals (matching typical experiment duration)
3. Download all data
4. Run full analysis pipeline
5. Verify results look similar to historical wired system data

**Deliverable:** Full analysis output (lick counts, correlation plot) for 24 animals. Comparison to wired system historical baseline.

---

### Task 4.5 — Update Documentation

**Prerequisites:** Task 4.4

**What to do:**
Update the following files to reflect the wireless system:

1. `README.md` — add wireless system section (installation, hardware setup, workflow)
2. `Manuscript Supplemental/Hardware Assembly.docx` — add wireless unit assembly instructions
3. `Manuscript Supplemental/Parts List.xlsx` — add wireless BOM

**Deliverable:** Updated documentation files committed to git.