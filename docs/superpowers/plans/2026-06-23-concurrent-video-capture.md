# Concurrent Video Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the CLiQR desktop GUI start a Raspberry Pi 5 / Pi Camera 3 video recording in sync with capacitive recording, bookmarking the exact video frame when a sensor's Start button is clicked.

**Architecture:** Pi 5 runs a TCP server (`pi/pi_camera_server.py`) wrapping picamera2; it pre-rolls the camera at session start and reports the current frame PTS on a BOOKMARK command. The desktop GUI talks to it via a thin TCP client (`hardware/pi_camera.py`), guarded by an optional, non-blocking "camera enabled" flag. A shared pure-Python protocol module (`video/protocol.py`) is used by both ends. The capacitive↔video alignment (frame index + PTS) is written into the existing HDF5 file alongside `start_time`, replacing the post-hoc sipper-step alignment for new recordings.

**Tech Stack:** Python 3.13, stdlib `socket`/`socketserver`/`json`, solara (GUI reactives), h5py (HDF5), picamera2 (Pi only), pytest (new dev dependency).

## Global Constraints

- Python 3.13; standard library only on the desktop side except existing deps (solara, h5py, numpy, pandas). No new desktop runtime dependency beyond `pytest` (dev/test only).
- picamera2 is a **Pi-only** dependency — never imported on the desktop import path. Desktop tests must run with no picamera2 and no physical Pi.
- The camera is **auxiliary and non-blocking**: any camera/network failure must be caught, logged via `state.add_log_message`, and must never interrupt capacitive recording.
- Follow existing module patterns: lazy intra-package imports inside functions (as `sensor_card.start_sensor` already does), `solara.reactive` for state, module-global singletons for stateful managers.
- Timestamps are Unix epoch seconds (`time.time()`) on the desktop; video PTS is the Pi camera `SensorTimestamp` in seconds. Both stored so analysis can cross-reference.
- Frame-offset `.txt` files must remain loadable by `false_positive_analysis.load_frame_offsets()` (one numeric frame timestamp per line, `numpy.loadtxt`-compatible).

---

### Task 1: Test scaffolding + shared protocol module

**Files:**
- Create: `video/__init__.py`
- Create: `video/protocol.py`
- Create: `tests/__init__.py`
- Create: `tests/test_protocol.py`
- Modify: `requirements.txt` (add `pytest`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - Command constants `PING`, `START_SESSION`, `BOOKMARK`, `STOP_SESSION`, `GET_FILE` (all `str`).
  - `encode_message(msg: dict) -> bytes` (newline-terminated UTF-8 JSON).
  - `decode_message(line: bytes) -> dict`.
  - `make_request(cmd: str, **params) -> dict` → `{"cmd": cmd, **params}`.
  - `make_ok(**fields) -> dict` → `{"ok": True, **fields}`.
  - `make_error(message: str) -> dict` → `{"ok": False, "error": message}`.

- [ ] **Step 1: Add pytest to requirements**

Append to `requirements.txt`:

```
pytest==8.3.4
```

- [ ] **Step 2: Create package markers**

Create `video/__init__.py` (empty) and `tests/__init__.py` (empty).

- [ ] **Step 3: Write the failing test**

Create `tests/test_protocol.py`:

```python
from video import protocol


def test_encode_decode_roundtrip():
    msg = {"cmd": protocol.PING, "n": 1}
    line = protocol.encode_message(msg)
    assert line.endswith(b"\n")
    assert protocol.decode_message(line) == msg


def test_decode_tolerates_missing_newline():
    assert protocol.decode_message(b'{"cmd": "PING"}') == {"cmd": "PING"}


def test_make_request():
    assert protocol.make_request(protocol.BOOKMARK, sensor_id=3) == {
        "cmd": "BOOKMARK",
        "sensor_id": 3,
    }


def test_make_ok_and_error():
    assert protocol.make_ok(frame_index=5) == {"ok": True, "frame_index": 5}
    err = protocol.make_error("boom")
    assert err == {"ok": False, "error": "boom"}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'video.protocol'`

- [ ] **Step 5: Implement the protocol module**

Create `video/protocol.py`:

```python
"""Shared wire protocol for the Pi camera server and desktop client.

Pure standard library so it can run unchanged on the Raspberry Pi and the
desktop. Messages are newline-delimited UTF-8 JSON objects.
"""
import json

PING = "PING"
START_SESSION = "START_SESSION"
BOOKMARK = "BOOKMARK"
STOP_SESSION = "STOP_SESSION"
GET_FILE = "GET_FILE"


def encode_message(msg: dict) -> bytes:
    """Serialize a message dict to a newline-terminated UTF-8 JSON frame."""
    return (json.dumps(msg) + "\n").encode("utf-8")


def decode_message(line: bytes) -> dict:
    """Parse one JSON frame; a trailing newline is optional."""
    return json.loads(line.decode("utf-8").rstrip("\n"))


def make_request(cmd: str, **params) -> dict:
    """Build a request message: {"cmd": cmd, ...params}."""
    return {"cmd": cmd, **params}


def make_ok(**fields) -> dict:
    """Build a success response: {"ok": True, ...fields}."""
    return {"ok": True, **fields}


def make_error(message: str) -> dict:
    """Build a failure response: {"ok": False, "error": message}."""
    return {"ok": False, "error": message}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add video/__init__.py video/protocol.py tests/__init__.py tests/test_protocol.py requirements.txt
git commit -m "feat: add shared camera wire protocol + test scaffolding"
```

---

### Task 2: Hardware-independent server core

**Files:**
- Create: `pi/__init__.py`
- Create: `pi/server_core.py`
- Create: `tests/test_server_core.py`

**Interfaces:**
- Consumes: `video.protocol`.
- Produces:
  - `CameraServer(backend)` with method `handle(request: dict) -> dict`.
  - Backend contract (duck-typed, implemented for real in Task 3, faked in tests):
    - property `is_active -> bool`
    - `start_session(name: str) -> str` (returns the video filename)
    - `bookmark(sensor_id) -> dict` with keys `frame_index` (int|None), `pts` (float), `pi_monotonic` (float)
    - `stop_session() -> list[dict]` each `{"name": str, "size": int}`

- [ ] **Step 1: Create package marker**

Create `pi/__init__.py` (empty).

- [ ] **Step 2: Write the failing test**

Create `tests/test_server_core.py`:

```python
import pytest
from pi.server_core import CameraServer
from video import protocol


class FakeBackend:
    def __init__(self):
        self._active = False
        self.bookmarks = 0

    @property
    def is_active(self):
        return self._active

    def start_session(self, name):
        self._active = True
        self.name = name
        return f"{name}.mp4"

    def bookmark(self, sensor_id):
        self.bookmarks += 1
        return {"frame_index": self.bookmarks, "pts": 1.5, "pi_monotonic": 99.0}

    def stop_session(self):
        self._active = False
        return [{"name": f"{self.name}.mp4", "size": 10}]


@pytest.fixture
def server():
    return CameraServer(FakeBackend())


def test_ping(server):
    resp = server.handle(protocol.make_request(protocol.PING))
    assert resp["ok"] is True
    assert resp["status"] == "ready"


def test_start_then_bookmark(server):
    start = server.handle(protocol.make_request(protocol.START_SESSION, name="vid"))
    assert start == {"ok": True, "video_filename": "vid.mp4"}
    mark = server.handle(protocol.make_request(protocol.BOOKMARK, sensor_id=1))
    assert mark["ok"] is True
    assert mark["frame_index"] == 1
    assert mark["pts"] == 1.5


def test_start_requires_name(server):
    resp = server.handle(protocol.make_request(protocol.START_SESSION))
    assert resp["ok"] is False


def test_bookmark_without_session(server):
    resp = server.handle(protocol.make_request(protocol.BOOKMARK, sensor_id=1))
    assert resp == {"ok": False, "error": "no active session"}


def test_stop_without_session(server):
    resp = server.handle(protocol.make_request(protocol.STOP_SESSION))
    assert resp == {"ok": False, "error": "no active session"}


def test_unknown_command(server):
    resp = server.handle({"cmd": "NOPE"})
    assert resp["ok"] is False
    assert "unknown" in resp["error"].lower()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_server_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pi.server_core'`

- [ ] **Step 4: Implement the server core**

Create `pi/server_core.py`:

```python
"""Hardware-independent request dispatcher for the Pi camera server.

Holds no networking and no camera code: it maps decoded protocol requests
to a camera backend and returns response dicts. This keeps the command
logic unit-testable with a fake backend (no picamera2, no hardware).
"""
from video import protocol


class CameraServer:
    """Dispatches protocol requests to a camera backend."""

    def __init__(self, backend):
        self.backend = backend

    def handle(self, request: dict) -> dict:
        cmd = request.get("cmd")
        try:
            if cmd == protocol.PING:
                return protocol.make_ok(status="ready", active=self.backend.is_active)

            if cmd == protocol.START_SESSION:
                name = request.get("name")
                if not name:
                    return protocol.make_error("START_SESSION requires 'name'")
                video_filename = self.backend.start_session(name)
                return protocol.make_ok(video_filename=video_filename)

            if cmd == protocol.BOOKMARK:
                if not self.backend.is_active:
                    return protocol.make_error("no active session")
                mark = self.backend.bookmark(request.get("sensor_id"))
                return protocol.make_ok(**mark)

            if cmd == protocol.STOP_SESSION:
                if not self.backend.is_active:
                    return protocol.make_error("no active session")
                files = self.backend.stop_session()
                return protocol.make_ok(files=files)

            return protocol.make_error(f"unknown command: {cmd}")
        except Exception as exc:  # backend/hardware errors never crash the server
            return protocol.make_error(str(exc))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_server_core.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add pi/__init__.py pi/server_core.py tests/test_server_core.py
git commit -m "feat: add hardware-independent camera server core"
```

---

### Task 3: TCP server + picamera2 backend (Pi side)

**Files:**
- Create: `pi/pi_camera_server.py`
- Create: `pi/camera_backend.py`
- Create: `tests/test_tcp_server.py`

**Interfaces:**
- Consumes: `pi.server_core.CameraServer`, `video.protocol`.
- Produces:
  - `pi/pi_camera_server.py`: `serve(backend, host="0.0.0.0", port=8770) -> socketserver.ThreadingTCPServer` (caller runs `.serve_forever()`); a `__main__` entry that builds a `Picamera2Backend` and serves. The TCP handler reads one newline-delimited request, calls `CameraServer.handle`, writes one response line. For `GET_FILE` it writes a `{"ok": True, "size": N}` line followed by exactly `N` raw bytes of the file.
  - `pi/camera_backend.py`: `Picamera2Backend(output_dir=".")` implementing the Task 2 backend contract.

**Note:** `Picamera2Backend` requires real hardware and is NOT unit-tested here; it is validated manually on the Pi (see Task 11 docs). The TCP framing IS tested using `CameraServer` + a fake backend.

- [ ] **Step 1: Write the failing test (TCP framing, fake backend)**

Create `tests/test_tcp_server.py`:

```python
import socket
import threading
from pathlib import Path

from pi.pi_camera_server import serve
from pi.server_core import CameraServer
from video import protocol


class FakeBackend:
    def __init__(self, tmp_path):
        self._active = False
        self.tmp_path = tmp_path

    @property
    def is_active(self):
        return self._active

    def start_session(self, name):
        self._active = True
        self.video = self.tmp_path / f"{name}.mp4"
        self.video.write_bytes(b"VIDEODATA")
        return self.video.name

    def bookmark(self, sensor_id):
        return {"frame_index": 7, "pts": 2.0, "pi_monotonic": 1.0}

    def stop_session(self):
        self._active = False
        return [{"name": self.video.name, "size": self.video.stat().st_size}]


def _send(sock, msg):
    sock.sendall(protocol.encode_message(msg))
    return protocol.decode_message(_recv_line(sock))


def _recv_line(sock):
    buf = b""
    while not buf.endswith(b"\n"):
        buf += sock.recv(1)
    return buf


def test_tcp_roundtrip(tmp_path):
    backend = FakeBackend(tmp_path)
    server = serve(CameraServer(backend), host="127.0.0.1", port=0)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with socket.create_connection((host, port)) as sock:
            assert _send(sock, protocol.make_request(protocol.PING))["ok"] is True
            start = _send(sock, protocol.make_request(protocol.START_SESSION, name="clip"))
            assert start["video_filename"] == "clip.mp4"
            mark = _send(sock, protocol.make_request(protocol.BOOKMARK, sensor_id=1))
            assert mark["frame_index"] == 7

            # GET_FILE: header line then raw bytes
            sock.sendall(protocol.encode_message(
                protocol.make_request(protocol.GET_FILE, name="clip.mp4")))
            header = protocol.decode_message(_recv_line(sock))
            assert header["ok"] is True
            payload = b""
            while len(payload) < header["size"]:
                payload += sock.recv(4096)
            assert payload == b"VIDEODATA"
    finally:
        server.shutdown()
        server.server_close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tcp_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pi.pi_camera_server'`

- [ ] **Step 3: Implement the TCP server**

Create `pi/pi_camera_server.py`:

```python
"""TCP front-end for the Pi camera server.

Each connection handles a stream of newline-delimited JSON requests. Most
commands return one JSON response line. GET_FILE returns a JSON header line
({"ok": true, "size": N}) followed by exactly N raw file bytes.

Run on the Pi:  python -m pi.pi_camera_server --port 8770 --output-dir ~/clips
"""
import argparse
import socketserver
from pathlib import Path

from pi.server_core import CameraServer
from video import protocol


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        server: "_Server" = self.server  # type: ignore[assignment]
        core = server.core
        for raw in self.rfile:
            if not raw.strip():
                continue
            request = protocol.decode_message(raw)
            if request.get("cmd") == protocol.GET_FILE:
                self._send_file(server.output_dir / request.get("name", ""))
                continue
            response = core.handle(request)
            self.wfile.write(protocol.encode_message(response))
            self.wfile.flush()

    def _send_file(self, path: Path):
        if not path.is_file():
            self.wfile.write(protocol.encode_message(
                protocol.make_error(f"no such file: {path.name}")))
            self.wfile.flush()
            return
        size = path.stat().st_size
        self.wfile.write(protocol.encode_message(protocol.make_ok(size=size)))
        with open(path, "rb") as fh:
            while chunk := fh.read(65536):
                self.wfile.write(chunk)
        self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, core: CameraServer, output_dir: Path):
        self.core = core
        self.output_dir = output_dir
        super().__init__(server_address, _Handler)


def serve(core: CameraServer, host: str = "0.0.0.0", port: int = 8770,
          output_dir: str = ".") -> _Server:
    """Create (but do not run) a threaded TCP server bound to host:port."""
    return _Server((host, port), core, Path(output_dir))


def main():
    parser = argparse.ArgumentParser(description="CLiQR Pi camera server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    from pi.camera_backend import Picamera2Backend
    core = CameraServer(Picamera2Backend(output_dir=args.output_dir))
    server = serve(core, host=args.host, port=args.port, output_dir=args.output_dir)
    print(f"CLiQR camera server listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tcp_server.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Implement the picamera2 backend (Pi hardware; not unit-tested)**

Create `pi/camera_backend.py`:

```python
"""picamera2 backend for the Pi camera server (Raspberry Pi 5 + Camera 3).

Pre-rolls the camera at session start, logging every frame's SensorTimestamp
to a `.txt` file (one nanosecond timestamp per line, numpy.loadtxt-compatible).
BOOKMARK reports the live frame count and current PTS so the desktop can map a
sensor's Start click to a video frame.

picamera2 is imported lazily so this module only loads on the Pi.
"""
import time
from pathlib import Path


class Picamera2Backend:
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._active = False
        self._picam2 = None
        self._encoder = None
        self._pts_fh = None
        self._pts_path = None
        self._video_path = None
        self._frame_count = 0

    @property
    def is_active(self) -> bool:
        return self._active

    def start_session(self, name: str) -> str:
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FfmpegOutput

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration()
        self._picam2.configure(config)

        self._video_path = self.output_dir / f"{name}.mp4"
        self._pts_path = self.output_dir / f"{name}.txt"
        self._pts_fh = open(self._pts_path, "w")
        self._frame_count = 0
        self._picam2.pre_callback = self._on_frame

        self._encoder = H264Encoder()
        self._picam2.start_recording(self._encoder, FfmpegOutput(str(self._video_path)))
        self._active = True
        return self._video_path.name

    def _on_frame(self, request):
        metadata = request.get_metadata()
        timestamp = metadata.get("SensorTimestamp", 0)
        self._pts_fh.write(f"{timestamp}\n")
        self._frame_count += 1

    def bookmark(self, sensor_id) -> dict:
        metadata = self._picam2.capture_metadata()
        pts = metadata.get("SensorTimestamp", 0) / 1e9
        return {
            "frame_index": self._frame_count,
            "pts": pts,
            "pi_monotonic": time.monotonic(),
        }

    def stop_session(self) -> list:
        self._picam2.stop_recording()
        self._pts_fh.close()
        self._active = False
        files = []
        for path in (self._video_path, self._pts_path):
            files.append({"name": path.name, "size": path.stat().st_size})
        return files
```

- [ ] **Step 6: Commit**

```bash
git add pi/pi_camera_server.py pi/camera_backend.py tests/test_tcp_server.py
git commit -m "feat: add TCP server and picamera2 backend for Pi camera"
```

---

### Task 4: Desktop TCP client

**Files:**
- Create: `hardware/pi_camera.py`
- Create: `tests/test_pi_camera_client.py`

**Interfaces:**
- Consumes: `video.protocol`. Tested against a real loopback `serve(CameraServer(FakeBackend))` from Task 3.
- Produces:
  - `PiCameraClient(host: str, port: int, timeout: float = 5.0)` with:
    - `ping() -> bool`
    - `start_session(name: str) -> dict` (response dict; `{"ok": False, "error": ...}` on failure)
    - `bookmark(sensor_id) -> dict`
    - `stop_session() -> dict`
    - `fetch_files(names: list[str], dest_dir: str) -> list[Path]` (downloaded paths)

- [ ] **Step 1: Write the failing test**

Create `tests/test_pi_camera_client.py`:

```python
import threading
from pathlib import Path

import pytest

from hardware.pi_camera import PiCameraClient
from pi.pi_camera_server import serve
from pi.server_core import CameraServer


class FakeBackend:
    def __init__(self, tmp_path):
        self._active = False
        self.tmp_path = tmp_path

    @property
    def is_active(self):
        return self._active

    def start_session(self, name):
        self._active = True
        self.video = self.tmp_path / f"{name}.mp4"
        self.video.write_bytes(b"VID")
        self.txt = self.tmp_path / f"{name}.txt"
        self.txt.write_text("100\n200\n")
        return self.video.name

    def bookmark(self, sensor_id):
        return {"frame_index": 3, "pts": 4.5, "pi_monotonic": 1.0}

    def stop_session(self):
        self._active = False
        return [
            {"name": self.video.name, "size": self.video.stat().st_size},
            {"name": self.txt.name, "size": self.txt.stat().st_size},
        ]


@pytest.fixture
def client(tmp_path):
    backend = FakeBackend(tmp_path)
    server = serve(CameraServer(backend), host="127.0.0.1", port=0,
                   output_dir=str(tmp_path))
    host, port = server.server_address
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield PiCameraClient(host, port)
    server.shutdown()
    server.server_close()


def test_ping(client):
    assert client.ping() is True


def test_full_flow_and_fetch(client, tmp_path):
    start = client.start_session("clip")
    assert start["ok"] is True
    assert start["video_filename"] == "clip.mp4"

    mark = client.bookmark(1)
    assert mark["frame_index"] == 3
    assert mark["pts"] == 4.5

    stop = client.stop_session()
    names = [f["name"] for f in stop["files"]]

    dest = tmp_path / "desktop"
    fetched = client.fetch_files(names, str(dest))
    assert {p.name for p in fetched} == {"clip.mp4", "clip.txt"}
    assert (dest / "clip.mp4").read_bytes() == b"VID"


def test_ping_unreachable():
    # Nothing listening on this port -> graceful False, no exception.
    assert PiCameraClient("127.0.0.1", 1, timeout=0.2).ping() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pi_camera_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hardware.pi_camera'`

- [ ] **Step 3: Implement the client**

Create `hardware/pi_camera.py`:

```python
"""Desktop-side TCP client for the Pi camera server.

Every method is best-effort: connection or protocol failures are caught and
returned as error dicts (or False for ping). The camera is auxiliary and must
never raise into the recording loop.
"""
import socket
from pathlib import Path

from video import protocol


class PiCameraClient:
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _request(self, msg: dict) -> dict:
        """Send one request, return the decoded response dict (or error dict)."""
        try:
            with socket.create_connection((self.host, self.port), self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(protocol.encode_message(msg))
                return protocol.decode_message(self._recv_line(sock))
        except Exception as exc:
            return protocol.make_error(str(exc))

    @staticmethod
    def _recv_line(sock: socket.socket) -> bytes:
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        return buf

    def ping(self) -> bool:
        return self._request(protocol.make_request(protocol.PING)).get("ok", False)

    def start_session(self, name: str) -> dict:
        return self._request(protocol.make_request(protocol.START_SESSION, name=name))

    def bookmark(self, sensor_id) -> dict:
        return self._request(protocol.make_request(protocol.BOOKMARK, sensor_id=sensor_id))

    def stop_session(self) -> dict:
        return self._request(protocol.make_request(protocol.STOP_SESSION))

    def fetch_files(self, names, dest_dir: str) -> list:
        """Download each named file via GET_FILE into dest_dir. Returns paths."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        try:
            with socket.create_connection((self.host, self.port), self.timeout) as sock:
                sock.settimeout(self.timeout)
                for name in names:
                    sock.sendall(protocol.encode_message(
                        protocol.make_request(protocol.GET_FILE, name=name)))
                    header = protocol.decode_message(self._recv_line(sock))
                    if not header.get("ok"):
                        continue
                    saved.append(self._recv_file(sock, dest / name, header["size"]))
        except Exception:
            return saved
        return [p for p in saved if p is not None]

    @staticmethod
    def _recv_file(sock: socket.socket, path: Path, size: int):
        received = 0
        with open(path, "wb") as fh:
            while received < size:
                chunk = sock.recv(min(65536, size - received))
                if not chunk:
                    return None
                fh.write(chunk)
                received += len(chunk)
        return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pi_camera_client.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add hardware/pi_camera.py tests/test_pi_camera_client.py
git commit -m "feat: add desktop TCP client for Pi camera"
```

---

### Task 5: Mock client for no-hardware GUI + tests

**Files:**
- Create: `hardware/pi_camera_mock.py`
- Create: `tests/test_pi_camera_mock.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `MockPiCameraClient()` with the same method surface as `PiCameraClient`
    (`ping`, `start_session`, `bookmark`, `stop_session`, `fetch_files`).
    Deterministic: `ping()` → True; `bookmark()` returns an incrementing
    `frame_index` starting at 1 and `pts == frame_index / 30.0`; `fetch_files`
    creates empty placeholder files in `dest_dir`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pi_camera_mock.py`:

```python
from hardware.pi_camera_mock import MockPiCameraClient


def test_mock_flow(tmp_path):
    client = MockPiCameraClient()
    assert client.ping() is True

    start = client.start_session("clip")
    assert start["ok"] is True
    assert start["video_filename"] == "clip.mp4"

    first = client.bookmark(1)
    second = client.bookmark(1)
    assert first["frame_index"] == 1
    assert second["frame_index"] == 2
    assert second["pts"] == 2 / 30.0

    stop = client.stop_session()
    names = [f["name"] for f in stop["files"]]
    fetched = client.fetch_files(names, str(tmp_path))
    assert {p.name for p in fetched} == {"clip.mp4", "clip.txt"}
    assert (tmp_path / "clip.mp4").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pi_camera_mock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hardware.pi_camera_mock'`

- [ ] **Step 3: Implement the mock**

Create `hardware/pi_camera_mock.py`:

```python
"""In-memory mock of PiCameraClient for mock-mode GUI and tests.

No sockets, no Pi. Returns deterministic, monotonically increasing frame
indices so behavior is assertable.
"""
import time
from pathlib import Path


class MockPiCameraClient:
    def __init__(self):
        self._active = False
        self._frame_count = 0
        self._name = None

    def ping(self) -> bool:
        return True

    def start_session(self, name: str) -> dict:
        self._active = True
        self._frame_count = 0
        self._name = name
        return {"ok": True, "video_filename": f"{name}.mp4"}

    def bookmark(self, sensor_id) -> dict:
        if not self._active:
            return {"ok": False, "error": "no active session"}
        self._frame_count += 1
        return {
            "ok": True,
            "frame_index": self._frame_count,
            "pts": self._frame_count / 30.0,
            "pi_monotonic": time.monotonic(),
        }

    def stop_session(self) -> dict:
        self._active = False
        return {"ok": True, "files": [
            {"name": f"{self._name}.mp4", "size": 0},
            {"name": f"{self._name}.txt", "size": 0},
        ]}

    def fetch_files(self, names, dest_dir: str) -> list:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        for name in names:
            path = dest / name
            path.touch()
            saved.append(path)
        return saved
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pi_camera_mock.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add hardware/pi_camera_mock.py tests/test_pi_camera_mock.py
git commit -m "feat: add mock Pi camera client"
```

---

### Task 6: Camera reactive state + client factory

**Files:**
- Modify: `utils/state.py` (append a new "Video Capture State" section)
- Create: `tests/test_camera_state.py`

**Interfaces:**
- Consumes: `hardware.pi_camera.PiCameraClient`, `hardware.pi_camera_mock.MockPiCameraClient`.
- Produces (in `utils/state.py`):
  - `camera_enabled = solara.reactive(False)`
  - `camera_host = solara.reactive("raspberrypi.local")`
  - `camera_port = solara.reactive(8770)`
  - `camera_sensor_id = solara.reactive(None)`  # int | None
  - `camera_status = solara.reactive("unknown")`  # str
  - `camera_mock = solara.reactive(False)`
  - `camera_video_filename = solara.reactive("")`
  - `make_camera_client()` → returns a `MockPiCameraClient` if `camera_mock` is set, else a `PiCameraClient` built from `camera_host`/`camera_port`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_state.py`:

```python
from utils import state
from hardware.pi_camera import PiCameraClient
from hardware.pi_camera_mock import MockPiCameraClient


def test_defaults():
    assert state.camera_enabled.value is False
    assert state.camera_port.value == 8770
    assert state.camera_sensor_id.value is None


def test_make_client_real_vs_mock():
    state.camera_mock.set(False)
    assert isinstance(state.make_camera_client(), PiCameraClient)
    state.camera_mock.set(True)
    assert isinstance(state.make_camera_client(), MockPiCameraClient)
    state.camera_mock.set(False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_camera_state.py -v`
Expected: FAIL — `AttributeError: module 'utils.state' has no attribute 'camera_enabled'`

- [ ] **Step 3: Append the state + factory**

Add to the end of `utils/state.py`:

```python
# ============================================================================
# Video Capture State (Pi camera)
# ============================================================================

camera_enabled = solara.reactive(False)
"""Whether concurrent Pi video capture is active for this session."""

camera_host = solara.reactive("raspberrypi.local")
"""Hostname or IP of the Raspberry Pi camera server."""

camera_port = solara.reactive(8770)
"""TCP port of the Pi camera server."""

camera_sensor_id = solara.reactive(None)
"""Sensor ID (1-24) whose Start button bookmarks the video, or None."""

camera_status = solara.reactive("unknown")
"""Last known camera connection status string for the UI."""

camera_mock = solara.reactive(False)
"""Use the in-memory mock camera client (set by recording_gui_mock.py)."""

camera_video_filename = solara.reactive("")
"""Video filename reported by the Pi at session start."""


def make_camera_client():
    """Build the appropriate camera client based on mock/real state."""
    if camera_mock.value:
        from hardware.pi_camera_mock import MockPiCameraClient
        return MockPiCameraClient()
    from hardware.pi_camera import PiCameraClient
    return PiCameraClient(camera_host.value, camera_port.value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_camera_state.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add utils/state.py tests/test_camera_state.py
git commit -m "feat: add camera reactive state and client factory"
```

---

### Task 7: HDF5 video metadata writer

**Files:**
- Modify: `recording/recorder.py` (add `_serial_for_sensor` helper + `write_video_metadata`)
- Create: `tests/test_video_metadata.py`

**Interfaces:**
- Consumes: existing `SensorRecorder`, `utils.state.SERIAL_NUMBER_SENSOR_MAP`.
- Produces:
  - `SensorRecorder.write_video_metadata(sensor_id, frame_index=None, pts=None, video_filename=None, cycle=0)` — writes datasets `video_frame_index{suffix}`, `video_pts{suffix}`, `video_filename{suffix}` into `board_{sn}/sensor_{sensor_id}`, where `suffix` is `""` for cycle 0 else `str(cycle)`, matching the `start_time` naming convention.

- [ ] **Step 1: Write the failing test**

Create `tests/test_video_metadata.py`:

```python
import h5py
from recording.recorder import SensorRecorder
from utils.state import SERIAL_NUMBER_SENSOR_MAP


def _make_recorder(tmp_path):
    serial = next(iter(SERIAL_NUMBER_SENSOR_MAP))  # a real serial from the map
    controllers = {serial: object()}
    rec = SensorRecorder(mpr121_manager=None,
                         filename=str(tmp_path / "raw.h5"),
                         controllers=controllers)
    rec.initialize_hdf5_file()
    return rec, serial, SERIAL_NUMBER_SENSOR_MAP[serial][0]


def test_write_video_metadata_cycle0(tmp_path):
    rec, serial, sensor_id = _make_recorder(tmp_path)
    rec.write_video_metadata(sensor_id=sensor_id, frame_index=42, pts=1.25,
                             video_filename="clip.mp4", cycle=0)
    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert grp["video_frame_index"][()] == 42
        assert abs(grp["video_pts"][()] - 1.25) < 1e-9
        assert grp["video_filename"][()].decode() == "clip.mp4"


def test_write_video_metadata_cycle1_suffix(tmp_path):
    rec, serial, sensor_id = _make_recorder(tmp_path)
    rec.write_video_metadata(sensor_id=sensor_id, frame_index=7, pts=0.5,
                             video_filename="clip.mp4", cycle=1)
    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert grp["video_frame_index1"][()] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_video_metadata.py -v`
Expected: FAIL — `AttributeError: 'SensorRecorder' object has no attribute 'write_video_metadata'`

- [ ] **Step 3: Implement the method**

In `recording/recorder.py`, add these two methods to the `SensorRecorder` class (place after `write_sensor_metadata`):

```python
    def _serial_for_sensor(self, sensor_id: int) -> str:
        """Return the board serial number that owns the given sensor."""
        from utils.state import SERIAL_NUMBER_SENSOR_MAP
        import numpy as np

        sn_idx = [sensor_id in sensors for sensors in SERIAL_NUMBER_SENSOR_MAP.values()]
        return str(np.array(list(SERIAL_NUMBER_SENSOR_MAP.keys()))[sn_idx].item())

    def write_video_metadata(self, sensor_id: int, frame_index=None, pts=None,
                             video_filename=None, cycle=0):
        """Write video bookmark metadata for a sensor's recording cycle.

        Datasets mirror the start_time cycle-suffix convention:
        cycle 0 -> "video_frame_index", cycle 1 -> "video_frame_index1", etc.
        """
        sn = self._serial_for_sensor(sensor_id)
        suffix = "" if cycle == 0 else str(cycle)

        with h5py.File(self.filename, "r+") as h5f:
            group_path = f"board_{sn}/sensor_{sensor_id}"
            if group_path not in h5f:
                h5f[f"board_{sn}"].create_group(f"sensor_{sensor_id}")
            group = h5f[group_path]

            for base, value in (
                (f"video_frame_index{suffix}", frame_index),
                (f"video_pts{suffix}", pts),
                (f"video_filename{suffix}", video_filename),
            ):
                if value is None:
                    continue
                if base in group:
                    del group[base]
                group.create_dataset(base, data=value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_video_metadata.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add recording/recorder.py tests/test_video_metadata.py
git commit -m "feat: write video bookmark metadata into HDF5"
```

---

### Task 8: Wire camera into the recording lifecycle

**Files:**
- Modify: `components/session_controls.py` (`start_recording`, `stop_recording`, add module global `camera_client`)
- Modify: `components/sensor_card.py` (`start_sensor`)
- Create: `tests/test_camera_integration.py`

**Interfaces:**
- Consumes: `state.make_camera_client`, `state.camera_enabled`, `state.camera_sensor_id`, `state.camera_video_filename`, `SensorRecorder.write_video_metadata`, `PiCameraClient`/`MockPiCameraClient` surface.
- Produces: module global `components.session_controls.camera_client` (set on session start, used by `sensor_card.start_sensor` and `stop_recording`).

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_camera_integration.py`:

```python
import asyncio
import h5py

from utils import state
from utils.state import SERIAL_NUMBER_SENSOR_MAP
from recording.recorder import SensorRecorder
import components.session_controls as sc
import components.sensor_card as scard


def test_bookmark_written_on_sensor_start(tmp_path):
    serial = next(iter(SERIAL_NUMBER_SENSOR_MAP))
    sensor_id = SERIAL_NUMBER_SENSOR_MAP[serial][0]

    # Build a real recorder over a temp HDF5 file.
    rec = SensorRecorder(mpr121_manager=None,
                         filename=str(tmp_path / "raw.h5"),
                         controllers={serial: object()})
    rec.initialize_hdf5_file()
    sc.current_recorder = rec

    # Camera: mock + enabled, designate this sensor as the camera sensor.
    state.camera_mock.set(True)
    state.camera_enabled.set(True)
    state.camera_sensor_id.set(sensor_id)
    state.recording_all.set(True)

    # Simulate session start wiring (start_recording's camera block).
    sc.camera_client = state.make_camera_client()
    resp = sc.camera_client.start_session("clip")
    state.camera_video_filename.set(resp["video_filename"])

    # Drive the per-sensor Start inside a running loop (start_sensor uses create_task).
    async def _run():
        scard.start_sensor(sensor_id)
    asyncio.run(_run())

    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert grp["video_frame_index"][()] == 1
        assert grp["video_filename"][()].decode() == "clip.mp4"

    # Cleanup global state for other tests.
    state.camera_enabled.set(False)
    state.camera_mock.set(False)
    state.recording_all.set(False)
    sc.current_recorder = None
    sc.camera_client = None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_camera_integration.py -v`
Expected: FAIL — `AttributeError: module 'components.session_controls' has no attribute 'camera_client'`

- [ ] **Step 3: Add the camera_client global to session_controls**

In `components/session_controls.py`, just below the existing globals (after line `recording_task = None`), add:

```python
# Active Pi camera client for the current session (None when camera disabled)
camera_client = None
```

- [ ] **Step 4: Start camera pre-roll in `start_recording`**

In `components/session_controls.py`, inside `start_recording`, immediately after the line `recording_task = asyncio.create_task(run_recording())`, insert:

```python
    # Optionally start concurrent Pi video pre-roll (non-blocking).
    global camera_client
    camera_client = None
    if state.camera_enabled.value:
        try:
            camera_client = state.make_camera_client()
            video_base = os.path.splitext(os.path.basename(full_path))[0]
            resp = camera_client.start_session(video_base)
            if resp.get("ok"):
                state.camera_video_filename.set(resp.get("video_filename", ""))
                state.add_log_message(
                    f"Camera pre-roll started: {resp.get('video_filename')}")
            else:
                state.add_log_message(
                    f"WARNING: Camera start failed: {resp.get('error')}")
        except Exception as exc:
            state.add_log_message(f"WARNING: Camera start error: {exc}")
```

- [ ] **Step 5: Stop + fetch in `stop_recording`**

In `components/session_controls.py`, inside `stop_recording`, immediately before the final `state.recording_all.set(False)` line, insert:

```python
    # Stop the camera and copy its files back (non-blocking, best-effort).
    global camera_client
    if state.camera_enabled.value and camera_client is not None:
        try:
            resp = camera_client.stop_session()
            if resp.get("ok"):
                names = [f["name"] for f in resp.get("files", [])]
                fetched = camera_client.fetch_files(names, state.output_directory.value)
                state.add_log_message(
                    f"Camera stopped; copied {len(fetched)} file(s)")
            else:
                state.add_log_message(
                    f"WARNING: Camera stop failed: {resp.get('error')}")
        except Exception as exc:
            state.add_log_message(f"WARNING: Camera stop error: {exc}")
    camera_client = None
```

- [ ] **Step 6: Bookmark in `start_sensor`**

In `components/sensor_card.py`, inside `start_sensor`, immediately after the `state.add_log_message(f"Sensor {sensor_id}: Recording started{cycle_text}")` line, insert:

```python
    # Bookmark the concurrent video for the designated camera sensor.
    if state.camera_enabled.value and sensor_id == state.camera_sensor_id.value:
        from components import session_controls
        client = session_controls.camera_client
        if client is not None:
            try:
                resp = client.bookmark(sensor_id)
                if resp.get("ok"):
                    current_recorder.write_video_metadata(
                        sensor_id=sensor_id,
                        frame_index=resp.get("frame_index"),
                        pts=resp.get("pts"),
                        video_filename=state.camera_video_filename.value,
                        cycle=current_cycle,
                    )
                    state.add_log_message(
                        f"Sensor {sensor_id}: video bookmark "
                        f"frame={resp.get('frame_index')} pts={resp.get('pts'):.3f}")
                else:
                    state.add_log_message(
                        f"WARNING: Sensor {sensor_id}: bookmark failed: {resp.get('error')}")
            except Exception as exc:
                state.add_log_message(
                    f"WARNING: Sensor {sensor_id}: bookmark error: {exc}")
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/test_camera_integration.py -v`
Expected: PASS (1 passed)

- [ ] **Step 8: Run the full suite (no regressions)**

Run: `python -m pytest -v`
Expected: PASS (all tests green)

- [ ] **Step 9: Commit**

```bash
git add components/session_controls.py components/sensor_card.py tests/test_camera_integration.py
git commit -m "feat: wire Pi camera bookmark into recording lifecycle"
```

---

### Task 9: Camera controls UI card + GUI wiring

**Files:**
- Create: `components/camera_controls.py`
- Modify: `recording_gui.py` (import + render the card)
- Modify: `recording_gui_mock.py` (enable mock camera)
- Create: `tests/test_camera_controls_import.py`

**Interfaces:**
- Consumes: `utils.state` camera reactives, `state.make_camera_client`.
- Produces: `CameraControlsCard()` solara component; a `test_connection()` helper that pings via a fresh client and updates `state.camera_status`.

- [ ] **Step 1: Write the failing import/smoke test**

Create `tests/test_camera_controls_import.py`:

```python
from utils import state


def test_card_is_callable():
    from components.camera_controls import CameraControlsCard, test_connection
    assert callable(CameraControlsCard)
    assert callable(test_connection)


def test_test_connection_sets_status_mock():
    state.camera_mock.set(True)
    test_connection_result = _run_test_connection()
    assert state.camera_status.value in ("connected", "disconnected")
    state.camera_mock.set(False)


def _run_test_connection():
    from components.camera_controls import test_connection
    return test_connection()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_camera_controls_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'components.camera_controls'`

- [ ] **Step 3: Implement the card**

Create `components/camera_controls.py`:

```python
"""Camera (Pi video) controls UI card.

Lets the operator enable concurrent video, point at the Pi, designate which
sensor's Start bookmarks the video, and test the connection.
"""
import solara
from utils import state


def test_connection() -> bool:
    """Ping the Pi camera server and update camera_status. Returns reachability."""
    try:
        client = state.make_camera_client()
        ok = client.ping()
    except Exception:
        ok = False
    state.camera_status.set("connected" if ok else "disconnected")
    return ok


@solara.component
def CameraControlsCard():
    """UI for configuring and testing concurrent Pi video capture."""
    with solara.Card(title="Video Capture (Pi Camera)",
                     style={"margin-bottom": "20px"}):
        solara.Switch(label="Enable concurrent video",
                      value=state.camera_enabled.value,
                      on_value=state.camera_enabled.set,
                      disabled=state.recording_all.value)

        if state.camera_enabled.value:
            with solara.Row(style={"gap": "10px"}):
                solara.InputText(label="Pi host/IP",
                                 value=state.camera_host.value,
                                 on_value=state.camera_host.set,
                                 disabled=state.recording_all.value)
                solara.InputInt(label="Port",
                                value=state.camera_port.value,
                                on_value=state.camera_port.set,
                                disabled=state.recording_all.value)

            solara.Select(
                label="Camera sensor (whose Start bookmarks the video)",
                value=state.camera_sensor_id.value,
                values=list(range(1, 25)),
                on_value=state.camera_sensor_id.set,
                disabled=state.recording_all.value)

            with solara.Row(style={"margin-top": "10px", "gap": "10px"}):
                solara.Button(label="Test connection", on_click=test_connection,
                              color="secondary",
                              disabled=state.recording_all.value)
                status = state.camera_status.value
                if status == "connected":
                    solara.Success("✓ Connected")
                elif status == "disconnected":
                    solara.Error("✗ Not reachable")
                else:
                    solara.Text("Status: unknown")
```

- [ ] **Step 4: Render the card in the main GUI**

In `recording_gui.py`, add the import after the existing component imports (after line `from components.plot_dialog import TestPlotDialog`):

```python
from components.camera_controls import CameraControlsCard
```

Then, inside `Page`, render it immediately after `SessionControlsCard()`:

```python
        # Video Capture (Pi camera) Section
        CameraControlsCard()
```

- [ ] **Step 5: Enable mock camera in the mock GUI**

In `recording_gui_mock.py`, after the existing `use_mock_hardware()` call and before `from recording_gui import Page`, add:

```python
# Use the in-memory mock camera client (no Pi needed)
from utils import state
state.camera_mock.set(True)
```

- [ ] **Step 6: Run the import/smoke test**

Run: `python -m pytest tests/test_camera_controls_import.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Manually verify the mock GUI renders**

Run: `solara run recording_gui_mock.py`
Expected: App loads at http://localhost:8765; the "Video Capture (Pi Camera)" card appears; toggling "Enable concurrent video" reveals host/port/sensor fields; "Test connection" shows "✓ Connected" (mock pings True). Stop with Ctrl-C.

- [ ] **Step 8: Commit**

```bash
git add components/camera_controls.py recording_gui.py recording_gui_mock.py tests/test_camera_controls_import.py
git commit -m "feat: add camera controls UI and wire into GUI"
```

---

### Task 10: False-positive direct-bookmark alignment path

**Files:**
- Modify: `false_positive_analysis.py` (add `alignment_from_bookmark`)
- Create: `tests/test_bookmark_alignment.py`

**Interfaces:**
- Consumes: existing `false_positive_analysis` alignment conventions.
- Produces:
  - `alignment_from_bookmark(start_time_abs: float, video_pts: float) -> dict` returning `{"offset": start_time_abs - video_pts, "method": "bookmark"}`, where `video_relative_to_abs(t_video) = t_video + offset`. This lets analysis map video PTS to Unix time directly from the HDF5 bookmark, bypassing `detect_sipper_step`.

**Note:** This adds a new path; the existing sipper-step functions stay for legacy data (ACG-26-3). Read the top of `false_positive_analysis.py` first to match the existing offset sign convention used by `video_relative_to_abs` / `establish_alignment`; if that convention differs from the formula below, mirror the existing one and adjust the test accordingly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bookmark_alignment.py`:

```python
from false_positive_analysis import alignment_from_bookmark


def test_offset_maps_pts_to_abs():
    # Sipper inserted at Unix t=1000.0, which was video PTS=12.5.
    align = alignment_from_bookmark(start_time_abs=1000.0, video_pts=12.5)
    assert align["method"] == "bookmark"
    # A later frame at PTS=20.0 maps to abs = 20.0 + offset.
    abs_time = 20.0 + align["offset"]
    assert abs(abs_time - 1007.5) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bookmark_alignment.py -v`
Expected: FAIL — `ImportError: cannot import name 'alignment_from_bookmark'`

- [ ] **Step 3: Implement the function**

Add to `false_positive_analysis.py` (near `establish_alignment`):

```python
def alignment_from_bookmark(start_time_abs, video_pts):
    """Build an alignment from a CLiQR video bookmark (frame PTS at sipper-in).

    The recording GUI records, at sipper insertion, the Unix start_time and the
    concurrent video frame PTS. Their difference is a constant offset mapping
    video PTS to absolute Unix seconds: abs = pts + offset.

    Returns a dict {"offset": float, "method": "bookmark"} compatible with
    video_relative_to_abs.
    """
    return {"offset": float(start_time_abs) - float(video_pts), "method": "bookmark"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bookmark_alignment.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add false_positive_analysis.py tests/test_bookmark_alignment.py
git commit -m "feat: add direct-bookmark alignment path for false-positive analysis"
```

---

### Task 11: Documentation

**Files:**
- Create: `docs/VIDEO_CAPTURE.md`
- Modify: `docs/CLAUDE.md` (add video-capture subsystem; the file already has uncommitted modernization edits — commit those together)
- Modify: `CURRENT_STATE.md` (add 4th subsystem entry)

**Interfaces:** none (docs only).

- [ ] **Step 1: Write the Pi setup / operation guide**

Create `docs/VIDEO_CAPTURE.md`:

```markdown
# Concurrent Video Capture (Pi Camera)

CLiQR can record video from a Raspberry Pi 5 + Pi Camera 3 in sync with
capacitive recording. When the operator clicks a designated sensor's **Start**
button (sipper inserted), the desktop GUI bookmarks the current video frame so
capacitive data and video share a common time reference.

## Topology

- One Pi 5 + Camera 3 films one cage.
- Desktop GUI talks to the Pi over **TCP/LAN** (one link for trigger, ack, and
  file copy). No USB serial/gadget is used — Pi 5 does not support USB device
  mode.
- The camera is **optional and non-blocking**: if the Pi is unreachable,
  capacitive recording proceeds normally.

## Timing model: pre-roll + bookmark

The camera starts recording at **global session Start** and logs every frame's
`SensorTimestamp` to a `.txt` file (one nanosecond value per line). The
per-sensor Start click sends a `BOOKMARK`; the Pi replies with the current
frame index and PTS, stored in the HDF5 as `video_frame_index` / `video_pts` /
`video_filename`. Sync is bounded by one frame interval (~33 ms @ 30 fps), with
no camera-warmup slop.

## Pi setup (one-time)

1. Raspberry Pi OS (Bookworm or later) on the Pi 5.
2. Install picamera2 (preinstalled on recent Pi OS; otherwise
   `sudo apt install -y python3-picamera2`).
3. Copy this repository (or at least the `pi/` and `video/` packages) to the Pi.
4. Ensure the Pi and desktop are on the same LAN; note the Pi's hostname/IP.

## Running the Pi server

```bash
python -m pi.pi_camera_server --port 8770 --output-dir ~/cliqr_clips
```

The server pre-rolls on `START_SESSION`, bookmarks on `BOOKMARK`, finalizes the
MP4 + `.txt` on `STOP_SESSION`, and serves files on `GET_FILE`.

## Desktop usage

1. In the Recording GUI, open the **Video Capture (Pi Camera)** card.
2. Toggle **Enable concurrent video**.
3. Enter the Pi host/IP and port (default 8770).
4. Choose the **camera sensor** — the sensor whose Start button bookmarks video.
5. Click **Test connection** (expects "✓ Connected").
6. Start the session, then click that sensor's Start when the sipper goes in.
7. On Stop, the MP4 + `.txt` are copied into the session output directory.

## Output

- `<session>.mp4` — H.264 video.
- `<session>.txt` — per-frame `SensorTimestamp` (numpy.loadtxt-compatible),
  consumed by `false_positive_analysis.load_frame_offsets()`.
- HDF5 datasets per camera sensor: `video_frame_index`, `video_pts`,
  `video_filename` (cycle-suffixed like `start_time`).

## Analysis

Use `false_positive_analysis.alignment_from_bookmark(start_time_abs, video_pts)`
to map video PTS to Unix time directly from the bookmark — no sipper-step
detection needed for recordings made this way. Legacy data (e.g. ACG-26-3)
still uses `detect_sipper_step` / `establish_alignment`.

## Troubleshooting

- **"✗ Not reachable":** verify the Pi server is running, host/IP is correct,
  and the LAN allows the port. The session still records capacitively.
- **No video files copied:** they remain on the Pi under `--output-dir`; copy
  manually (`scp`).
```

- [ ] **Step 2: Add the subsystem to `docs/CLAUDE.md`**

In `docs/CLAUDE.md`, change the "Three subsystems" list to four and add the focus note. Replace the existing block:

```
Three subsystems:
1. **Recording GUI** — complete, in production
2. **Data analysis pipeline** — functional, actively maintained for manuscript
3. **False positive analysis** — implemented, validating detection accuracy against CVAT video annotations
```

with:

```
Four subsystems:
1. **Recording GUI** — complete, in production
2. **Data analysis pipeline** — functional, actively maintained for manuscript
3. **False positive analysis** — implemented, validating detection accuracy against CVAT video annotations
4. **Concurrent video capture** — Pi 5 + Camera 3 over TCP; bookmarks video frame at sipper insertion. See `docs/VIDEO_CAPTURE.md`.
```

Also add a row to the Architecture section describing the new packages:

```
### Concurrent Video Capture (`pi/`, `video/`, `hardware/pi_camera.py`)

Desktop ↔ Pi 5 over TCP. `video/protocol.py` (shared wire format), `pi/server_core.py` (dispatcher), `pi/pi_camera_server.py` (TCP + picamera2 entry), `pi/camera_backend.py` (picamera2), `hardware/pi_camera.py` (desktop client), `hardware/pi_camera_mock.py` (no-Pi mock), `components/camera_controls.py` (UI). Bookmarks stored in HDF5 as `video_frame_index`/`video_pts`/`video_filename`. Full guide: `docs/VIDEO_CAPTURE.md`.
```

- [ ] **Step 3: Add the subsystem to `CURRENT_STATE.md`**

In `CURRENT_STATE.md`, add a new component-status section after the False Positive Analysis section:

```markdown
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
```

- [ ] **Step 4: Run the full test suite once more**

Run: `python -m pytest -v`
Expected: PASS (all green)

- [ ] **Step 5: Commit (includes the pre-existing CLAUDE.md modernization)**

```bash
git add docs/VIDEO_CAPTURE.md docs/CLAUDE.md CURRENT_STATE.md
git commit -m "docs: document concurrent video capture subsystem"
```

---

### Task 12: Synced video analysis (trim + side-by-side render)

**Files:**
- Create: `video/sync_video.py`
- Create: `tests/test_sync_video.py`
- Modify: `requirements.txt` (add `imageio`, `imageio-ffmpeg`)
- Modify: `docs/VIDEO_CAPTURE.md` (add an "Analysis: synced video" section)

**Context:** When data analysis runs, it should (a) trim the recorded video to just the
sensor's recording window using the HDF5 start/stop timestamps and the video bookmark, and
(b) render an MP4 with the video on the left and the capacitive trace on the right. The
trace is zoomed to ±1 s around the current time, with the current time fixed at center and
the window sliding as playback advances. Trace shows raw `cap_data` plus detected lick
markers. Output frame rate matches the source video.

**Alignment math (no separate offset needed):** at the bookmark frame the video PTS equals
the sensor `start_time`, so each frame's absolute time is
`abs(f) = start_time_abs + (frame_offsets_ns[f] / 1e9 - bookmark_pts)`. The recording window
in video frames is the contiguous run where `start_time_abs <= abs(f) <= stop_time_abs`
(its first frame is the bookmark frame).

**Interfaces:**
- Consumes: HDF5 datasets from Task 7 (`video_frame_index`, `video_pts`, `video_filename`,
  plus existing `start_time`/`stop_time`, `cap_data`/`time_data`), `imageio`, `matplotlib`.
- Produces:
  - `compute_trim_frames(frame_offsets_ns, bookmark_pts, start_time_abs, stop_time_abs) -> tuple[int, int]`
    — `(start_frame, stop_frame)` inclusive indices.
  - `frame_abs_times(frame_offsets_ns, bookmark_pts, start_time_abs) -> np.ndarray`.
  - `trim_video(input_path, output_path, start_frame, stop_frame, fps=None) -> str`.
  - `render_synced_video(video_path, frame_offsets_ns, cap_time, cap_data, lick_times, bookmark_pts, start_time_abs, stop_time_abs, output_path, window_sec=1.0, fps=None) -> str`.
  - `make_sync_video_from_hdf5(raw_h5, sensor_id, video_path, frame_offsets_path, output_path, lick_times=None, cycle=0, window_sec=1.0, fps=None) -> str` — thin loader that pulls bookmark/window/trace from the HDF5 and calls `render_synced_video`.

- [ ] **Step 1: Add analysis video dependencies**

Append to `requirements.txt`:

```
imageio==2.36.1
imageio-ffmpeg==0.5.1
```

Install: `pip install imageio==2.36.1 imageio-ffmpeg==0.5.1`

- [ ] **Step 2: Write the failing test (pure helpers + render smoke)**

Create `tests/test_sync_video.py`:

```python
import numpy as np
import pytest

from video.sync_video import (
    frame_abs_times,
    compute_trim_frames,
    trim_video,
    render_synced_video,
)


def test_frame_abs_times_anchors_bookmark_to_start():
    # 5 frames, 0.1 s apart; bookmark is frame 2 (pts in seconds = its offset/1e9).
    offsets_ns = np.array([0, 1e8, 2e8, 3e8, 4e8])
    bookmark_pts = 2e8 / 1e9  # frame 2's pts in seconds
    abs_t = frame_abs_times(offsets_ns, bookmark_pts, start_time_abs=1000.0)
    assert abs(abs_t[2] - 1000.0) < 1e-9          # bookmark frame == start_time
    assert abs(abs_t[4] - 1000.2) < 1e-9          # 0.2 s later


def test_compute_trim_frames():
    offsets_ns = np.array([0, 1e8, 2e8, 3e8, 4e8])
    bookmark_pts = 2e8 / 1e9
    start, stop = compute_trim_frames(offsets_ns, bookmark_pts,
                                      start_time_abs=1000.0, stop_time_abs=1000.15)
    assert start == 2          # bookmark frame
    assert stop == 3           # last frame within window (1000.1 <= 1000.15)


@pytest.fixture
def tiny_video(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "src.mp4"
    writer = imageio.get_writer(str(path), fps=10)
    for i in range(10):
        frame = np.full((32, 32, 3), i * 20, dtype=np.uint8)
        writer.append_data(frame)
    writer.close()
    return path


def test_trim_video(tiny_video, tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    out = tmp_path / "trim.mp4"
    trim_video(str(tiny_video), str(out), start_frame=2, stop_frame=5)
    assert out.exists()
    reader = imageio.get_reader(str(out))
    n = sum(1 for _ in reader)
    reader.close()
    assert n == 4          # frames 2,3,4,5 inclusive


def test_render_synced_video_smoke(tiny_video, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("matplotlib")
    offsets_ns = np.arange(10) * 1e8            # 0.1 s apart, 10 frames
    bookmark_pts = 0.0                          # frame 0 is the bookmark
    start_time_abs = 1000.0
    stop_time_abs = 1000.9
    cap_time = np.linspace(999.5, 1001.5, 400)
    cap_data = np.sin(cap_time * 30.0)
    lick_times = np.array([1000.2, 1000.6])
    out = tmp_path / "synced.mp4"
    render_synced_video(
        str(tiny_video), offsets_ns, cap_time, cap_data, lick_times,
        bookmark_pts, start_time_abs, stop_time_abs, str(out),
        window_sec=1.0,
    )
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_sync_video.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'video.sync_video'`

- [ ] **Step 4: Implement the module**

Create `video/sync_video.py`:

```python
"""Trim recorded video to the capacitive recording window and render a
side-by-side synced MP4 (video left, sliding capacitive trace right).

Alignment: the video bookmark (frame PTS at sipper insertion) equals the
sensor start_time, so frame f's absolute Unix time is
    abs(f) = start_time_abs + (frame_offsets_ns[f] / 1e9 - bookmark_pts).
"""
import numpy as np


def frame_abs_times(frame_offsets_ns, bookmark_pts, start_time_abs):
    """Absolute Unix seconds for every video frame."""
    offsets = np.asarray(frame_offsets_ns, dtype=float)
    return start_time_abs + (offsets / 1e9 - bookmark_pts)


def compute_trim_frames(frame_offsets_ns, bookmark_pts, start_time_abs, stop_time_abs):
    """Inclusive (start_frame, stop_frame) covering [start_time_abs, stop_time_abs]."""
    abs_t = frame_abs_times(frame_offsets_ns, bookmark_pts, start_time_abs)
    mask = (abs_t >= start_time_abs) & (abs_t <= stop_time_abs)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("no video frames fall within the recording window")
    return int(idx[0]), int(idx[-1])


def trim_video(input_path, output_path, start_frame, stop_frame, fps=None):
    """Write frames [start_frame, stop_frame] (inclusive) to a new MP4."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(input_path)
    out_fps = fps or reader.get_meta_data().get("fps", 30)
    writer = imageio.get_writer(output_path, fps=out_fps)
    try:
        for i, frame in enumerate(reader):
            if i < start_frame:
                continue
            if i > stop_frame:
                break
            writer.append_data(frame)
    finally:
        writer.close()
        reader.close()
    return output_path


def render_synced_video(video_path, frame_offsets_ns, cap_time, cap_data, lick_times,
                        bookmark_pts, start_time_abs, stop_time_abs, output_path,
                        window_sec=1.0, fps=None):
    """Render an MP4: left = video frame, right = capacitive trace zoomed to
    ±window_sec around the current time (center fixed, window slides), with
    detected licks marked."""
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    abs_t = frame_abs_times(frame_offsets_ns, bookmark_pts, start_time_abs)
    start_frame, stop_frame = compute_trim_frames(
        frame_offsets_ns, bookmark_pts, start_time_abs, stop_time_abs)

    cap_time = np.asarray(cap_time, dtype=float)
    cap_data = np.asarray(cap_data, dtype=float)
    lick_times = np.asarray(lick_times, dtype=float) if lick_times is not None else np.array([])

    window_mask = (cap_time >= start_time_abs) & (cap_time <= stop_time_abs)
    if window_mask.any():
        ylim = (cap_data[window_mask].min(), cap_data[window_mask].max())
    else:
        ylim = (cap_data.min(), cap_data.max())

    reader = imageio.get_reader(video_path)
    out_fps = fps or reader.get_meta_data().get("fps", 30)

    fig, (ax_vid, ax_tr) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1]})
    ax_vid.axis("off")
    image = ax_vid.imshow(reader.get_data(start_frame))

    ax_tr.plot(cap_time, cap_data, color="steelblue", lw=0.8)
    if lick_times.size:
        ax_tr.scatter(lick_times, np.interp(lick_times, cap_time, cap_data),
                      color="red", s=20, zorder=5, label="detected licks")
        ax_tr.legend(loc="upper right", fontsize=8)
    center0 = abs_t[start_frame]
    vline = ax_tr.axvline(center0, color="k", lw=1)
    ax_tr.set_ylim(*ylim)
    ax_tr.set_xlabel("time (s, Unix)")
    ax_tr.set_ylabel("capacitance")

    def update(f):
        image.set_data(reader.get_data(f))
        center = abs_t[f]
        ax_tr.set_xlim(center - window_sec, center + window_sec)
        vline.set_xdata([center, center])
        return image, vline

    anim = FuncAnimation(fig, update, frames=range(start_frame, stop_frame + 1),
                         blit=False)
    try:
        anim.save(output_path, writer=FFMpegWriter(fps=out_fps))
    finally:
        reader.close()
        plt.close(fig)
    return output_path


def make_sync_video_from_hdf5(raw_h5, sensor_id, video_path, frame_offsets_path,
                              output_path, lick_times=None, cycle=0,
                              window_sec=1.0, fps=None):
    """Load bookmark, recording window, and trace for a sensor from the raw HDF5,
    then render the synced video. `lick_times` (absolute Unix seconds) come from
    the filtered-data analysis; pass them in to mark detected licks.

    NOTE: verify the HDF5 group path and lick-time source against the analysis
    notebook before relying on this in batch.
    """
    import h5py

    suffix = "" if cycle == 0 else str(cycle)
    frame_offsets_ns = np.loadtxt(frame_offsets_path)

    from utils.state import SERIAL_NUMBER_SENSOR_MAP
    sn = [s for s, sensors in SERIAL_NUMBER_SENSOR_MAP.items() if sensor_id in sensors][0]

    with h5py.File(raw_h5, "r") as h5f:
        grp = h5f[f"board_{sn}/sensor_{sensor_id}"]
        bookmark_pts = float(grp[f"video_pts{suffix}"][()])
        start_time_abs = float(grp[f"start_time{suffix}"][()])
        stop_time_abs = float(grp[f"stop_time{suffix}"][()])
        cap_time = grp["time_data"][()]
        cap_data = grp["cap_data"][()]

    return render_synced_video(
        video_path, frame_offsets_ns, cap_time, cap_data, lick_times,
        bookmark_pts, start_time_abs, stop_time_abs, output_path,
        window_sec=window_sec, fps=fps)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_sync_video.py -v`
Expected: PASS (4 passed; render/trim tests skip only if imageio-ffmpeg is unavailable)

- [ ] **Step 6: Document the analysis step**

Add to `docs/VIDEO_CAPTURE.md`, after the "Analysis" section:

```markdown
### Analysis: synced video

`video/sync_video.py` turns a recorded session into review media:

- `make_sync_video_from_hdf5(raw_h5, sensor_id, video_path, frame_offsets_path, output_path, lick_times=...)`
  trims the video to the sensor's `start_time`/`stop_time` window (via the
  bookmark) and renders an MP4 with the video on the left and the capacitive
  trace on the right. The trace is zoomed to ±1 s around the current time, with
  the current time fixed at center and the window sliding; detected licks are
  marked in red. Output frame rate matches the source video.
- `trim_video(...)` alone writes just the trimmed recording-window MP4.

Call it from `DataAnalysis.ipynb` after lick detection, passing the absolute
lick times for the camera sensor.
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all green)

- [ ] **Step 8: Commit**

```bash
git add video/sync_video.py tests/test_sync_video.py requirements.txt docs/VIDEO_CAPTURE.md
git commit -m "feat: add synced video trim + side-by-side render for analysis"
```

---

## Self-Review Notes

- **Spec coverage:** Pi server (T3), TCP transport (T3/T4), pre-roll+bookmark (T3 backend + T8 integration), operator-selected camera sensor (T6 state + T8 guard + T9 UI), auto-copy after stop (T4 fetch_files + T8 stop_recording), drop sipper / direct bookmark (T7 HDF5 + T10 alignment), non-blocking camera (try/except in T8, graceful client errors in T4), mock for no-Pi tests (T5 + T9), docs incl. committing modernized CLAUDE.md (T11). Analysis-side video trim + side-by-side synced render with ±1 s sliding centered trace and lick markers (T12). All spec sections mapped.
- **T12 alignment reuse:** `frame_abs_times`/`compute_trim_frames` use the same bookmark↔start_time identity as T10's `alignment_from_bookmark` (offset cancels at the bookmark frame). Consistent.
- **Placeholder scan:** none — every code/test step contains complete code.
- **Type consistency:** backend contract (`is_active`, `start_session`, `bookmark`, `stop_session`) identical across T2/T3/T4/T5; response dict shape (`ok`, `frame_index`, `pts`, `pi_monotonic`, `video_filename`, `files[].name/size`) consistent client↔server; `write_video_metadata` signature matches its T8 call site; `make_camera_client` used in T8/T9.
- **Known soft spot (flagged in-task):** the picamera2 PTS/encoder API (T3 Step 5) and the false-positive offset sign convention (T10) need on-device / in-context verification; both are isolated and noted in their tasks.
