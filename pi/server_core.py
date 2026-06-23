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
