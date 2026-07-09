"""Hardware-independent request dispatcher for the Pi camera server.

Holds no networking and no camera code: it maps decoded protocol requests
to a camera backend and returns response dicts. This keeps the command
logic unit-testable with a fake backend (no picamera2, no hardware).

picamera2's encoder/FfmpegOutput pipeline is bound to the thread that calls
start_recording: if that thread exits, the mp4 is never finalized. The TCP
front-end is a ThreadingTCPServer and the desktop client opens a fresh
connection per request, so each request would otherwise run on a throwaway
handler thread that dies after responding. We therefore funnel every backend
call through a single long-lived worker thread so the camera lifecycle is
owned by one persistent thread.
"""
from concurrent.futures import ThreadPoolExecutor

from video import protocol


class CameraServer:
    """Dispatches protocol requests to a camera backend."""

    def __init__(self, backend):
        self.backend = backend
        # max_workers=1 -> one persistent thread owns every backend/camera call.
        self._camera = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera")

    def handle(self, request: dict) -> dict:
        return self._camera.submit(self._handle, request).result()

    def _handle(self, request: dict) -> dict:
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

            if cmd == protocol.SNAPSHOT:
                # Idle-only: the camera is busy while a session records.
                if self.backend.is_active:
                    return protocol.make_error(
                        "cannot snapshot during active recording")
                import base64
                jpeg = self.backend.snapshot()
                return protocol.make_ok(
                    image=base64.b64encode(jpeg).decode("ascii"), format="jpeg")

            if cmd == protocol.STOP_SESSION:
                if not self.backend.is_active:
                    return protocol.make_error("no active session")
                files = self.backend.stop_session()
                return protocol.make_ok(files=files)

            return protocol.make_error(f"unknown command: {cmd}")
        except Exception as exc:  # backend/hardware errors never crash the server
            return protocol.make_error(str(exc))
