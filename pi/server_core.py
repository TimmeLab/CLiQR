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
import logging
from concurrent.futures import ThreadPoolExecutor

from video import protocol

log = logging.getLogger(__name__)


class CameraServer:
    """Dispatches protocol requests to a camera backend."""

    def __init__(self, backend):
        self.backend = backend
        # max_workers=1 -> one persistent thread owns every backend/camera call.
        self._camera = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera")

    def handle(self, request: dict) -> dict:
        # Every command is low-frequency (a handful per session), so logging
        # all of them costs nothing and makes a failed session reconstructable.
        # Without this a refused or failed START_SESSION left no trace at all.
        cmd = request.get("cmd")
        response = self._camera.submit(self._handle, request).result()
        if response.get("ok"):
            log.info("%s -> ok", cmd)
        else:
            log.warning("%s -> error: %s", cmd, response.get("error"))
        return response

    def _handle(self, request: dict) -> dict:
        cmd = request.get("cmd")
        try:
            if cmd == protocol.PING:
                return protocol.make_ok(status="ready", active=self.backend.is_active)

            if cmd == protocol.START_SESSION:
                name = request.get("name")
                if not name:
                    return protocol.make_error("START_SESSION requires 'name'")
                # Reclaim disk space BEFORE recording starts: a crashed run
                # never reaches the post-stop cleanup, so the leftover videos
                # (including a crashed run's own) are deleted here, oldest
                # first, until 5 GB is free. Best-effort, like the post-stop
                # pass.
                try:
                    cleanup = self.backend.reclaim_disk_space()
                except Exception:
                    cleanup = {}
                video_filename = self.backend.start_session(name)
                return protocol.make_ok(video_filename=video_filename, **cleanup)

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
                # stalls: segments the watchdog restarted because frame
                # delivery died mid-session. Reported so the desktop can warn
                # instead of the run looking clean (2026-07-21: it did not, and
                # 90 min of video was silently missing). getattr keeps older /
                # test backends without a watchdog working.
                response = protocol.make_ok(
                    files=files,
                    stalls=getattr(self.backend, "stalls", []),
                    low_disk_during_run=getattr(
                        self.backend, "low_disk_during_run", False),
                    ffmpeg_log_overflows=getattr(
                        self.backend, "ffmpeg_log_overflows", 0))
                # Disk cleanup is best-effort: the stop succeeded and the
                # desktop needs the file list to fetch the recording, so a
                # cleanup error must not turn this reply into an error.
                try:
                    response.update(self.backend.reclaim_disk_space())
                except Exception:
                    pass
                return response

            return protocol.make_error(f"unknown command: {cmd}")
        except Exception as exc:  # backend/hardware errors never crash the server
            return protocol.make_error(str(exc))
