"""In-memory mock of PiCameraClient for mock-mode GUI and tests.

No sockets, no Pi. Returns deterministic, monotonically increasing frame
indices so behavior is assertable.
"""
import time
from pathlib import Path

# A real 1x1 JPEG so the mock GUI's snapshot data URI renders a valid image.
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAAB"
    "AAAAAAAAAAAAAAAAAAAAAP/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AfwD/2Q=="
)


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

    def snapshot(self) -> dict:
        return {"ok": True, "image": _TINY_JPEG_B64, "format": "jpeg"}

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
