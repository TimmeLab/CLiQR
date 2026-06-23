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
        # TODO: Change to write relative nanoseconds (timestamp - first_frame_SensorTimestamp)
        # and add 2-line header (frame count and first_frame_SensorTimestamp) for
        # compatibility with false_positive_analysis.load_frame_offsets().
        # Pending validation on Pi.
        self._pts_fh.write(f"{timestamp}\n")
        self._frame_count += 1

    def bookmark(self, sensor_id) -> dict:
        metadata = self._picam2.capture_metadata()
        pts = metadata.get("SensorTimestamp", 0) / 1e9
        # TODO: Change to report pts as relative seconds (relative to session start /
        # first_frame_SensorTimestamp) for consistency with alignment_from_bookmark()
        # and the planned .txt format change. Pending validation on Pi.
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
