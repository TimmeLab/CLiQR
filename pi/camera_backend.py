"""picamera2 backend for the Pi camera server (Raspberry Pi 5 + Camera 3).

Pre-rolls the camera at session start, logging every frame's SensorTimestamp
to a `.txt` file (one nanosecond timestamp per line, numpy.loadtxt-compatible).
BOOKMARK reports the live frame count and current PTS so the desktop can map a
sensor's Start click to a video frame.

picamera2 is imported lazily so this module only loads on the Pi.
"""
import shutil
import time
from pathlib import Path

# imx708 fast readout mode: the sensor's 1536x864 mode runs at up to 120 fps.
# Requesting a 1536x864 *raw* stream pins the sensor to that fast mode; locking
# both FrameDurationLimits to 1e6/120 us then fixes it at 120 fps (subject to
# exposure headroom in low light).
#
# The Pi 5 has no hardware H.264 encoder, so start_recording() runs libav's
# software encoder. Encoding the full 1536x864 main stream at 120 fps saturates
# the CPU and drops ~10-20% of frames (see docs/superpowers/specs/
# 2026-07-08-framerate-bottleneck-diagnosis-design.md). Downscaling the encoded
# main stream to 1280x720 and capping the bitrate keeps the encoder inside its
# budget: measured drop-free over a 60 s run at 120 fps. Frame *capture* stays at
# the sensor's native 1536x864/120; only the recorded video is downscaled.
# A session's video must always fit on disk: after each run, old recordings
# are deleted (oldest first) until at least this much space is free for the
# next session. Videos are otherwise kept as long as possible.
MIN_FREE_BYTES = 5 * 1024 ** 3  # 5 GB

SENSOR_FAST_MODE_SIZE = (1536, 864)
RECORD_SIZE = (1280, 720)
TARGET_FPS = 120
BITRATE = 3_000_000  # bits/s; keeps 720p120 software H.264 drop-free
_FRAME_DURATION_US = round(1_000_000 / TARGET_FPS)  # 8333 us


def video_config_kwargs() -> dict:
    """Kwargs for Picamera2.create_video_configuration.

    Encodes a 1280x720 main stream while the 1536x864 raw stream pins the
    sensor to its fast 120 fps mode. Pure (no picamera2 import) so the intended
    resolution/fps are testable off-hardware.
    """
    return {
        "main": {"size": RECORD_SIZE},
        "raw": {"size": SENSOR_FAST_MODE_SIZE},
        "controls": {
            "FrameDurationLimits": (_FRAME_DURATION_US, _FRAME_DURATION_US),
            "AfMode": 0,
            "LensPosition": 1000,
        },
    }


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
        self._last_frame_ts_ns = 0

    @property
    def is_active(self) -> bool:
        return self._active

    # picamera2 construction is funnelled through these hooks so the session
    # lifecycle is testable off-hardware (a test backend overrides them).
    def _create_camera(self):
        from picamera2 import Picamera2
        return Picamera2()

    def _create_encoder(self):
        from picamera2.encoders import H264Encoder
        return H264Encoder(bitrate=BITRATE)

    def _create_output(self, video_path):
        from picamera2.outputs import FfmpegOutput
        return FfmpegOutput(str(video_path))

    def _capture_jpeg(self, cam):
        """Capture one full-FOV still as JPEG bytes from an already-started cam."""
        from io import BytesIO
        buf = BytesIO()
        cam.capture_file(buf, format="jpeg")
        return buf.getvalue()

    def snapshot(self) -> bytes:
        """Grab one still JPEG for a pre-recording alignment check.

        Idle-only: opens a throwaway camera, captures a full-FOV still at
        RECORD_SIZE (so framing matches the recorded video), and releases the
        device. Does not touch the session state; the server refuses SNAPSHOT
        while a session is active, so self._picam2 is None here.
        """
        cam = self._create_camera()
        try:
            cam.configure(cam.create_still_configuration(main={"size": RECORD_SIZE}))
            cam.start()
            return self._capture_jpeg(cam)
        finally:
            try:
                cam.stop()
            except Exception:
                pass
            cam.close()

    def _release_camera(self):
        """Close and drop the camera so the device is released for reuse."""
        if self._picam2 is not None:
            try:
                self._picam2.close()
            finally:
                self._picam2 = None

    def start_session(self, name: str) -> str:
        # Defensive: free any camera a prior session left held (e.g. a stop that
        # never ran) so acquire() can't fail with "camera already in use".
        self._release_camera()
        self._picam2 = self._create_camera()
        config = self._picam2.create_video_configuration(**video_config_kwargs())
        self._picam2.configure(config)

        self._video_path = self.output_dir / f"{name}.mp4"
        self._pts_path = self.output_dir / f"{name}.txt"
        self._pts_fh = open(self._pts_path, "w")
        self._frame_count = 0
        self._picam2.pre_callback = self._on_frame

        self._encoder = self._create_encoder()
        self._picam2.start_recording(self._encoder, self._create_output(self._video_path))
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
        self._last_frame_ts_ns = timestamp

    def bookmark(self, sensor_id) -> dict:
        # frame_index and pts must describe the SAME frame: the most recent one
        # the pre_callback logged. capture_metadata() would instead fetch (and
        # block until) a newer frame, whose timestamp would not match
        # frame_index, biasing the video<->trace anchor.
        # TODO: Change to report pts as relative seconds (relative to session start /
        # first_frame_SensorTimestamp) for consistency with alignment_from_bookmark()
        # and the planned .txt format change. Pending validation on Pi.
        return {
            "frame_index": self._frame_count,
            "pts": self._last_frame_ts_ns / 1e9,
            "pi_monotonic": time.monotonic(),
        }

    def stop_session(self) -> list:
        try:
            self._picam2.stop_recording()
        finally:
            # Always release the device and close the file, even if
            # stop_recording raised, so the next session can acquire the camera.
            self._release_camera()
            if self._pts_fh is not None:
                self._pts_fh.close()
                self._pts_fh = None
            self._active = False
        files = []
        for path in (self._video_path, self._pts_path):
            files.append({"name": path.name, "size": path.stat().st_size})
        return files

    def _disk_free_bytes(self) -> int:
        return shutil.disk_usage(self.output_dir).free

    def reclaim_disk_space(self) -> dict:
        """Delete the oldest recordings until MIN_FREE_BYTES is free.

        Never deletes the current session's video (self._video_path): the
        desktop has not fetched it yet, and losing the run just recorded is
        never worth the space. Each deleted video takes its companion .txt
        timestamp file with it (useless without the frames it indexes).

        Returns {"deleted": [names], "low_disk": bool, "free_bytes": int};
        low_disk means even deleting every old video left < MIN_FREE_BYTES,
        so the user must free space manually before the next session.
        """
        deleted = []
        free = self._disk_free_bytes()
        if free < MIN_FREE_BYTES:
            candidates = sorted(
                (p for p in self.output_dir.glob("*.mp4") if p != self._video_path),
                key=lambda p: p.stat().st_mtime)
            for mp4 in candidates:
                if free >= MIN_FREE_BYTES:
                    break
                try:
                    mp4.unlink()
                    mp4.with_suffix(".txt").unlink(missing_ok=True)
                except OSError:
                    continue  # skip undeletable files; keep reclaiming
                deleted.append(mp4.name)
                free = self._disk_free_bytes()
        return {
            "deleted": deleted,
            "low_disk": free < MIN_FREE_BYTES,
            "free_bytes": free,
        }
