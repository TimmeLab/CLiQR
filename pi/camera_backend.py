"""picamera2 backend for the Pi camera server (Raspberry Pi 5 + Camera 3).

Pre-rolls the camera at session start, logging every frame's SensorTimestamp
to a `.txt` file (one nanosecond timestamp per line, numpy.loadtxt-compatible).
BOOKMARK reports the live frame count and current PTS so the desktop can map a
sensor's Start click to a video frame.

picamera2 is imported lazily so this module only loads on the Pi.
"""
import shutil
import threading
import time
from pathlib import Path

from pi import ffmpeg_output

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

# Frame-delivery watchdog. On 2026-07-21 the camera stopped delivering frames
# 44 min into a 2 h 19 min session and nothing noticed: the server kept
# answering TCP, and the Stop bookmark happily returned a frame that was 90 min
# stale. The watchdog polls the frame counter and, if it has not advanced,
# restarts recording into a `_partN` segment -- so a stall costs seconds of
# video instead of the rest of the run, and never fails the session (the
# capacitance trace is what must survive).
#
# 3 s is ~360 frames at 120 fps: far beyond any legitimate scheduling hiccup
# (the worst gap in the 2026-07-21 sidecar was 25 ms) but short enough that a
# stall costs little.
STALL_TIMEOUT_S = 3.0
WATCHDOG_POLL_S = 1.0

# Ceiling on watchdog restarts per session. Total video bytes are bitrate-bound
# no matter how often we restart, but each segment costs a .mp4 + .txt +
# .ffmpeg.log that the desktop fetches one file at a time over TCP, and every
# segment is protected from disk reclaim until the session ends. A camera that
# stalls this many times is broken, not flaky: stop restarting, keep the
# session alive (the capacitance trace is the thing that must survive), and say
# so loudly.
MAX_SEGMENTS = 10

# The 2026-07-21 muxer failure emitted two error lines per frame — ~240 lines/s
# at 120 fps, ~170 MB over a full session. That flood used to go to the
# terminal; now it goes to disk, competing for space with the very video it is
# complaining about. Cap it: past this size the log is truncated, so a wedged
# muxer costs bounded disk instead of unbounded.
FFMPEG_LOG_MAX_BYTES = 2 * 1024 ** 2  # 2 MB

# Disk reclaim only runs at session start and after stop, so nothing frees
# space mid-run. Warn (do not delete — the desktop may be fetching an old file)
# when the Pi drops below this while recording.
LOW_DISK_WARN_BYTES = 1024 ** 3  # 1 GB
LOW_DISK_WARN_INTERVAL_S = 60.0


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


def encoder_kwargs() -> dict:
    """Kwargs for the H.264 encoder.

    On the Pi 5, picamera2 aliases H264Encoder to the libav software encoder,
    whose framerate defaults to 30. x264 rate control budgets
    bitrate/framerate bits per frame, so leaving the default while feeding
    120 fps overshoots the target bitrate 4x (observed: a 2 h run ballooned
    past 9 GB — ~12 Mb/s instead of 3 — and filled the disk). Passing the true
    frame rate makes `bitrate` mean bits per second of wall time. Pure (no
    picamera2 import) so it is testable off-hardware.
    """
    return {"bitrate": BITRATE, "framerate": TARGET_FPS}


def segment_stem(name: str, segment: int) -> str:
    """File stem for a session's Nth recording segment (1-based).

    Segment 1 keeps the plain session name so the normal case produces exactly
    the filenames it always did; a watchdog restart appends `_part2`, `_part3`,
    ... Each segment carries its own `.txt` sidecar, whose SensorTimestamps are
    absolute Pi boot-clock nanoseconds — so a later segment still aligns to the
    capacitance trace through the Start bookmark, with no bookmark of its own.
    """
    return name if segment <= 1 else f"{name}_part{segment}"


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
        self._mux_proc = None
        self._frame_count = 0
        self._frame_errors = 0
        self._last_frame = None  # (frame_index, timestamp_ns) of the newest frame
        # Wall-clock (monotonic) of the last frame, for the stall watchdog.
        self._last_frame_monotonic = 0.0
        self._session_name = None
        self._segment = 0
        self._segment_paths = []  # [(video_path, pts_path)] for every segment
        self._segment_cap_reached = False
        self._last_disk_warn = 0.0
        self.stalls = []  # [{"segment", "reason", "idle_seconds", "frames"}]
        self.ffmpeg_log_overflows = 0
        self.low_disk_during_run = False
        # Guards segment open/close so the watchdog thread can never restart a
        # segment while stop_session is tearing one down.
        self._segment_lock = threading.RLock()
        self._watchdog = None
        self._watchdog_stop = threading.Event()

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
        return H264Encoder(**encoder_kwargs())

    def _create_output(self, video_path):
        """Start this segment's private ffmpeg and return its picamera2 Output.

        Not picamera2's FfmpegOutput: that one lets ffmpeg guess presentation
        timestamps from packet arrival times, which is what killed the
        2026-07-21 recording. See pi/ffmpeg_output.py.
        """
        output, proc = ffmpeg_output.spawn_muxer(video_path, TARGET_FPS)
        self._mux_proc = proc
        return output

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

    def _configure_camera(self):
        """Acquire the camera and apply the session video configuration."""
        self._picam2 = self._create_camera()
        config = self._picam2.create_video_configuration(**video_config_kwargs())
        self._picam2.configure(config)

    def start_session(self, name: str) -> str:
        # Defensive: free any camera a prior session left held (e.g. a stop that
        # never ran) so acquire() can't fail with "camera already in use".
        self._release_camera()
        self._session_name = name
        self._segment = 0
        self._segment_paths = []
        self._segment_cap_reached = False
        self._last_disk_warn = 0.0
        self.stalls = []
        self.ffmpeg_log_overflows = 0
        self.low_disk_during_run = False
        self._configure_camera()
        self._open_segment()
        self._active = True
        self._start_watchdog()
        return self._video_path.name

    def _open_segment(self):
        """Open the next segment's files and start recording into them.

        self._segment is published LAST, once the segment is fully open, so it
        is a safe "recording again" signal for anything watching from another
        thread.
        """
        with self._segment_lock:
            segment = self._segment + 1
            stem = segment_stem(self._session_name, segment)
            self._video_path = self.output_dir / f"{stem}.mp4"
            self._pts_path = self.output_dir / f"{stem}.txt"
            self._frame_count = 0
            self._last_frame = None
            self._last_frame_monotonic = time.monotonic()
            self._mux_proc = None
            self._pts_fh = open(self._pts_path, "w")
            self._picam2.pre_callback = self._on_frame

            self._encoder = self._create_encoder()
            self._picam2.start_recording(
                self._encoder, self._create_output(self._video_path))
            self._segment_paths.append((self._video_path, self._pts_path))
            self._segment = segment

    def _close_segment(self):
        """Stop recording, close the sidecar, and finalize this segment's mp4."""
        with self._segment_lock:
            try:
                if self._picam2 is not None:
                    self._picam2.stop_recording()
            finally:
                if self._pts_fh is not None:
                    try:
                        self._pts_fh.close()
                    except Exception:
                        pass
                    self._pts_fh = None
                # Closing ffmpeg's stdin is what writes the moov atom; skip it
                # and the segment is an unplayable file.
                ffmpeg_output.finish_muxer(self._mux_proc)
                self._mux_proc = None

    def _on_frame(self, request):
        # Runs inside picamera2's request loop. An exception escaping here kills
        # that thread, and with it frame delivery AND encoding -- the camera
        # goes silent while the server keeps answering TCP, which is precisely
        # the failure mode this file exists to prevent. Nothing in here is
        # allowed to raise.
        try:
            # The watchdog closes this handle while it swaps segments; frames
            # arriving in that window belong to no segment and are dropped
            # rather than written to a stale file. _frame_count must not
            # advance for them: it indexes lines in the CURRENT sidecar.
            fh = self._pts_fh
            if fh is None or fh.closed:
                return
            metadata = request.get_metadata()
            timestamp = metadata.get("SensorTimestamp", 0)
            # TODO: Change to write relative nanoseconds (timestamp - first_frame_SensorTimestamp)
            # and add 2-line header (frame count and first_frame_SensorTimestamp) for
            # compatibility with false_positive_analysis.load_frame_offsets().
            # Pending validation on Pi.
            fh.write(f"{timestamp}\n")
            index = self._frame_count
            self._frame_count += 1
            # Publish index and timestamp as ONE tuple. bookmark() runs on another
            # thread; a tuple rebind is atomic under the GIL, so it can never observe
            # a half-updated pair (frame_index of frame k with pts of frame k-1).
            self._last_frame = (index, timestamp)
            self._last_frame_monotonic = time.monotonic()
        except Exception as exc:
            # Rate-limited: a persistent fault (e.g. a full disk) would
            # otherwise flood the log at 120 lines/s -- the same flood that
            # deadlocked ffmpeg on 2026-07-21.
            self._frame_errors += 1
            if self._frame_errors <= 5 or self._frame_errors % 1000 == 0:
                print(f"camera: frame callback error #{self._frame_errors}: {exc}",
                      flush=True)

    # ---- stall watchdog -------------------------------------------------

    def _start_watchdog(self):
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="camera-watchdog", daemon=True)
        self._watchdog.start()

    def _stop_watchdog(self):
        self._watchdog_stop.set()
        if self._watchdog is not None:
            self._watchdog.join(WATCHDOG_POLL_S + STALL_TIMEOUT_S)
            self._watchdog = None

    def frames_stale_seconds(self) -> float:
        """Seconds since the last frame reached _on_frame (0 if none yet)."""
        if self._last_frame is None:
            return 0.0
        return time.monotonic() - self._last_frame_monotonic

    def _watchdog_loop(self):
        while not self._watchdog_stop.wait(WATCHDOG_POLL_S):
            if not self._active:
                continue
            try:
                self._cap_ffmpeg_log()
                self._warn_if_disk_low()
                self._check_segment_health()
            except Exception as exc:
                # A failed poll must not kill the watchdog thread; the next one
                # retries. The session is degraded, not lost.
                print(f"camera watchdog: poll failed: {exc}", flush=True)

    def _muxer_died(self) -> bool:
        """True if this segment's ffmpeg exited while the session is recording.

        Frames can keep flowing into the sidecar long after the muxer dies (a
        full disk kills ffmpeg, not the camera), so frame staleness alone would
        miss it and the mp4 would be silently truncated. Test backends never
        spawn a process, so _mux_proc is None and this is always False.
        """
        proc = self._mux_proc
        return proc is not None and proc.poll() is not None

    def _check_segment_health(self):
        """Restart the segment if frames stopped or the muxer died."""
        if self._segment_cap_reached:
            return
        if self._muxer_died():
            self._restart_stalled_segment("muxer exited")
            return
        # Only judge a session that has actually produced frames: camera warmup
        # legitimately takes a second or two, and restarting during it would
        # loop forever instead of surfacing the real failure.
        if self._last_frame is None:
            return
        idle = self.frames_stale_seconds()
        if idle >= STALL_TIMEOUT_S:
            self._restart_stalled_segment(f"no frames for {idle:.1f}s")

    def _cap_ffmpeg_log(self):
        """Truncate this segment's ffmpeg stderr log if it has run away.

        Normal runs leave it empty (-loglevel error plus timestamps ffmpeg no
        longer has to guess). A muxer stuck rejecting packets fills it at
        ~240 lines/s, so bound it rather than let it eat the space the video
        needs. The truncation itself is the signal that something is wrong.
        """
        if self._mux_proc is None or self._video_path is None:
            return
        log_path = ffmpeg_output.log_path_for(self._video_path)
        try:
            size = log_path.stat().st_size
        except OSError:
            return
        if size < FFMPEG_LOG_MAX_BYTES:
            return
        # ffmpeg's handle keeps writing at the new end-of-file after this.
        with open(log_path, "r+b") as fh:
            fh.truncate(0)
        self.ffmpeg_log_overflows += 1
        print(f"camera watchdog: {log_path.name} exceeded "
              f"{FFMPEG_LOG_MAX_BYTES} bytes and was truncated "
              f"(overflow #{self.ffmpeg_log_overflows}); the muxer is erroring "
              f"on nearly every packet", flush=True)

    def _warn_if_disk_low(self):
        """Warn (rate-limited) when the Pi runs low on space mid-session."""
        now = time.monotonic()
        if now - self._last_disk_warn < LOW_DISK_WARN_INTERVAL_S:
            return
        try:
            free = self._disk_free_bytes()
        except OSError:
            return
        if free >= LOW_DISK_WARN_BYTES:
            return
        self._last_disk_warn = now
        self.low_disk_during_run = True
        print(f"camera watchdog: only {free / 1024 ** 3:.2f} GB free on the Pi "
              f"while recording; the video will be truncated when it fills "
              f"(~{1.38:.2f} GB/h at the current bitrate)", flush=True)

    def _restart_stalled_segment(self, reason: str):
        """Tear down the stalled segment and record into a fresh one."""
        with self._segment_lock:
            # Re-check under the lock: stop_session may have ended the session,
            # or the fault may have cleared, between the poll and acquiring it.
            if not self._active or self._watchdog_stop.is_set():
                return
            if self._segment_cap_reached:
                return
            if not self._muxer_died() and self.frames_stale_seconds() < STALL_TIMEOUT_S:
                return

            self.stalls.append({
                "segment": self._segment,
                "reason": reason,
                "idle_seconds": round(self.frames_stale_seconds(), 3),
                "frames": self._frame_count,
            })

            if self._segment >= MAX_SEGMENTS:
                self._segment_cap_reached = True
                print(f"camera watchdog: {reason} in segment {self._segment}; "
                      f"segment cap ({MAX_SEGMENTS}) reached, giving up on "
                      f"video. The session continues and the capacitance data "
                      f"is unaffected.", flush=True)
                self._close_segment()
                return

            print(f"camera watchdog: {reason} after {self._frame_count} frames "
                  f"in segment {self._segment}; restarting recording",
                  flush=True)

            self._close_segment()
            # A stalled pipeline is not reusable: drop the camera entirely and
            # re-acquire, the same way start_session does.
            self._release_camera()
            self._configure_camera()
            self._open_segment()

    # ---------------------------------------------------------------------

    def bookmark(self, sensor_id) -> dict:
        # frame_index and pts must describe the SAME frame: the most recent one
        # the pre_callback logged. capture_metadata() would instead fetch (and
        # block until) a newer frame, whose timestamp would not match
        # frame_index, biasing the video<->trace anchor.
        #
        # _on_frame publishes (frame_index, timestamp) as one tuple, where
        # frame_index is the 0-based sidecar line the timestamp was written to, so
        # the pair read here always lines up. No frame yet means no valid anchor:
        # fail loudly rather than return frame_index -1 (which downstream reads as
        # pts_ns[-1], the LAST frame, anchoring the whole video wrong).
        # TODO: Change to report pts as relative seconds (relative to session start /
        # first_frame_SensorTimestamp) for consistency with alignment_from_bookmark()
        # and the planned .txt format change. Pending validation on Pi.
        last = self._last_frame
        if last is None:
            raise RuntimeError("no video frame captured yet; cannot bookmark")
        frame_index, timestamp = last
        return {
            "frame_index": frame_index,
            "pts": timestamp / 1e9,
            "pi_monotonic": time.monotonic(),
            # Which segment this index refers to, and how stale the frame is.
            # The desktop compares pi_monotonic - pts to spot a frozen camera;
            # frames_stale_s says the same thing without clock assumptions.
            "video_filename": self._video_path.name if self._video_path else "",
            "frames_stale_s": round(self.frames_stale_seconds(), 3),
        }

    def stop_session(self) -> list:
        self._stop_watchdog()
        try:
            self._close_segment()
        finally:
            # Always release the device even if stop_recording raised, so the
            # next session can acquire the camera.
            self._release_camera()
            self._active = False
        files = []
        for video_path, pts_path in self._segment_paths:
            for path in (video_path, pts_path):
                if path.exists():
                    files.append({"name": path.name, "size": path.stat().st_size})
            # The ffmpeg log is listed only when it has something in it, so a
            # healthy run still transfers exactly the two files it always did,
            # and a bad one carries its own diagnosis to the desktop.
            log_path = ffmpeg_output.log_path_for(video_path)
            if log_path.exists() and log_path.stat().st_size > 0:
                files.append({"name": log_path.name,
                              "size": log_path.stat().st_size})
        return files

    def _disk_free_bytes(self) -> int:
        return shutil.disk_usage(self.output_dir).free

    def reclaim_disk_space(self) -> dict:
        """Delete the oldest recordings until MIN_FREE_BYTES is free.

        Never deletes the current session's videos (every segment in
        self._segment_paths): the desktop has not fetched them yet, and losing
        the run just recorded is never worth the space. Each deleted video takes
        its companion .txt timestamp file (useless without the frames it indexes)
        and .ffmpeg.log with it.

        Returns {"deleted": [names], "low_disk": bool, "free_bytes": int};
        low_disk means even deleting every old video left < MIN_FREE_BYTES,
        so the user must free space manually before the next session.
        """
        deleted = []
        free = self._disk_free_bytes()
        keep = {video for video, _ in self._segment_paths}
        if self._video_path is not None:
            keep.add(self._video_path)
        if free < MIN_FREE_BYTES:
            candidates = sorted(
                (p for p in self.output_dir.glob("*.mp4") if p not in keep),
                key=lambda p: p.stat().st_mtime)
            for mp4 in candidates:
                if free >= MIN_FREE_BYTES:
                    break
                try:
                    mp4.unlink()
                    mp4.with_suffix(".txt").unlink(missing_ok=True)
                    ffmpeg_output.log_path_for(mp4).unlink(missing_ok=True)
                except OSError:
                    continue  # skip undeletable files; keep reclaiming
                deleted.append(mp4.name)
                free = self._disk_free_bytes()
        return {
            "deleted": deleted,
            "low_disk": free < MIN_FREE_BYTES,
            "free_bytes": free,
        }
