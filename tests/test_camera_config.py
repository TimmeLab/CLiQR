"""The Pi camera captures in its fast 1536x864p120 sensor mode but records a
downscaled 1280x720 stream so the Pi 5 software H.264 encoder can keep up.

camera_backend builds the picamera2 config from a pure helper so the intended
resolution and frame rate are testable off-hardware (picamera2 itself is
imported lazily only on the Pi).
"""
from pi.camera_backend import (
    video_config_kwargs, encoder_kwargs, TARGET_FPS, BITRATE)


def test_target_fps_is_120():
    assert TARGET_FPS == 120


def test_config_locks_frame_duration_for_120fps():
    kwargs = video_config_kwargs()
    # 1e6 / 120 == 8333.33us; both limits equal -> fixed frame rate.
    lo, hi = kwargs["controls"]["FrameDurationLimits"]
    assert lo == hi == 8333


def test_raw_stream_pins_fast_sensor_mode():
    # The 1536x864 raw stream forces the sensor's 120 fps fast readout mode.
    kwargs = video_config_kwargs()
    assert kwargs["raw"]["size"] == (1536, 864)


def test_main_stream_downscaled_for_encoder_budget():
    # Encoded stream is 1280x720 so software H.264 stays drop-free at 120 fps.
    kwargs = video_config_kwargs()
    assert kwargs["main"]["size"] == (1280, 720)


def test_bitrate_capped_for_drop_free_encode():
    assert BITRATE == 3_000_000


def test_encoder_told_the_real_framerate():
    # Pi 5's H264Encoder is libav software encoding whose framerate defaults
    # to 30. x264 rate control budgets bitrate/framerate bits PER FRAME, so a
    # 30 fps default fed 120 real fps overshoots 4x (12 Mb/s: a 2 h run
    # ballooned past 9 GB and filled the disk). The encoder must be told the
    # true frame rate so `bitrate` means what it says.
    kwargs = encoder_kwargs()
    assert kwargs["framerate"] == TARGET_FPS
    assert kwargs["bitrate"] == BITRATE
