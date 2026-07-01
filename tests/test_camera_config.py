"""The Pi camera must be configured for its fast 1536x864p120 mode.

camera_backend builds the picamera2 config from a pure helper so the intended
resolution and frame rate are testable off-hardware (picamera2 itself is
imported lazily only on the Pi).
"""
from pi.camera_backend import video_config_kwargs, TARGET_FPS


def test_target_fps_is_120():
    assert TARGET_FPS == 120


def test_config_locks_frame_duration_for_120fps():
    kwargs = video_config_kwargs()
    # 1e6 / 120 == 8333.33us; both limits equal -> fixed frame rate.
    lo, hi = kwargs["controls"]["FrameDurationLimits"]
    assert lo == hi == 8333


def test_config_uses_full_fast_mode_resolution():
    kwargs = video_config_kwargs()
    assert kwargs["main"]["size"] == (1536, 864)
    assert kwargs["raw"]["size"] == (1536, 864)
