"""A bookmark describing a long-stale frame must warn, not pass silently.

On 2026-07-21 the Stop bookmark returned pi_monotonic 1146289.505 against pts
1140875.064 — the frame it described was 5414 s old because the camera had
frozen 90 min earlier. Every layer reported success and the operator found out
the next day. The healthy Start bookmark from the same run measured 0.017 s.
"""
import pytest

from components import session_controls
from components.sensor_card import VIDEO_STALL_WARN_S, _warn_if_video_frozen
from utils import state


@pytest.fixture
def log(monkeypatch):
    messages = []
    monkeypatch.setattr(state, "add_log_message", messages.append)
    return messages


def test_healthy_bookmark_does_not_warn(log):
    # Real values from the 2026-07-21 Start bookmark: a 17 ms capture->exec gap.
    _warn_if_video_frozen(7, {"pi_monotonic": 1138571.007523709,
                              "pts": 1138570.990607}, "start")
    assert log == []


def test_frozen_camera_warns_from_the_clock_gap(log):
    # Real values from the 2026-07-21 Stop bookmark.
    _warn_if_video_frozen(7, {"pi_monotonic": 1146289.50493759,
                              "pts": 1140875.064128}, "stop")
    assert len(log) == 1
    assert "WARNING" in log[0]
    assert "5414s stale" in log[0]


def test_explicit_staleness_field_is_preferred(log):
    # A server that reports frames_stale_s needs no clock-epoch assumptions.
    _warn_if_video_frozen(7, {"frames_stale_s": 5414.4,
                              "pi_monotonic": 1.0, "pts": 0.9}, "stop")
    assert len(log) == 1
    assert "5414s stale" in log[0]


def test_threshold_is_far_above_healthy_and_far_below_a_real_stall():
    # Healthy gap 0.017 s, observed stall 5414 s. 5 s is ~250x healthy.
    assert 0.017 < VIDEO_STALL_WARN_S < 60.0


def test_bookmark_without_timing_fields_is_ignored(log):
    _warn_if_video_frozen(7, {"frame_index": 3}, "start")
    assert log == []


def test_stop_reply_stalls_are_surfaced(log, monkeypatch):
    monkeypatch.setattr(state.camera_stall_warning, "set",
                        lambda v: setattr(state.camera_stall_warning, "_test", v))
    session_controls._report_camera_stalls(
        {"ok": True, "stalls": [{"segment": 1, "idle_seconds": 3.2, "frames": 317292}]})
    assert len(log) == 1
    assert "317292 frames" in log[0]
    assert "restarted" in log[0]
    assert "_part2" in state.camera_stall_warning._test


def test_clean_stop_reply_says_nothing(log):
    session_controls._report_camera_stalls({"ok": True, "stalls": []})
    session_controls._report_camera_stalls({"ok": True})
    assert log == []
