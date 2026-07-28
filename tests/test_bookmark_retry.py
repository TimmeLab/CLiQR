"""A single failed bookmark must not cost the session its video alignment.

2026-07-22, 11:48:39:

    WARNING: Sensor 9: bookmark failed: [WinError 10061] No connection could be
    made because the target machine actively refused it

The Pi server had accepted START_SESSION and was gone by the time BOOKMARK went
out. The Start bookmark is the ONLY thing tying frame numbers to session time,
so that one refused round-trip left the whole session's video unalignable — and
it was reported as a single quiet WARNING among many.
"""
import pytest

from components import sensor_card
from components.sensor_card import (BOOKMARK_ATTEMPTS, _bookmark_with_retry,
                                    _report_bookmark_failure)
from utils import state
from video import protocol


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sensor_card.time_module, "sleep", lambda s: None)


@pytest.fixture
def log(monkeypatch):
    messages = []
    monkeypatch.setattr(state, "add_log_message", messages.append)
    return messages


class _Client:
    """Fails the first `failures` bookmarks, then succeeds."""

    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def bookmark(self, sensor_id):
        self.calls += 1
        if self.calls <= self.failures:
            return protocol.make_error(
                "[WinError 10061] No connection could be made because the "
                "target machine actively refused it")
        return protocol.make_ok(frame_index=42, pts=1.5, pi_monotonic=1.51)


def test_first_attempt_succeeding_makes_no_extra_calls(log):
    client = _Client(failures=0)
    resp, before, after = _bookmark_with_retry(client, 9, "start")

    assert resp["ok"] is True
    assert client.calls == 1
    assert log == [], "a clean bookmark must stay quiet"


def test_transient_refusal_is_retried_and_recovers(log):
    client = _Client(failures=1)
    resp, before, after = _bookmark_with_retry(client, 9, "start")

    assert resp["ok"] is True
    assert resp["frame_index"] == 42
    assert client.calls == 2
    assert any("retrying" in m for m in log)
    assert any("succeeded on attempt 2" in m for m in log)


def test_host_brackets_belong_to_the_successful_attempt(log, monkeypatch):
    # The latency correction works back from host_after to the bookmarked
    # frame's true host time. Carrying the first attempt's bracket across a
    # retry would bias the video<->trace anchor by the whole retry delay.
    #
    # monkeypatch, not direct assignment: sensor_card.time_module IS the stdlib
    # time module, so a hand-rolled save/restore would capture the already
    # patched function and leak a fake clock into every later test.
    stamps = iter([100.0, 100.1, 200.0, 200.1])
    monkeypatch.setattr(sensor_card.time_module, "time", lambda: next(stamps))

    client = _Client(failures=1)
    resp, before, after = _bookmark_with_retry(client, 9, "start")

    assert resp["ok"] is True
    assert (before, after) == (200.0, 200.1)


def test_gives_up_after_the_attempt_limit(log):
    client = _Client(failures=99)
    resp, before, after = _bookmark_with_retry(client, 9, "start")

    assert resp["ok"] is False
    assert client.calls == BOOKMARK_ATTEMPTS


def test_failed_start_bookmark_says_the_video_is_unalignable(log):
    _report_bookmark_failure(9, {"error": "[WinError 10061] ... refused it"},
                             "start")

    assert any(m.startswith("ERROR:") and "FAILED" in m for m in log)
    assert any("CANNOT be aligned" in m for m in log)


def test_failed_stop_bookmark_does_not_claim_the_video_is_unalignable(log):
    # Losing the Stop bookmark only costs the drift-slope refinement; the Start
    # anchor still aligns the video.
    _report_bookmark_failure(9, {"error": "boom"}, "stop")

    assert any("FAILED" in m for m in log)
    assert not any("CANNOT be aligned" in m for m in log)
