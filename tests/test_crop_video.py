import os

import numpy as np
import pytest

import crop_video as cv
from video.trimcrop import VideoAnchor


def _anchor(frame_index=2, start=110.0, stop=110.5,
            host_before=None, host_after=None):
    return VideoAnchor(
        sensor_number=1, video_filename="v.mp4", video_frame_index=frame_index,
        start_time=start, stop_time=stop,
        host_before=host_before, host_after=host_after,
    )


def test_compute_crop_window_spans_the_session():
    # frames every 0.1 s over 1.0 s; bookmark frame 2 -> session zero at 0.2 s
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    sf, ef, start_sec, end_sec = cv.compute_crop_window(
        _anchor(frame_index=2, start=110.0, stop=110.5), pts_ns)
    assert (sf, ef) == (2, 7)              # session [0, 0.5] s
    assert start_sec == pytest.approx(0.2)  # video-file seconds
    assert end_sec == pytest.approx(0.7 + 0.3)  # tail margin


def test_compute_crop_window_applies_bookmark_latency():
    """The bookmarked frame was captured mid-round-trip, so raw frame session
    times run early (the video leads the trace). Correcting for it shifts every
    frame's session label LATER, so the same session window resolves to EARLIER
    frames in the file.

    The bracket is chosen so the latency is exactly representable in binary
    (0.25). Values like 0.2 are not: (110.1+110.3)/2 - 110.0 evaluates to
    0.19999999999998863, which leaves video_base_eff a hair ABOVE zero, and the
    `sess >= start` test in compute_trim_frames then drops frame 0 on an
    epsilon. That knife-edge would make this test assert floating-point noise
    rather than the behavior it is here to pin down.
    """
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    plain = cv.compute_crop_window(_anchor(frame_index=2, start=110.0, stop=110.3), pts_ns)
    assert plain[0] == 2 and plain[2] == pytest.approx(0.2)
    # bracket midpoint 0.25 s after start_time -> latency 0.25 (exact)
    shifted = cv.compute_crop_window(
        _anchor(frame_index=2, start=110.0, stop=110.3,
                host_before=110.0, host_after=110.5), pts_ns)
    assert shifted[0] == 0
    assert shifted[0] < plain[0]           # earlier start frame
    assert shifted[2] < plain[2]           # earlier start second


def test_compute_crop_window_empty_raises():
    """A reversed window (stop_time before start_time — corrupt h5) is the only
    way this raises, so it is what we test.

    For any sane recording the window CANNOT be empty: the bookmarked frame sits
    at session time == latency by construction, which is inside [0, duration]
    unless the latency exceeds the whole session. The guard is therefore about
    propagating compute_trim_frames' error on degenerate data, not a case real
    recordings reach.
    """
    pts_ns = (np.arange(0, 3) * 100_000_000).astype(np.int64)
    with pytest.raises(ValueError):
        cv.compute_crop_window(_anchor(frame_index=0, start=900.0, stop=500.0), pts_ns)


def test_resolve_out_path_default(tmp_path):
    v = str(tmp_path / "v.mp4")
    assert cv.resolve_out_path(v, None, False) == str(tmp_path / "v_cropped.mp4")


def test_resolve_out_path_explicit(tmp_path):
    v = str(tmp_path / "v.mp4")
    assert cv.resolve_out_path(v, "/o/x.mp4", False) == "/o/x.mp4"


def test_resolve_out_path_existing_raises(tmp_path):
    v = str(tmp_path / "v.mp4")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    with pytest.raises(ValueError, match="--force"):
        cv.resolve_out_path(v, None, False)


def test_resolve_out_path_existing_with_force(tmp_path):
    v = str(tmp_path / "v.mp4")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    assert cv.resolve_out_path(v, None, True) == str(tmp_path / "v_cropped.mp4")


def test_reject_cropped_input():
    with pytest.raises(ValueError, match="already-cropped"):
        cv.reject_cropped_input("/d/v_cropped.mp4")
    cv.reject_cropped_input("/d/v.mp4")  # no raise


def test_build_arg_parser_defaults():
    args = cv.build_arg_parser().parse_args(["--h5", "r.h5"])
    assert args.h5 == "r.h5"
    assert args.size == 360
    assert args.video is None and args.pts_txt is None
    assert args.out is None and args.force is False
