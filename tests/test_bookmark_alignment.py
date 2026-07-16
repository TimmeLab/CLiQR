from false_positive_analysis import alignment_from_bookmark, video_relative_to_abs


def test_offset_maps_pts_to_abs():
    # Sipper inserted at Unix t=1000.0, which was video PTS=12.5.
    align = alignment_from_bookmark(start_time_abs=1000.0, video_pts=12.5)
    assert align["method"] == "bookmark"
    # video_start_unix_s = 1000.0 - 12.5 = 987.5 (Unix time when video started, at PTS=0).
    # A frame at PTS=20.0 has video_relative_s = 20.0 (seconds since video start).
    # So abs_time = video_start_unix_s + video_relative_s = 987.5 + 20.0 = 1007.5.
    video_relative_s = 20.0
    abs_time = video_relative_to_abs(video_relative_s, align)
    assert abs(abs_time - 1007.5) < 1e-9


def test_alignment_from_bookmark_round_trip():
    """Test that alignment_from_bookmark produces a dict compatible with video_relative_to_abs."""
    # Sipper inserted at Unix t=1000.0, which was video PTS=12.5.
    align = alignment_from_bookmark(start_time_abs=1000.0, video_pts=12.5)

    # Check that the alignment has the required keys.
    assert 'video_start_unix_s' in align
    assert 'drift_corrected' in align
    assert 'drift_s' in align
    assert align['method'] == 'bookmark'

    # Verify video_start_unix_s is computed correctly.
    assert abs(align['video_start_unix_s'] - (1000.0 - 12.5)) < 1e-9

    # Round-trip: convert video-relative timestamp through the alignment.
    # At video PTS=12.5 (the bookmark frame), video_relative_s = 0.0 (start of video).
    abs_time_at_start = video_relative_to_abs(0.0, align)
    assert abs(abs_time_at_start - 987.5) < 1e-9  # 1000.0 - 12.5

    # At video PTS=20.0, video_relative_s = 20.0 - 12.5 = 7.5.
    abs_time_at_pts20 = video_relative_to_abs(7.5, align)
    assert abs(abs_time_at_pts20 - 995.0) < 1e-9  # 987.5 + 7.5


def test_latency_bracket_shifts_video_start_later():
    """The bookmarked frame was captured at the END of the round-trip (~host_after),
    ~L after start_time, so the video panel leads the trace by L. Feeding the host
    bracket must push video_start_unix_s later by L (== host_after - start_time)."""
    # Round-trip: before=1000.0, after=1005.0 -> L = host_after - start = 5.0 s.
    align = alignment_from_bookmark(
        start_time_abs=1000.0, video_pts=12.5,
        host_before=1000.0, host_after=1005.0)
    # video_start = (start_time + L) - video_pts = (1000 + 5.0) - 12.5 = 992.5
    assert abs(align["video_start_unix_s"] - 992.5) < 1e-9
    assert abs(align["bookmark_latency_s"] - 5.0) < 1e-9


def test_latency_subtracts_pi_capture_gap():
    """pi_monotonic (bookmark exec) minus video_pts (grabbed frame), both on the
    SensorTimestamp clock, backs the capture->exec gap off host_after."""
    align = alignment_from_bookmark(
        start_time_abs=1000.0, video_pts=12.5,
        host_before=1000.0, host_after=1005.0, pi_monotonic=12.6)
    # L = (host_after - start) - (pi_monotonic - video_pts) = 5.0 - 0.1 = 4.9
    assert abs(align["bookmark_latency_s"] - 4.9) < 1e-9
    assert abs(align["video_start_unix_s"] - (1000.0 + 4.9 - 12.5)) < 1e-9


def test_missing_bracket_leaves_latency_zero():
    """Older recordings without the host bracket: L defaults to 0, unchanged."""
    align = alignment_from_bookmark(start_time_abs=1000.0, video_pts=12.5)
    assert abs(align["video_start_unix_s"] - 987.5) < 1e-9
    assert align["bookmark_latency_s"] == 0.0
