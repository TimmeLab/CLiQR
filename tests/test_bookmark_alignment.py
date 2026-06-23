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
