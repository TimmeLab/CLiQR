from false_positive_analysis import alignment_from_bookmark


def test_offset_maps_pts_to_abs():
    # Sipper inserted at Unix t=1000.0, which was video PTS=12.5.
    align = alignment_from_bookmark(start_time_abs=1000.0, video_pts=12.5)
    assert align["method"] == "bookmark"
    # A later frame at PTS=20.0 maps to abs = 20.0 + offset.
    abs_time = 20.0 + align["offset"]
    assert abs(abs_time - 1007.5) < 1e-9
