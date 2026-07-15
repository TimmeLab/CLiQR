import numpy as np
import pytest
import h5py

from video import trimcrop as tc


def test_compute_video_base():
    pts_ns = np.array([1_000_000_000, 1_100_000_000, 1_250_000_000], dtype=np.int64)
    assert tc.compute_video_base(pts_ns, 2) == pytest.approx(0.25)


def test_bookmark_latency_from_bracket():
    assert tc.bookmark_latency(1000.0, 1000.2, 998.0) == pytest.approx(2.1)
    assert tc.bookmark_latency(999.9, 1000.1, 1000.0) == pytest.approx(0.0)


def test_bookmark_latency_missing_is_zero():
    assert tc.bookmark_latency(None, None, 1000.0) == 0.0
    assert tc.bookmark_latency(1000.0, None, 998.0) == 0.0


def test_frame_session_times_and_trim_frames():
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)  # 0.0 .. 1.0 s
    vb = tc.compute_video_base(pts_ns, 2)  # 0.2 s
    sess = tc.frame_session_times(pts_ns, vb)
    assert sess[2] == pytest.approx(0.0)
    assert sess[0] == pytest.approx(-0.2)
    sf, ef = tc.compute_trim_frames(pts_ns, vb, 0.0, 0.5)
    assert sf == 2 and ef == 7


def test_compute_trim_frames_window_edges_inclusive():
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    # video_base 0 -> session time == file time; [0.2, 0.5] covers frames 2..5
    sf, ef = tc.compute_trim_frames(pts_ns, 0.0, 0.2, 0.5)
    assert sf == 2 and ef == 5


def test_compute_trim_frames_empty_window_raises():
    pts_ns = (np.arange(0, 5) * 100_000_000).astype(np.int64)
    with pytest.raises(ValueError):
        tc.compute_trim_frames(pts_ns, 0.0, 100.0, 200.0)


def _write_sensor(path, groups):
    """groups: {sensor_name: {dataset: value}} under one board."""
    with h5py.File(path, "w") as f:
        for name, datasets in groups.items():
            g = f.create_group(f"board_FT232H0/{name}")
            for k, v in datasets.items():
                g[k] = v


def test_resolve_start_stop_picks_highest_numbered(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "start_time": 110.0, "stop_time": 120.0,
        "start_time1": 130.0, "stop_time1": 140.0,
    }})
    with h5py.File(p, "r") as f:
        assert tc._resolve_start_stop(f["board_FT232H0/sensor_1"]) == (130.0, 140.0)


def test_resolve_start_stop_unnumbered_pair(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "start_time": 110.0, "stop_time": 120.0,
    }})
    with h5py.File(p, "r") as f:
        assert tc._resolve_start_stop(f["board_FT232H0/sensor_1"]) == (110.0, 120.0)


def test_resolve_start_stop_no_pair_falls_back_to_time_data(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {"time_data": np.array([100.0, 150.0, 200.0])}})
    with h5py.File(p, "r") as f:
        assert tc._resolve_start_stop(f["board_FT232H0/sensor_1"]) == (100.0, 200.0)


def test_find_video_sensor(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {
        "sensor_0": {"time_data": np.array([1.0])},
        "sensor_2": {"video_filename": b"vid.mp4", "video_frame_index": 7},
    })
    with h5py.File(p, "r") as f:
        board_id, name, num = tc.find_video_sensor(f)
    assert board_id == "board_FT232H0"
    assert name == "sensor_2"
    assert num == 2


def test_find_video_sensor_none_raises(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_0": {"time_data": np.array([1.0])}})
    with h5py.File(p, "r") as f:
        with pytest.raises(ValueError):
            tc.find_video_sensor(f)


def test_read_session_window(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "video_filename": b"vid.mp4", "video_frame_index": 3,
        "start_time": 110.0, "stop_time": 175.0,
    }})
    assert tc.read_session_window(str(p)) == (110.0, 175.0)


def test_clamp_origin_interior_unchanged():
    assert tc.clamp_origin(452, 180, 1280, 720, 360) == (452, 180)


def test_clamp_origin_rounds_down_to_even():
    # yuv420p needs even offsets
    assert tc.clamp_origin(451, 181, 1280, 720, 360) == (450, 180)


def test_clamp_origin_clamps_each_edge():
    assert tc.clamp_origin(-50, -50, 1280, 720, 360) == (0, 0)
    # right/bottom clamp to frame - size, which stays inside
    assert tc.clamp_origin(9999, 9999, 1280, 720, 360) == (920, 360)


def test_clamp_origin_clamped_edge_stays_even():
    # frame_h - size = 365 is odd -> must round down, not out of frame
    assert tc.clamp_origin(9999, 9999, 1280, 725, 360) == (920, 364)


def test_clamp_origin_size_exceeding_frame_raises():
    with pytest.raises(ValueError):
        tc.clamp_origin(0, 0, 320, 720, 360)
    with pytest.raises(ValueError):
        tc.clamp_origin(0, 0, 1280, 200, 360)


def test_read_video_anchor(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {
        "sensor_0": {"time_data": np.array([1.0])},
        "sensor_1": {
            "time_data": np.array([100.0, 200.0]),
            "video_filename": b"vid.mp4",
            "video_frame_index": 42,
            "start_time": 110.0,
            "stop_time": 175.0,
            "video_bookmark_host_before": 111.0,
            "video_bookmark_host_after": 111.4,
        },
    })
    a = tc.read_video_anchor(str(p))
    assert a.sensor_number == 1
    assert a.video_filename == "vid.mp4"   # decoded, not bytes
    assert a.video_frame_index == 42
    assert a.session_duration == pytest.approx(65.0)
    assert a.latency == pytest.approx(1.2)  # (111.0 + 111.4)/2 - 110.0


def test_read_video_anchor_without_host_bracket(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "video_filename": b"vid.mp4",
        "video_frame_index": 3,
        "start_time": 110.0, "stop_time": 175.0,
    }})
    a = tc.read_video_anchor(str(p))
    assert a.host_before is None and a.host_after is None
    assert a.latency == 0.0
