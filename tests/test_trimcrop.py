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
