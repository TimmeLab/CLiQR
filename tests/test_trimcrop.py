import numpy as np
import pytest
import h5py

from video import trimcrop as tc


def _clock(pts_start_sec, latency=0.0, slope=1.0):
    """A SessionClock with an explicit anchor. video_base==pts_start_sec at
    latency 0; a former ``video_base - L`` is now ``latency=L``."""
    return tc.SessionClock(pts_start_sec=pts_start_sec, latency=latency, slope=slope)


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
    clock = _clock(pts_start_sec=float(pts_ns[2]) / 1e9)  # bookmark frame 2
    sess = tc.frame_session_times(clock, pts_ns)
    assert sess[2] == pytest.approx(0.0)
    assert sess[0] == pytest.approx(-0.2)
    sf, ef = tc.compute_trim_frames(clock, pts_ns, 0.0, 0.5)
    assert sf == 2 and ef == 7


def test_compute_trim_frames_window_edges_inclusive():
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    # pts_start 0 -> session time == file time; [0.2, 0.5] covers frames 2..5
    sf, ef = tc.compute_trim_frames(_clock(0.0), pts_ns, 0.2, 0.5)
    assert sf == 2 and ef == 5


def test_compute_trim_frames_empty_window_raises():
    pts_ns = (np.arange(0, 5) * 100_000_000).astype(np.int64)
    with pytest.raises(ValueError):
        tc.compute_trim_frames(_clock(0.0), pts_ns, 100.0, 200.0)


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


def test_read_video_anchor_multi_cycle_uses_matching_cycle(tmp_path):
    # A 2-cycle recording: the window comes from cycle 1 (start_time1/stop_time1),
    # so the frame index and host bracket must come from cycle 1 too, not the
    # unsuffixed cycle-0 datasets.
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 300.0]),
        "video_filename": b"vid.mp4",
        "video_frame_index": 42,
        "start_time": 110.0, "stop_time": 175.0,
        "video_bookmark_host_before": 111.0, "video_bookmark_host_after": 111.4,
        "video_filename1": b"vid.mp4",
        "video_frame_index1": 900,
        "start_time1": 210.0, "stop_time1": 260.0,
        "video_bookmark_host_before1": 210.5, "video_bookmark_host_after1": 210.9,
    }})
    a = tc.read_video_anchor(str(p))
    assert a.video_frame_index == 900
    assert a.session_duration == pytest.approx(50.0)  # 260 - 210
    assert a.latency == pytest.approx(0.7)  # (210.5 + 210.9)/2 - 210.0


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


def test_session_clock_slope1_is_offset_only():
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)  # 0.0 .. 1.0 s
    clock = tc.SessionClock(pts_start_sec=float(pts_ns[2]) / 1e9,
                            latency=0.0, slope=1.0)
    sess = clock.session_time(pts_ns / 1e9)
    assert sess[2] == pytest.approx(0.0)   # bookmark frame -> session 0
    assert sess[0] == pytest.approx(-0.2)


def test_drift_slope_recovers_known_skew():
    # Video clock runs slightly slow vs host -> slope = host-s per video-s.
    slope_true = 1.0 / 1.001
    pts_ns = (np.arange(0, 1000) * 1_000_000).astype(np.int64)  # 0..0.999 s
    anchor = tc.VideoAnchor(
        sensor_number=1, video_filename="v.mp4", video_frame_index=0,
        start_time=1000.0, stop_time=1001.0,
        host_before=1000.0, host_after=1000.0,            # mid_start = 1000.0
        stop_frame_index=999,
        stop_host_before=1000.0 + 0.999 * slope_true,
        stop_host_after=1000.0 + 0.999 * slope_true)      # mid_stop
    assert anchor.drift_slope(pts_ns) == pytest.approx(slope_true, rel=1e-6)


def test_drift_slope_defaults_to_one_without_stop():
    pts_ns = (np.arange(0, 10) * 1_000_000).astype(np.int64)
    anchor = tc.VideoAnchor(
        sensor_number=1, video_filename="v.mp4", video_frame_index=0,
        start_time=1000.0, stop_time=1001.0,
        host_before=1000.0, host_after=1000.2)
    assert anchor.drift_slope(pts_ns) == 1.0


def test_session_clock_builder_reads_anchor():
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    anchor = tc.VideoAnchor(
        sensor_number=1, video_filename="v.mp4", video_frame_index=2,
        start_time=110.0, stop_time=175.0,
        host_before=110.1, host_after=110.3)  # latency 0.2, no stop -> slope 1
    clock = tc.session_clock(anchor, pts_ns)
    assert clock.pts_start_sec == pytest.approx(0.2)
    assert clock.latency == pytest.approx(0.2)
    assert clock.slope == 1.0


def test_read_video_anchor_reads_stop_bookmark(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "video_filename": b"vid.mp4", "video_frame_index": 3,
        "start_time": 110.0, "stop_time": 175.0,
        "video_bookmark_host_before": 111.0, "video_bookmark_host_after": 111.4,
        "video_stop_frame_index": 900,
        "video_stop_bookmark_host_before": 176.0,
        "video_stop_bookmark_host_after": 176.4,
    }})
    a = tc.read_video_anchor(str(p))
    assert a.stop_frame_index == 900
    assert a.stop_host_before == pytest.approx(176.0)
    assert a.stop_host_after == pytest.approx(176.4)


def test_read_video_anchor_stop_fields_none_when_absent(tmp_path):
    p = tmp_path / "r.h5"
    _write_sensor(p, {"sensor_1": {
        "time_data": np.array([100.0, 200.0]),
        "video_filename": b"vid.mp4", "video_frame_index": 3,
        "start_time": 110.0, "stop_time": 175.0,
    }})
    a = tc.read_video_anchor(str(p))
    assert a.stop_frame_index is None
    assert a.stop_host_before is None
    assert a.stop_host_after is None


import types


def _fake_run(calls, stdout="", returncode=0):
    def run(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="boom")
    return run


def test_probe_start_pts(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, stdout="30.000000\n"))
    assert tc.probe_start_pts("in.mp4") == pytest.approx(30.0)
    assert "start_time" in " ".join(calls[0])


def test_probe_start_pts_missing_is_zero(monkeypatch):
    # ffprobe prints "N/A" for containers without a start_time
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, stdout="N/A\n"))
    assert tc.probe_start_pts("in.mp4") == 0.0


def test_probe_start_pts_failure_raises(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, returncode=1))
    with pytest.raises(RuntimeError):
        tc.probe_start_pts("in.mp4")


def test_trim_and_crop_builds_argv(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls))
    tc.trim_and_crop("in.mp4", 100.0, 110.0, "out.mp4", 452, 180, 360)
    cmd = calls[0]
    assert "crop=360:360:452:180" in cmd
    assert "-copyts" in cmd
    # trim_and_crop reads the original video (start_time 0) -> plain margin seek
    assert cmd[cmd.index("-ss") + 1] == "95.000000"
    assert cmd[cmd.index("-to") + 1] == "110.000000"
    assert "libx264" in cmd  # re-encode: a filter is applied


def test_trim_and_crop_seek_floors_at_zero(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls))
    tc.trim_and_crop("in.mp4", 2.0, 8.0, "out.mp4", 0, 0, 360)
    assert calls[0][calls[0].index("-ss") + 1] == "0.000000"


def test_trim_and_crop_failure_raises(monkeypatch):
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, returncode=1))
    with pytest.raises(RuntimeError):
        tc.trim_and_crop("in.mp4", 100.0, 110.0, "out.mp4", 0, 0, 360)


def test_subclip_copy_builds_argv(monkeypatch):
    calls = []
    # first call is probe_start_pts, second is the ffmpeg subclip
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, stdout="0.000000\n"))
    tc.subclip_copy("in.mp4", 100.0, 110.0, "out.mp4")
    cmd = calls[-1]
    assert cmd[cmd.index("-c") + 1] == "copy"   # stream copy, no re-encode
    assert "-copyts" in cmd
    assert "libx264" not in cmd
    assert not any(str(a).startswith("crop=") for a in cmd)
    assert cmd[cmd.index("-ss") + 1] == "95.000000"
    assert cmd[cmd.index("-to") + 1] == "110.000000"


def test_subclip_copy_seek_is_file_relative(monkeypatch):
    """Input -ss is relative to the container start_time, but -to under -copyts is
    absolute. On a cropped file whose PTS start at 30 s, seeking to original-timeline
    second 100 means -ss 65 (=100-30-5), while -to stays at 110."""
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, stdout="30.000000\n"))
    tc.subclip_copy("cropped.mp4", 100.0, 110.0, "out.mp4")
    cmd = calls[-1]
    assert cmd[cmd.index("-ss") + 1] == "65.000000"
    assert cmd[cmd.index("-to") + 1] == "110.000000"


import os


def _anchor(video_filename="v.mp4"):
    return tc.VideoAnchor(
        sensor_number=1, video_filename=video_filename, video_frame_index=3,
        start_time=110.0, stop_time=175.0, host_before=None, host_after=None,
    )


def test_cropped_path_for():
    assert tc.cropped_path_for("/d/v.mp4") == "/d/v_cropped.mp4"


def test_resolve_paths_plain(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    video, pts = tc.resolve_paths(str(tmp_path / "r.h5"), _anchor())
    assert video == str(tmp_path / "v.mp4")
    assert pts == str(tmp_path / "v.txt")


def test_resolve_paths_prefers_cropped_when_present(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    video, pts = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), prefer_cropped=True)
    assert video == str(tmp_path / "v_cropped.mp4")
    # THE TRAP: the sidecar belongs to the ORIGINAL video and must not follow
    # the cropped name. The cropped file has no sidecar and needs none.
    assert pts == str(tmp_path / "v.txt")


def test_resolve_paths_falls_back_when_no_cropped(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    video, pts = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), prefer_cropped=True)
    assert video == str(tmp_path / "v.mp4")
    assert pts == str(tmp_path / "v.txt")


def test_resolve_paths_ignores_cropped_when_not_preferred(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    video, _ = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), prefer_cropped=False)
    assert video == str(tmp_path / "v.mp4")


def test_resolve_paths_explicit_overrides(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    (tmp_path / "v_cropped.mp4").write_bytes(b"")
    video, pts = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), video="/elsewhere/x.mp4",
        prefer_cropped=True)
    assert video == "/elsewhere/x.mp4"
    # sidecar still derives from the h5's video_filename, not from --video
    assert pts == str(tmp_path / "v.txt")


def test_resolve_paths_explicit_pts_overrides(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"")
    _, pts = tc.resolve_paths(
        str(tmp_path / "r.h5"), _anchor(), pts_txt="/elsewhere/x.txt")
    assert pts == "/elsewhere/x.txt"


def test_trim_window_seconds():
    # frames every 0.1 s; bookmark frame 2 -> session zero at 0.2 s
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    clock = _clock(pts_start_sec=float(pts_ns[2]) / 1e9)  # 0.2
    sf, ef, start_sec, end_sec = tc.trim_window_seconds(clock, pts_ns, 0.0, 0.3)
    assert (sf, ef) == (2, 5)
    assert start_sec == pytest.approx(0.2)          # original-timeline seconds
    assert end_sec == pytest.approx(0.5 + tc.TAIL_MARGIN)


def test_trim_window_seconds_honors_the_anchor():
    """A larger latency labels every frame later, so the same session window
    resolves to earlier frames and earlier video seconds."""
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    ps = float(pts_ns[2]) / 1e9
    plain = tc.trim_window_seconds(_clock(ps, latency=0.0), pts_ns, 0.0, 0.3)
    shifted = tc.trim_window_seconds(_clock(ps, latency=0.25), pts_ns, 0.0, 0.3)
    assert shifted[0] < plain[0]
    assert shifted[2] < plain[2]


def test_trim_window_seconds_empty_raises():
    pts_ns = (np.arange(0, 3) * 100_000_000).astype(np.int64)
    with pytest.raises(ValueError):
        tc.trim_window_seconds(_clock(0.0), pts_ns, 0.0, -400.0)


def test_frame_session_times_scales_with_slope():
    # A non-unit slope scales video-elapsed-since-bookmark; the bookmark frame
    # itself stays at τ=latency.
    pts_ns = (np.arange(0, 600) * 1_000_000).astype(np.int64)  # 0 .. 0.599 s
    clock = tc.SessionClock(pts_start_sec=0.1, latency=0.05, slope=1.002)
    sess = tc.frame_session_times(clock, pts_ns)
    assert sess[100] == pytest.approx(0.05)  # frame at pts 0.1 == pts_start
    assert sess[300] == pytest.approx(0.05 + 1.002 * (0.3 - 0.1))


def test_probe_frame_session_times_applies_same_clock(monkeypatch):
    # ffprobe path must use the identical transform as the sidecar path.
    calls = []
    monkeypatch.setattr(tc.subprocess, "run", _fake_run(calls, stdout="0.3\n"))
    clock = tc.SessionClock(pts_start_sec=0.1, latency=0.05, slope=1.002)
    got = tc.probe_frame_session_times("clip.mp4", clock)
    assert got[0] == pytest.approx(0.05 + 1.002 * (0.3 - 0.1))
