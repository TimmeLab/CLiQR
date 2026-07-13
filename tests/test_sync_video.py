import numpy as np
import pytest

from video.sync_video import (
    frame_abs_times,
    compute_trim_frames,
    trim_video,
    render_synced_video,
)


def test_frame_abs_times_anchors_bookmark_to_start():
    # 5 frames, 0.1 s apart; bookmark is frame 2 (pts in seconds = its offset/1e9).
    offsets_ns = np.array([0, 1e8, 2e8, 3e8, 4e8])
    bookmark_pts = 2e8 / 1e9  # frame 2's pts in seconds
    abs_t = frame_abs_times(offsets_ns, bookmark_pts, start_time_abs=1000.0)
    assert abs(abs_t[2] - 1000.0) < 1e-9          # bookmark frame == start_time
    assert abs(abs_t[4] - 1000.2) < 1e-9          # 0.2 s later


def test_compute_trim_frames():
    offsets_ns = np.array([0, 1e8, 2e8, 3e8, 4e8])
    bookmark_pts = 2e8 / 1e9
    start, stop = compute_trim_frames(offsets_ns, bookmark_pts,
                                      start_time_abs=1000.0, stop_time_abs=1000.15)
    assert start == 2          # bookmark frame
    assert stop == 3           # last frame within window (1000.1 <= 1000.15)


@pytest.fixture
def tiny_video(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "src.mp4"
    writer = imageio.get_writer(str(path), fps=10)
    for i in range(10):
        frame = np.full((32, 32, 3), i * 20, dtype=np.uint8)
        writer.append_data(frame)
    writer.close()
    return path


def test_trim_video(tiny_video, tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    out = tmp_path / "trim.mp4"
    trim_video(str(tiny_video), str(out), start_frame=2, stop_frame=5)
    assert out.exists()
    reader = imageio.get_reader(str(out))
    n = sum(1 for _ in reader)
    reader.close()
    assert n == 4          # frames 2,3,4,5 inclusive


def test_render_synced_video_smoke(tiny_video, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("matplotlib")
    offsets_ns = np.arange(10) * 1e8            # 0.1 s apart, 10 frames
    bookmark_pts = 0.0                          # frame 0 is the bookmark
    start_time_abs = 1000.0
    stop_time_abs = 1000.9
    cap_time = np.linspace(999.5, 1001.5, 400)
    cap_data = np.sin(cap_time * 30.0)
    lick_times = np.array([1000.2, 1000.6])
    out = tmp_path / "synced.mp4"
    render_synced_video(
        str(tiny_video), offsets_ns, cap_time, cap_data, lick_times,
        bookmark_pts, start_time_abs, stop_time_abs, str(out),
        window_sec=1.0,
    )
    assert out.exists()
    assert out.stat().st_size > 0
