import numpy as np
import pytest

import make_sync_video as msv


def test_compute_video_base():
    pts_ns = np.array([1_000_000_000, 1_100_000_000, 1_250_000_000], dtype=np.int64)
    # (1_250_000_000 - 1_000_000_000) / 1e9 = 0.25
    assert msv.compute_video_base(pts_ns, 2) == pytest.approx(0.25)


def test_video_sec_at_session_start():
    # tau=0 -> video_base + (start_time - t0_raw)
    got = msv.video_sec(0.0, video_base=32.0, start_time=127.0, t0_raw=100.0)
    assert got == pytest.approx(32.0 + 27.0)


def test_video_sec_linear_and_offset():
    base = dict(video_base=32.0, start_time=127.0, t0_raw=100.0)
    assert msv.video_sec(5.0, **base) - msv.video_sec(0.0, **base) == pytest.approx(5.0)
    assert msv.video_sec(0.0, sync_offset=2.0, **base) - msv.video_sec(0.0, **base) == pytest.approx(2.0)


def test_video_sec_vectorized():
    taus = np.array([0.0, 1.0, 2.0])
    got = msv.video_sec(taus, video_base=32.0, start_time=127.0, t0_raw=100.0)
    assert np.allclose(got, np.array([59.0, 60.0, 61.0]))


def test_n_output_frames():
    assert msv.n_output_frames(10.0, 20.0, 30.0) == 300
    assert msv.n_output_frames(0.0, 5.0, 30.0) == 150


def test_frame_times():
    ft = msv.frame_times(10.0, 12.0, 30.0)
    assert len(ft) == 60
    assert ft[0] == pytest.approx(10.0)
    assert ft[1] == pytest.approx(10.0 + 1 / 30.0)


def test_window_mask():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mask = msv.window_mask(times, 1.0, 3.0)
    assert list(mask) == [False, True, True, True, False]


def test_nearest_index_interior_and_clamp():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    assert msv.nearest_index(times, 1.4) == 1
    assert msv.nearest_index(times, 1.6) == 2
    assert msv.nearest_index(times, -5.0) == 0
    assert msv.nearest_index(times, 99.0) == 3


import os

REC_DIR = "Lickometry Data/ACG-26-3"
H5 = os.path.join(REC_DIR, "raw_data_2026-07-13_11-59-47.h5")
VIDEO = os.path.join(REC_DIR, "raw_data_2026-07-13_11-59-47.mp4")
PTS = os.path.join(REC_DIR, "raw_data_2026-07-13_11-59-47.txt")
LAYOUT = os.path.join(REC_DIR, "layout_w_controls.csv")

needs_reference = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (H5, PTS, LAYOUT)),
    reason="reference recording files not present",
)


@needs_reference
def test_load_recording_reference():
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO)
    assert rec.animal == "ACG-26-3-1"
    assert rec.sensor == 1
    assert rec.cap.shape == rec.time.shape
    assert rec.cap.size > 1000
    # session-relative time starts at ~0 and increases
    assert rec.time[0] == pytest.approx(0.0, abs=1.0)
    assert rec.time[-1] > rec.time[0]
    # licks detected, indices valid, vals consistent
    assert rec.lick_indices.size == rec.lick_times.size
    assert rec.lick_indices.max() < rec.cap.size
    assert np.allclose(rec.lick_vals, rec.cap[rec.lick_indices])
    # sync fields populated; video_base ~ 32 s for this recording
    assert rec.video_base == pytest.approx(31.97, abs=0.1)
    assert rec.session_duration > 3600


needs_video = pytest.mark.skipif(
    not os.path.exists(VIDEO), reason="reference video not present"
)


@needs_video
def test_frame_grabber_reads_rgb_and_advances():
    start = 60.0
    g = msv.FrameGrabber(VIDEO, clip_start_sec=start)
    try:
        assert g.src_fps > 1
        f0 = g.get(start)
        assert f0 is not None
        assert f0.ndim == 3 and f0.shape[2] == 3
        # advancing ~1s forward returns a frame of the same shape
        f1 = g.get(start + 1.0)
        assert f1.shape == f0.shape
    finally:
        g.close()
