import numpy as np
import pytest

import make_sync_video as msv


def test_compute_video_base():
    pts_ns = np.array([1_000_000_000, 1_100_000_000, 1_250_000_000], dtype=np.int64)
    # (1_250_000_000 - 1_000_000_000) / 1e9 = 0.25
    assert msv.compute_video_base(pts_ns, 2) == pytest.approx(0.25)


def test_bookmark_latency_from_bracket():
    # frame's true host time ~ midpoint of the round-trip bracket; its offset
    # from start_time is the latency the video would otherwise lead the trace by.
    assert msv.bookmark_latency(1000.0, 1000.2, 998.0) == pytest.approx(2.1)
    # symmetric bracket around start -> ~0
    assert msv.bookmark_latency(999.9, 1000.1, 1000.0) == pytest.approx(0.0)


def test_bookmark_latency_missing_is_zero():
    assert msv.bookmark_latency(None, None, 1000.0) == 0.0
    assert msv.bookmark_latency(1000.0, None, 998.0) == 0.0


def test_video_sec_at_bookmark_anchor():
    # tau=0 (start_time bookmark) -> video_base, no spurious offset
    got = msv.video_sec(0.0, video_base=32.0)
    assert got == pytest.approx(32.0)


def test_video_sec_linear_and_offset():
    base = dict(video_base=32.0)
    assert msv.video_sec(5.0, **base) - msv.video_sec(0.0, **base) == pytest.approx(5.0)
    assert msv.video_sec(0.0, sync_offset=2.0, **base) - msv.video_sec(0.0, **base) == pytest.approx(2.0)


def test_video_sec_vectorized():
    taus = np.array([0.0, 1.0, 2.0])
    got = msv.video_sec(taus, video_base=32.0)
    assert np.allclose(got, np.array([32.0, 33.0, 34.0]))


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


def test_frame_session_times_and_trim_frames():
    # frames every 0.1 s; bookmark frame 2 -> session zero
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)  # 0.0 .. 1.0 s
    vb = msv.compute_video_base(pts_ns, 2)  # 0.2 s
    sess = msv.frame_session_times(pts_ns, vb)
    assert sess[2] == pytest.approx(0.0)
    assert sess[0] == pytest.approx(-0.2)
    # session 0 at frame 2, session 0.5 at frame 7
    sf, ef = msv.compute_trim_frames(pts_ns, vb, 0.0, 0.5)
    assert sf == 2 and ef == 7


def test_compute_trim_frames_empty_window_raises():
    pts_ns = (np.arange(0, 5) * 100_000_000).astype(np.int64)
    with pytest.raises(ValueError):
        msv.compute_trim_frames(pts_ns, 0.0, 100.0, 200.0)


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
    # PTS sidecar retained for trim/crop + per-frame timing
    assert rec.pts_ns.size > 1000
    # this recording predates bookmark-latency recording -> no correction
    assert rec.bookmark_latency == 0.0


needs_video = pytest.mark.skipif(
    not os.path.exists(VIDEO), reason="reference video not present"
)


@needs_reference
@needs_video
def test_trim_and_crop_and_frame_source(tmp_path):
    import imageio
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO)
    sf, ef = msv.compute_trim_frames(rec.pts_ns, rec.video_base, 120.0, 123.0)
    start_sec = float(rec.pts_ns[sf] - rec.pts_ns[0]) / 1e9
    end_sec = float(rec.pts_ns[ef] - rec.pts_ns[0]) / 1e9 + 0.3
    out = str(tmp_path / "trim.mp4")
    msv.trim_and_crop(VIDEO, start_sec, end_sec, out, 452, 180, 360)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    r = imageio.get_reader(out, "ffmpeg")
    size = r.get_meta_data()["size"]
    r.close()
    assert size == (360, 360)  # (width, height)

    frame_sess = msv.probe_frame_session_times(out, rec.video_base)
    assert frame_sess[0] <= 120.0 and frame_sess[-1] >= 123.0
    assert np.all(np.diff(frame_sess) >= 0)  # monotonic

    src = msv.TrimmedFrameSource(out, frame_sess)
    try:
        f0 = src.get(120.0)
        assert f0 is not None and f0.shape[:2] == (360, 360)  # (h, w)
        f1 = src.get(122.0)
        assert f1.shape == f0.shape
    finally:
        src.close()


@needs_reference
@needs_video
def test_subclip_copy_lands_on_a_cropped_file(tmp_path):
    """A cropped file's PTS start at the session start, not 0. Stream-copying a
    window out of it must still cover that window."""
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO)
    sf, ef = msv.compute_trim_frames(rec.pts_ns, rec.video_base, 120.0, 130.0)
    start_sec = float(rec.pts_ns[sf] - rec.pts_ns[0]) / 1e9
    end_sec = float(rec.pts_ns[ef] - rec.pts_ns[0]) / 1e9 + 0.3
    cropped = str(tmp_path / "cropped.mp4")
    msv.trim_and_crop(VIDEO, start_sec, end_sec, cropped, 452, 180, 360)
    assert msv.probe_start_pts(cropped) > 1.0  # not a zero-based timeline

    sub = str(tmp_path / "sub.mp4")
    msv.subclip_copy(cropped, start_sec + 2.0, start_sec + 5.0, sub)
    sess = msv.probe_frame_session_times(sub, rec.video_base)
    assert sess.size > 0
    assert sess[0] <= 122.0 and sess[-1] >= 124.0


@needs_reference
@needs_video
def test_trimmed_frame_source_decode_matches_pts(tmp_path):
    # This footage is VFR (coded 240 fps, real ~120). imageio's default reader
    # forces CFR and DUPLICATES frames, so its sequential decode count exceeds
    # the ffprobe pts list; TrimmedFrameSource counts frames by decode but times
    # them by pts, so the mismatch slips the frame<->session mapping ~1 s per
    # ~300 s (different frame shown at the same session in clips of different
    # length). The reader must decode passthrough: one decoded frame per pts.
    import h5py
    pts_ns = np.loadtxt(PTS, dtype=np.int64)
    with h5py.File(H5, "r") as f:
        board, sensor, _ = msv.find_video_sensor(f)
        fi = int(f[board][sensor]["video_frame_index"][()])
    video_base = msv.compute_video_base(pts_ns, fi)
    # a long-ish window so any per-frame slip accumulates past rounding
    sf, ef = msv.compute_trim_frames(pts_ns, video_base, 100.0, 160.0)
    start_sec = float(pts_ns[sf] - pts_ns[0]) / 1e9
    end_sec = float(pts_ns[ef] - pts_ns[0]) / 1e9 + 0.3
    out = str(tmp_path / "long.mp4")
    msv.trim_and_crop(VIDEO, start_sec, end_sec, out, 452, 180, 360)

    frame_sess = msv.probe_frame_session_times(out, video_base)
    src = msv.TrimmedFrameSource(out, frame_sess)
    decoded = 0
    try:
        while True:
            try:
                src._reader.get_next_data()
            except (IndexError, StopIteration):
                break
            decoded += 1
    finally:
        src.close()
    # one decoded frame per pts entry -> mapping can't drift
    assert decoded == frame_sess.size


import subprocess


def _video_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", path,
    ])
    return float(out.strip())


@needs_reference
@needs_video
def test_render_clip_smoke(tmp_path):
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO)
    # sync anchor: tau=0 (start_time bookmark) maps to video_base seconds into the file
    assert msv.video_sec(0.0, rec.video_base) == pytest.approx(rec.video_base)
    out = str(tmp_path / "clip.mp4")
    start, end, fps = 120.0, 124.0, 30.0
    msv.render_clip(rec, start, end, out, fps=fps)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    # duration ~ (end - start), within a couple frames
    assert _video_duration(out) == pytest.approx(end - start, abs=2.0 / fps)


@needs_reference
def test_read_session_duration_reference():
    duration = msv.read_session_duration(H5)
    assert isinstance(duration, float)
    assert duration > 0


def test_validate_window_ok():
    msv.validate_window(10.0, 20.0, 100.0)  # no raise


@pytest.mark.parametrize("start,end,dur", [
    (20.0, 10.0, 100.0),   # inverted
    (-1.0, 10.0, 100.0),   # negative start
    (10.0, 200.0, 100.0),  # past session end
])
def test_validate_window_rejects(start, end, dur):
    with pytest.raises(ValueError):
        msv.validate_window(start, end, dur)


def test_build_arg_parser_parses_required():
    p = msv.build_arg_parser()
    args = p.parse_args([
        "--h5", "r.h5", "--layout", "l.csv",
        "--start", "5", "--end", "9", "--out", "o.mp4",
    ])
    assert args.h5 == "r.h5" and args.start == 5.0 and args.end == 9.0
    assert args.fps == 30.0 and args.window == 2.5 and args.sync_offset == 0.0
    assert args.crop_w == 640 and args.crop_h == 360 and args.intermediate is None


@needs_reference
def test_resolve_paths_defaults_from_h5():
    anchor = msv.read_video_anchor(H5)
    video, pts = msv.resolve_paths(H5, anchor)
    assert video.endswith("raw_data_2026-07-13_11-59-47.mp4")
    assert pts.endswith("raw_data_2026-07-13_11-59-47.txt")
    assert os.path.dirname(video) == os.path.dirname(H5)
