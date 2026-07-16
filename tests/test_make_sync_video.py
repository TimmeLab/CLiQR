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
    clock = msv.SessionClock(pts_start_sec=float(pts_ns[2]) / 1e9,
                             latency=0.0, slope=1.0)
    sess = msv.frame_session_times(clock, pts_ns)
    assert sess[2] == pytest.approx(0.0)
    assert sess[0] == pytest.approx(-0.2)
    # session 0 at frame 2, session 0.5 at frame 7
    sf, ef = msv.compute_trim_frames(clock, pts_ns, 0.0, 0.5)
    assert sf == 2 and ef == 7


def test_compute_trim_frames_empty_window_raises():
    pts_ns = (np.arange(0, 5) * 100_000_000).astype(np.int64)
    clock = msv.SessionClock(pts_start_sec=0.0, latency=0.0, slope=1.0)
    with pytest.raises(ValueError):
        msv.compute_trim_frames(clock, pts_ns, 100.0, 200.0)


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
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO, msv.read_video_anchor(H5))
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
    # sync fields populated; bookmark frame PTS ~ 32 s for this recording
    assert rec.clock.pts_start_sec == pytest.approx(31.97, abs=0.1)
    assert rec.session_duration > 3600
    # PTS sidecar retained for trim/crop + per-frame timing
    assert rec.pts_ns.size > 1000
    # this recording predates the latency bracket AND the stop bookmark ->
    # no latency correction and no drift correction
    assert rec.clock.latency == 0.0
    assert rec.clock.slope == 1.0


needs_video = pytest.mark.skipif(
    not os.path.exists(VIDEO), reason="reference video not present"
)


@needs_reference
@needs_video
def test_trim_and_crop_and_frame_source(tmp_path):
    import imageio
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO, msv.read_video_anchor(H5))
    sf, ef = msv.compute_trim_frames(rec.clock, rec.pts_ns, 120.0, 123.0)
    start_sec = float(rec.pts_ns[sf] - rec.pts_ns[0]) / 1e9
    end_sec = float(rec.pts_ns[ef] - rec.pts_ns[0]) / 1e9 + 0.3
    out = str(tmp_path / "trim.mp4")
    msv.trim_and_crop(VIDEO, start_sec, end_sec, out, 452, 180, 360)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    r = imageio.get_reader(out, "ffmpeg")
    size = r.get_meta_data()["size"]
    r.close()
    assert size == (360, 360)  # (width, height)

    frame_sess = msv.probe_frame_session_times(out, rec.clock)
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
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO, msv.read_video_anchor(H5))
    sf, ef = msv.compute_trim_frames(rec.clock, rec.pts_ns, 120.0, 130.0)
    start_sec = float(rec.pts_ns[sf] - rec.pts_ns[0]) / 1e9
    end_sec = float(rec.pts_ns[ef] - rec.pts_ns[0]) / 1e9 + 0.3
    cropped = str(tmp_path / "cropped.mp4")
    msv.trim_and_crop(VIDEO, start_sec, end_sec, cropped, 452, 180, 360)
    assert msv.probe_start_pts(cropped) > 1.0  # not a zero-based timeline

    sub = str(tmp_path / "sub.mp4")
    msv.subclip_copy(cropped, start_sec + 2.0, start_sec + 5.0, sub)
    sess = msv.probe_frame_session_times(sub, rec.clock)
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
    clock = msv.SessionClock(pts_start_sec=msv.compute_video_base(pts_ns, fi),
                             latency=0.0, slope=1.0)
    # a long-ish window so any per-frame slip accumulates past rounding
    sf, ef = msv.compute_trim_frames(clock, pts_ns, 100.0, 160.0)
    start_sec = float(pts_ns[sf] - pts_ns[0]) / 1e9
    end_sec = float(pts_ns[ef] - pts_ns[0]) / 1e9 + 0.3
    out = str(tmp_path / "long.mp4")
    msv.trim_and_crop(VIDEO, start_sec, end_sec, out, 452, 180, 360)

    frame_sess = msv.probe_frame_session_times(out, clock)
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
    rec = msv.load_recording(H5, LAYOUT, PTS, VIDEO, msv.read_video_anchor(H5))
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
    assert args.intermediate is None
    # cropping is crop_video.py's job now
    assert not hasattr(args, "crop_w")
    assert not hasattr(args, "crop_h")


@needs_reference
def test_resolve_paths_defaults_from_h5():
    anchor = msv.read_video_anchor(H5)
    video, pts = msv.resolve_paths(H5, anchor)
    assert video.endswith("raw_data_2026-07-13_11-59-47.mp4")
    assert pts.endswith("raw_data_2026-07-13_11-59-47.txt")
    assert os.path.dirname(video) == os.path.dirname(H5)


def _synthetic_rec(pts_ns, video_base, latency, n=3, slope=1.0):
    """A Recording carrying only what clip_trim_window reads. ``n`` widens
    cap/time (default 3, matching the original clip_trim_window-only callers)
    for callers that need a renderable trace panel, e.g. render_clip.

    ``video_base`` is the bookmark frame's PTS in seconds (== pts_start_sec here,
    since these synthetic pts_ns start at 0)."""
    if n == 3:
        cap, time = np.zeros(3), np.zeros(3)
    else:
        time = np.linspace(0.0, 5.0, n)
        cap = np.sin(time)
    clock = msv.SessionClock(pts_start_sec=video_base, latency=latency, slope=slope)
    return msv.Recording(
        animal="X", sensor=1, cap=cap, time=time,
        lick_times=np.array([]), lick_indices=np.array([], dtype=int),
        lick_vals=np.array([]), clock=clock, video_path="v.mp4",
        session_duration=10.0, pts_ns=pts_ns,
    )


def test_clip_trim_window_applies_bookmark_latency():
    """FAILS if render_clip's anchor drops the bookmark-latency correction.

    The reference recording's latency is 0.0, so no reference-backed test can
    catch that regression — this synthetic one is the guard. The bracket gives a
    latency of exactly 0.25: values like 0.2 are not exactly representable and
    would put the assertion on a floating-point knife-edge.
    """
    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    vb = msv.compute_video_base(pts_ns, 2)  # 0.2
    plain = msv.clip_trim_window(_synthetic_rec(pts_ns, vb, 0.0), 0.0, 0.3)
    assert plain[0] == 2 and plain[2] == pytest.approx(0.2)
    shifted = msv.clip_trim_window(_synthetic_rec(pts_ns, vb, 0.25), 0.0, 0.3)
    assert shifted[0] == 0
    assert shifted[0] < plain[0]           # earlier start frame
    assert shifted[2] < plain[2]           # earlier start second


def test_clip_trim_window_matches_crop_window():
    """The renderer and the crop tool MUST resolve the same session window to the
    same video seconds. If they diverge, crop_video trims to one window while
    render_clip places frames using another, and every cropped video silently
    misaligns against its trace. Uses a NONZERO latency, which the reference
    recording cannot exercise.
    """
    import crop_video as cv
    from video.trimcrop import VideoAnchor

    pts_ns = (np.arange(0, 11) * 100_000_000).astype(np.int64)
    anchor = VideoAnchor(
        sensor_number=1, video_filename="v.mp4", video_frame_index=2,
        start_time=110.0, stop_time=110.3,
        host_before=110.0, host_after=110.5,   # latency exactly 0.25
    )
    assert anchor.latency == pytest.approx(0.25)
    vb = msv.compute_video_base(pts_ns, anchor.video_frame_index)
    rec = _synthetic_rec(pts_ns, vb, anchor.latency)

    assert (msv.clip_trim_window(rec, 0.0, anchor.session_duration)[:4]
            == cv.compute_crop_window(anchor, pts_ns))


def test_render_clip_probes_frame_session_with_latency_corrected_anchor(tmp_path, monkeypatch):
    """Regression guard for the one anchor bug the reference recording (latency
    0.0) can never catch: if render_clip's probe_frame_session_times call used a
    clock without the latency, the trim WINDOW would still be right
    (clip_trim_window handles that), but every frame's session LABEL would be
    `latency` seconds early, so src.get(tau) would return the frame captured at
    tau + latency — video running ahead of the trace by exactly the bookmark
    latency. render_clip must hand probe the SAME rec.clock (latency inside it).
    """
    pts_ns = (np.arange(0, 41) * 100_000_000).astype(np.int64)  # 0.0..4.0 s
    vb = msv.compute_video_base(pts_ns, 2)  # 0.2
    latency = 0.25
    rec = _synthetic_rec(pts_ns, vb, latency, n=50)

    recorded = {}

    def fake_subclip_copy(video_path, start_sec, end_sec, out_path, *a, **kw):
        return out_path

    def fake_probe(path, clock):
        recorded["clock"] = clock
        return np.linspace(0.0, 0.3, 5)

    class FakeSource:
        def __init__(self, path, frame_sess):
            pass

        def get(self, target_session):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(msv, "subclip_copy", fake_subclip_copy)
    monkeypatch.setattr(msv, "probe_frame_session_times", fake_probe)
    monkeypatch.setattr(msv, "TrimmedFrameSource", FakeSource)

    out = str(tmp_path / "clip.mp4")
    msv.render_clip(rec, 0.0, 0.2, out, fps=5.0)

    assert "clock" in recorded
    # the clock carries the latency (bookmark frame at τ=latency), so the probe
    # labels frames correctly; a latency-less clock would be the regression.
    assert recorded["clock"].latency == pytest.approx(latency)
    assert recorded["clock"].pts_start_sec == pytest.approx(vb)
