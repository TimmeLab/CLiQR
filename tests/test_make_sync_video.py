import numpy as np
import pytest

import make_sync_video as msv


def test_load_container_pts_prefers_encoded_sidecar(tmp_path):
    capture = (np.arange(0, 5) * 100_000_000).astype(np.int64)
    enc = tmp_path / "rec.encpts.txt"
    # encoder emitted only frames 0,1,3 (2 and 4 dropped) -> exact container times
    encoded = capture[[0, 1, 3]]
    np.savetxt(str(enc), encoded, fmt="%d")
    got = msv.load_container_pts(str(tmp_path / "rec.txt"), capture)
    assert np.array_equal(got, encoded)


def test_load_container_pts_falls_back_to_capture(tmp_path):
    capture = (np.arange(0, 5) * 100_000_000).astype(np.int64)
    # no .encpts.txt written -> fall back to the capture sidecar unchanged
    got = msv.load_container_pts(str(tmp_path / "rec.txt"), capture)
    assert got is capture


def test_compute_video_base():
    pts_ns = np.array([1_000_000_000, 1_100_000_000, 1_250_000_000], dtype=np.int64)
    # (1_250_000_000 - 1_000_000_000) / 1e9 = 0.25
    assert msv.compute_video_base(pts_ns, 2) == pytest.approx(0.25)


def test_bookmark_latency_from_end_of_bracket():
    # frame's true host time ~ host_after (bookmark runs at the END of the Pi
    # round-trip); its offset from start_time is the latency the video would
    # otherwise lead the trace by, backed off the Pi capture->exec gap.
    assert msv.bookmark_latency(1000.2, 998.0) == pytest.approx(2.2)
    assert msv.bookmark_latency(1000.2, 998.0, pi_monotonic=500.05, pts=500.0) \
        == pytest.approx(2.15)


def test_bookmark_latency_missing_is_zero():
    assert msv.bookmark_latency(None, 1000.0) == 0.0
    assert msv.bookmark_latency(None, 998.0, pi_monotonic=1.0, pts=0.9) == 0.0


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

    frame_sess = msv.probe_frame_session_times(
        out, rec.clock, rec.pts_ns, msv.probe_frame_rate(out))
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
    sess = msv.probe_frame_session_times(
        sub, rec.clock, rec.pts_ns, msv.probe_frame_rate(sub))
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

    frame_sess = msv.probe_frame_session_times(
        out, clock, pts_ns, msv.probe_frame_rate(out))
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
    assert args.fps is None and args.window == 2.5
    # default sync-offset is the measured constant 2-frame residual lead, not 0
    assert args.sync_offset == pytest.approx(msv.DEFAULT_SYNC_OFFSET)
    assert args.sync_offset == pytest.approx(2.0 / 120.0)
    assert args.intermediate is None
    # crop is a display-time slice: no re-encode. crop_video.py writes the box to
    # a sidecar auto-picked by default; --no-crop forces the full frame.
    assert args.crop_params is None
    assert args.no_crop is False


def test_load_crop_reads_sidecar(tmp_path):
    from video.trimcrop import crop_params_path, write_crop_params
    video = str(tmp_path / "v.mp4")
    write_crop_params(crop_params_path(video), 452, 180, 360)
    box = msv.load_crop(video)
    assert (box.x, box.y, box.size) == (452, 180, 360)


def test_load_crop_missing_sidecar_returns_none(tmp_path):
    assert msv.load_crop(str(tmp_path / "v.mp4")) is None


def test_load_crop_no_crop_flag_skips_sidecar(tmp_path):
    from video.trimcrop import crop_params_path, write_crop_params
    video = str(tmp_path / "v.mp4")
    write_crop_params(crop_params_path(video), 1, 2, 3)
    assert msv.load_crop(video, no_crop=True) is None


def test_load_crop_explicit_params_path(tmp_path):
    from video.trimcrop import write_crop_params
    p = str(tmp_path / "elsewhere.json")
    write_crop_params(p, 5, 6, 7)
    box = msv.load_crop(str(tmp_path / "v.mp4"), params_path=p)
    assert (box.x, box.y, box.size) == (5, 6, 7)


def test_trimmed_frame_source_applies_crop(monkeypatch):
    """TrimmedFrameSource must slice each decoded frame to its crop box, so the
    left panel shows only the sipper region while the video it TIMES stays the
    untouched original (crop never re-encodes)."""
    from video.trimcrop import CropBox

    full = np.arange(8 * 8 * 3).reshape(8, 8, 3).astype(np.uint8)

    class FakeReader:
        def get_next_data(self):
            return full

        def close(self):
            pass

    monkeypatch.setattr(msv.imageio, "get_reader", lambda *a, **k: FakeReader())
    src = msv.TrimmedFrameSource("x.mp4", np.array([0.0, 0.1]),
                                 crop=CropBox(x=2, y=3, size=4))
    out = src.get(0.0)
    assert out.shape == (4, 4, 3)
    assert np.array_equal(out, full[3:7, 2:6])


def test_render_clip_passes_crop_to_source(tmp_path, monkeypatch):
    """render_clip must forward its crop box to TrimmedFrameSource; a regression
    that dropped it would render the full frame instead of the sipper crop."""
    from video.trimcrop import CropBox
    pts_ns = (np.arange(0, 41) * 100_000_000).astype(np.int64)
    vb = msv.compute_video_base(pts_ns, 2)
    rec = _synthetic_rec(pts_ns, vb, 0.0, n=50)

    seen = {}

    class RecordingSource:
        def __init__(self, path, frame_sess, crop=None):
            seen["crop"] = crop

        def get(self, target_session):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(msv, "subclip_copy", lambda *a, **kw: a[3] if len(a) > 3 else kw["out_path"])
    monkeypatch.setattr(msv, "probe_frame_rate", lambda path: 10.0)
    monkeypatch.setattr(msv, "probe_frame_session_times",
                        lambda *a, **kw: np.linspace(0.0, 0.3, 5))
    monkeypatch.setattr(msv, "TrimmedFrameSource", RecordingSource)

    box = CropBox(x=1, y=2, size=3)
    msv.render_clip(rec, 0.0, 0.2, str(tmp_path / "clip.mp4"), fps=5.0, crop=box)
    assert seen["crop"] is box


REC27 = os.path.join(REC_DIR, "raw_data_2026-07-27_11-56-15.mp4")
needs_ref27 = pytest.mark.skipif(
    not os.path.exists(REC27), reason="07-27 reference video not present"
)


@needs_ref27
def test_probe_frame_rate_uses_true_cfr_rate():
    """r_frame_rate rounds the Pi's real capture rate to 120/1, but the container
    is CFR at ~120.0048 fps (avg_frame_rate == frames/duration).
    probe_frame_session_times recovers each frame's ordinal with
    round(pts_time * rate); a rate rounded even 0.005 fps low slips the ordinal ~1
    frame per 25k frames, dragging the video ~33 frames (0.27 s) ahead of the trace
    by the end of a 2-hour recording. probe_frame_rate must return the true rate.
    """
    rate = msv.probe_frame_rate(REC27)
    assert rate == pytest.approx(120.0048, abs=1e-3)
    assert abs(rate - 120.0) > 1e-3  # the rounded r_frame_rate (120/1) would fail


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
        session_duration=10.0, pts_ns=pts_ns, container_pts_ns=pts_ns,
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
        host_before=110.0, host_after=110.25,  # latency = after-start = 0.25 (exact)
    )
    assert anchor.latency == pytest.approx(0.25)
    vb = msv.compute_video_base(pts_ns, anchor.video_frame_index)
    rec = _synthetic_rec(pts_ns, vb, anchor.latency)

    assert (msv.clip_trim_window(rec, 0.0, anchor.session_duration)[:4]
            == cv.compute_crop_window(anchor, pts_ns))


def test_render_clip_default_sync_offset_delays_video(tmp_path, monkeypatch):
    """render_clip must apply DEFAULT_SYNC_OFFSET so each output time tau fetches
    the source frame at tau - offset (an earlier frame shown later => video
    delayed), cancelling the measured constant ~2-frame lead. A regression to
    sync_offset=0 would show the frame AT tau, leaving the video ahead."""
    pts_ns = (np.arange(0, 41) * 100_000_000).astype(np.int64)
    vb = msv.compute_video_base(pts_ns, 2)
    rec = _synthetic_rec(pts_ns, vb, 0.0, n=50)

    targets = []

    class RecordingSource:
        def __init__(self, path, frame_sess, crop=None):
            pass

        def get(self, target_session):
            targets.append(target_session)
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(msv, "subclip_copy",
                        lambda *a, **kw: a[-1] if a else kw["out_path"])
    monkeypatch.setattr(msv, "probe_frame_rate", lambda path: 10.0)
    monkeypatch.setattr(msv, "probe_frame_session_times",
                        lambda *a, **kw: np.linspace(0.0, 0.3, 5))
    monkeypatch.setattr(msv, "TrimmedFrameSource", RecordingSource)

    msv.render_clip(rec, 0.0, 0.2, str(tmp_path / "clip.mp4"), fps=5.0)

    # first fetch is at start - offset; every fetch is shifted by the same offset
    assert targets[0] == pytest.approx(0.0 - msv.DEFAULT_SYNC_OFFSET)
    assert msv.DEFAULT_SYNC_OFFSET > 0  # positive => delays the video


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

    def fake_probe(path, clock, pts_ns, framerate):
        recorded["clock"] = clock
        return np.linspace(0.0, 0.3, 5)

    class FakeSource:
        def __init__(self, path, frame_sess, crop=None):
            pass

        def get(self, target_session):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(msv, "subclip_copy", fake_subclip_copy)
    monkeypatch.setattr(msv, "probe_frame_rate", lambda path: 10.0)
    monkeypatch.setattr(msv, "probe_frame_session_times", fake_probe)
    monkeypatch.setattr(msv, "TrimmedFrameSource", FakeSource)

    out = str(tmp_path / "clip.mp4")
    msv.render_clip(rec, 0.0, 0.2, out, fps=5.0)

    assert "clock" in recorded
    # the clock carries the latency (bookmark frame at τ=latency), so the probe
    # labels frames correctly; a latency-less clock would be the regression.
    assert recorded["clock"].latency == pytest.approx(latency)
    assert recorded["clock"].pts_start_sec == pytest.approx(vb)


def test_read_trace_from_combined_reads_named_cycle(tmp_path):
    # A minimal combined-results file: <animal>/<cycle>/{cap_data,time_data,lick_times,lick_indices}.
    # _read_trace_from_combined must return exactly that cycle's arrays (the fast path that avoids
    # re-running filter_data on the raw recording).
    import h5py
    path = tmp_path / "results_combined.h5"
    cap = np.array([676, 675, 660, 676], dtype=np.int64)
    time = np.array([0.0, 0.01, 0.02, 0.03])
    lick_times = np.array([0.02])
    lick_indices = np.array([2], dtype=int)
    with h5py.File(str(path), "w") as f:
        g = f.create_group("ACG-1").create_group("3")   # animal ACG-1, cycle 3
        g.create_dataset("cap_data", data=cap)
        g.create_dataset("time_data", data=time)
        g.create_dataset("lick_times", data=lick_times)
        g.create_dataset("lick_indices", data=lick_indices)

    got_cap, got_time, got_lt, got_li = msv._read_trace_from_combined(str(path), "ACG-1", 3)
    assert np.array_equal(got_cap, cap)
    assert np.array_equal(got_time, time)
    assert np.array_equal(got_lt, lick_times)
    assert np.array_equal(got_li, lick_indices)


def test_read_trace_from_combined_missing_cycle_raises(tmp_path):
    import h5py
    path = tmp_path / "results_combined.h5"
    with h5py.File(str(path), "w") as f:
        f.create_group("ACG-1").create_group("0")
    with pytest.raises(ValueError, match="cycle 5 not found"):
        msv._read_trace_from_combined(str(path), "ACG-1", 5)
