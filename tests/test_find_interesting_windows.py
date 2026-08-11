"""Unit tests for the pure logic in scripts/find_interesting_windows.py.

The HDF5 reading, provenance resolution, and shell-script writing are exercised by hand on real
recordings; only the deterministic, self-contained helpers (variance, masking, selection, window
construction, command formatting) are unit-tested here.
"""
import os
import sys

import numpy as np
import pytest

# Make the repository root importable so `scripts.find_interesting_windows` resolves regardless of
# where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video.trimcrop import SessionClock, frame_session_times  # noqa: E402

from scripts.find_interesting_windows import (  # noqa: E402
    sliding_variance, center_sample_indices, mask_bout_windows, select_climbing_centers,
    clip_window, count_licks_in_window, build_rois_for_cycle, parse_offsets, build_command,
    is_control,
    parse_dlc_video_stem, read_dlc_windows,
    frame_window_to_session, clamp_to_trace,
    load_frame_session_times,
    build_dlc_rois,
    build_cycles_for_dlc,
)


def test_sliding_variance_matches_numpy_var():
    # For a small signal, every sliding window's variance must equal numpy's population variance
    # of that same slice. This checks the prefix-sum formula against a trusted reference.
    signal = np.array([1.0, 3.0, 2.0, 8.0, 4.0, 4.0, 4.0])
    window = 3
    result = sliding_variance(signal, window)
    expected = np.array([np.var(signal[i:i + window]) for i in range(len(signal) - window + 1)])
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_sliding_variance_constant_signal_is_zero():
    # A flat signal has zero variance in every window; round-off must not make it negative.
    signal = np.full(50, 5.0)
    result = sliding_variance(signal, 10)
    assert result.shape[0] == 50 - 10 + 1
    assert np.all(result >= 0.0)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_sliding_variance_signal_shorter_than_window_is_empty():
    assert sliding_variance(np.array([1.0, 2.0]), 5).size == 0


def test_center_sample_indices():
    # Window width 3 -> each window's center is offset by 1 (== 3 // 2) from its start index.
    np.testing.assert_array_equal(center_sample_indices(4, 3), np.array([1, 2, 3, 4]))


def test_mask_bout_windows_sets_neg_inf_near_bouts():
    # Centers every 1 s from 0..9 s. A single bout at t = 5 s (duration 0) with a 2 s guard must
    # disqualify the windows centered at 3, 4, 5, 6, 7 s (within 2 s of the bout), leaving the rest.
    center_times = np.arange(10, dtype=float)
    variance = np.ones(10)
    mask_bout_windows(variance, center_times, [5.0], [0.0], guard_seconds=2.0)
    disqualified = ~np.isfinite(variance)
    np.testing.assert_array_equal(np.where(disqualified)[0], np.array([3, 4, 5, 6, 7]))
    # Windows outside the guard keep their original value.
    assert variance[0] == 1.0 and variance[9] == 1.0


def test_select_climbing_centers_respects_separation_and_order():
    # Variance peaks at indices 2 (highest) and 3 (second), which are only 1 s apart. With a 5 s
    # minimum separation, the greedy picker must take index 2 and then SKIP index 3, taking the
    # next far-enough, lower peak (index 8).
    center_times = np.arange(10, dtype=float)
    variance = np.array([0, 0, 10, 9, 0, 0, 0, 0, 5, 0], dtype=float)
    picks = select_climbing_centers(variance, center_times, n_wanted=2, min_separation_s=5.0)
    picked_centers = [c for c, _ in picks]
    assert picked_centers == [2.0, 8.0]
    # Scores are reported alongside, strongest first.
    assert picks[0][1] == 10.0 and picks[1][1] == 5.0


def test_select_climbing_centers_stops_at_masked():
    # Only two finite windows exist; the rest are -inf (masked). Even asking for 4, we get 2.
    center_times = np.arange(5, dtype=float)
    variance = np.array([3.0, -np.inf, -np.inf, 1.0, -np.inf])
    picks = select_climbing_centers(variance, center_times, n_wanted=4, min_separation_s=1.0)
    assert [c for c, _ in picks] == [0.0, 3.0]


def test_clip_window_clamps_to_recording_bounds():
    # A centered window is kept full width in the interior...
    assert clip_window(50.0, 12.0, span_s=100.0) == (44.0, 56.0)
    # ...but is clipped (not shifted) at the start and end edges.
    assert clip_window(2.0, 12.0, span_s=100.0) == (0.0, 8.0)
    assert clip_window(99.0, 12.0, span_s=100.0) == (93.0, 100.0)


def test_count_licks_in_window_is_inclusive():
    licks = np.array([1.0, 5.0, 5.5, 9.0])
    assert count_licks_in_window(licks, 5.0, 6.0) == 2   # 5.0 and 5.5
    assert count_licks_in_window(np.array([]), 0.0, 10.0) == 0


def test_build_rois_separates_licking_and_climbing():
    # Synthetic 60 s / 100 Hz trace. A slow, large "climbing" excursion sits at ~10 s (a smooth
    # bump, no licks). A detected bout sits at ~40 s. The builder must return the bump as a
    # lick-free 'climb' window and the bout as a 'lick' window, with no overlap between them.
    fs = 100.0
    t = np.arange(0, 60, 1.0 / fs)
    cap = np.full(t.shape, 676.0)                     # flat baseline (median capacitance)
    # Climbing bump: a wide Gaussian centered at 10 s -> high local variance, slow, no licks.
    cap += 120.0 * np.exp(-((t - 10.0) ** 2) / (2 * 1.5 ** 2))

    bout_start_times = np.array([40.0])
    bout_durations = np.array([0.5])
    bout_lick_counts = np.array([7])
    lick_times = np.linspace(40.0, 40.5, 7)           # the 7 licks of that bout

    params = {"n_lick": 3, "n_climb": 3, "roi_seconds": 6.0, "climb_skip_edges": 0.0,
              "var_window": 1.0, "min_var": 0.0}
    rois = build_rois_for_cycle(cap, t, bout_start_times, bout_durations,
                                bout_lick_counts, lick_times, params)

    licks = [r for r in rois if r["category"] == "lick"]
    climbs = [r for r in rois if r["category"] == "climb"]
    assert len(licks) == 1                            # only one bout exists
    assert licks[0]["center"] == 40.25               # bout_start + duration/2
    assert len(climbs) >= 1
    # The strongest climbing window sits on the 10 s bump and contains no licks.
    top_climb = climbs[0]
    assert 7.0 <= top_climb["center"] <= 13.0
    assert top_climb["n_licks_in_window"] == 0
    # Licking and climbing windows do not overlap.
    for climb in climbs:
        assert climb["end"] <= licks[0]["start"] or climb["start"] >= licks[0]["end"]


def test_build_rois_lick_window_contains_whole_bout():
    # A bout LONGER than roi_seconds must not be truncated: the emitted window has to cover the
    # entire bout (plus the pad), even though that makes it wider than roi_seconds. Truncating it
    # cuts licks off the end of the clip, which is exactly what we watch the clip for.
    fs = 100.0
    t = np.arange(0, 120, 1.0 / fs)
    cap = np.full(t.shape, 676.0)

    bout_start, bout_duration = 50.0, 25.0            # 25 s bout vs a 12 s roi_seconds
    lick_times = np.arange(bout_start, bout_start + bout_duration, 0.15)
    params = {"n_lick": 1, "n_climb": 0, "roi_seconds": 12.0, "climb_skip_edges": 0.0,
              "var_window": 1.0, "min_var": 0.0, "lick_pad": 2.0}
    rois = build_rois_for_cycle(cap, t, [bout_start], [bout_duration], [len(lick_times)],
                                lick_times, params)

    lick = [r for r in rois if r["category"] == "lick"][0]
    assert lick["start"] <= bout_start
    assert lick["end"] >= bout_start + bout_duration
    # Every lick of the bout is inside the emitted window.
    assert lick["n_licks_in_window"] == len(lick_times)
    # The pad is applied on both sides.
    assert lick["start"] == bout_start - params["lick_pad"]
    assert lick["end"] == bout_start + bout_duration + params["lick_pad"]


def test_build_rois_short_bout_keeps_roi_seconds_width():
    # A bout much SHORTER than roi_seconds still yields a full-width roi_seconds window centered on
    # it, so the clip has context around the bout rather than a 1 s sliver.
    fs = 100.0
    t = np.arange(0, 120, 1.0 / fs)
    cap = np.full(t.shape, 676.0)
    lick_times = np.linspace(50.0, 50.5, 5)
    params = {"n_lick": 1, "n_climb": 0, "roi_seconds": 12.0, "climb_skip_edges": 0.0,
              "var_window": 1.0, "min_var": 0.0, "lick_pad": 2.0}
    rois = build_rois_for_cycle(cap, t, [50.0], [0.5], [5], lick_times, params)
    lick = [r for r in rois if r["category"] == "lick"][0]
    assert lick["end"] - lick["start"] == params["roi_seconds"]
    assert lick["start"] == 50.25 - 6.0


def test_build_rois_climbing_uses_real_time_base_not_sample_rate():
    # The capacitance trace is NOT uniformly sampled: the hardware stalls, so sample index / average
    # rate drifts away from the real timestamp (seconds of error on a real recording). The climbing
    # window must be positioned from `time_data` itself, not from an assumed constant rate.
    #
    # Here the first half of the trace is sampled at 100 Hz and the second half at 20 Hz, so the
    # average rate (~36 Hz) matches neither. A bump sits at a known time in the SLOW half.
    fast = np.arange(0, 60, 1.0 / 100.0)
    slow = np.arange(60, 180, 1.0 / 20.0)
    t = np.concatenate([fast, slow])
    bump_time = 120.0
    cap = np.full(t.shape, 676.0)
    cap += 120.0 * np.exp(-((t - bump_time) ** 2) / (2 * 1.5 ** 2))

    params = {"n_lick": 0, "n_climb": 1, "roi_seconds": 12.0, "climb_skip_edges": 0.0,
              "var_window": 1.0, "min_var": 0.0, "lick_pad": 2.0}
    rois = build_rois_for_cycle(cap, t, [], [], [], [], params)
    climbs = [r for r in rois if r["category"] == "climb"]
    assert len(climbs) == 1
    # The window has to land ON the bump. With the index/rate assumption it lands far away.
    assert abs(climbs[0]["center"] - bump_time) < 2.0


def test_build_rois_climbing_mask_uses_real_time_base():
    # Same non-uniform trace, but now the bump IS a detected licking bout. Masking is done in
    # seconds, so with a wrong time base the mask misses the bout and the bout gets reported as
    # "climbing". No lick-free window is loud enough here, so nothing should be returned.
    fast = np.arange(0, 60, 1.0 / 100.0)
    slow = np.arange(60, 180, 1.0 / 20.0)
    t = np.concatenate([fast, slow])
    bout_start, bout_duration = 118.5, 3.0
    cap = np.full(t.shape, 676.0)
    cap += 120.0 * np.exp(-((t - 120.0) ** 2) / (2 * 1.5 ** 2))
    lick_times = np.arange(bout_start, bout_start + bout_duration, 0.15)

    params = {"n_lick": 0, "n_climb": 1, "roi_seconds": 12.0, "climb_skip_edges": 0.0,
              "var_window": 1.0, "min_var": 1.0, "lick_pad": 2.0}
    rois = build_rois_for_cycle(cap, t, [bout_start], [bout_duration], [len(lick_times)],
                                lick_times, params)
    climbs = [r for r in rois if r["category"] == "climb"]
    assert climbs == []


def test_build_rois_climbing_skips_session_edges():
    # The first and last few minutes of a session are dominated by start-up / shut-down transients
    # (the operator handling the cage, sipper insertion, sensor settling), not by climbing. Those
    # stretches must be excluded from the climbing search even though they are the loudest ones.
    fs = 100.0
    t = np.arange(0, 1800, 1.0 / fs)                  # 30 min session
    cap = np.full(t.shape, 676.0)
    cap += 300.0 * np.exp(-((t - 60.0) ** 2) / (2 * 1.5 ** 2))     # start-up transient (loudest)
    cap += 300.0 * np.exp(-((t - 1740.0) ** 2) / (2 * 1.5 ** 2))   # shut-down transient
    cap += 120.0 * np.exp(-((t - 900.0) ** 2) / (2 * 1.5 ** 2))    # real climbing, mid-session

    params = {"n_lick": 0, "n_climb": 3, "roi_seconds": 12.0, "climb_skip_edges": 300.0,
              "var_window": 1.0, "min_var": 1.0, "lick_pad": 2.0}
    rois = build_rois_for_cycle(cap, t, [], [], [], [], params)
    climbs = [r for r in rois if r["category"] == "climb"]
    assert len(climbs) == 1
    assert abs(climbs[0]["center"] - 900.0) < 2.0
    # The whole window, not just its center, is clear of both excluded edges.
    assert climbs[0]["start"] >= 300.0
    assert climbs[0]["end"] <= 1800.0 - 300.0


def test_build_rois_climbing_skip_edges_does_not_touch_licking():
    # The edge exclusion is a CLIMBING heuristic only: a real bout in the first five minutes is
    # still real drinking and must survive.
    fs = 100.0
    t = np.arange(0, 1800, 1.0 / fs)
    cap = np.full(t.shape, 676.0)
    lick_times = np.linspace(30.0, 33.0, 20)
    params = {"n_lick": 3, "n_climb": 0, "roi_seconds": 12.0, "climb_skip_edges": 300.0,
              "var_window": 1.0, "min_var": 0.0, "lick_pad": 2.0}
    rois = build_rois_for_cycle(cap, t, [30.0], [3.0], [20], lick_times, params)
    licks = [r for r in rois if r["category"] == "lick"]
    assert len(licks) == 1 and licks[0]["n_licks_in_window"] == 20


def test_parse_offsets():
    assert parse_offsets(["0=280", "3=-1.5"]) == {0: 280.0, 3: -1.5}
    assert parse_offsets(None) == {}


def test_build_command_restart_warns_without_offset():
    # A restart cycle with no offset must emit a WARNING block and the command with raw start/end.
    row = {"animal": "A1", "cycle": 0, "category": "lick", "rank": 0,
           "start": 100.0, "end": 112.0, "restart": True,
           "raw_h5": "data/raw_07-22.h5", "layout": "data/layout.csv"}
    lines = build_command(row, out_dir="clips", offsets={}, combined_h5="results_combined.h5")
    text = "\n".join(lines)
    assert "WARNING" in text
    assert "--start 100.000 --end 112.000" in text
    assert "clips/A1_c0_lick0.mp4" in text
    # The command reads the trace from the combined file (fast path), not by re-running filter_data.
    assert "--combined-h5 results_combined.h5 --cycle 0" in text


def test_build_command_offset_shifts_and_silences_warning():
    # Supplying an offset for the restart cycle shifts start/end and suppresses the warning
    # (the user has taken responsibility for the alignment).
    row = {"animal": "A1", "cycle": 0, "category": "lick", "rank": 0,
           "start": 100.0, "end": 112.0, "restart": True,
           "raw_h5": "data/raw_07-22.h5", "layout": "data/layout.csv"}
    lines = build_command(row, out_dir="clips", offsets={0: 280.0}, combined_h5="results_combined.h5")
    text = "\n".join(lines)
    assert "WARNING" not in text
    assert "--start 380.000 --end 392.000" in text


def test_build_command_climb_renders_full_frame():
    # A climbing clip must show the WHOLE frame: the crop box is framed on the sipper tip for
    # licking, so cropping a climbing clip cuts out the sipper/cage the animal is climbing on.
    row = {"animal": "A1", "cycle": 0, "category": "climb", "rank": 0,
           "start": 100.0, "end": 112.0, "restart": False,
           "raw_h5": "data/raw_07-23.h5", "layout": "data/layout.csv"}
    lines = build_command(row, out_dir="clips", offsets={}, combined_h5="results_combined.h5")
    assert "--no-crop" in "\n".join(lines)


def test_build_command_lick_keeps_crop():
    # Licking clips keep the crop: the crop is what makes the tongue visible at all.
    row = {"animal": "A1", "cycle": 0, "category": "lick", "rank": 0,
           "start": 100.0, "end": 112.0, "restart": False,
           "raw_h5": "data/raw_07-23.h5", "layout": "data/layout.csv"}
    lines = build_command(row, out_dir="clips", offsets={}, combined_h5="results_combined.h5")
    assert "--no-crop" not in "\n".join(lines)


def _lick_row():
    return {"animal": "A1", "cycle": 0, "category": "lick", "rank": 0,
            "start": 100.0, "end": 112.0, "restart": False,
            "raw_h5": "data/raw_07-23.h5", "layout": "data/layout.csv"}


def test_build_command_passes_speed_through():
    # Slow motion is a per-batch choice: every emitted command must carry it, or you
    # get one slow clip and a directory of real-time ones.
    lines = build_command(_lick_row(), out_dir="clips", offsets={},
                          combined_h5="results_combined.h5", speed=0.25)
    assert "--speed 0.25" in "\n".join(lines)


def test_build_command_omits_speed_at_real_time():
    # Default real time emits no flag, so existing make_clips.sh files are unchanged.
    lines = build_command(_lick_row(), out_dir="clips", offsets={},
                          combined_h5="results_combined.h5")
    assert "--speed" not in "\n".join(lines)


def test_is_control():
    assert is_control("Control1") and is_control("Control12")
    assert not is_control("ACG-26-3-5")


# ---------------------------------------------------------------------------
# DLC window mode
# ---------------------------------------------------------------------------
def test_parse_dlc_video_stem_strips_cfr():
    # DLC ran on a CFR re-encode: the stem the raw .h5 is named after is the one WITHOUT _cfr,
    # and the caller needs to know it was a re-encode (those rows are not renderable).
    stem, is_cfr = parse_dlc_video_stem(
        "/N/lustre/project/proj-530/videos/raw_data_2026-07-13_11-59-47_cfr.mp4")
    assert stem == "raw_data_2026-07-13_11-59-47"
    assert is_cfr is True


def test_parse_dlc_video_stem_leaves_original_alone():
    stem, is_cfr = parse_dlc_video_stem("videos/raw_data_2026-07-24_12-02-14.mp4")
    assert stem == "raw_data_2026-07-24_12-02-14"
    assert is_cfr is False


def test_parse_dlc_video_stem_only_strips_a_trailing_cfr():
    # "_cfr" in the middle of a name is part of the name, not the re-encode marker.
    stem, is_cfr = parse_dlc_video_stem("raw_data_cfr_test_2026-07-24.mp4")
    assert stem == "raw_data_cfr_test_2026-07-24"
    assert is_cfr is False


def test_read_dlc_windows_parses_numbers(tmp_path):
    csv_path = tmp_path / "windows.csv"
    csv_path.write_text(
        "task_id,label,video,h5,scorer,bodypart,start_frame,end_frame,n_frames,"
        "start_sec,end_sec,duration_sec,frac_above,mean_likelihood,mean_nose_dist,"
        "min_nose_dist,tongue_rate\n"
        "1,w000,/videos/raw_data_2026-07-24_12-02-14.mp4,/p.h5,SCORER,nose,"
        "2606,2799,193,21.717,23.325,1.608,0.1347,0.4471,45.37,20.29,7.119\n"
    )
    rows = read_dlc_windows(str(csv_path))
    assert len(rows) == 1
    assert rows[0]["label"] == "w000"
    assert rows[0]["video"] == "/videos/raw_data_2026-07-24_12-02-14.mp4"
    assert rows[0]["start_frame"] == 2606
    assert rows[0]["end_frame"] == 2799
    assert rows[0]["tongue_rate"] == 7.119


def test_read_dlc_windows_empty_tongue_rate_is_zero(tmp_path):
    # tongue_rate is written on every run, but a hand-edited or older CSV may leave it blank.
    csv_path = tmp_path / "windows.csv"
    csv_path.write_text(
        "task_id,label,video,start_frame,end_frame,tongue_rate\n"
        "1,w000,/videos/v.mp4,10,20,\n"
    )
    rows = read_dlc_windows(str(csv_path))
    assert rows[0]["tongue_rate"] == 0.0


def _fake_session_times(n_frames=100, latency=1.5, slope=1.002, fps=120.0):
    """Per-frame session times built the way the renderer builds them: absolute SensorTimestamps
    through a SessionClock that carries a bookmark latency and a drift slope."""
    pts_ns = (np.arange(n_frames) / fps * 1e9).astype(np.int64) + 123_456_789
    clock = SessionClock(pts_start_sec=0.0, latency=latency, slope=slope)
    return frame_session_times(clock, pts_ns)


def test_frame_window_to_session_matches_frame_session_times():
    # The window must be the session time of the frames DLC scored -- latency and drift included,
    # not frame/fps. sess[k] is the reference the renderer itself uses to place frame k.
    sess = _fake_session_times()
    start, end = frame_window_to_session(sess, 10, 20)
    assert start == sess[10]
    # end_frame is EXCLUSIVE in the DLC CSV: the last frame in the window is 19.
    assert end == sess[19]
    # And that is emphatically not frame/fps: the clock's latency alone shifts it by 1.5 s.
    assert start > 10 / 120.0 + 1.0


def test_frame_window_to_session_clamps_end_past_sidecar():
    sess = _fake_session_times(n_frames=50)
    start, end = frame_window_to_session(sess, 40, 999)
    assert start == sess[40]
    assert end == sess[-1]


def test_frame_window_to_session_returns_none_when_start_past_sidecar():
    sess = _fake_session_times(n_frames=50)
    assert frame_window_to_session(sess, 50, 60) is None


def test_frame_window_to_session_returns_none_for_negative_start_frame():
    # A negative index into `sess` does not fail -- it silently wraps to the END of the recording,
    # which would emit a clip from a completely different part of the session. Reject it.
    sess = _fake_session_times(n_frames=50)
    assert frame_window_to_session(sess, -5, 10) is None
    assert frame_window_to_session(sess, -1, 0) is None


def test_clamp_to_trace_clips_partial_overlap():
    assert clamp_to_trace(95.0, 110.0, 0.0, 100.0) == (95.0, 100.0)
    assert clamp_to_trace(-5.0, 10.0, 0.0, 100.0) == (0.0, 10.0)


def test_clamp_to_trace_returns_none_when_disjoint():
    assert clamp_to_trace(120.0, 130.0, 0.0, 100.0) is None
    assert clamp_to_trace(-20.0, -10.0, 0.0, 100.0) is None


def test_clamp_to_trace_returns_none_when_clamped_window_is_empty():
    # Touching the edge leaves no time to render.
    assert clamp_to_trace(100.0, 110.0, 0.0, 100.0) is None


def test_load_frame_session_times_missing_h5_returns_none():
    # Provenance pointing at a file that isn't on this machine is the normal case for an older
    # combined file, so it must degrade to "no commands for this video", never to a traceback.
    assert load_frame_session_times("/nonexistent/raw_data_2026-01-01_00-00-00.h5") is None


def test_load_frame_session_times_none_path_returns_none():
    assert load_frame_session_times(None) is None


def test_load_frame_session_times_no_video_sensor_returns_none(tmp_path):
    # A raw .h5 with no video sensor at all: read_video_anchor raises, and we swallow it.
    import h5py
    h5_path = tmp_path / "raw_data_2026-01-01_00-00-00.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_group("board0")
    assert load_frame_session_times(str(h5_path)) is None


def _write_raw_with_video_anchor(tmp_path, video_filename="raw_data_2026-01-02_00-00-00.mp4",
                                 frame_index=0):
    """A raw .h5 whose video anchor RESOLVES (video sensor group with the bookmark datasets
    read_video_anchor needs), so the sidecar lookup downstream of it is what gets exercised."""
    import h5py
    h5_path = tmp_path / "raw_data_2026-01-02_00-00-00.h5"
    with h5py.File(h5_path, "w") as f:
        group = f.create_group("board0").create_group("sensor_1")
        group.create_dataset("video_filename", data=np.bytes_(video_filename))
        group.create_dataset("video_frame_index", data=frame_index)
        group.create_dataset("time_data", data=np.array([0.0, 1.0]))
        group.create_dataset("start_time", data=0.0)
        group.create_dataset("stop_time", data=1.0)
    return h5_path


def test_load_frame_session_times_times_encoded_frames_not_captured_ones(tmp_path):
    # Container ordinal k -- which is what a DLC frame index IS -- indexes the ENCODED frames. When
    # the encoder dropped frames, the capture sidecar has more lines than the container has frames,
    # so timing frames off it would shift every window later and later through the recording. The
    # renderer's `load_container_pts` owns that choice; this pins that we go through it.
    h5_path = _write_raw_with_video_anchor(tmp_path, frame_index=1)
    capture = (np.arange(6) * 100_000_000).astype(np.int64)     # 6 captured frames, 0.1 s apart
    encoded = capture[[0, 1, 3, 5]]                             # the encoder kept only 4 of them
    np.savetxt(str(tmp_path / "raw_data_2026-01-02_00-00-00.txt"), capture, fmt="%d")
    np.savetxt(str(tmp_path / "raw_data_2026-01-02_00-00-00.encpts.txt"), encoded, fmt="%d")

    sess = load_frame_session_times(str(h5_path))
    assert sess is not None
    # One time per ENCODED frame, not per captured frame.
    assert sess.size == encoded.size
    # Bookmark frame 1 is session zero (0.1 s into the video file), so the encoded frames sit at
    # -0.1, 0.0, 0.2, 0.4 s.
    np.testing.assert_allclose(sess, [-0.1, 0.0, 0.2, 0.4], atol=1e-9)


def test_load_frame_session_times_missing_pts_sidecar_returns_none(tmp_path):
    # The anchor resolves, but the recording's <stem>.txt PTS sidecar is not on this machine (the
    # normal case for a combined file that names recordings kept elsewhere). No sidecar means no
    # per-frame times, which must degrade to None rather than a traceback.
    h5_path = _write_raw_with_video_anchor(tmp_path)
    from video.trimcrop import read_video_anchor, resolve_paths
    _, pts_txt = resolve_paths(str(h5_path), read_video_anchor(str(h5_path)))
    assert not os.path.exists(pts_txt)          # the branch under test is reachable
    assert load_frame_session_times(str(h5_path)) is None


# --- Positive path, gated on a real recording being present on this machine ------------------
# Mirrors tests/test_make_sync_video.py's `needs_reference`: the conversion this whole mode rests
# on can only be checked end-to-end against a real anchor + PTS sidecar. Without this, every
# load_frame_session_times test asserts None and a regression that dropped the SessionClock (or
# read the raw capture sidecar instead of the container PTS) would pass the suite.
REC_DIR = "Lickometry Data/ACG-26-3"
DLC_H5 = os.path.join(REC_DIR, "raw_data_2026-07-24_12-02-14.h5")
DLC_PTS = os.path.join(REC_DIR, "raw_data_2026-07-24_12-02-14.txt")

needs_dlc_reference = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (DLC_H5, DLC_PTS)),
    reason="reference recording files not present",
)


@needs_dlc_reference
def test_load_frame_session_times_matches_the_renderers_own_clock():
    from make_sync_video import load_container_pts
    from video.trimcrop import read_video_anchor, resolve_paths, session_clock

    got = load_frame_session_times(DLC_H5)
    assert got is not None

    anchor = read_video_anchor(DLC_H5)
    _, pts_txt = resolve_paths(DLC_H5, anchor)
    pts_ns = np.loadtxt(pts_txt, dtype=np.int64)
    expected = frame_session_times(session_clock(anchor, pts_ns),
                                   load_container_pts(pts_txt, pts_ns))

    # Same clock, same sidecar choice, same per-frame times the renderer will place frames with.
    np.testing.assert_array_equal(got, expected)
    # One session time per ENCODED container frame, strictly increasing (a frame that did not move
    # forward in time would make a window's start/end order meaningless).
    assert got.size == np.asarray(load_container_pts(pts_txt, pts_ns)).size
    assert np.all(np.diff(got) > 0)
    # And it is emphatically not frame/fps: the SessionClock shifts frame 0 off video-file zero.
    assert got[0] != 0.0


def _cycle_info(**overrides):
    info = {
        "cycle": 2,
        "raw_h5": "/data/raw_data_2026-07-24_12-02-14.h5",
        "layout": "/data/layout.csv",
        "animal": "ACG-26-3-1",
        "restart": False,
        "first_s": 0.0,
        "span_s": 1000.0,
        "lick_times": np.array([]),
    }
    info.update(overrides)
    return info


def _dlc_row(video="/videos/raw_data_2026-07-24_12-02-14.mp4", start_frame=10, end_frame=20,
             tongue_rate=5.0, label="w000"):
    return {"label": label, "video": video, "start_frame": start_frame,
            "end_frame": end_frame, "tongue_rate": tongue_rate, }


def test_build_dlc_rois_emits_one_row_per_window():
    sess = np.arange(1000, dtype=np.float64) / 10.0  # frame k at k/10 s
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info(lick_times=np.array([1.5]))}
    rows, skips = build_dlc_rois(
        [_dlc_row(start_frame=10, end_frame=20), _dlc_row(start_frame=30, end_frame=41)],
        cycles, sess_loader=lambda path: sess)

    assert [r["category"] for r in rows] == ["dlc", "dlc"]
    assert [r["rank"] for r in rows] == [0, 1]
    assert rows[0]["start"] == 1.0 and rows[0]["end"] == 1.9
    assert rows[1]["start"] == 3.0 and rows[1]["end"] == 4.0
    assert rows[0]["center"] == (1.0 + 1.9) / 2.0
    assert rows[0]["score"] == 5.0
    # the lick at 1.5 s falls inside the first window only
    assert rows[0]["n_licks_in_window"] == 1
    assert rows[1]["n_licks_in_window"] == 0
    assert rows[0]["animal"] == "ACG-26-3-1"
    assert rows[0]["cycle"] == 2
    assert rows[0]["filmed"] is True
    assert rows[0]["raw_h5"] == "/data/raw_data_2026-07-24_12-02-14.h5"
    assert rows[0]["layout"] == "/data/layout.csv"
    assert sum(skips.values()) == 0


def test_build_dlc_rois_skips_cfr_videos():
    # The _cfr files are re-encodes; their frame indices are not the original container's ordinals,
    # so a window built from them could be silently misaligned. Never emit one.
    cycles = {"raw_data_2026-07-13_11-59-47": _cycle_info()}
    rows, skips = build_dlc_rois(
        [_dlc_row(video="/videos/raw_data_2026-07-13_11-59-47_cfr.mp4")],
        cycles, sess_loader=lambda path: np.arange(1000, dtype=np.float64) / 10.0)
    assert rows == []
    assert skips["cfr_video"] == 1


def test_build_dlc_rois_skips_video_with_no_matching_cycle():
    rows, skips = build_dlc_rois(
        [_dlc_row(video="/videos/raw_data_1999-01-01_00-00-00.mp4")],
        {}, sess_loader=lambda path: np.arange(100, dtype=np.float64))
    assert rows == []
    assert skips["no_cycle"] == 1


def test_build_dlc_rois_skips_video_without_frame_times():
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    rows, skips = build_dlc_rois([_dlc_row(), _dlc_row(label="w001")], cycles,
                                 sess_loader=lambda path: None)
    assert rows == []
    # counted per ROW, so the report says how much output was lost, not how many files were missing
    assert skips["no_video_times"] == 2


def test_build_dlc_rois_loads_frame_times_once_per_video():
    calls = []

    def loader(path):
        calls.append(path)
        return np.arange(1000, dtype=np.float64) / 10.0

    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    rows, _ = build_dlc_rois([_dlc_row(), _dlc_row(label="w001", start_frame=30, end_frame=40)],
                             cycles, sess_loader=loader)
    assert len(rows) == 2
    assert calls == ["/data/raw_data_2026-07-24_12-02-14.h5"]


def test_build_dlc_rois_skips_row_past_end_of_sidecar():
    sess = np.arange(100, dtype=np.float64) / 10.0
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    rows, skips = build_dlc_rois([_dlc_row(start_frame=500, end_frame=510)], cycles,
                                 sess_loader=lambda path: sess)
    assert rows == []
    assert skips["frame_out_of_range"] == 1


def test_build_dlc_rois_clamps_and_drops_against_the_trace():
    sess = np.arange(1000, dtype=np.float64) / 10.0  # 0 .. 99.9 s
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info(first_s=0.0, span_s=50.0)}
    rows, skips = build_dlc_rois(
        [_dlc_row(start_frame=495, end_frame=520),   # 49.5 .. 51.9 -> clamped to 50.0
         _dlc_row(start_frame=700, end_frame=710, label="w001")],  # 70 .. 70.9 -> gone
        cycles, sess_loader=lambda path: sess)
    assert len(rows) == 1
    assert rows[0]["start"] == 49.5
    assert rows[0]["end"] == 50.0
    assert skips["outside_trace"] == 1


def test_build_dlc_rois_unfilmed_cycle_produces_unfilmed_rows():
    # No filmed animal (layout doesn't name the video sensor): the row is still reported in the
    # CSV, but write_shell_script will not emit a command for it.
    sess = np.arange(1000, dtype=np.float64) / 10.0
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info(animal=None)}
    rows, _ = build_dlc_rois([_dlc_row()], cycles, sess_loader=lambda path: sess)
    assert rows[0]["filmed"] is False
    assert rows[0]["animal"] == ""


def test_build_dlc_rois_carries_the_restart_flag():
    sess = np.arange(1000, dtype=np.float64) / 10.0
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info(restart=True)}
    rows, _ = build_dlc_rois([_dlc_row()], cycles, sess_loader=lambda path: sess)
    assert rows[0]["restart"] is True


def test_build_dlc_rois_ranks_within_each_video():
    sess = np.arange(1000, dtype=np.float64) / 10.0
    cycles = {
        "raw_data_2026-07-24_12-02-14": _cycle_info(cycle=2),
        "raw_data_2026-07-27_11-56-15": _cycle_info(
            cycle=3, raw_h5="/data/raw_data_2026-07-27_11-56-15.h5"),
    }
    rows, _ = build_dlc_rois(
        [_dlc_row(video="/videos/raw_data_2026-07-24_12-02-14.mp4"),
         _dlc_row(video="/videos/raw_data_2026-07-27_11-56-15.mp4"),
         _dlc_row(video="/videos/raw_data_2026-07-27_11-56-15.mp4", start_frame=30, end_frame=40)],
        cycles, sess_loader=lambda path: sess)
    ranks = {(r["cycle"], r["rank"]) for r in rows}
    assert ranks == {(2, 0), (3, 0), (3, 1)}


def test_build_dlc_rois_emits_no_lick_or_climb_rows():
    sess = np.arange(1000, dtype=np.float64) / 10.0
    cycles = {"raw_data_2026-07-24_12-02-14": _cycle_info()}
    rows, _ = build_dlc_rois([_dlc_row()], cycles, sess_loader=lambda path: sess)
    assert {r["category"] for r in rows} == {"dlc"}


def test_dlc_row_builds_a_cropped_command_with_speed():
    # Only "climb" gets --no-crop; a DLC clip is nose-at-sipper footage, where the crop is what
    # makes the tongue visible.
    row = {"animal": "ACG-26-3-1", "cycle": 2, "category": "dlc", "rank": 0,
           "start": 10.0, "end": 12.0, "raw_h5": "/data/raw.h5", "layout": "/data/layout.csv",
           "restart": False}
    lines = build_command(row, "clips", {}, "/data/combined.h5", speed=0.25)
    assert len(lines) == 1
    assert "--no-crop" not in lines[0]
    assert "--speed 0.25" in lines[0]
    assert "clips/ACG-26-3-1_c2_dlc0.mp4" in lines[0]


def _write_combined(tmp_path, raw_h5_path):
    """A minimal results_combined_*.h5: one animal, one cycle, provenance attrs, and the arrays
    build_cycles_for_dlc reads."""
    import h5py
    combined = tmp_path / "combined.h5"
    with h5py.File(combined, "w") as f:
        g = f.create_group("ACG-26-3-1").create_group("0")
        g.attrs["raw_h5"] = str(raw_h5_path)
        g.attrs["layout"] = "/data/layout.csv"
        g.create_dataset("time_data", data=np.linspace(0.0, 100.0, 1001))
        g.create_dataset("lick_times", data=np.array([5.0, 6.0]))
    return str(combined)


def test_build_cycles_for_dlc_keys_on_raw_stem(tmp_path, monkeypatch):
    import scripts.find_interesting_windows as fiw
    raw_h5 = tmp_path / "raw_data_2026-07-24_12-02-14.h5"
    raw_h5.write_text("")  # never opened: both readers below are stubbed
    combined = _write_combined(tmp_path, raw_h5)
    monkeypatch.setattr(fiw, "resolve_filmed_animal", lambda h5, layout: "ACG-26-3-1")
    monkeypatch.setattr(fiw, "is_restart_recording", lambda h5: False)

    cycles = build_cycles_for_dlc(combined)

    assert list(cycles) == ["raw_data_2026-07-24_12-02-14"]
    info = cycles["raw_data_2026-07-24_12-02-14"]
    assert info["cycle"] == 0
    assert info["animal"] == "ACG-26-3-1"
    assert info["layout"] == "/data/layout.csv"
    assert info["first_s"] == 0.0
    assert info["span_s"] == 100.0
    assert info["restart"] is False
    np.testing.assert_allclose(info["lick_times"], [5.0, 6.0])


def test_build_cycles_for_dlc_unfilmed_cycle_has_no_animal(tmp_path, monkeypatch):
    import scripts.find_interesting_windows as fiw
    raw_h5 = tmp_path / "raw_data_2026-07-24_12-02-14.h5"
    raw_h5.write_text("")
    combined = _write_combined(tmp_path, raw_h5)
    monkeypatch.setattr(fiw, "resolve_filmed_animal", lambda h5, layout: None)
    monkeypatch.setattr(fiw, "is_restart_recording", lambda h5: False)

    info = build_cycles_for_dlc(combined)["raw_data_2026-07-24_12-02-14"]
    assert info["animal"] is None
    # No filmed animal means no trace to draw either, so the bounds stay empty rather than
    # borrowing some other animal's.
    assert info["lick_times"].size == 0


def test_build_cycles_for_dlc_animals_filter_drops_other_cycles(tmp_path, monkeypatch):
    import scripts.find_interesting_windows as fiw
    raw_h5 = tmp_path / "raw_data_2026-07-24_12-02-14.h5"
    raw_h5.write_text("")
    combined = _write_combined(tmp_path, raw_h5)
    monkeypatch.setattr(fiw, "resolve_filmed_animal", lambda h5, layout: "ACG-26-3-1")
    monkeypatch.setattr(fiw, "is_restart_recording", lambda h5: False)

    assert build_cycles_for_dlc(combined, animals=["SOMEONE-ELSE"]) == {}


def test_main_dlc_mode_writes_csv_and_commands(tmp_path, monkeypatch):
    # End-to-end through main(): a DLC CSV plus a combined file becomes a .sh of make_sync_video
    # calls, with no lick/climb search anywhere in it.
    import scripts.find_interesting_windows as fiw

    raw_h5 = tmp_path / "raw_data_2026-07-24_12-02-14.h5"
    raw_h5.write_text("")
    combined = _write_combined(tmp_path, raw_h5)
    windows = tmp_path / "windows.csv"
    windows.write_text(
        "task_id,label,video,start_frame,end_frame,tongue_rate\n"
        "1,w000,/videos/raw_data_2026-07-24_12-02-14.mp4,100,200,7.1\n"
        "2,w001,/videos/raw_data_2026-07-13_11-59-47_cfr.mp4,100,200,7.1\n"
    )
    monkeypatch.setattr(fiw, "resolve_filmed_animal", lambda h5, layout: "ACG-26-3-1")
    monkeypatch.setattr(fiw, "is_restart_recording", lambda h5: False)
    # frame k at k/100 s, so window [100, 200) is 1.0 .. 1.99 s
    monkeypatch.setattr(fiw, "load_frame_session_times",
                        lambda path: np.arange(5000, dtype=np.float64) / 100.0)

    out_csv = tmp_path / "dlc_rois.csv"
    out_sh = tmp_path / "make_dlc_clips.sh"
    rc = fiw.main([combined, "--dlc-windows", str(windows),
                   "--csv", str(out_csv), "--sh", str(out_sh),
                   "--out-dir", str(tmp_path / "clips"), "--speed", "0.25"])
    assert rc == 0

    csv_text = out_csv.read_text()
    assert "dlc" in csv_text
    assert "climb" not in csv_text
    assert ",lick," not in csv_text

    sh_text = out_sh.read_text()
    commands = [ln for ln in sh_text.splitlines() if ln.startswith("python make_sync_video.py")]
    assert len(commands) == 1   # the _cfr row produced nothing
    assert "--start 1.000 --end 1.990" in sh_text
    assert "--speed 0.25" in sh_text
    assert "--no-crop" not in sh_text
    assert "--cycle 0" in sh_text
