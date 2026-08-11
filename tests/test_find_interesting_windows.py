"""Unit tests for the pure logic in scripts/find_interesting_windows.py.

The HDF5 reading, provenance resolution, and shell-script writing are exercised by hand on real
recordings; only the deterministic, self-contained helpers (variance, masking, selection, window
construction, command formatting) are unit-tested here.
"""
import os
import sys

import numpy as np

# Make the repository root importable so `scripts.find_interesting_windows` resolves regardless of
# where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.find_interesting_windows import (  # noqa: E402
    sliding_variance, center_sample_indices, mask_bout_windows, select_climbing_centers,
    clip_window, count_licks_in_window, build_rois_for_cycle, parse_offsets, build_command,
    is_control,
    parse_dlc_video_stem, read_dlc_windows,
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
