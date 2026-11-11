"""End-to-end tests for analysis_core (the logic behind analysis_gui.py).

Uses small synthetic recordings written in the recorder's HDF5 layout
(scripts/make_synthetic_recording.py), since the example .h5 files in the repo
are Git LFS pointers.
"""
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis_core as core  # noqa: E402
from data_analysis import parse_time_fix, SENSOR_BOARD_MAP  # noqa: E402
from scripts.make_synthetic_recording import write_cohort  # noqa: E402

LAYOUT = {1: "T-1", 3: "T-2", 6: "T-3", 7: "T-4", 22: "Control1"}
HOURS = 0.1  # 6 minutes per session keeps the suite fast


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    base = tmp_path_factory.mktemp("data")
    cohort_dir = base / "COH"
    raws = write_cohort(str(cohort_dir), days=3, hours=HOURS, layout=LAYOUT, seed=1)
    return str(base), str(cohort_dir), raws


@pytest.fixture(scope="module")
def combined(cohort):
    base, _cohort_dir, raws = cohort
    lines = []
    path = core.run_pipeline(list(reversed(raws)), base, recording_length=HOURS * 3600,
                             log=lines.append)
    assert path is not None, "\n".join(lines)
    return path, raws, lines


def test_discovery(cohort):
    base, cohort_dir, raws = cohort
    assert core.discover_cohorts(base) == ["COH"]
    assert core.list_raw_files(cohort_dir) == sorted(raws)
    assert core.layout_ids(core.load_layout(cohort_dir)) == list(LAYOUT.values())


def test_compact_date_range():
    assert core.compact_date_range(["2026-01-20", "2026-01-21", "2026-02-03", "2027-01-01"]) \
        == "2026-01-20_21_02-03_2027-01-01"


def test_pipeline_writes_combined_file(combined):
    path, raws, lines = combined
    assert os.path.basename(path) == "results_combined_COH_2026-01-20_21_22.h5"
    with h5py.File(path) as f:
        assert sorted(f.keys()) == sorted(LAYOUT.values())
        # sessions ordered by date even though files were passed in reverse
        assert json.loads(f.attrs['files']) == sorted(raws)
        assert f.attrs['recording_length'] == pytest.approx(HOURS * 3600)
        g = f["T-1"]["0"]
        for key in ('consumed_vols', 'weights', 'lick_indices', 'lick_times', 'cap_data',
                    'time_data', 'used_start_idx', 'used_stop_idx', 'ILIs',
                    'bout_lick_counts', 'bout_durations', 'bout_start_times',
                    'lick_durations', 'lick_depths', 'lick_onsets', 'lick_offsets',
                    'bout_median_lick_durations'):
            assert key in g, key
        assert g['lick_durations'].shape == g['lick_indices'].shape
        assert g.attrs['raw_h5'] == sorted(raws)[0]
        assert g['lick_indices'].shape[0] > 0
        assert 'weights' not in f["Control1"]["0"]
    # intermediates removed by default
    assert not any(os.path.exists(core.filtered_path_for(r)) for r in raws)


def test_load_combined_and_analyses(combined, tmp_path):
    path, raws, _ = combined
    res = core.load_combined(path)
    assert res.animal_ids == ["T-1", "T-2", "T-3", "T-4"]
    assert res.control_ids == ["Control1"]
    assert res.dates == ["2026-01-20", "2026-01-21", "2026-01-22"]
    assert len(res.sessions) == 15
    assert (res.sessions['duration_s'] <= HOURS * 3600 + 1).all()

    det = core.detection_summary(res)
    assert det['x'].size == 12 and 0 <= det['r2'] <= 1

    m = core.pooled_metrics(res)
    s = core.metric_summary(m)
    # synthetic licks are at 7 Hz
    assert 130 < s['median_ili_ms'] < 155
    # synthetic dips are a graded 4-sample shape at 110 Hz -> ~20-40 ms at half depth
    assert 15 < s['median_lick_duration_ms'] < 45
    assert s['n_lick_durations'] > 0.9 * len(m['ILIs'])
    licked = ~res.sessions['is_control'] & (res.sessions['n_licks'] > 0)
    assert res.sessions.loc[licked, 'median_lick_duration_ms'].notna().all()

    tc = core.timecourse(res, bin_s=60, length_s=res.recording_length)
    assert tc['licks'].shape == (12, 6)
    assert tc['licks'].sum() == res.animals['n_licks'].sum()

    ctrl = core.control_summary(res)
    assert list(ctrl['control'].unique()) == ["Control1"]

    corr = core.correlation_analysis(res)
    assert corr['ok'] and len(corr['x']) == 12
    excl = {(0, "T-1", 0)}
    clean = core.refit_excluding(res, excl)
    assert clean['n_excluded'] == 1 and clean['x'].size == 11
    m2 = core.pooled_metrics(res, excl)
    assert len(m2['ILIs']) < len(m['ILIs'])

    table, fits, skipped = core.per_animal_fits(res)
    assert list(table.columns)[:2] == ['animal_id', 'n_sessions'] and len(table) == 4
    assert core.correlation_ceiling(res) is not None

    out = str(tmp_path)
    written = (core.export_licks_vs_vol(det, out) + core.export_metrics(m, out)
               + core.export_metrics(m2, out, "_outliers_removed")
               + core.export_cumulative(tc, out) + core.export_clean_regression(clean, out)
               + core.export_per_animal(table, out) + core.export_session_table(res, out))
    assert all(os.path.exists(p) for p in written)
    cum = pd.read_csv(os.path.join(out, "cliqr_cumulative_licks.csv"))
    assert list(cum.columns) == ["Time (min)", "Cumulative Licks", "Standard Error"]
    assert pd.read_csv(os.path.join(out, "cliqr_ILIs_outliers_removed.csv")).columns[0] == "ILI (ms)"
    assert pd.read_csv(os.path.join(out, "cliqr_lick_durations_outliers_removed.csv")).columns[0] \
        == "Lick Duration (ms)"


def test_session_window_decimates_and_keeps_dips(combined):
    path, _, _ = combined
    res = core.load_combined(path)
    with h5py.File(path) as f:
        cap = f["T-1"]["0"]["cap_data"][()]
        t = f["T-1"]["0"]["time_data"][()]
        n_licks = f["T-1"]["0"]["lick_indices"].shape[0]
    full = core.session_window(res, "T-1", 0, max_points=500)
    assert full['decimated'] and len(full['t']) <= 520
    # min/max envelope preserves the extremes of the whole trace
    assert full['y'].min() == cap.min() and full['y'].max() == cap.max()
    assert full['n_licks_window'] == n_licks
    # a short window is returned raw, with lick markers on the trace
    w = core.session_window(res, "T-1", 0, t0=60, t1=70, max_points=5000)
    assert not w['decimated']
    i0, i1 = np.searchsorted(t, 60), np.searchsorted(t, 70) + 1
    np.testing.assert_array_equal(w['y'], cap[i0:i1])
    assert np.isin(w['lick_y'], cap[i0:i1]).all()


def test_h5_searchsorted_matches_numpy(tmp_path):
    arr = np.cumsum(np.random.default_rng(0).uniform(0.001, 0.02, 10000))
    with h5py.File(tmp_path / "x.h5", "w") as f:
        ds = f.create_dataset("t", data=arr)
        for v in [-1, 0, arr[0], arr[123], arr[123] + 1e-6, arr[-1], arr[-1] + 1]:
            assert core.h5_searchsorted(ds, v) == np.searchsorted(arr, v)


def test_raw_sensor_overview(cohort):
    _, _, raws = cohort
    ov = core.raw_sensor_overview(raws[0], 1, max_points=1000)
    assert ov['board'] == SENSOR_BOARD_MAP[1]
    assert len(ov['cycles']) == 1
    c = ov['cycles'][0]
    assert c['duration_min'] == pytest.approx(HOURS * 60)
    assert ov['t_first'] < c['start_time'] < c['stop_time'] < ov['t_last']


def test_time_fix_is_applied(cohort, tmp_path):
    _, cohort_dir, raws = cohort
    raw = raws[0]
    with h5py.File(raw) as f:
        g = f[SENSOR_BOARD_MAP[1]]["sensor_1"]
        start = float(g["start_time"][()])
    layout = core.load_layout(cohort_dir)
    fix = pd.DataFrame({"Sensor": [1], "New Start Time": [start + 60],
                        "New End Time": [start + 180]})
    assert parse_time_fix(fix) == {1: (start + 60, start + 180)}
    fix_path = core.time_fix_path_for(raw)
    fix.to_excel(fix_path, index=False)
    try:
        logs = []
        core.process_raw_file(raw, layout, recording_length=HOURS * 3600, log=logs.append)
        with h5py.File(core.filtered_path_for(raw)) as f:
            fixed = f["T-1"]["time_data"][-1]
            other = f["T-2"]["time_data"][-1]
        assert fixed == pytest.approx(120, abs=0.1)        # 2 minutes, from the fix
        assert other == pytest.approx(HOURS * 3600, abs=1)  # untouched sensor
        assert any("time fix" in line.lower() for line in logs)

        core.process_raw_file(raw, layout, recording_length=HOURS * 3600,
                              apply_time_fix=False, log=logs.append)
        with h5py.File(core.filtered_path_for(raw)) as f:
            assert f["T-1"]["time_data"][-1] == pytest.approx(HOURS * 3600, abs=1)
    finally:
        os.remove(fix_path)
        os.remove(core.filtered_path_for(raw))


def test_gui_tabs_render(combined, cohort):
    """Every tab renders headlessly against real results (catches API/typo errors)."""
    solara = pytest.importorskip("solara")
    import analysis_gui as gui

    path, _, _ = combined
    base, _, _ = cohort
    res = core.load_combined(path)
    gui.base_dir.set(base)
    gui.selected_cohorts.set(["COH"])
    for comp in (gui.Page, gui.RunTab, gui.StartStopTab):
        solara.render(comp(), handle_error=False)
    gui.results.set(res)
    gui.excluded.set(frozenset({(0, "T-1", 0)}))
    for comp in (gui.SummaryTab, gui.TracesTab, gui.CorrelationTab, gui.PerAnimalTab,
                 gui.CleanTab):
        solara.render(comp(res), handle_error=False)
    solara.render(gui.MetricsBlock(res, frozenset(), "", ""), handle_error=False)


# ---------------------------------------------------------------------------
# 2-bottle choice
# ---------------------------------------------------------------------------
# The synthetic 2-bottle cohort licks fluid 2 ~3x more than fluid 1 and drains
# more of it (scripts/make_synthetic_recording.TWO_BOTTLE_FLUIDS).
NAMES = {1: "Water", 2: "Sucrose"}


@pytest.fixture(scope="module")
def cohort2(tmp_path_factory):
    base = tmp_path_factory.mktemp("data2")
    cohort_dir = base / "TB"
    raws = write_cohort(str(cohort_dir), days=3, hours=HOURS, layout=LAYOUT, seed=5,
                        two_bottle=True)
    return str(base), str(cohort_dir), raws


@pytest.fixture(scope="module")
def combined2(cohort2):
    base, _cohort_dir, raws = cohort2
    lines = []
    path = core.run_pipeline(raws, base, recording_length=HOURS * 3600, two_bottle=True,
                             fluid_names=NAMES, log=lines.append)
    assert path is not None, "\n".join(lines)
    return path, raws, lines


def test_two_bottle_combined_layout(combined2):
    path, raws, _ = combined2
    assert all(core.file_is_two_bottle(r) for r in raws)
    with h5py.File(path) as f:
        assert bool(f.attrs['two_bottle'])
        assert json.loads(f.attrs['fluid_names']) == {"1": "Water", "2": "Sucrose"}
        g = f["T-1"]["0"]
        assert 'weights' in g and 'consumed_vols' not in g
        for fl in ("fluid1", "fluid2"):
            for key in ('consumed_vols', 'lick_indices', 'lick_times', 'cap_data',
                        'time_data', 'ILIs', 'bout_lick_counts'):
                assert key in g[fl], (fl, key)
        assert 'weights' not in f["Control1"]["0"]
        # each bottle was detected on its own trace
        assert g["fluid1"]["lick_indices"].shape[0] < g["fluid2"]["lick_indices"].shape[0]


def test_two_bottle_mode_mismatch_is_skipped(cohort, cohort2):
    base1, _, raws1 = cohort
    _, _, raws2 = cohort2
    lines = []
    # 1-bottle files under a 2-bottle analysis (and vice versa) are skipped.
    assert core.run_pipeline(raws1[:1], base1, recording_length=HOURS * 3600,
                             two_bottle=True, log=lines.append) is None
    assert any("1-bottle mode" in line for line in lines)
    lines.clear()
    assert core.run_pipeline(raws2[:1], base1, recording_length=HOURS * 3600,
                             two_bottle=False, log=lines.append) is None
    assert any("2-bottle mode" in line for line in lines)


def test_two_bottle_analyses_split_by_fluid(combined2, tmp_path):
    path, _, _ = combined2
    res = core.load_combined(path)
    assert res.two_bottle and res.fluids == [1, 2]
    assert res.fluid_label(2) == "Sucrose"
    assert res.animal_ids == ["T-1", "T-2", "T-3", "T-4"]
    assert len(res.sessions) == 30  # 5 cages x 3 days x 2 bottles
    assert (res.sessions.groupby(['animal_id', 'session'])['weight'].nunique(dropna=False) == 1).all()

    for fl in (1, 2):
        det = core.detection_summary(res, fluid=fl)
        assert det['x'].size == 12
        corr = core.correlation_analysis(res, fluid=fl)
        assert corr['ok'] and len(corr['x']) == 12 and (corr['data']['fluid'] == fl).all()
        table, _fits, _sk = core.per_animal_fits(res, fluid=fl)
        assert len(table) == 4
        tc = core.timecourse(res, bin_s=60, length_s=res.recording_length, fluid=fl)
        assert tc['licks'].sum() == res.animals.query("fluid == @fl")['n_licks'].sum()
        w = core.session_window(res, "T-1", 0, fluid=fl, max_points=500)
        assert w['n_licks_window'] == res.sessions.query(
            "animal_id == 'T-1' and session == 0 and fluid == @fl")['n_licks'].item()

    # excluding one bottle of one session only touches that bottle
    excl = {(0, "T-1", 1)}
    assert core.refit_excluding(res, excl, fluid=1)['n_excluded'] == 1
    assert core.refit_excluding(res, excl, fluid=2)['n_excluded'] == 0
    full = core.pooled_metrics(res, fluid=1)
    assert len(core.pooled_metrics(res, excl, fluid=1)['ILIs']) < len(full['ILIs'])
    assert len(core.pooled_metrics(res, excl, fluid=2)['ILIs']) == \
        len(core.pooled_metrics(res, fluid=2)['ILIs'])

    ctrl = core.control_summary(res)
    assert sorted(ctrl['fluid'].unique()) == ["Sucrose", "Water"]

    # --- preference ---------------------------------------------------------
    pt = core.preference_table(res, target=2)
    assert len(pt) == 12
    row = pt.iloc[0]
    t, o = row['licks_f2'], row['licks_f1']
    assert row['licks_diff'] == t - o
    assert row['licks_pref'] == pytest.approx(t / (t + o))
    assert row['licks_di'] == pytest.approx((t - o) / (t + o))
    assert (pt['vol_pref'] > 0.5).all() and pt['licks_pref'].mean() > 0.5
    # flipping the target mirrors the ratios
    pt1 = core.preference_table(res, target=1)
    np.testing.assert_allclose(pt1['vol_pref'], 1 - pt['vol_pref'])
    # excluding either bottle drops that whole session from the preference
    assert len(core.preference_table(res, excl, target=2)) == 11

    summ = core.preference_summary(pt, target=2)
    assert list(summ['per_animal']['animal_id']) == res.animal_ids
    vol_t = summ['tests'].query("measure == 'Volume (mL)'")
    assert (vol_t['n_animals'] == 4).all()
    assert vol_t['mean_target'].iloc[0] > vol_t['mean_other'].iloc[0]
    assert vol_t.query("test.str.startswith('paired')", engine='python')['p'].item() < 0.05

    by_s = core.preference_by_session(pt)
    assert list(by_s['date']) == res.dates

    ptc = core.preference_timecourse(res, bin_s=60, length_s=res.recording_length, target=2)
    assert ptc['cum_licks'][2].shape == (12, 6)
    np.testing.assert_allclose(ptc['cum_licks'][1][:, -1] + ptc['cum_licks'][2][:, -1],
                               pt['licks_total'].to_numpy())

    out = str(tmp_path)
    written = (core.export_preference(pt, summ, out)
               + core.export_metrics(full, out, tag=core.fluid_tag(1))
               + core.export_session_table(res, out))
    assert all(os.path.exists(p) for p in written)
    assert os.path.exists(os.path.join(out, "cliqr_ILIs_fluid1.csv"))
    st = pd.read_csv(os.path.join(out, "cliqr_session_summary.csv"))
    assert set(st['fluid_name']) == {"Water", "Sucrose"}


def test_preference_ratio_clips_negative_volumes():
    # A bottle that reads fuller afterwards counts as 0 consumed, not negative.
    np.testing.assert_allclose(core._ratio([0.4, -0.1, 0.0], [-0.1, 0.4, 0.0]),
                               [1.0, 0.0, np.nan])


def test_two_bottle_missing_weight_policy(cohort2, tmp_path):
    _, cohort_dir, raws = cohort2
    raw = str(tmp_path / os.path.basename(raws[0]))
    import shutil
    shutil.copy(raws[0], raw)
    with h5py.File(raw, "r+") as f:
        del f[SENSOR_BOARD_MAP[1]]["sensor_1"]["weight"]
    layout = core.load_layout(cohort_dir)
    try:
        assert core.process_raw_file(raw, layout, recording_length=HOURS * 3600,
                                     log=lambda *_: None) is False
        with h5py.File(core.filtered_path_for(raw)) as f:
            assert "T-1/fluid1" not in f and "T-1/fluid2" not in f  # weight required by default
        core.process_raw_file(raw, layout, recording_length=HOURS * 3600,
                              require_weight=False, log=lambda *_: None)
        with h5py.File(core.filtered_path_for(raw)) as f:
            assert np.isnan(f["T-1/fluid1/weight"][()])
            assert "lick_times" in f["T-1/fluid2"]
    finally:
        os.remove(core.filtered_path_for(raw))


def test_raw_sensor_overview_two_bottle(cohort2):
    _, _, raws = cohort2
    ov1 = core.raw_sensor_overview(raws[0], 1, max_points=1000)
    ov2 = core.raw_sensor_overview(raws[0], 1, max_points=1000, fluid=2)
    assert ov1['two_bottle'] and ov1['fluid'] == 1 and ov2['fluid'] == 2
    assert len(ov1['cycles']) == 1 and ov1['cycles'] == ov2['cycles']


def test_gui_tabs_render_two_bottle(combined2, cohort2):
    solara = pytest.importorskip("solara")
    import analysis_gui as gui

    path, _, _ = combined2
    base, _, _ = cohort2
    res = core.load_combined(path)
    gui.base_dir.set(base)
    gui.selected_cohorts.set(["TB"])
    gui.two_bottle.set(True)
    for comp in (gui.Page, gui.RunTab, gui.StartStopTab):
        solara.render(comp(), handle_error=False)
    gui.results.set(res)
    gui.excluded.set(frozenset({(0, "T-1", 2)}))
    for comp in (gui.SummaryTab, gui.TracesTab, gui.CorrelationTab, gui.PerAnimalTab,
                 gui.CleanTab):
        solara.render(comp(res), handle_error=False)
    solara.render(gui.MetricsBlock(res, frozenset(), "", ""), handle_error=False)
    gui.two_bottle.set(False)


def test_lick_durations_computed_for_old_combined_files(combined, tmp_path):
    """Combined files made before lick durations existed get them on the fly."""
    import shutil
    path, _, _ = combined
    old = str(tmp_path / "results_combined_old.h5")
    shutil.copy(path, old)
    with h5py.File(old, 'a') as f:
        for a in f:
            for k in f[a]:
                for key in ('lick_durations', 'lick_depths', 'lick_onsets', 'lick_offsets',
                            'bout_median_lick_durations'):
                    if key in f[a][k]:
                        del f[a][k][key]
    new_m = core.pooled_metrics(core.load_combined(path))
    old_res = core.load_combined(old)
    old_m = core.pooled_metrics(old_res)
    np.testing.assert_allclose(np.sort(old_m['lick_durations']), np.sort(new_m['lick_durations']))
    assert old_res.sessions['median_lick_duration_ms'].isna().all()


def test_lick_events_table_and_export(combined, tmp_path):
    path, _, _ = combined
    res = core.load_combined(path)
    ev = core.lick_events(res)
    assert list(ev.columns) == ['animal_id', 'session', 'date', 'lick_number', 'detected_s',
                                'start_s', 'end_s', 'duration_ms', 'start_lead_ms',
                                'depth_counts', 'duration_valid']
    # every non-control lick, nothing from controls
    assert len(ev) == res.animals['n_licks'].sum()
    assert not ev['animal_id'].str.startswith('Control').any()
    ok = ev['duration_valid']
    assert ok.mean() > 0.9
    # start before the detected (deepest) sample, end after it
    assert (ev.loc[ok, 'start_s'] < ev.loc[ok, 'detected_s']).all()
    assert (ev.loc[ok, 'end_s'] > ev.loc[ok, 'detected_s']).all()
    assert 0 < ev.loc[ok, 'start_lead_ms'].median() < 40
    # exclusions drop that session's licks
    excl = {(0, "T-1", 0)}
    ev2 = core.lick_events(res, excl)
    assert not ((ev2['animal_id'] == "T-1") & (ev2['session'] == 0)).any()
    p, = core.export_lick_events(ev, str(tmp_path), "_outliers_removed")
    assert os.path.basename(p) == "cliqr_lick_events_outliers_removed.csv"
    back = pd.read_csv(p)
    assert len(back) == len(ev)
    np.testing.assert_allclose(back['start_s'], ev['start_s'], atol=1e-4)


def test_lick_events_two_bottle(combined2):
    path, _, _ = combined2
    res = core.load_combined(path)
    ev = core.lick_events(res)
    assert 'fluid' in ev.columns and set(ev['fluid']) <= {1, 2}
    only1 = core.lick_events(res, fluid=1)
    assert set(only1['fluid']) == {1}
    assert len(only1) == res.animals.query("fluid == 1")['n_licks'].sum()
