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
                    'bout_lick_counts', 'bout_durations', 'bout_start_times'):
            assert key in g, key
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

    tc = core.timecourse(res, bin_s=60, length_s=res.recording_length)
    assert tc['licks'].shape == (12, 6)
    assert tc['licks'].sum() == res.animals['n_licks'].sum()

    ctrl = core.control_summary(res)
    assert list(ctrl['control'].unique()) == ["Control1"]

    corr = core.correlation_analysis(res)
    assert corr['ok'] and len(corr['x']) == 12
    excl = {(0, "T-1")}
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
    gui.excluded.set(frozenset({(0, "T-1")}))
    for comp in (gui.SummaryTab, gui.TracesTab, gui.CorrelationTab, gui.PerAnimalTab,
                 gui.CleanTab):
        solara.render(comp(res), handle_error=False)
    solara.render(gui.MetricsBlock(res, frozenset(), "", ""), handle_error=False)
