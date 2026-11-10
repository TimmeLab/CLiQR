"""
CLiQR Data Analysis - Solara GUI

Replaces DataAnalysis.ipynb for batch analysis. Built for long recordings
(24 h runs, ~5 GB/day): lick detection streams one sensor at a time and runs
in a background thread, and every trace plot reads only the visible window,
min/max-decimated to a few thousand points.

Run with:

    solara run analysis_gui.py        # opens http://localhost:8765

(or double-click start_analysis.bat on the lab PC). If the recording GUI is
already running on 8765, add  --port 8766.

All analysis logic lives in analysis_core.py (no GUI code); this file only
wires widgets to it.

Author: Christopher Parker (parkecp@ucmail.uc.edu)
Lab: Timme Lab, University of Cincinnati
"""
import datetime
import os
import threading

import numpy as np
import pandas as pd
import solara
import solara.lab  # noqa: F401  (solara.lab.task)
from matplotlib.figure import Figure

import analysis_core as core

# ---------------------------------------------------------------------------
# State (module-level reactives are per browser session in Solara)
# ---------------------------------------------------------------------------
base_dir = solara.reactive(os.environ.get("CLIQR_DATA_DIR", "Lickometry Data"))
selected_cohorts = solara.reactive([])
selected_files = solara.reactive([])
algorithm = solara.reactive("basic_threshold")
recording_hours = solara.reactive(2.0)
apply_time_fix = solara.reactive(True)
keep_intermediate = solara.reactive(False)
export_dir = solara.reactive(".")
bin_minutes = solara.reactive(5.0)

log_lines = solara.reactive([])
progress = solara.reactive((0.0, ""))
results = solara.reactive(None)          # core.CombinedResults
excluded = solara.reactive(frozenset())  # {(session, animal_id)} dropped as outliers
tab = solara.reactive("Run")

_stop_event = threading.Event()
MAX_LOG_LINES = 3000
PLOT_POINTS = 4000
BLUE, RED, GREY = "#1f77b4", "#d62728", "#7f7f7f"


def add_log(message: str):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_lines.set((log_lines.value + [f"[{stamp}] {message}"])[-MAX_LOG_LINES:])


def set_results(res):
    results.set(res)
    excluded.set(frozenset())
    if res is not None:
        tab.set("Summary")
        add_log(f"Loaded {os.path.basename(res.path)}: {res.sessions['session'].nunique()} "
                f"session(s), {len(res.animal_ids)} animal(s), {len(res.control_ids)} control(s)")


def export(paths_fn, what: str):
    out = export_dir.value or "."
    try:
        os.makedirs(out, exist_ok=True)
        paths = paths_fn(out)
    except Exception as exc:
        add_log(f"ERROR exporting {what}: {exc!r}")
        return
    add_log(f"Saved {what}: " + ", ".join(os.path.basename(p) for p in paths)
            + f"  (in {os.path.abspath(out)})")


# ---------------------------------------------------------------------------
# Background run
# ---------------------------------------------------------------------------
@solara.lab.task
def run_analysis_task():
    _stop_event.clear()
    files = list(selected_files.value)
    add_log(f"Starting analysis of {len(files)} file(s) with {algorithm.value}, "
            f"recording length {recording_hours.value:g} h")

    def _progress(frac, msg):
        progress.set((float(frac), msg))

    path = core.run_pipeline(
        files, base_dir.value,
        algorithm=algorithm.value,
        recording_length=recording_hours.value * 3600,
        apply_time_fix=apply_time_fix.value,
        keep_intermediate=keep_intermediate.value,
        log=add_log, progress=_progress, should_stop=_stop_event.is_set,
    )
    if path:
        set_results(core.load_combined(path))
    return path


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _time_unit(seconds: float):
    """Pick hours for long spans (24 h runs), minutes otherwise."""
    return (3600.0, "h") if seconds > 3 * 3600 else (60.0, "min")


def _fig(w=7.0, h=3.2):
    fig = Figure(figsize=(w, h), dpi=100)
    return fig, fig.add_subplot(111)


def Plot(fig, deps):
    fig.tight_layout()
    return solara.FigureMatplotlib(fig, dependencies=deps, format="png")


def _hist(ax, data, bins, title, xlabel, x_range=None):
    kw = {'range': x_range} if x_range else {}
    if len(data):
        ax.hist(data, bins=bins, color=BLUE, edgecolor="white", **kw)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Count")


def _band(ax, x, mat, label, color=BLUE):
    mean, sem = core.mean_sem(mat)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.3, lw=0)
    ax.plot(x, mean, "-o", color=color, ms=3, lw=1.5, label=label)


def _trace_plot(w, title, unit=None, vlines=(), t_offset=0.0):
    """Plot a (possibly decimated) window from analysis_core.read_window/session_window."""
    fig, ax = _fig(10, 3.2)
    span = (w['t'][-1] - w['t'][0]) if len(w['t']) else 0
    scale, uname = unit or _time_unit(span)
    t = (w['t'] - t_offset) / scale
    ax.plot(t, w['y'], color=BLUE, lw=0.6)
    if len(w.get('lick_t', [])):
        ax.plot((np.asarray(w['lick_t']) - t_offset) / scale, w['lick_y'], "o", color=RED,
                ms=3, label=f"Licks ({len(w['lick_t'])})")
        ax.legend(loc="lower left", fontsize=8)
    for x, color, label in vlines:
        ax.axvline((x - t_offset) / scale, color=color, lw=1.2, ls="--", label=label)
    if vlines:
        ax.legend(loc="lower left", fontsize=8)
    note = f" — decimated ×{w['k']}" if w.get('decimated') else ""
    ax.set_title(title + note, fontsize=10)
    ax.set_xlabel(f"Time ({uname})"); ax.set_ylabel("Capacitance (counts)")
    return fig


# ---------------------------------------------------------------------------
# Components: setup / run
# ---------------------------------------------------------------------------
@solara.component
def FileSelection():
    cohorts = solara.use_memo(lambda: core.discover_cohorts(base_dir.value), [base_dir.value])
    with solara.Card("1. Data"):
        solara.InputText("Base data directory", value=base_dir, continuous_update=False)
        if not cohorts:
            solara.Warning(f"No cohort folders with a layout file found in '{base_dir.value}'.")
            return
        solara.SelectMultiple("Cohorts", values=selected_cohorts, all_values=cohorts)
        for cohort in selected_cohorts.value:
            cohort_dir = os.path.join(base_dir.value, cohort)
            files = core.list_raw_files(cohort_dir)
            with solara.Details(f"{cohort}: {len(files)} raw file(s)", expand=True):
                with solara.Row():
                    solara.Button("Select all", text=True, on_click=lambda f=files: selected_files.set(
                        sorted(set(selected_files.value) | set(f))))
                    solara.Button("Select none", text=True, on_click=lambda f=files: selected_files.set(
                        [x for x in selected_files.value if x not in set(f)]))
                for f in files:
                    def toggle(on, f=f):
                        cur = set(selected_files.value)
                        cur.add(f) if on else cur.discard(f)
                        selected_files.set(sorted(cur))
                    tf = " · time fix" if os.path.exists(core.time_fix_path_for(f)) else ""
                    solara.Checkbox(label=f"{os.path.basename(f)}  ({core.file_size_gb(f):.2f} GB{tf})",
                                    value=f in selected_files.value, on_value=toggle)
        n = len(selected_files.value)
        total = sum(core.file_size_gb(f) for f in selected_files.value)
        solara.Text(f"{n} file(s) selected, {total:.2f} GB total")


@solara.component
def RunOptions():
    running = run_analysis_task.pending
    with solara.Card("2. Detect licks"):
        with solara.Row(style={"flex-wrap": "wrap", "gap": "16px", "align-items": "center"}):
            solara.Select("Algorithm", value=algorithm, values=list(core.ALGORITHMS),
                          style={"min-width": "220px"})
            solara.InputFloat("Recording length (h)", value=recording_hours,
                              style={"max-width": "180px"})
            solara.Checkbox(label="Apply *_time_fix.xlsx files", value=apply_time_fix)
            solara.Checkbox(label="Keep intermediate filtered_*.h5", value=keep_intermediate)
        solara.Markdown(
            f"*{core.ALGORITHMS[algorithm.value]}.* Each sensor is trimmed to its last "
            f"start/stop cycle and then to **{recording_hours.value:g} h** "
            "(use 24 for day-long runs).")
        with solara.Row():
            solara.Button("Run analysis", color="primary", icon_name="mdi-play",
                          disabled=running or not selected_files.value,
                          on_click=run_analysis_task)
            solara.Button("Stop after current file", icon_name="mdi-stop",
                          disabled=not running, on_click=_stop_event.set)
        if running:
            frac, msg = progress.value
            solara.ProgressLinear(value=frac * 100)
            solara.Text(msg, style={"font-size": "12px"})
        elif run_analysis_task.error:
            solara.Error(f"Analysis failed: {run_analysis_task.exception!r}")
        elif run_analysis_task.finished and run_analysis_task.value:
            solara.Success(f"Done: {os.path.basename(run_analysis_task.value)}")


@solara.component
def OpenExisting():
    files = solara.use_memo(lambda: core.list_combined_files(base_dir.value),
                            [base_dir.value, results.value, run_analysis_task.finished])
    choice = solara.use_reactive(None)
    with solara.Card("…or open existing results"):
        if not files:
            solara.Text(f"No results_combined_*.h5 in '{base_dir.value}' yet.")
            return
        current = choice.value if choice.value in files else files[0]
        names = {os.path.basename(f): f for f in files}
        solara.Select("Combined results file (newest first)", value=os.path.basename(current),
                      values=list(names), on_value=lambda n: choice.set(names[n]),
                      style={"min-width": "420px"})

        def _load():
            try:
                set_results(core.load_combined(current))
            except Exception as exc:
                add_log(f"ERROR loading {current}: {exc!r}")
        solara.Button("Load", color="primary", on_click=_load,
                      disabled=run_analysis_task.pending)


@solara.component
def LogCard():
    with solara.Card("Activity log"):
        with solara.Row():
            solara.InputText("Export folder for CSVs", value=export_dir, style={"min-width": "320px"})
            solara.Button("Clear log", text=True, on_click=lambda: log_lines.set([]))
        with solara.Column(style={"max-height": "260px", "overflow-y": "auto"}):
            solara.Preformatted("\n".join(reversed(log_lines.value[-400:])) or "(empty)")


@solara.component
def RunTab():
    with solara.Columns([1, 1]):
        with solara.Column():
            FileSelection()
        with solara.Column():
            RunOptions()
            OpenExisting()
    LogCard()


# ---------------------------------------------------------------------------
# Start/stop inspector (raw files)
# ---------------------------------------------------------------------------
@solara.component
def StartStopTab():
    cohorts = selected_cohorts.value or core.discover_cohorts(base_dir.value)
    raw_files = [f for c in cohorts
                 for f in core.list_raw_files(os.path.join(base_dir.value, c))]
    raw_choice = solara.use_reactive(None)
    sensor_choice = solara.use_reactive(None)
    window_state = solara.use_reactive((None, None))  # (key, (min, min) from file start)

    solara.Markdown(
        "Inspect a raw recording to check its start/stop times. If they are wrong, create "
        "`<raw_filename>_time_fix.xlsx` next to it with columns **Sensor | New Start Time | "
        "New End Time** (Unix seconds, shown in the table below); it is applied on the next run.")
    if not raw_files:
        solara.Info(f"No raw_data_*.h5 files found under '{base_dir.value}'.")
        return
    raw_file = raw_choice.value if raw_choice.value in raw_files else raw_files[0]
    layout = _safe(core.load_layout, os.path.dirname(raw_file))
    if isinstance(layout, Exception):
        solara.Error(str(layout))
        return
    sensors = {f"Sensor {s} — {a}": int(s) for s, a in zip(layout.index, layout.iloc[:, 0])}
    sensor = sensor_choice.value if sensor_choice.value in sensors else next(iter(sensors))

    with solara.Row(style={"flex-wrap": "wrap", "gap": "16px"}):
        labels = {os.path.relpath(f, base_dir.value): f for f in raw_files}
        solara.Select("Raw file", value=os.path.relpath(raw_file, base_dir.value),
                      values=list(labels), on_value=lambda n: raw_choice.set(labels[n]),
                      style={"min-width": "420px"})
        solara.Select("Sensor", value=sensor, values=list(sensors), on_value=sensor_choice.set,
                      style={"min-width": "240px"})
    RawSensorView(raw_file, sensors[sensor], sensor, window_state)


@solara.component
def RawSensorView(raw_file, sensor_num, sensor_label, window_state):
    key = (raw_file, sensor_num)
    overview = solara.use_memo(
        lambda: _safe(core.raw_sensor_overview, raw_file, sensor_num, max_points=PLOT_POINTS),
        [raw_file, sensor_num])
    ok = not isinstance(overview, Exception)
    t_first = overview['t_first'] if ok else 0.0
    total_min = max((overview['t_last'] - t_first) / 60, 0.01) if ok else 1.0
    window = window_state.value[1] if window_state.value[0] == key else (0.0, total_min)
    zoomed = solara.use_memo(
        lambda: overview if (not ok or window == (0.0, total_min)) else _safe(
            core.raw_sensor_overview, raw_file, sensor_num, t_first + window[0] * 60,
            t_first + window[1] * 60, max_points=PLOT_POINTS),
        [raw_file, sensor_num, window])
    for r in (overview, zoomed):
        if isinstance(r, Exception):
            solara.Error(str(r))
            return
    sensor = sensor_label

    solara.SliderRangeFloat("View window (minutes from file start)", value=window,
                            on_value=lambda v: window_state.set((key, tuple(v))),
                            min=0.0, max=float(total_min), step=1 / 60)  # 1 s steps
    vlines = []
    for c in overview['cycles']:
        tag = "" if c['cycle'] < 0 else f" {c['cycle']}"
        vlines.append((c['start_time'], "green", f"start{tag}"))
        if not np.isnan(c['stop_time']):
            vlines.append((c['stop_time'], RED, f"stop{tag}"))
    lo, hi = t_first + window[0] * 60, t_first + window[1] * 60
    vlines = [v for v in vlines if lo <= v[0] <= hi]
    fig = _trace_plot(zoomed, f"{os.path.basename(raw_file)} — {sensor} ({overview['board']})",
                      unit=_time_unit((hi - lo)), vlines=vlines, t_offset=t_first)
    Plot(fig, [raw_file, sensor, window])

    if overview['cycles']:
        df = pd.DataFrame(overview['cycles'])
        for col in ('start_time', 'stop_time'):
            df[col.replace('_time', ' (local)')] = [
                datetime.datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S")
                if not np.isnan(v) else "—" for v in df[col]]
        solara.DataFrame(df, items_per_page=10)
    else:
        solara.Warning("No start/stop times recorded for this sensor; the whole trace would be "
                       "used. Consider a time fix file.")
    solara.Text(f"File spans {datetime.datetime.fromtimestamp(t_first):%Y-%m-%d %H:%M:%S} → "
                f"{datetime.datetime.fromtimestamp(overview['t_last']):%H:%M:%S} "
                f"(Unix {t_first:.0f} – {overview['t_last']:.0f})")


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return exc


# ---------------------------------------------------------------------------
# Results tabs
# ---------------------------------------------------------------------------
@solara.component
def SummaryTab(res):
    det = solara.use_memo(lambda: core.detection_summary(res), [res])
    controls = solara.use_memo(lambda: core.control_summary(res), [res])

    with solara.Card("Detection summary"):
        solara.Markdown(
            f"**{len(res.animal_ids)}** animals · **{res.sessions['session'].nunique()}** sessions · "
            f"recording length **{res.recording_length / 3600:g} h** · algorithm "
            f"`{res.attrs.get('algorithm', '?')}`\n\n"
            f"CLiQR: **{det['mean']:.0f} ± {det['sd']:.0f}** licks/session (mean ± SD) · "
            f"lick count vs volume **R² = {det['r2']:.4f}**")
        fig, ax = _fig(5, 4)
        ax.scatter(det['x'], det['y'], s=18, color=BLUE, alpha=0.7)
        if not np.isnan(det['slope']) and len(det['x']):
            xs = np.array([det['x'].min(), det['x'].max()])
            ax.plot(xs, det['intercept'] + det['slope'] * xs, color=GREY, lw=1)
        ax.set_title(f"CLiQR (R²={det['r2']:.3f})"); ax.set_xlabel("Lick Count")
        ax.set_ylabel("Volume Consumed (mL)")
        Plot(fig, [res])
        solara.Button("Export cliqr_licks_vs_vol.csv", text=True,
                      on_click=lambda: export(lambda d: core.export_licks_vs_vol(det, d),
                                              "licks vs volume"))

    with solara.Card("Control cages"):
        solara.Markdown("Controls have sippers but no animal: any detected licks are false "
                        "positives and any volume change is evaporation/leak. They are excluded "
                        "from all other statistics.")
        if controls.empty:
            solara.Text("No control cages in this results file.")
        else:
            solara.Markdown(
                f"False-positive licks / control-session: mean "
                f"**{controls['false_positive_licks'].mean():.1f}**, max "
                f"**{controls['false_positive_licks'].max()}** · evaporation: mean "
                f"**{controls['evaporation_mL'].mean():.3f} mL**, max "
                f"**{controls['evaporation_mL'].max():.3f} mL**")
            fig, ax = _fig(max(5, 0.5 * len(controls)), 3)
            labels = [f"{r.control} ({r.date})" for r in controls.itertuples()]
            ax.bar(range(len(labels)), controls['false_positive_licks'], color="firebrick")
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("False-positive licks")
            Plot(fig, [res])
            solara.DataFrame(controls, items_per_page=10)

    with solara.Card("All sessions"):
        cols = ['animal_id', 'date', 'session', 'n_licks', 'n_bouts', 'consumed_vol', 'weight',
                'duration_s']
        solara.DataFrame(res.sessions[cols], items_per_page=15)
        solara.Button("Export cliqr_session_summary.csv", text=True,
                      on_click=lambda: export(lambda d: core.export_session_table(res, d),
                                              "session table"))


@solara.component
def MetricsBlock(res, excl, suffix, title):
    """Behavioral metric histograms + time course (shared by the all-data and
    outliers-removed tabs)."""
    bin_s = max(float(bin_minutes.value), 0.5) * 60
    length_s = res.recording_length
    m = solara.use_memo(lambda: core.pooled_metrics(res, excl), [res, excl])
    tc = solara.use_memo(lambda: core.timecourse(res, bin_s=bin_s, length_s=length_s,
                                                 exclude=excl), [res, excl, bin_s])
    s = core.metric_summary(m)

    with solara.Card(f"Behavioral metrics{title}"):
        solara.Markdown(
            f"Median ILI **{s['median_ili_ms']:.1f} ms** ({s['median_ili_hz']:.1f} Hz) · "
            f"mean licks/bout **{s['mean_licks_per_bout']:.1f}** (median "
            f"{s['median_licks_per_bout']:.0f}) · mean bout duration "
            f"**{s['mean_bout_duration']:.2f} s** (median {s['median_bout_duration']:.2f} s) · "
            f"{s['n_bouts']} bouts")
        fig = Figure(figsize=(13, 3.2), dpi=100)
        a1, a2, a3 = fig.subplots(1, 3)
        _hist(a1, m['ILIs'] * 1000, 100, "ILI", "ILI (ms)", (0, 1000))
        max_b = int(m['bout_lick_counts'].max()) if len(m['bout_lick_counts']) else 1
        _hist(a2, m['bout_lick_counts'], np.arange(1, min(max_b, 300) + 2), "Licks per Bout",
              "Licks per Bout")
        _hist(a3, m['bout_durations'], 50, "Bout Duration", "Duration (s)")
        Plot(fig, [res, excl])
        solara.Button("Export ILI / bout CSVs", text=True,
                      on_click=lambda: export(lambda d: core.export_metrics(m, d, suffix),
                                              "behavioral metrics"))

    with solara.Card(f"Temporal dynamics{title}"):
        solara.InputFloat("Bin size (minutes)", value=bin_minutes, style={"max-width": "200px"})
        scale, uname = _time_unit(length_s)
        x = tc['centers_min'] * 60 / scale
        solara.Text(f"{tc['licks'].shape[0]} sessions, mean ± SEM, "
                    f"{len(x)} bins over {length_s / 3600:g} h")
        fig = Figure(figsize=(13, 3.4), dpi=100)
        (b1, b2) = fig.subplots(1, 2)
        if tc['licks'].shape[0]:
            _band(b1, x, tc['licks'], "Licks")
            _band(b2, x, np.cumsum(tc['licks'], axis=1), "Licks")
        if tc['bouts'].shape[0]:
            _band(b1, x, tc['bouts'], "Bouts", color="#ff7f0e")
            _band(b2, x, np.cumsum(tc['bouts'], axis=1), "Bouts", color="#ff7f0e")
        b1.set_title(f"Per {bin_s / 60:g}-min bin"); b2.set_title("Cumulative")
        for ax in (b1, b2):
            ax.set_xlabel(f"Time ({uname})"); ax.legend(fontsize=8)
        b1.set_ylabel("Count per bin"); b2.set_ylabel("Cumulative count")
        Plot(fig, [res, excl, bin_s])
        solara.Button("Export cumulative CSVs", text=True,
                      on_click=lambda: export(lambda d: core.export_cumulative(tc, d, suffix),
                                              "cumulative licks/bouts"))


@solara.component
def TracesTab(res):
    animals = res.animal_ids + res.control_ids
    animal_choice = solara.use_reactive(None)
    session_choice = solara.use_reactive(None)
    window_state = solara.use_reactive((None, None))  # (key, (lo, hi)) in display units
    if not animals:
        solara.Info("No sessions in this results file.")
        return
    animal = animal_choice.value if animal_choice.value in animals else animals[0]
    sess = res.sessions[res.sessions['animal_id'] == animal]
    labels = {f"{r.date} (#{r.session}) — {r.n_licks} licks": int(r.session)
              for r in sess.itertuples()}
    session = session_choice.value if session_choice.value in labels else next(iter(labels))
    s_idx = labels[session]
    row = sess[sess['session'] == s_idx].iloc[0]
    duration = max(float(row['duration_s']), 1.0)
    scale, uname = _time_unit(duration)
    total = duration / scale
    key = (res.path, animal, s_idx)
    lo, hi = window_state.value[1] if window_state.value[0] == key else (0.0, total)
    span = hi - lo

    def set_window(v):
        window_state.set((key, (float(v[0]), float(v[1]))))

    with solara.Row(style={"flex-wrap": "wrap", "gap": "16px"}):
        solara.Select("Animal / control", value=animal, values=animals,
                      on_value=animal_choice.set, style={"min-width": "220px"})
        solara.Select("Session", value=session, values=list(labels),
                      on_value=session_choice.set, style={"min-width": "320px"})

    def pan(direction):
        shift = direction * span * 0.8
        a = min(max(lo + shift, 0.0), total - span)
        set_window((a, a + span))

    with solara.Row(style={"align-items": "center"}):
        solara.Button("◀", on_click=lambda: pan(-1), disabled=lo <= 0)
        solara.Button("▶", on_click=lambda: pan(1), disabled=hi >= total)
        for label, width_s in (("1 min", 60), ("10 min", 600), ("1 h", 3600)):
            if width_s < duration:
                solara.Button(label, text=True, on_click=lambda w=width_s / scale: set_window(
                    (min(lo, total - w), min(lo, total - w) + w)))
        solara.Button("Full", text=True, on_click=lambda: set_window((0.0, total)))
    solara.SliderRangeFloat(f"Window ({uname})", value=(lo, hi), on_value=set_window,
                            min=0.0, max=float(total), step=1 / scale)  # 1 s steps

    SessionTraceView(res, animal, s_idx, row['date'], lo * scale, hi * scale)
    solara.Text("Long windows are min/max decimated (every dip is kept). Zoom in to see "
                "individual samples.", style={"font-size": "12px", "color": GREY})


@solara.component
def SessionTraceView(res, animal, s_idx, date, t0, t1):
    w = solara.use_memo(lambda: core.session_window(res, animal, s_idx, t0, t1,
                                                    max_points=PLOT_POINTS),
                        [res, animal, s_idx, t0, t1])
    title = (f"{animal} — {date} — consumed {w['consumed_vol']:.2f} mL — "
             f"{w['n_licks_window']} licks in view")
    Plot(_trace_plot(w, title, unit=_time_unit(t1 - t0)), [res, animal, s_idx, t0, t1])


@solara.component
def CorrelationTab(res):
    from scipy import stats
    corr = solara.use_memo(lambda: core.correlation_analysis(res), [res])
    clean = solara.use_memo(lambda: core.refit_excluding(res, set(excluded.value)),
                            [res, excluded.value])
    ceiling = solara.use_memo(lambda: core.correlation_ceiling(res), [res])
    if not corr['ok']:
        solara.Warning("Need at least 3 animal sessions with volumes for the correlation.")
        return
    mdl, robust = corr['ols'], corr['robust']

    with solara.Card("Lick count vs volume consumed (all animal sessions)"):
        solara.Markdown(f"OLS **R² = {mdl.rsquared:.4f}** · HC3-robust slope "
                        f"**{robust.params[1]:.4g}** mL/lick, p = {robust.pvalues[1]:.3e} · "
                        f"n = {len(corr['x'])}")
        fig = Figure(figsize=(13, 3.6), dpi=100)
        a0, a1, a2 = fig.subplots(1, 3)
        out = corr['is_outlier']
        a0.scatter(corr['x'][~out], corr['y'][~out], s=16, color=BLUE)
        a0.scatter(corr['x'][out], corr['y'][out], s=40, color=RED, marker="x", label="MAD outlier")
        a0.set_xlabel("Number of Licks"); a0.set_ylabel("Volume Consumed (mL)")
        a0.set_title("Outliers detected"); a0.legend(fontsize=8)
        stats.probplot(mdl.resid, dist="norm", plot=a1)
        a1.set_title(f"Linear fit residuals, R²={mdl.rsquared:.4f}", fontsize=10)
        stats.probplot(corr['resid_r'], dist="norm", plot=a2)
        a2.set_title("Robust (Huber) fit residuals", fontsize=10)
        Plot(fig, [res])
        with solara.Details("Full OLS summary (HC3)"):
            solara.Preformatted(str(robust.summary()))

    outliers = corr['outliers']
    with solara.Card(f"Outliers ({len(outliers)}) — MAD > 3.5 on robust residuals"):
        if outliers.empty:
            solara.Success("No outliers detected.")
        else:
            solara.Markdown("Tick a session to **exclude** it from the cleaned regression and "
                            "from the *Outliers removed* tab. Nothing is excluded by default.")
            for r in outliers.itertuples():
                k = (int(r.session), r.animal_id)

                def toggle(on, k=k):
                    cur = set(excluded.value)
                    cur.add(k) if on else cur.discard(k)
                    excluded.set(frozenset(cur))
                solara.Checkbox(label=f"Exclude {r.label}: {r.n_licks} licks, "
                                      f"{r.consumed_vol:.2f} mL",
                                value=k in excluded.value, on_value=toggle)
            with solara.Details("Outlier traces"):
                for r in outliers.itertuples():
                    w = core.session_window(res, r.animal_id, int(r.session), max_points=2000)
                    Plot(_trace_plot(w, f"{r.animal_id}, date: {r.date}"),
                         [res, r.animal_id, int(r.session)])

    with solara.Card("Cleaned fit"):
        if clean['ok']:
            mc, rc = clean['ols'], clean['robust']
            solara.Markdown(
                f"Outliers excluded: **{clean['n_excluded']}**\n\n"
                f"- Original: R² {mdl.rsquared:.4f} | slope {mdl.params[1]:.4g} | "
                f"HC3 p {robust.pvalues[1]:.3e}\n"
                f"- Cleaned: R² {mc.rsquared:.4f} | slope {mc.params[1]:.4g} | "
                f"HC3 p {rc.pvalues[1]:.3e}")
            with solara.Details("Cleaned OLS summary (HC3)"):
                solara.Preformatted(str(rc.summary()))
            solara.Button("Export nLicksVsVol_outliers_removed.csv", text=True,
                          on_click=lambda: export(lambda d: core.export_clean_regression(clean, d),
                                                  "cleaned regression data"))
        else:
            solara.Warning("Fewer than 3 sessions left after exclusion.")
        if ceiling is not None:
            solara.Markdown(f"**Correlation ceiling:** predicted volumes rounded to the nearest "
                            f"0.05 mL give R² = {ceiling.rsquared:.4f} (the best achievable with "
                            f"0.05 mL volume resolution).")


@solara.component
def PerAnimalTab(res):
    table, fits, skipped = solara.use_memo(lambda: core.per_animal_fits(res), [res])
    solara.Markdown(
        "`volume ~ licks` fitted **separately for each animal** across its own sessions (raw "
        "data, independent of outlier exclusion). Each slope is that animal's mL per lick; the "
        "spread across animals is the between-individual variability pooling hides. Animals "
        "with fewer than 3 sessions are skipped.")
    if skipped:
        solara.Text("Skipped: " + ", ".join(f"{a} (n={n})" for a, n in skipped))
    if table.empty:
        solara.Info("No animal has 3 or more sessions.")
        return
    sl = table['slope_mL_per_lick']
    spread = f", {sl.max() / sl.min():.1f}× spread" if sl.min() > 0 else ""
    solara.Markdown(f"Slope (mL/lick): mean **{sl.mean():.2e}**, range "
                    f"{sl.min():.2e}–{sl.max():.2e}{spread}")
    solara.DataFrame(table, items_per_page=20)
    fig, ax = _fig(9, 5.5)
    cmap = __import__("matplotlib").colormaps["tab20"]
    for i, animal_id in enumerate(table['animal_id']):
        x, y, b0, b1 = fits[animal_id]
        c = cmap(i % 20)
        ax.scatter(x, y, s=16, color=c)
        xl = np.array([x.min(), x.max()])
        ax.plot(xl, b0 + b1 * xl, color=c, lw=1.5, label=animal_id)
    ax.set_xlabel("Lick Count"); ax.set_ylabel("Volume Consumed (mL)")
    ax.set_title("Per-animal: volume consumed vs lick count")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    Plot(fig, [res])
    solara.Button("Export per_animal_lick_vs_volume_fits.csv", text=True,
                  on_click=lambda: export(lambda d: core.export_per_animal(table, d),
                                          "per-animal fits"))


@solara.component
def CleanTab(res):
    excl = excluded.value
    if excl:
        solara.Info("Excluded sessions: " + ", ".join(
            f"{a} ({res.session_label(s)})" for s, a in sorted(excl)))
    else:
        solara.Info("No sessions excluded (tick outliers on the Correlation tab). "
                    "Exports from this tab use the *_outliers_removed.csv names.")
    MetricsBlock(res, excl, "_outliers_removed", " (outliers removed)")


@solara.component
def Page():
    solara.Title("CLiQR Data Analysis")
    res = results.value
    with solara.Column(style={"padding": "16px", "max-width": "1600px", "margin": "0 auto"}):
        with solara.Card():
            solara.Markdown("# CLiQR Data Analysis")
            solara.Markdown("*Capacitive Lick Quantification in Rodents: batch analysis*"
                            + (f" · results: `{os.path.basename(res.path)}`" if res else ""))
        # A plain toggle-button tab bar: only the visible tab is rendered, so
        # nothing reads data for tabs you are not looking at.
        names = ["Run", "Start/Stop check"] + ([] if res is None else list(RESULT_TABS))
        current = tab.value if tab.value in names else "Run"
        solara.ToggleButtonsSingle(value=current, values=names, on_value=tab.set,
                                   style={"flex-wrap": "wrap", "margin-bottom": "12px"})
        if res is None:
            solara.Text("Run an analysis or open existing results to unlock the result tabs.",
                        style={"color": GREY})
        if current == "Run":
            RunTab()
        elif current == "Start/Stop check":
            StartStopTab()
        else:
            RESULT_TABS[current](res)


def _behavior_tab(res):
    return MetricsBlock(res, frozenset(), "", "")


RESULT_TABS = {
    "Summary": SummaryTab,
    "Behavior": _behavior_tab,
    "Traces": TracesTab,
    "Correlation": CorrelationTab,
    "Per-animal": PerAnimalTab,
    "Outliers removed": CleanTab,
}
