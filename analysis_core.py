"""GUI-independent core of the CLiQR batch analysis (used by analysis_gui.py).

This is the logic that used to live in DataAnalysis.ipynb, reorganised so that
long (e.g. 24 h, ~5 GB/day) recordings stay responsive:

* Raw files are processed one sensor at a time (see data_analysis.filter_data),
  never loaded whole.
* Summary statistics only read the small per-session datasets (lick times,
  volumes, bout metrics). Full capacitance traces are never pulled into memory
  just to count licks: `lick_indices.shape` is HDF5 metadata, free to read.
* Traces are only read for the window being looked at, and are min/max
  decimated to a few thousand points before plotting, so a 24 h trace
  (~10 M samples) draws as fast as a 2 h one. Lick dips are minima, so the
  min/max envelope never hides one.

2-bottle choice recordings (raw file attr two_bottle=True) are analysed per
bottle: every session row carries a `fluid` (1 or 2; 0 for ordinary 1-bottle
data) and every analysis takes a `fluid=` argument (None = all fluids). The
preference_* functions compare the two bottles of each session.

Nothing in here imports Solara; every function can be used from a script or a
notebook, and is unit-tested in tests/test_analysis_core.py.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable

import h5py
import numpy as np
import pandas as pd

from data_analysis import (FLUIDS, SENSOR_BOARD_MAP, compute_lick_durations, filter_data,
                           fluid_group_name, is_control)

LAYOUT_NAMES = ("layout_w_controls.csv", "layout.csv")
RAW_PATTERN = "raw_data_*.h5"
_DATE_RE = re.compile(r'raw_data_(\d{4}-\d{2}-\d{2})_')

ALGORITHMS = {
    "basic_threshold": "Basic threshold (manuscript)",
    "hilbert": "Hilbert envelope (experimental)",
    "ml": "ML cascade (needs torch)",
}

# Big per-session datasets copied file-to-file inside HDF5 (never through numpy).
_SESSION_DATASETS = ('lick_indices', 'lick_times', 'cap_data', 'time_data',
                     'used_start_idx', 'used_stop_idx')
_OPTIONAL_DATASETS = ('ILIs', 'bout_lick_counts', 'bout_durations', 'bout_start_times',
                      'lick_durations', 'lick_depths', 'lick_onsets', 'lick_offsets',
                      'bout_median_lick_durations')

DEFAULT_FLUID_NAMES = {1: "Fluid 1", 2: "Fluid 2"}

LogFn = Callable[[str], None]


def _noop(*_args, **_kwargs):
    pass


# ---------------------------------------------------------------------------
# Discovery / naming
# ---------------------------------------------------------------------------
def find_layout(cohort_dir: str) -> str | None:
    """Prefer layout_w_controls.csv (lists control cages), else layout.csv."""
    for fname in LAYOUT_NAMES:
        path = os.path.join(cohort_dir, fname)
        if os.path.exists(path):
            return path
    return None


def load_layout(cohort_dir: str) -> pd.DataFrame:
    path = find_layout(cohort_dir)
    if path is None:
        raise FileNotFoundError(f"No layout_w_controls.csv or layout.csv found in {cohort_dir}")
    return pd.read_csv(path, header=None, index_col=0)


def layout_ids(layout: pd.DataFrame) -> list[str]:
    return [str(row[0]) for row in layout.values]


def discover_cohorts(base_dir: str) -> list[str]:
    """Cohort folder names under base_dir: any sub-folder holding a layout file."""
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
        and find_layout(os.path.join(base_dir, name)) is not None
    )


def list_raw_files(cohort_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(cohort_dir, RAW_PATTERN)))


def raw_date(path: str) -> str | None:
    m = _DATE_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def filtered_path_for(raw_path: str) -> str:
    base = os.path.basename(raw_path)
    return os.path.join(os.path.dirname(raw_path), "filtered" + base[3:])


def time_fix_path_for(raw_path: str) -> str:
    return raw_path[:-3] + "_time_fix.xlsx"


def compact_date_range(dates: Iterable[str]) -> str:
    """'2026-01-20', '2026-01-21', '2026-02-03' -> '2026-01-20_21_02-03'."""
    parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    if not parsed:
        return "nodate"
    out = [parsed[0].strftime("%Y-%m-%d")]
    year, month = parsed[0].year, parsed[0].month
    for d in parsed[1:]:
        if d.year == year and d.month == month:
            out.append(d.strftime("%d"))
        elif d.year == year:
            month = d.month
            out.append(d.strftime("%m-%d"))
        else:
            year, month = d.year, d.month
            out.append(d.strftime("%Y-%m-%d"))
    return "_".join(out)


def combined_filename(prefixes: Iterable[str], dates: Iterable[str]) -> str:
    return f"results_combined_{'-'.join(prefixes)}_{compact_date_range(dates)}.h5"


def file_is_two_bottle(path: str) -> bool | None:
    """Recording mode of a raw file (attr two_bottle), or None if unreadable."""
    try:
        with h5py.File(path, 'r') as h5f:
            return bool(h5f.attrs.get('two_bottle', False))
    except OSError:
        return None


def file_size_gb(path: str) -> float:
    try:
        return os.path.getsize(path) / 1e9
    except OSError:
        return float('nan')


# ---------------------------------------------------------------------------
# Running detection
# ---------------------------------------------------------------------------
class _LineForwarder(io.TextIOBase):
    """stdout replacement that forwards complete lines to a log callback
    (and still echoes them to the real stdout / server console)."""

    def __init__(self, log: LogFn, echo):
        self._log, self._echo, self._buf = log, echo, ""

    def write(self, s):
        if self._echo is not None:
            self._echo.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._log(line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._log(self._buf)
        self._buf = ""


@contextlib.contextmanager
def capture_prints(log: LogFn):
    fwd = _LineForwarder(log, sys.__stdout__)
    with contextlib.redirect_stdout(fwd):
        try:
            yield
        finally:
            fwd.flush()


def process_raw_file(raw_path: str, layout: pd.DataFrame, *, algorithm="basic_threshold",
                     recording_length=2 * 60 * 60, apply_time_fix=True,
                     require_weight=True, log: LogFn = print, progress=None) -> bool:
    """Run lick detection on one raw file -> filtered_*.h5. Returns True if data was missing.

    2-bottle files are detected from their two_bottle attr (see filter_data)."""
    filtered = filtered_path_for(raw_path)
    time_fix = None
    tf_path = time_fix_path_for(raw_path)
    if os.path.exists(tf_path):
        if apply_time_fix:
            time_fix = pd.read_excel(tf_path, index_col=False, header=0)
            log(f"Using time fix file {os.path.basename(tf_path)}")
        else:
            log(f"Ignoring time fix file {os.path.basename(tf_path)} (disabled)")
    with h5py.File(raw_path, 'r') as raw_h5, h5py.File(filtered, 'w') as filtered_h5:
        if 'comments' in raw_h5:
            comment = raw_h5['comments'][()]
            if isinstance(comment, bytes):
                comment = comment.decode('utf-8', errors='replace')
            if str(comment).strip():
                log(f"Comments: {comment}")
        with capture_prints(log):
            return filter_data(raw_h5, filtered_h5, layout, raw_path[:-3] + '.log',
                               time_fix=time_fix, algorithm=algorithm,
                               recording_length=recording_length, progress=progress,
                               require_weight=require_weight)


def _copy_channel(in_h5, src, dst):
    """Copy one channel's volume + per-session datasets (filtered -> combined)."""
    dst.create_dataset('consumed_vols', data=src['consumed_vol'][()])
    for key in _SESSION_DATASETS + _OPTIONAL_DATASETS:
        if key in src:
            in_h5.copy(src[key], dst, name=key)


def build_combined(file_infos: list[dict], out_path: str, *, recording_length: float,
                   algorithm: str = "", log: LogFn = print, two_bottle: bool = False,
                   fluid_names: dict | None = None) -> None:
    """Merge filtered_*.h5 files into one results_combined_*.h5, grouped by animal ID.

    file_infos: dicts with keys raw, cohort, cohort_dir, layout (DataFrame),
    already sorted in session order (index i -> subgroup "i").
    Large datasets are copied HDF5-to-HDF5 (h5py Group.copy) so they never pass
    through Python memory.

    2-bottle layout: <animal>/<i>/weights is shared and each bottle gets its own
    <animal>/<i>/fluid1, fluid2 subgroup (consumed_vols + the per-session
    datasets). A session is kept only if BOTH bottles have a volume, so the
    two fluids of every stored session can always be compared.
    """
    all_ids = sorted({a for fi in file_infos for a in layout_ids(fi['layout'])})
    with h5py.File(out_path, 'w') as out_h5:
        for animal_id in all_ids:
            out_h5.create_group(animal_id)
        for i, fi in enumerate(file_infos):
            raw_path = fi['raw']
            layout_path = find_layout(fi['cohort_dir'])
            with h5py.File(filtered_path_for(raw_path), 'r') as in_h5:
                for animal_id in layout_ids(fi['layout']):
                    if animal_id not in in_h5:
                        continue  # animal missing from this file
                    src = in_h5[animal_id]
                    channels = ([src.get(fluid_group_name(f)) for f in FLUIDS]
                                if two_bottle else [src])
                    if any(c is None or 'consumed_vol' not in c or (
                            not is_control(animal_id) and 'weight' not in c)
                           for c in channels):
                        # weight / volume wasn't properly recorded: skip this entry
                        log(f"Skipping {animal_id} in {os.path.basename(raw_path)}: "
                            f"missing weight or volume"
                            + (" (either bottle)" if two_bottle else ""))
                        continue
                    grp = out_h5[animal_id].create_group(str(i))
                    # Control cages have no animal, so no weight was recorded.
                    # (2-bottle: both bottles carry the same sensor-level weight.)
                    if not is_control(animal_id):
                        grp.create_dataset('weights', data=channels[0]['weight'][()])
                    if two_bottle:
                        for f, c in zip(FLUIDS, channels):
                            _copy_channel(in_h5, c, grp.create_group(fluid_group_name(f)))
                    else:
                        _copy_channel(in_h5, src, grp)
                    # Provenance for downstream tools (scripts/find_interesting_windows.py)
                    grp.attrs['raw_h5'] = raw_path
                    if layout_path is not None:
                        grp.attrs['layout'] = layout_path
                    grp.attrs['date'] = raw_date(raw_path) or ""
        out_h5.attrs['files'] = json.dumps([fi['raw'] for fi in file_infos])
        out_h5.attrs['dates'] = json.dumps([raw_date(fi['raw']) or str(i)
                                            for i, fi in enumerate(file_infos)])
        out_h5.attrs['cohorts'] = json.dumps(sorted({fi['cohort'] for fi in file_infos}))
        out_h5.attrs['recording_length'] = float(recording_length)
        out_h5.attrs['algorithm'] = algorithm
        out_h5.attrs['two_bottle'] = bool(two_bottle)
        if two_bottle:
            names = {**DEFAULT_FLUID_NAMES, **(fluid_names or {})}
            out_h5.attrs['fluid_names'] = json.dumps({str(f): names[f] for f in FLUIDS})
        out_h5.attrs['created'] = datetime.now().isoformat(timespec='seconds')


def run_pipeline(selected_files: list[str], base_dir: str, *, algorithm="basic_threshold",
                 recording_length=2 * 60 * 60, apply_time_fix=True, keep_intermediate=False,
                 two_bottle=False, fluid_names: dict | None = None, require_weight=True,
                 log: LogFn = print, progress: Callable[[float, str], None] = _noop,
                 should_stop: Callable[[], bool] = lambda: False) -> str | None:
    """Detect licks in every selected raw file, then build the combined results file.

    Cohort = name of each raw file's parent folder (expects base_dir/<cohort>/raw_*.h5).
    Returns the combined file path (or None if nothing usable was produced).
    progress(fraction 0..1, message) is called as work proceeds.

    `two_bottle` is the analysis mode: files recorded in the other mode are
    skipped (with a log line), since 1- and 2-bottle sessions cannot share one
    combined file. `fluid_names` ({1: "Water", 2: "Sucrose"}) label the bottles.
    """
    layouts, infos = {}, []
    for f in selected_files:
        cohort_dir = os.path.dirname(f)
        cohort = os.path.basename(cohort_dir)
        if cohort not in layouts:
            layouts[cohort] = load_layout(cohort_dir)
        infos.append({'raw': f, 'cohort': cohort, 'cohort_dir': cohort_dir,
                      'layout': layouts[cohort]})
    # Session order = date order (stable for same-day files)
    infos.sort(key=lambda fi: (raw_date(fi['raw']) or "", fi['raw']))

    n = len(infos)
    usable = []
    for k, fi in enumerate(infos):
        if should_stop():
            log("Stopped by user.")
            return None
        name = os.path.basename(fi['raw'])
        log(f"[{k + 1}/{n}] {fi['cohort']}/{name} ({file_size_gb(fi['raw']):.2f} GB)")
        progress(k / n, f"Detecting licks: {name}")
        file_mode = file_is_two_bottle(fi['raw'])
        if file_mode is not None and file_mode != bool(two_bottle):
            log(f"Skipping {name}: recorded in {'2' if file_mode else '1'}-bottle mode but "
                f"the analysis is set to {'2' if two_bottle else '1'}-bottle")
            continue

        def _p(done, total, animal, k=k, name=name):
            progress((k + done / max(total, 1)) / n, f"{name}: {animal} ({done}/{total})")

        try:
            missing = process_raw_file(fi['raw'], fi['layout'], algorithm=algorithm,
                                       recording_length=recording_length,
                                       apply_time_fix=apply_time_fix,
                                       require_weight=require_weight, log=log, progress=_p)
        except Exception as exc:  # keep going with the other files
            log(f"ERROR processing {name}: {exc!r}")
            continue
        if missing:
            log(f"Skipping {name} due to missing data")
        else:
            usable.append(fi)

    if not usable:
        log("No files produced usable results.")
        return None

    progress(0.99, "Combining results")
    dates = [raw_date(fi['raw']) for fi in usable]
    out_path = os.path.join(base_dir, combined_filename(
        sorted({fi['cohort'] for fi in usable}), [d for d in dates if d]))
    build_combined(usable, out_path, recording_length=recording_length,
                   algorithm=algorithm, log=log, two_bottle=two_bottle,
                   fluid_names=fluid_names)
    log(f"Combined results written to {out_path}")

    if not keep_intermediate:
        for fi in infos:
            fp = filtered_path_for(fi['raw'])
            if os.path.exists(fp):
                os.remove(fp)
        log("Removed intermediate filtered_*.h5 files")
    progress(1.0, "Done")
    return out_path


# ---------------------------------------------------------------------------
# Loading results (cheap: never reads full traces)
# ---------------------------------------------------------------------------
@dataclass
class CombinedResults:
    path: str
    sessions: pd.DataFrame          # one row per (animal_id, session, fluid)
    dates: list[str]                # session index -> date label
    files: list[str]                # session index -> raw file ('' if unknown)
    recording_length: float         # seconds
    attrs: dict = field(default_factory=dict)
    two_bottle: bool = False
    fluid_names: dict = field(default_factory=dict)  # {1: "Fluid 1", 2: "Fluid 2"}

    @property
    def animal_ids(self) -> list[str]:
        return sorted(a for a in self.sessions['animal_id'].unique() if not is_control(a))

    @property
    def control_ids(self) -> list[str]:
        return sorted(a for a in self.sessions['animal_id'].unique() if is_control(a))

    @property
    def animals(self) -> pd.DataFrame:
        return self.sessions[~self.sessions['is_control']]

    @property
    def fluids(self) -> list[int]:
        """Fluid codes present: [1, 2] for 2-bottle results, [0] otherwise."""
        return list(FLUIDS) if self.two_bottle else [0]

    def fluid_label(self, fluid: int) -> str:
        return self.fluid_names.get(fluid, f"Fluid {fluid}") if fluid else ""

    def session_label(self, session: int) -> str:
        return self.dates[session] if 0 <= session < len(self.dates) else str(session)


def _scalar(ds):
    v = ds[()]
    return float(np.asarray(v).ravel()[0]) if np.size(v) else float('nan')


def _median_ms(ds) -> float:
    """Median of the finite values of a durations dataset, in ms (NaN if none)."""
    if ds is None:
        return float('nan')
    a = ds[()]
    a = a[np.isfinite(a)]
    return float(np.median(a) * 1000) if a.size else float('nan')


_LICK_MEASURES = ('lick_durations', 'lick_depths', 'lick_onsets', 'lick_offsets')


def _lick_measures(g, keys=_LICK_MEASURES) -> dict:
    """Per-lick duration/depth/onset/offset arrays for one channel group. Read from
    the file when stored; otherwise (combined files made before these were added)
    measured from the stored trace and lick indices, so older results still get them."""
    if all(k in g for k in keys):
        return {k: g[k][()] for k in keys}
    if not all(k in g for k in ('cap_data', 'time_data', 'lick_indices')):
        return {k: np.array([]) for k in keys}
    m = compute_lick_durations(g['cap_data'][()], g['time_data'][()], g['lick_indices'][()])
    return {k: m[k] for k in keys}


def _lick_durations(g) -> np.ndarray:
    """Per-lick durations (s) for one channel group (see _lick_measures)."""
    return _lick_measures(g, ('lick_durations',))['lick_durations']


def _channel(g, fluid: int):
    """The group holding one channel's data: the session group itself
    (1-bottle, fluid 0) or its fluid1/fluid2 subgroup."""
    return g[fluid_group_name(fluid)] if fluid else g


def load_combined(path: str) -> CombinedResults:
    rows = []
    with h5py.File(path, 'r') as h5f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h5f.attrs.items()}
        two_bottle = bool(attrs.get('two_bottle', False))
        fluids = list(FLUIDS) if two_bottle else [0]
        for animal_id, agrp in h5f.items():
            if not isinstance(agrp, h5py.Group):
                continue
            for key, g in agrp.items():
                if not key.isdigit():
                    continue
                weight = _scalar(g['weights']) if 'weights' in g else np.nan
                for fluid in fluids:
                    if fluid and fluid_group_name(fluid) not in g:
                        continue
                    c = _channel(g, fluid)
                    t = c.get('time_data')
                    n_samples = int(t.shape[0]) if t is not None else 0
                    duration = float(t[-1]) if n_samples else 0.0
                    rows.append({
                        'animal_id': animal_id,
                        'session': int(key),
                        'fluid': fluid,
                        'n_licks': int(c['lick_indices'].shape[0]) if 'lick_indices' in c else 0,
                        'consumed_vol': _scalar(c['consumed_vols']) if 'consumed_vols' in c else np.nan,
                        'weight': weight,
                        'n_bouts': int(c['bout_lick_counts'].shape[0]) if 'bout_lick_counts' in c else 0,
                        # NaN for combined files made before lick durations existed
                        # (pooled_metrics computes those on the fly instead).
                        'median_lick_duration_ms': _median_ms(c.get('lick_durations')),
                        'duration_s': duration,
                        'n_samples': n_samples,
                        'raw_h5': str(g.attrs.get('raw_h5', '')),
                        'date_attr': str(g.attrs.get('date', '')),
                        'is_control': is_control(animal_id),
                    })
    sessions = pd.DataFrame(rows, columns=[
        'animal_id', 'session', 'fluid', 'n_licks', 'consumed_vol', 'weight', 'n_bouts',
        'median_lick_duration_ms',
        'duration_s', 'n_samples', 'raw_h5', 'date_attr', 'is_control'])
    n_sessions = int(sessions['session'].max()) + 1 if len(sessions) else 0

    dates = json.loads(attrs['dates']) if 'dates' in attrs else []
    files = json.loads(attrs['files']) if 'files' in attrs else []
    if len(dates) < n_sessions:
        # Older combined files: recover per-session dates from provenance attrs.
        by_session = {}
        for r in sessions.itertuples():
            by_session.setdefault(r.session, r.date_attr or raw_date(r.raw_h5) or "")
        dates = [by_session.get(i) or f"session {i}" for i in range(n_sessions)]
        files = [""] * n_sessions
        for r in sessions.itertuples():
            if r.raw_h5:
                files[r.session] = r.raw_h5
    sessions['date'] = [dates[s] if s < len(dates) else str(s) for s in sessions['session']]
    sessions = sessions.drop(columns=['date_attr']).sort_values(
        ['animal_id', 'session', 'fluid']).reset_index(drop=True)

    if 'recording_length' in attrs:
        rec_len = float(attrs['recording_length'])
    else:
        rec_len = float(sessions['duration_s'].max()) if len(sessions) else 2 * 60 * 60
    fluid_names = {}
    if two_bottle:
        stored = json.loads(attrs['fluid_names']) if 'fluid_names' in attrs else {}
        fluid_names = {f: stored.get(str(f), DEFAULT_FLUID_NAMES[f]) for f in FLUIDS}
    return CombinedResults(path=path, sessions=sessions, dates=dates, files=files,
                           recording_length=rec_len, attrs=attrs, two_bottle=two_bottle,
                           fluid_names=fluid_names)


def list_combined_files(base_dir: str) -> list[str]:
    files = glob.glob(os.path.join(base_dir, "results_combined_*.h5"))
    return sorted(files, key=os.path.getmtime, reverse=True)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
# Every analysis takes `fluid`: 1 or 2 restricts a 2-bottle result to one
# bottle, 0 is the only channel of a 1-bottle result, None means all channels.
# Excluded outliers are (session, animal_id, fluid) triples.


def _fluid_rows(df: pd.DataFrame, fluid: int | None) -> pd.DataFrame:
    return df if fluid is None else df[df['fluid'] == fluid]


def _iter_session_groups(res: CombinedResults, animals: Iterable[str],
                         exclude: set | frozenset = frozenset(), fluid: int | None = None):
    """Yield (animal_id, session, fluid, channel group) for every kept channel."""
    fluids = res.fluids if fluid is None else [fluid]
    with h5py.File(res.path, 'r') as h5f:
        for animal_id in animals:
            if animal_id not in h5f:
                continue
            for key, g in h5f[animal_id].items():
                if not key.isdigit():
                    continue
                for f in fluids:
                    if (int(key), animal_id, f) in exclude:
                        continue
                    if f and fluid_group_name(f) not in g:
                        continue
                    yield animal_id, int(key), f, _channel(g, f)


def detection_summary(res: CombinedResults, fluid: int | None = None) -> dict:
    import statsmodels.api as sm
    animals = _fluid_rows(res.animals, fluid)
    df = animals.dropna(subset=['consumed_vol'])
    x = df['n_licks'].to_numpy(float)
    y = df['consumed_vol'].to_numpy(float)
    out = {
        'mean': float(animals['n_licks'].mean()) if len(animals) else np.nan,
        'sd': float(animals['n_licks'].std(ddof=0)) if len(animals) else np.nan,
        'x': x, 'y': y, 'r2': np.nan, 'slope': np.nan, 'intercept': np.nan,
    }
    if x.size >= 3:
        mdl = sm.OLS(y, sm.add_constant(x)).fit()
        out.update(r2=float(mdl.rsquared), intercept=float(mdl.params[0]),
                   slope=float(mdl.params[1]))
    return out


def pooled_metrics(res: CombinedResults, exclude: set | frozenset = frozenset(),
                   fluid: int | None = None) -> dict:
    """ILIs, licks per bout, bout durations and lick durations pooled across animals
    (controls excluded). Lick durations keep only measurable (finite) licks."""
    acc = {'ILIs': [], 'bout_lick_counts': [], 'bout_durations': [], 'lick_durations': []}
    for _a, _s, _f, g in _iter_session_groups(res, res.animal_ids, exclude, fluid):
        for key in ('ILIs', 'bout_lick_counts', 'bout_durations'):
            if key in g:
                acc[key].append(g[key][()])
        d = _lick_durations(g)
        acc['lick_durations'].append(d[np.isfinite(d)])
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in acc.items()}


def metric_summary(m: dict) -> dict:
    def _f(fn, a):
        return float(fn(a)) if len(a) else float('nan')
    med_ili = _f(np.median, m['ILIs'])
    return {
        'median_ili_ms': med_ili * 1000,
        'median_ili_hz': 1 / med_ili if med_ili and not np.isnan(med_ili) else float('nan'),
        'mean_licks_per_bout': _f(np.mean, m['bout_lick_counts']),
        'median_licks_per_bout': _f(np.median, m['bout_lick_counts']),
        'mean_bout_duration': _f(np.mean, m['bout_durations']),
        'median_bout_duration': _f(np.median, m['bout_durations']),
        'n_ilis': len(m['ILIs']), 'n_bouts': len(m['bout_durations']),
        'median_lick_duration_ms': _f(np.median, m.get('lick_durations', [])) * 1000,
        'n_lick_durations': len(m.get('lick_durations', [])),
    }


def lick_events(res: CombinedResults, exclude: set | frozenset = frozenset(),
                fluid: int | None = None) -> pd.DataFrame:
    """One row per detected lick (controls excluded): its start and end time.

    start_s / end_s are the half-depth crossings of the lick's dip (see
    data_analysis.compute_lick_durations), in seconds from the session's used
    start, the same clock as lick_times. They replace a fixed offset: the detected
    lick time (detected_s) is the dip's deepest sample, which comes after the dip
    has begun, and start_lead_ms is how much earlier the measured start is.
    duration_valid is False where the duration was rejected (sampling gap inside
    the dip, or longer than 1/6 s); start/end are still the best estimate there,
    and are NaN only for a dip with no measurable depth.
    """
    cols = ['animal_id', 'session', 'date', 'fluid', 'lick_number', 'detected_s', 'start_s',
            'end_s', 'duration_ms', 'start_lead_ms', 'depth_counts', 'duration_valid']
    frames = []
    for animal_id, session, f, g in _iter_session_groups(res, res.animal_ids, exclude, fluid):
        if 'lick_indices' not in g or g['lick_indices'].shape[0] == 0:
            continue
        times = g['lick_times'][()]
        lm = _lick_measures(g)
        start, end = lm['lick_onsets'], lm['lick_offsets']
        durs = lm['lick_durations']
        frames.append(pd.DataFrame({
            'animal_id': animal_id, 'session': session,
            'date': res.dates[session] if session < len(res.dates) else str(session),
            'fluid': f, 'lick_number': np.arange(1, len(times) + 1),
            'detected_s': times, 'start_s': start, 'end_s': end,
            'duration_ms': (end - start) * 1000, 'start_lead_ms': (times - start) * 1000,
            'depth_counts': lm['lick_depths'], 'duration_valid': np.isfinite(durs),
        }))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    df = df[cols].sort_values(['animal_id', 'session', 'fluid', 'detected_s'],
                              kind='stable').reset_index(drop=True)
    if not res.two_bottle:
        df = df.drop(columns=['fluid'])
    return df


def bin_edges_for(length_s: float, bin_s: float) -> np.ndarray:
    return np.arange(0, length_s + bin_s, bin_s)


def timecourse(res: CombinedResults, *, bin_s: float, length_s: float,
               exclude: set | frozenset = frozenset(), fluid: int | None = None) -> dict:
    """Per-session lick / bout counts in fixed bins (rows = sessions)."""
    edges = bin_edges_for(length_s, bin_s)
    licks, bouts = [], []
    for _a, _s, _f, g in _iter_session_groups(res, res.animal_ids, exclude, fluid):
        if 'lick_times' not in g:
            continue
        licks.append(np.histogram(g['lick_times'][()], bins=edges)[0])
        if 'bout_start_times' in g:
            bouts.append(np.histogram(g['bout_start_times'][()], bins=edges)[0])
    nb = len(edges) - 1
    return {
        'edges': edges,
        'centers_min': (edges[:-1] + edges[1:]) / 2 / 60,
        'licks': np.array(licks).reshape(-1, nb),
        'bouts': np.array(bouts).reshape(-1, nb),
    }


def mean_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and SEM, ignoring NaNs (e.g. undefined preference bins)."""
    mat = np.asarray(mat, dtype=float)
    if mat.shape[0] == 0:
        nan = np.full(mat.shape[1], np.nan)
        return nan, nan
    n = np.sum(~np.isnan(mat), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN columns -> NaN
        mean = np.nanmean(mat, axis=0)
        if mat.shape[0] < 2:
            return mean, np.zeros_like(mean)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)
    return mean, np.where(n >= 2, sem, 0.0)


def control_summary(res: CombinedResults) -> pd.DataFrame:
    c = res.sessions[res.sessions['is_control']]
    out = pd.DataFrame({
        'control': c['animal_id'], 'date': c['date'],
        'false_positive_licks': c['n_licks'].astype(int),
        'evaporation_mL': c['consumed_vol'].astype(float),
    })
    keys = ['control', 'date']
    if res.two_bottle:
        out.insert(2, 'fluid', [res.fluid_label(f) for f in c['fluid']])
        keys.append('fluid')
    return out.sort_values(keys).reset_index(drop=True)


def mad_based_outliers(r, thresh=3.5):
    """Flag points where |r - median(r)| > thresh * (1.4826 * MAD)."""
    r = np.asarray(r, dtype=float)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad == 0:
        return np.zeros_like(r, dtype=bool)
    return np.abs(r - med) > thresh * 1.4826 * mad


def correlation_data(res: CombinedResults, fluid: int | None = None) -> pd.DataFrame:
    """Animal sessions with both a lick count and a volume (the regression rows)."""
    df = _fluid_rows(res.animals, fluid).dropna(subset=['consumed_vol'])
    return df[['animal_id', 'session', 'fluid', 'date', 'n_licks', 'consumed_vol']].reset_index(drop=True)


def correlation_analysis(res: CombinedResults, mad_thresh: float = 3.5,
                         fluid: int | None = None) -> dict:
    """OLS (+HC3-robust inference) and Huber RLM of volume ~ licks, with MAD outliers
    flagged on the RLM residuals (same procedure as the notebook)."""
    import statsmodels.api as sm
    df = correlation_data(res, fluid)
    out = {'data': df, 'ok': len(df) >= 3}
    if not out['ok']:
        out['outliers'] = df.iloc[0:0].assign(label=[])
        return out
    x = df['n_licks'].to_numpy(float)
    y = df['consumed_vol'].to_numpy(float)
    X = sm.add_constant(x)
    mdl = sm.OLS(y, X).fit()
    robust = mdl.get_robustcov_results(cov_type='HC3')
    rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    resid_r = y - rlm.predict(X)
    is_out = mad_based_outliers(resid_r, thresh=mad_thresh)
    outliers = df[is_out].copy()
    outliers['label'] = [f"{a}  —  {d}" + (f"  —  {res.fluid_label(f)}" if f else "")
                         for a, d, f in zip(outliers['animal_id'], outliers['date'],
                                            outliers['fluid'])]
    out.update(ols=mdl, robust=robust, rlm=rlm, resid_r=resid_r, is_outlier=is_out,
               outliers=outliers.reset_index(drop=True), x=x, y=y)
    return out


def refit_excluding(res: CombinedResults, excluded: set, fluid: int | None = None) -> dict:
    """OLS + HC3 after dropping excluded (session, animal_id, fluid) triples."""
    import statsmodels.api as sm
    df = correlation_data(res, fluid)
    keep = np.array([(s, a, f) not in excluded
                     for s, a, f in zip(df['session'], df['animal_id'], df['fluid'])],
                    dtype=bool)
    x = df['n_licks'].to_numpy(float)[keep]
    y = df['consumed_vol'].to_numpy(float)[keep]
    out = {'x': x, 'y': y, 'n_excluded': int((~keep).sum()), 'ok': x.size >= 3}
    if out['ok']:
        mdl = sm.OLS(y, sm.add_constant(x)).fit()
        out.update(ols=mdl, robust=mdl.get_robustcov_results(cov_type='HC3'))
    return out


def per_animal_fits(res: CombinedResults, min_sessions: int = 3, fluid: int | None = None):
    """Per-animal OLS volume ~ licks. Returns (table, fits{animal: (x, y, b0, b1)}, skipped)."""
    import statsmodels.api as sm
    df = correlation_data(res, fluid)
    rows, fits, skipped = [], {}, []
    for animal_id in res.animal_ids:
        d = df[df['animal_id'] == animal_id]
        x = d['n_licks'].to_numpy(float)
        y = d['consumed_vol'].to_numpy(float)
        if x.size < min_sessions:
            skipped.append((animal_id, int(x.size)))
            continue
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        robust = fit.get_robustcov_results(cov_type='HC3')
        b0, b1 = fit.params
        rows.append({'animal_id': animal_id, 'n_sessions': int(x.size),
                     'slope_mL_per_lick': b1, 'intercept_mL': b0,
                     'r2': fit.rsquared, 'slope_p': robust.pvalues[1]})
        fits[animal_id] = (x, y, b0, b1)
    cols = ['animal_id', 'n_sessions', 'slope_mL_per_lick', 'intercept_mL', 'r2', 'slope_p']
    table = (pd.DataFrame(rows, columns=cols)
             .sort_values('slope_mL_per_lick').reset_index(drop=True))
    return table, fits, skipped


def correlation_ceiling(res: CombinedResults, step_ml: float = 0.05, fluid: int | None = None):
    """R² achievable if volumes were read perfectly but rounded to `step_ml`."""
    import statsmodels.api as sm
    df = correlation_data(res, fluid)
    if len(df) < 3:
        return None
    X = sm.add_constant(df['n_licks'].to_numpy(float))
    mdl = sm.OLS(df['consumed_vol'].to_numpy(float), X).fit()
    rounded = np.round(mdl.predict(X) / step_ml) * step_ml
    return sm.OLS(rounded, X).fit()


# ---------------------------------------------------------------------------
# 2-bottle fluid preference
# ---------------------------------------------------------------------------
# Measures are oriented toward a `target` fluid (T) against the other (O):
#   difference          T - O            (mL, licks or bouts)
#   preference ratio    T / (T + O)      0.5 = no preference, 1 = only T
#   discrimination idx  (T - O)/(T + O)  0 = no preference, +1 = only T
# Ratios use amounts clipped at 0: a bottle that reads slightly fuller after
# the session (meniscus/reading error) counts as 0 consumed, not negative
# consumption. A ratio is NaN when nothing was consumed from either bottle.
PREFERENCE_MEASURES = {'vol': ('consumed_vol', "Volume (mL)"),
                       'licks': ('n_licks', "Licks"),
                       'bouts': ('n_bouts', "Bouts")}


def _ratio(t, o):
    t = np.clip(np.asarray(t, float), 0, None)
    o = np.clip(np.asarray(o, float), 0, None)
    total = t + o
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, t / total, np.nan)


def _sessions_with_excluded_fluid(exclude) -> set:
    return {(s, a) for s, a, *_ in exclude}


def preference_table(res: CombinedResults, exclude: set | frozenset = frozenset(),
                     target: int = 1) -> pd.DataFrame:
    """One row per animal session of a 2-bottle result, both bottles side by side.

    Columns per measure m in (vol, licks, bouts): m_f1, m_f2 (raw per bottle),
    m_total, m_diff (T - O), m_pref (T/(T+O)), m_di ((T-O)/(T+O)).
    A session with either bottle excluded as an outlier is dropped entirely,
    because a preference needs both bottles.
    """
    if not res.two_bottle:
        raise ValueError("fluid preference needs 2-bottle results")
    if target not in FLUIDS:
        raise ValueError(f"target must be one of {FLUIDS}")
    other = 3 - target
    drop = _sessions_with_excluded_fluid(exclude)
    a = res.animals
    a = a[[(s, an) not in drop for s, an in zip(a['session'], a['animal_id'])]]
    cols = ['animal_id', 'session', 'date', 'weight']
    if a.empty:
        return pd.DataFrame(columns=cols + [f"{m}_{k}" for m in PREFERENCE_MEASURES
                                            for k in ('f1', 'f2', 'total', 'diff', 'pref', 'di')])
    values = [c for c, _ in PREFERENCE_MEASURES.values()] + ['weight']
    wide = a.set_index(['animal_id', 'session', 'date', 'fluid'])[values].unstack('fluid')
    out = wide.index.to_frame(index=False)
    out['weight'] = wide['weight'].bfill(axis=1).iloc[:, 0].to_numpy()
    for m, (col, _label) in PREFERENCE_MEASURES.items():
        f = {fl: wide[(col, fl)].to_numpy(float) if (col, fl) in wide else
             np.full(len(wide), np.nan) for fl in FLUIDS}
        t, o = f[target], f[other]
        out[f"{m}_f1"], out[f"{m}_f2"] = f[1], f[2]
        out[f"{m}_total"] = f[1] + f[2]
        out[f"{m}_diff"] = t - o
        out[f"{m}_pref"] = _ratio(t, o)
        out[f"{m}_di"] = 2 * out[f"{m}_pref"] - 1
    return out.sort_values(['animal_id', 'session']).reset_index(drop=True)


def preference_summary(table: pd.DataFrame, target: int = 1) -> dict:
    """Per-animal means (animals are the independent unit) and group tests.

    Tests, per measure, across animals:
      * paired t-test and Wilcoxon signed-rank: target bottle vs other bottle
      * one-sample t-test: preference ratio vs 0.5 (no preference)
    """
    from scipy import stats
    other = 3 - target
    num = [c for c in table.columns if c not in ('animal_id', 'session', 'date')]
    per_animal = (table.groupby('animal_id')[num].mean()
                  .assign(n_sessions=table.groupby('animal_id').size())
                  .reset_index())
    rows = []
    for m, (_col, label) in PREFERENCE_MEASURES.items():
        d = per_animal[[f"{m}_f{target}", f"{m}_f{other}", f"{m}_pref"]].dropna()
        t, o, p = (d.iloc[:, i].to_numpy(float) for i in range(3))
        base = {'measure': label, 'n_animals': int(len(d)),
                'mean_target': float(np.mean(t)) if len(d) else np.nan,
                'mean_other': float(np.mean(o)) if len(d) else np.nan,
                'mean_pref': float(np.mean(p)) if len(d) else np.nan}

        def _run(name, fn, *args):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r = fn(*args)
                stat, pval = float(r.statistic), float(r.pvalue)
            except Exception:  # too few animals / all differences zero
                stat, pval = np.nan, np.nan
            rows.append({**base, 'test': name, 'statistic': stat, 'p': pval})
        _run("paired t (target vs other)", stats.ttest_rel, t, o)
        _run("Wilcoxon (target vs other)", stats.wilcoxon, t, o)
        _run("one-sample t (pref vs 0.5)", stats.ttest_1samp, p, 0.5)
    tests = pd.DataFrame(rows, columns=['measure', 'test', 'n_animals', 'mean_target',
                                        'mean_other', 'mean_pref', 'statistic', 'p'])
    return {'per_animal': per_animal, 'tests': tests}


def preference_by_session(table: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SEM preference ratios across animals for each session (date)."""
    rows = []
    for (s, date), d in table.groupby(['session', 'date'], sort=True):
        r = {'session': s, 'date': date}
        for m in PREFERENCE_MEASURES:
            v = d[f"{m}_pref"].dropna().to_numpy(float)
            r[f"{m}_pref_mean"] = float(v.mean()) if v.size else np.nan
            r[f"{m}_pref_sem"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
            r[f"{m}_n"] = int(v.size)
        rows.append(r)
    return pd.DataFrame(rows)


def preference_timecourse(res: CombinedResults, *, bin_s: float, length_s: float,
                          exclude: set | frozenset = frozenset(), target: int = 1) -> dict:
    """Cumulative licks per bottle and cumulative lick preference over the session.

    Rows = animal sessions with both bottles kept. pref[i, j] = cumulative target
    licks / cumulative total licks by the end of bin j (NaN until the first lick).
    """
    if not res.two_bottle:
        raise ValueError("fluid preference needs 2-bottle results")
    edges = bin_edges_for(length_s, bin_s)
    drop = _sessions_with_excluded_fluid(exclude)
    counts = {f: [] for f in FLUIDS}
    with h5py.File(res.path, 'r') as h5f:
        for animal_id in res.animal_ids:
            if animal_id not in h5f:
                continue
            for key, g in h5f[animal_id].items():
                if not key.isdigit() or (int(key), animal_id) in drop:
                    continue
                chans = [g.get(fluid_group_name(f)) for f in FLUIDS]
                if any(c is None or 'lick_times' not in c for c in chans):
                    continue
                for f, c in zip(FLUIDS, chans):
                    counts[f].append(np.histogram(c['lick_times'][()], bins=edges)[0])
    nb = len(edges) - 1
    cum = {f: np.cumsum(np.array(v).reshape(-1, nb), axis=1) for f, v in counts.items()}
    return {
        'edges': edges,
        'centers_min': (edges[:-1] + edges[1:]) / 2 / 60,
        'cum_licks': cum,
        'pref': _ratio(cum[target], cum[3 - target]),
    }


# ---------------------------------------------------------------------------
# Windowed, decimated trace access
# ---------------------------------------------------------------------------
def h5_searchsorted(ds, value: float) -> int:
    """np.searchsorted(ds, value, 'left') on a sorted 1-D HDF5 dataset,
    by bisection (~log2(N) single-element reads instead of loading N values)."""
    lo, hi = 0, int(ds.shape[0])
    while lo < hi:
        mid = (lo + hi) // 2
        if ds[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _minmax_reduce(t, y, k):
    """Per-bin (t_first, min, max) for consecutive bins of k samples."""
    n = len(y)
    nb = int(np.ceil(n / k))
    pad = nb * k - n
    yp = np.concatenate([y, np.full(pad, y[-1])]) if pad else y
    yb = yp.reshape(nb, k)
    return t[::k][:nb], yb.min(axis=1), yb.max(axis=1)


def read_window(group, t0: float | None = None, t1: float | None = None,
                max_points: int = 4000, chunk: int = 2_000_000) -> dict:
    """Read [t0, t1] of a group's time_data/cap_data, min/max-decimated.

    Returns dict(t, y, decimated, i0, i1, k): when decimated, t/y are an
    interleaved min/max envelope (2 points per bin of k samples), otherwise raw.
    Reads the window in chunks so even a whole 24 h trace never needs more than
    `chunk` samples in memory at once.
    """
    tds, cds = group['time_data'], group['cap_data']
    n = int(tds.shape[0])
    if n == 0:
        return {'t': np.array([]), 'y': np.array([]), 'decimated': False, 'i0': 0, 'i1': 0, 'k': 1}
    i0 = 0 if t0 is None else h5_searchsorted(tds, t0)
    i1 = n if t1 is None else min(n, h5_searchsorted(tds, t1) + 1)
    i1 = max(i1, min(i0 + 2, n))
    m = i1 - i0
    half = max(max_points // 2, 1)
    if m <= max_points:
        return {'t': tds[i0:i1].astype(float), 'y': cds[i0:i1].astype(float),
                'decimated': False, 'i0': i0, 'i1': i1, 'k': 1}
    k = int(np.ceil(m / half))
    step = max(k, (chunk // k) * k)
    ts, mins, maxs = [], [], []
    for a in range(i0, i1, step):
        b = min(a + step, i1)
        tb, lo, hi = _minmax_reduce(tds[a:b], cds[a:b].astype(float), k)
        ts.append(tb); mins.append(lo); maxs.append(hi)
    tb = np.concatenate(ts); lo = np.concatenate(mins); hi = np.concatenate(maxs)
    return {'t': np.repeat(tb, 2), 'y': np.column_stack([lo, hi]).ravel(),
            'decimated': True, 'i0': i0, 'i1': i1, 'k': k,
            'bin_t': tb, 'bin_min': lo}


def session_window(res: CombinedResults, animal_id: str, session: int,
                   t0: float | None = None, t1: float | None = None,
                   max_points: int = 4000, fluid: int = 0) -> dict:
    """Decimated trace + lick markers for one session (one bottle of a 2-bottle
    session if `fluid` is 1 or 2), restricted to [t0, t1] (s)."""
    with h5py.File(res.path, 'r') as h5f:
        g = _channel(h5f[animal_id][str(session)], fluid)
        w = read_window(g, t0, t1, max_points=max_points)
        n = int(g['time_data'].shape[0])
        w['duration'] = float(g['time_data'][-1]) if n else 0.0
        li = g['lick_indices'][()].astype(np.int64) if 'lick_indices' in g else np.array([], int)
        li = li[(li >= w['i0']) & (li < w['i1'])]
        rel = li - w['i0']
        if not w['decimated']:
            w['lick_t'], w['lick_y'] = w['t'][rel], w['y'][rel]
        elif li.size > 2000:
            # Too many licks to read one by one (HDF5 point reads are slow):
            # put each marker on the minimum of the decimation bin holding it.
            b = np.clip(rel // w['k'], 0, len(w['bin_t']) - 1)
            w['lick_t'], w['lick_y'] = w['bin_t'][b], w['bin_min'][b]
        else:
            # Read just the lick samples exactly, so markers sit on the trace
            # value at the lick (h5py needs sorted, unique indices).
            uniq = np.unique(li)
            vals = g['cap_data'][uniq] if uniq.size else np.array([])
            times = g['time_data'][uniq] if uniq.size else np.array([])
            pos = np.searchsorted(uniq, li)
            w['lick_t'] = np.asarray(times, float)[pos] if uniq.size else np.array([])
            w['lick_y'] = np.asarray(vals, float)[pos] if uniq.size else np.array([])
        w['n_licks_window'] = int(li.size)
        w['consumed_vol'] = _scalar(g['consumed_vols']) if 'consumed_vols' in g else np.nan
    return w


def raw_sensor_overview(raw_path: str, sensor: int, t0=None, t1=None,
                        max_points: int = 4000, fluid: int | None = None) -> dict:
    """Decimated raw trace (absolute Unix time) + every recorded start/stop cycle,
    for checking start/stop times and writing *_time_fix.xlsx files.

    For a 2-bottle sensor the trace is one bottle's (`fluid`, default 1); the
    start/stop cycles are shared. w['two_bottle'] says which kind it was."""
    board = SENSOR_BOARD_MAP.get(int(sensor))
    if board is None:
        raise KeyError(f"Sensor {sensor} is not part of the current rack layout")
    with h5py.File(raw_path, 'r') as h5f:
        if board not in h5f or f"sensor_{sensor}" not in h5f[board]:
            raise KeyError(f"{board}/sensor_{sensor} not found in {os.path.basename(raw_path)}")
        g = h5f[board][f"sensor_{sensor}"]
        two_bottle = 'time_data' not in g and fluid_group_name(FLUIDS[0]) in g
        fluid = (fluid or FLUIDS[0]) if two_bottle else None
        data_g = g[fluid_group_name(fluid)] if two_bottle else g
        w = read_window(data_g, t0, t1, max_points=max_points)
        w['two_bottle'], w['fluid'] = two_bottle, fluid
        tds = data_g['time_data']
        w['t_first'] = float(tds[0]) if tds.shape[0] else np.nan
        w['t_last'] = float(tds[-1]) if tds.shape[0] else np.nan
        cycles = []
        for key in g.keys():
            m = re.match(r'^start_time(\d+)?$', key)
            if not m:
                continue
            num = int(m.group(1)) if m.group(1) else -1
            stop_key = 'stop' + key[5:]
            start = float(g[key][()])
            stop = float(g[stop_key][()]) if stop_key in g else np.nan
            cycles.append({'cycle': num, 'start_time': start, 'stop_time': stop,
                           'duration_min': (stop - start) / 60 if not np.isnan(stop) else np.nan})
        w['cycles'] = sorted(cycles, key=lambda c: c['cycle'])
        w['board'] = board
    return w


# ---------------------------------------------------------------------------
# CSV exports (same file names / layouts the notebook produced, for Prism)
# ---------------------------------------------------------------------------
def _clip_list(a, cap):
    return [v if v < cap else cap for v in a]


def fluid_tag(fluid: int) -> str:
    """CSV name tag for one bottle of 2-bottle results ('' for 1-bottle)."""
    return f"_fluid{fluid}" if fluid else ""


def export_licks_vs_vol(det: dict, out_dir: str, tag: str = "") -> list[str]:
    p = os.path.join(out_dir, f"cliqr_licks_vs_vol{tag}.csv")
    pd.DataFrame(np.vstack([det['x'], det['y']]).T,
                 columns=["Licks Detected", "Volume Consumed (mL)"]).to_csv(p)
    return [p]


def export_metrics(m: dict, out_dir: str, suffix: str = "", tag: str = "") -> list[str]:
    """suffix '' -> notebook's headerless '0' column CSVs; '_outliers_removed' -> named headers.
    tag (e.g. '_fluid1') names one bottle of 2-bottle results."""
    paths = []
    specs = [("cliqr_ILIs", m['ILIs'] * 1000, 700, "ILI (ms)"),
             ("cliqr_bout_durations", m['bout_durations'], 15, "Bout Duration (s)"),
             ("cliqr_licks_per_bout", m['bout_lick_counts'], 100, "Licks per Bout")]
    if 'lick_durations' in m:
        # Cap at the 1/6 s detection limit, same idea as the other caps.
        specs.append(("cliqr_lick_durations", m['lick_durations'] * 1000, 1000 / 6,
                      "Lick Duration (ms)"))
    for name, arr, cap, header in specs:
        p = os.path.join(out_dir, f"{name}{tag}{suffix}.csv")
        df = pd.DataFrame(_clip_list(arr, cap))
        if suffix:
            df.to_csv(p, index=False, header=[header])
        else:
            df.to_csv(p, index=False)
        paths.append(p)
    return paths


def export_lick_events(df: pd.DataFrame, out_dir: str, suffix: str = "",
                       tag: str = "") -> list[str]:
    p = os.path.join(out_dir, f"cliqr_lick_events{tag}{suffix}.csv")
    df.to_csv(p, index=False, float_format="%.4f")
    return [p]


def export_cumulative(tc: dict, out_dir: str, suffix: str = "", tag: str = "") -> list[str]:
    paths = []
    for key, label in (('licks', "Cumulative Licks"), ('bouts', "Cumulative Bouts")):
        mean, sem = mean_sem(np.cumsum(tc[key], axis=1))
        p = os.path.join(out_dir, f"cliqr_cumulative_{key}{tag}{suffix}.csv")
        pd.DataFrame(np.vstack([tc['centers_min'], mean, sem]).T,
                     columns=["Time (min)", label, "Standard Error"]).to_csv(p, index=False)
        paths.append(p)
    return paths


def export_clean_regression(fit: dict, out_dir: str, tag: str = "") -> list[str]:
    p = os.path.join(out_dir, f"nLicksVsVol{tag}_outliers_removed.csv")
    pd.DataFrame(np.vstack([fit['x'], fit['y']]).T,
                 columns=["Number of Licks", "Volume Consumed"]).to_csv(p, index=False)
    return [p]


def export_per_animal(table: pd.DataFrame, out_dir: str, tag: str = "") -> list[str]:
    p = os.path.join(out_dir, f"per_animal_lick_vs_volume_fits{tag}.csv")
    table.to_csv(p, index=False)
    return [p]


def export_session_table(res: CombinedResults, out_dir: str) -> list[str]:
    p = os.path.join(out_dir, "cliqr_session_summary.csv")
    df = res.sessions.drop(columns=['raw_h5'])
    if res.two_bottle:
        df.insert(df.columns.get_loc('fluid') + 1, 'fluid_name',
                  [res.fluid_label(f) for f in df['fluid']])
    else:
        df = df.drop(columns=['fluid'])
    df.to_csv(p, index=False)
    return [p]


def export_preference(table: pd.DataFrame, summary: dict, out_dir: str,
                      suffix: str = "") -> list[str]:
    """Per-session, per-animal and test tables of a 2-bottle preference analysis."""
    paths = []
    for name, df in (("cliqr_preference_by_session", table),
                     ("cliqr_preference_by_animal", summary['per_animal']),
                     ("cliqr_preference_tests", summary['tests'])):
        p = os.path.join(out_dir, f"{name}{suffix}.csv")
        df.to_csv(p, index=False)
        paths.append(p)
    return paths
