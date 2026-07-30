"""
Find "interesting" time windows in already-analyzed capacitance recordings and turn them into
ready-to-run make_sync_video.py calls.

Motivation
----------
When we want to eyeball the synchronised video + capacitance trace, we do not want to watch two
hours of mostly-flat signal. We want a short list of windows where SOMETHING happens:

  * LICKING  -- the animal drank. These are already detected: the lick-detection analysis writes
                per-bout `bout_start_times` / `bout_durations` into the combined results file, so we
                do NOT re-detect anything. We simply pick a few of the busiest bouts.

  * CLIMBING -- the animal moved on/around the sipper without drinking. These are NOT licks, so the
                detector ignores them, but they show up as large, slow excursions in the raw
                capacitance. We surface them as high-VARIANCE stretches that do NOT overlap a
                detected bout (masking the bouts is what keeps licking out of this category).

This script is meant to run AFTER the lick-detection analysis (i.e. on a `results_combined_*.h5`
produced by DataAnalysis.ipynb). It never re-runs detection.

What it produces
----------------
1. A CSV (`--csv`, default rois.csv) with one row per region of interest, across every requested
   animal and recording cycle. Columns are documented in `write_csv` below. The `start` / `end`
   columns are in the SAME zero-based per-cycle time reference that make_sync_video.py's
   `--start` / `--end` expect (both come from `filter_data`, which rebases each cycle's time axis
   so t = 0 is the recording's Start bookmark).

2. A runnable shell script (`--sh`, default make_clips.sh) with one make_sync_video.py command per
   region -- but ONLY for the single animal that each recording's camera actually filmed. A
   recording has exactly one video sensor (see video.trimcrop.find_video_sensor), hence one filmed
   animal; the other animals in the same cycle have no video, so we cannot make a sync clip for
   them (they still appear in the CSV for reference).

Time-base caveat (restart recordings)
--------------------------------------
For a plain single-cycle recording, the combined-file time base and make_sync_video's `--start`
reference agree, because both are produced by `filter_data`. But a RESTART recording (the operator
stopped and restarted within one raw file, writing numbered `start_time1`, `start_time2`, ...) has
a known history of the two references disagreeing by a fixed offset (~280 s was seen for the
2026-07-22 recording). We do NOT silently shift times to "fix" this, because getting the shift
subtly wrong would misalign every clip. Instead, for any cycle whose raw file is a restart
recording, the emitted command is preceded by a loud WARNING comment telling you to verify the
alignment and, if needed, pass `--offset` to make_sync_video (or re-run this script with
`--offset <cycle>=<seconds>`), which adds the offset to that cycle's start/end.

Provenance
----------
To build make_sync_video commands we need, per cycle, the raw recording `.h5` and the layout CSV.
Going forward DataAnalysis.ipynb stores these as attributes (`raw_h5`, `layout`) on each per-cycle
subgroup. For an older combined file that predates that change, pass a `--raw-map` JSON to supply
them (see `load_raw_map`); without either, the CSV is still written but the shell script will be
empty (a note is printed).

Usage (from the repository root, cliqr-gui environment):
    python scripts/find_interesting_windows.py \
        "Lickometry Data/results_combined_ACG-26-3_2026-07-22_23_24_27_28_29_basic-algorithm.h5" \
        --csv rois.csv --sh make_clips.sh \
        --n-lick 3 --n-climb 3 --roi-seconds 12

    # Backfilling provenance for an older combined file:
    python scripts/find_interesting_windows.py results_combined_old.h5 \
        --raw-map raw_map.json
"""

import argparse
import csv
import json
import os
import re
import sys

import h5py
import numpy as np

# Allow running from anywhere: make sure the repository root (this file's parent's parent) is on
# the import path so `import video.trimcrop` and `import data_analysis` resolve even when the
# current working directory is not the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# These primitives already know how the recordings are laid out; reuse them rather than
# re-implementing the (fiddly) video-sensor / cycle-suffix logic here.
from video.trimcrop import find_video_sensor, read_video_anchor, _resolve_cycle  # noqa: E402


# ---------------------------------------------------------------------------
# Control-cage detection
# ---------------------------------------------------------------------------
# Control cages ("Control1", "Control2", ...) have a sipper but no animal, so their "licks" are
# noise and they never climb. We skip them by default.
_CONTROL_RE = re.compile(r"^Control\d+$")


def is_control(animal_id):
    """True if `animal_id` names a control cage (no animal), which we skip by default."""
    return bool(_CONTROL_RE.match(str(animal_id)))


# ---------------------------------------------------------------------------
# Sliding-window variance
# ---------------------------------------------------------------------------
def sliding_variance(signal_values, window_samples):
    """Variance of `signal_values` inside a sliding window `window_samples` wide.

    Returned array element i is the variance of the window covering samples [i, i + window_samples),
    so it has length ``len(signal_values) - window_samples + 1``. Its "position" is the CENTER of
    that window, i.e. sample ``i + window_samples // 2`` (see `center_sample_indices`).

    We compute it in one pass using prefix sums rather than a Python loop, because a recording has
    ~800,000 samples and a naive per-window loop would be far too slow:

        Var(window) = E[x^2] - (E[x])^2

    where E[x] and E[x^2] over each window are differences of cumulative sums. This is exact for
    the (integer-valued) capacitance data; tiny negative values from floating-point round-off are
    clamped to zero.
    """
    x = np.asarray(signal_values, dtype=np.float64)
    w = int(window_samples)
    if w < 1:
        raise ValueError(f"window_samples must be >= 1 (got {window_samples})")
    if x.size < w:
        # Recording shorter than one window: no variance windows exist.
        return np.array([], dtype=np.float64)

    # Prefix sums of x and x^2. Prepending a 0 lets us get any window sum as a single subtraction:
    # sum over [i, i+w) == prefix[i+w] - prefix[i].
    prefix_sum = np.cumsum(np.insert(x, 0, 0.0))
    prefix_sum_sq = np.cumsum(np.insert(x * x, 0, 0.0))

    window_sum = prefix_sum[w:] - prefix_sum[:-w]
    window_sum_sq = prefix_sum_sq[w:] - prefix_sum_sq[:-w]

    window_mean = window_sum / w
    window_variance = window_sum_sq / w - window_mean * window_mean

    # Round-off can push a should-be-zero variance slightly negative; clamp so downstream logic
    # (and any sqrt a reader might add) stays well-defined.
    np.maximum(window_variance, 0.0, out=window_variance)
    return window_variance


def center_sample_indices(n_windows, window_samples):
    """Sample index at the CENTER of each sliding window produced by `sliding_variance`."""
    return np.arange(n_windows) + window_samples // 2


# ---------------------------------------------------------------------------
# Masking out detected bouts
# ---------------------------------------------------------------------------
def mask_bout_windows(window_variance, center_times_s, bout_start_times, bout_durations,
                      guard_seconds):
    """Set the variance to -inf for every window centered near a detected licking bout.

    We are hunting for CLIMBING (non-licking activity), so any window that sits on top of, or too
    close to, a detected bout must be disqualified -- otherwise the loud variance of a lick bout
    would masquerade as climbing. `guard_seconds` is how far (in seconds) on each side of a bout we
    extend the exclusion, so that a subsequently chosen fixed-width window cannot overlap the bout.

    The variance array is modified in place (and also returned for convenience).
    """
    if len(bout_start_times) == 0:
        return window_variance
    bout_start_times = np.asarray(bout_start_times, dtype=np.float64)
    bout_durations = np.asarray(bout_durations, dtype=np.float64)
    for bout_start, bout_duration in zip(bout_start_times, bout_durations):
        exclusion_start = bout_start - guard_seconds
        exclusion_end = bout_start + bout_duration + guard_seconds
        inside_exclusion = (center_times_s >= exclusion_start) & (center_times_s <= exclusion_end)
        window_variance[inside_exclusion] = -np.inf
    return window_variance


# ---------------------------------------------------------------------------
# Greedy non-overlapping peak selection
# ---------------------------------------------------------------------------
def select_climbing_centers(window_variance, center_times_s, n_wanted, min_separation_s):
    """Greedily pick up to `n_wanted` window centers with the highest variance, keeping every pick
    at least `min_separation_s` apart so the resulting fixed-width windows do not overlap.

    Returns a list of (center_time_s, variance) tuples, highest variance first. Windows that were
    masked to -inf (i.e. bout windows) are never chosen.
    """
    # Visit candidate centers from most to least variable.
    order = np.argsort(window_variance)[::-1]
    chosen_centers = []
    chosen_scores = []
    for idx in order:
        score = window_variance[idx]
        if not np.isfinite(score):
            # Everything from here on is masked (-inf) or invalid -- stop.
            break
        center_time = center_times_s[idx]
        # Reject if too close to an already-chosen center (would overlap it).
        too_close = any(abs(center_time - c) < min_separation_s for c in chosen_centers)
        if too_close:
            continue
        chosen_centers.append(center_time)
        chosen_scores.append(float(score))
        if len(chosen_centers) >= n_wanted:
            break
    return list(zip(chosen_centers, chosen_scores))


# ---------------------------------------------------------------------------
# Per-cycle region-of-interest construction
# ---------------------------------------------------------------------------
def count_licks_in_window(lick_times, start_s, end_s):
    """How many detected licks fall inside [start_s, end_s]. Used only for the CSV, so a reader can
    see at a glance whether a 'climbing' window is truly lick-free."""
    lick_times = np.asarray(lick_times, dtype=np.float64)
    if lick_times.size == 0:
        return 0
    return int(np.sum((lick_times >= start_s) & (lick_times <= end_s)))


def clip_window(center_s, roi_seconds, span_s):
    """Fixed-width window of `roi_seconds` centered on `center_s`, clipped to [0, span_s].

    The window is kept `roi_seconds` wide whenever possible; it only shrinks if the recording edge
    forces it (near t = 0 or near the end). Returns (start_s, end_s).
    """
    half = roi_seconds / 2.0
    start_s = max(0.0, center_s - half)
    end_s = min(span_s, center_s + half)
    return start_s, end_s


def build_rois_for_cycle(cap_data, time_data, bout_start_times, bout_durations,
                         bout_lick_counts, lick_times, params):
    """Build the licking and climbing regions of interest for a single (animal, cycle) trace.

    Returns a list of dicts, each describing one region:
        {category, rank, start, end, center, score, n_licks_in_window}
    where `category` is "lick" or "climb", `rank` is 0-based within its category (0 = strongest),
    `score` is the bout lick count (licking) or the window variance (climbing).
    """
    rois = []
    span_s = float(time_data[-1]) if len(time_data) else 0.0

    # --- Sampling rate (Hz), needed to turn the variance window from seconds into samples. ---
    # The capacitance is sampled at ~112 Hz but not exactly; derive it from the actual trace.
    n_samples = len(time_data)
    duration_s = span_s - float(time_data[0]) if n_samples > 1 else 0.0
    sampling_rate_hz = (n_samples - 1) / duration_s if duration_s > 0 else 0.0

    # ----------------------------------------------------------------------
    # LICKING regions: take the busiest detected bouts (most licks).
    # ----------------------------------------------------------------------
    bout_start_times = np.asarray(bout_start_times, dtype=np.float64)
    bout_durations = np.asarray(bout_durations, dtype=np.float64)
    bout_lick_counts = np.asarray(bout_lick_counts)
    if params["n_lick"] > 0 and len(bout_start_times) > 0:
        # Visit bouts busiest-first, but keep the chosen windows non-overlapping: skip a bout whose
        # window center falls within `roi_seconds` of an already-chosen licking window (two windows
        # of width `roi_seconds` overlap iff their centers are closer than `roi_seconds`).
        busiest = np.argsort(bout_lick_counts)[::-1]
        chosen_lick_centers = []
        rank = 0
        for bout_index in busiest:
            if len(chosen_lick_centers) >= params["n_lick"]:
                break
            bout_center = bout_start_times[bout_index] + bout_durations[bout_index] / 2.0
            if any(abs(bout_center - c) < params["roi_seconds"] for c in chosen_lick_centers):
                continue
            chosen_lick_centers.append(bout_center)
            start_s, end_s = clip_window(bout_center, params["roi_seconds"], span_s)
            rois.append({
                "category": "lick",
                "rank": rank,
                "start": start_s,
                "end": end_s,
                "center": bout_center,
                "score": float(bout_lick_counts[bout_index]),
                "n_licks_in_window": count_licks_in_window(lick_times, start_s, end_s),
            })
            rank += 1

    # ----------------------------------------------------------------------
    # CLIMBING regions: high-variance windows that avoid the detected bouts.
    # ----------------------------------------------------------------------
    variance_window_samples = max(1, int(round(params["var_window"] * sampling_rate_hz)))
    window_variance = sliding_variance(cap_data, variance_window_samples)
    if window_variance.size > 0 and sampling_rate_hz > 0:
        centers = center_sample_indices(window_variance.size, variance_window_samples)
        center_times_s = centers / sampling_rate_hz

        # Disqualify windows on/near any detected bout. Guard by the FULL ROI width: a licking
        # window is centered on the bout and is `roi_seconds` wide, and a climbing window is also
        # `roi_seconds` wide, so keeping climbing centers at least `roi_seconds` away from every
        # bout guarantees a climbing window overlaps neither the bout nor the bout's licking window.
        guard_seconds = params["roi_seconds"]
        mask_bout_windows(window_variance, center_times_s, bout_start_times, bout_durations,
                          guard_seconds)

        # Optional variance floor: ignore windows below this, so we don't dredge up flat-signal
        # "climbing" on a quiet recording. 0.0 (the default) disables the floor.
        if params["min_var"] > 0.0:
            window_variance[window_variance < params["min_var"]] = -np.inf

        picks = select_climbing_centers(
            window_variance, center_times_s,
            n_wanted=params["n_climb"],
            min_separation_s=params["roi_seconds"],
        )
        for rank, (center_time, variance) in enumerate(picks):
            start_s, end_s = clip_window(center_time, params["roi_seconds"], span_s)
            rois.append({
                "category": "climb",
                "rank": rank,
                "start": start_s,
                "end": end_s,
                "center": center_time,
                "score": variance,
                "n_licks_in_window": count_licks_in_window(lick_times, start_s, end_s),
            })

    return rois


# ---------------------------------------------------------------------------
# Provenance: raw .h5 + layout per cycle, and the filmed animal
# ---------------------------------------------------------------------------
def load_raw_map(path):
    """Load an optional JSON that supplies raw-h5 / layout provenance for combined files that
    predate storing it as attributes.

    Expected shape (keys are cycle indices, as strings):
        {
          "0": {"raw_h5": ".../raw_data_..._07-22.h5", "layout": ".../layout_w_controls.csv"},
          "1": {"raw_h5": "...", "layout": "..."}
        }
    Returns {} when `path` is None.
    """
    if path is None:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def cycle_provenance(combined_h5, animal_id, cycle_key, raw_map):
    """Resolve (raw_h5_path, layout_path) for one cycle, preferring the attributes written by
    DataAnalysis.ipynb and falling back to the `--raw-map` JSON. Returns (None, None) if neither
    supplies them, in which case no make_sync_video command can be built for this cycle."""
    group = combined_h5[animal_id][cycle_key]
    raw_h5 = group.attrs.get("raw_h5")
    layout = group.attrs.get("layout")
    if raw_h5 is not None and layout is not None:
        return str(raw_h5), str(layout)
    # Fall back to the supplied map (keyed by cycle index).
    entry = raw_map.get(str(cycle_key))
    if entry:
        return entry.get("raw_h5"), entry.get("layout")
    return None, None


def resolve_filmed_animal(raw_h5_path, layout_path):
    """The single animal that this recording's camera filmed, or None if it can't be determined.

    A recording has exactly one video sensor; `read_video_anchor` reports its sensor number, and
    the layout CSV maps that sensor number to an animal ID. Returns None (and stays quiet) if the
    raw file or layout is missing/unreadable -- provenance being absent is expected for old files.
    """
    if not raw_h5_path or not layout_path:
        return None
    if not os.path.exists(raw_h5_path) or not os.path.exists(layout_path):
        return None
    try:
        import pandas as pd
        anchor = read_video_anchor(raw_h5_path)
        layout = pd.read_csv(layout_path, header=None, index_col=0)
        return str(layout.loc[anchor.sensor_number].iloc[0])
    except (ValueError, KeyError, OSError, FileNotFoundError):
        return None


def is_restart_recording(raw_h5_path):
    """True if the raw recording is a RESTART recording (stopped/restarted mid-file), which is the
    case that carries the known time-base offset risk.

    A restart recording writes numbered cycle keys (`start_time1`, ...) on the video sensor group;
    `_resolve_cycle` returns a non-empty suffix in that case. Returns False if we can't tell (no
    file / no video sensor) -- we only warn when we're sure it's a restart."""
    if not raw_h5_path or not os.path.exists(raw_h5_path):
        return False
    try:
        with h5py.File(raw_h5_path, "r") as raw:
            board_id, sensor_name, _ = find_video_sensor(raw)
            _, cycle_suffix = _resolve_cycle(raw[board_id][sensor_name])
            return cycle_suffix != ""
    except (ValueError, KeyError, OSError):
        return False


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "animal", "cycle", "category", "rank",
    "start", "end", "center", "score",
    "n_licks_in_window", "filmed", "raw_h5", "layout",
]


def write_csv(rows, csv_path):
    """Write every region of interest to `csv_path`. One row per region; see CSV_COLUMNS."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def build_command(row, out_dir, offsets, combined_h5):
    """Build the make_sync_video.py command (a list of lines) for one filmed-animal region.

    Returns a list of text lines: any WARNING comment(s) followed by the command itself. `offsets`
    maps a cycle index (int) to a manual offset in seconds that is ADDED to start/end (used to
    correct a restart recording's time base once you've measured it). `combined_h5` is the analyzed
    results file; the command reads the trace from it (via --combined-h5/--cycle) so make_sync_video
    does not re-run filter_data on the raw recording."""
    lines = []
    cycle = row["cycle"]
    start_s = row["start"]
    end_s = row["end"]

    # Apply a user-supplied per-cycle offset, if any (see --offset).
    offset = offsets.get(cycle, 0.0)
    if offset:
        start_s = start_s + offset
        end_s = end_s + offset

    # Loudly flag restart recordings whose time base may be shifted (unless the user already
    # supplied an offset for this cycle, in which case we assume it's handled).
    if row.get("restart") and not offset:
        lines.append(f"# WARNING cycle {cycle} ({os.path.basename(row['raw_h5'])}) is a RESTART "
                     f"recording;")
        lines.append(f"#   the combined-file time base may be offset from make_sync_video's "
                     f"Start-bookmark reference.")
        lines.append(f"#   Verify alignment; if off, re-run with --offset {cycle}=<seconds> "
                     f"(or pass --sync-offset to make_sync_video).")

    out_name = f"{row['animal']}_c{cycle}_{row['category']}{row['rank']}.mp4"
    out_path = os.path.join(out_dir, out_name)
    lines.append(
        f"python make_sync_video.py --h5 {shquote(row['raw_h5'])} "
        f"--layout {shquote(row['layout'])} "
        f"--combined-h5 {shquote(combined_h5)} --cycle {cycle} "
        f"--start {start_s:.3f} --end {end_s:.3f} "
        f"--out {shquote(out_path)}"
    )
    return lines


def shquote(path):
    """Minimal shell quoting: wrap in double quotes if the path contains whitespace."""
    return f'"{path}"' if re.search(r"\s", str(path)) else str(path)


def write_shell_script(rows, sh_path, out_dir, offsets, combined_h5):
    """Write a runnable shell script with one make_sync_video command per FILMED-animal region.

    Rows that are not for the filmed animal (or lack provenance) are skipped -- there is no video
    for them. Returns the number of commands written."""
    command_blocks = []
    for row in rows:
        if not row.get("filmed"):
            continue
        if not row.get("raw_h5") or not row.get("layout"):
            continue
        command_blocks.append(build_command(row, out_dir, offsets, combined_h5))

    with open(sh_path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Auto-generated by scripts/find_interesting_windows.py\n")
        f.write("# One make_sync_video.py call per region of interest, for the filmed animal only.\n")
        f.write("# Review the WARNINGs (restart recordings) before running.\n")
        f.write("set -euo pipefail\n")
        f.write(f"mkdir -p {shquote(out_dir)}\n\n")
        for block in command_blocks:
            f.write("\n".join(block) + "\n\n")
    # Make it executable for convenience.
    os.chmod(sh_path, 0o755)
    return len(command_blocks)


# ---------------------------------------------------------------------------
# Offset parsing (--offset CYCLE=SECONDS)
# ---------------------------------------------------------------------------
def parse_offsets(offset_args):
    """Parse repeated --offset CYCLE=SECONDS arguments into {cycle_index: seconds}."""
    offsets = {}
    for item in offset_args or []:
        if "=" not in item:
            raise ValueError(f"--offset expects CYCLE=SECONDS (got {item!r})")
        cycle_str, seconds_str = item.split("=", 1)
        offsets[int(cycle_str)] = float(seconds_str)
    return offsets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Find interesting (licking / climbing) windows in an analyzed combined results "
                    "HDF5 and emit make_sync_video.py calls.")
    parser.add_argument("combined_h5", help="results_combined_*.h5 produced by DataAnalysis.ipynb")
    parser.add_argument("--csv", default="rois.csv", help="output CSV path (default rois.csv)")
    parser.add_argument("--sh", default="make_clips.sh",
                        help="output shell script path (default make_clips.sh)")
    parser.add_argument("--out-dir", default="clips",
                        help="directory the generated commands write their .mp4 clips to "
                             "(default clips)")
    parser.add_argument("--n-lick", type=int, default=3,
                        help="licking windows per (animal, cycle), busiest bouts first (default 3)")
    parser.add_argument("--n-climb", type=int, default=3,
                        help="climbing windows per (animal, cycle), highest variance first "
                             "(default 3)")
    parser.add_argument("--roi-seconds", type=float, default=12.0,
                        help="width of each emitted window in seconds (default 12)")
    parser.add_argument("--var-window", type=float, default=1.0,
                        help="sliding variance window in seconds for climbing detection (default 1)")
    parser.add_argument("--min-var", type=float, default=0.0,
                        help="ignore climbing windows below this variance (default 0 = no floor)")
    parser.add_argument("--animals", nargs="*", default=None,
                        help="restrict to these animal IDs (default: all non-control animals)")
    parser.add_argument("--include-controls", action="store_true",
                        help="also analyze control cages (skipped by default)")
    parser.add_argument("--raw-map", default=None,
                        help="JSON mapping cycle index -> {raw_h5, layout} for combined files that "
                             "predate stored provenance attributes")
    parser.add_argument("--offset", action="append", default=None, metavar="CYCLE=SECONDS",
                        help="add SECONDS to a cycle's start/end (repeatable) to correct a restart "
                             "recording's time base once measured")
    args = parser.parse_args(argv)

    raw_map = load_raw_map(args.raw_map)
    offsets = parse_offsets(args.offset)
    params = {
        "n_lick": args.n_lick,
        "n_climb": args.n_climb,
        "roi_seconds": args.roi_seconds,
        "var_window": args.var_window,
        "min_var": args.min_var,
    }

    all_rows = []
    # Cache filmed-animal / restart lookups per (cycle -> raw_h5) so we don't re-open a raw file
    # once per animal.
    filmed_cache = {}
    restart_cache = {}

    with h5py.File(args.combined_h5, "r") as combined:
        animal_ids = list(combined.keys())
        if args.animals:
            animal_ids = [a for a in animal_ids if a in set(args.animals)]
        elif not args.include_controls:
            animal_ids = [a for a in animal_ids if not is_control(a)]

        for animal_id in animal_ids:
            for cycle_key in combined[animal_id].keys():
                group = combined[animal_id][cycle_key]
                # A cycle must have the analysis outputs we depend on; skip anything malformed.
                required = ("cap_data", "time_data", "bout_start_times", "bout_durations",
                            "bout_lick_counts", "lick_times")
                if not all(name in group for name in required):
                    continue

                cap_data = group["cap_data"][:]
                time_data = group["time_data"][:]
                if len(time_data) < 2:
                    continue

                rois = build_rois_for_cycle(
                    cap_data, time_data,
                    group["bout_start_times"][:], group["bout_durations"][:],
                    group["bout_lick_counts"][:], group["lick_times"][:],
                    params,
                )

                # Resolve provenance + the filmed animal once per cycle (cached on raw_h5).
                raw_h5_path, layout_path = cycle_provenance(combined, animal_id, cycle_key, raw_map)
                cache_key = (cycle_key, raw_h5_path)
                if cache_key not in filmed_cache:
                    filmed_cache[cache_key] = resolve_filmed_animal(raw_h5_path, layout_path)
                    restart_cache[cache_key] = is_restart_recording(raw_h5_path)
                filmed_animal = filmed_cache[cache_key]
                restart = restart_cache[cache_key]

                # cycle_key comes from the HDF5 as a string; the CSV/commands use it verbatim, but
                # the integer form is what --offset keys on.
                try:
                    cycle_int = int(cycle_key)
                except ValueError:
                    cycle_int = cycle_key

                for roi in rois:
                    roi.update({
                        "animal": animal_id,
                        "cycle": cycle_int,
                        "filmed": (filmed_animal is not None and animal_id == filmed_animal),
                        "restart": restart,
                        "raw_h5": raw_h5_path or "",
                        "layout": layout_path or "",
                    })
                    all_rows.append(roi)

    # Sort for a tidy, deterministic CSV: by animal, then cycle, then category, then rank.
    all_rows.sort(key=lambda r: (str(r["animal"]), str(r["cycle"]), r["category"], r["rank"]))

    write_csv(all_rows, args.csv)
    n_commands = write_shell_script(all_rows, args.sh, args.out_dir, offsets, args.combined_h5)

    n_filmed_rows = sum(1 for r in all_rows if r["filmed"])
    print(f"Wrote {len(all_rows)} regions of interest to {args.csv}.")
    print(f"Wrote {n_commands} make_sync_video command(s) to {args.sh} "
          f"(filmed-animal regions only; {n_filmed_rows} filmed rows found).")
    if n_commands == 0:
        print("No commands were emitted. This usually means the combined file has no stored "
              "raw_h5/layout provenance and no --raw-map was given, or the raw files aren't at the "
              "stored paths. The CSV is still complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
