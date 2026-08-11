"""
Find "interesting" time windows in already-analyzed capacitance recordings and turn them into
ready-to-run make_sync_video.py calls.

Motivation
----------
When we want to eyeball the synchronised video + capacitance trace, we do not want to watch two
hours of mostly-flat signal. We want a short list of windows where SOMETHING happens:

  * LICKING  -- the animal drank. These are already detected: the lick-detection analysis writes
                per-bout `bout_start_times` / `bout_durations` into the combined results file, so we
                do NOT re-detect anything. We simply pick a few of the busiest bouts. Each window
                spans the WHOLE bout plus `--lick-pad` seconds of context on each side, so a long
                bout is never cut off mid-drink; `--roi-seconds` is only a minimum width.

  * CLIMBING -- the animal moved on/around the sipper without drinking. These are NOT licks, so the
                detector ignores them, but they show up as large, slow excursions in the raw
                capacitance. We surface them as high-VARIANCE stretches that do NOT overlap a
                detected bout (masking the bouts is what keeps licking out of this category).
                The first and last `--climb-skip-edges` seconds of a session are excluded: those
                are start-up / shut-down transients (cage handling, sipper insertion, sensor
                settling), which are usually the loudest excursions in the recording and would
                otherwise crowd out the real climbing. Licking is not edge-restricted.
                Both the variance windows and the mask are positioned using the trace's own
                `time_data`: the capacitance is NOT sampled uniformly (spacing ranges from a few ms
                to several hundred ms), so converting a sample index with an average rate drifts by
                tens of seconds over a 2 h recording.

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
   region -- climbing commands get `--no-crop` (the sipper-tip crop box hides the cage the animal
   climbs on; licking commands keep the crop, which is what makes the tongue visible) -- but ONLY
   for the single animal that each recording's camera actually filmed. A
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
from video.trimcrop import (  # noqa: E402
    find_video_sensor, read_video_anchor, _resolve_cycle,
    frame_session_times, resolve_paths, session_clock,
)


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


def clip_window(center_s, roi_seconds, span_s, first_s=0.0):
    """Fixed-width window of `roi_seconds` centered on `center_s`, clipped to [first_s, span_s].

    The window is kept `roi_seconds` wide whenever possible; it only shrinks if the recording edge
    forces it (near the first sample or near the end). Returns (start_s, end_s).
    """
    half = roi_seconds / 2.0
    start_s = max(first_s, center_s - half)
    end_s = min(span_s, center_s + half)
    return start_s, end_s


def bout_window(bout_start_s, bout_duration_s, roi_seconds, lick_pad_s, span_s, first_s=0.0):
    """Window for a licking bout: the WHOLE bout plus `lick_pad_s` of context on each side.

    A fixed `roi_seconds` window centered on the bout would truncate any bout longer than
    `roi_seconds` (real recordings have bouts of 25 s+), cutting licks off the end of the clip --
    which defeats the point of watching the clip. So the window grows to fit the bout, and
    `roi_seconds` acts only as a MINIMUM width, so a one-second bout still gets context around it.

    Returns (start_s, end_s), clipped to the recording bounds [first_s, span_s].
    """
    start_s = bout_start_s - lick_pad_s
    end_s = bout_start_s + bout_duration_s + lick_pad_s
    if end_s - start_s < roi_seconds:
        # Too short: widen symmetrically about the bout center to the minimum width.
        center_s = bout_start_s + bout_duration_s / 2.0
        start_s = center_s - roi_seconds / 2.0
        end_s = center_s + roi_seconds / 2.0
    return max(first_s, start_s), min(span_s, end_s)


def bout_window_half_width(bout_duration_s, roi_seconds, lick_pad_s):
    """How far a bout's licking window extends beyond the bout's CENTER, in seconds.

    Used to size the climbing exclusion guard: it must cover the bout's own licking window, not
    just the bout, so climbing and licking clips never show the same stretch of recording.
    """
    return max(bout_duration_s / 2.0 + lick_pad_s, roi_seconds / 2.0)


def build_rois_for_cycle(cap_data, time_data, bout_start_times, bout_durations,
                         bout_lick_counts, lick_times, params):
    """Build the licking and climbing regions of interest for a single (animal, cycle) trace.

    Returns a list of dicts, each describing one region:
        {category, rank, start, end, center, score, n_licks_in_window}
    where `category` is "lick" or "climb", `rank` is 0-based within its category (0 = strongest),
    `score` is the bout lick count (licking) or the window variance (climbing).
    """
    rois = []
    time_data = np.asarray(time_data, dtype=np.float64)
    span_s = float(time_data[-1]) if len(time_data) else 0.0
    first_s = float(time_data[0]) if len(time_data) else 0.0
    lick_pad_s = float(params.get("lick_pad", 2.0))

    # --- Typical sample spacing, needed to turn the variance window from seconds into samples. ---
    # The capacitance is nominally sampled at ~112 Hz, but the real trace is NOT uniform: the
    # hardware stalls, so the spacing ranges from ~2 ms to several hundred ms. We use the MEDIAN
    # spacing (robust to those stalls) purely to size the variance window in samples; every window
    # POSITION is read from `time_data` itself (see below), never inferred from a rate.
    n_samples = len(time_data)
    median_dt_s = float(np.median(np.diff(time_data))) if n_samples > 1 else 0.0

    # ----------------------------------------------------------------------
    # LICKING regions: take the busiest detected bouts (most licks).
    # ----------------------------------------------------------------------
    bout_start_times = np.asarray(bout_start_times, dtype=np.float64)
    bout_durations = np.asarray(bout_durations, dtype=np.float64)
    bout_lick_counts = np.asarray(bout_lick_counts)
    if params["n_lick"] > 0 and len(bout_start_times) > 0:
        # Visit bouts busiest-first, but keep the chosen windows non-overlapping. Windows are no
        # longer all the same width (a long bout makes a wide one), so we compare the actual
        # intervals rather than the distance between centers.
        busiest = np.argsort(bout_lick_counts)[::-1]
        chosen_lick_windows = []
        rank = 0
        for bout_index in busiest:
            if len(chosen_lick_windows) >= params["n_lick"]:
                break
            bout_center = bout_start_times[bout_index] + bout_durations[bout_index] / 2.0
            start_s, end_s = bout_window(bout_start_times[bout_index], bout_durations[bout_index],
                                         params["roi_seconds"], lick_pad_s, span_s, first_s)
            if any(start_s < chosen_end and end_s > chosen_start
                   for chosen_start, chosen_end in chosen_lick_windows):
                continue
            chosen_lick_windows.append((start_s, end_s))
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
    variance_window_samples = (max(1, int(round(params["var_window"] / median_dt_s)))
                               if median_dt_s > 0 else 1)
    window_variance = sliding_variance(cap_data, variance_window_samples)
    if window_variance.size > 0 and median_dt_s > 0:
        centers = center_sample_indices(window_variance.size, variance_window_samples)
        # Read each window's time from the trace's own time base. Converting a sample index with an
        # average rate silently assumes uniform sampling; on a real recording that drifts by tens of
        # SECONDS by the end (the stalls above), which both mis-times the emitted clip and makes the
        # bout mask below disqualify the wrong stretches.
        center_times_s = time_data[centers]

        # Disqualify windows on/near any detected bout. Guard by `roi_seconds + lick_pad`: that is
        # at least the half-width of the bout's own licking window plus the half-width of a climbing
        # window, so a surviving climbing window overlaps neither the bout nor its licking clip.
        guard_seconds = params["roi_seconds"] + lick_pad_s
        mask_bout_windows(window_variance, center_times_s, bout_start_times, bout_durations,
                          guard_seconds)

        # Disqualify the start and end of the session. Both edges are dominated by transients that
        # have nothing to do with the animal climbing -- the operator handling the cage, inserting
        # the sipper, the sensor settling at the start, and the reverse at the end -- and those
        # transients are usually the LOUDEST excursions in the whole recording, so without this they
        # crowd out the real climbing. The whole window (not just its center) must clear the edge,
        # hence the extra half-window. Licking is untouched: a real bout near an edge is real.
        skip_edges_s = float(params.get("climb_skip_edges", 300.0))
        if skip_edges_s > 0.0:
            half_window_s = params["roi_seconds"] / 2.0
            too_early = center_times_s < first_s + skip_edges_s + half_window_s
            too_late = center_times_s > span_s - skip_edges_s - half_window_s
            window_variance[too_early | too_late] = -np.inf

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
            start_s, end_s = clip_window(center_time, params["roi_seconds"], span_s, first_s)
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
# DLC window mode
# ---------------------------------------------------------------------------
# `dlc_integration/find_dlc_windows.py` writes one row per stretch of video where the animal was
# confidently at the sipper. Those rows are frame ranges into the ORIGINAL recording's container
# (half-open, `[start_frame, end_frame)`), which is a different time reference from the one
# make_sync_video takes -- see `frame_window_to_session` for the conversion.


def parse_dlc_video_stem(video_path):
    """(raw_stem, is_cfr) for a DLC CSV `video` path.

    DLC was run on a CFR re-encode for the older sessions (`<stem>_cfr.mp4`) and on the original
    recording for the newer ones. The raw `.h5` is always named after the stem WITHOUT `_cfr`, so
    strip it -- but report that we did, because a re-encode's frame indices are NOT guaranteed to be
    the original container's ordinals (`-fps_mode cfr` drops frames to hit a flat rate), which makes
    those rows unrenderable rather than merely inconvenient.
    """
    stem = os.path.splitext(os.path.basename(str(video_path)))[0]
    if stem.endswith("_cfr"):
        return stem[: -len("_cfr")], True
    return stem, False


def read_dlc_windows(path):
    """Read a find_dlc_windows.py CSV into plain dicts with the fields this script needs.

    Only `label`, `video`, `start_frame`, `end_frame` and `tongue_rate` are used; the rest of the
    columns (likelihoods, distances, the derived seconds) are diagnostics for a human reading that
    CSV. `tongue_rate` is blank on a row written before the column existed, which reads as 0.0.
    """
    rows = []
    with open(path, "r", newline="") as f:
        for record in csv.DictReader(f):
            rate = (record.get("tongue_rate") or "").strip()
            rows.append({
                "label": record.get("label", ""),
                "video": record.get("video", ""),
                "start_frame": int(record["start_frame"]),
                "end_frame": int(record["end_frame"]),
                "tongue_rate": float(rate) if rate else 0.0,
            })
    return rows


def frame_window_to_session(sess, start_frame, end_frame):
    """Session-time window (start_s, end_s) for a DLC frame range, or None if it starts past the
    end of the recording's per-frame time array.

    `sess` is `frame_session_times(clock, container_pts_ns)`: the session time of every container
    frame, latency- and drift-corrected. Indexing it is the whole conversion -- do NOT compute
    `start_frame / fps`, which is video-file time and drifts against session time by seconds over a
    long recording (see docs/video-sync-alignment-bugs.md).

    `end_frame` is EXCLUSIVE (the CSV mirrors `Frames2plot=range(start, end)`), so the window ends
    at frame `end_frame - 1`, clamped to the last frame we have a time for.
    """
    sess = np.asarray(sess, dtype=np.float64)
    if sess.size == 0 or start_frame >= sess.size:
        return None
    last = min(int(end_frame) - 1, sess.size - 1)
    if last < start_frame:
        return None
    return float(sess[int(start_frame)]), float(sess[last])


def clamp_to_trace(start_s, end_s, first_s, span_s):
    """Clip a window to the trace's own [first_s, span_s], or None if nothing is left.

    A window can fall outside the trace because the video and the capacitance recording do not
    start and stop at exactly the same instant. Clipping keeps a partially-overlapping window
    renderable (make_sync_video rejects an --end past the session), and dropping a disjoint one
    keeps a clip with no trace to draw out of the shell script.
    """
    start_s = max(float(start_s), float(first_s))
    end_s = min(float(end_s), float(span_s))
    if end_s <= start_s:
        return None
    return start_s, end_s


def load_frame_session_times(raw_h5_path):
    """Session time of every container frame of a recording, or None if we cannot get it.

    This is the same three-step the renderer does: read the video anchor out of the raw .h5, build
    the SessionClock (bookmark latency + two-bookmark drift slope), and time every frame of the
    PTS sidecar with it. Returning None -- rather than raising -- for a missing raw file, missing
    video sensor, or missing sidecar is deliberate: a combined file routinely names recordings that
    are not on this machine, and those videos simply produce no clips.
    """
    if not raw_h5_path or not os.path.exists(raw_h5_path):
        return None
    # Lazy: make_sync_video drags in matplotlib/imageio/pandas, which the trace-search mode has no
    # use for. It owns the "which sidecar times container frames" rule (the Pi's encoded-frame
    # sidecar when present, else the capture sidecar); reuse it so a window computed here always
    # selects the frames the renderer will place.
    from make_sync_video import load_container_pts
    try:
        anchor = read_video_anchor(raw_h5_path)
        _, pts_txt = resolve_paths(raw_h5_path, anchor)
        if not os.path.exists(pts_txt):
            return None
        pts_ns = np.loadtxt(pts_txt, dtype=np.int64)
        if np.asarray(pts_ns).size < 2:
            return None
        clock = session_clock(anchor, pts_ns)
        return frame_session_times(clock, load_container_pts(pts_txt, pts_ns))
    except (ValueError, KeyError, OSError, IndexError):
        return None


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


def build_command(row, out_dir, offsets, combined_h5, speed=1.0):
    """Build the make_sync_video.py command (a list of lines) for one filmed-animal region.

    Returns a list of text lines: any WARNING comment(s) followed by the command itself. `offsets`
    maps a cycle index (int) to a manual offset in seconds that is ADDED to start/end (used to
    correct a restart recording's time base once you've measured it). `combined_h5` is the analyzed
    results file; the command reads the trace from it (via --combined-h5/--cycle) so make_sync_video
    does not re-run filter_data on the raw recording. `speed` < 1 renders slow motion (see
    make_sync_video's --speed); at the default 1.0 no flag is emitted, so the script is unchanged."""
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
    # The crop box (crop_video.py's sidecar) is framed tightly on the sipper tip so the tongue is
    # visible during licking. A CLIMBING clip needs the opposite: the animal is on/around the sipper
    # and the cage, mostly OUTSIDE that box, so cropping hides the very behaviour we are reviewing.
    # Render climbing clips full-frame.
    crop_flag = " --no-crop" if row["category"] == "climb" else ""
    speed_flag = f" --speed {speed:g}" if speed != 1.0 else ""
    lines.append(
        f"python make_sync_video.py --h5 {shquote(row['raw_h5'])} "
        f"--layout {shquote(row['layout'])} "
        f"--combined-h5 {shquote(combined_h5)} --cycle {cycle} "
        f"--start {start_s:.3f} --end {end_s:.3f}{crop_flag}{speed_flag} "
        f"--out {shquote(out_path)}"
    )
    return lines


def shquote(path):
    """Minimal shell quoting: wrap in double quotes if the path contains whitespace."""
    return f'"{path}"' if re.search(r"\s", str(path)) else str(path)


def write_shell_script(rows, sh_path, out_dir, offsets, combined_h5, speed=1.0):
    """Write a runnable shell script with one make_sync_video command per FILMED-animal region.

    Rows that are not for the filmed animal (or lack provenance) are skipped -- there is no video
    for them. Returns the number of commands written."""
    command_blocks = []
    for row in rows:
        if not row.get("filmed"):
            continue
        if not row.get("raw_h5") or not row.get("layout"):
            continue
        command_blocks.append(build_command(row, out_dir, offsets, combined_h5, speed))

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
                        help="MINIMUM width of each emitted window in seconds (default 12); a "
                             "licking window grows beyond this when the bout is longer")
    parser.add_argument("--lick-pad", type=float, default=2.0,
                        help="seconds of context kept on each side of a licking bout (default 2); "
                             "the window always contains the whole bout")
    parser.add_argument("--climb-skip-edges", type=float, default=300.0,
                        help="seconds at the START and END of each session excluded from the "
                             "CLIMBING search (default 300); start-up/shut-down transients are not "
                             "climbing. Licking bouts near the edges are still reported")
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
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed for every emitted clip (default 1.0 = real time); "
                             "0.25 gives quarter-speed slow motion with no frames dropped")
    args = parser.parse_args(argv)

    raw_map = load_raw_map(args.raw_map)
    offsets = parse_offsets(args.offset)
    params = {
        "n_lick": args.n_lick,
        "n_climb": args.n_climb,
        "roi_seconds": args.roi_seconds,
        "lick_pad": args.lick_pad,
        "climb_skip_edges": args.climb_skip_edges,
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
    n_commands = write_shell_script(all_rows, args.sh, args.out_dir, offsets, args.combined_h5,
                                    args.speed)

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
