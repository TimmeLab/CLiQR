"""
False positive detection rate quantification for CLiQR.

Compares CVAT-annotated licking bouts against CLiQR-detected lick times to
compute false positive rates per session and sensor.

Completely standalone — no imports from data_analysis.py.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from utils.state import SERIAL_NUMBER_SENSOR_MAP

logger = logging.getLogger(__name__)

# ── Sensor → board mapping ──
# Derived from the single source of truth in utils.state so it can never drift
# out of sync with the rack wiring the recorder actually writes to HDF5 (the
# previous hardcoded copy still encoded the retired 4-board layout and produced
# KeyErrors on 8-board recordings). The HDF5 groups are named "board_{serial}".
SENSOR_BOARD_MAP = {
    sensor: f"board_{serial}"
    for serial, sensors in SERIAL_NUMBER_SENSOR_MAP.items()
    for sensor in sensors
}

LABEL_BOUT_START   = 'Licking Bout Start'
LABEL_BOUT_END     = 'Licking Bout End'
LABEL_INCONCL_START = 'Inconclusive Region Start'
LABEL_INCONCL_END   = 'Inconclusive Region End'
LABEL_SIPPER_IN    = 'Sipper Inserted'
LABEL_SIPPER_OUT   = 'Sipper Removed'


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Parse CVAT XML
# ═══════════════════════════════════════════════════════════════════════════════

def parse_job_annotations(xml_path):
    """
    Parse a CVAT per-job XML export.

    Per-job exports use local 0-indexed frame IDs, so global_frame_offset is
    always 0 and no task_id lookup is required.

    Returns
    -------
    dict {frame_id (int) → [label_str, ...]}
        Only frames with at least one tag are included.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = {}
    for img in root.findall('image'):
        tags = img.findall('tag')
        if not tags:
            continue
        fid = int(img.get('id'))
        annotations[fid] = [t.get('label') for t in tags]

    logger.info(f"Parsed {Path(xml_path).name}: {len(annotations)} labeled frames")
    return annotations


def parse_annotations(xml_path):
    """
    Parse CVAT project-level XML export.

    Returns
    -------
    task_meta : dict
        {task_id (int) → {'name', 'source', 'size', 'start_frame', 'stop_frame'}}
    annotations : dict
        {task_id (int) → {frame_id (int) → [label_str, ...]}}
        Only frames with at least one tag are included.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    task_meta = {}
    tasks_elem = root.find('meta/project/tasks')
    if tasks_elem is not None:
        for task in tasks_elem.findall('task'):
            tid = int(task.find('id').text)
            source_elem = task.find('source')
            task_meta[tid] = {
                'name':        task.find('name').text,
                'source':      source_elem.text if source_elem is not None else None,
                'size':        int(task.find('size').text),
                'start_frame': int(task.find('start_frame').text),
                'stop_frame':  int(task.find('stop_frame').text),
            }

    # CVAT project exports use global frame IDs: task N's frames start at the
    # cumulative sum of all preceding task sizes. Compute per-task offset so
    # callers can convert global → local frame index for the per-task txt file.
    cumulative = 0
    for tid in sorted(task_meta.keys()):
        task_meta[tid]['global_frame_offset'] = cumulative
        cumulative += task_meta[tid]['size']

    annotations = {}
    for img in root.findall('image'):
        tags = img.findall('tag')
        if not tags:
            continue
        tid = int(img.get('task_id'))
        fid = int(img.get('id'))
        annotations.setdefault(tid, {})[fid] = [t.get('label') for t in tags]

    logger.info(
        f"Parsed {len(task_meta)} tasks, "
        f"{sum(len(v) for v in annotations.values())} labeled frames total."
    )
    return task_meta, annotations


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Frame IDs → video-relative timestamps
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame_offsets(txt_path):
    """
    Load Pi camera .txt timestamp file.

    Format written by pi/camera_backend.py:_on_frame — one integer per line,
    no header, each the frame's absolute SensorTimestamp in nanoseconds (a
    monotonic sensor clock, NOT Unix), 0-indexed by capture order. The absolute
    epoch cancels downstream: alignment_from_bookmark() subtracts the bookmark's
    video_pts, which is the same clock, so only elapsed time survives.

    Returns numpy int64 array of per-frame absolute-ns timestamps.
    """
    offsets = np.loadtxt(txt_path, dtype=np.int64)
    return offsets


def frames_to_relative_seconds(frame_ids, frame_offsets_ns, global_frame_offset=0):
    """
    Convert frame IDs to seconds since video start.

    Parameters
    ----------
    frame_ids : iterable of int — global frame IDs from CVAT XML
    frame_offsets_ns : numpy array returned by load_frame_offsets()
    global_frame_offset : int — subtract from each frame ID to get per-task index.
        Use task_meta[task_id]['global_frame_offset'] from parse_annotations().

    Returns
    -------
    dict {global_frame_id → relative_s (float)}
    """
    result = {}
    n = len(frame_offsets_ns)
    for fid in frame_ids:
        local_idx = fid - global_frame_offset
        if 0 <= local_idx < n:
            result[fid] = float(frame_offsets_ns[local_idx]) / 1e9
        else:
            logger.warning(
                f"Frame {fid} (local {local_idx}) out of range "
                f"(txt has {n} offsets) — skipped"
            )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Build ground truth intervals (in video-relative seconds)
# ═══════════════════════════════════════════════════════════════════════════════

def _pair_events(starts, ends, start_label, end_label):
    """Pair sorted start/end timestamps into (start_s, end_s) intervals."""
    starts = sorted(starts)
    ends   = sorted(ends)
    intervals = []
    ei = 0
    for s in starts:
        while ei < len(ends) and ends[ei] <= s:
            ei += 1
        if ei < len(ends):
            intervals.append((s, ends[ei]))
            ei += 1
        else:
            logger.warning(
                f"Unpaired {start_label} at {s:.2f}s — no matching {end_label} after it"
            )
    return intervals


def build_ground_truth(task_annotations, frame_times):
    """
    Build ground truth event structure from annotated frames.

    Parameters
    ----------
    task_annotations : dict {frame_id → [label_str, ...]}
    frame_times : dict {frame_id → relative_s}  (from frames_to_relative_seconds)

    Returns
    -------
    dict with keys:
        'licking_bouts'  : list of (start_s, end_s) in video-relative seconds
        'inconclusive'   : list of (start_s, end_s)
        'sipper_in_s'    : float or None
        'sipper_out_s'   : float or None
    """
    # Collect all events by label
    by_label = {}
    for fid, labels in task_annotations.items():
        rel_s = frame_times.get(fid)
        if rel_s is None:
            continue
        for lbl in labels:
            by_label.setdefault(lbl, []).append(rel_s)

    bouts = _pair_events(
        by_label.get(LABEL_BOUT_START, []),
        by_label.get(LABEL_BOUT_END, []),
        LABEL_BOUT_START, LABEL_BOUT_END,
    )
    inconclusive = _pair_events(
        by_label.get(LABEL_INCONCL_START, []),
        by_label.get(LABEL_INCONCL_END, []),
        LABEL_INCONCL_START, LABEL_INCONCL_END,
    )

    sipper_in_times  = sorted(by_label.get(LABEL_SIPPER_IN,  []))
    sipper_out_times = sorted(by_label.get(LABEL_SIPPER_OUT, []))

    return {
        'licking_bouts': bouts,
        'inconclusive':  inconclusive,
        'sipper_in_s':   sipper_in_times[0]   if sipper_in_times  else None,
        'sipper_out_s':  sipper_out_times[-1]  if sipper_out_times else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Load raw HDF5 + detect sipper step
# ═══════════════════════════════════════════════════════════════════════════════

def load_sensor_data(raw_h5_path, sensor_num):
    """
    Load cap_data and time_data for one sensor from a raw HDF5 file.

    Returns
    -------
    dict with keys:
        'cap_data'   : float64 numpy array
        'time_data'  : float64 numpy array (absolute Unix seconds)
        'start_time' : float or None
        'stop_time'  : float or None
        'has_recording' : bool (True if start_time present)
    """
    board = SENSOR_BOARD_MAP.get(sensor_num)
    if board is None:
        raise ValueError(f"Unknown sensor number: {sensor_num}")
    path = f"{board}/sensor_{sensor_num}"

    with h5py.File(raw_h5_path, 'r') as f:
        if path not in f:
            raise KeyError(f"Sensor path '{path}' not found in {raw_h5_path}")
        grp = f[path]
        cap_data  = grp['cap_data'][:].astype(np.float64)
        time_data = grp['time_data'][:]
        start_time = float(grp['start_time'][()]) if 'start_time' in grp else None
        stop_time  = float(grp['stop_time'][()])  if 'stop_time'  in grp else None

    return {
        'cap_data':      cap_data,
        'time_data':     time_data,
        'start_time':    start_time,
        'stop_time':     stop_time,
        'has_recording': start_time is not None,
    }


def detect_sipper_step(cap_data, time_data, t_min, t_max, direction='down',
                       smooth_s=2.0, fs=56.0):
    """
    Detect the sipper handling event (insertion or removal) in a cap_data window.

    Physical behavior: both insertion and removal involve a large, brief dip in
    capacitance while personnel handle the sipper, followed by stabilisation at
    a new level. The bottom of that dip is the most reliably detectable feature
    and is used as the anchor.

      - Insertion (direction='down'): baseline ~220 pF, dips during handling,
        settles lower (~180 pF). Search window: pre-start (t[0] → start_time).
      - Removal  (direction='up'):   baseline ~180 pF, dips during handling,
        rises back to ~200+ pF. Search window: post-stop (stop_time → t[-1]).

    The 'direction' label is carried through for logging only; detection always
    finds the minimum of the smoothed trace (deepest point of the handling dip).
    Smoothing (default 2 s rolling median) suppresses brief lick transients while
    preserving the sustained handling dip.

    Parameters
    ----------
    cap_data, time_data : arrays from load_sensor_data()
    t_min, t_max : restrict search to this absolute time window
    direction : 'down' for insertion, 'up' for removal (affects log message only)
    smooth_s : rolling-median window in seconds
    fs : nominal sampling rate

    Returns
    -------
    (step_abs_time, dip_depth) or (None, None) if detection fails.
    dip_depth = (stable level before dip) − (dip minimum); always positive.
    """
    mask = (time_data >= t_min) & (time_data <= t_max)
    n = int(mask.sum())
    if n < 10:
        logger.warning(
            f"Search window {t_min:.0f}–{t_max:.0f} has <10 samples — detection skipped"
        )
        return None, None

    t_win = time_data[mask]
    c_win = cap_data[mask]

    # Rolling median suppresses lick transients (~0.1 s) while preserving the
    # sustained handling dip (several seconds wide)
    win  = max(3, int(smooth_s * fs))
    half = win // 2
    # Centered rolling median, C-level (was an O(n*win) Python comprehension).
    # window=2*half+1 with min_periods=1 reproduces the old clamped-edge windows.
    smoothed = (
        pd.Series(c_win)
        .rolling(window=2 * half + 1, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    # Minimum of smoothed trace = deepest point of handling dip
    dip_idx  = int(np.argmin(smoothed))
    dip_time = float(t_win[dip_idx])

    # Score = how far the dip drops from the preceding stable level
    pre_samples = max(1, dip_idx - win)
    pre_level   = float(np.median(c_win[:pre_samples])) if pre_samples > 0 else float(smoothed[0])
    dip_depth   = pre_level - float(smoothed[dip_idx])  # positive

    event = 'insertion' if direction == 'down' else 'removal'
    logger.info(
        f"Sipper {event} detected at {dip_time:.3f} "
        f"(dip_depth {dip_depth:.1f}) in window [{t_min:.0f}, {t_max:.0f}]"
    )
    return dip_time, dip_depth


def alignment_from_bookmark(start_time_abs, video_pts,
                            host_before=None, host_after=None):
    """Build an alignment from a CLiQR video bookmark (frame PTS at the Start click).

    The recording GUI records, at the sensor Start click, the Unix start_time and
    the concurrent video frame's PTS. The video's Unix start (its PTS=0 instant)
    is that frame's true host time minus its PTS.

    Bookmark-latency correction: the bookmarked frame was NOT captured at
    start_time. Start_time is stamped on the host, then the bookmark is a wireless
    round-trip to the Pi, which returns whatever frame it had captured by the time
    it handled the request — ~L seconds after start_time (L = one-way latency).
    Its true host time is ~midpoint(host_before, host_after), so:
        video_start_unix_s = (start_time_abs + L) - video_pts
    with L from the shared trimcrop.bookmark_latency formula (0.0 when the host
    bracket wasn't recorded — older recordings). Without this correction the video
    panel leads the trace by L (~2.5 s on the reference wireless link); see
    docs/video-sync-alignment-bugs.md.

    NOTE: video_pts is stored in the SensorTimestamp clock (seconds), the SAME
    clock as load_frame_offsets(). Do NOT "fix" this to a relative-seconds pts —
    the large absolute epoch is intentional and cancels when video_relative_to_abs()
    adds the frame's absolute offset back (abs = start_time_abs + (offset/1e9 -
    video_pts)).

    Drift correction is unavailable (only one anchor), so drift_corrected=False.
    """
    from video.trimcrop import bookmark_latency
    latency = bookmark_latency(host_before, host_after, start_time_abs)
    video_start = (float(start_time_abs) + latency) - float(video_pts)
    return {
        'video_start_unix_s':    video_start,
        'sipper_in_hdf5_abs_s':  float(start_time_abs),
        'sipper_out_hdf5_abs_s': None,
        'drift_s':               None,
        'drift_corrected':       False,
        'step_magnitude':        None,
        'removal_magnitude':     None,
        'bookmark_latency_s':    latency,
        'method':                'bookmark',
    }


def establish_alignment(sensor_data, sipper_in_video_s, sipper_out_video_s=None,
                        post_stop_grace_s=5.0):
    """
    Compute video_start_unix_s using sipper insertion as the primary anchor.
    Optionally computes drift using sipper removal as a second anchor.

    Parameters
    ----------
    sensor_data : dict from load_sensor_data()
    sipper_in_video_s : float — sipper insertion time in video-relative seconds
    sipper_out_video_s : float or None — sipper removal in video-relative seconds
    post_stop_grace_s : float — skip this many seconds at the start of the post-stop
        window before searching for sipper removal. Avoids boundary artifacts
        (e.g. a transient drop right after the recording stop) being mistaken for
        the removal event. Default 5.0 s.

    Returns
    -------
    dict with keys:
        'video_start_unix_s'    : float
        'sipper_in_hdf5_abs_s'  : float
        'sipper_out_hdf5_abs_s' : float or None
        'drift_s'               : float or None (+ = video clock ran fast)
        'drift_corrected'       : bool
        'step_magnitude'        : float
    """
    if not sensor_data['has_recording']:
        raise ValueError("Sensor has no start_time — it was empty during this session")

    t = sensor_data['time_data']
    c = sensor_data['cap_data']
    start_time = sensor_data['start_time']
    stop_time  = sensor_data['stop_time']

    # Primary anchor: sipper insertion in the pre-start window.
    # Sipper insertion DECREASES capacitance (high baseline → lower elevated).
    sipper_in_hdf5, step_mag = detect_sipper_step(
        c, t, t_min=t[0], t_max=start_time, direction='down'
    )
    if sipper_in_hdf5 is None:
        raise RuntimeError(
            "Sipper insertion step not detected in pre-start window. "
            "Check the raw capacitance trace manually."
        )

    video_start = sipper_in_hdf5 - sipper_in_video_s

    result = {
        'video_start_unix_s':    video_start,
        'sipper_in_hdf5_abs_s':  sipper_in_hdf5,
        'sipper_out_hdf5_abs_s': None,
        'drift_s':               None,
        'drift_corrected':       False,
        'step_magnitude':        step_mag,
        'removal_magnitude':     None,
    }

    # Optional: drift correction via sipper removal anchor.
    # Removal happens near stop_time, most likely a few seconds after.
    # Skip the first post_stop_grace_s seconds to avoid boundary artifacts.
    # Only accept detection if magnitude >= 25% of insertion magnitude.
    if sipper_out_video_s is not None and stop_time is not None:
        t_removal_search_start = stop_time + post_stop_grace_s
        sipper_out_hdf5, out_mag = detect_sipper_step(
            c, t, t_min=t_removal_search_start, t_max=t[-1], direction='up'
        )
        min_removal_mag = abs(step_mag) * 0.25
        if sipper_out_hdf5 is not None and abs(out_mag) >= min_removal_mag:
            video_start_v2 = sipper_out_hdf5 - sipper_out_video_s
            drift = video_start_v2 - video_start
            result['sipper_out_hdf5_abs_s'] = sipper_out_hdf5
            result['removal_magnitude']     = out_mag
            result['drift_s'] = drift
            if abs(drift) > 5.0:
                logger.warning(f"Large clock drift: {drift:.2f}s — flag for manual review")
            if abs(drift) > 1.0:
                result['drift_corrected'] = True
                logger.info(f"Drift correction enabled: {drift:.3f}s over session")
        else:
            if out_mag is not None:
                logger.info(
                    f"Sipper removal detected (mag={out_mag:.1f}) below threshold "
                    f"({min_removal_mag:.1f}) — drift correction skipped"
                )
            else:
                logger.info("Sipper removal step not found in post-stop window — drift correction skipped")

    return result


def video_relative_to_abs(video_relative_s, alignment, sipper_in_video_s=None):
    """
    Convert video-relative timestamps to absolute Unix seconds,
    applying linear drift correction if available.

    Parameters
    ----------
    video_relative_s : float or array — seconds since video start
    alignment : dict from establish_alignment()
    sipper_in_video_s : float — needed only when drift_corrected=True

    Returns
    -------
    float or array of absolute Unix timestamps
    """
    video_start = alignment['video_start_unix_s']

    if not alignment['drift_corrected'] or alignment['drift_s'] is None:
        return video_start + np.asarray(video_relative_s)

    drift = alignment['drift_s']
    sipper_out_abs = alignment['sipper_out_hdf5_abs_s']
    sipper_out_video = sipper_out_abs - (video_start + drift)
    session_dur = sipper_out_video - sipper_in_video_s

    if session_dur <= 0:
        return video_start + np.asarray(video_relative_s)

    frac = np.clip(
        (np.asarray(video_relative_s) - sipper_in_video_s) / session_dur, 0, 1
    )
    return video_start + np.asarray(video_relative_s) + drift * frac


def intervals_to_abs(intervals_rel_s, alignment, sipper_in_video_s=None):
    """
    Convert list of (start_s, end_s) video-relative intervals to absolute Unix seconds.
    """
    return [
        (
            float(video_relative_to_abs(s, alignment, sipper_in_video_s)),
            float(video_relative_to_abs(e, alignment, sipper_in_video_s)),
        )
        for s, e in intervals_rel_s
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4b — Load filtered HDF5 lick times
# ═══════════════════════════════════════════════════════════════════════════════

def load_lick_times_abs(filtered_h5_path, raw_h5_path, animal_id, sensor_num):
    """
    Load detected lick times in absolute Unix seconds.

    lick_times in filtered HDF5 are relative to the start of the analysis window
    (the trimmed recording, not time_data[0] of the raw file).
    used_start_idx points into the raw time_data array to recover the absolute start.

    Parameters
    ----------
    filtered_h5_path : path to filtered HDF5
    raw_h5_path : path to corresponding raw HDF5
    animal_id : str, e.g. 'ACG-26-3-1'
    sensor_num : int

    Returns
    -------
    numpy float64 array of absolute Unix lick times, or None if animal not found
    """
    with h5py.File(filtered_h5_path, 'r') as ff:
        if animal_id not in ff:
            logger.warning(f"{animal_id} not in {filtered_h5_path}")
            return None
        grp = ff[animal_id]
        lick_times_rel  = grp['lick_times'][:]
        used_start_idx  = int(grp['used_start_idx'][()])

    board = SENSOR_BOARD_MAP[sensor_num]
    with h5py.File(raw_h5_path, 'r') as rf:
        raw_time_data = rf[f"{board}/sensor_{sensor_num}/time_data"][:]

    abs_start = raw_time_data[used_start_idx]
    return abs_start + lick_times_rel


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6 — Classify licks
# ═══════════════════════════════════════════════════════════════════════════════

def classify_licks(lick_times_abs, bouts_abs, inconclusive_abs):
    """
    Classify each detected lick as TP, FP, or excluded (inconclusive).

    Parameters
    ----------
    lick_times_abs : numpy array of absolute Unix lick times
    bouts_abs : list of (start, end) absolute Unix intervals (ground truth licking bouts)
    inconclusive_abs : list of (start, end) absolute Unix intervals

    Returns
    -------
    dict with:
        'tp', 'fp', 'excluded' : arrays of lick times per class
        'tp_mask', 'fp_mask', 'excluded_mask' : boolean arrays
        'n_tp', 'n_fp', 'n_excluded', 'n_total' : int counts
        'fp_rate' : float = n_fp / (n_tp + n_fp)
    """
    n = len(lick_times_abs)
    excl_mask = np.zeros(n, dtype=bool)
    tp_mask   = np.zeros(n, dtype=bool)

    for s, e in inconclusive_abs:
        excl_mask |= (lick_times_abs >= s) & (lick_times_abs <= e)

    available = ~excl_mask
    for s, e in bouts_abs:
        tp_mask |= available & (lick_times_abs >= s) & (lick_times_abs <= e)

    fp_mask = available & ~tp_mask

    n_tp   = int(tp_mask.sum())
    n_fp   = int(fp_mask.sum())
    n_excl = int(excl_mask.sum())
    denom  = n_tp + n_fp

    return {
        'tp':            lick_times_abs[tp_mask],
        'fp':            lick_times_abs[fp_mask],
        'excluded':      lick_times_abs[excl_mask],
        'tp_mask':       tp_mask,
        'fp_mask':       fp_mask,
        'excluded_mask': excl_mask,
        'n_tp':          n_tp,
        'n_fp':          n_fp,
        'n_excluded':    n_excl,
        'n_total':       n,
        'fp_rate':       n_fp / denom if denom > 0 else float('nan'),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7 — Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_session(
    session_label,
    sensor_data,
    alignment,
    ground_truth_abs,
    classification,
    show_raw=True,
    width=1400,
    height=380,
):
    """
    Plot capacitance trace with bout annotations and lick classifications.

    Returns an interactive Bokeh figure (zoom/pan via toolbar; click legend to hide).

    X-axis: minutes since first HDF5 sample.
    Vertical lines mark detected sipper insertion/removal and start/stop_time boundaries.

    Parameters
    ----------
    session_label : str — used as plot title
    sensor_data : dict from load_sensor_data()
    alignment : dict from establish_alignment()
    ground_truth_abs : dict with 'licking_bouts' and 'inconclusive' in absolute time
    classification : dict from classify_licks(), or None
    show_raw : bool — if True, plot full raw trace; if False, plot analysis window only
    width, height : figure dimensions in pixels

    Returns
    -------
    bokeh.plotting.figure
    """
    from bokeh.plotting import figure
    from bokeh.models import BoxAnnotation, Span, Label, HoverTool

    t_full = sensor_data['time_data']
    c_full = sensor_data['cap_data']

    def to_idx(abs_time):
        """Absolute time → absolute sample index in t_full (float-preserving for arrays)."""
        arr = np.asarray(abs_time, dtype=float)
        idx = np.searchsorted(t_full, arr)
        return np.clip(idx, 0, len(t_full) - 1)

    t_plot = t_full
    c_plot = c_full
    x_plot = np.arange(len(t_full))
    if sensor_data['has_recording'] and not show_raw:
        stop = sensor_data['stop_time'] or t_full[-1]
        mask = (t_full >= sensor_data['start_time']) & (t_full <= stop)
        t_plot = t_full[mask]
        c_plot = c_full[mask]
        x_plot = np.where(mask)[0]

    fp_rate = classification['fp_rate'] if classification else float('nan')
    n_tp    = classification['n_tp']    if classification else 0
    n_fp    = classification['n_fp']    if classification else 0
    if classification and not np.isnan(fp_rate):
        title = f"{session_label}   |   TP={n_tp}, FP={n_fp}, FP rate={fp_rate:.1%}"
    else:
        title = session_label

    p = figure(
        width=width, height=height,
        title=title,
        x_axis_label='Sample index',
        y_axis_label='Capacitance (pF)',
        tools='pan,box_zoom,wheel_zoom,reset,save',
        active_drag='box_zoom',
    )

    # Capacitance trace with hover tooltip
    r_cap = p.line(
        x_plot.tolist(), c_plot.tolist(),
        color='#555555', line_width=0.6, alpha=0.7,
    )
    p.add_tools(HoverTool(
        renderers=[r_cap],
        tooltips=[('Index', '@x{0}'), ('Cap', '@y{0.0} pF')],
        mode='vline',
    ))

    # Bout and inconclusive shading via BoxAnnotation
    for s, e in ground_truth_abs.get('licking_bouts', []):
        p.add_layout(BoxAnnotation(
            left=float(to_idx(s)), right=float(to_idx(e)),
            fill_color='#2ecc71', fill_alpha=0.6, line_width=0,
        ))
    for s, e in ground_truth_abs.get('inconclusive', []):
        p.add_layout(BoxAnnotation(
            left=float(to_idx(s)), right=float(to_idx(e)),
            fill_color='#aaaaaa', fill_alpha=0.4, line_width=0,
        ))

    # Dummy off-screen rects so BoxAnnotation colors appear in legend
    if ground_truth_abs.get('licking_bouts'):
        p.rect(x=[0], y=[-1e9], width=[0.001], height=[0.001],
               fill_color='#2ecc71', fill_alpha=0.5, line_width=0,
               legend_label='Licking bout (GT)')
    if ground_truth_abs.get('inconclusive'):
        p.rect(x=[0], y=[-1e9], width=[0.001], height=[0.001],
               fill_color='#aaaaaa', fill_alpha=0.5, line_width=0,
               legend_label='Inconclusive')

    # Lick markers — placed at the interpolated cap value at each lick time
    c_range = float(c_full.max() - c_full.min()) if len(c_full) > 0 else 1.0
    c_min   = float(c_full.min())

    if classification is not None:
        tp_t = classification['tp']
        fp_t = classification['fp']
        ex_t = classification['excluded']
        if len(tp_t):
            y_tp = np.interp(tp_t, t_full, c_full)
            p.scatter(to_idx(tp_t).tolist(), y_tp.tolist(),
                      color='#2980b9', size=6, marker='circle',
                      line_color='white', line_width=1.5,
                      legend_label='TP lick')
        if len(fp_t):
            y_fp = np.interp(fp_t, t_full, c_full)
            p.scatter(to_idx(fp_t).tolist(), y_fp.tolist(),
                      color='#e74c3c', size=8, marker='x',
                      line_width=3,
                      legend_label='FP lick')
        if len(ex_t):
            y_ex = np.interp(ex_t, t_full, c_full)
            p.scatter(to_idx(ex_t).tolist(), y_ex.tolist(),
                      color='#888888', size=10, marker='dash',
                      line_width=2)

    # Vertical event lines with text labels
    def _vspan_label(abs_time, color, line_dash, label_text, y_frac=0.95):
        xi = float(to_idx(abs_time))
        p.add_layout(Span(location=xi, dimension='height',
                          line_color=color, line_width=1.5, line_dash=line_dash))
        p.add_layout(Label(x=xi, y=c_min + c_range * y_frac,
                           text=label_text, text_color=color,
                           text_font_size='8pt', x_offset=4))

    if sensor_data['has_recording']:
        _vspan_label(sensor_data['start_time'], '#2471a3', 'dashed', 'start', y_frac=0.52)
        if sensor_data['stop_time'] is not None:
            _vspan_label(sensor_data['stop_time'], '#2471a3', 'dashed', 'stop', y_frac=0.52)

    si_hdf5 = alignment.get('sipper_in_hdf5_abs_s')
    if si_hdf5 is not None:
        _vspan_label(si_hdf5, '#8e44ad', 'dotted', 'Sipper In')

    so_hdf5 = alignment.get('sipper_out_hdf5_abs_s')
    if so_hdf5 is not None:
        _vspan_label(so_hdf5, '#e67e22', 'dotted', 'Sipper Out')

    if p.legend:
        p.legend.location = 'top_right'
        p.legend.label_text_font_size = '8pt'
        p.legend.click_policy = 'hide'

    # Pin y_range to actual data — dummy legend rects at y=-1e9 would corrupt auto-range
    from bokeh.models import Range1d
    pad = c_range * 0.05
    p.y_range = Range1d(c_min - pad, c_min + c_range + pad)

    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Step 8 — Summary
# ═══════════════════════════════════════════════════════════════════════════════

def build_results_dataframe(all_results):
    """
    Compile per-session results into a summary DataFrame.

    Parameters
    ----------
    all_results : list of dicts, each containing:
        'session', 'animal_id', 'sensor_num', 'task_id',
        'n_total', 'n_tp', 'n_fp', 'n_excluded', 'fp_rate',
        'drift_s', 'step_magnitude'

    Returns
    -------
    pandas DataFrame
    """
    df = pd.DataFrame(all_results)
    cols = [c for c in [
        'session', 'animal_id', 'sensor_num', 'task_id',
        'n_total', 'n_tp', 'n_fp', 'n_excluded', 'fp_rate',
        'drift_s', 'alignment_offset', 'step_magnitude',
    ] if c in df.columns]
    return df[cols]
