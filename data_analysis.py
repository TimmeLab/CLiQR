# File Name: data_analysis.py
# Author: Christopher Parker
# Created: Thu May 08, 2025 | 11:48P EDT
# Last Modified: Wed May 21, 2025 | 12:12P EDT

"""Contains the data analysis functions to be called when the user stops
a recording."""


import re
import h5py
import numpy as np
import scipy.signal as scs

from utils.state import SERIAL_NUMBER_SENSOR_MAP


# ---------------------------------------------------------------------------
# Sensor number -> HDF5 board group name
# ---------------------------------------------------------------------------
# The recorder stores each board's data under a group named "board_{serial}"
# (e.g. "board_FT232H0"). To find a given sensor's data we need the reverse of
# SERIAL_NUMBER_SENSOR_MAP (which maps serial -> list of sensor numbers).
#
# We DERIVE this reverse map from that single source of truth in utils.state
# instead of hardcoding it. A previously hardcoded copy here silently encoded
# only the retired 4-board rack and produced KeyErrors on 8-board recordings;
# deriving it means the analysis follows whatever rack the recorder actually
# wrote (selected by CLIQR_RACK / RACK_DESIGN).
SENSOR_BOARD_MAP = {
    sensor_number: f"board_{serial_number}"
    for serial_number, sensor_numbers in SERIAL_NUMBER_SENSOR_MAP.items()
    for sensor_number in sensor_numbers
}


# Control cages have sippers (so volume is recorded) but no animal, hence no
# weight. IDs are "Control1", "Control2", ... (1-indexed) in the layout file.
_CONTROL_RE = re.compile(r'^Control\d+$')

def is_control(animal_id):
    """True if animal_id is a control cage (no animal, no weight recorded).

    Accepts 2-bottle per-fluid keys ("Control1/fluid2") as well as bare IDs.
    """
    return bool(_CONTROL_RE.match(str(animal_id).split('/')[0]))


# ---------------------------------------------------------------------------
# 2-bottle choice
# ---------------------------------------------------------------------------
# A 2-bottle recording (file attr two_bottle=True) stores each sensor's two
# sippers in fluid subgroups. Start/stop times and weight stay at the sensor
# level; cap_data, time_data and the volumes live in the fluid subgroups:
#     sensor_<id>/start_time, stop_time, weight
#     sensor_<id>/fluid1/cap_data, time_data, start_vol, stop_vol
#     sensor_<id>/fluid2/...
FLUIDS = (1, 2)


def fluid_group_name(fluid):
    return f"fluid{int(fluid)}"


def is_two_bottle(raw_h5f):
    """True if an open raw HDF5 file was recorded in 2-bottle choice mode."""
    return bool(raw_h5f.attrs.get('two_bottle', False))


_START_TIME_RE = re.compile(r'^start_time(\d+)?$')


def _last_cycle_start_stop(sensor_group, time_data):
    """Return (start_time, stop_time, had_start) for the LAST recording cycle.

    A sensor may have been started and stopped several times in one session.
    Each cycle writes a start_time key: the first is plain "start_time", later
    ones are "start_time1", "start_time2", ... We want the LAST cycle, so
    collect every start_time key and index it by its cycle number (the
    unnumbered first cycle is treated as cycle -1 so it sorts before the
    numbered ones).
    """
    cycle_number_to_key = {}
    for k in sensor_group.keys():
        m = _START_TIME_RE.match(k)
        if m:
            cycle_number = int(m.group(1)) if m.group(1) else -1
            cycle_number_to_key[cycle_number] = k
    if not cycle_number_to_key:
        return time_data[0], time_data[-1], False
    # The most recent cycle is the one with the largest cycle number.
    last_start = cycle_number_to_key[max(cycle_number_to_key)]
    start_time = sensor_group[last_start][()]
    # Try the stop_time corresponding to the start_time above
    stop_key = 'stop' + last_start[5:]
    if stop_key in sensor_group:
        stop_time = sensor_group[stop_key][()]
    else:
        # Stop time wasn't recorded (likely clicked stop all button too soon)
        stop_time = time_data[-1]
    return start_time, stop_time, True


def load_trimmed_sensor(sensor_group, label, recording_length, start_stop_override=None,
                        fluid=None):
    """Read ONE sensor's group from a raw HDF5 file and trim it for analysis.

    The trace is cut to the last start/stop cycle, rebased so time starts at 0,
    and then cut (from the end) to `recording_length` seconds. Only the samples
    that survive the trim are read from disk for cap_data, so peak memory is a
    single sensor's time axis plus its trimmed trace -- not the whole file.
    (A 24 h, 24-sensor recording is several GB; loading it all at once is what
    made long runs crawl.)

    Returns a plain dict with the same keys filter_data has always produced:
    every dataset in the sensor group (scalars read eagerly), plus cap_data,
    time_data, fs, consumed_vol (when volumes exist), used_start_idx and
    used_stop_idx.

    `start_stop_override`, if given, is a (start_time, stop_time) pair in Unix
    seconds (from a *_time_fix.xlsx file) that replaces the recorded times.

    `fluid` (1 or 2) selects one bottle of a 2-bottle sensor: the trace and
    volumes come from that fluid's subgroup, start/stop times and weight from
    the sensor group (both bottles share them).
    """
    data_group = sensor_group if fluid is None else sensor_group[fluid_group_name(fluid)]
    time_data = data_group['time_data'][()]
    if start_stop_override is not None:
        start_time, stop_time = start_stop_override
        had_start = True
        print(f"Applying time fix for {label}: start={start_time}, stop={stop_time}")
    else:
        start_time, stop_time, had_start = _last_cycle_start_stop(sensor_group, time_data)
    if not had_start:
        print(f"Warning: no start/stop times recorded for {label}, create a time fix file or we will default to the entire trace.")
    start_idx = int(np.argmin(np.abs(time_data - start_time)))
    stop_idx = int(np.argmin(np.abs(time_data - stop_time)))

    sensor_data = {}
    # Sensor-level scalars first, then (2-bottle) the fluid's own volumes.
    for grp in ((sensor_group,) if fluid is None else (sensor_group, data_group)):
        for k, v in grp.items():
            if k in ('time_data', 'cap_data') or not isinstance(v, h5py.Dataset):
                continue
            sensor_data[k] = v[()]

    t = time_data[start_idx:stop_idx+1] - time_data[start_idx]
    del time_data
    end = stop_idx + 1
    fs = (stop_idx - start_idx) / (t[-1] - t[0])

    # Check if we still have at least recording_length
    remaining_time = t[-1] - t[0]
    if remaining_time < recording_length:
        print(f"Warning: {label} was left with only {remaining_time} seconds of recording after trimming to recorded start/stop times. Consider a time fix file.")
    else:
        # Now we trim any excess to the desired recording length (from the end)
        revised_stop_idx = int(np.argmin(np.abs(t - recording_length)))
        t = t[:revised_stop_idx]
        end = start_idx + revised_stop_idx

    sensor_data['time_data'] = t
    sensor_data['cap_data'] = data_group['cap_data'][start_idx:end]
    sensor_data['fs'] = fs

    if 'stop_vol' in sensor_data:
        sensor_data['consumed_vol'] = sensor_data['start_vol'] - sensor_data['stop_vol']

    # Record which start/stop indices we actually used in the analysis
    sensor_data['used_start_idx'] = start_idx
    sensor_data['used_stop_idx'] = stop_idx
    return sensor_data


def parse_time_fix(time_fix):
    """Turn a time-fix DataFrame into {sensor_number: (start_time, stop_time)}.

    Columns are taken by position (Sensor, New Start Time, New End Time) so
    small header spelling differences in hand-made spreadsheets don't matter.
    Rows with a missing value are ignored.
    """
    if time_fix is None or len(time_fix) == 0:
        return {}
    if time_fix.shape[1] < 3:
        raise ValueError("time fix file needs 3 columns: Sensor, New Start Time, New End Time")
    overrides = {}
    for sensor, start, stop in time_fix.iloc[:, :3].itertuples(index=False):
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (sensor, start, stop)):
            continue
        overrides[int(sensor)] = (float(start), float(stop))
    return overrides


def filter_data(raw_h5f, filtered_h5f, sensor_animal_map, logfile, time_fix=None, algorithm='basic_threshold', recording_length=2*60*60, progress=None, require_weight=True):
    """This organizes the data by animal and handles some common issues relating to
    recording start/stop times, volumes, and weights. Current algorithm choices:
        - basic_threshold
        - hilbert (Hilbert envelope threshold)
        - ml

    Sensors are processed ONE AT A TIME (load -> trim -> detect -> save -> free)
    so memory stays bounded for long (e.g. 24 h) recordings. Only sensors listed
    in `sensor_animal_map` are read from disk.

    `progress`, if given, is called as progress(done, total, animal_id) after
    each animal finishes (used by the analysis GUI for a progress bar).

    `time_fix` is an optional DataFrame read from `<raw>_time_fix.xlsx` with
    columns [Sensor, New Start Time, New End Time] (Unix epoch seconds). Listed
    sensors use those times instead of the recorded start/stop times.

    2-bottle recordings (file attr two_bottle=True) are detected automatically:
    each bottle is analysed as its own channel and saved under
    "<animal>/fluid1" and "<animal>/fluid2" in the filtered file (a time fix
    applies to both bottles of a sensor).

    `require_weight=False` keeps animals whose weight was never recorded (their
    weight is saved as NaN) instead of dropping them as missing data.
    """
    algorithms = {
        'basic_threshold': basic_algorithm,
        'hilbert': hilbert_algorithm,
        'ml': ml_algorithm,
    }
    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm {algorithm!r}; expected one of {sorted(algorithms)}")

    # Reorganize by animal ID, agnostic wrt any board/sensor numbering
    jobs = []
    for idx, row in sensor_animal_map.iterrows():
        sensor = row.name
        animal = row.item()
        # Look up which board's HDF5 group holds this sensor (derived from the
        # rack wiring in utils.state, so it stays correct across rack designs).
        board_id = SENSOR_BOARD_MAP.get(sensor)
        if board_id is None:
            print(f"Sensor {sensor} is not part of the current rack layout; skipping.")
            continue
        if board_id not in raw_h5f or f"sensor_{sensor}" not in raw_h5f[board_id]:
            print(f"Missing key in data_dict: {board_id}/sensor_{sensor}")
            continue
        jobs.append((animal, board_id, sensor))

    two_bottle = is_two_bottle(raw_h5f)
    if two_bottle:
        # One detection job per bottle; the output key nests the fluid under
        # the animal so save_filtered_data writes "<animal>/fluid<n>".
        fluid_jobs = []
        for animal, board_id, sensor in jobs:
            sensor_grp = raw_h5f[board_id][f"sensor_{sensor}"]
            for fluid in FLUIDS:
                if fluid_group_name(fluid) not in sensor_grp:
                    print(f"Missing key in data_dict: {board_id}/sensor_{sensor}/{fluid_group_name(fluid)}")
                    continue
                fluid_jobs.append((f"{animal}/{fluid_group_name(fluid)}", board_id, sensor, fluid))
        jobs = fluid_jobs
    else:
        jobs = [(animal, board_id, sensor, None) for animal, board_id, sensor in jobs]

    overrides = parse_time_fix(time_fix)

    extra = {}
    if algorithm == 'ml':
        # Load the networks once for the whole file, not once per animal.
        extra['nets'] = _load_ml_nets(None)

    for n_done, (animal, board_id, sensor, fluid) in enumerate(jobs, start=1):
        label = f"{board_id} sensor_{sensor}" + (f" {fluid_group_name(fluid)}" if fluid else "")
        data = load_trimmed_sensor(raw_h5f[board_id][f"sensor_{sensor}"],
                                   label, recording_length,
                                   start_stop_override=overrides.get(int(sensor)),
                                   fluid=fluid)
        if not require_weight and 'weight' not in data and not is_control(animal):
            data['weight'] = np.nan  # kept on purpose; see require_weight
        missing_data = algorithms[algorithm]({animal: data}, filtered_h5f, logfile, **extra)
        del data
        if progress is not None:
            progress(n_done, len(jobs), animal)
        if missing_data:
            return missing_data
    return False


def _record_zero_licks(data, animal, logfile, reason):
    """Mark a channel as having zero detectable licks (instead of dropping it).

    A trace with too few discrete capacitance levels cannot hold a dip deep
    enough to count as a lick, so zero licks is the correct answer for it -- and
    the channel's volume/trace still belong in the output. Silently skipping such
    channels used to make control cages vanish from some sessions.
    """
    data['lick_times'] = np.array([])
    data['lick_indices'] = np.array([], dtype=int)
    data['num_licks'] = 0
    with open(logfile, 'a') as lf:
        lf.write(f"No licks detectable for {animal}: {reason}; recorded 0 licks\n")


def _narrow_int_trace(trace):
    """Return (trace in the narrowest safe integer dtype, True) for integer
    traces, or (trace, False) otherwise."""
    if not np.issubdtype(trace.dtype, np.integer) or trace.size == 0:
        return trace, False
    lo, hi = int(trace.min()), int(trace.max())
    for dt in (np.int16, np.int32):
        info = np.iinfo(dt)
        # leave headroom so ceil(threshold) always fits the comparison
        if info.min < lo and hi < info.max:
            return trace.astype(dt), True
    return trace, True


def _span_min(trace, starts, ends):
    """min(trace[s:e+1]) for each (s, e) pair (all e >= s).

    Lick-sized spans are short (< 1/6 s, ~20 samples), so the minimum is taken
    by stepping an offset across all spans at once: O(spans * longest span)
    instead of a pass over the whole trace.
    """
    if starts.size == 0:
        return np.array([], dtype=trace.dtype)
    lengths = ends - starts + 1
    short = lengths <= 64
    out = np.empty(starts.size, dtype=trace.dtype)
    s, e = starts[short], ends[short]
    if s.size:
        m = trace[s]
        for off in range(1, int(lengths[short].max())):
            m = np.minimum(m, trace[np.minimum(s + off, e)])
        out[short] = m
    for j in np.flatnonzero(~short):  # rare (irregular sampling): do it directly
        out[j] = trace[starts[j]:ends[j] + 1].min()
    return out


def basic_algorithm(data_by_animal, filtered_h5f, logfile):
    """Detect licks by scanning every possible amplitude threshold.

    Scientific idea
    ---------------
    A lick briefly pulls the sipper capacitance DOWN, so each lick is a narrow
    downward dip in the trace. A dip is detected by asking "where does the trace
    drop below some threshold?". The hard part is choosing the threshold. Rather
    than guess one value, this routine tries every candidate threshold and keeps
    the one that yields the most licks that also satisfy the shape rules below.

    Because the MPR121 reports capacitance as discrete integer counts, there are
    only finitely many meaningful thresholds: one sitting halfway between each
    pair of adjacent observed values. Scanning those midpoints covers every
    distinguishable threshold with no wasted work.

    For a candidate threshold, a "peak" (dip) is a run of consecutive samples
    below the threshold. A run is accepted as a real lick only if it is:
      1. short enough in time  (< max_lick_time; a real lick is brief), and
      2. deep enough           (it must also cross a threshold `depth_back`
                                levels lower, the "depth" rule, so shallow noise
                                wiggles are rejected).
    The lick time is taken at the minimum (deepest) sample of the accepted dip.

    The depth rule (2) is why the candidate-threshold loop below starts at index
    `depth_back` rather than 0: for candidate index i_thr the depth comparison
    uses thresholds[i_thr - depth_back], and starting any earlier would make that
    a NEGATIVE index. In numpy a negative index does not error, it silently wraps
    to the HIGH end of the sorted threshold array, which disables the depth rule
    and lets noise wiggles through -- historically this produced tens of
    thousands of false "licks" on quiet/control channels and, downstream, an
    O(N^2) memory blow-up. Keep the loop start tied to `depth_back`.

    Scientific caveat: `depth_back` is an ABSOLUTE count of discrete capacitance
    levels, not a fixed voltage/amplitude. Because different animals/sensors span
    different numbers of unique capacitance values, the same `depth_back` imposes
    a slightly different physical depth requirement per channel. If the sensor
    charge/discharge time (CDT) or gain changes, revisit this value.

    This mirrors the original MATLAB lickDetector logic, reimplemented in numpy.

    Parameters
    ----------
    data_by_animal : dict {animal_id -> data dict with 'cap_data', 'time_data'}
    filtered_h5f : open writable h5py.File to save per-animal results into
    logfile : path to a text log for recording missing weights/volumes

    Returns
    -------
    bool : True if any animal was missing required weight/volume data.
    """
    for (animal, data) in data_by_animal.items():
        trace = data['cap_data']
        times = data['time_data']

        # Longest a single lick may stay below threshold. Mice lick at roughly
        # 6-10 Hz, so one lick lasts well under 1/6 s (~167 ms); anything longer
        # below threshold is a sustained contact or artifact, not a lick.
        max_lick_time = 1. / 6. # in seconds

        peak_info = {}

        # A peak must cross a threshold this many levels lower to count as a
        # real lick (see depth check below). The threshold loop MUST start at
        # depth_back so that i_thr - depth_back is never negative; a negative
        # index would silently wrap to a HIGH threshold, disabling the depth
        # filter and inflating counts (and IndexError on channels with <
        # depth_back levels).
        depth_back = 20

        # Because the capacitance values are discretized, we can focus only
        # on the thresholds between the discrete values
        unique_vals = np.unique(trace)
        # The scan needs more thresholds than depth_back, otherwise the loop
        # below never runs. A channel that flat cannot contain a dip deep enough
        # to be a lick, so it is a ZERO-LICK channel, not a missing one -- we
        # still save it (control cages sit here: no animal means a nearly flat
        # trace, and their evaporation volume has to reach the combined file).
        if len(unique_vals) - 1 > depth_back:
            # Get thresholds between each unique capacitance value
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            n_peaks = np.full((len(thresholds),), np.nan)
            # Accepted (start, end) sample pairs per threshold. The lick sample
            # (argmin inside each dip) is only needed for the WINNING threshold,
            # so it is computed once after the scan instead of for every one.
            good_spans = [None] * len(thresholds)
            empty = np.array([], dtype=int)
            # Speed-up for long recordings: the trace is integer counts, so
            # `trace < thr` is exactly `trace < ceil(thr)`, which can be done in
            # a narrow integer type (4x less memory traffic than int64-vs-float).
            work, as_int = _narrow_int_trace(trace)

            for i_thr in range(depth_back, len(thresholds)):
                thr = thresholds[i_thr]
                below = work < (int(np.ceil(thr)) if as_int else thr)
                # Identify peaks below threshold. 1 marks the time bin just before a
                # peak starts (the last bin that isn't in the peak), -1 marks the time
                # bin just before a peak ends (the last bin in the peak)
                peak_trans = np.diff(below.view(np.int8))
                # Add 1 so we are getting the actual start of the lick, not the time
                # 1 sample before
                peak_starts = np.where(peak_trans == 1)[0] + 1
                peak_ends = np.where(peak_trans == -1)[0]

                # Handle the first and last time point separately
                if trace[0] < thr:
                    peak_starts = np.r_[0, peak_starts]
                if trace[-1] < thr:
                    peak_ends = np.r_[peak_ends, len(trace) - 1]

                # If we still have no peaks, skip
                if peak_starts.size == 0 or peak_ends.size == 0:
                    good_spans[i_thr] = (empty, empty)
                    n_peaks[i_thr] = 0
                    continue
                if peak_starts.size != peak_ends.size:
                    # Align to the whichever has the shortest count
                    m = min(peak_starts.size, peak_ends.size)
                    peak_starts = peak_starts[:m]
                    peak_ends = peak_ends[:m]

                # Remove peaks that are below threshold for too long (set by max_lick_time)
                peak_t = times[peak_ends] - times[peak_starts]
                good_peaks = np.where(peak_t < max_lick_time)[0]
                # Check if we have any good peaks to start with, if not just move on
                if good_peaks.size == 0:
                    good_spans[i_thr] = (empty, empty)
                    n_peaks[i_thr] = 0
                    continue

                # Only look at peaks that are at least depth_back thresholds deep
                # MATLAB checks cap(start:end) < thresholds(iThresh-N) (inclusive end)
                # i.e. keep a peak iff ANY sample in trace[start:end+1] is below
                # depth_thr, i.e. iff the span's minimum is below it. Vectorized
                # (see _span_min); this replaces a per-peak Python loop with
                # np.delete (O(peaks^2) per threshold) that made 24 h recordings
                # crawl. Results are identical (tests/test_basic_algorithm_parity.py).
                depth_thr = thresholds[i_thr - depth_back]
                ps_good = peak_starts[good_peaks]
                pe_good = peak_ends[good_peaks]
                # An inverted (end < start) pair is an empty slice: never deep enough.
                valid = pe_good >= ps_good
                deep = np.zeros(ps_good.size, dtype=bool)
                deep[valid] = _span_min(trace, ps_good[valid], pe_good[valid]) < depth_thr
                ps_good = ps_good[deep]
                pe_good = pe_good[deep]

                # Number of total good peaks with this threshold
                n_peaks[i_thr] = int(ps_good.size)
                good_spans[i_thr] = (ps_good, pe_good)

            peak_info[animal] = {
                    'thr_and_peaks': np.vstack([thresholds, n_peaks]),
            }

            peaks_row = peak_info[animal]['thr_and_peaks'][1, :]
            if np.all(np.isnan(peaks_row)):
                # No threshold was ever evaluated -- same zero-lick case as a
                # too-flat channel above. Save it rather than dropping it.
                _record_zero_licks(data, animal, logfile,
                                   f"no thresholds evaluated ({len(unique_vals)} capacitance levels)")
                missing_data = save_filtered_data(data, animal, filtered_h5f, logfile)
                if missing_data:
                    del filtered_h5f[animal]
                continue
            max_peaks = np.nanmax(peaks_row)
            i_thr = int(np.where(peaks_row == max_peaks)[0][0])

            # Lick sample = deepest (first minimum) sample of each accepted dip.
            ps_best, pe_best = good_spans[i_thr]
            pb_ = np.array([ps_ + int(np.argmin(trace[ps_:pe_+1]))
                            for ps_, pe_ in zip(ps_best.tolist(), pe_best.tolist())],
                           dtype=int)
            lick_times_arr = times[pb_]

            # Isolated single dips are usually noise, not licking (mice lick in
            # rhythmic bouts). So keep a lick only if it has at least one OTHER
            # detected lick within 1.0 s. A lick's nearest neighbor is always one
            # of its two adjacent licks in time, so we sort the times and compare
            # each to its previous/next neighbor -- O(N log N) and O(N) memory.
            # (The earlier full N x N pairwise-difference matrix was O(N^2) memory
            # and would exhaust RAM/swap when detection over-counted noise.)
            # The 1.0 s window is intentional here and differs from the hilbert
            # algorithm's 0.5 s window below.
            if len(lick_times_arr) >= 2:
                order = np.argsort(lick_times_arr)
                st = lick_times_arr[order]
                gaps = np.diff(st)  # gap between each pair of adjacent sorted licks
                nn = np.full(st.shape, np.inf)  # distance to nearest neighbor
                nn[:-1] = np.minimum(nn[:-1], gaps)   # gap to next
                nn[1:] = np.minimum(nn[1:], gaps)     # gap to previous
                keep_sorted = nn <= 1.0
                keep = np.zeros(len(lick_times_arr), dtype=bool)
                keep[order[keep_sorted]] = True
                data['lick_times'] = lick_times_arr[keep]
                data['lick_indices'] = pb_[keep]
            else:
                data['lick_times'] = np.array([])
                data['lick_indices'] = np.array([], dtype=int)
            data['num_licks'] = int(len(data['lick_times']))
        else:
            # Too few capacitance levels for the threshold scan: a dip cannot be
            # depth_back levels deep, so there are no licks to find here.
            _record_zero_licks(data, animal, logfile,
                               f"only {len(unique_vals)} capacitance levels, "
                               f"need > {depth_back + 1} for the threshold scan")
        print(f"Animal {animal} had {data['num_licks']} licks detected")
        missing_data = save_filtered_data(data, animal, filtered_h5f, logfile)
        if missing_data:
            # Rather than doing a return here to fully skip the file, we can
            # just delete the group for the animal with missing data and continue
            del filtered_h5f[animal]
            continue
    return False # no missing data


def hilbert_algorithm(data_by_animal, filtered_h5f, logfile):
    """Trying a mix of using raw trace and Hilbert envelope to find licks"""
    for (animal, data) in data_by_animal.items():
        fs = data['fs']
        trace = data['cap_data']

        # 8–12 Hz band-pass applied as high- then low-pass
        bh, ah = scs.butter(4, 8, btype='high', fs=fs)
        bl, al = scs.butter(8, 12, btype='low', fs=fs)
        # scs.filtfilt applies the filter forward and then backward. This
        # doubles the effective filter order but produces ZERO phase shift, so
        # detected lick times are not delayed relative to the raw trace. That
        # timing fidelity is why filtfilt is used instead of a single-pass filter.
        filtered_data = scs.filtfilt(bh, ah, trace)
        filtered_data = scs.filtfilt(bl, al, filtered_data)

        # A second high-pass then low-pass pass, applied once each. This sharpens
        # the band edges (steeper roll-off) at a small cost in signal amplitude.
        #
        # NOTE / correction of a previous "clever" one-liner: these two lines
        # used to read
        #     filtered_data = [scs.filtfilt(bh, ah, filtered_data)
        #                      for _ in range(6)][-1]
        # which LOOKS like six stacked passes but is not. Inside a list
        # comprehension `filtered_data` is never reassigned, so all six entries
        # are identical (one pass over the same input) and taking [-1] keeps
        # exactly ONE of them. The net effect was a single extra pass; the five
        # discarded copies only wasted memory and time, and the old docs claiming
        # "7 filter passes" were inaccurate. The plain single application below
        # reproduces the ACTUAL past numerical behavior (two high-pass and two
        # low-pass passes total) while being honest about it. Do NOT "restore" a
        # real 6x loop without re-validating downstream results -- it would
        # change every lick count this function produces.
        filtered_data = scs.filtfilt(bh, ah, filtered_data)
        filtered_data = scs.filtfilt(bl, al, filtered_data)

        # The analytic (Hilbert) transform turns the oscillating band-passed
        # signal into a smooth amplitude envelope: env[i] is the instantaneous
        # strength of the 8-12 Hz rhythm at sample i. Licking bouts show up as
        # sustained high-envelope regions, so the envelope is a good gate for
        # "is the animal rhythmically licking right now?".
        env = np.abs(scs.hilbert(filtered_data))

        # Keep only samples where the licking-band envelope is strong. The
        # threshold is an empirically tuned fraction (0.261) of this trace's peak
        # envelope; it scales per-animal because absolute capacitance amplitude
        # varies between sensors/animals. Samples below it are treated as no
        # rhythmic licking.
        env_thr  = 0.261 * np.max(env)
        env_mask = env > env_thr

        # A lick is marked on the RAW trace by a sharp downward step: a single
        # lick briefly pulls the sipper capacitance down. `downs` are the sample
        # indices where the raw value drops by more than 15 capacitance units
        # from one sample to the next. The +15 deadband ignores small
        # sample-to-sample wiggle so only genuine lick-sized drops qualify.
        # (+1 shifts the index to the sample AFTER the drop, i.e. the low point.)
        downs = np.where((trace[:-1] > trace[1:] + 15))[0] + 1

        # Apply envelope mask
        candidates = [i for i in downs if env_mask[i]]

        # At least ~80 ms (0.08 s) between licks (mice shouldn't be licking faster than that)
        min_dist = int(0.08 * fs)
        lick_idxs = []
        for idx in candidates:
            if not lick_idxs or (idx - lick_idxs[-1]) > min_dist:
                lick_idxs.append(idx)

        # At most 500 ms between licks — isolated licks with no neighbors within that window are excluded
        max_dist = int(0.5 * fs)
        lick_idxs_ = []
        if len(lick_idxs) < 3:
            data['lick_indices'] = []
            data['lick_times'] = []
            print(f'No licks recorded for {animal}')
            continue

        for i, idx in enumerate(lick_idxs):
            # Only check to the right on the first loop and left on the last:
            if i == 0: 
                ir = lick_idxs[i+1]
                ir2 = lick_idxs[i+2]
                if np.abs(idx - ir) < max_dist and np.abs(idx - ir2) < 2*max_dist:
                    lick_idxs_.append(idx)
                continue
            if i == len(lick_idxs)-1: 
                il = lick_idxs[i-1]
                il2 = lick_idxs[i-2]
                if np.abs(idx - il) < max_dist and np.abs(idx - il2) < 2*max_dist:
                    lick_idxs_.append(idx)
                continue
            il = lick_idxs[i-1]
            ir = lick_idxs[i+1]
            if np.abs(idx - il) < max_dist and np.abs(idx - ir) < max_dist:
                lick_idxs_.append(idx)
                continue
            elif np.abs(idx - il) < max_dist:
                if i == 1: continue # Make sure we have 2 points to the left
                il2 = lick_idxs[i-2]
                if np.abs(idx - il2) < 2*max_dist:
                    lick_idxs_.append(idx)
                continue
            elif np.abs(idx - ir) < max_dist:
                if i >= len(lick_idxs)-2: continue # There must be 2 points right
                ir2 = lick_idxs[i+2]
                if np.abs(idx - ir2) < 2*max_dist:
                    lick_idxs_.append(idx)
                continue
        lick_idxs = lick_idxs_
        lick_times = np.array(lick_idxs) / fs
        data['lick_times'] = lick_times
        data['lick_indices'] = lick_idxs

        num_licks = len(lick_times)
        data['num_licks'] = num_licks
        print(f"Animal {animal} had {num_licks} licks detected")
        missing_data = save_filtered_data(data, animal, filtered_h5f, logfile)
        if missing_data:
            # Rather than doing a return here to fully skip the file, we can
            # just delete the group for the animal with missing data and continue
            del filtered_h5f[animal]
            continue
    return False # no data missing


# --- ML lick detection (opt-in via algorithm='ml') --------------------------------------------
# Imported lazily so the heavy torch import is only paid when the ML path is actually used.
def _load_ml_nets(checkpoint=None):
    """Load the bout/point networks. Defaults to the fine-tuned checkpoint, falling back to the
    weights ported straight from MATLAB. Returns (bout_net, point_net) in eval mode."""
    import os
    import torch
    from ml_detection.nets import LickBoutNet, LickPointNet
    from ml_detection.weights_io import load_matlab_nets
    default_finetuned = os.path.join("ml_detection", "checkpoints", "finetuned.pt")
    ckpt = checkpoint or default_finetuned
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        bout, point = LickBoutNet(), LickPointNet()
        bout.load_state_dict(state["bout"]); point.load_state_dict(state["point"])
        bout.eval(); point.eval()
        return bout, point
    # Fall back to the raw MATLAB port (old-scale weights) if no fine-tuned checkpoint exists yet.
    return load_matlab_nets(os.path.join("ML Detection MATLAB Code", "lickNets.mat"))


# Module-level indirection so tests can monkeypatch `data_analysis.detect_licks`. Left as None
# here (rather than imported eagerly) so `import data_analysis` never pulls in torch via
# ml_detection.infer -- the default basic_threshold path shouldn't pay that cost. The real
# function is imported lazily on first use inside ml_algorithm (see below).
detect_licks = None


def ml_algorithm(data_by_animal, filtered_h5f, logfile, checkpoint=None, nets=None):
    """Detect licks with the ML cascade. Mirrors basic_algorithm's I/O contract exactly:
    writes lick_times, lick_indices, num_licks per animal (num_licks is in-memory only, same
    as basic_algorithm -- neither is persisted via save_filtered_data)."""
    global detect_licks
    if detect_licks is None:
        from ml_detection.infer import detect_licks as _dl
        detect_licks = _dl
    # `nets` lets filter_data load the networks once per file instead of once per animal.
    bout_net, point_net = nets if nets is not None else _load_ml_nets(checkpoint)
    for animal, data in data_by_animal.items():
        cap = np.asarray(data["cap_data"])
        t = np.asarray(data["time_data"])
        if len(cap) == 0:
            data["lick_times"] = np.array([]); data["lick_indices"] = np.array([], dtype=int)
            data["num_licks"] = 0
            missing_data = save_filtered_data(data, animal, filtered_h5f, logfile)
            if missing_data:
                del filtered_h5f[animal]
                continue
            continue
        lick_times = np.asarray(detect_licks(t, cap, bout_net, point_net))
        # Map lick TIMES back to indices in the (trimmed) original trace: np.searchsorted returns
        # the leftmost insertion point, i.e. the first sample with t >= lick_time (not nearest).
        lick_indices = np.searchsorted(t, lick_times)
        lick_indices = np.clip(lick_indices, 0, len(t) - 1)
        data["lick_times"] = lick_times
        data["lick_indices"] = lick_indices
        data["num_licks"] = int(len(lick_times))
        print(f"Animal {animal} had {len(lick_times)} licks detected (ML)")
        missing_data = save_filtered_data(data, animal, filtered_h5f, logfile)
        if missing_data: return missing_data
    return False


def compute_lick_durations(trace, times, lick_indices, *, rel_height=0.5, wlen=41,
                           max_gap=0.05, max_duration=1. / 6.):
    """Measure how long each detected lick keeps the capacitance down.

    Scientific idea
    ---------------
    A lick pulls the sipper capacitance into a roughly square dip. Its duration
    is the dip's width at `rel_height` of its depth (0.5 = half depth), measured
    between linearly interpolated crossing points so it is not limited to whole
    samples. Measuring relative to each dip's own depth keeps durations
    comparable across animals, sensors and CDT settings whose dips differ several
    fold in absolute counts (checked on ACG-26-3 / ACG-26-5: ~40 ms medians for
    dips 20-350 counts deep).

    The dip's baseline is found within `wlen` samples centred on the lick (about
    +-180 ms at ~110 Hz, enough to reach the baseline between licks at 6-10 Hz).
    Without that window the search can run across the whole trace and use a
    far-away outlier spike as the baseline.

    Sampling is non-uniform, so crossing positions are converted to seconds
    through `times`, never through an assumed rate. A lick gets NaN when:
      - a gap between samples inside its span is >= `max_gap` s (a recording
        stall would otherwise read as a long lick), or
      - it lasts >= `max_duration` s (the same limit lick detection uses), or
      - the dip has no measurable depth.
    NaN keeps the outputs index-aligned with `lick_indices`.

    Resolution caveat: at ~110 Hz most dips span only 2-5 samples, so absolute
    durations shift with `rel_height` (~30/39/48 ms medians at 0.25/0.5/0.75).
    Compare durations measured with the same setting.

    The crossing times themselves are returned as each lick's start ('lick_onsets')
    and end ('lick_offsets'), in the units of `times`. They are the better lick
    start: the detected lick time is the dip's deepest sample, which comes after
    the tongue has already made contact. Onsets/offsets are kept even when the
    duration is NaN for a gap or length (they are still the best estimate there);
    they are NaN only when the dip has no measurable depth.

    Returns {'lick_durations': seconds, 'lick_depths': capacitance counts,
    'lick_onsets', 'lick_offsets'}, all float arrays the same length and order as
    `lick_indices`.
    """
    from scipy.signal import peak_prominences, peak_widths
    import warnings

    li = np.asarray(lick_indices, dtype=np.int64).ravel()
    durations = np.full(li.shape, np.nan)
    depths = np.full(li.shape, np.nan)
    onsets = np.full(li.shape, np.nan)
    offsets = np.full(li.shape, np.nan)
    if li.size == 0:
        return {'lick_durations': durations, 'lick_depths': depths,
                'lick_onsets': onsets, 'lick_offsets': offsets}
    neg = -np.asarray(trace, dtype=float)
    times = np.asarray(times, dtype=float)
    n = neg.size

    # Snap each index to the deepest sample within +-3 samples. Basic threshold
    # indices are already the minimum; ML/hilbert indices can sit on the edge.
    lo = np.clip(li - 3, 0, n - 1)
    offs = np.arange(7)
    win = np.clip(lo[:, None] + offs[None, :], 0, n - 1)
    peaks = win[np.arange(li.size), np.argmax(neg[win], axis=1)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zero-prominence peaks -> handled below
        prom = peak_prominences(neg, peaks, wlen=wlen)[0]
        _w, _h, left, right = peak_widths(neg, peaks, rel_height=rel_height, wlen=wlen)

    idx = np.arange(n)
    t_left = np.interp(left, idx, times)
    t_right = np.interp(right, idx, times)
    dur = t_right - t_left

    # Largest sampling interval inside each span (from the sample before the
    # left crossing to the sample after the right one).
    dt = np.diff(times)
    a = np.clip(np.floor(left).astype(np.int64), 0, n - 1)
    b = np.clip(np.ceil(right).astype(np.int64), 0, n - 1)
    max_dt = np.array([dt[i:j].max() if j > i else 0.0 for i, j in zip(a, b)])

    ok = (prom > 0) & (max_dt < max_gap) & (dur < max_duration)
    durations[ok] = dur[ok]
    has_dip = prom > 0
    depths[has_dip] = prom[has_dip]
    onsets[has_dip] = t_left[has_dip]
    offsets[has_dip] = t_right[has_dip]
    return {'lick_durations': durations, 'lick_depths': depths,
            'lick_onsets': onsets, 'lick_offsets': offsets}


def compute_bout_structure(lick_times, ibi_threshold=0.25, min_licks=3, lick_durations=None):
    """Compute ILIs and bout structure from an array of lick timestamps.
    A bout requires >= min_licks licks with all consecutive ILIs < ibi_threshold (seconds).
    Returned ILIs are within-bout only.

    If `lick_durations` (aligned with lick_times, NaN = unmeasurable) is given,
    also returns 'bout_median_lick_durations': the median measurable lick
    duration in each bout (NaN if none were measurable)."""
    if len(lick_times) < min_licks:
        out = {
            'ILIs': np.array([]),
            'bout_lick_counts': np.array([], dtype=int),
            'bout_durations': np.array([]),
            'bout_start_times': np.array([]),
        }
        if lick_durations is not None:
            out['bout_median_lick_durations'] = np.array([])
        return out
    all_ILIs = np.diff(lick_times)
    bout_boundaries = np.where(all_ILIs >= ibi_threshold)[0]
    bout_start_idxs = np.r_[0, bout_boundaries + 1]
    bout_end_idxs = np.r_[bout_boundaries, len(lick_times) - 1]
    lick_counts = bout_end_idxs - bout_start_idxs + 1

    valid = lick_counts >= min_licks
    bout_start_idxs = bout_start_idxs[valid]
    bout_end_idxs = bout_end_idxs[valid]
    lick_counts = lick_counts[valid]

    within_ILIs = [all_ILIs[s:e] for s, e in zip(bout_start_idxs, bout_end_idxs)]
    out = {
        'ILIs': np.concatenate(within_ILIs) if within_ILIs else np.array([]),
        'bout_lick_counts': lick_counts.astype(int),
        'bout_durations': lick_times[bout_end_idxs] - lick_times[bout_start_idxs],
        'bout_start_times': lick_times[bout_start_idxs],
    }
    if lick_durations is not None:
        durs = np.asarray(lick_durations, dtype=float)
        meds = []
        for s, e in zip(bout_start_idxs, bout_end_idxs):
            d = durs[s:e + 1]
            d = d[np.isfinite(d)]
            meds.append(float(np.median(d)) if d.size else np.nan)
        out['bout_median_lick_durations'] = np.array(meds, dtype=float)
    return out


def save_filtered_data(data, animal, filtered_h5f, logfile):
    missing_data = False # flag to track if we missed any weights or volumes

    grp = filtered_h5f.create_group(animal)
    # We need to check each of these things to make sure they were actually populated
    try:
        grp.create_dataset('cap_data', data=data['cap_data'])
        grp.create_dataset('time_data', data=data['time_data'])
        grp.create_dataset('lick_times', data=data['lick_times'])
        grp.create_dataset('lick_indices', data=data['lick_indices'])
        grp.create_dataset('used_start_idx', data=data['used_start_idx'])
        grp.create_dataset('used_stop_idx', data=data['used_stop_idx'])
    except KeyError as e:
        with open(logfile, 'a') as lf:
            lf.write(f"Caught KeyError {e}, no licks recorded for {animal}\n")
        print(f'Caught KeyError {e}, no licks recorded for {animal}')
        missing_data = True
    lick_durations = None
    try:
        # Per-lick duration/depth, index-aligned with lick_indices (NaN = unmeasurable).
        lk = compute_lick_durations(data['cap_data'], data['time_data'], data['lick_indices'])
        grp.create_dataset('lick_durations', data=lk['lick_durations'])
        grp.create_dataset('lick_depths', data=lk['lick_depths'])
        grp.create_dataset('lick_onsets', data=lk['lick_onsets'])
        grp.create_dataset('lick_offsets', data=lk['lick_offsets'])
        lick_durations = lk['lick_durations']
    except Exception as e:
        with open(logfile, 'a') as lf:
            lf.write(f"Warning: could not measure lick durations for {animal}: {e}\n")
    try:
        metrics = compute_bout_structure(data['lick_times'], ibi_threshold=1., min_licks=2,
                                         lick_durations=lick_durations)
        grp.create_dataset('ILIs', data=metrics['ILIs'])
        grp.create_dataset('bout_lick_counts', data=metrics['bout_lick_counts'])
        grp.create_dataset('bout_durations', data=metrics['bout_durations'])
        grp.create_dataset('bout_start_times', data=metrics['bout_start_times'])
        if 'bout_median_lick_durations' in metrics:
            grp.create_dataset('bout_median_lick_durations',
                               data=metrics['bout_median_lick_durations'])
    except Exception as e:
        with open(logfile, 'a') as lf:
            lf.write(f"Warning: could not save supplementary metrics for {animal}: {e}\n")
    try:
        grp.create_dataset('consumed_vol', data=data['consumed_vol'])
    except KeyError as e:
        with open(logfile, 'a') as lf:
            lf.write(f"Caught KeyError {e}, volumes not recorded for {animal}\n")
        print(f'Caught KeyError {e}, volumes were likely not recorded for {animal}')
        missing_data = True
    try:
        grp.create_dataset('weight', data=data['weight'])
    except KeyError as e:
        # Control cages have no animal, so no weight. Expected: don't flag missing.
        if is_control(animal):
            with open(logfile, 'a') as lf:
                lf.write(f"No weight for control {animal} (expected), skipping weight check\n")
        else:
            with open(logfile, 'a') as lf:
                lf.write(f"Caught KeyError {e}, weight not recorded for {animal}\n")
            print(f'Caught KeyError {e}, weight was likely not recorded for {animal}')
            missing_data = True

    return missing_data
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#                                 MIT License                                 #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#     Copyright (c) 2025 Christopher John Parker <parkecp@mail.uc.edu>        #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
