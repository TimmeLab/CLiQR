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
    """True if animal_id is a control cage (no animal, no weight recorded)."""
    return bool(_CONTROL_RE.match(str(animal_id)))


def filter_data(raw_h5f, filtered_h5f, sensor_animal_map, logfile, time_fix=None, algorithm='basic_threshold', recording_length=2*60*60):
    """This organizes the data by animal and handles some common issues relating to
    recording start/stop times, volumes, and weights. Current algorithm choices:
        - basic_threshold
        - hilbert (Hilbert envelope threshold)"""
    data_dict = {}
    # We only expect up to 3 nested levels based on the DataRecording notebook
    for k,v in raw_h5f.items():
        data_dict[k] = {} if isinstance(v, h5py._hl.group.Group) else v
        if not isinstance(data_dict[k], dict): continue
        for k2,v2 in v.items():
            data_dict[k][k2] = (
                {} if isinstance(v2, h5py._hl.group.Group) else v2[()]
            )
            if not isinstance(data_dict[k][k2], dict): continue
            for k3,v3 in v2.items():
                data_dict[k][k2][k3] = v3[()]
    # Loop through all boards and sensors and truncate at start_time and
    # stop_time, then subtract the first time point from the data
    for board_id, board_data in data_dict.items():
        if board_id == 'comments': continue
        for sensor_id, sensor_data in board_data.items():
            # Initialize with the start and stop indices indicating the entire time series
            start_idx = 0
            stop_idx = -1
            # A sensor may have been started and stopped several times in one
            # session. Each cycle writes a start_time key: the first is plain
            # "start_time", later ones are "start_time1", "start_time2", ...
            # We want the LAST cycle, so collect every start_time key and index
            # it by its cycle number (the unnumbered first cycle is treated as
            # cycle -1 so it sorts before the numbered ones).
            pattern = re.compile(r'^start_time(\d+)?$')
            cycle_number_to_key = {}
            for k in sensor_data.keys():
                m = pattern.match(k)
                if m:
                    cycle_number = int(m.group(1)) if m.group(1) else -1
                    cycle_number_to_key[cycle_number] = k
            if cycle_number_to_key:
                # The most recent cycle is the one with the largest cycle number.
                last_cycle_number = max(cycle_number_to_key)
                last_start = cycle_number_to_key[last_cycle_number]
                start_time = sensor_data[last_start][()]
                # Try the stop_time corresponding to the start_time above
                try:
                    stop_time = sensor_data['stop' + last_start[5:]][()]
                except KeyError:
                    # Stop time wasn't recorded (likely clicked stop all button too soon)
                    stop_time = sensor_data['time_data'][-1]
            else: # no start time recorded
                print(f"Warning: no start/stop times recorded for {board_id} {sensor_id}, create a time fix file or we will default to the entire trace.")
                start_time = sensor_data['time_data'][0] # start at beginning
                stop_time = sensor_data['time_data'][-1] # stop at end
            total_recording_time = stop_time - start_time
            start_idx = np.argmin(
                np.abs(sensor_data['time_data'] - start_time)
            )
            stop_idx = np.argmin(
                np.abs(sensor_data['time_data'] - stop_time)
            )

            sensor_data['time_data'] = (
                sensor_data['time_data'][start_idx:stop_idx+1] -
                    sensor_data['time_data'][start_idx]
			)
            sensor_data['cap_data'] = (
                sensor_data['cap_data'][start_idx:stop_idx+1]
			)
            sensor_data['fs'] = (
				(stop_idx - start_idx) /
					(sensor_data['time_data'][-1] - sensor_data['time_data'][0])
			)

            # Check if we still have at least recording_length
            remaining_time = sensor_data['time_data'][-1] - sensor_data['time_data'][0]
            if remaining_time < recording_length:
                print(f"Warning: {board_id} {sensor_id} was left with only {remaining_time} seconds of recording after trimming to recorded start/stop times. Consider a time fix file.")
            else:
                # Now we trim any excess to the desired recording length (from the end)
                revised_stop_idx = np.argmin(
                    np.abs(sensor_data['time_data'] - recording_length)
                )
                sensor_data['time_data'] = sensor_data['time_data'][:revised_stop_idx]
                sensor_data['cap_data'] = sensor_data['cap_data'][:revised_stop_idx]

			
            if 'stop_vol' in sensor_data.keys():
                sensor_data['consumed_vol'] = (
					sensor_data['start_vol'] - sensor_data['stop_vol']
				)

            # Add in datasets to indicate which start/stop indices we actually used in the analysis
            sensor_data['used_start_idx'] = start_idx
            sensor_data['used_stop_idx'] = stop_idx


    # Reorganize the data to be by animal ID, agnostic wrt any board/sensor numbering
    data_by_animal = {}
    for idx,row in sensor_animal_map.iterrows():
        sensor = row.name
        animal = row.item()
        # Look up which board's HDF5 group holds this sensor (derived from the
        # rack wiring in utils.state, so it stays correct across rack designs).
        board_id = SENSOR_BOARD_MAP.get(sensor)
        if board_id is None:
            print(f"Sensor {sensor} is not part of the current rack layout; skipping.")
            continue
        try:
            data_by_animal[animal] = data_dict[board_id][f"sensor_{sensor}"]
        except KeyError as e:
            print(f"Missing key in data_dict: {e}")

    # Flag to track if we had any animals without necessary data recorded
    missing_data = False
    _run_optimal_threshold(data_by_animal)
    if algorithm == 'basic_threshold':
        missing_data = basic_algorithm(data_by_animal, filtered_h5f, logfile)
    elif algorithm == 'hilbert':
        missing_data = hilbert_algorithm(data_by_animal, filtered_h5f, logfile)
    return missing_data


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
      2. deep enough           (it must also cross a threshold two levels lower,
                                the "2-threshold-deep" rule, so shallow noise
                                wiggles are rejected).
    The lick time is taken at the minimum (deepest) sample of the accepted dip.

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

        # Because the capacitance values are discretized, we can focus only
        # on the thresholds between the discrete values
        unique_vals = np.unique(trace)
        if len(unique_vals) > 3: # We need at least 3 different values for the algorithm
            # Get thresholds between each unique capacitance value
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            n_peaks = np.full((len(thresholds),), np.nan)
            peak_bins = [None] * len(thresholds)
            
            for i_thr in range(2, len(thresholds)):
                thr = thresholds[i_thr]
                # Identify peaks below threshold. 1 marks the time bin just before a
                # peak starts (the last bin that isn't in the peak), -1 marks the time
                # bin just before a peak ends (the last bin in the peak)
                peak_trans = np.diff((trace < thr).astype(np.int8))
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
                    peak_bins[i_thr] = np.array([], dtype=int)
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
                    peak_bins[i_thr] = np.array([], dtype=int)
                    n_peaks[i_thr] = 0
                    continue

                # Only look at peaks that are at least two thresholds deep
                # MATLAB checks cap(start:end) < thresholds(iThresh-2) (inclusive end)
                depth_thr = thresholds[i_thr - 2]
                i_peak = 0
                good_peaks = good_peaks.astype(int, copy=False)
                while i_peak < good_peaks.size:
                    p_idx = good_peaks[i_peak]
                    # If we don't cross the 2-deep threshold, remove that peak from the "good" list
                    ps_ = int(peak_starts[p_idx])
                    pe_ = int(peak_ends[p_idx])
                    peak_trace = trace[ps_:pe_+1]
                    if np.count_nonzero(peak_trace < depth_thr) == 0:
                        good_peaks = np.delete(good_peaks, i_peak, 0)
                    else:
                        # It was a good peak still, increment i_peak
                        i_peak += 1

                # Check again if we have any good peaks, as we may have deleted them all
                if good_peaks.size == 0:
                    peak_bins[i_thr] = np.array([], dtype=int)
                    n_peaks[i_thr] = 0
                    continue
                    
                # Number of total good peaks with this threshold
                n_peaks[i_thr] = int(good_peaks.size)

                # Record the peak time bins
                temp_peak_bins = np.full((good_peaks.size,), -1, dtype=int)
                for j, p_idx in enumerate(good_peaks):
                    ps_ = int(peak_starts[p_idx])
                    pe_ = int(peak_ends[p_idx])
                    peak_trace = trace[ps_:pe_+1]
                    if peak_trace.size == 0: continue
                    peak_rel_min = int(np.argmin(peak_trace))  # relative min index in the peak
                    temp_peak_bins[j] = ps_ + peak_rel_min
                peak_bins[i_thr] = temp_peak_bins

            peak_info[animal] = {
                    'thr_and_peaks': np.vstack([thresholds, n_peaks]),
                    'peak_bins': peak_bins,
            }

            peaks_row = peak_info[animal]['thr_and_peaks'][1, :]
            if np.all(np.isnan(peaks_row)):
                continue
            max_peaks = np.nanmax(peaks_row)
            i_thr = int(np.where(peaks_row == max_peaks)[0][0])

            pb_ = peak_info[animal]['peak_bins'][i_thr]
            lick_times_arr = times[pb_]

            # Isolated single dips are usually noise, not licking (mice lick in
            # rhythmic bouts). So keep a lick only if it has at least one OTHER
            # detected lick within 1.0 s. Implementation: build the full matrix
            # of pairwise time differences, blank the diagonal (a lick's distance
            # to itself) with infinity, then keep a lick when its nearest
            # neighbor is <= 1.0 s away. (The 1.0 s window is intentional here and
            # differs from the hilbert algorithm's 0.5 s window below.)
            if len(lick_times_arr) >= 2:
                diffs = np.abs(lick_times_arr[:, None] - lick_times_arr[None, :])
                np.fill_diagonal(diffs, np.inf)
                keep = diffs.min(axis=1) <= 1.0
                data['lick_times'] = lick_times_arr[keep]
                data['lick_indices'] = pb_[keep]
            else:
                data['lick_times'] = np.array([])
                data['lick_indices'] = np.array([], dtype=int)
            data['num_licks'] = int(len(data['lick_times']))
        else:
            # Didn't have more than 3 separate capacitance values, so probably nothing was recorded
            continue
        print(f"Animal {animal} had {data['num_licks']} licks detected")
        missing_data = save_filtered_data(data, animal, filtered_h5f, logfile)
        if missing_data: return True
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
        if missing_data: return missing_data
    return False # no data missing

def _optimize_simple_threshold(data_by_animal, n_steps=200):
    """Grid-search threshold fraction f ∈ [0, 1] of each animal's dynamic range
    that maximizes R² between per-animal lick count and volume consumed."""
    animals = [a for a in data_by_animal
               if 'consumed_vol' in data_by_animal[a]
               and len(data_by_animal[a].get('cap_data', [])) > 0]
    if len(animals) < 3:
        return 0.5  # not enough data; fall back to midpoint

    vols = np.array([data_by_animal[a]['consumed_vol'] for a in animals])
    traces = [data_by_animal[a]['cap_data'] for a in animals]

    fractions = np.linspace(0.01, 0.99, n_steps)
    r2_vals = np.full(n_steps, np.nan)

    for k, frac in enumerate(fractions):
        counts = np.zeros(len(animals))
        for j, trace in enumerate(traces):
            thr = trace.min() + frac * (trace.max() - trace.min())
            below = (trace < thr).astype(np.int8)
            counts[j] = np.sum(np.diff(below) == 1)
        if np.std(counts) == 0 or np.std(vols) == 0:
            continue
        r2_vals[k] = np.corrcoef(counts, vols)[0, 1] ** 2

    best_k = int(np.nanargmax(r2_vals))
    return fractions[best_k]


def _run_optimal_threshold(data_by_animal):
    """Add optimal threshold lick detection to each animal's data dict.
    Threshold fraction (of each animal's dynamic range) is optimized to maximize
    R² between lick count and volume consumed. Every downward crossing counts as one lick."""
    opt_frac = _optimize_simple_threshold(data_by_animal)
    print(f"Optimal threshold: fraction = {opt_frac:.3f} of each animal's dynamic range")

    for animal, data in data_by_animal.items():
        trace = data['cap_data']
        times = data['time_data']
        thr = float(trace.min() + opt_frac * (trace.max() - trace.min()))
        below = (trace < thr).astype(np.int8)
        lick_starts = np.where(np.diff(below) == 1)[0] + 1
        data['optimal_lick_times'] = times[lick_starts]
        data['optimal_lick_indices'] = lick_starts


def compute_bout_structure(lick_times, ibi_threshold=0.25, min_licks=3):
    """Compute ILIs and bout structure from an array of lick timestamps.
    A bout requires >= min_licks licks with all consecutive ILIs < ibi_threshold (seconds).
    Returned ILIs are within-bout only."""
    if len(lick_times) < min_licks:
        return {
            'ILIs': np.array([]),
            'bout_lick_counts': np.array([], dtype=int),
            'bout_durations': np.array([]),
            'bout_start_times': np.array([]),
        }
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
    return {
        'ILIs': np.concatenate(within_ILIs) if within_ILIs else np.array([]),
        'bout_lick_counts': lick_counts.astype(int),
        'bout_durations': lick_times[bout_end_idxs] - lick_times[bout_start_idxs],
        'bout_start_times': lick_times[bout_start_idxs],
    }


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
            lf.write(f"Caught KeyError {e}, volumes not recorded for {animal}\n")
        print(f'Caught KeyError {e}, no licks recorded for {animal}')
        missing_data = True
    try:
        grp.create_dataset('optimal_lick_times', data=data.get('optimal_lick_times', np.array([])))
        grp.create_dataset('optimal_lick_indices', data=data.get('optimal_lick_indices', np.array([], dtype=int)))
        metrics = compute_bout_structure(data['lick_times'], ibi_threshold=1., min_licks=2)
        grp.create_dataset('ILIs', data=metrics['ILIs'])
        grp.create_dataset('bout_lick_counts', data=metrics['bout_lick_counts'])
        grp.create_dataset('bout_durations', data=metrics['bout_durations'])
        grp.create_dataset('bout_start_times', data=metrics['bout_start_times'])
        opt_metrics = compute_bout_structure(data.get('optimal_lick_times', np.array([])), ibi_threshold=1., min_licks=2)
        grp.create_dataset('optimal_ILIs', data=opt_metrics['ILIs'])
        grp.create_dataset('optimal_bout_lick_counts', data=opt_metrics['bout_lick_counts'])
        grp.create_dataset('optimal_bout_durations', data=opt_metrics['bout_durations'])
        grp.create_dataset('optimal_bout_start_times', data=opt_metrics['bout_start_times'])
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
    
