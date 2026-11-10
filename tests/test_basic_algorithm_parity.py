"""The vectorized basic_algorithm must give exactly the same licks as the original.

The depth check used to be a per-peak Python loop with np.delete (O(peaks^2)
per threshold), which made 24 h recordings crawl. It was replaced by a
cumulative-count formulation. `_reference_basic_algorithm` below is a verbatim
copy of the pre-vectorization implementation, kept ONLY as an oracle: every
synthetic trace must yield identical lick indices/times and bout metrics.
"""

import h5py
import numpy as np
import pytest

from data_analysis import basic_algorithm, save_filtered_data, _record_zero_licks


# ---------------------------------------------------------------------------
# Verbatim copy of the original implementation (oracle). Do not "improve".
# ---------------------------------------------------------------------------
def _reference_basic_algorithm(data_by_animal, filtered_h5f, logfile):
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
            peak_bins = [None] * len(thresholds)

            for i_thr in range(depth_back, len(thresholds)):
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

                # Only look at peaks that are at least depth_back thresholds deep
                # MATLAB checks cap(start:end) < thresholds(iThresh-N) (inclusive end)
                depth_thr = thresholds[i_thr - depth_back]
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

            pb_ = peak_info[animal]['peak_bins'][i_thr]
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


# ---------------------------------------------------------------------------
# Synthetic traces
# ---------------------------------------------------------------------------
def _trace(seed, n=30000, fs=110.0, lick_rate=7.0, n_bouts=12, edge_dips=False):
    rng = np.random.default_rng(seed)
    trace = (800 - rng.integers(0, 45, size=n)).astype(np.int64)  # noisy baseline
    t = np.arange(n) / fs
    for _ in range(n_bouts):
        start = int(rng.integers(0, n - 2000))
        n_licks = int(rng.integers(1, 40))
        for k in range(n_licks):
            s = start + int(k * fs / lick_rate) + int(rng.integers(-2, 3))
            width = int(rng.integers(1, 25))          # some dips too long to be licks
            depth = int(rng.integers(5, 120))         # some dips too shallow
            trace[max(s, 0):s + width] -= depth
    if edge_dips:
        trace[:4] -= 100   # trace starts below threshold
        trace[-3:] -= 100  # trace ends below threshold
    return t, trace


def _run(fn, t, trace, logfile):
    data = {
        'cap_data': trace.copy(), 'time_data': t, 'fs': 110.0,
        'consumed_vol': 0.4, 'weight': 25.0,
        'used_start_idx': 0, 'used_stop_idx': len(trace) - 1,
    }
    f = h5py.File(f'mem{id(data)}', driver='core', backing_store=False, mode='w')
    try:
        fn({'A1': data}, f, str(logfile))
        return {k: f['A1'][k][()] for k in f['A1'].keys()}
    finally:
        f.close()


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('edge_dips', [False, True])
def test_vectorized_matches_reference(seed, edge_dips, tmp_path):
    t, trace = _trace(seed, edge_dips=edge_dips)
    new = _run(basic_algorithm, t, trace, tmp_path / 'new.log')
    ref = _run(_reference_basic_algorithm, t, trace, tmp_path / 'ref.log')
    assert new.keys() == ref.keys()
    assert ref['lick_indices'].size > 0
    for k in ref:
        np.testing.assert_array_equal(new[k], ref[k], err_msg=k)


@pytest.mark.parametrize('seed', range(3))
def test_parity_float_trace_and_high_rate(seed, tmp_path):
    """Float traces skip the integer fast path; at 1 kHz lick spans exceed the
    64-sample vectorized window and take the direct fallback."""
    t, trace = _trace(seed, n=60000, fs=1000.0, lick_rate=7.0)
    trace = np.repeat(trace[::9], 9)[:len(t)]  # widen dips to ~100+ samples
    for tr in (trace, trace.astype(np.float64) + 0.25):
        new = _run(basic_algorithm, t, tr, tmp_path / 'new.log')
        ref = _run(_reference_basic_algorithm, t, tr, tmp_path / 'ref.log')
        assert new.keys() == ref.keys()
        for k in ref:
            np.testing.assert_array_equal(new[k], ref[k], err_msg=k)


def test_span_min_matches_naive_including_long_spans():
    from data_analysis import _span_min
    rng = np.random.default_rng(3)
    trace = rng.integers(0, 1000, size=5000)
    starts = np.sort(rng.choice(4800, size=200, replace=False))
    lengths = rng.integers(1, 150, size=200)  # some > 64 -> fallback path
    ends = np.minimum(starts + lengths - 1, len(trace) - 1)
    expected = np.array([trace[s:e + 1].min() for s, e in zip(starts, ends)])
    np.testing.assert_array_equal(_span_min(trace, starts, ends), expected)
    assert (lengths > 64).any()
