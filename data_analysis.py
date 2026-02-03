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


def filter_data(raw_h5f, filtered_h5f, sensor_animal_map, logfile, time_fix=None, algorithm='basic_threshold'):
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
            # Determine which start time keys we have
            pattern = re.compile(r'^start_time(\d+)?$')
            matches = {}
            for k in sensor_data.keys():
                m = pattern.match(k)
                if m:
                    num = int(m.group(1)) if m.group(1) else -1
                    matches[num] = k
            if matches:
                num = -np.inf
                for n in matches.keys():
                    if n > num: num = n
                last_start = matches[num]
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
                sensor_data['time_data'][start_idx:stop_idx] -
                    sensor_data['time_data'][start_idx]
			)
            sensor_data['cap_data'] = (
                sensor_data['cap_data'][start_idx:stop_idx]
			)
            sensor_data['fs'] = (
				(stop_idx - start_idx) /
					(sensor_data['time_data'][-1] - sensor_data['time_data'][0])
			)
			
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
        # We need to determine which FT232H was used for the recordings
        if sensor in [1,2,3,7,8,9]:
            board_id = 'board_FT232H0'
        elif sensor in [4,5,6,10,11,12]:
            board_id = 'board_FT232H1'
        elif sensor in [13,14,15,19,20,21]:
            board_id = 'board_FT232H2'
        elif sensor in [16,17,18,22,23,24]:
            board_id = 'board_FT232H3'
        try:
            data_by_animal[animal] = data_dict[board_id][f"sensor_{sensor}"]
        except KeyError as e:
            print(f"Missing key in data_dict: {e}")

    if algorithm == 'basic_threshold':
        basic_algorithm(data_by_animal, filtered_h5f, logfile)
    elif algorithm == 'hilbert':
        hilbert_algorithm(data_by_animal, filtered_h5f, logfile)

def basic_algorithm(data_by_animal, filtered_h5f, logfile):
    """Basic algorithm based on thresholding"""
    for (animal, data) in data_by_animal.items():
        #print(f"Animal ID {animal}")
        trace = data['cap_data']
        times = data['time_data']

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

                # If we still have no peaks (or mismatched start/end), skip.
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
            if len(pb_) == 1:
                pb_ = int(pb_)
                data['lick_times'] = np.array(times[pb_]).reshape(1,)
            else:
                data['lick_times'] = times[pb_]
            data['num_licks'] = int(len(data['lick_times']))
            data['lick_indices'] = pb_
        else:
            # Didn't have more than 3 separate capacitance values, so probably nothing was recorded
            continue
        print(f"Animal {animal} had {data['num_licks']} licks detected")
        save_filtered_data(data, animal, filtered_h5f, logfile)


def hilbert_algorithm(data_by_animal, filtered_h5f, logfile):
    """Trying a mix of using raw trace and Hilbert envelope to find licks"""
    for (animal, data) in data_by_animal.items():
        fs = data['fs']
        trace = data['cap_data']

        # 8–12 Hz band-pass applied as high- then low-pass
        bh, ah = scs.butter(4, 8, btype='high', fs=fs)
        bl, al = scs.butter(8, 12, btype='low', fs=fs)
        filtered_data = scs.filtfilt(bh, ah, trace)
        filtered_data = scs.filtfilt(bl, al, filtered_data)
        filtered_data = [scs.filtfilt(bh, ah, filtered_data) for _ in range(6)][-1]
        filtered_data = [scs.filtfilt(bl, al, filtered_data) for _ in range(6)][-1]

        # Convert filtered data to Hilbert envelope
        env = np.abs(scs.hilbert(filtered_data))

        # Thresholding with Hilbert envelope
        env_thr  = 0.261 * np.max(env)
        env_mask = env > env_thr

        # Where the value decreases from one point to the next
        downs = np.where((trace[:-1] > trace[1:] + 15))[0] + 1

        # Apply envelope mask
        candidates = [i for i in downs if env_mask[i]]

        # At least ~80 ms (0.08 s) between licks (mice shouldn't be licking faster than that)
        min_dist = int(0.08 * fs)
        lick_idxs = []
        for idx in candidates:
            if not lick_idxs or (idx - lick_idxs[-1]) > min_dist:
                lick_idxs.append(idx)

        # At most 500 ms between licks. This is probably overly permissive,
        # but we need some way to say 
        # "it's impossible to tell if a single lick is really a lick"
        # convert to timestamps
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

        # print(f"lick_times: {lick_times}")
        num_licks = len(lick_times)
        data['num_licks'] = num_licks
        print(f"Animal {animal} had {num_licks} licks detected")
        save_filtered_data(data, animal, filtered_h5f, logfile)

def save_filtered_data(data, animal, filtered_h5f, logfile):
    grp = filtered_h5f.create_group(animal)
    # We need to check each of these things to make sure they were actually populated
    try:
        grp.create_dataset('lick_times', data=data['lick_times'])
        grp.create_dataset('lick_indices', data=data['lick_indices'])
        grp.create_dataset('used_start_idx', data=data['used_start_idx'])
        grp.create_dataset('used_stop_idx', data=data['used_stop_idx'])
    except KeyError as e:
        with open(logfile, 'a') as lf:
            lf.write(f"Caught KeyError {e}, volumes not recorded for {animal}\n")
        print(f'Caught KeyError {e}, no licks recorded for {animal}')
    try:
        grp.create_dataset('consumed_vol', data=data['consumed_vol'])
    except KeyError as e:
        with open(logfile, 'a') as lf:
            lf.write(f"Caught KeyError {e}, volumes not recorded for {animal}\n")
        print(f'Caught KeyError {e}, volumes were likely not recorded for {animal}')
    try:
        grp.create_dataset('weight', data=data['weight'])
    except KeyError as e:
        with open(logfile, 'a') as lf:
            lf.write(f"Caught KeyError {e}, weight not recorded for {animal}\n")
        print(f'Caught KeyError {e}, weight was likely not recorded for {animal}')
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
    
