"""
Seed an initial training set for a new-scale recording.

Strategy (port of MATLAB `buildInitialTrainingSet`, but seeded by CLiQR's existing threshold
detector instead of findpeaks): resample to 100 Hz, run the threshold detector to get candidate
lick times, then sample category-balanced 3 s windows (0 / 1-3 / >=4 licks in the central 1 s).
Each window is offset per-window (training convention).
"""
import numpy as np

from ml_detection.preprocess import (
    resample_to_100hz, offset_window, WIN_SAMPLES, CENTER_SAMPLES, FS,
)


def bootstrap_segments(time_s, cap, threshold_lick_times, n_samples=200, seed=0):
    """
    Build initial labeled 3 s segments from one recording.

    Parameters
    ----------
    threshold_lick_times : np.ndarray
        Lick times (seconds, original base) from the existing basic_algorithm threshold detector.

    Returns a training dict compatible with save_training_h5 / prepare_point_segments.
    """
    rng = np.random.RandomState(seed)
    t, y = resample_to_100hz(time_s, cap)
    # Convert seeded lick times to 100 Hz sample indices.
    lick_samples = np.clip(np.searchsorted(t, threshold_lick_times), 0, len(t) - 1)
    lick_flags = np.zeros(len(t), dtype=bool)
    lick_flags[lick_samples] = True

    center_start = round(WIN_SAMPLES / 2 - CENTER_SAMPLES / 2)
    per_cat = round(n_samples / 3)
    segments, times, lick_idx, labels = [], [], [], []

    def count_center_licks(start):
        cs = start + center_start
        return int(lick_flags[cs:cs + CENTER_SAMPLES].sum())

    max_start = len(t) - WIN_SAMPLES
    if max_start <= 0:
        raise ValueError("Recording shorter than one 3 s window after resampling.")

    for cat in range(3):
        got = 0
        for _ in range(100000):
            if got >= per_cat:
                break
            s = rng.randint(0, max_start)
            n_c = count_center_licks(s)
            if cat == 0 and n_c != 0: continue
            if cat == 1 and not (1 <= n_c <= 3): continue
            if cat == 2 and n_c < 4: continue
            win = offset_window(y[s:s + WIN_SAMPLES])
            in_win = np.nonzero(lick_flags[s:s + WIN_SAMPLES])[0]
            segments.append(win.astype(np.float32))
            times.append((t[s:s + WIN_SAMPLES] - t[s]).astype(np.float64))
            lick_idx.append(in_win.astype(np.int64))
            labels.append(1 if count_center_licks(s) > 0 else 0)
            got += 1

    return {
        "samples": np.asarray(segments, dtype=np.float32),
        "t": np.asarray(times, dtype=np.float64),
        "lick_idx": lick_idx,
        "labels_bout": np.asarray(labels, dtype=np.int64),
        "fs": FS, "win_sec": 3, "center_sec": 1,
    }
