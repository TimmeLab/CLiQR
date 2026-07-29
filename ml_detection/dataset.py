"""
Training-set schema (HDF5) and point-window builder for the lick nets.

We store curated 3 s segments plus per-segment lick indices and bout labels. This mirrors the
MATLAB `training` struct but in HDF5 with our own field names. `prepare_point_segments` ports
MATLAB `preparePointSegments`: for each segment whose central 1 s contains >=1 lick, it emits one
21-sample window centered on every sample in the central 1 s, labeled by whether that center
sample is a lick.
"""
import h5py
import numpy as np

from ml_detection.preprocess import WIN_SAMPLES, CENTER_SAMPLES, POINT_WIN


def save_training_h5(path, segments, times, lick_idx, labels_bout, meta):
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=np.asarray(segments, dtype=np.float32))
        f.create_dataset("t", data=np.asarray(times, dtype=np.float64))
        f.create_dataset("labels_bout", data=np.asarray(labels_bout, dtype=np.int64))
        # Ragged lick indices: store one variable-length dataset per segment under a group.
        g = f.create_group("lick_idx")
        for i, li in enumerate(lick_idx):
            g.create_dataset(str(i), data=np.asarray(li, dtype=np.int64))
        f.attrs["fs"] = 100
        f.attrs["win_sec"] = 3
        f.attrs["center_sec"] = 1
        for k, v in meta.items():
            f.attrs[f"meta_{k}"] = v


def load_training_h5(path):
    with h5py.File(path, "r") as f:
        n = f["samples"].shape[0]
        lick_idx = [np.asarray(f["lick_idx"][str(i)]) for i in range(n)]
        return {
            "samples": np.asarray(f["samples"]),
            "t": np.asarray(f["t"]),
            "lick_idx": lick_idx,
            "labels_bout": np.asarray(f["labels_bout"]),
            "fs": int(f.attrs["fs"]),
            "win_sec": int(f.attrs["win_sec"]),
            "center_sec": int(f.attrs["center_sec"]),
        }


def prepare_point_segments(training, win_pt=POINT_WIN):
    """
    Build point-level training windows. Returns (X [n,1,win_pt] float32, y [n] int64).

    Only segments with labels_bout == 1 contribute; only window centers inside the central 1 s are
    used; windows that would run past the segment edge are skipped. Matches MATLAB exactly.
    """
    half_pt = (win_pt - 1) // 2
    win_samples = WIN_SAMPLES
    center_samples = CENTER_SAMPLES
    center_start = round(win_samples / 2 - center_samples / 2)   # 0-based
    center_range = range(center_start, center_start + center_samples)

    all_x, all_y = [], []
    for seg, li, lab in zip(training["samples"], training["lick_idx"], training["labels_bout"]):
        if lab != 1:
            continue
        L = len(seg)
        lick_mask = np.zeros(L, dtype=bool)
        lick_mask[np.asarray(li, dtype=int)] = True
        for c in center_range:
            if c - half_pt < 0 or c + half_pt >= L:
                continue
            all_x.append(seg[c - half_pt:c + half_pt + 1])
            all_y.append(1 if lick_mask[c] else 0)
    X = np.asarray(all_x, dtype=np.float32)[:, None, :] if all_x else np.empty((0, 1, win_pt), np.float32)
    y = np.asarray(all_y, dtype=np.int64)
    return X, y
