"""
Vectorized inference cascade: the PyTorch equivalent of MATLAB `detectLicksFromRaw`.

Faithful behavior, with ONE output-identical optimization: the point net is evaluated for every
candidate sample in a single batched forward pass instead of MATLAB's per-sample Python loop
(`_point_mask_naive` documents and tests the equivalence). Because the point net input uses the
GLOBAL offset (MATLAB inference convention), a sample's classification does not depend on which
bout window reached it, so overlapping positive bout windows collapse to their union losslessly —
matching MATLAB's OR semantics (`lickMaskGlobal` is set and never unset).
"""
import numpy as np
import torch

from ml_detection.preprocess import (
    resample_to_100hz, offset_global, offset_window,
    FS, WIN_SAMPLES, POINT_WIN, CENTER_SAMPLES, BOUT_STEP,
)

HALF_PT = (POINT_WIN - 1) // 2
MERGE_SAMPLES = int(round(0.02 * FS))   # 20 ms


def _positive_bout_samples(y, bout_net):
    """
    Slide the 3 s bout window (step 0.5 s), classify each with the per-window offset, and return
    the SORTED UNIQUE set of global sample indices that fall inside at least one positive window.
    """
    L = len(y)
    starts = range(0, L - WIN_SAMPLES + 1, BOUT_STEP)
    windows = []
    win_starts = []
    for s in starts:
        w = offset_window(y[s:s + WIN_SAMPLES])
        windows.append(w)
        win_starts.append(s)
    if not windows:
        return np.array([], dtype=int)
    X = torch.tensor(np.stack(windows), dtype=torch.float32).unsqueeze(1)  # [nWin,1,300]
    with torch.no_grad():
        pred = bout_net(X).argmax(dim=1).numpy()          # 1 == 'lick'
    positive = np.zeros(L, dtype=bool)
    for s, is_lick in zip(win_starts, pred):
        if is_lick == 1:
            positive[s:s + WIN_SAMPLES] = True
    return np.nonzero(positive)[0]


def _gather_point_windows(y_offset, samples):
    """Build [n,1,21] batch of point windows centered on each valid sample (edges dropped)."""
    L = len(y_offset)
    valid = samples[(samples - HALF_PT >= 0) & (samples + HALF_PT < L)]
    if len(valid) == 0:
        return valid, np.empty((0, 1, POINT_WIN), dtype=np.float32)
    idx = valid[:, None] + np.arange(-HALF_PT, HALF_PT + 1)[None, :]
    batch = y_offset[idx].astype(np.float32)[:, None, :]
    return valid, batch


def _point_mask_vectorized(y_offset, samples, point_net):
    L = len(y_offset)
    mask = np.zeros(L, dtype=bool)
    valid, batch = _gather_point_windows(y_offset, samples)
    if len(valid) == 0:
        return mask
    with torch.no_grad():
        pred = point_net(torch.tensor(batch)).argmax(dim=1).numpy()
    mask[valid[pred == 1]] = True
    return mask


def _point_mask_naive(y_offset, samples, point_net):
    """Reference per-sample loop (MATLAB-style), for test equivalence only."""
    L = len(y_offset)
    mask = np.zeros(L, dtype=bool)
    for c in samples:
        if c - HALF_PT < 0 or c + HALF_PT >= L:
            continue
        seg = y_offset[c - HALF_PT:c + HALF_PT + 1].astype(np.float32)
        with torch.no_grad():
            p = point_net(torch.tensor(seg)[None, None, :]).argmax(dim=1).item()
        if p == 1:
            mask[c] = True
    return mask


def _merge_lick_points(mask, t):
    """Merge True samples within 20 ms into one lick; representative time = cluster center."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return np.array([])
    clusters = [[idx[0]]]
    for i in idx[1:]:
        if i - clusters[-1][-1] <= MERGE_SAMPLES:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    centers = [c[len(c) // 2] for c in clusters]
    return t[np.array(centers)]


def detect_licks(time_s, cap, bout_net, point_net):
    """
    Detect lick times from a raw (irregular) capacitance recording.

    Returns lick times in the ORIGINAL time base (seconds), obtained by resampling to 100 Hz,
    running the cascade, and mapping the 100 Hz cluster-center times back through the original
    recording start.
    """
    bout_net.eval(); point_net.eval()
    t, y = resample_to_100hz(time_s, cap)
    positive = _positive_bout_samples(y, bout_net)
    y_glob = offset_global(y)                              # point-net inference offset
    mask = _point_mask_vectorized(y_glob, positive, point_net)
    lick_t = _merge_lick_points(mask, t)                   # seconds relative to resampled t0
    # t already starts at time_s[0], so lick_t is in the original time base.
    return lick_t
