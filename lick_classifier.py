"""
lick_classifier.py — CNN + Transformer for capacitive lick detection.

Architecture
------------
Raw cap trace → resample to 100 Hz → z-score normalize
→ 1D CNN encoder (4× downsample) → Transformer encoder
→ per-frame lick probability

Input:  (B, 6000)   100 Hz, 60 s chunk
Output: (B, 1500)   40 ms / frame probability

Training data format: .pt files produced by LabelingTool.ipynb
Each sample dict has keys: 'cap', 'time', 'lick_times' (all torch.Tensor).
"""

import math
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Fixed hyperparameters ─────────────────────────────────────────────────────
TARGET_FS   = 100        # Hz — all traces resampled to this
CHUNK_S     = 60         # seconds per chunk
N_SAMPLES   = TARGET_FS * CHUNK_S   # 6000 input samples
CNN_STRIDE  = 4          # total CNN downsampling factor
N_FRAMES    = N_SAMPLES  // CNN_STRIDE   # 1500 output frames
FRAME_DT    = CNN_STRIDE / TARGET_FS     # 0.04 s / frame
GAUSS_SIGMA = 1.5        # frames — label smoothing width (~60 ms)


# ── Dataset ───────────────────────────────────────────────────────────────────
class LickDataset(Dataset):
    """Wrap .pt files from LabelingTool into (x, y, has_lick) tensors.

    x        : (N_SAMPLES,) float32 — z-scored capacitance at TARGET_FS
    y        : (N_FRAMES,)  float32 — Gaussian-smoothed lick label in [0, 1]
    has_lick : bool — at least one lick labeled in this chunk
    """

    def __init__(self, pt_files, augment=False):
        self.augment = augment
        self.samples = []
        for f in pt_files:
            data = torch.load(f, weights_only=False)
            self.samples.extend(data['samples'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s         = self.samples[idx]
        cap       = s['cap'].numpy().astype(np.float32)
        time      = s['time'].numpy().astype(np.float32)
        lick_times = s['lick_times'].numpy().astype(np.float32)

        # Resample to fixed 100 Hz grid over [0, CHUNK_S)
        t_grid = np.linspace(0.0, float(CHUNK_S), N_SAMPLES, endpoint=False)
        cap_rs = np.interp(t_grid, time, cap).astype(np.float32)

        # Snap each lick time to deflection onset (pre-z-score, so polarity is intact)
        lick_times = np.array(
            [_snap_lick_to_onset(cap_rs, t_grid, lt) for lt in lick_times],
            dtype=np.float32,
        )

        # Z-score per chunk
        mu, sigma = cap_rs.mean(), cap_rs.std()
        cap_rs = (cap_rs - mu) / (sigma + 1e-6)

        # Build soft label on the downsampled frame grid
        frame_centers = np.arange(N_FRAMES, dtype=np.float32) * FRAME_DT + FRAME_DT / 2
        label = np.zeros(N_FRAMES, dtype=np.float32)
        for lt in lick_times:
            dist_frames = (frame_centers - lt) / FRAME_DT
            label += np.exp(-0.5 * (dist_frames / GAUSS_SIGMA) ** 2)
        label = label.clip(0.0, 1.0)

        # Augmentation: flip sign — some sensors dip on lick, others rise
        if self.augment and np.random.rand() < 0.5:
            cap_rs = -cap_rs
        # Additive noise (SNR ~20 dB)
        if self.augment:
            cap_rs += np.random.randn(*cap_rs.shape).astype(np.float32) * 0.1

        x        = torch.from_numpy(cap_rs)
        y        = torch.from_numpy(label)
        has_lick = torch.tensor(len(lick_times) > 0, dtype=torch.bool)
        return x, y, has_lick


# ── Building blocks ───────────────────────────────────────────────────────────
class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.drop(x + self.pe[:x.size(1)])


class _CNNEncoder(nn.Module):
    """Extract local lick-waveform features and downsample 4×.

    Stride schedule: (2, 2, 1) so the last layer adds depth without
    further compressing the sequence — preserves 40 ms / frame resolution.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,       32,      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32,      64,      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64,      d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) → (B, 1, T) → (B, d_model, T//4) → (B, T//4, d_model)
        return self.net(x.unsqueeze(1)).transpose(1, 2)


# ── Main model ────────────────────────────────────────────────────────────────
class LickTransformer(nn.Module):
    """
    Lick event detector.

    Input:  (B, N_SAMPLES) z-scored capacitance at TARGET_FS
    Output: (B, N_FRAMES)  lick logits (pass through sigmoid for probability)

    Parameters
    ----------
    d_model   : transformer embedding dimension
    n_heads   : number of attention heads (must divide d_model)
    n_layers  : number of transformer encoder layers
    d_ff      : feed-forward hidden dimension
    dropout   : dropout rate
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = _CNNEncoder(d_model)
        self.pe  = _SinusoidalPE(d_model, max_len=N_FRAMES + 64, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,  # pre-norm: more stable training
        )
        # enable_nested_tensor requires norm_first=False; suppress the spurious warning
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cnn(x)           # (B, N_FRAMES, d_model)
        z = self.pe(z)
        z = self.transformer(z)   # (B, N_FRAMES, d_model)
        return self.head(z).squeeze(-1)  # (B, N_FRAMES) — raw logits


# ── Loss ──────────────────────────────────────────────────────────────────────
def lick_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float = 25.0) -> torch.Tensor:
    """BCE with upweighted positive class to handle sparse lick events."""
    pw = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(
    model: LickTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pos_weight: float = 25.0,
    grad_clip: float  = 1.0,
) -> float:
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = lick_loss(model(x), y, pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(
    model: LickTransformer,
    loader: DataLoader,
    device: str,
    pos_weight: float = 25.0,
) -> float:
    model.eval()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        total += lick_loss(model(x), y, pos_weight).item()
    return total / len(loader)


@torch.no_grad()
def eval_detection_metrics(
    model: LickTransformer,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
    match_window_s: float = 0.12,
) -> dict:
    """Compute precision, recall, F1 by matching predicted licks to ground-truth.

    A predicted lick matches a ground-truth lick if it falls within
    ±match_window_s (default 120 ms, covers the Gaussian label width).
    """
    model.eval()
    match_window_frames = match_window_s / FRAME_DT
    tp = fp = fn = 0

    for x, y, _ in loader:
        x = x.to(device)
        probs  = torch.sigmoid(model(x)).cpu().numpy()  # (B, N_FRAMES)
        labels = y.numpy()                               # (B, N_FRAMES)

        for b in range(probs.shape[0]):
            pred_peaks = _pick_peaks(probs[b], threshold, min_gap_frames=int(0.08 / FRAME_DT))
            gt_peaks   = _pick_peaks(labels[b], 0.3,     min_gap_frames=int(0.08 / FRAME_DT))

            matched_gt = set()
            for pp in pred_peaks:
                dists = [abs(pp - gp) for gp in gt_peaks]
                if dists and min(dists) <= match_window_frames:
                    best = int(np.argmin(dists))
                    if best not in matched_gt:
                        tp += 1
                        matched_gt.add(best)
                    else:
                        fp += 1
                else:
                    fp += 1
            fn += len(gt_peaks) - len(matched_gt)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_licks(
    model: LickTransformer,
    cap: np.ndarray,
    time: np.ndarray,
    device: str,
    threshold: float  = 0.5,
    min_gap_s: float  = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect lick times in a full raw capacitance trace.

    Slides non-overlapping CHUNK_S windows across the trace and runs the
    model on each, then merges results.  Compatible with data_analysis.py's
    expected output format.

    Parameters
    ----------
    cap, time   : array-like — raw capacitance and timestamp arrays from HDF5
    threshold   : probability threshold for declaring a lick (default 0.5)
    min_gap_s   : minimum inter-lick interval in seconds (default 80 ms)

    Returns
    -------
    lick_times : (N_licks,) ndarray — absolute timestamps (seconds)
    all_probs  : (N_chunks * N_FRAMES,) ndarray — frame-level probabilities
    """
    model.eval()
    cap  = np.asarray(cap,  dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    t0   = float(time[0])
    time = time - t0   # relative time within recording

    n_chunks  = max(1, int(np.ceil(float(time[-1]) / CHUNK_S)))
    all_frame_times: list[np.ndarray] = []
    all_probs_list:  list[np.ndarray] = []

    for ci in range(n_chunks):
        ts = ci * CHUNK_S
        te = ts + CHUNK_S
        t_grid = np.linspace(ts, te, N_SAMPLES, endpoint=False)
        cap_rs = np.interp(t_grid, time, cap,
                           left=float(cap[0]), right=float(cap[-1])).astype(np.float32)
        mu, sigma = cap_rs.mean(), cap_rs.std()
        cap_rs = (cap_rs - mu) / (sigma + 1e-6)

        x     = torch.from_numpy(cap_rs).unsqueeze(0).to(device)
        probs = torch.sigmoid(model(x))[0].cpu().numpy()

        frame_times = ts + np.arange(N_FRAMES) * FRAME_DT + FRAME_DT / 2
        all_frame_times.append(frame_times)
        all_probs_list.append(probs)

    all_times = np.concatenate(all_frame_times)
    all_probs = np.concatenate(all_probs_list)

    min_gap_frames = max(1, int(min_gap_s / FRAME_DT))
    peak_frames = _pick_peaks(all_probs, threshold, min_gap_frames)
    lick_times  = (all_times[peak_frames] + t0) if len(peak_frames) else np.array([])

    return lick_times, all_probs


# ── Utilities ─────────────────────────────────────────────────────────────────
def _snap_lick_to_onset(cap_rs: np.ndarray, t_grid: np.ndarray, lick_t: float,
                         threshold_frac: float = 0.05,
                         max_lookback_s: float = 1.0) -> float:
    """Walk left from lick_t while cap doesn't rise more than 5% above value at lick_t,
    capped at max_lookback_s before lick_t."""
    n = len(t_grid)
    click_idx  = max(int(np.searchsorted(t_grid, lick_t, side='right')) - 1, 0)
    limit_idx  = int(np.searchsorted(t_grid, max(float(t_grid[0]), float(t_grid[click_idx]) - max_lookback_s)))
    v_hi       = float(cap_rs[click_idx]) * (1.0 + threshold_frac)
    result_idx = click_idx
    for i in range(click_idx - 1, limit_idx - 1, -1):
        if float(cap_rs[i]) <= v_hi:
            result_idx = i
        else:
            break
    return float(t_grid[result_idx])


def _pick_peaks(probs: np.ndarray, threshold: float, min_gap_frames: int) -> list[int]:
    """Find local maxima above threshold separated by at least min_gap_frames."""
    above = probs >= threshold
    peaks: list[int] = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            peak_i = i + int(np.argmax(probs[i:j]))
            if not peaks or (peak_i - peaks[-1]) >= min_gap_frames:
                peaks.append(peak_i)
            i = j
        else:
            i += 1
    return peaks


def build_model(
    device: str | None = None,
    **kwargs,
) -> tuple['LickTransformer', str]:
    """Construct and move model to best available device."""
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    model = LickTransformer(**kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LickTransformer: {n_params:,} parameters | device={device}')
    return model, device


def build_dataloaders(
    pt_files: list[str | Path],
    val_split: float        = 0.15,
    test_split: float       = 0.15,
    batch_size: int         = 16,
    seed: int               = 42,
    neg_to_pos_ratio: float | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split .pt files into train / val / test DataLoaders.

    Splits are at the sample level. Test set is held out entirely and must
    not influence hyperparameter or architecture decisions.

    neg_to_pos_ratio: subsample negatives so n_neg <= n_pos * ratio before
        splitting. None keeps all samples.

    Returns (train_loader, val_loader, test_loader).
    """
    all_samples = []
    for f in pt_files:
        data = torch.load(str(f), weights_only=False)
        all_samples.extend(data['samples'])

    rng = np.random.default_rng(seed)

    pos_idx = [i for i, s in enumerate(all_samples) if len(s['lick_times']) > 0]
    neg_idx = [i for i, s in enumerate(all_samples) if len(s['lick_times']) == 0]
    if neg_to_pos_ratio is not None and len(pos_idx) > 0:
        max_neg = int(len(pos_idx) * neg_to_pos_ratio)
        rng.shuffle(neg_idx)
        neg_idx = neg_idx[:max_neg]
        print(f'Balancing: {len(pos_idx)} pos  +  {len(neg_idx)} neg  '
              f'(ratio {neg_to_pos_ratio:.1f}:1,  {len(all_samples) - len(pos_idx) - len(neg_idx)} neg discarded)')
    else:
        print(f'No balancing: {len(pos_idx)} pos  +  {len(neg_idx)} neg')

    order  = rng.permutation(np.array(pos_idx + neg_idx, dtype=int))
    n      = len(order)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))
    test_idx  = order[:n_test].tolist()
    val_idx   = order[n_test : n_test + n_val].tolist()
    train_idx = order[n_test + n_val:].tolist()

    class _SubsetDataset(Dataset):
        def __init__(self, samples, augment):
            self._samples = samples
            self._augment = augment

        def __len__(self):
            return len(self._samples)

        def __getitem__(self, i):
            s          = self._samples[i]
            cap        = s['cap'].numpy().astype(np.float32)
            time       = s['time'].numpy().astype(np.float32)
            lick_times = s['lick_times'].numpy().astype(np.float32)

            t_grid = np.linspace(0.0, float(CHUNK_S), N_SAMPLES, endpoint=False)
            cap_rs = np.interp(t_grid, time, cap).astype(np.float32)

            lick_times = np.array(
                [_snap_lick_to_onset(cap_rs, t_grid, lt) for lt in lick_times],
                dtype=np.float32,
            )

            mu, sigma = cap_rs.mean(), cap_rs.std()
            cap_rs = (cap_rs - mu) / (sigma + 1e-6)

            frame_centers = np.arange(N_FRAMES, dtype=np.float32) * FRAME_DT + FRAME_DT / 2
            label = np.zeros(N_FRAMES, dtype=np.float32)
            for lt in lick_times:
                dist_frames = (frame_centers - lt) / FRAME_DT
                label += np.exp(-0.5 * (dist_frames / GAUSS_SIGMA) ** 2)
            label = label.clip(0.0, 1.0)

            if self._augment and np.random.rand() < 0.5:
                cap_rs = -cap_rs
            if self._augment:
                cap_rs += np.random.randn(*cap_rs.shape).astype(np.float32) * 0.1

            return (
                torch.from_numpy(cap_rs),
                torch.from_numpy(label),
                torch.tensor(len(lick_times) > 0, dtype=torch.bool),
            )

    train_ds = _SubsetDataset([all_samples[i] for i in train_idx], augment=True)
    val_ds   = _SubsetDataset([all_samples[i] for i in val_idx],   augment=False)
    test_ds  = _SubsetDataset([all_samples[i] for i in test_idx],  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def _n_pos(idx): return sum(1 for i in idx if len(all_samples[i]['lick_times']) > 0)
    print(f'Train: {len(train_ds):3d} samples ({_n_pos(train_idx)} with licks)')
    print(f'Val:   {len(val_ds):3d} samples ({_n_pos(val_idx)} with licks)')
    print(f'Test:  {len(test_ds):3d} samples ({_n_pos(test_idx)} with licks)  ← held out')
    return train_loader, val_loader, test_loader


def load_pt_files(training_dir: str = 'Training Data') -> list[Path]:
    files = sorted(Path(training_dir).glob('*.pt'))
    if not files:
        print(f'No .pt files found in {training_dir}/')
    else:
        print(f'Found {len(files)} .pt file(s): {[f.name for f in files]}')
    return files


def save_checkpoint(model: LickTransformer, path: str | Path, **meta) -> None:
    torch.save({'state_dict': model.state_dict(), **meta}, str(path))


def load_checkpoint(path: str | Path, device: str | None = None, **model_kwargs) -> tuple['LickTransformer', dict]:
    ckpt = torch.load(str(path), map_location=device or 'cpu', weights_only=False)
    model = LickTransformer(**{k: v for k, v in model_kwargs.items()})
    model.load_state_dict(ckpt['state_dict'])
    if device:
        model = model.to(device)
    return model, {k: v for k, v in ckpt.items() if k != 'state_dict'}
