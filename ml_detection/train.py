"""
Transfer learning: adapt the MATLAB-ported nets to new high-CDT data.

Steps (per the spec): load ported weights, refit the scalar zscore normalization from the new
curated segments (the primary fix for the CDT magnitude change), split by SESSION to avoid leakage
between adjacent overlapping windows, then fine-tune ALL layers at a low learning rate with early
stopping. The bout net trains on 3 s segments; the point net trains on 21-sample central windows
(via prepare_point_segments).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ml_detection.nets import LickBoutNet, LickPointNet
from ml_detection.weights_io import load_matlab_nets
from ml_detection.dataset import load_training_h5, prepare_point_segments
from ml_detection.preprocess import POINT_WIN


def refit_zscore(net, segments):
    """Set net.norm_mean / net.norm_std to the global scalar mean/std of the new segments."""
    with torch.no_grad():
        net.norm_mean.fill_(float(np.mean(segments)))
        net.norm_std.fill_(float(np.std(segments)) or 1.0)


def session_split(session_ids, val_fraction=0.25, seed=0):
    """Partition unique session ids into disjoint (train, val) sets, holding out whole sessions."""
    uniq = sorted(set(session_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, round(len(uniq) * val_fraction))
    val = set(uniq[:n_val])
    train = set(uniq[n_val:])
    return train, val


def _train_one(net, X, y, Xval, yval, epochs, lr, batch_size):
    """Fine-tune a single net; return best val accuracy (early-stopped)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    best, best_state, patience, bad = 0.0, None, 5, 0
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(torch.tensor(Xval)).argmax(1) == torch.tensor(yval)).float().mean().item()
        if acc > best:
            best, best_state, bad = acc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return best


def fine_tune(training_files, out_path, epochs=100, lr=1e-4, batch_size=64,
              mat_path="ML Detection MATLAB Code/lickNets.mat"):
    """
    Fine-tune both nets on curated training HDF5 files (one per session).

    `training_files` is a dict {session_id: path}. Sessions are split into train/val; both nets are
    fine-tuned; the best checkpoint is saved to out_path as {'bout', 'point', 'meta'}.
    Returns a metrics dict.
    """
    bout, point = load_matlab_nets(mat_path)
    train_ids, val_ids = session_split(list(training_files))

    def gather(ids):
        bout_x, bout_y, pt_x, pt_y = [], [], [], []
        for sid in ids:
            d = load_training_h5(training_files[sid])
            bout_x.append(d["samples"]); bout_y.append(d["labels_bout"])
            X, y = prepare_point_segments(d)
            if len(y):
                pt_x.append(X); pt_y.append(y)
        BX = np.concatenate(bout_x)[:, None, :].astype(np.float32) if bout_x \
            else np.empty((0, 1, 0), np.float32)
        BY = np.concatenate(bout_y).astype(np.int64) if bout_y else np.empty((0,), np.int64)
        if pt_x:
            PX = np.concatenate(pt_x, axis=0).astype(np.float32)
            PY = np.concatenate(pt_y).astype(np.int64)
        else:
            # No point-labeled segments in this split (all labels_bout==0) -- produce
            # correctly-shaped empty arrays instead of letting np.concatenate([]) raise.
            PX = np.empty((0, 1, POINT_WIN), np.float32)
            PY = np.empty((0,), np.int64)
        return BX, BY, PX, PY

    BXt, BYt, PXt, PYt = gather(train_ids)
    BXv, BYv, PXv, PYv = gather(val_ids)

    # Refit normalization from TRAIN segments only (no val leakage).
    refit_zscore(bout, BXt[:, 0, :])
    point_acc = 0.0
    if len(PXt) == 0:
        print("WARNING: no point-training samples in this split (all labels_bout==0); "
              "skipping point-net fine-tuning.")
    else:
        refit_zscore(point, PXt[:, 0, :])

    bout_acc = _train_one(bout, BXt, BYt, BXv, BYv, epochs, lr, batch_size)
    if len(PXt) > 0:
        point_acc = _train_one(point, PXt, PYt, PXv, PYv, epochs=20, lr=lr, batch_size=128)

    meta = {"fs": 100, "win_sec": 3, "point_win": 21,
            "train_sessions": sorted(train_ids), "val_sessions": sorted(val_ids)}
    torch.save({"bout": bout.state_dict(), "point": point.state_dict(), "meta": meta}, out_path)
    return {"bout_val_acc": bout_acc, "point_val_acc": point_acc, "meta": meta}
