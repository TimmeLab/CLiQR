"""
Fine-tune the lick-detection networks on a hand-labeled training set (validation gate #3).

Input is the curated HDF5 written by the labeler (`..._curated.h5`). The labeler saves via
`save_training_h5`, which does NOT carry the per-segment `source` ("animal/session") provenance
that the bootstrap step stored — but the labeler never adds, drops, or reorders segments, so the
i-th curated segment corresponds to the i-th segment of the ORIGINAL bootstrap file. We recover the
session of each segment by index from that original file, which lets us split TRAIN/VALIDATION by
whole session (no leakage between the two) exactly as the design intends.

We then reuse the tested `ml_detection.train.fine_tune`: it loads the MATLAB-ported weights, refits
the zscore normalization from the TRAIN segments only (the key fix for the new CDT magnitude
scale), fine-tunes both nets with early stopping, and saves the checkpoint. Because `fine_tune`
takes a {session_id -> file_path} mapping, we write one small temporary HDF5 per session.

After training we reload the checkpoint and report precision / recall / F1 on the held-out sessions
(both the bout-level central-lick decision and the point-level per-sample decision), which is the
actual gate-#3 metric.

Usage (from the repository root, cliqr-gui environment):
    PYTHONPATH=. python scripts/finetune_from_labeled.py \
        ml_training_ACG-26-3_bootstrap_curated_curated.h5

By default the segment->session source is read from the de-curated original filename (strip the
trailing `_curated`s and add `.h5`); override with --source-from if needed. The checkpoint is
written to ml_detection/checkpoints/finetuned.pt so data_analysis._load_ml_nets auto-discovers it.
"""

import argparse
import os
import tempfile

import h5py
import numpy as np
import torch

from ml_detection.dataset import load_training_h5, save_training_h5, prepare_point_segments
from ml_detection.nets import LickBoutNet, LickPointNet
from ml_detection.train import fine_tune


def derive_source_path(labeled_path):
    """Strip trailing '_curated' suffixes to recover the original bootstrap filename that holds the
    per-segment `source` dataset. e.g. 'x_curated_curated.h5' -> 'x.h5'."""
    stem, ext = os.path.splitext(labeled_path)
    while stem.endswith("_curated"):
        stem = stem[: -len("_curated")]
    return stem + ext


def read_sources(source_path, n_expected):
    """Read the per-segment 'source' string dataset from the original bootstrap file, verifying it
    has one entry per labeled segment (the ordering guarantee that makes index-recovery valid)."""
    with h5py.File(source_path, "r") as f:
        if "source" not in f:
            raise SystemExit(
                f"{source_path} has no 'source' dataset — cannot recover session provenance. "
                f"Pass the original bootstrap file with --source-from, or regenerate it."
            )
        sources = [s.decode() if isinstance(s, bytes) else str(s) for s in f["source"][()]]
    if len(sources) != n_expected:
        raise SystemExit(
            f"Segment count mismatch: labeled file has {n_expected} segments but "
            f"{source_path} has {len(sources)} sources. They must correspond 1:1 by index."
        )
    return sources


def subset_dict(training, indices):
    """Build a training dict containing only the segments at `indices` (preserving all fields)."""
    return {
        "samples": training["samples"][indices],
        "t": training["t"][indices],
        "lick_idx": [training["lick_idx"][i] for i in indices],
        "labels_bout": training["labels_bout"][indices],
        "fs": training["fs"], "win_sec": training["win_sec"], "center_sec": training["center_sec"],
    }


def _prf(pred, true):
    """Precision, recall, F1 for binary arrays, treating class 1 as the positive class."""
    pred = np.asarray(pred).astype(bool)
    true = np.asarray(true).astype(bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_split(checkpoint_path, labeled, sources, val_sessions):
    """Load the fine-tuned checkpoint and score it on the held-out sessions (bout + point levels)."""
    state = torch.load(checkpoint_path, map_location="cpu")
    bout, point = LickBoutNet(), LickPointNet()
    bout.load_state_dict(state["bout"]); point.load_state_dict(state["point"])
    bout.eval(); point.eval()

    val_idx = [i for i, s in enumerate(sources) if s in val_sessions]
    val = subset_dict(labeled, val_idx)

    # Bout level: does the net's central-lick decision match the curated label?
    BX = torch.tensor(val["samples"][:, None, :], dtype=torch.float32)
    with torch.no_grad():
        bout_pred = bout(BX).argmax(1).numpy()
    bout_prf = _prf(bout_pred, val["labels_bout"])

    # Point level: for the central-1 s windows of positive segments, does the net's per-sample
    # decision match the curated per-sample lick labels?
    PX, py = prepare_point_segments(val)
    if len(py):
        with torch.no_grad():
            point_pred = point(torch.tensor(PX, dtype=torch.float32)).argmax(1).numpy()
        point_prf = _prf(point_pred, py)
        n_point = len(py)
    else:
        point_prf, n_point = (0.0, 0.0, 0.0), 0

    return {"n_val_segments": len(val_idx), "bout_prf": bout_prf,
            "n_point": n_point, "point_prf": point_prf}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune lick nets on curated labels (gate 3).")
    parser.add_argument("labeled_h5", help="Curated training HDF5 from the labeler.")
    parser.add_argument("--source-from", default=None,
                        help="Original bootstrap HDF5 holding the per-segment 'source' dataset "
                             "(default: de-curated labeled filename).")
    parser.add_argument("--out", default=os.path.join("ml_detection", "checkpoints", "finetuned.pt"),
                        help="Checkpoint output path (default: the path _load_ml_nets discovers).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    labeled = load_training_h5(args.labeled_h5)
    n = len(labeled["samples"])
    source_path = args.source_from or derive_source_path(args.labeled_h5)
    sources = read_sources(source_path, n)
    unique_sessions = sorted(set(sources))
    print(f"Loaded {n} labeled segments across {len(unique_sessions)} sessions "
          f"({int(labeled['labels_bout'].sum())} central-lick positives).")

    # Write one temporary HDF5 per session so fine_tune can split by whole session.
    work_dir = tempfile.mkdtemp(prefix="ml_finetune_")
    session_files = {}
    for session in unique_sessions:
        idx = [i for i, s in enumerate(sources) if s == session]
        sub = subset_dict(labeled, idx)
        safe = session.replace("/", "_")
        path = os.path.join(work_dir, f"{safe}.h5")
        save_training_h5(path, sub["samples"], sub["t"], sub["lick_idx"], sub["labels_bout"],
                         {"session": session})
        session_files[session] = path

    print(f"Fine-tuning (epochs={args.epochs}, lr={args.lr}) ...")
    metrics = fine_tune(session_files, args.out, epochs=args.epochs, lr=args.lr,
                        batch_size=args.batch_size)

    val_sessions = set(metrics["meta"]["val_sessions"])
    print(f"\nSaved fine-tuned checkpoint to {args.out}")
    print(f"  train sessions: {len(metrics['meta']['train_sessions'])}   "
          f"val sessions: {len(val_sessions)}")
    print(f"  fine_tune val accuracy — bout: {metrics['bout_val_acc']:.4f}   "
          f"point: {metrics['point_val_acc']:.4f}")

    # Held-out precision/recall/F1 (the real gate-3 metric).
    ev = evaluate_split(args.out, labeled, sources, val_sessions)
    bp, br, bf = ev["bout_prf"]
    pp, pr, pf = ev["point_prf"]
    print(f"\nHeld-out session metrics ({ev['n_val_segments']} segments):")
    print(f"  bout  (central-lick present):  P={bp:.3f}  R={br:.3f}  F1={bf:.3f}")
    print(f"  point (per-sample, {ev['n_point']} windows):  P={pp:.3f}  R={pr:.3f}  F1={pf:.3f}")


if __name__ == "__main__":
    main()
