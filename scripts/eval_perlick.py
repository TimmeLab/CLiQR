"""
Per-lick held-out evaluation of the fine-tuned cascade (the meaningful gate-#3 metric).

The fine-tune report scores the point net PER SAMPLE (every 10 ms), which is a pessimistic proxy:
a real lick spans several samples and the cascade merges detections within 20 ms, so per-LICK
recall is normally much higher than per-sample recall. This script measures per-lick precision /
recall / F1 on the held-out sessions by running the actual detection stages on each curated segment
and matching the merged lick times against the hand-labeled licks (20 ms tolerance).

Two numbers are reported, because the curated data is isolated 3 s segments while the cascade is
designed for continuous recordings:

  1. POINT STAGE — run the point net across the whole segment (as it runs inside a positive bout
     window), merge, match. This isolates the point net's true per-lick skill without the bout gate
     or the edge effects that isolated segments introduce.

  2. FULL CASCADE — run detect_licks (bout gate + point pass + merge) on the segment as a mini
     recording. More end-to-end, but understates recall for licks near a segment edge (in a real
     continuous recording the 0.5 s sliding step gives every lick a centered window; an isolated
     segment does not), so treat it as a lower bound.

The bout-level decision is reported separately by the fine-tune driver (held-out F1 ~0.98).

Usage:
    PYTHONPATH=. python scripts/eval_perlick.py \
        ml_training_ACG-26-3_bootstrap_curated_curated.h5 \
        --checkpoint ml_detection/checkpoints/finetuned.pt
"""

import argparse
import os

import numpy as np
import torch

from ml_detection.dataset import load_training_h5
from ml_detection.nets import LickBoutNet, LickPointNet
from ml_detection.train import session_split
from ml_detection.preprocess import offset_global, offset_window, FS, WIN_SAMPLES
from ml_detection.infer import _point_mask_vectorized, _merge_lick_points
from ml_detection.validate import compare_lick_times

# Reuse the exact provenance-recovery helpers from the fine-tune driver so the val split matches.
from scripts.finetune_from_labeled import derive_source_path, read_sources, subset_dict


def _aggregate_perlick(detected_times_per_seg, curated_times_per_seg, tol_s):
    """Sum matched / missed / extra across all segments, then compute overall P / R / F1.

    `compare_lick_times(python, matlab)` treats its 2nd arg as ground truth, so we pass
    (detected, curated): n_matched = true positives, n_missed = curated licks with no detection
    (false negatives), n_extra = detections with no curated lick (false positives).
    """
    tp = miss = extra = 0
    for detected, curated in zip(detected_times_per_seg, curated_times_per_seg):
        r = compare_lick_times(detected, curated, tol_s=tol_s)
        tp += r["n_matched"]
        miss += r["n_missed"]
        extra += r["n_extra"]
    precision = tp / (tp + extra) if (tp + extra) else 0.0
    recall = tp / (tp + miss) if (tp + miss) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fn": miss, "fp": extra,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Per-lick held-out eval of the fine-tuned cascade.")
    parser.add_argument("labeled_h5", help="Curated training HDF5 (same one used for fine-tuning).")
    parser.add_argument("--checkpoint", default=os.path.join("ml_detection", "checkpoints", "finetuned.pt"))
    parser.add_argument("--source-from", default=None,
                        help="Original bootstrap HDF5 with the per-segment 'source' dataset "
                             "(default: de-curated labeled filename).")
    parser.add_argument("--tol-ms", type=float, default=20.0,
                        help="Match tolerance in milliseconds (default 20 ms = the merge window).")
    args = parser.parse_args()

    labeled = load_training_h5(args.labeled_h5)
    n = len(labeled["samples"])
    source_path = args.source_from or derive_source_path(args.labeled_h5)
    sources = read_sources(source_path, n)
    unique_sessions = sorted(set(sources))

    # Reproduce the SAME held-out split the fine-tune driver used (session_split is deterministic
    # for a given seed, and the driver passed the default val_fraction=0.25, seed=0).
    _, val_sessions = session_split(unique_sessions, val_fraction=0.25, seed=0)
    val_idx = [i for i, s in enumerate(sources) if s in val_sessions]
    val = subset_dict(labeled, val_idx)
    print(f"{len(unique_sessions)} sessions; held out {len(val_sessions)} "
          f"({len(val_idx)} segments) for evaluation.")

    # Load the fine-tuned nets.
    state = torch.load(args.checkpoint, map_location="cpu")
    bout, point = LickBoutNet(), LickPointNet()
    bout.load_state_dict(state["bout"]); point.load_state_dict(state["point"])
    bout.eval(); point.eval()

    tol_s = args.tol_ms / 1000.0
    all_samples_idx = np.arange(WIN_SAMPLES)
    seg_time = all_samples_idx / FS       # within-segment time axis (seconds)

    point_detected, cascade_detected, curated_licks = [], [], []
    for seg, lick_idx in zip(val["samples"], val["lick_idx"]):
        curated_t = np.asarray(lick_idx, dtype=int) / FS   # curated lick times (s within segment)
        curated_licks.append(curated_t)

        # (1) POINT STAGE: run the point net across the whole segment, merge to lick times.
        y_glob = offset_global(seg)                        # segment is already max=0; no-op-ish
        mask = _point_mask_vectorized(y_glob, all_samples_idx, point)
        point_licks = _merge_lick_points(mask, seg_time)
        point_detected.append(point_licks)

        # (2) FULL CASCADE: the bout gate applied to this one 300-sample window, then the point
        # stage. We do NOT call detect_licks here because it resamples (a 2.99 s / 300-sample
        # segment shrinks to 299 samples, below the 300-sample bout window, yielding no window).
        # The segment is already 100 Hz and exactly one bout window, so we gate it directly.
        Xb = torch.tensor(offset_window(seg)[None, None, :], dtype=torch.float32)
        with torch.no_grad():
            is_bout = int(bout(Xb).argmax(1).item()) == 1
        cascade_detected.append(point_licks if is_bout else np.array([]))

    point_res = _aggregate_perlick(point_detected, curated_licks, tol_s)
    cascade_res = _aggregate_perlick(cascade_detected, curated_licks, tol_s)
    total_curated = sum(len(c) for c in curated_licks)

    print(f"\nPer-lick metrics on held-out sessions ({total_curated} labeled licks, "
          f"{args.tol_ms:.0f} ms tolerance):")
    print(f"  POINT STAGE  : P={point_res['precision']:.3f}  R={point_res['recall']:.3f}  "
          f"F1={point_res['f1']:.3f}   (TP={point_res['tp']} FP={point_res['fp']} FN={point_res['fn']})")
    print(f"  FULL CASCADE : P={cascade_res['precision']:.3f}  R={cascade_res['recall']:.3f}  "
          f"F1={cascade_res['f1']:.3f}   (TP={cascade_res['tp']} FP={cascade_res['fp']} FN={cascade_res['fn']})")
    print("\nNote: FULL CASCADE understates recall for licks near a segment edge (isolated 3 s "
          "windows lack the overlapping sliding-window coverage a continuous recording has).")


if __name__ == "__main__":
    main()
