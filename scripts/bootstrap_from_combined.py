"""
Build an initial ML training set from a combined results HDF5 file.

This pulls each animal/session's capacitance trace and its existing threshold-detector lick times
out of a `results_combined_*.h5` file, runs `bootstrap_segments` on every (animal, session) to
produce category-balanced 3 s candidate windows seeded by those threshold detections, and writes
all the segments into a single training HDF5 that the Solara labeler can open for curation.

The combined file's layout (produced by data_analysis.save_filtered_data) is:
    <animal>/<session_index>/{cap_data, time_data, lick_times, lick_indices, ...}
where `lick_times` is the basic_algorithm (threshold) detector's output — exactly the seed
`bootstrap_segments` expects. `time_data` is already zero-based (filter_data subtracts the start
time), so no epoch handling is needed here.

Control channels have no animal, so their "lick_times" are noise/false positives; seeding those
as licks would create bad labels to hand-correct. They are skipped by default (--include-controls
to override).

Usage (from the repository root, cliqr-gui environment):
    python scripts/bootstrap_from_combined.py \
        "Lickometry Data/results_combined_ACG-26-3_2026-07-22_23_24_27_28.h5" \
        --out ml_training_ACG-26-3_bootstrap.h5 \
        --per-session 100
"""

import argparse
import os

import h5py
import numpy as np

# The bootstrap sampler and the training-set writer live in the ml_detection package.
from ml_detection.bootstrap import bootstrap_segments
from ml_detection.dataset import save_training_h5


def is_control(animal_name):
    """Control cages (no animal) are named like 'Control1'/'Control2'. Their trace is noise."""
    return animal_name.lower().startswith("control")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap an ML training set from a results_combined_*.h5 file."
    )
    parser.add_argument("h5_path", help="Path to the results_combined_*.h5 file.")
    parser.add_argument(
        "--out", default=None,
        help="Output training HDF5 path. Default: <input stem>_bootstrap.h5 in the cwd.",
    )
    parser.add_argument(
        "--per-session", type=int, default=100,
        help="How many candidate 3 s windows to sample per (animal, session). Default 100. "
             "Total labeling load = per-session x number of animal-sessions (printed at the end).",
    )
    parser.add_argument(
        "--include-controls", action="store_true",
        help="Also bootstrap from Control channels (off by default; their licks are noise).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base RNG seed for reproducible sampling. Each session offsets it so sessions differ.",
    )
    parser.add_argument(
        "--min-separation", type=int, default=300,
        help="Minimum spacing (in 100 Hz samples) between accepted window starts, to cut "
             "near-duplicate segments from overlapping windows. Default 300 = fully "
             "non-overlapping 3 s windows. Use 0 to allow overlap.",
    )
    args = parser.parse_args()

    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.h5_path))[0]
        args.out = f"{stem}_bootstrap.h5"

    # -----------------------------------------------------------------------------------------
    # Walk every (animal, session) and bootstrap candidate segments
    # -----------------------------------------------------------------------------------------
    # We accumulate the per-session outputs and concatenate at the end into one training set.
    all_samples = []     # each: [n_i, 300] float32
    all_times = []       # each: [n_i, 300] float64  (within-segment relative time)
    all_lick_idx = []    # flat list of per-segment lick-index arrays
    all_labels = []      # each: [n_i] int64
    all_source = []      # flat list of "animal/session" strings, one per segment (provenance)

    skipped = []         # (animal, session, reason) for the summary

    with h5py.File(args.h5_path, "r") as f:
        animals = list(f.keys())
        for animal in animals:
            if is_control(animal) and not args.include_controls:
                skipped.append((animal, "*", "control (use --include-controls to keep)"))
                continue
            animal_group = f[animal]
            for session in animal_group.keys():
                session_group = animal_group[session]
                # A well-formed session has cap_data, time_data, and the seed lick_times.
                if not all(k in session_group for k in ("cap_data", "time_data", "lick_times")):
                    skipped.append((animal, session, "missing cap/time/lick_times"))
                    continue

                time_s = np.asarray(session_group["time_data"][()], dtype=float)
                cap = np.asarray(session_group["cap_data"][()], dtype=float)
                seed_lick_times = np.asarray(session_group["lick_times"][()], dtype=float)

                # Give each session a distinct but reproducible seed so their random windows differ.
                session_seed = args.seed + hash((animal, session)) % 10_000

                try:
                    training = bootstrap_segments(
                        time_s, cap, seed_lick_times,
                        n_samples=args.per_session, seed=session_seed,
                        min_separation_samples=args.min_separation,
                    )
                except ValueError as exc:
                    # Raised when a session is shorter than one 3 s window after resampling.
                    skipped.append((animal, session, f"bootstrap failed: {exc}"))
                    continue

                n_segments = len(training["samples"])
                if n_segments == 0:
                    skipped.append((animal, session, "0 segments produced"))
                    continue

                all_samples.append(training["samples"])
                all_times.append(training["t"])
                all_lick_idx.extend(training["lick_idx"])
                all_labels.append(training["labels_bout"])
                all_source.extend([f"{animal}/{session}"] * n_segments)

    if not all_samples:
        raise SystemExit("No segments were produced from any session. Nothing to save.")

    # -----------------------------------------------------------------------------------------
    # Concatenate into one training set and save
    # -----------------------------------------------------------------------------------------
    samples = np.concatenate(all_samples, axis=0)
    times = np.concatenate(all_times, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    meta = {
        "source_file": os.path.basename(args.h5_path),
        "per_session": str(args.per_session),
        "include_controls": str(args.include_controls),
    }
    save_training_h5(args.out, samples, times, all_lick_idx, labels, meta)

    # Store per-segment provenance ("animal/session") so a later session-level train/val split can
    # group segments by their originating session. save_training_h5 doesn't carry per-segment
    # strings, so we append a variable-length-string dataset directly.
    with h5py.File(args.out, "a") as out_file:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        out_file.create_dataset(
            "source", data=np.array(all_source, dtype=object), dtype=string_dtype
        )

    # -----------------------------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------------------------
    total = len(samples)
    n_positive = int(np.sum(labels == 1))
    print(f"\nSaved {total} candidate segments to {args.out}")
    print(f"  central-lick positives : {n_positive}  ({n_positive / total:.1%})")
    print(f"  no-central-lick        : {total - n_positive}")

    # Per-source counts, so the labeling load per session is visible.
    sources, counts = np.unique(np.array(all_source), return_counts=True)
    print(f"  from {len(sources)} animal-sessions:")
    for source, count in zip(sources, counts):
        print(f"    {source:>16} : {count}")

    if skipped:
        print("\nSkipped:")
        for animal, session, reason in skipped:
            print(f"    {animal}/{session}: {reason}")

    print(f"\nNext: curate in the labeler ->  panel serve ml_detection/labeler/app.py --show  "
          f"(load {args.out})")


if __name__ == "__main__":
    main()
