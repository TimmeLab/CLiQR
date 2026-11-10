"""Write a synthetic raw_data_*.h5 recording (same layout the recorder writes).

Useful for trying the analysis GUI, or for load-testing it with long runs,
without real data (the example .h5 files in the repo are Git LFS pointers):

    python scripts/make_synthetic_recording.py "Lickometry Data/Synthetic" --hours 24 --days 2

This creates the cohort folder with a layout_w_controls.csv and one raw file
per day. Add --two-bottle for a 2-bottle choice recording (fluid subgroups,
file attr two_bottle=True) in which fluid 2 is preferred ~3:1. Each sensor gets a noisy baseline with rhythmic (~7 Hz) lick dips
grouped in bouts; control cages get baseline only. A 24 h day with the default
10 sensors is roughly 1 GB on disk.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_analysis import SENSOR_BOARD_MAP  # noqa: E402

DEFAULT_LAYOUT = {1: "SYN-1", 3: "SYN-2", 6: "SYN-3", 7: "SYN-4", 9: "SYN-5",
                  12: "SYN-6", 13: "SYN-7", 15: "SYN-8", 22: "Control1", 24: "Control2"}


def synthetic_trace(n, fs, rng, licking=True, bouts_per_hour=20.0):
    """int64 capacitance trace with lick dips; returns (trace, true lick sample indices)."""
    # Quiet baseline (a few counts of noise plus slow drift), like a real sipper.
    drift = (6 * np.sin(np.arange(n) * 2 * np.pi / (fs * 1800))).astype(np.int64)
    trace = 760 + drift + rng.integers(-2, 3, size=n)
    licks = []
    if licking:
        n_bouts = max(1, int(bouts_per_hour * n / fs / 3600))
        starts = np.sort(rng.integers(0, max(n - int(30 * fs), 1), size=n_bouts))
        period = fs / 7.0
        shape = np.array([-40, -95, -110, -60])  # graded dip, ~36 ms at 110 Hz
        for s in starts:
            for k in range(int(rng.integers(5, 120))):
                i = int(s + k * period)
                if i + 6 >= n:
                    break
                trace[i:i + 4] += shape + rng.integers(-8, 9, size=4)
                licks.append(i)
    return trace.astype(np.int64), np.asarray(licks, dtype=np.int64)


# 2-bottle mode: (bouts per hour, consumed-volume range in mL) per fluid.
TWO_BOTTLE_FLUIDS = {1: (8.0, (0.2, 0.8)), 2: (24.0, (1.0, 2.5))}


def write_raw(path, layout, *, hours=2.0, fs=110.0, start_epoch=None, seed=0,
              lead_s=60.0, tail_s=60.0, comments="synthetic recording", chunk_s=3600.0,
              two_bottle=False):
    """Write one raw file. Sensors start `lead_s` after file start, stop `tail_s` before end.

    Data is written in `chunk_s` blocks so multi-GB files never sit in memory.
    With `two_bottle`, each sensor gets fluid1/fluid2 subgroups (trace + volumes)
    and fluid 2 is licked more and drained more (see TWO_BOTTLE_FLUIDS).
    """
    rng = np.random.default_rng(seed)
    t_start = start_epoch if start_epoch is not None else datetime(2026, 1, 1).timestamp()
    total_s = lead_s + hours * 3600 + tail_s
    n = int(total_s * fs)
    block = int(chunk_s * fs)
    def write_channel(g, is_ctrl, bouts_per_hour, vol_range):
        tds = g.create_dataset("time_data", shape=(n,), dtype="float64", chunks=(500,))
        cds = g.create_dataset("cap_data", shape=(n,), dtype="int64", chunks=(500,))
        for a in range(0, n, block):
            b = min(a + block, n)
            tds[a:b] = t_start + np.arange(a, b) / fs
            cds[a:b] = synthetic_trace(b - a, fs, rng, licking=not is_ctrl,
                                       bouts_per_hour=bouts_per_hour)[0]
        g.create_dataset("start_vol", data=10.0)
        g.create_dataset("stop_vol", data=float(10.0 - (0.02 if is_ctrl else rng.uniform(*vol_range))))

    with h5py.File(path, 'w') as h5f:
        h5f.attrs["two_bottle"] = bool(two_bottle)
        h5f.create_dataset("comments", data=comments)
        for sensor, animal in layout.items():
            board = SENSOR_BOARD_MAP[sensor]
            g = h5f.require_group(board).create_group(f"sensor_{sensor}")
            is_ctrl = str(animal).startswith("Control")
            if two_bottle:
                for fluid, (bph, vol_range) in TWO_BOTTLE_FLUIDS.items():
                    write_channel(g.create_group(f"fluid{fluid}"), is_ctrl, bph, vol_range)
            else:
                write_channel(g, is_ctrl, 20.0, (0.5, 2.5))
            g.create_dataset("start_time", data=t_start + lead_s)
            g.create_dataset("stop_time", data=t_start + lead_s + hours * 3600)
            if not is_ctrl:
                g.create_dataset("weight", data=float(rng.uniform(20, 30)))
    return path


def write_cohort(cohort_dir, *, days=2, hours=2.0, fs=110.0, layout=None, seed=0,
                 two_bottle=False):
    layout = layout or DEFAULT_LAYOUT
    os.makedirs(cohort_dir, exist_ok=True)
    with open(os.path.join(cohort_dir, "layout_w_controls.csv"), "w") as f:
        for sensor, animal in layout.items():
            f.write(f"{sensor},{animal}\n")
    paths = []
    day0 = datetime(2026, 1, 20, 11, 0, 0)
    for d in range(days):
        when = day0 + timedelta(days=d)
        p = os.path.join(cohort_dir, f"raw_data_{when:%Y-%m-%d_%H-%M-%S}.h5")
        write_raw(p, layout, hours=hours, fs=fs, start_epoch=when.timestamp(), seed=seed + d,
                  two_bottle=two_bottle)
        paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cohort_dir")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--fs", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--two-bottle", action="store_true", help="2-bottle choice recording")
    args = ap.parse_args()
    for p in write_cohort(args.cohort_dir, days=args.days, hours=args.hours, fs=args.fs,
                          seed=args.seed, two_bottle=args.two_bottle):
        print(f"wrote {p} ({os.path.getsize(p) / 1e9:.2f} GB)")
