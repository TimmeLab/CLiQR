"""
Find the stretches of an already-analyzed video where the mouse is actually in frame, and write
them out as one-row-per-window jobs for `dlc_label_window.py` / the SLURM array.

Motivation
----------
A session video is ~45 min long but the animal is only at the sipper for a tiny fraction of it: in
`raw_data_2026-07-21_12-59-50_cfr` the `nose` keypoint clears likelihood 0.8 in ~2% of frames.
Rendering a labeled video of the whole recording to eyeball how DeepLabCut did is therefore mostly
a waste of GPU-hours and of your time. Instead we use the pose predictions themselves as an
"is the mouse here?" detector: contiguous runs of high-confidence `nose` frames are the only parts
worth looking at.

This script does NOT re-run inference and does NOT cut any video. It reads the DLC prediction .h5
that `analyze_videos` already wrote, thresholds one bodypart's likelihood, and emits frame ranges.
The actual subsetting happens later, inside `create_labeled_video(..., fastmode=False,
Frames2plot=range(start, end))`, so frame numbers stay in the ORIGINAL video's index space and
remain relatable to the capacitance trace.

How a window is built (in order)
--------------------------------
1. `mask = likelihood >= --pcutoff` over the chosen `--bodypart`.
2. Runs of `True` separated by a gap of <= `--merge-gap` frames are merged. Raw runs are extremely
   choppy (median 3 frames) because the detector flickers while the animal moves; without merging
   you get hundreds of useless sub-100 ms windows.
3. Windows shorter than `--min-frames` are dropped (isolated false positives), as are windows
   holding fewer than `--min-confident` confident frames in total. The second test matters: with a
   1 s merge gap, two single-frame detections a second apart otherwise become a 120-frame "window"
   containing almost no evidence that the animal was there at all.
4. `--pad` frames of context are added on each side, then any windows that now overlap are merged
   again.
5. Windows longer than `--max-frames` are split into equal-ish chunks, so no single array task has
   to render an unbounded number of frames.

Frame ranges are half-open, `[start_frame, end_frame)`, matching
`Frames2plot=list(range(start_frame, end_frame))`.

Usage (from the repository root):
    # every analyzed video in a directory (the normal case)
    python dlc_integration/find_dlc_windows.py "Lickometry Data/ACG-26-3" --csv dlc_windows.csv

    # or one specific prediction file
    python dlc_integration/find_dlc_windows.py \
        "Lickometry Data/ACG-26-3/raw_data_2026-07-21_12-59-50_cfrDLC_Resnet50_CLiQR_ValidationJul27shuffle1_snapshot_best-90.h5" \
        --csv dlc_windows.csv

Inputs default to the current directory, and any directory given is scanned for DLC prediction
.h5 files. Multiple .h5 files and/or directories may be given; every window from every file lands
in the same CSV, numbered by `task_id`, which is exactly the SLURM array index -- so one array
covers all videos.

Re-running with `--append` adds the new windows to an existing CSV instead of replacing it,
continuing the `task_id` numbering and skipping windows that are already in the file.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# DLC names its output `<video stem>DLC_<network>_<task><date>shuffle<n>_<snapshot>.h5`, plus an
# optional `_filtered` / `_el` / `_bx` suffix. Everything before the `DLC_`/`DeepCut_` marker is the
# video stem; everything from the marker on is the "scorer" string DLC uses internally.
_SCORER_RE = re.compile(r"(DLC_|DeepCut_|DLCnet)")

CSV_FIELDS = [
    "task_id",
    "label",
    "video",
    "h5",
    "scorer",
    "bodypart",
    "start_frame",
    "end_frame",
    "n_frames",
    "start_sec",
    "end_sec",
    "duration_sec",
    "frac_above",
    "mean_likelihood",
]


# --------------------------------------------------------------------------------------------
# reading DLC predictions
# --------------------------------------------------------------------------------------------
def load_dlc_h5(path):
    """Return (scorer, bodyparts, coords) from a DLC prediction file.

    `coords[bodypart]` is a dict of `x`, `y`, `likelihood` float arrays, one entry per frame.

    Uses pandas/pytables when available; otherwise falls back to reading the pytables `table`
    dataset directly with h5py and unpickling the column MultiIndex out of the group attributes.
    The fallback exists so this script runs in the plain `cliqr-gui` environment (no pytables)
    as well as inside the DLC conda env on the cluster.
    """
    try:
        import pandas as pd

        df = pd.read_hdf(path)
        scorer = df.columns.get_level_values(0)[0]
        bodyparts = list(dict.fromkeys(df.columns.get_level_values(1)))
        coords = {
            bp: {
                coord: np.asarray(df[(scorer, bp, coord)].values, dtype=float)
                for coord in ("x", "y", "likelihood")
            }
            for bp in bodyparts
        }
        return scorer, bodyparts, coords
    except ImportError:
        pass  # no pytables -> h5py fallback below

    import h5py

    with h5py.File(path, "r") as fh:
        # DLC always writes a single group; find it rather than hard-coding "df_with_missing".
        group_names = [k for k in fh.keys() if isinstance(fh[k], h5py.Group) and "table" in fh[k]]
        if not group_names:
            raise ValueError(f"{path}: no pytables frame group found")
        grp = fh[group_names[0]]
        table = grp["table"]
        blocks = sorted(n for n in table.dtype.names if n.startswith("values_block_"))
        if len(blocks) != 1:
            raise ValueError(
                f"{path}: {len(blocks)} value blocks; install pytables "
                "(`pip install tables`) so pandas can read this file"
            )
        values = np.asarray(table[blocks[0]], dtype=float)
        columns = pickle.loads(bytes(grp.attrs["non_index_axes"]))[0][1]

    if values.shape[1] != len(columns):
        raise ValueError(f"{path}: {values.shape[1]} columns of data vs {len(columns)} names")

    scorer = columns[0][0]
    bodyparts = list(dict.fromkeys(col[1] for col in columns))
    coords = {}
    for i, (_scorer, bp, coord) in enumerate(columns):
        if coord in ("x", "y", "likelihood"):
            coords.setdefault(bp, {})[coord] = values[:, i]
    return scorer, bodyparts, coords


def guess_video(h5_path):
    """Map `<stem>DLC_<scorer>.h5` back to the video it was produced from.

    Returns the first existing sibling video, or the `.mp4` guess if none exists (so the CSV is
    still usable when the videos live somewhere else on the cluster; fix the paths with --video).
    """
    h5_path = Path(h5_path)
    match = _SCORER_RE.search(h5_path.stem)
    stem = h5_path.stem[: match.start()] if match else h5_path.stem
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        candidate = h5_path.with_name(stem + ext)
        if candidate.exists():
            return candidate
    return h5_path.with_name(stem + ".mp4")


def probe_fps(video):
    """True constant frame rate of `video` via ffprobe, or None.

    Deliberately reads `avg_frame_rate`, not `r_frame_rate`: our recordings report a rounded 120
    for r_frame_rate while the real rate is 120.0048, and that rounding is enough to drift the
    reported window times by seconds late in a session.
    """
    video = Path(video)
    if not video.exists():
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if "/" not in out:
        return None
    num, den = out.split("/")
    try:
        num, den = float(num), float(den)
    except ValueError:
        return None
    return num / den if den else None


# --------------------------------------------------------------------------------------------
# sipper geometry and tongue rhythm
# --------------------------------------------------------------------------------------------
# The four sipper_* keypoints trace the curve of the sipper tip. Anatomical order matters: the
# polyline models the physical tube, so the points must be joined top -> bottom, never re-sorted
# by coordinate (the sipper sits diagonally in most recordings).
SIPPER_BODYPARTS = ("sipper_top", "sipper_midtop", "sipper_midbottom", "sipper_bottom")


def point_to_polyline_distance(px, py, points):
    """Distance from each (px, py) to the nearest point ON the polyline through `points`.

    Segment distance, not nearest-vertex distance: adjacent sipper keypoints are ~40-50 px apart,
    so a nose resting midway between two of them reads as up to ~25 px farther from the sipper
    than it really is if you only measure to the vertices.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    if len(points) < 2:
        raise ValueError("need at least two points to form a polyline")

    best = None
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
        vx, vy = bx - ax, by - ay
        length_sq = vx * vx + vy * vy
        if length_sq == 0:
            # Degenerate segment (two identical keypoint medians): fall back to the endpoint.
            d = np.hypot(px - ax, py - ay)
        else:
            t = np.clip(((px - ax) * vx + (py - ay) * vy) / length_sq, 0.0, 1.0)
            d = np.hypot(px - (ax + t * vx), py - (ay + t * vy))
        best = d if best is None else np.minimum(best, d)
    return best


def sipper_anchor(coords, pcutoff=0.6, min_frames=100):
    """Static per-session sipper position: (points, arc_length_px).

    The sipper does not move within a recording -- across the ten analyzed ACG-26-3 sessions each
    keypoint's position IQR over confident frames is 0.5-3.5 px -- so one median per keypoint is
    both more robust and cheaper than tracking it frame by frame, and it survives the 1-23% of
    frames where a keypoint drops below `pcutoff`.

    `arc_length` is the length of the polyline through the surviving keypoints. It is the natural
    scale of the sipper in this recording (140-165 px across our sessions, varying with camera
    distance), which is what makes a proximity threshold expressed as a fraction of it portable
    between sessions.
    """
    points = []
    for bp in SIPPER_BODYPARTS:
        if bp not in coords:
            continue
        confident = coords[bp]["likelihood"] >= pcutoff
        if confident.sum() < min_frames:
            continue
        points.append(
            (float(np.median(coords[bp]["x"][confident])),
             float(np.median(coords[bp]["y"][confident])))
        )
    if len(points) < 2:
        raise ValueError(
            f"no usable sipper keypoints: fewer than two of {list(SIPPER_BODYPARTS)} have "
            f"{min_frames}+ frames at likelihood >= {pcutoff}"
        )
    arc_length = sum(
        float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(points[:-1], points[1:])
    )
    return points, arc_length


def tongue_upcross_rate(likelihood, start, end, pcutoff, fps):
    """Upward crossings of `pcutoff` per second within [start, end).

    The tongue is only visible at the top of each lick, so during drinking its likelihood PULSES
    rather than staying high: in the analyzed sessions, drinking stretches cross 0.6 upward 3.4-7.8
    times per second (a 7.9-9.1 Hz rhythm, part of it below the cutoff) while non-drinking
    stretches near the sipper cross it 0-0.4 times per second. Counting crossings separates the two
    without an FFT.

    A window that opens already above the cutoff contributes no crossing for that leading run. That
    undercounts by at most one crossing, which is immaterial against a 3+/s threshold.
    """
    duration = (end - start) / fps
    if duration <= 0:
        return 0.0
    confident = np.asarray(likelihood[start:end]) >= pcutoff
    crossings = int(np.sum(np.diff(confident.astype(np.int8)) == 1))
    return crossings / duration


# --------------------------------------------------------------------------------------------
# window construction
# --------------------------------------------------------------------------------------------
def runs_of_true(mask):
    """Half-open [start, end) index pairs for each contiguous True run in a boolean array."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends))


def merge_close(windows, gap):
    """Merge windows separated by <= `gap` frames (gap=0 merges only touching/overlapping ones)."""
    merged = []
    for start, end in sorted(windows):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def split_long(windows, max_frames):
    """Split any window longer than `max_frames` into equal-ish consecutive chunks."""
    if not max_frames or max_frames <= 0:
        return list(windows)
    out = []
    for start, end in windows:
        length = end - start
        if length <= max_frames:
            out.append((start, end))
            continue
        n_chunks = int(np.ceil(length / max_frames))
        edges = np.linspace(start, end, n_chunks + 1).round().astype(int)
        out.extend((int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a)
    return out


def find_windows(mask, merge_gap, min_frames, min_confident, pad, max_frames):
    """Full window pipeline over a per-frame "the animal is here" mask.

    Takes the mask rather than a likelihood array because the gate is no longer a single
    threshold: a frame counts when the bodypart is confidently detected AND close to the sipper.
    Callers build the mask; everything below is unchanged.

    Returns half-open [start, end) frame ranges.
    """
    mask = np.asarray(mask, dtype=bool)
    windows = merge_close(runs_of_true(mask), merge_gap)
    windows = [
        (s, e) for s, e in windows
        if e - s >= min_frames and mask[s:e].sum() >= min_confident
    ]
    if pad:
        n = mask.size
        windows = [(max(0, s - pad), min(n, e + pad)) for s, e in windows]
        windows = merge_close(windows, 0)
    return split_long(windows, max_frames)


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------
def collect_h5_paths(inputs, recursive=False):
    """Expand a mix of .h5 files and directories into a sorted, de-duplicated file list.

    Directories are scanned for DLC prediction files (`*DLC_*.h5`). If a directory holds .h5 files
    but none carry the scorer marker, every .h5 in it is taken instead -- the directories we point
    this at contain nothing but the predictions we care about, and silently finding zero files is
    worse than trying to read one.
    """
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(p.rglob("*.h5") if recursive else p.glob("*.h5"))
            predictions = [q for q in found if _SCORER_RE.search(q.stem)]
            paths.extend(predictions or found)
        else:
            paths.append(p)
    seen, unique = set(), []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def window_key(row):
    """Identity of a window for de-duplication across runs: which video, which frames."""
    return (str(row["video"]), int(row["start_frame"]), int(row["end_frame"]))


def read_existing_csv(path):
    """Rows already in `path` (empty list if it doesn't exist), for --append."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(CSV_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{path}: existing CSV is missing column(s) {sorted(missing)}; "
                "it was written by a different version -- drop --append or use a new --csv"
            )
        return [{k: row[k] for k in CSV_FIELDS} for row in reader]


def rows_for_file(h5_path, args, task_id_start):
    scorer, bodyparts, coords = load_dlc_h5(h5_path)
    if args.bodypart not in coords:
        raise ValueError(
            f"bodypart '{args.bodypart}' not in this file. Available: {bodyparts}"
        )
    like = coords[args.bodypart]["likelihood"]

    video = Path(args.video) if args.video else guess_video(h5_path)
    if args.video_dir:
        # The CSV is usually written on the laptop but consumed on the cluster, where the videos
        # live in the DLC project's videos/ directory rather than next to the recordings.
        video = Path(args.video_dir) / video.name
    fps = args.fps or probe_fps(video) or 120.0

    windows = find_windows(
        like >= args.pcutoff,
        merge_gap=args.merge_gap,
        min_frames=args.min_frames,
        min_confident=args.min_confident,
        pad=args.pad,
        max_frames=args.max_frames,
    )

    rows = []
    for i, (start, end) in enumerate(windows):
        chunk = like[start:end]
        rows.append(
            {
                "task_id": task_id_start + i,
                "label": f"{video.stem}_w{i:03d}_f{start}-{end}",
                "video": str(video),
                "h5": str(h5_path),
                "scorer": scorer,
                "bodypart": args.bodypart,
                "start_frame": start,
                "end_frame": end,
                "n_frames": end - start,
                "start_sec": round(start / fps, 3),
                "end_sec": round(end / fps, 3),
                "duration_sec": round((end - start) / fps, 3),
                "frac_above": round(float(np.mean(chunk >= args.pcutoff)), 4),
                "mean_likelihood": round(float(np.mean(chunk)), 4),
            }
        )
    return rows, dict(
        h5=str(h5_path), video=str(video), fps=fps, n_frames=int(like.size),
        frames_above=int(np.sum(like >= args.pcutoff)), n_windows=len(windows),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract high-confidence frame windows from DLC prediction .h5 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("h5", nargs="*", default=["."],
                        help="director(ies) to scan for DLC prediction .h5 files, or individual "
                             ".h5 files; every window found goes into the one CSV")
    parser.add_argument("--recursive", action="store_true",
                        help="also scan subdirectories of any directory given")
    parser.add_argument("--csv", default="dlc_windows.csv", help="output CSV path")
    parser.add_argument("--append", action="store_true",
                        help="add to an existing --csv instead of replacing it: task_id numbering "
                             "continues from the last row and windows already present are skipped")
    parser.add_argument("--bodypart", default="nose",
                        help="bodypart whose likelihood decides 'mouse is in frame'")
    parser.add_argument("--pcutoff", type=float, default=0.8,
                        help="minimum likelihood for a frame to count as confident")
    parser.add_argument("--merge-gap", type=int, default=120,
                        help="merge confident runs separated by at most this many frames "
                             "(120 = 1 s at 120 fps; the raw runs are very choppy)")
    parser.add_argument("--min-frames", type=int, default=30,
                        help="discard merged windows shorter than this many frames")
    parser.add_argument("--min-confident", type=int, default=15,
                        help="discard merged windows containing fewer than this many frames that "
                             "actually clear --pcutoff (guards against gap-merging two flickers "
                             "into a long, near-empty window)")
    parser.add_argument("--pad", type=int, default=60,
                        help="frames of context added to each side of a window")
    parser.add_argument("--max-frames", type=int, default=3600,
                        help="split windows longer than this (0 disables splitting)")
    parser.add_argument("--fps", type=float, default=None,
                        help="frame rate for the *_sec columns; default probes the video with "
                             "ffprobe (avg_frame_rate) and falls back to 120.0")
    parser.add_argument("--video", default=None,
                        help="explicit video path (only valid with a single .h5)")
    parser.add_argument("--video-dir", default=None,
                        help="rewrite every video path into this directory, keeping the filename "
                             "(e.g. the DLC project's videos/ dir on the cluster)")
    parser.add_argument("--summary-json", default=None,
                        help="also write a per-file summary here")
    args = parser.parse_args(argv)

    h5_paths = collect_h5_paths(args.h5, recursive=args.recursive)
    if not h5_paths:
        raise SystemExit(f"no .h5 files found in: {', '.join(map(str, args.h5))}")
    if args.video and len(h5_paths) > 1:
        raise SystemExit("--video only makes sense with a single .h5")

    out = Path(args.csv)
    existing_rows = read_existing_csv(out) if args.append else []
    existing_keys = {window_key(r) for r in existing_rows}
    next_task_id = max((int(r["task_id"]) for r in existing_rows), default=0) + 1

    all_rows, summaries, n_skipped = [], [], 0
    for h5_path in h5_paths:
        try:
            rows, summary = rows_for_file(h5_path, args, task_id_start=next_task_id)
        except Exception as exc:
            # One unreadable / non-DLC file shouldn't kill a whole-directory run; with a single
            # explicit file there is nothing to salvage, so fail loudly instead.
            msg = str(exc) if str(h5_path) in str(exc) else f"{h5_path}: {exc}"
            if len(h5_paths) == 1:
                raise SystemExit(msg)
            print(f"skipping {msg}", file=sys.stderr)
            continue
        if existing_keys:
            kept = [r for r in rows if window_key(r) not in existing_keys]
            n_skipped += len(rows) - len(kept)
            # task_id is the SLURM array index, so it must stay gapless after dropping duplicates.
            for i, row in enumerate(kept):
                row["task_id"] = next_task_id + i
            rows = kept
        existing_keys.update(window_key(r) for r in rows)
        next_task_id += len(rows)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"{Path(h5_path).name}: {summary['frames_above']}/{summary['n_frames']} frames "
            f">= {args.pcutoff} on '{args.bodypart}' -> {summary['n_windows']} windows",
            file=sys.stderr,
        )

    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    # Always rewrite the whole file, existing rows included: one header, one contiguous task_id
    # range, and re-running with --append twice can't produce a half-written CSV.
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(all_rows)

    n_total = len(existing_rows) + len(all_rows)
    total_sec = sum(float(r["duration_sec"]) for r in existing_rows) \
        + sum(r["duration_sec"] for r in all_rows)
    added = f"added {len(all_rows)} windows to {len(existing_rows)} already in" if existing_rows \
        else f"wrote {len(all_rows)} windows from {len(h5_paths)} file(s) to"
    skipped = f" (skipped {n_skipped} duplicates)" if n_skipped else ""
    print(
        f"{added} {out}{skipped}; {n_total} windows total ({total_sec:.1f} s of video)",
        file=sys.stderr,
    )
    if n_total:
        print(
            f"submit with: sbatch --array=1-{n_total} "
            "dlc_integration/slurm_dlc_label_windows.sbatch",
            file=sys.stderr,
        )

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
