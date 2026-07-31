"""
Render ONE labeled-video window from `find_dlc_windows.py`'s CSV. This is the body of the SLURM
array task: `--row $SLURM_ARRAY_TASK_ID`.

Why it looks like this
----------------------
`deeplabcut.create_labeled_video` can restrict itself to a frame subset, but only on the slow
(matplotlib) path:

    create_labeled_video(config, [video], fastmode=False, save_frames=True,
                         Frames2plot=list(range(start, end)))

so the output video contains only frames `start .. end-1` of the ORIGINAL video, with the original
frame indexing intact. No clip is cut, no inference is re-run: the existing prediction .h5 is
reused, which is what keeps these renders honest as a check on the analysis.

Two consequences drive the rest of this script:

  * DLC finds the predictions by globbing `destfolder` for `<video stem><scorer>*.h5`, and it names
    its output `<video stem><scorer>_labeled.mp4` -- with no window in the name. Two array tasks
    working on the same source video therefore collide on both the temp folder and the output file.
    So each task gets a private staging directory holding SYMLINKS to the video and the .h5, uses
    that as `destfolder`, and moves the finished video out under a window-specific name.

  * The slow path renders with matplotlib, i.e. it is CPU/IO bound. A GPU is not actually needed
    here; the array runs on gpu-a100 only because that is where the DLC environment lives.

Usage:
    python scripts/dlc_label_window.py \
        --config /N/lustre/project/proj-530/dlc_projects/CLiQR_Validation-parkecp-2026-07-27/config.yaml \
        --csv dlc_windows.csv --row 1 --out-dir labeled_windows
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


def read_row(csv_path, row):
    """Return the `row`-th data row (1-based, header excluded) of the windows CSV."""
    with Path(csv_path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{csv_path}: no rows")
    # Prefer an explicit task_id match so the CSV can be reordered without breaking array indices.
    for entry in rows:
        if entry.get("task_id") and int(entry["task_id"]) == row:
            return entry
    if not 1 <= row <= len(rows):
        raise SystemExit(f"row {row} out of range (CSV has {len(rows)} windows)")
    return rows[row - 1]


def stage_inputs(video, h5, workdir):
    """Symlink the video and its predictions into a private directory, return the staged video.

    Symlinks (not copies) because the videos are multi-GB and live on Lustre; DLC only reads them.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    video, h5 = Path(video), Path(h5)
    # The CSV is often written on the laptop, where the .h5 sits next to the recording; on the
    # cluster DLC keeps it beside the video instead. Fall back to that before giving up.
    if not h5.exists() and (video.parent / h5.name).exists():
        h5 = video.parent / h5.name
    staged = None
    for src in (video, h5):
        if not src.exists():
            raise SystemExit(f"missing input: {src}")
        dst = workdir / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        if src.suffix.lower() != ".h5":
            staged = dst
    return staged


def find_output(workdir, before):
    """The .mp4 that appeared in `workdir` during the render."""
    after = {p for p in workdir.glob("*.mp4") if not p.is_symlink()}
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if new:
        return new[-1]
    labeled = sorted(workdir.glob("*labeled*.mp4"))
    return labeled[-1] if labeled else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render a labeled video for one high-confidence window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="DLC project config.yaml")
    parser.add_argument("--csv", required=True, help="CSV from find_dlc_windows.py")
    parser.add_argument("--row", type=int, required=True,
                        help="1-based window / task_id to render (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--out-dir", default="labeled_windows", help="where finished videos go")
    parser.add_argument("--work-dir", default=None,
                        help="staging root (default <out-dir>/.work); use node-local scratch "
                             "($SLURM_TMPDIR) when available -- the temp PNGs are written here")
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--trainingsetindex", type=int, default=0)
    parser.add_argument("--videotype", default="", help="passed through to DLC; '' auto-detects")
    parser.add_argument("--filtered", action="store_true",
                        help="use the *_filtered.h5 predictions instead of the raw ones")
    parser.add_argument("--pcutoff", type=float, default=None,
                        help="likelihood below which a keypoint is not drawn "
                             "(default: the project config's value)")
    parser.add_argument("--dotsize", type=int, default=None)
    parser.add_argument("--trailpoints", type=int, default=0,
                        help="draw this many previous positions as a trail")
    parser.add_argument("--draw-skeleton", action="store_true")
    parser.add_argument("--color-by", default="bodypart", choices=["bodypart", "individual"])
    parser.add_argument("--outputframerate", type=float, default=None,
                        help="playback fps of the rendered window; 120 fps recordings are much "
                             "easier to judge at 15-30")
    parser.add_argument("--keep-frames", action="store_true",
                        help="keep the per-frame PNGs and the staging directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-render even if the output already exists")
    args = parser.parse_args(argv)

    entry = read_row(args.csv, args.row)
    start, end = int(entry["start_frame"]), int(entry["end_frame"])
    label = entry.get("label") or f"{Path(entry['video']).stem}_f{start}-{end}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{label}_labeled.mp4"
    if final.exists() and not args.overwrite:
        print(f"[{args.row}] {final} exists; skipping (use --overwrite)")
        return 0

    work_root = Path(args.work_dir) if args.work_dir else out_dir / ".work"
    workdir = work_root / label
    if workdir.exists():
        shutil.rmtree(workdir)
    staged_video = stage_inputs(entry["video"], entry["h5"], workdir)

    print(
        f"[{args.row}] {label}: frames {start}-{end} ({end - start} frames, "
        f"{entry.get('duration_sec', '?')} s) from {Path(entry['video']).name}",
        flush=True,
    )

    import deeplabcut  # imported late: it is slow and pulls in torch

    kwargs = dict(
        config=args.config,
        videos=[str(staged_video)],
        videotype=args.videotype,
        shuffle=args.shuffle,
        trainingsetindex=args.trainingsetindex,
        filtered=args.filtered,
        # The subset only takes effect on the slow path, and the DLC docstring ties Frames2plot to
        # save_frames, so both are pinned here regardless of what is otherwise convenient.
        fastmode=False,
        save_frames=True,
        Frames2plot=list(range(start, end)),
        destfolder=str(workdir),
        draw_skeleton=args.draw_skeleton,
        trailpoints=args.trailpoints,
        color_by=args.color_by,
    )
    if args.pcutoff is not None:
        kwargs["pcutoff"] = args.pcutoff
    if args.dotsize is not None:
        kwargs["dotsize"] = args.dotsize
    if args.outputframerate is not None:
        kwargs["outputframerate"] = args.outputframerate

    before = {p for p in workdir.glob("*.mp4") if not p.is_symlink()}
    try:
        deeplabcut.create_labeled_video(**kwargs)
    except TypeError as exc:
        # Older DLC releases do not accept every kwarg (pcutoff/dotsize/color_by moved around
        # between 2.2, 2.3 and 3.x). Retry with only the arguments every version has.
        print(f"[{args.row}] retrying without optional kwargs ({exc})", file=sys.stderr)
        for key in ("pcutoff", "dotsize", "color_by"):
            kwargs.pop(key, None)
        deeplabcut.create_labeled_video(**kwargs)

    produced = find_output(workdir, before)
    if produced is None:
        raise SystemExit(f"[{args.row}] DLC produced no video in {workdir}")
    shutil.move(str(produced), str(final))
    print(f"[{args.row}] wrote {final}", flush=True)

    if not args.keep_frames:
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.rmdir(work_root)  # only succeeds once the last task is done
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
