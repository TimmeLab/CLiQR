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

  * Every name DLC uses comes from the video's stem -- output `<stem><scorer>_labeled.mp4`, scratch
    folder `temp-<stem>`, predictions `<stem><scorer>.h5` -- and none of them mention a frame
    range. So all N windows of one recording look like the same job to DLC: the first task renders,
    and every later task finds the output already there and skips with "Labeled video already
    created". Symlinking the video per window does not help, because DLC resolves the path first
    and every link collapses back to the one recording. Each task therefore HARDLINKS the video
    (and its predictions) into a private staging directory under a window-specific name -- a real
    path, no bytes copied -- which gives the window its own stem and so its own everything. See
    `stage_inputs`.

  * The slow path renders with matplotlib, i.e. it is CPU/IO bound. A GPU is not actually needed
    here; the array runs on gpu-a100 only because that is where the DLC environment lives.

Usage:
    python dlc_integration/dlc_label_window.py \
        --config /N/lustre/project/proj-530/dlc_projects/CLiQR_Validation-parkecp-2026-07-27/config.yaml \
        --csv dlc_windows.csv --row 1 --out-dir labeled_windows
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
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


def stage_inputs(video, h5, stagedir, label):
    """HARDLINK the video and its analysis outputs into `stagedir` under a per-window name.

    Everything DLC does is keyed off the video's stem: the output is
    `<stem><scorer>_labeled.mp4`, the scratch folder is `temp-<stem>`, and -- the part that bites
    -- it short-circuits with "Labeled video already created" when that output already exists. The
    stem carries no frame range, so all 18 windows of one recording are one and the same job as far
    as DLC is concerned: the first task renders, every later task finds the file and skips.

    Symlinks do not fix this, because DLC resolves the video path before deriving those names --
    every link collapses back to the one real recording. Hardlinks do: a hardlink IS a real path
    with its own name and no copied bytes, so `<label>.mp4` gives this window a private stem, a
    private output name and a private scratch folder.

    The predictions are hardlinked alongside under the matching renamed prefix, since DLC now looks
    for `<label><scorer>.h5`. `<h5 stem>*` also matches `<h5 stem>_labeled.mp4` from an earlier
    render, which must NOT come along -- find_output would hand that stale video back as this
    window's result -- so only .h5/.pickle are taken. The `_meta.pickle` is required: without it
    DLC aborts with "No metadata found in ... for video ... and scorer ...".

    Returns the staged video path.
    """
    video, h5 = Path(video), Path(h5)
    # The CSV is often written on the laptop, where the .h5 sits next to the recording; on the
    # cluster DLC keeps it beside the video instead. Fall back to that before giving up.
    if not h5.exists() and (video.parent / h5.name).exists():
        h5 = video.parent / h5.name
    if not video.exists():
        raise SystemExit(f"missing input: {video}")
    if not h5.exists():
        raise SystemExit(f"missing input: {h5}")
    if not h5.name.startswith(video.stem):
        raise SystemExit(
            f"{h5.name} does not start with the video stem {video.stem!r}; cannot rename it "
            "consistently with the staged video"
        )

    sidecars = sorted(
        p for p in h5.parent.glob(h5.stem + "*")
        if p.suffix.lower() in (".h5", ".pickle")
    )
    if not any(p.name.endswith("_meta.pickle") for p in sidecars):
        print(
            f"warning: no {h5.stem}_meta.pickle next to {h5}; "
            "create_labeled_video will not be able to load the video metadata",
            file=sys.stderr,
        )

    stagedir.mkdir(parents=True, exist_ok=True)
    staged = None
    for src in [video] + sidecars:
        # `<video stem>` -> `<label>`, keeping the scorer suffix DLC matches on.
        dst = stagedir / (label + src.name[len(video.stem):])
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError as exc:
            raise SystemExit(
                f"cannot hardlink {src} -> {dst} ({exc}). The staging directory must be on the "
                "same filesystem as the recording; pass --stage-dir with a path on that "
                "filesystem (a symlink will not do -- DLC resolves it and the windows collide)."
            ) from exc
        if src == video:
            staged = dst
    return staged


def real_mp4s(directory):
    """Real .mp4 files in `directory`, never symlinks.

    Symlinks are excluded everywhere in this module: the staged inputs are symlinks, and moving one
    would produce an output that merely points at a shared file instead of owning its own frames.
    """
    return {p for p in Path(directory).glob("*.mp4") if not p.is_symlink()}


def find_output(workdir, before, started_at):
    """The .mp4 this render produced in `workdir`, or None."""
    new = sorted(real_mp4s(workdir) - before, key=lambda p: p.stat().st_mtime)
    if new:
        return new[-1]
    # Fall back to any real, freshly written labeled video; `started_at` keeps a leftover from a
    # previous attempt out of the result.
    fresh = [
        p for p in real_mp4s(workdir)
        if "labeled" in p.name and p.stat().st_mtime >= started_at
    ]
    return max(fresh, key=lambda p: p.stat().st_mtime) if fresh else None


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
    parser.add_argument("--stage-dir", default=None,
                        help="staging root (default .dlc_label_staging beside the recording). "
                             "MUST be on the same filesystem as the recording: the video is "
                             "hardlinked in, which is what gives each window its own DLC output "
                             "name. Node-local scratch will not work.")
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

    # The staging dir must share a filesystem with the recording (hardlinks), so it defaults to a
    # hidden directory beside it rather than to node-local scratch.
    stage_root = (
        Path(args.stage_dir) if args.stage_dir
        else Path(entry["video"]).parent / ".dlc_label_staging"
    )
    workdir = stage_root / label
    if workdir.exists():
        shutil.rmtree(workdir)
    staged_video = stage_inputs(entry["video"], entry["h5"], workdir, label)

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

    before = real_mp4s(workdir)
    started_at = time.time()
    try:
        deeplabcut.create_labeled_video(**kwargs)
    except TypeError as exc:
        # Older DLC releases do not accept every kwarg (pcutoff/dotsize/color_by moved around
        # between 2.2, 2.3 and 3.x). Retry with only the arguments every version has.
        print(f"[{args.row}] retrying without optional kwargs ({exc})", file=sys.stderr)
        for key in ("pcutoff", "dotsize", "color_by"):
            kwargs.pop(key, None)
        deeplabcut.create_labeled_video(**kwargs)

    produced = find_output(workdir, before, started_at)
    if produced is None:
        # Listing the staging dir is the fastest way to tell "DLC wrote nothing" apart from
        # "DLC wrote somewhere other than destfolder".
        listing = "\n  ".join(sorted(p.name for p in workdir.iterdir())) or "(empty)"
        raise SystemExit(
            f"[{args.row}] DLC produced no video in {workdir}. Contents:\n  {listing}"
        )
    if final.is_symlink():
        final.unlink()
    shutil.move(str(produced), str(final))
    if final.is_symlink():
        raise SystemExit(f"[{args.row}] refusing a symlinked result: {final} -> {os.readlink(final)}")
    print(f"[{args.row}] wrote {final} ({final.stat().st_size / 1e6:.1f} MB)", flush=True)

    if not args.keep_frames:
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.rmdir(stage_root)  # only succeeds once the last task is done
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
