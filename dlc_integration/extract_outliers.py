"""
Run `deeplabcut.extract_outlier_frames` over every analyzed video in a DLC project's videos/
directory, so the next round of manual labeling can target the frames the network is worst at.

Why a wrapper instead of calling DLC directly
---------------------------------------------
`extract_outlier_frames` takes a list of videos, but in practice you cannot hand it the whole
directory and walk away:

  * It raises on any video that has not been analyzed for the requested shuffle. One unanalyzed
    file therefore aborts the whole batch. Here each video is a separate call inside try/except,
    and videos with no predictions are skipped up front with a message.
  * It is interactive by default ("Do you want to proceed with extracting ... frames?"), which
    hangs a SLURM job forever. `automatic=True` is the default here; --interactive opts back in.
  * How many frames it picks comes from `numframes2pick` in config.yaml, not from an argument.
    --numframes2pick edits the config for the duration of the run and restores it afterwards.
  * The outlier criterion is one algorithm per call, but 'jump' and 'uncertain' find different
    failures -- 'jump' catches the keypoint teleporting between frames, 'uncertain' catches frames
    the network simply is not confident about. Both run by default, one after the other.

Extracted frames land in `<project>/labeled-data/<video stem>/` alongside a `machinelabels-iter<N>.h5`
of the current predictions, which is what makes refinement fast: run `refine_labels` and you are
correcting the network's guesses rather than clicking from scratch. Nothing here modifies the
training set; `merge_datasets` + `create_training_dataset` is still a separate, deliberate step.

Usage:
    python dlc_integration/extract_outliers.py \
        --config /N/lustre/project/proj-530/dlc_projects/CLiQR_Validation-parkecp-2026-07-27/config.yaml \
        --numframes2pick 20

    # see what would run, and which videos lack predictions, without touching anything
    python dlc_integration/extract_outliers.py --config .../config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Same marker as find_dlc_windows.py: everything before it in an .h5 name is the video stem.
_SCORER_RE = re.compile(r"(DLC_|DeepCut_|DLCnet)")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def read_config(config):
    """Load the project config.yaml as a dict (via DLC so its custom loader rules apply)."""
    from deeplabcut.utils import auxiliaryfunctions

    return auxiliaryfunctions.read_config(str(config))


def find_videos(config_path, videos_dir=None, exts=VIDEO_EXTS):
    """Videos to consider: `--videos-dir` if given, else `<project_path>/videos`.

    The project's own `video_sets` is deliberately not used -- it holds the paths as they were when
    the project was created (often a different machine), while the directory holds what is actually
    there to work with now.
    """
    if videos_dir:
        root = Path(videos_dir)
    else:
        cfg = read_config(config_path)
        root = Path(cfg["project_path"]) / "videos"
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    videos = sorted(
        p for p in root.iterdir()
        if p.suffix.lower() in exts and not p.name.startswith(".")
    )
    # A previous create_labeled_video run leaves *_labeled.mp4 next to the source; extracting
    # outliers from an already-annotated render would put burnt-in dots into the training set.
    return [p for p in videos if "_labeled" not in p.stem]


def prediction_files(video, destfolder=None):
    """DLC prediction .h5 files for `video` (raw only, not _filtered/_labeled leftovers)."""
    folder = Path(destfolder) if destfolder else Path(video).parent
    return sorted(
        p for p in folder.glob(Path(video).stem + "*.h5")
        if _SCORER_RE.search(p.stem) and not p.stem.endswith("_filtered")
    )


def count_frames(project_path, video):
    """PNGs currently sitting in `labeled-data/<video stem>/`, to report what a run added."""
    folder = Path(project_path) / "labeled-data" / Path(video).stem
    return len(list(folder.glob("*.png"))) if folder.is_dir() else 0


def set_numframes2pick(config_path, value):
    """Set `numframes2pick` in config.yaml, returning the previous value.

    extract_outlier_frames reads this from the config rather than taking it as an argument, so
    changing how many frames get picked means editing the project. The caller restores it.
    """
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(str(config_path))
    previous = cfg.get("numframes2pick")
    if value is not None and value != previous:
        auxiliaryfunctions.edit_config(str(config_path), {"numframes2pick": int(value)})
    return previous


def extract_one(deeplabcut, video, args, algorithm):
    """One `extract_outlier_frames` call. Raises whatever DLC raises."""
    kwargs = dict(
        config=args.config,
        videos=[str(video)],
        videotype=video.suffix,
        shuffle=args.shuffle,
        trainingsetindex=args.trainingsetindex,
        outlieralgorithm=algorithm,
        comparisonbodyparts=args.bodyparts,
        epsilon=args.epsilon,
        p_bound=args.p_bound,
        ARdegree=args.ARdegree,
        MAdegree=args.MAdegree,
        alpha=args.alpha,
        extractionalgorithm=args.extractionalgorithm,
        automatic=not args.interactive,
        cluster_color=args.cluster_color,
        savelabeled=args.savelabeled,
    )
    if args.destfolder:
        kwargs["destfolder"] = args.destfolder
    try:
        deeplabcut.extract_outlier_frames(**kwargs)
    except TypeError as exc:
        # Kwargs have moved between DLC 2.2/2.3/3.x; retry with the ones every version accepts.
        print(f"  retrying with reduced kwargs ({exc})", file=sys.stderr)
        for key in ("cluster_color", "savelabeled", "destfolder"):
            kwargs.pop(key, None)
        deeplabcut.extract_outlier_frames(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract outlier frames from every analyzed video in a DLC project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="DLC project config.yaml")
    parser.add_argument("--videos-dir", default=None,
                        help="directory of videos to scan (default <project_path>/videos)")
    parser.add_argument("--destfolder", default=None,
                        help="where the prediction .h5 files live, if not beside the videos")
    parser.add_argument("--outlieralgorithm", default="jump,uncertain",
                        help="comma-separated list run in order; 'jump' catches keypoints "
                             "teleporting, 'uncertain' catches low-likelihood frames "
                             "(also: fitting, manual)")
    parser.add_argument("--numframes2pick", type=int, default=None,
                        help="frames per video per algorithm; temporarily overrides the project "
                             "config value, which is restored on exit")
    parser.add_argument("--bodyparts", default="all",
                        help="comma-separated bodyparts to judge outliers on, or 'all'")
    parser.add_argument("--epsilon", type=float, default=20,
                        help="jump/fitting: pixels of frame-to-frame movement that counts as an "
                             "outlier")
    parser.add_argument("--p-bound", dest="p_bound", type=float, default=0.01,
                        help="uncertain: likelihood below which a prediction counts as an outlier")
    parser.add_argument("--ARdegree", type=int, default=3, help="fitting: AR order of the SARIMAX fit")
    parser.add_argument("--MAdegree", type=int, default=1, help="fitting: MA order of the SARIMAX fit")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="fitting: significance level for flagging a residual")
    parser.add_argument("--extractionalgorithm", default="kmeans", choices=["kmeans", "uniform"],
                        help="how the flagged frames are down-sampled to numframes2pick; kmeans "
                             "spreads the picks across visually different frames")
    parser.add_argument("--cluster-color", action="store_true",
                        help="cluster on color instead of grayscale (slower)")
    parser.add_argument("--savelabeled", action="store_true",
                        help="also save a copy of each extracted frame with the current "
                             "predictions drawn on it, for eyeballing what the network did")
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--trainingsetindex", type=int, default=0)
    parser.add_argument("--interactive", action="store_true",
                        help="let DLC ask before extracting (do NOT use in a SLURM job: the "
                             "prompt will hang the task until it times out)")
    parser.add_argument("--include-unanalyzed", action="store_true",
                        help="attempt videos with no prediction .h5 as well (they will fail; "
                             "useful only to see the DLC error)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the videos and what would run, extract nothing")
    args = parser.parse_args(argv)

    algorithms = [a.strip() for a in args.outlieralgorithm.split(",") if a.strip()]
    if not algorithms:
        raise SystemExit("--outlieralgorithm is empty")
    if args.bodyparts != "all":
        args.bodyparts = [b.strip() for b in args.bodyparts.split(",") if b.strip()]

    videos = find_videos(args.config, args.videos_dir)
    if not videos:
        raise SystemExit("no videos found")

    ready, skipped = [], []
    for video in videos:
        if prediction_files(video, args.destfolder) or args.include_unanalyzed:
            ready.append(video)
        else:
            skipped.append(video)
    for video in skipped:
        print(f"skipping {video.name}: no DLC prediction .h5 (run analyze_videos first)",
              file=sys.stderr)
    if not ready:
        raise SystemExit("no analyzed videos to work on")

    print(
        f"{len(ready)} video(s) x {len(algorithms)} algorithm(s) "
        f"({', '.join(algorithms)}), {args.numframes2pick or 'config'} frames each",
        file=sys.stderr,
    )
    if args.dry_run:
        for video in ready:
            print(f"  would extract from {video}", file=sys.stderr)
        return 0

    import deeplabcut  # imported late: slow, pulls in torch

    project_path = Path(read_config(args.config)["project_path"])
    previous_numframes = set_numframes2pick(args.config, args.numframes2pick)

    failures, totals = [], {}
    try:
        for video in ready:
            before = count_frames(project_path, video)
            for algorithm in algorithms:
                print(f"[{video.name}] {algorithm}", flush=True)
                try:
                    extract_one(deeplabcut, video, args, algorithm)
                except Exception as exc:  # one bad video must not end the batch
                    print(f"  FAILED ({algorithm}): {exc}", file=sys.stderr)
                    failures.append((video.name, algorithm, str(exc)))
            after = count_frames(project_path, video)
            totals[video.name] = after - before
            print(f"[{video.name}] {after - before} new frame(s) "
                  f"in labeled-data/{video.stem} ({after} total)", flush=True)
    finally:
        # Restore the project's own numframes2pick even on Ctrl-C, so the config is not left
        # carrying a value from a one-off run.
        if args.numframes2pick is not None and previous_numframes is not None:
            set_numframes2pick(args.config, previous_numframes)

    added = sum(totals.values())
    print(
        f"\nextracted {added} new frame(s) across {len(ready)} video(s)\n"
        f"next: deeplabcut.refine_labels('{args.config}')  # correct the machine labels\n"
        f"then: deeplabcut.merge_datasets(...) + create_training_dataset(...)",
        file=sys.stderr,
    )
    if failures:
        print(f"{len(failures)} call(s) failed:", file=sys.stderr)
        for name, algorithm, exc in failures:
            print(f"  {name} [{algorithm}]: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
