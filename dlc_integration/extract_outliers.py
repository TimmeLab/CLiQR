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

Why the frames are gated to windows (the important part)
--------------------------------------------------------
Run unmodified, `outlieralgorithm='jump'` flagged 181,215 of ~317,000 frames (57%) on one of these
recordings -- while the animal is only actually in frame for about 2% of them. The reason is that
`jump` thresholds frame-to-frame keypoint displacement with no regard for likelihood: in an empty
cage the predictions bounce around the arena every frame, so nearly every empty frame looks like a
huge "jump". kmeans then picks its frames out of a pool that is almost entirely empty cage, and you
spend an afternoon refining pictures of bedding. `uncertain` fails the same way from the other
direction -- empty frames are exactly the low-likelihood ones.

So by default this script builds the candidate pool itself: frames are first restricted to the
mouse-is-present windows that `find_dlc_windows.py` computes (contiguous runs where the gate
bodypart clears its likelihood cutoff, merged and padded), and the outlier test is applied only
inside those. The surviving indices are handed to DLC's own
`ExtractFramesbasedonPreselection`, which is the exact function `extract_outlier_frames` calls
once it has decided on its indices -- so the output, the `machinelabels-iter<N>.h5` and the
`refine_labels` workflow are identical to a normal DLC run. `--no-gate` restores the plain
whole-video behaviour.

Three criteria are available inside the gate:
    jump       keypoint moved more than --epsilon px between consecutive frames. Catches the
               confident-but-wrong cases (nose/tongue swaps), which is where the labels help most.
    uncertain  some bodypart below --p-bound. Inside a window the animal IS present, so a low
               likelihood is a genuine miss rather than an empty cage.
    window     every gated frame is a candidate; kmeans then just spreads picks over visually
               different mouse-present frames. Use this to grow the training set generally
               rather than to chase specific failures.

Extracted frames land in `<project>/labeled-data/<video stem>/` alongside a `machinelabels-iter<N>.h5`
of the current predictions, which is what makes refinement fast: run `refine_labels` and you are
correcting the network's guesses rather than clicking from scratch. Nothing here modifies the
training set; `merge_datasets` + `create_training_dataset` is still a separate, deliberate step.

Usage:
    python dlc_integration/extract_outliers.py \
        --config /N/lustre/project/proj-530/dlc_projects/CLiQR_Validation-parkecp-2026-07-27/config.yaml \
        --outlieralgorithm window,jump,uncertain --numframes2pick 20

    # see the pool sizes each criterion would produce, extract nothing
    python dlc_integration/extract_outliers.py --config .../config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import find_dlc_windows  # noqa: E402  (same directory; the window logic lives there, unduplicated)

# Same marker as find_dlc_windows.py: everything before it in an .h5 name is the video stem.
_SCORER_RE = re.compile(r"(DLC_|DeepCut_|DLCnet)")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# Criteria this script evaluates itself, inside the gate. Anything else is passed through to
# deeplabcut.extract_outlier_frames unchanged (which means: whole video, no gating).
GATED_ALGORITHMS = ("jump", "uncertain", "window")


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


# --------------------------------------------------------------------------------------------
# candidate-frame selection (the gated path)
# --------------------------------------------------------------------------------------------
def load_predictions(h5_path):
    """Return (scorer, raw DataFrame or None, per-bodypart dict of Nx3 [x, y, likelihood] arrays).

    The raw frame is kept as-is because that is what gets handed back to DLC: with
    `with_annotations=True`, ExtractFramesbasedonPreselection slices it to build the
    machinelabels-iter<N>.h5, and that file needs the scorer level of the column MultiIndex intact.

    The DataFrame is None when pandas cannot read the file for lack of pytables -- enough to
    inspect pools with --dry-run off-cluster, not enough to extract. Same fallback as
    find_dlc_windows.load_dlc_h5, which exists for the same reason.
    """
    try:
        import pandas as pd

        df = pd.read_hdf(h5_path)
    except ImportError:
        return (*_load_predictions_h5py(h5_path),)

    scorer = df.columns.get_level_values(0)[0]
    body = df[scorer]
    bodyparts = list(dict.fromkeys(body.columns.get_level_values(0)))
    arrays = {
        bp: np.column_stack([
            np.asarray(body[(bp, "x")].values, dtype=float),
            np.asarray(body[(bp, "y")].values, dtype=float),
            np.asarray(body[(bp, "likelihood")].values, dtype=float),
        ])
        for bp in bodyparts
    }
    return scorer, df, arrays


def _load_predictions_h5py(h5_path):
    """(scorer, None, arrays) read straight from the pytables layout with h5py."""
    import pickle

    import h5py

    with h5py.File(h5_path, "r") as fh:
        groups = [k for k in fh.keys() if isinstance(fh[k], h5py.Group) and "table" in fh[k]]
        if not groups:
            raise ValueError(f"{h5_path}: no pytables frame group found")
        grp = fh[groups[0]]
        table = grp["table"]
        blocks = sorted(n for n in table.dtype.names if n.startswith("values_block_"))
        if len(blocks) != 1:
            raise ValueError(
                f"{h5_path}: {len(blocks)} value blocks; install pytables (`pip install tables`)"
            )
        values = np.asarray(table[blocks[0]], dtype=float)
        columns = pickle.loads(bytes(grp.attrs["non_index_axes"]))[0][1]

    scorer = columns[0][0]
    order = {"x": 0, "y": 1, "likelihood": 2}
    arrays = {}
    for i, (_scorer, bp, coord) in enumerate(columns):
        if coord not in order:
            continue
        arrays.setdefault(bp, np.full((values.shape[0], 3), np.nan))[:, order[coord]] = values[:, i]
    return scorer, None, arrays


def resolve_bodyparts(requested, available):
    """`all` -> every bodypart; otherwise validate the requested subset."""
    if requested == "all":
        return list(available)
    missing = [bp for bp in requested if bp not in available]
    if missing:
        raise ValueError(f"unknown bodypart(s) {missing}; available: {list(available)}")
    return list(requested)


def gate_mask_from_csv(csv_path, video, n_frames):
    """Boolean per-frame mask from a find_dlc_windows.py CSV, for rows matching `video`."""
    stem = Path(video).stem
    mask = np.zeros(n_frames, dtype=bool)
    matched = 0
    with Path(csv_path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if Path(row["video"]).stem != stem:
                continue
            start = max(0, int(row["start_frame"]))
            end = min(n_frames, int(row["end_frame"]))
            if end > start:
                mask[start:end] = True
                matched += 1
    if not matched:
        raise ValueError(f"no rows in {csv_path} for video stem {stem!r}")
    return mask


def gate_mask(arrays, args, video, n_frames):
    """Frames where the animal is judged to be in view.

    Either replayed from a find_dlc_windows.py CSV (--windows-csv) or recomputed here with that
    module's own window pipeline, so the definition of "the mouse is here" is the same one used to
    pick the review clips. `max_frames=0` because splitting long windows only matters when each
    window becomes a separate render job.
    """
    if args.windows_csv:
        return gate_mask_from_csv(args.windows_csv, video, n_frames)

    if args.gate_bodypart not in arrays:
        raise ValueError(
            f"gate bodypart '{args.gate_bodypart}' not in this file; "
            f"available: {list(arrays)}"
        )
    windows = find_dlc_windows.find_windows(
        arrays[args.gate_bodypart][:, 2] >= args.gate_pcutoff,
        merge_gap=args.gate_merge_gap,
        min_frames=args.gate_min_frames,
        min_confident=args.gate_min_confident,
        pad=args.gate_pad,
        max_frames=0,
    )
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in windows:
        mask[start:end] = True
    return mask


def candidate_indices(arrays, bodyparts, algorithm, args):
    """Frames flagged by `algorithm`, before gating. Mirrors DLC's own tests."""
    if algorithm == "window":
        # No test at all: every gated frame is a candidate and kmeans does the choosing.
        return np.arange(len(next(iter(arrays.values()))))

    if algorithm == "uncertain":
        like = np.column_stack([arrays[bp][:, 2] for bp in bodyparts])
        return np.flatnonzero(np.any(like < args.p_bound, axis=1))

    if algorithm == "jump":
        # Same test as DLC: per-bodypart squared frame-to-frame displacement, flagged if ANY
        # bodypart moved more than epsilon. np.diff drops one frame, so shift the indices back up.
        flagged = np.zeros(len(next(iter(arrays.values()))), dtype=bool)
        for bp in bodyparts:
            xy = arrays[bp][:, :2]
            step2 = np.sum(np.diff(xy, axis=0) ** 2, axis=1)
            flagged[1:] |= step2 > args.epsilon ** 2
        return np.flatnonzero(flagged)

    raise ValueError(f"'{algorithm}' is not a gated criterion; expected one of {GATED_ALGORITHMS}")


def gated_pool(arrays, args, video, algorithm):
    """(indices to extract, gated frame count, ungated flag count) for one criterion."""
    n_frames = len(next(iter(arrays.values())))
    bodyparts = resolve_bodyparts(args.bodyparts, list(arrays))
    mask = gate_mask(arrays, args, video, n_frames)
    flagged = candidate_indices(arrays, bodyparts, algorithm, args)
    return flagged[mask[flagged]], int(mask.sum()), len(flagged)


def warn_if_undiscriminating(algorithm, n_index, n_gated, args, out=sys.stderr):
    """Flag the case where a criterion has stopped selecting anything in particular.

    With --bodyparts all, one chronically-invisible keypoint decides everything: the tongue sits
    near likelihood 0.1 whenever it is inside the mouth, so 'uncertain' fires on ~90% of gated
    frames and degenerates into 'window'. That is not an error -- it just means kmeans, not the
    criterion, is choosing the frames, and you should know which one is doing the work.
    """
    if algorithm == "window" or not n_gated:
        return
    share = n_index / n_gated
    if share >= 0.7:
        print(
            f"  note: '{algorithm}' flags {share:.0%} of gated frames, so it is barely selecting "
            f"anything (kmeans is doing the choosing). Restrict --bodyparts to keypoints that "
            f"should always be visible, or tighten "
            f"{'--p-bound' if algorithm == 'uncertain' else '--epsilon'}; --stats shows which.",
            file=out,
        )


def print_bodypart_stats(video, arrays, args, out=sys.stderr):
    """Per-bodypart likelihood/jump summary INSIDE the gate, to choose --bodyparts and thresholds.

    Whether a criterion discriminates depends entirely on which bodyparts feed it. On these
    recordings the four sipper points are static and tracked near-perfectly, while the tongue sits
    at a median likelihood of ~0.1 because it is only visible during a lick -- so 'uncertain' over
    all bodyparts flags ~90% of gated frames and stops being a signal. This table is how you find
    that out before spending a GPU-hour on it.
    """
    n_frames = len(next(iter(arrays.values())))
    mask = gate_mask(arrays, args, video, n_frames)
    print(f"  {video.name}: {int(mask.sum())}/{n_frames} frames gated", file=out)
    print(f"    {'bodypart':<18}{'med_lik':>9}{'frac<p_bound':>14}{'med_jump_px':>13}"
          f"{'frac_jump>eps':>15}", file=out)
    for bp, arr in arrays.items():
        like = arr[mask, 2]
        step = np.concatenate([[0.0], np.sqrt(np.sum(np.diff(arr[:, :2], axis=0) ** 2, axis=1))])
        step = step[mask]
        print(
            f"    {bp:<18}{np.median(like):>9.3f}{np.mean(like < args.p_bound):>14.3f}"
            f"{np.median(step):>13.2f}{np.mean(step > args.epsilon):>15.3f}",
            file=out,
        )


def extract_preselected(video, data, index, args, cfg):
    """Hand our own frame indices to DLC's extractor.

    `ExtractFramesbasedonPreselection` is what extract_outlier_frames itself calls once it has
    chosen indices, so everything downstream -- the PNGs in labeled-data/<stem>/, the
    machinelabels-iter<N>.h5, refine_labels -- behaves exactly as in a stock DLC run. It reads
    `numframes2pick`, `start`, `stop` and `iteration` from `cfg`.
    """
    from deeplabcut.refine_training_dataset.outlier_frames import (
        ExtractFramesbasedonPreselection,
    )

    if data is None:
        raise ValueError(
            "predictions were read through the h5py fallback, which cannot produce the DataFrame "
            "DLC needs for machinelabels; install pytables (`pip install tables`)"
        )
    ExtractFramesbasedonPreselection(
        np.sort(np.asarray(index, dtype=int)),
        args.extractionalgorithm,
        data,
        str(video),
        cfg,
        str(args.config),
        opencv=True,
        cluster_resizewidth=args.cluster_resizewidth,
        cluster_color=args.cluster_color,
        savelabeled=args.savelabeled,
    )


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
    parser.add_argument("--outlieralgorithm", default="window,jump,uncertain",
                        help="comma-separated list run in order. Gated criteria: 'jump' "
                             "(keypoint moved > --epsilon px), 'uncertain' (bodypart below "
                             "--p-bound), 'window' (every mouse-present frame). Anything else "
                             "(fitting, manual) is passed to DLC ungated, over the whole video")
    parser.add_argument("--numframes2pick", type=int, default=None,
                        help="frames per video per algorithm; temporarily overrides the project "
                             "config value, which is restored on exit")
    parser.add_argument("--bodyparts", default="all",
                        help="comma-separated bodyparts to judge outliers on, or 'all'")
    parser.add_argument("--epsilon", type=float, default=20,
                        help="jump/fitting: pixels of frame-to-frame movement that counts as an "
                             "outlier")
    parser.add_argument("--p-bound", dest="p_bound", type=float, default=0.6,
                        help="uncertain: likelihood below which a prediction counts as an outlier. "
                             "Higher than DLC's 0.01 default on purpose: inside a gated window the "
                             "animal really is there, so anything the network is lukewarm about is "
                             "a miss worth labeling")
    parser.add_argument("--ARdegree", type=int, default=3, help="fitting: AR order of the SARIMAX fit")
    parser.add_argument("--MAdegree", type=int, default=1, help="fitting: MA order of the SARIMAX fit")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="fitting: significance level for flagging a residual")
    parser.add_argument("--extractionalgorithm", default="kmeans", choices=["kmeans", "uniform"],
                        help="how the flagged frames are down-sampled to numframes2pick; kmeans "
                             "spreads the picks across visually different frames")
    gate = parser.add_argument_group(
        "mouse-present gate",
        "Restricts candidate frames to the windows where the animal is actually in view. Without "
        "this, 'jump' flags ~57%% of an empty-cage recording and kmeans picks pictures of bedding.",
    )
    gate.add_argument("--no-gate", dest="gate", action="store_false",
                      help="disable gating: hand the whole video to deeplabcut.extract_outlier_frames")
    gate.add_argument("--windows-csv", default=None,
                      help="reuse windows from find_dlc_windows.py instead of recomputing them; "
                           "rows are matched to each video by stem")
    gate.add_argument("--gate-bodypart", default="nose",
                      help="bodypart whose likelihood decides 'the mouse is in frame'")
    gate.add_argument("--gate-pcutoff", type=float, default=0.8,
                      help="likelihood the gate bodypart must clear")
    gate.add_argument("--gate-merge-gap", type=int, default=120,
                      help="merge confident runs separated by at most this many frames "
                           "(120 = 1 s at 120 fps)")
    gate.add_argument("--gate-min-frames", type=int, default=30,
                      help="discard merged windows shorter than this")
    gate.add_argument("--gate-min-confident", type=int, default=15,
                      help="discard merged windows with fewer confident frames than this")
    gate.add_argument("--gate-pad", type=int, default=60,
                      help="frames of context added to each side of a window")

    parser.add_argument("--cluster-resizewidth", type=int, default=30,
                        help="width the frames are downscaled to before kmeans clustering")
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
                        help="report the candidate pool each criterion would draw from, "
                             "extract nothing")
    parser.add_argument("--stats", action="store_true",
                        help="print per-bodypart likelihood and jump statistics inside the gate "
                             "(use this to pick --bodyparts, --p-bound and --epsilon), then exit")
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

    if not args.gate and "window" in algorithms:
        raise SystemExit(
            "'window' only exists as a gated criterion (it means 'every mouse-present frame'); "
            "drop it or drop --no-gate"
        )
    ungated = [a for a in algorithms if a not in GATED_ALGORITHMS or not args.gate]
    if ungated and args.gate:
        print(
            f"note: {', '.join(ungated)} has no gated implementation and will run over the whole "
            "video via deeplabcut.extract_outlier_frames",
            file=sys.stderr,
        )
    print(
        f"{len(ready)} video(s) x {len(algorithms)} algorithm(s) "
        f"({', '.join(algorithms)}), {args.numframes2pick or 'config'} frames each"
        + ("" if args.gate else " [gate disabled]"),
        file=sys.stderr,
    )

    if args.stats:
        for video in ready:
            preds = prediction_files(video, args.destfolder)
            try:
                _scorer, _data, arrays = load_predictions(preds[0])
                print_bodypart_stats(video, arrays, args)
            except Exception as exc:
                print(f"  {video.name}: FAILED {exc}", file=sys.stderr)
        return 0

    if args.dry_run:
        # Report the pool each criterion would draw from. This is the number that told us the
        # ungated run was broken (181k flagged of 317k frames), so it is worth seeing up front.
        for video in ready:
            preds = prediction_files(video, args.destfolder)
            if not preds or not args.gate:
                print(f"  would extract from {video}", file=sys.stderr)
                continue
            try:
                _scorer, _data, arrays = load_predictions(preds[0])
                for algorithm in algorithms:
                    if algorithm in ungated:
                        continue
                    index, n_gated, n_flagged = gated_pool(arrays, args, video, algorithm)
                    n = len(next(iter(arrays.values())))
                    print(
                        f"  {video.name} [{algorithm}]: {len(index)} candidates "
                        f"({n_gated}/{n} frames gated, {n_flagged} flagged before gating)",
                        file=sys.stderr,
                    )
                    warn_if_undiscriminating(algorithm, len(index), n_gated, args)
            except Exception as exc:
                print(f"  {video.name}: FAILED {exc}", file=sys.stderr)
        return 0

    import deeplabcut  # imported late: slow, pulls in torch

    project_path = Path(read_config(args.config)["project_path"])
    previous_numframes = set_numframes2pick(args.config, args.numframes2pick)

    failures, totals = [], {}
    try:
        # Read AFTER the numframes2pick edit: ExtractFramesbasedonPreselection takes its frame
        # count from this dict, not from the file.
        cfg = read_config(args.config)
        if (cfg.get("start"), cfg.get("stop")) != (0, 1):
            print(
                f"note: project config restricts extraction to start={cfg.get('start')} "
                f"stop={cfg.get('stop')} of each video",
                file=sys.stderr,
            )
        for video in ready:
            before = count_frames(project_path, video)
            data = arrays = None
            for algorithm in algorithms:
                print(f"[{video.name}] {algorithm}", flush=True)
                try:
                    if algorithm in ungated:
                        extract_one(deeplabcut, video, args, algorithm)
                        continue
                    if arrays is None:
                        preds = prediction_files(video, args.destfolder)
                        if len(preds) > 1:
                            print(f"  {len(preds)} prediction files; using {preds[0].name}",
                                  file=sys.stderr)
                        _scorer, data, arrays = load_predictions(preds[0])
                    index, n_gated, n_flagged = gated_pool(arrays, args, video, algorithm)
                    n = len(next(iter(arrays.values())))
                    print(f"  {len(index)} candidate frame(s) "
                          f"({n_gated}/{n} gated, {n_flagged} flagged before gating)", flush=True)
                    warn_if_undiscriminating(algorithm, len(index), n_gated, args)
                    if len(index) == 0:
                        print("  nothing to extract; loosen the gate or the criterion",
                              file=sys.stderr)
                        continue
                    extract_preselected(video, data, index, args, cfg)
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
