# DLC window selection: sipper proximity + tongue rhythmicity

**Date:** 2026-08-10
**Status:** design approved, not yet implemented
**Touches:** `dlc_integration/find_dlc_windows.py`, `dlc_integration/extract_outliers.py`
(one call site), `tests/test_dlc_windows.py` (new)

## Problem

`find_dlc_windows.py` decides "the mouse is here, render this window" from a single test:
`nose.likelihood >= --pcutoff`. That is too permissive. A confident nose anywhere in the frame
qualifies — the animal wandering past the sipper, sitting in the corner, or grooming produces
windows we then spend GPU-hours labeling and human time reviewing.

The DLC model now predicts four keypoints along the curve of the sipper tip
(`sipper_top`, `sipper_midtop`, `sipper_midbottom`, `sipper_bottom`), plus `nose`, `tongue`, `jaw`.
Those sipper keypoints let us ask the question we actually care about: *is the nose confidently
detected AND close to the sipper?* And, optionally, *is the tongue appearing rhythmically*, which is
what licking looks like in the likelihood trace (the tongue is only visible at the top of each lick,
so its likelihood pulses rather than staying high).

## Measurements that drove the design

Probed all ten `*snapshot_best-140*.h5` files in
`Lickometry Data/ACG-26-3/dlc_analysis_results/`:

- **Sipper keypoints are near-static within a session.** Each clears likelihood 0.8 in 77–99% of
  frames, and over those frames the per-keypoint position IQR is 0.5–3.5 px. A per-session median
  is effectively exact.
- **Sipper size varies between sessions.** Polyline arc length through the four points runs
  140–165 px (camera distance differs per recording). A pixel threshold therefore does not transfer
  between sessions; a threshold expressed as a fraction of that arc does.
- **Proximity is a real filter.** The nose clears 0.8 in only 0.3–2.4% of frames, and of those, the
  fraction within ~0.6 arc of the sipper ranges from 34% to 90% depending on the session.
- **Tongue rhythm separates cleanly.** In near-sipper stretches of the 07-22 session, drinking
  stretches show a 7.9–9.1 Hz spectral peak in `tongue.likelihood` and 3.4–7.8 upward crossings of
  0.6 per second; non-drinking stretches show 0–0.4 crossings/s and no lick-band peak. Counting
  crossings is enough — no FFT required.
- **Tongue detection is not trustworthy session-wide.** The 07-23 session has 18,442 frames with
  `tongue.likelihood > 0.6` yet almost none of them fall in a rhythmic near-sipper stretch. This is
  why the tongue test is opt-in, not part of the default gate.

Prototype run of the design below, per session, windows / total seconds:

| session | current (nose only) | + proximity | + tongue rhythm |
|---|---|---|---|
| 07-13 | 95 / 146.5 s | 26 / 67.3 s | 14 / 33.8 s |
| 07-14 | 39 / 75.5 s  | 13 / 37.2 s | 5 / 12.8 s |
| 07-16 | 31 / 72.1 s  | 13 / 34.4 s | 10 / 30.3 s |
| 07-21 | 30 / 86.8 s  | 9 / 53.3 s  | 6 / 48.9 s |
| 07-22 | 56 / 188.8 s | 27 / 122.8 s | 9 / 86.6 s |
| 07-23 | 99 / 178.5 s | 35 / 61.2 s | 1 / 9.7 s |
| 07-24 | 44 / 129.2 s | 17 / 73.5 s | 13 / 71.9 s |
| 07-27 | 44 / 112.7 s | 17 / 61.5 s | 13 / 58.3 s |
| 07-28 | 53 / 97.7 s  | 18 / 38.1 s | 3 / 14.6 s |
| 07-29 | 22 / 53.8 s  | 9 / 14.5 s  | 6 / 12.6 s |

(`--max-nose-dist 0.6`, all other parameters at their current defaults.)

## Design

### 1. Read coordinates, not just likelihood

`load_dlc_h5` currently returns `(scorer, bodyparts, likelihood_dict)` and discards x/y. It changes
to return `(scorer, bodyparts, coords)` where `coords[bodypart]` is a dict with keys `x`, `y`,
`likelihood`, each a float array of length n_frames. Both the pandas path and the h5py/pytables
fallback populate all three; the fallback already reads every column, it just filters to
`likelihood` today.

All existing callers within the file are updated. `extract_outliers.py` imports the module but has
its own reader (`load_predictions`, which already returns Nx3 x/y/likelihood arrays), so it is
unaffected by this change; `dlc_label_window.py` does not import either. The two readers stay
separate — merging them is a bigger refactor than this work justifies.

### 2. Sipper anchor

New `sipper_anchor(coords, bodyparts, pcutoff, min_frames=100)`:

- For each of `sipper_top`, `sipper_midtop`, `sipper_midbottom`, `sipper_bottom` present in the
  file, take frames where that keypoint's likelihood `>= pcutoff` (`--sipper-pcutoff`, default 0.6)
  and return the median (x, y) over them.
- A keypoint with fewer than `min_frames` confident frames is dropped from the polyline.
- Returns `(points, arc_length)` where `points` is the surviving keypoints **in anatomical order**
  (top → midtop → midbottom → bottom, not sorted by position) and `arc_length` is the sum of the
  segment lengths between consecutive surviving points.
- Fewer than two surviving points raises `ValueError("<path>: no usable sipper keypoints ...")`.

Ordering is anatomical because the polyline models the physical curve of the tip; re-sorting the
points would connect them in the wrong sequence when the sipper is oriented diagonally in frame.

### 3. Nose-to-sipper distance

New `point_to_polyline_distance(px, py, points)`: for each consecutive pair of points, compute the
standard vectorized point-to-segment distance (project onto the segment, clamp the parameter to
[0, 1], take the Euclidean distance to the clamped foot); return the elementwise minimum over
segments. Degenerate zero-length segments fall back to the distance to the shared endpoint.

Segment distance rather than nearest-keypoint distance, because adjacent keypoints are ~40–50 px
apart and a nose resting midway between two of them reads as up to ~25 px farther away than it is.

### 4. The gate

`find_windows` takes a boolean per-frame mask instead of `(likelihood, pcutoff)`; thresholding moves
out to the callers. The caller here (`rows_for_file`) builds:

```
thr  = args.max_nose_dist_px  if args.max_nose_dist_px is not None
       else args.max_nose_dist * arc_length
mask = (nose.likelihood >= args.pcutoff) & (nose_dist <= thr)
```

with `--max-nose-dist` defaulting to 0.6 (fraction of arc). `--max-nose-dist 0` and no
`--max-nose-dist-px` disables the proximity term entirely, reproducing today's behavior exactly.
`--max-nose-dist-px` (default None) overrides the fraction with a raw pixel threshold.

Everything downstream of the mask — merge by `--merge-gap`, drop by `--min-frames` and
`--min-confident`, pad by `--pad`, re-merge, split by `--max-frames` — is unchanged. `--bodypart`
stays and still names the keypoint whose likelihood and position are tested, so the gate can be
pointed at `jaw` instead of `nose` if that ever proves more reliable.

The other caller of `find_windows` is `extract_outliers.gate_mask`, which passes a likelihood array
today. It is updated to pass `arrays[args.gate_bodypart][:, 2] >= args.gate_pcutoff` — its behavior
is unchanged, and it does not gain proximity gating in this work (its `--windows-csv` path already
lets it replay the proximity-gated windows this script writes).

### 5. Tongue rhythmicity

New `tongue_upcross_rate(tongue_likelihood, start, end, pcutoff, fps)`: count upward crossings of
`pcutoff` within `[start, end)` (`np.diff` of the boolean-as-int8 mask equal to +1) and divide by
the window duration in seconds. A window that begins already above the cutoff contributes no
crossing for that leading run — an accepted small undercount, immaterial at 3+ crossings/s.

Applied **after the min-frames / min-confident filters but before padding and splitting**, so the
rate describes the detected behavior rather than the context padding, and so a long bout that later
gets split into chunks is judged as one bout.

- `--tongue-pcutoff` (default 0.6), `--tongue-min-rate` (default 3.0 crossings/s).
- `--require-tongue` (flag, default off) drops windows below the rate.
- The rate is **always computed and written to the CSV**, filtered only when the flag is set. One
  unfiltered run therefore tells you what threshold to use on the next.
- If the file has no `tongue` bodypart, the column is empty and `--require-tongue` raises
  `ValueError` naming the available bodyparts.

### 6. CSV and summary

`CSV_FIELDS` gains three columns, appended after `mean_likelihood` so existing column order is
preserved:

- `mean_nose_dist` — mean distance in px over the frames in the window that pass the near mask;
  empty when proximity is disabled.
- `min_nose_dist` — minimum distance in px over the same frames; empty when disabled.
- `tongue_rate` — upward crossings per second, as defined above.

Distances are reported in pixels even when the threshold was given as a fraction, because pixels are
what you can check against a labeled frame. `sipper_scale` (arc length, px) and `nose_dist_thresh_px`
go into the per-file `--summary-json` record.

Because `CSV_FIELDS` changes, `--append` onto a CSV written by the current version raises the
existing missing-column error at `read_existing_csv`. That is the correct outcome — the old rows
have no distance data — and the error message already tells the user to use a new `--csv`.

The per-file stderr progress line reports frames passing the full gate, not just the likelihood
test, plus the arc length and threshold in px.

### 7. Testing

New `tests/test_dlc_windows.py`, pure-numpy, no DLC or video fixtures:

1. `point_to_polyline_distance` against hand-computed values: a point perpendicular to a segment's
   interior, a point beyond an endpoint (clamps to the endpoint), a point exactly on the polyline
   (0.0), and a degenerate zero-length segment.
2. `sipper_anchor` on synthetic arrays: median ignores low-likelihood outlier frames; a keypoint
   with too few confident frames is dropped; arc length matches the hand-summed segment lengths;
   anatomical ordering preserved when the keypoints are diagonal; fewer than two usable points
   raises.
3. `tongue_upcross_rate` on a square wave of known period → known rate; all-below → 0.0;
   all-above → 0.0; leading partial run does not count.
4. Gate equivalence: with `--max-nose-dist 0`, `find_windows` returns exactly the windows the
   likelihood-only mask produces on the same synthetic likelihood array.
5. Proximity actually excludes: a synthetic session with a confident nose far from the sipper for
   one stretch and near it for another yields only the near stretch.
6. `extract_outliers.gate_mask` still produces the same mask it does today for a given likelihood
   array and `--gate-pcutoff`, after the signature change.

## Out of scope

- Spectral (FFT) rhythmicity detection. Upward-crossing rate separates the measured data cleanly and
  is one line to explain in the paper; the FFT variant also flagged one window that had almost no
  confident tongue frames.
- Per-frame sipper tracking. The keypoints move by ~1–3 px within a session; a static per-session
  median is more robust than tracking through the 1–23% of frames where a keypoint drops out.
- Proximity or tongue gating inside `extract_outliers.py` (it only gets the one-line call-site fix),
  and any change to `dlc_label_window.py`, the SLURM scripts, or
  `scripts/find_interesting_windows.py` (the capacitance-trace ROI tool, which is unrelated).
- Unifying the two prediction readers (`find_dlc_windows.load_dlc_h5` and
  `extract_outliers.load_predictions`).
- Re-running DLC inference or cutting video.
