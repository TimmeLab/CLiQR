# Licks DLC never saw -> synced videos (`--dlc-exclude`)

Date: 2026-08-11
Status: approved, ready for implementation plan

## Problem

`--dlc-windows` (see `2026-08-11-dlc-sync-video-windows-design.md`) renders a clip for every stretch
where DeepLabCut says the animal was at the sipper. That answers "when the animal was drinking, did
the capacitance detector agree?"

The opposite question is the one that bounds the false-positive rate: **when the capacitance
detector reported licks, was the animal actually there?** Today nothing selects those stretches. The
default trace search picks the busiest bouts regardless of what the video shows, so its output is
dominated by real drinking and the suspicious blocks are buried.

We want the complement: bouts the detector found that fall in no DLC window, rendered as synced
clips so they can be adjudicated by eye.

## Solution overview

Add `--dlc-exclude PATH` to `scripts/find_interesting_windows.py` -- a third mode, mutually
exclusive with `--dlc-windows`.

The mode is **bout-seeded** (like the default trace search) but **DLC-filtered**: it reads the same
CSV `find_dlc_windows.py` writes, converts every window to session seconds, and then emits the
busiest detected bouts that touch none of them.

Which DLC CSV you pass is the strictness knob:

* a permissive presence CSV (no `--require-tongue`, loose `--max-nose-dist`) excludes anything where
  the animal was merely near the sipper, so what survives is licks reported while the animal was
  demonstrably elsewhere -- unambiguous false positives;
* a strict drinking CSV (`--require-tongue`) also lets through "at the sipper but not drinking",
  which is a real and separate failure mode worth looking at.

No new DLC gating logic: the thresholds live in `find_dlc_windows.py`, where they already are.

Everything downstream is unchanged -- the same `rois.csv` writer, the same `make_clips.sh` writer,
the same `make_sync_video.py` command shape.

## Data flow

1. `build_cycles_for_dlc(combined_h5, raw_map, animals)` -- reused unchanged. Per cycle: the filmed
   animal, `raw_h5`, `layout`, trace bounds, restart flag, `lick_times`.
2. New `dlc_exclusion_spans(dlc_rows, cycles, sess_loader=None)` ->
   `({stem: {"spans": [(start_s, end_s), ...], "coverage": (first_s, last_s)}}, skips)`.
   Rows are grouped by video so the per-frame session times are read once per recording, exactly as
   `build_dlc_rois` does. Conversion uses the same clock as the renderer:
   `load_frame_session_times(raw_h5)` then `frame_window_to_session(sess, start_frame, end_frame)`.
   `coverage` is `(sess[0], sess[-1])`.
   **No clamp to the trace.** `build_dlc_rois` clamps because it is about to render; here a window
   outside the trace still proves the animal was at the sipper, so clamping it away would leak a
   false-positive candidate.
3. For each cycle that has a usable exclusion entry, read that cycle's `cap_data`, `time_data`,
   `bout_start_times`, `bout_durations`, `bout_lick_counts`, `lick_times` for the **filmed animal
   only** (the other cages have no camera, so nothing can be adjudicated for them).
4. Drop disqualified bouts (rules below), then call the existing `build_rois_for_cycle` with the
   surviving bout arrays and `n_climb=0`. Climbing windows are not lick false positives.
5. Relabel each returned region `category="no_dlc"` and attach the usual provenance
   (`animal`, `cycle`, `filmed`, `restart`, `raw_h5`, `layout`).
6. `write_outputs(all_rows, args)` -- unchanged.

The only genuinely new code is step 2 and the bout filter in step 4. Frame -> session time is not
re-derived here; see the `--dlc-windows` design for why the sidecar clock, not `frame / fps`, is the
only correct conversion.

## Disqualification rules

A bout occupies `[bout_start, bout_start + bout_duration]`. It is kept only if **both** hold:

* **Inside DLC coverage.** The bout lies within the video's `coverage` span. The trace can start
  before the camera and run past it; outside the video DLC saw nothing, so the absence of a window
  there is not evidence of anything.
* **No overlap with any guarded span.** Each DLC span is grown by `--dlc-guard` seconds (default
  1.0) on both sides; any intersection with any grown span disqualifies the bout.

The guard band exists because DLC window edges are ragged -- they come out of `--merge-gap`,
`--min-frames` and `--pad` in `find_dlc_windows.py`, so a window's boundary is not the moment the
animal arrived or left. Without a guard, bouts that start a few hundred milliseconds before a window
opens survive and are almost all real licks.

The test is applied to the **bout**, not to the padded clip window. The clip is deliberately wider
(`--lick-pad` of context each side), and a clip that happens to show the edge of a DLC window is
useful context, not a disqualification.

## Skips

Nothing is silently dropped. Skips are counted by reason and printed at the end, in the style of
`print_dlc_skips`.

Whole-cycle skips (the cycle emits nothing):

| Reason | Why |
|---|---|
| `no_dlc_video` | No CSV row names this cycle's recording. "DLC found no windows" and "DLC never ran on this video" are indistinguishable from the CSV, and the second one would dump the cycle's entire bout list into the false-positive pile. |
| `cfr_video` | The video is a `_cfr` re-encode; its frame indices are not guaranteed to be the original container's, so no span can be placed on the session clock. |
| `no_cycle` | A CSV video with no matching cycle in the combined file (or filtered out by `--animals`). Counted against the rows, as in `--dlc-windows`. |
| `no_video_times` | The recording's mp4 or PTS sidecar is missing on this machine, so frames cannot be timed. |
| `no_trace` | The filmed animal cannot be resolved, or its group has no usable `time_data`. |

Per-bout skips:

| Reason | Why |
|---|---|
| `outside_video` | The bout falls outside the video's coverage span. |
| `in_dlc_window` | The bout intersects a guarded DLC span -- i.e. DLC agrees the animal was there. |

Per-row skip, when converting a DLC window to a span:

| Reason | Why |
|---|---|
| `frame_out_of_range` | `start_frame` is negative or past the end of the PTS sidecar. |

`no_dlc_video` is the correctness crux of this mode and is called out explicitly in the printed
summary.

## Known limitation

A bout where the animal really was drinking but DLC missed it -- likelihood never cleared
`--pcutoff`, or the tongue never crossed `--tongue-pcutoff` -- still lands in the output. That is
intended: the clips are a review queue and the human adjudicates. Stricter gating in the input CSV
produces more such rows; that is the trade-off the choice of CSV controls.

## CLI

```
python scripts/find_interesting_windows.py "Lickometry Data/results_combined_ACG-26-3_*.h5" \
    --dlc-exclude second_iteration_windows_with_tongue.csv \
    --dlc-guard 1.0 --n-lick 5 \
    --csv fp_rois.csv --sh make_fp_clips.sh --speed 0.25
```

* `--dlc-exclude PATH` -- CSV written by `dlc_integration/find_dlc_windows.py`. Enables the mode.
* `--dlc-guard SECONDS` -- guard band around each DLC window (default 1.0). `0` disables it.
* Passing `--dlc-exclude` together with `--dlc-windows` is a `parser.error`: they select windows by
  opposite criteria and there is no sensible combination.
* Honoured: `--n-lick` (bouts per cycle, busiest first), `--roi-seconds`, `--lick-pad`, `--csv`,
  `--sh`, `--out-dir`, `--animals` (filters on the resolved filmed animal), `--raw-map`, `--offset`,
  `--speed`.
* Unused (passing one is not an error): `--n-climb` (forced to 0), `--var-window`, `--min-var`,
  `--climb-skip-edges`, `--include-controls` -- the filmed animal is never a control cage.

## Output

The existing `CSV_COLUMNS`, with:

* `category` = `"no_dlc"`,
* `rank` = 0-based within the cycle, busiest surviving bout first,
* `score` = the bout's lick count,
* `n_licks_in_window` = licks inside the emitted window, as elsewhere.

Clips are named `<animal>_c<cycle>_no_dlc<rank>.mp4` by the existing `build_command`. Restart-cycle
WARNING comments are emitted unchanged.

## Testing

All in `tests/test_find_interesting_windows.py`, with an injected `sess_loader`, so no recording or
video is needed on disk.

* `dlc_exclusion_spans` converts frames to session seconds and does **not** clamp to the trace.
* Each whole-video skip reason is counted: `cfr_video`, `no_cycle`, `no_video_times`, `no_trace`.
* `frame_out_of_range` is counted per row.
* Guard band: a bout ending 0.5 s before a DLC window is dropped at `--dlc-guard 1.0` and kept at
  `--dlc-guard 0`.
* Coverage: a bout after `coverage[1]` is skipped as `outside_video`.
* A cycle with no rows in the CSV emits nothing and is counted `no_dlc_video`.
* End to end: a bout inside a DLC window is absent from the output; a bout far from every window is
  present with `category="no_dlc"`, the right `rank`, `score` and `n_licks_in_window`.
* `--dlc-exclude` with `--dlc-windows` exits non-zero.
