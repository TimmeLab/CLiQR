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
| `no_dlc_video` | No CSV row names this cycle's recording, **or** rows do name it but every single one failed to convert to a session-time span (see `frame_out_of_range` below). "DLC found no windows", "DLC never ran on this video" and "DLC ran but nothing it wrote is usable" are all indistinguishable from the caller's side, and any of them would dump the cycle's entire bout list into the false-positive pile if treated as "zero exclusion spans". `dlc_exclusion_spans` counts the second case itself (per video, the same unit as `cfr_video`/`no_cycle`/`no_video_times`); the driver counts the first. |
| `cfr_video` | The video is a `_cfr` re-encode; its frame indices are not guaranteed to be the original container's, so no span can be placed on the session clock. |
| `no_cycle` | A CSV video with no matching cycle in the combined file (or filtered out by `--animals`). Counted against the rows, as in `--dlc-windows`. |
| `no_video_times` | The recording's mp4 or PTS sidecar is missing on this machine, so frames cannot be timed. |
| `no_trace` | The filmed animal cannot be resolved, or its group has no usable `time_data` -- **or** the group exists with usable `time_data` but predates the bout-array outputs (an older analysis pipeline never wrote `bout_start_times`/`bout_durations`/`bout_lick_counts`). Two different causes, one reason: the printed explanation names both so a reader is not sent hunting layout/video-sensor resolution when the real cause is a stale combined file. |

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

Per-cycle evidence, not just skip counts: `no_dlc_video` only fires when a cycle has NO usable DLC
spans at all. A cycle that has one or two spans, against which nearly every bout survives, passes
that guard cleanly and still produces a misleading false-positive pile -- one DLC window over a
multi-hour session is barely more informative than zero. `run_dlc_exclude_mode` therefore also
returns a `cycle_reports` list (one entry per searched cycle: `n_spans`, `n_bouts`, `n_kept`,
`restart`), and `print_dlc_exclude_skips` prints a line per cycle plus a `WARNING` when a cycle has
few spans (`<= 2`) AND a near-total kept fraction (`>= 90%`) -- see
`DLC_EXCLUDE_THIN_EVIDENCE_MAX_SPANS`/`DLC_EXCLUDE_THIN_EVIDENCE_MIN_KEPT_FRACTION` in
`scripts/find_interesting_windows.py` for the exact thresholds and their rationale. This does not
change which bouts are selected -- it is a reporting-only signal for the human adjudicating the
output.

## Clock assumption for restart recordings

DLC spans come from the session clock (`load_frame_session_times`/`SessionClock`, anchored to the
recording's own PTS sidecar); bout times come from the combined file's clock (`filter_data`'s
per-cycle rebasing). For a plain single-cycle recording the two agree, because both trace back to
the same `filter_data` call. `--dlc-windows` never has to care whether they agree, because a DLC
window IS the renderer's own reference frame -- it is never compared against anything else. This
mode is different: it directly compares a DLC span against a bout's start/end time on what has to
be the SAME clock for the comparison to mean anything.

For a RESTART recording (operator stopped and restarted within one raw file), the module docstring
already documents a known history of the two clocks disagreeing by a fixed offset (~280 s was
observed for the 2026-07-22 recording). A silent disagreement here would convert real drinking into
a reported "false positive" (bout looks outside every DLC span, because the spans are shifted) and
hide genuine ones -- and it would do so with no visible symptom: `coverage` comes from the PTS
sidecar and is typically far wider than the trace (observed `(-604.5, 7682.8)` against a `(0, 7200)`
trace for the 2026-07-22 recording), so `outside_video` cannot catch a shift of a few hundred
seconds the way it would catch a shift of hours.

We do not correct for this automatically, for the same reason the module docstring gives for
`--offset`/`--sync-offset`: getting a silent correction subtly wrong is worse than not correcting.
Instead there is a cheap manual check: a restart cycle's spans should contain MOST of its
`lick_times`, since DLC and the detector are both looking at the same drinking. This was checked by
hand for the current dataset's one restart cycle (2026-07-22, cycle 0): 663/922 licks fall inside
spans at shift 0, and 0/922 at either +-280.33 s (the known offset), so the two clocks resolve this
cycle the same way and no correction is needed today. This is a point-in-time check, not something
re-verified automatically per run -- a future restart recording's `--dlc-exclude` output should be
sanity-checked the same way before being trusted. `cycle_reports` marks restart cycles so
`print_dlc_exclude_skips` can flag them inline (see above) without a second reporting mechanism; the
per-clip restart WARNING `build_command` already emits addresses rendering only (`--offset` is
applied after selection), so it cannot fix a mis-selected clip set and is not a substitute for this
check.

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

`no_dlc` clips are rendered with `--no-crop`, showing the full frame. The crop box is framed tightly
on the sipper tip to make the tongue visible during licking; a `no_dlc` clip exists precisely to
show where the animal WAS instead, which is outside that box, so cropping would hide the very
evidence the clip exists to demonstrate.

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
* A cycle whose video HAS rows but every one fails conversion (`frame_out_of_range`) also produces
  no searchable entry and is counted `no_dlc_video`, never searched against an empty spans list --
  the regression this whole-branch review caught.
* Two video path strings resolving to the same recording stem accumulate spans rather than the
  second overwriting the first.
* `run_dlc_exclude_mode`'s `cycle_reports`: `n_kept` counts every surviving bout, not just the
  `--n-lick` emitted rows; `print_dlc_exclude_skips` prints a per-cycle line and a thin-evidence
  `WARNING` when spans are few and the kept fraction is near 1.0, and marks restart cycles inline.
* End to end: a bout inside a DLC window is absent from the output; a bout far from every window is
  present with `category="no_dlc"`, the right `rank`, `score` and `n_licks_in_window`.
* `--dlc-exclude` with `--dlc-windows` exits non-zero.
