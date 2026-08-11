# DLC windows -> synced videos (`--dlc-windows`)

Date: 2026-08-11
Status: approved, ready for implementation plan

## Problem

`dlc_integration/find_dlc_windows.py` already finds the stretches of a session video where the
animal is confidently at the sipper (and, with `--require-tongue`, actually drinking). It writes one
row per window to a CSV. Those windows are exactly the ones worth reviewing as side-by-side
video + capacitance clips, but today the only way to get such a clip is
`scripts/find_interesting_windows.py`, which finds its own windows from the capacitance trace
(busiest bouts, high-variance non-bout stretches) and knows nothing about DLC.

The two selections disagree by construction: DLC gates on pose, the trace search gates on
capacitance. Reviewing the DLC windows in a synced clip is how we see whether the capacitance
detector agreed with what the animal was actually doing.

The DLC *labeled* videos (`create_labeled_video` output) are not involved. We only read the CSV and
render from the original recording, so the clip shows the raw footage next to the trace.

## Solution overview

Add `--dlc-windows PATH` to `scripts/find_interesting_windows.py`. When given, the script runs in
**pure DLC mode**: it does not search the trace at all; every emitted region comes from a row of the
DLC CSV, converted from video frame indices into session seconds.

Everything downstream is unchanged -- the same `rois.csv` writer, the same `make_clips.sh` writer,
the same `make_sync_video.py` command shape.

## CLI

```
python scripts/find_interesting_windows.py <combined_h5> \
    --dlc-windows second_iteration_windows_with_tongue.csv \
    --csv dlc_rois.csv --sh make_dlc_clips.sh --speed 0.25
```

* `--dlc-windows PATH` -- CSV written by `dlc_integration/find_dlc_windows.py`.
* The `combined_h5` positional stays **required**. It supplies:
  * the per-cycle `raw_h5` / `layout` provenance attributes,
  * the filmed animal (via `resolve_filmed_animal`),
  * the trace the emitted commands read through `--combined-h5` / `--cycle`,
  * `lick_times`, used to annotate each DLC window with how many detected licks fall inside it.
* Ignored in DLC mode (they only parameterise the trace search): `--n-lick`, `--n-climb`,
  `--roi-seconds`, `--lick-pad`, `--climb-skip-edges`, `--var-window`, `--min-var`. Passing one
  alongside `--dlc-windows` is not an error; the value is simply unused.
* Still honoured: `--csv`, `--sh`, `--out-dir`, `--animals`, `--raw-map`, `--offset`, `--speed`.
  `--animals` filters on the resolved (filmed) animal of each mapped video.
* Also unused in DLC mode: `--include-controls`. A DLC row's animal is the cycle's filmed animal,
  which is never a control cage.
* No new pass-through flags for `make_sync_video` (`--window`, `--sync-offset`, `--fps`,
  `--keep-intermediate`). `--speed` already passes through and is enough.

## Frame -> session time

This is the only genuinely new computation.

A DLC row's `start_frame` / `end_frame` are half-open indices into the **original container**
(`Frames2plot=range(start, end)`), and its `start_sec` / `end_sec` columns are just `frame / fps` --
video-file time, which is NOT the reference `make_sync_video --start/--end` expect (session seconds
since the Start bookmark). Convert with exactly the clock the renderer uses to place frames:

```python
anchor = read_video_anchor(raw_h5)
video, pts_txt = resolve_paths(raw_h5, anchor)
pts_ns = np.loadtxt(pts_txt, dtype=np.int64)
clock = session_clock(anchor, pts_ns)                       # latency + drift slope
sess = frame_session_times(clock, load_container_pts(pts_txt, pts_ns))
start_s = sess[start_frame]
end_s = sess[end_frame - 1]        # end_frame is EXCLUSIVE in the DLC CSV
```

`load_container_pts` (imported from `make_sync_video`) prefers the Pi's `<stem>.encpts.txt`, which
has one line per ENCODED frame, so encoder drops are already excluded and container ordinal `k`
indexes it directly. Falling back to the capture sidecar `<stem>.txt` is what
`make_sync_video` does for recordings made before drop-marking, so the two stay consistent.

Because the window is derived from the same `SessionClock` and the same sidecar that
`make_sync_video` uses to time frames, the frames DLC scored are the frames the clip shows. Do not
recompute this from `start_sec` and a nominal fps: the sidecar clock drifts against the container's
constant rate by seconds over a session (see `docs/video-sync-alignment-bugs.md`).

`resolve_paths` / `read_video_anchor` are called ONCE per video (per cycle), not per row.

## Row -> cycle mapping

DLC CSV rows carry a `video` path (often a cluster path that does not exist locally), never the raw
`.h5`. Map by basename:

1. `basename(row["video"])` -> stem, e.g. `raw_data_2026-07-24_12-02-14_cfr.mp4` ->
   `raw_data_2026-07-24_12-02-14_cfr`.
2. Strip one trailing `_cfr` -> `raw_data_2026-07-24_12-02-14`.
3. Match against `basename(raw_h5)` without extension, for every cycle in the combined file (the
   `raw_h5` attribute, or the `--raw-map` fallback).

Verified against the current data: the combined file's cycles 0-5 are the 07-22 .. 07-29
recordings, and `second_iteration_windows.csv` rows for those dates map cleanly.

The animal for a mapped row is that cycle's filmed animal. Rows are grouped by video so each
video's anchor/pts work happens once.

## Skips

Nothing is silently dropped: every skip is counted by reason and the counts are printed at the end.

| Case | Granularity | Why |
|---|---|---|
| stem ended in `_cfr` | whole video | The `_cfr.mp4` files are re-encodes; their frame indices are not guaranteed 1:1 with the original container (`-fps_mode cfr` drops ~1 frame per 25k), so a frame index cannot be trusted as a container ordinal. Those sessions (07-13/14/16/21) also have no local mp4 or pts sidecar, so no clip could be rendered anyway. |
| no matching cycle in the combined file | whole video | Nothing to read a trace from. |
| `raw_h5`, mp4, or pts sidecar missing on disk | whole video | `make_sync_video` could not run. |
| `start_frame >= len(sess)` | single row | Frame index past the end of the sidecar. |
| window entirely outside the cycle's `time_data` span | single row | Nothing to plot. |
| window partially outside that span | single row, clamped | Keeps the clip inside the trace, same as the trace-search path's `clip_window`. |

One more group-level skip, `no_trace`: if the cycle has no trace to draw -- because its filmed
animal cannot be resolved (`resolve_filmed_animal` returns `None`, e.g. the layout does not name
the video sensor), or that animal's group has no usable `time_data` -- the whole video's rows are
skipped under their own reason.

That reason exists to keep the report honest. Such a cycle carries `first_s == span_s == 0.0`, so
without it every one of its windows would fail the clamp and be counted as "the window falls
outside the cycle's trace" -- pointing the reader at DeepLabCut when the actual fault is animal or
layout resolution.

## Output

DLC regions are ordinary ROI dicts, so `write_csv` and `write_shell_script` are untouched:

| field | value in DLC mode |
|---|---|
| `category` | `"dlc"` |
| `rank` | 0-based row order within the (animal, cycle), following CSV order |
| `start` / `end` | converted session seconds, clamped to the trace span |
| `center` | midpoint of `start` / `end` |
| `score` | the row's `tongue_rate` (0.0 when the column is empty) |
| `n_licks_in_window` | detected licks inside the window, from the cycle's `lick_times` |

`CSV_COLUMNS` does not change.

Windows are used **as-is**: `find_dlc_windows.py --pad` already added context, and no cap is applied
(the upstream `--require-tongue` / `--min-frames` filters are where selection belongs). A 184-row
CSV yields up to 184 commands.

Clips keep the crop -- the DLC gate is nose-near-sipper, i.e. the same footage the licking clips
want, where the crop is what makes the tongue visible. Only `category == "climb"` gets `--no-crop`,
which DLC mode never produces. Output names come from the existing `build_command`:
`{animal}_c{cycle}_dlc{rank}.mp4`.

The restart-recording WARNING comment still fires, but its ADVICE differs by mode. A DLC window's
start/end are derived from the video clock, which is already `make_sync_video`'s reference, so
`--offset CYCLE=SECONDS` is the wrong remedy here: it shifts start and end together and would move
the clip off the DLC window entirely, while doing nothing about a trace/video mismatch. The DLC
warning therefore points at `make_sync_video --sync-offset` instead. Trace mode's wording is
untouched.

## Testing

New tests in `tests/test_find_interesting_windows.py`, all against pure functions with synthetic
inputs (no recordings, no ffmpeg):

* frame -> session conversion matches `frame_session_times` for a clock with latency and a drift
  slope != 1.0.
* `end_frame` is treated as exclusive: a row `[10, 20)` ends at `sess[19]`.
* stem -> cycle matching works with and without the `_cfr` suffix, and returns nothing for an
  unknown stem.
* a `_cfr` video is skipped as a group and counted.
* a video with no matching cycle, and one whose pts sidecar is absent, are each skipped and counted.
* a row whose `start_frame` exceeds the sidecar length is skipped; a row overlapping the end of the
  trace is clamped.
* a DLC row builds a command that keeps the crop (no `--no-crop`) and carries `--speed` when set.
* DLC mode emits no `lick` or `climb` rows.

## Non-goals

* No changes to `dlc_integration/find_dlc_windows.py` or its CSV schema.
* No use of the DLC labeled videos.
* No re-running of DLC inference, lick detection, or `filter_data`.
* No support for rendering `_cfr` sessions; if that is ever wanted, the fix is to re-run DLC on the
  original mp4 (as was already done for 07-22 onward), not to guess a frame mapping.
