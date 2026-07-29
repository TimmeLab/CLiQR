# ML Lick Detection — MATLAB → PyTorch Port + Transfer Learning

**Date:** 2026-07-29
**Status:** Approved design, ready for implementation planning

## Problem

Lick detection currently uses a threshold scanner (`basic_algorithm` in `data_analysis.py`,
a numpy port of the original MATLAB `lickDetector`). A separate MATLAB project
(`ML Detection MATLAB Code/`) trains a two-stage 1-D CNN cascade that detects licks more
robustly. We want that cascade available in the Python/CLiQR pipeline.

The saved MATLAB networks (`lickNets.mat`) were trained on capacitance data collected
**before** the MPR121 charge/discharge time (CDT) was increased. Raising CDT greatly
increased the magnitude (deflection depth) of the capacitance signal. The *shape* of a lick
deflection is essentially unchanged, but the amplitude scale is not — and the networks bake in
a **single global zscore normalization scalar** (netBout mean −6.87 / std 8.22; netPoint mean
−5.00 / std 8.97) fit to the old scale. Those constants are now wrong for new recordings.

Because the deflection shape is preserved, **transfer learning** (reuse the trained
convolutional/FC weights, refit normalization, fine-tune) should adapt the networks to the new
data efficiently, rather than retraining from scratch. We assume we will label additional
high-CDT training data.

## Goals

- Reproduce the MATLAB two-net cascade **exactly** in PyTorch (so the ported weights are
  meaningful).
- Port the trained weights from `lickNets.mat` into the PyTorch models.
- Adapt to new high-CDT data via transfer learning: refit zscore normalization + fine-tune all
  layers at low learning rate.
- Provide a Python labeling tool (Solara) to curate new training data, seeded by the existing
  threshold detector.
- Expose ML detection through the existing `filter_data(..., algorithm='ml')` flag, writing the
  same per-animal HDF5 datasets so downstream analysis is unchanged.

## Non-goals

- Replacing the threshold detector as the default (stays `basic_threshold`; ML is opt-in via the
  flag, may become default or be combined later).
- Bit-identical retraining of the MATLAB networks (we fine-tune; architecture is exact, weights
  become new).
- GPU training (networks and data are tiny; CPU is sufficient).
- Any change to the recording GUI or downstream plotting.

## Background: the MATLAB cascade (what we are reproducing)

Signal is resampled to **100 Hz** and each analysis window is offset so `max = 0`
(`y - max(y)`; deflections go negative).

**netBout** — coarse gate. Input = whole 3 s window (300 samples). Question: is there ≥1 lick in
the central 1 s?

```
imageInput[1x300x1], zscore (scalar mean/std)
conv2d[1x15]x16, pad same -> batchnorm -> relu
maxpool[1x4] stride[1x4]
conv2d[1x15]x32, pad same -> batchnorm -> relu
fullyConnected(32) -> relu -> fullyConnected(2) -> softmax
```

**netPoint** — fine classifier. Input = 21 samples (0.21 s). Question: is the center sample a
lick?

```
imageInput[1x21x1], zscore (scalar mean/std)
conv2d[1x5]x16, pad same -> batchnorm -> relu
conv2d[1x5]x32, pad same -> batchnorm -> relu
fullyConnected(32) -> relu -> fullyConnected(2) -> softmax
```

MATLAB's `[1xk]` conv2d over a `1xNx1` "image" is a 1-D convolution.

**Inference** (`detectLicksFromRaw`): slide the 3 s bout window with a 0.5 s step; on positive
windows, run netPoint centered on every sample; merge lick points within 20 ms; cluster center
is the lick time.

`meta` in `lickNets.mat`: fs=100, winSec=3, winSamples=300, centerSec=1, pointWin=21.

## Architecture

New in-repo package (committed, present on clone), selected by the `algorithm='ml'` flag — not a
separately installed opt-in package.

```
ml_detection/
  __init__.py
  nets.py         # LickBoutNet, LickPointNet — exact MATLAB topology in PyTorch
  weights_io.py   # read lickNets.mat (h5py) -> load conv/BN/FC + zscore scalars
  preprocess.py   # resample -> 100 Hz, per-window max-offset (shared train + infer)
  dataset.py      # h5 cap_data/time_data -> segments + labels; torch Dataset
  bootstrap.py    # seed labels via existing basic_algorithm threshold detector
  train.py        # refit normalization, fine-tune all layers, save checkpoints
  infer.py        # vectorized cascade -> lick_times; detect_licks() entry point
  labeler/app.py  # Solara curation app
  checkpoints/    # ported + fine-tuned weights (.pt) + norm constants + meta
```

Each module has one clear responsibility and a narrow interface:

- `nets.py` — pure model definitions, no I/O.
- `weights_io.py` — `.mat` → state_dict; depends on `nets.py` for shapes.
- `preprocess.py` — deterministic signal transforms, no torch; shared so train and infer see
  identical inputs.
- `dataset.py` / `bootstrap.py` — build/curate labeled segments.
- `train.py` — fine-tuning driver.
- `infer.py` — the public `detect_licks()`; the only module `data_analysis.py` imports.
- `labeler/` — standalone Solara app, reads/writes curated HDF5.

### Integration point

In `data_analysis.py::filter_data` (currently dispatches `basic_threshold` / `hilbert`), add:

```python
elif algorithm == 'ml':
    missing_data = ml_algorithm(data_by_animal, filtered_h5f, logfile)
```

`ml_algorithm` mirrors `basic_algorithm`'s I/O contract: it writes the same per-animal datasets
(`lick_times`, `lick_indices`, and whatever else `basic_algorithm` emits) into `filtered_h5f`,
so downstream analysis and plotting need zero changes. Default remains `basic_threshold`.

## Component detail

### nets.py

`LickBoutNet`: `Conv1d(1,16,15,padding='same') -> BatchNorm1d -> ReLU -> MaxPool1d(4,4) ->
Conv1d(16,32,15,padding='same') -> BatchNorm1d -> ReLU -> Flatten -> Linear(_,32) -> ReLU ->
Linear(32,2)`.

`LickPointNet`: `Conv1d(1,16,5,padding='same') -> BatchNorm1d -> ReLU ->
Conv1d(16,32,5,padding='same') -> BatchNorm1d -> ReLU -> Flatten -> Linear(_,32) -> ReLU ->
Linear(32,2)`.

zscore normalization is the first op in `forward`: `(x - mean) / std`, with `mean`/`std`
registered as buffers so they save/load with the model and are refit during transfer learning.
softmax is applied outside for probabilities (CrossEntropyLoss consumes logits in training).

### weights_io.py — porting `lickNets.mat`

`.mat` v7.3 is HDF5; read with h5py (already a dependency). Mapping:

| MATLAB field | shape | PyTorch target |
|---|---|---|
| conv `Weights` | `[1, k, inC, outC]` | Conv1d weight `[outC, inC, k]` (permute; both cross-correlate, no kernel flip) |
| conv `Bias` | `[outC]` | Conv1d bias |
| BN `TrainedMean`, `TrainedVariance`, `Scale`, `Offset`, `Epsilon` | per-channel | BatchNorm1d `running_mean`, `running_var`, `weight`, `bias`, `eps` |
| FC `Weights` | `[out, in]` | Linear weight (see flatten-order fix) |
| FC `Bias` | `[out]` | Linear bias |
| input layer `Mean` / `Std` | scalar | zscore buffers |

**Critical flatten-order fix:** MATLAB flattens the conv output column-major as `[1, W, C]`;
PyTorch `Flatten` is row-major `[C, W]`. The first `Linear`'s input axis must be reindexed from
MATLAB `(W, C)` ordering to PyTorch `(C, W)` ordering, or the port silently produces garbage.
Implement as a permutation of the FC weight columns.

### preprocess.py

- Resample irregular `(time, cap)` to uniform 100 Hz via linear interpolation with extrapolation
  (`np.interp` / `scipy`), matching MATLAB `resampleCapacitance`.
- Offset `y - max(y)` so `max = 0`.

**Guiding principle: faithful-first.** The initial implementation reproduces the MATLAB behavior
*exactly as written*, including its quirks, so the parity gate can pass. Every offset convention
below matches MATLAB verbatim:

- Bout net input: per-**bout-window** offset (`y(win) - max(y(win))`).
- Point net input **at inference**: **global** offset (`y - max(y)` over the whole recording).
- Point net windows **at training**: per-**segment** offset (each training segment offset by its
  own max, as `buildInitialTrainingSet` does).

This means the MATLAB point net is trained under one offset convention and inferred under another
(a latent train/inference mismatch). We deliberately preserve it for the faithful port, and log
it as a deferred improvement to test (see Potential Improvements). Offset conventions live in
`preprocess.py` behind named functions so the improvement experiment is a one-line swap.

### bootstrap.py + dataset.py

1. Read ACG-26-3 2026-07 recordings (`Lickometry Data/results_combined_ACG-26-3_2026-07*.h5` and
   related), per sensor: `cap_data` (int64), `time_data` (float64).
2. Resample + offset via `preprocess.py`.
3. Seed lick marks by running the existing `basic_algorithm` threshold detector on each recording
   (far better starting point than MATLAB's findpeaks).
4. Sample 3 s windows (300 samples), central 1 s labeling region, using the same three categories
   as `buildInitialTrainingSet` (0 / 1-3 / >=4 licks in center) for balance.
5. Persist a training struct to **HDF5** (our own schema): `samples`, `t`, `lickIdx`,
   `labelsBout`, `fs`, `winSec`, `centerSec`, plus provenance (source recording, sensor, cycle).

### labeler/app.py — Solara curation tool

Port `lickLabelerGUI` behavior to Solara (already in the env; matches cliqr-gui):

- Show one 3 s segment with the capacitance trace and red lick markers.
- Click on plot: add a lick at nearest sample, or select the nearest existing lick.
- Arrow keys / buttons: nudge selected lick +/-1 sample.
- Delete/remove selected lick.
- Prev/Next navigation across segments; list of lick times for the current segment.
- On every edit, recompute `labelsBout` = (>=1 lick in central 1 s), exactly as MATLAB.
- Save writes the curated segments back to HDF5.

### train.py — transfer learning

1. Load ported weights into both networks.
2. **Refit zscore:** recompute a single scalar mean/std over the new curated segments; overwrite
   the input-norm buffers. This is the primary fix for the CDT magnitude change.
3. **Split by recording/session** (hold out whole sessions for validation) to avoid leakage
   between adjacent overlapping windows.
4. Fine-tune **all** layers at a low learning rate (e.g. Adam 1e-4) with early stopping on the
   held-out split. Bout net trains on 3 s segments; point net trains on 21-sample windows over
   the central 1 s of positive segments (port of `preparePointSegments`).
5. Optional dataset transforms ported from MATLAB: `balanceTrainingByLickLocation` (central-lick
   vs no-central-lick, variance-spread negatives) and `makeThreeSecondTraining`-style
   augmentation.
6. Save fine-tuned `.pt` (state_dict incl. norm buffers) + meta (fs, winSec, pointWin, training
   provenance) to `checkpoints/`.

### infer.py — vectorized cascade

`detect_licks(cap_data, time_data, net_bout, net_point) -> lick_times`:

1. Resample -> 100 Hz, offset.
2. **Bout gate:** slide 3 s window, step 0.5 s (as MATLAB), but batch all windows through
   `LickBoutNet` in a single forward pass -> per-window lick/no-lick.
3. **Point pass (vectorized):** instead of MATLAB's per-sample re-classify loop, gather all
   21-sample windows centered on every sample inside positive bout spans as one batch (or run
   `LickPointNet` stride-1 over the positive span), single forward pass -> per-sample lick mask.
   This is the one intentional deviation from MATLAB-as-written, but it is **output-identical**
   (same inputs, same labels) — a pure speed optimization, not a behavior change — and validation
   gate #2 confirms it reproduces the loop's result. Overlapping positive bout windows are reduced
   to their **union** before the batch: because the point net input uses the global offset, a
   sample's label is independent of which bout window reached it, so the union is lossless and
   matches MATLAB's OR semantics (`lickMaskGlobal` is set and never unset).
4. Merge mask points within 20 ms; cluster center = lick time (port of MATLAB merge logic).
5. Map times back to the original time base; return.

`ml_algorithm` wraps `detect_licks` per animal and writes the same HDF5 datasets as
`basic_algorithm`.

## Validation (gates, in order)

1. **Weight-port parity.** Generate reference `(input, softmax-output)` pairs from the real
   MATLAB networks using headless MATLAB
   (`/Applications/MATLAB_R2025a.app/bin/matlab -batch "..."`; MATLAB R2025a is installed).
   Assert the PyTorch port matches to ~1e-4. **Must pass before any fine-tuning** — fine-tuning
   from mis-loaded weights is worthless.
2. **Pre-fine-tune sanity.** The ported (un-fine-tuned) net on an *old-CDT* recording reproduces
   the old MATLAB detection behavior.
3. **Post-fine-tune metrics.** Precision / recall / F1 on held-out sessions against curated
   labels; compare to the threshold detector on the same sessions.
4. **Timing spot-check.** Overlay ML lick times on the trace (and sync video where available) for
   a few bouts; confirm no systematic offset.

## Testing

- Unit tests, mirroring the existing `tests/` layout:
  - `preprocess`: resample invariants (length, endpoint handling), offset correctness.
  - `weights_io`: shape mapping, flatten-order permutation, parity against MATLAB reference
    vectors.
  - `infer`: bout-window slicing, point-window gathering equals the naive loop on a small
    fixture, 20 ms merge logic.
- Small fixture HDF5 recording committed for fast tests.

## Dependencies

- **New:** `torch` (CPU wheel) added to `requirements.txt` and `environment.yml`.
- Already present and reused: `h5py` (read `.mat` v7.3 and project h5), `numpy`, `scipy`
  (resample / threshold detector), `matplotlib`, `solara` / `panel` / `ipywidgets` (labeler).

## Potential improvements (deferred)

The first implementation is a faithful MATLAB port. While planning/building, log candidate
improvements here rather than acting on them, so the parity gate stays meaningful. Each is a
post-parity experiment, measured on held-out-session F1.

1. **Fix the point-net train/inference offset mismatch (prioritized to test).** MATLAB trains the
   point net on per-segment-offset windows but infers under global offset. Experiment: use the
   **per-segment offset for BOTH training and inference** and compare F1 to the faithful port.
   Requires the offset functions in `preprocess.py` to be swappable (already specified).
2. **Bout-gate overlap resolution.** MATLAB resolves overlapping bout windows with OR (any
   positive window makes the span positive). Alternative: `gate_mode='vote'` requiring a majority
   of covering bout windows to agree — more conservative, possibly better generalization. Add as
   a flag after parity.
3. **Reuse of prior ML artifacts.** The repo already contains `checkpoints/best.pt` and training
   data from earlier, un-integrated ML experiments (per project CLAUDE.md). During planning,
   inspect these — the checkpoint architecture, any labeled data, and how `best.pt` was produced —
   before generating fresh labels; they may seed or replace parts of this pipeline.
4. **Combine ML with the threshold detector.** Longer term, ML could become the default or be
   fused with `basic_algorithm` (e.g. ML gates, threshold refines timing, or ensemble). Out of
   scope for the faithful port.

## Open questions / risks

- **Flatten-order fix** is the highest-risk detail; the parity gate (validation #1) is designed
  specifically to catch it.
- Amount of new labeled data needed for stable fine-tuning is unknown; start with ACG-26-3
  2026-07 sessions and expand if held-out F1 is unsatisfactory.
- BatchNorm behavior during fine-tuning on a small dataset: consider freezing BN running stats or
  using a low momentum; decide during implementation based on validation.
