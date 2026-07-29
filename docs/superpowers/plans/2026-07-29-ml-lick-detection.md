# ML Lick Detection (PyTorch Port + Transfer Learning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the MATLAB two-net lick-detection cascade to PyTorch, load the trained weights, adapt to new high-CDT data via transfer learning, and expose it through `filter_data(algorithm='ml')`.

**Architecture:** A new in-repo `ml_detection/` package holds exact PyTorch reproductions of the MATLAB `netBout` (3 s / 300-sample gate) and `netPoint` (21-sample classifier). Weights are ported from `lickNets.mat`; a MATLAB parity gate confirms the port before any training. Inference runs a vectorized cascade producing the same per-animal HDF5 datasets the existing threshold detector writes. A Solara labeler and a fine-tuning script build/curate new-scale training data.

**Tech Stack:** Python 3.13, PyTorch 2.13.0 (CPU), h5py, numpy, scipy, Solara, pytest. MATLAB R2025a (headless `-batch`) as the parity oracle only.

## Global Constraints

- **Faithful-first.** The initial port reproduces MATLAB behavior *exactly as written*, including its offset quirks, so the parity gate is meaningful. Improvements are logged in the spec's "Potential improvements" section, NOT implemented here.
- **Offset conventions (verbatim MATLAB):** bout net input = per-bout-window offset `y(win) - max(y(win))`; point net input at **inference** = **global** offset `y - max(y)`; point net windows at **training** = per-segment offset. Keep each behind a named function in `preprocess.py`.
- **Network meta (from `lickNets.mat`):** `fs=100`, `winSec=3`, `winSamples=300`, `centerSec=1`, `pointWin=21`. Bout step = 0.5 s. Point merge threshold = 20 ms.
- **CDT cutoff = 2026-07-22.** Fine-tune data = files dated `2026-07-22` and onward (new scale). Old-scale files (`2026-07-21`, `2026-07-09`) are the oracle for the pre-fine-tune sanity gate. Never mix old-scale files into the new-CDT training set.
- **Default detector unchanged.** `filter_data`'s default stays `algorithm='basic_threshold'`. ML is opt-in via `algorithm='ml'`.
- **Output contract.** `ml_algorithm` writes the same per-animal datasets as `basic_algorithm` via `save_filtered_data`: `data['lick_times']`, `data['lick_indices']`, `data['num_licks']`.
- **Branch:** `feature/ml-lick-detection`. Commit after every task.
- **MATLAB binary:** `/Applications/MATLAB_R2025a.app/bin/matlab`. The first deep-learning `predict()` call is very slow (toolbox load + JIT) — allow a multi-minute timeout when generating parity vectors.
- **Test style:** mirror `tests/` — `pytest`, one `test_*.py` per module, numpy fixtures, no network/hardware.

---

## File Structure

- Create `ml_detection/__init__.py` — package marker, exports `detect_licks`.
- Create `ml_detection/nets.py` — `LickBoutNet`, `LickPointNet` (pure model defs).
- Create `ml_detection/weights_io.py` — read `lickNets.mat`, build state_dicts, flatten-order fix.
- Create `ml_detection/preprocess.py` — resample + offset functions (no torch).
- Create `ml_detection/infer.py` — vectorized cascade, `detect_licks()`.
- Create `ml_detection/dataset.py` — HDF5 training-set schema read/write, point-segment builder.
- Create `ml_detection/bootstrap.py` — seed labels via `basic_algorithm`, build initial segments.
- Create `ml_detection/train.py` — refit zscore, session split, fine-tune, save checkpoints.
- Create `ml_detection/labeler/app.py` — Solara curation UI.
- Create `ml_detection/checkpoints/` — ported + fine-tuned weights (`.pt`), plus `parity_refs.npz`.
- Modify `data_analysis.py` — add `ml_algorithm` + `algorithm=='ml'` dispatch in `filter_data`.
- Create `tests/test_ml_nets.py`, `tests/test_ml_weights_io.py`, `tests/test_ml_parity.py`, `tests/test_ml_preprocess.py`, `tests/test_ml_infer.py`, `tests/test_ml_integration.py`, `tests/test_ml_dataset.py`, `tests/test_ml_train.py`.
- Create `scripts/matlab_parity_refs.m` — MATLAB script dumping reference `(input, output)` pairs.

---

## Task 1: Package scaffold + network definitions

**Files:**
- Create: `ml_detection/__init__.py`, `ml_detection/nets.py`
- Test: `tests/test_ml_nets.py`

**Interfaces:**
- Produces:
  - `class LickBoutNet(nn.Module)` — `forward(x: Tensor[N,1,300]) -> Tensor[N,2]` logits. Buffers `norm_mean`, `norm_std` (scalars).
  - `class LickPointNet(nn.Module)` — `forward(x: Tensor[N,1,21]) -> Tensor[N,2]` logits. Buffers `norm_mean`, `norm_std`.
  - Both apply `(x - norm_mean) / norm_std` as the first forward op.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ml_nets.py
import torch
from ml_detection.nets import LickBoutNet, LickPointNet


def test_bout_net_output_shape():
    net = LickBoutNet()
    x = torch.randn(4, 1, 300)          # [batch, channels, samples]
    y = net(x)
    assert y.shape == (4, 2)            # 2-class logits per window


def test_point_net_output_shape():
    net = LickPointNet()
    x = torch.randn(7, 1, 21)
    y = net(x)
    assert y.shape == (7, 2)


def test_norm_buffers_apply():
    # With mean=0, std=1 the normalization is identity; a hand-set mean shifts input.
    net = LickPointNet()
    net.norm_mean.fill_(0.0)
    net.norm_std.fill_(1.0)
    x = torch.zeros(1, 1, 21)
    # forward must run without error and produce finite logits
    y = net(x)
    assert torch.isfinite(y).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml_nets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_detection'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml_detection/__init__.py
"""ML lick-detection package: PyTorch port of the MATLAB netBout/netPoint cascade."""
```

```python
# ml_detection/nets.py
"""
PyTorch reproductions of the MATLAB lick-detection networks.

MATLAB fakes 1-D convolution with a `convolution2dLayer([1 k], nFilters)` acting on a
`1 x N x 1` "image". That is exactly an `nn.Conv1d` with kernel length k. We reproduce
the two networks layer-for-layer so weights ported from `lickNets.mat` are meaningful.

Both networks begin with a zscore normalization using a SINGLE scalar mean/std, matching
MATLAB's `imageInputLayer(..., 'Normalization','zscore')` as trained. The scalars are stored
as buffers so they save/load with the model and can be refit during transfer learning.
"""
import torch
import torch.nn as nn


class LickBoutNet(nn.Module):
    """Coarse gate: does a 3 s (300-sample) window contain >=1 lick in its central 1 s?"""

    def __init__(self):
        super().__init__()
        # zscore constants (scalar); identity by default until real weights are loaded.
        self.register_buffer("norm_mean", torch.zeros(1))
        self.register_buffer("norm_std", torch.ones(1))

        # conv2d[1x15]x16 -> BN -> ReLU -> maxpool[1x4] stride4 -> conv2d[1x15]x32 -> BN -> ReLU
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, padding="same")
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, padding="same")
        self.bn2 = nn.BatchNorm1d(32)
        # After pool: 300 -> 75 samples, 32 channels -> flatten 32*75 = 2400
        self.fc1 = nn.Linear(32 * 75, 32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = (x - self.norm_mean) / self.norm_std
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class LickPointNet(nn.Module):
    """Fine classifier: is the CENTER sample of a 21-sample window a lick?"""

    def __init__(self):
        super().__init__()
        self.register_buffer("norm_mean", torch.zeros(1))
        self.register_buffer("norm_std", torch.ones(1))

        # conv2d[1x5]x16 -> BN -> ReLU -> conv2d[1x5]x32 -> BN -> ReLU (NO pooling: needs precise localization)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(32)
        # No pool: 21 samples preserved, 32 channels -> flatten 32*21 = 672
        self.fc1 = nn.Linear(32 * 21, 32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = (x - self.norm_mean) / self.norm_std
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml_nets.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ml_detection/__init__.py ml_detection/nets.py tests/test_ml_nets.py
git commit -m "feat(ml): PyTorch netBout/netPoint definitions matching MATLAB topology"
```

---

## Task 2: Weight porting from lickNets.mat

**Files:**
- Create: `ml_detection/weights_io.py`
- Test: `tests/test_ml_weights_io.py`

**Interfaces:**
- Consumes: `LickBoutNet`, `LickPointNet` from Task 1.
- Produces:
  - `load_matlab_nets(mat_path: str) -> tuple[LickBoutNet, LickPointNet]` — returns both nets with ported weights + norm scalars loaded, in `.eval()` mode.
  - `permute_fc_for_flatten(W_matlab: np.ndarray, n_channels: int, width: int) -> np.ndarray` — reorders a MATLAB FC weight matrix `[out, C*W]` from MATLAB `(W, C)` column-major flatten order to PyTorch `(C, W)` row-major order.

**Background for the implementer:** `lickNets.mat` is HDF5 (MATLAB v7.3). The trained `DAGNetwork` objects store layer arrays under `#refs#`. Each conv layer has `Weights [1,k,inC,outC]` and `Bias`; each batchnorm has `TrainedMean`, `TrainedVariance`, `Scale`, `Offset`, `Epsilon`; each FC has `Weights [out,in]` and `Bias`; the input layer's `Normalization` group has scalar `Mean` and `Std`. Because the exact `#refs#` layout is opaque, this task first writes a small explorer, then hardcodes the discovered ref paths.

- [ ] **Step 1: Explore the .mat layer layout (one-off, keep output in a comment)**

Run this and record the ref keys in a comment at the top of `weights_io.py`:

```bash
cd "ML Detection MATLAB Code" && python - <<'EOF'
import h5py, numpy as np
f = h5py.File('lickNets.mat', 'r')
names = []
f.visit(names.append)
for n in names:
    o = f[n]
    if isinstance(o, h5py.Dataset) and any(s in n for s in
        ['Weights','Bias','TrainedMean','TrainedVariance','Scale','Offset','Epsilon','Mean','Std']):
        print(n, o.shape, o.dtype)
EOF
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_ml_weights_io.py
import os
import numpy as np
import torch
import pytest
from ml_detection.weights_io import load_matlab_nets, permute_fc_for_flatten

MAT = os.path.join("ML Detection MATLAB Code", "lickNets.mat")


def test_permute_fc_for_flatten_reorders_columns():
    # 2 channels, 3 width. MATLAB flattens column-major as (w0c0..w2c0? ) -> our helper
    # must map MATLAB (W,C) ordering to PyTorch (C,W). Build a weight whose columns are
    # labeled by (c, w) so we can verify the permutation.
    n_out, C, W = 1, 2, 3
    # MATLAB order index = c*W + w  (channel-major here is what MATLAB produces for [1,W,C] col-major)
    matlab = np.arange(C * W, dtype=float).reshape(n_out, C * W)
    out = permute_fc_for_flatten(matlab, n_channels=C, width=W)
    assert out.shape == (n_out, C * W)
    # After reordering to PyTorch (C, W) row-major flatten, column k corresponds to (k//W, k%W)
    # The helper is its own inverse-consistent mapping; assert it is a pure permutation.
    assert sorted(out.ravel().tolist()) == sorted(matlab.ravel().tolist())


@pytest.mark.skipif(not os.path.exists(MAT), reason="lickNets.mat not present")
def test_load_matlab_nets_shapes_and_norm():
    bout, point = load_matlab_nets(MAT)
    # Norm scalars from the .mat (netBout mean approx -6.87 / std 8.22; point -5.0 / 8.97)
    assert bout.norm_mean.item() == pytest.approx(-6.8734145, abs=1e-3)
    assert bout.norm_std.item() == pytest.approx(8.215723, abs=1e-3)
    assert point.norm_mean.item() == pytest.approx(-4.9988613, abs=1e-3)
    # Forward runs and produces finite logits
    assert torch.isfinite(bout(torch.zeros(1, 1, 300))).all()
    assert torch.isfinite(point(torch.zeros(1, 1, 21))).all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_ml_weights_io.py -v`
Expected: FAIL (`ModuleNotFoundError` / function not defined).

- [ ] **Step 4: Write the implementation**

```python
# ml_detection/weights_io.py
"""
Port trained weights from the MATLAB `lickNets.mat` DAGNetworks into the PyTorch models.

MATLAB stores networks in HDF5 (v7.3). Conv/BN/FC parameters live under `#refs#`. The two
subtleties handled here:

1. Conv weight layout. MATLAB conv weights are `[1, k, inC, outC]`; PyTorch Conv1d wants
   `[outC, inC, k]`. Both frameworks CROSS-CORRELATE (no kernel flip), so we only permute axes.

2. The flatten-order fix. MATLAB flattens the final conv feature map column-major as `[1, W, C]`
   before the first fully-connected layer; PyTorch `flatten` is row-major `[C, W]`. If we copy the
   FC weight straight across, every input column lands on the wrong feature and the port silently
   produces garbage. `permute_fc_for_flatten` reindexes the FC input columns from MATLAB (W,C)
   order to PyTorch (C,W) order.

NOTE: fill in the exact `#refs#` dataset paths discovered by the Step-1 explorer below:
   # netBout conv1 weights: #refs#/.../Weights   (record real paths here)
   # ...
"""
import h5py
import numpy as np
import torch

from ml_detection.nets import LickBoutNet, LickPointNet


def _deref(f, name):
    """Read a dataset by name and return a numpy array with MATLAB's dim order squeezed sensibly."""
    return np.array(f[name])


def permute_fc_for_flatten(W_matlab, n_channels, width):
    """
    Reorder FC input columns from MATLAB column-major (W, C) flatten order to PyTorch (C, W).

    Parameters
    ----------
    W_matlab : np.ndarray, shape [n_out, n_channels * width]
        FC weight as stored by MATLAB, whose input axis follows MATLAB's `[1, W, C]` column-major
        flatten (fastest-varying dimension first = W, then C).
    n_channels, width : int
        Feature-map channel count and temporal width feeding the FC layer.

    Returns
    -------
    np.ndarray, shape [n_out, n_channels * width]
        FC weight whose input axis follows PyTorch row-major `[C, W]` flatten.
    """
    n_out = W_matlab.shape[0]
    # MATLAB column index for (channel c, width w): matlab_idx = c * width + w  when the stored
    # matrix already lists channel-major; we instead reconstruct the (C, W) grid and re-flatten.
    reshaped = W_matlab.reshape(n_out, n_channels, width)   # interpret as (out, C, W)
    # PyTorch flatten over (C, W) is exactly (out, C, W) row-major -> already correct grid order,
    # so the permutation is the identity re-flatten. If the explorer shows MATLAB stored (W, C),
    # swap axes here: reshaped = W_matlab.reshape(n_out, width, n_channels).transpose(0, 2, 1)
    return reshaped.reshape(n_out, n_channels * width)


def _load_conv(f, weights_name, bias_name):
    W = _deref(f, weights_name)          # MATLAB [1, k, inC, outC] (may come transposed via h5py)
    W = np.squeeze(W)                    # -> [k, inC, outC] or similar; normalize below
    # h5py reads MATLAB arrays with reversed dims: stored [outC, inC, k, 1]. Squeeze then move to
    # [outC, inC, k]. Assert final shape against the Conv1d layer at call site.
    b = np.squeeze(_deref(f, bias_name))
    return W, b


def load_matlab_nets(mat_path):
    """
    Build LickBoutNet and LickPointNet with weights ported from `lickNets.mat`.

    Returns both nets in eval() mode. Raises AssertionError if any tensor shape does not match the
    target layer (guards against a wrong `#refs#` path).
    """
    f = h5py.File(mat_path, "r")
    bout = LickBoutNet()
    point = LickPointNet()

    # For each network, load: zscore Mean/Std -> buffers; conv Weights/Bias -> conv layers with
    # axis permutation to [outC,inC,k]; BN TrainedMean/Variance/Scale/Offset/Epsilon -> BatchNorm1d
    # running_mean/running_var/weight/bias/eps; FC Weights/Bias with permute_fc_for_flatten on the
    # FIRST fc only. Exact ref paths come from the Step-1 explorer; wire them here with asserts:
    #
    #   with torch.no_grad():
    #       bout.norm_mean.fill_(float(_deref(f, '<bout input Mean path>')))
    #       bout.norm_std.fill_(float(_deref(f, '<bout input Std path>')))
    #       w, b = _load_conv(f, '<bout conv1 W>', '<bout conv1 b>')
    #       assert w.shape == (16, 1, 15), w.shape
    #       bout.conv1.weight.copy_(torch.tensor(w, dtype=torch.float32))
    #       bout.conv1.bias.copy_(torch.tensor(b, dtype=torch.float32))
    #       ... bn1, conv2, bn2 ...
    #       fc1_w = permute_fc_for_flatten(_deref(f, '<bout fc1 W>'), n_channels=32, width=75)
    #       assert fc1_w.shape == (32, 32 * 75)
    #       bout.fc1.weight.copy_(torch.tensor(fc1_w, dtype=torch.float32))
    #       ... fc1.bias, fc2 ...
    #   (repeat for point with width=21, kernel 5)
    #
    # Implement the block above using the real paths. Keep the asserts.
    raise NotImplementedError("Wire real #refs# paths from the Step-1 explorer, then remove this.")

    bout.eval()
    point.eval()
    return bout, point
```

Then replace the `raise NotImplementedError` block with the real per-layer copies using the ref paths from Step 1, keeping every shape assert. For BatchNorm set `bn.eps = float(epsilon)` and copy `running_mean/running_var/weight(=Scale)/bias(=Offset)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ml_weights_io.py -v`
Expected: PASS. `test_load_matlab_nets_shapes_and_norm` confirms norm scalars and finite forward.

- [ ] **Step 6: Save a ported checkpoint + commit**

```bash
python - <<'EOF'
import torch
from ml_detection.weights_io import load_matlab_nets
bout, point = load_matlab_nets("ML Detection MATLAB Code/lickNets.mat")
torch.save({"bout": bout.state_dict(), "point": point.state_dict()},
           "ml_detection/checkpoints/ported_matlab.pt")
print("saved ported_matlab.pt")
EOF
git add ml_detection/weights_io.py tests/test_ml_weights_io.py ml_detection/checkpoints/ported_matlab.pt
git commit -m "feat(ml): port lickNets.mat weights into PyTorch nets with flatten-order fix"
```

---

## Task 3: MATLAB parity gate (validation #1)

**Files:**
- Create: `scripts/matlab_parity_refs.m`, `ml_detection/checkpoints/parity_refs.npz`
- Test: `tests/test_ml_parity.py`

**Interfaces:**
- Consumes: `load_matlab_nets` (Task 2).
- Produces: `parity_refs.npz` with arrays `bout_in [K,300]`, `bout_out [K,2]`, `point_in [K,21]`, `point_out [K,2]` (softmax probabilities from MATLAB).

- [ ] **Step 1: Write the MATLAB reference-dumping script**

```matlab
% scripts/matlab_parity_refs.m
% Dump (input, softmax-output) reference pairs from the trained MATLAB nets so the PyTorch
% port can be validated numerically. Uses fixed seeds for reproducibility.
S = load(fullfile('ML Detection MATLAB Code', 'lickNets.mat'));
rng(0);
K = 8;
bout_in = single(randn(K, 300));
point_in = single(randn(K, 21));
bout_out = zeros(K, 2);
point_out = zeros(K, 2);
for i = 1:K
    xb = reshape(bout_in(i, :), [1 300 1 1]);
    yb = predict(S.netBout, xb);      % softmax probabilities
    bout_out(i, :) = yb(:)';
    xp = reshape(point_in(i, :), [1 21 1 1]);
    yp = predict(S.netPoint, xp);
    point_out(i, :) = yp(:)';
end
save(fullfile('ml_detection', 'checkpoints', 'parity_refs.mat'), ...
     'bout_in', 'bout_out', 'point_in', 'point_out', '-v7');
disp('PARITY_REFS_DONE');
```

- [ ] **Step 2: Generate the references (allow a long timeout — first predict() is slow)**

Run:
```bash
/Applications/MATLAB_R2025a.app/bin/matlab -batch "run('scripts/matlab_parity_refs.m')"
```
Expected: prints `PARITY_REFS_DONE`, writes `ml_detection/checkpoints/parity_refs.mat`. Then convert to npz:
```bash
python - <<'EOF'
import scipy.io as sio, numpy as np
m = sio.loadmat("ml_detection/checkpoints/parity_refs.mat")
np.savez("ml_detection/checkpoints/parity_refs.npz",
         bout_in=m["bout_in"], bout_out=m["bout_out"],
         point_in=m["point_in"], point_out=m["point_out"])
print("npz written")
EOF
```

- [ ] **Step 3: Write the failing parity test**

```python
# tests/test_ml_parity.py
import os
import numpy as np
import torch
import pytest
from ml_detection.weights_io import load_matlab_nets

REFS = "ml_detection/checkpoints/parity_refs.npz"
MAT = "ML Detection MATLAB Code/lickNets.mat"


@pytest.mark.skipif(not os.path.exists(REFS), reason="parity refs not generated")
def test_pytorch_matches_matlab_to_1e4():
    refs = np.load(REFS)
    bout, point = load_matlab_nets(MAT)
    with torch.no_grad():
        pb = torch.softmax(bout(torch.tensor(refs["bout_in"]).unsqueeze(1)), dim=1).numpy()
        pp = torch.softmax(point(torch.tensor(refs["point_in"]).unsqueeze(1)), dim=1).numpy()
    assert np.max(np.abs(pb - refs["bout_out"])) < 1e-4
    assert np.max(np.abs(pp - refs["point_out"])) < 1e-4
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_ml_parity.py -v`
Expected: PASS. **If it fails**, the most likely cause is the flatten-order fix in Task 2 — revisit `permute_fc_for_flatten` (try the transposed reshape branch) before proceeding. This gate MUST pass before any training.

- [ ] **Step 5: Commit**

```bash
git add scripts/matlab_parity_refs.m ml_detection/checkpoints/parity_refs.npz tests/test_ml_parity.py
git commit -m "test(ml): MATLAB parity gate for the ported weights (validation #1)"
```

---

## Task 4: preprocess.py (resample + offset conventions)

**Files:**
- Create: `ml_detection/preprocess.py`
- Test: `tests/test_ml_preprocess.py`

**Interfaces:**
- Produces:
  - `resample_to_100hz(time_s: np.ndarray, cap: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — uniform 100 Hz `(t, y)` via linear interp with extrapolation (matches MATLAB `resampleCapacitance`).
  - `offset_global(y: np.ndarray) -> np.ndarray` — `y - max(y)`.
  - `offset_window(y_window: np.ndarray) -> np.ndarray` — `y_window - max(y_window)` (same math, named for intent: bout-window / point-training use).
  - `FS = 100`, `WIN_SAMPLES = 300`, `POINT_WIN = 21`, `CENTER_SAMPLES = 100`, `BOUT_STEP = 50` (samples).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ml_preprocess.py
import numpy as np
from ml_detection.preprocess import resample_to_100hz, offset_global, offset_window, FS


def test_resample_produces_uniform_100hz_grid():
    # Irregular 2 s of data; resample must land on a 0.01 s grid.
    t = np.array([0.0, 0.017, 0.031, 0.5, 1.0, 2.0])
    y = np.array([10.0, 9.0, 8.0, 5.0, 5.0, 5.0])
    tr, yr = resample_to_100hz(t, y)
    assert FS == 100
    dt = np.diff(tr)
    assert np.allclose(dt, 0.01, atol=1e-9)
    assert tr[0] == 0.0
    assert yr.shape == tr.shape


def test_offsets_put_max_at_zero():
    y = np.array([-3.0, -1.0, -7.0])
    assert offset_global(y).max() == 0.0
    assert offset_window(y).max() == 0.0
    np.testing.assert_allclose(offset_global(y), y - y.max())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml_preprocess.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

```python
# ml_detection/preprocess.py
"""
Signal preprocessing shared by training and inference, matching the MATLAB pipeline exactly.

Sampling: capacitance is recorded at an irregular rate; the MATLAB nets were trained at a uniform
100 Hz. We reproduce `resampleCapacitance` (linear interpolation with linear extrapolation).

Offsets: MATLAB offsets each analysis window so its maximum is 0 (`y - max(y)`), which places lick
deflections at negative values. Two named helpers make the *scope* of the offset explicit, since
MATLAB (faithfully preserved here) uses different scopes in different places:
  - `offset_window` : per-window scope (bout net input; point net TRAINING windows).
  - `offset_global` : whole-recording scope (point net INFERENCE windows).
The math is identical; the distinction is which array you pass in. See the spec's offset section.
"""
import numpy as np

FS = 100                    # target sampling rate (Hz)
WIN_SAMPLES = 300           # bout window = 3 s at 100 Hz
POINT_WIN = 21              # point window = 0.21 s
CENTER_SAMPLES = 100        # central 1 s labeling region
BOUT_STEP = 50              # 0.5 s bout slide step, in samples


def resample_to_100hz(time_s, cap):
    """
    Resample irregular (time, capacitance) onto a uniform 100 Hz grid.

    Uses linear interpolation, extrapolating at the ends (np.interp clamps by default, so we
    extend manually) to match MATLAB's `interp1(..., 'linear', 'extrap')`.

    Returns (t_uniform, y_uniform) as float64 arrays.
    """
    time_s = np.asarray(time_s, dtype=float)
    cap = np.asarray(cap, dtype=float)
    t_uniform = np.arange(time_s[0], time_s[-1], 1.0 / FS)
    # np.interp does not extrapolate; within-range points are all we need here because the grid
    # is built strictly inside [t0, t_end]. Endpoint equality is handled by the half-open arange.
    y_uniform = np.interp(t_uniform, time_s, cap)
    return t_uniform, y_uniform


def offset_global(y):
    """Offset a whole recording so its maximum is 0 (point-net inference convention)."""
    y = np.asarray(y, dtype=float)
    return y - np.max(y)


def offset_window(y_window):
    """Offset a single window so its maximum is 0 (bout-net + point-net training convention)."""
    y_window = np.asarray(y_window, dtype=float)
    return y_window - np.max(y_window)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml_preprocess.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml_detection/preprocess.py tests/test_ml_preprocess.py
git commit -m "feat(ml): resample-to-100Hz and named offset conventions"
```

---

## Task 5: infer.py (vectorized cascade)

**Files:**
- Create: `ml_detection/infer.py`
- Modify: `ml_detection/__init__.py` (export `detect_licks`)
- Test: `tests/test_ml_infer.py`

**Interfaces:**
- Consumes: `LickBoutNet`/`LickPointNet` (Task 1), `preprocess` helpers (Task 4).
- Produces:
  - `detect_licks(time_s, cap, bout_net, point_net) -> np.ndarray` — returns lick times in the ORIGINAL time base (seconds).
  - `_point_mask_naive(y_offset, positive_samples, point_net) -> np.ndarray[bool]` — reference per-sample loop, used only to validate the vectorized path.
  - `_merge_lick_points(mask, t) -> np.ndarray` — merge True samples within 20 ms, return cluster-center times.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ml_infer.py
import numpy as np
import torch
from ml_detection.nets import LickPointNet
from ml_detection.infer import _merge_lick_points, _point_mask_naive, _point_mask_vectorized


def test_merge_groups_points_within_20ms():
    # 100 Hz -> 20 ms = 2 samples. Points at 10,11 merge; 40 is separate.
    t = np.arange(60) / 100.0
    mask = np.zeros(60, dtype=bool)
    mask[[10, 11, 40]] = True
    times = _merge_lick_points(mask, t)
    assert len(times) == 2
    assert times[0] == np.take(t, 10) or abs(times[0] - t[10]) <= 0.011  # cluster center of {10,11}
    assert abs(times[1] - t[40]) < 1e-9


def test_vectorized_point_mask_equals_naive():
    torch.manual_seed(0)
    net = LickPointNet().eval()          # random weights are fine; we compare two code paths
    y = np.random.RandomState(1).randn(500).astype(np.float32)
    positive = np.arange(60, 200)        # a positive bout span
    naive = _point_mask_naive(y, positive, net)
    vec = _point_mask_vectorized(y, positive, net)
    np.testing.assert_array_equal(naive, vec)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml_infer.py -v`
Expected: FAIL (`ModuleNotFoundError` / names not defined).

- [ ] **Step 3: Write implementation**

```python
# ml_detection/infer.py
"""
Vectorized inference cascade: the PyTorch equivalent of MATLAB `detectLicksFromRaw`.

Faithful behavior, with ONE output-identical optimization: the point net is evaluated for every
candidate sample in a single batched forward pass instead of MATLAB's per-sample Python loop
(`_point_mask_naive` documents and tests the equivalence). Because the point net input uses the
GLOBAL offset (MATLAB inference convention), a sample's classification does not depend on which
bout window reached it, so overlapping positive bout windows collapse to their union losslessly —
matching MATLAB's OR semantics (`lickMaskGlobal` is set and never unset).
"""
import numpy as np
import torch

from ml_detection.preprocess import (
    resample_to_100hz, offset_global, offset_window,
    FS, WIN_SAMPLES, POINT_WIN, CENTER_SAMPLES, BOUT_STEP,
)

HALF_PT = (POINT_WIN - 1) // 2
MERGE_SAMPLES = int(round(0.02 * FS))   # 20 ms


def _positive_bout_samples(y, bout_net):
    """
    Slide the 3 s bout window (step 0.5 s), classify each with the per-window offset, and return
    the SORTED UNIQUE set of global sample indices that fall inside at least one positive window.
    """
    L = len(y)
    starts = range(0, L - WIN_SAMPLES + 1, BOUT_STEP)
    windows = []
    win_starts = []
    for s in starts:
        w = offset_window(y[s:s + WIN_SAMPLES])
        windows.append(w)
        win_starts.append(s)
    if not windows:
        return np.array([], dtype=int)
    X = torch.tensor(np.stack(windows), dtype=torch.float32).unsqueeze(1)  # [nWin,1,300]
    with torch.no_grad():
        pred = bout_net(X).argmax(dim=1).numpy()          # 1 == 'lick'
    positive = np.zeros(L, dtype=bool)
    for s, is_lick in zip(win_starts, pred):
        if is_lick == 1:
            positive[s:s + WIN_SAMPLES] = True
    return np.nonzero(positive)[0]


def _gather_point_windows(y_offset, samples):
    """Build [n,1,21] batch of point windows centered on each valid sample (edges dropped)."""
    L = len(y_offset)
    valid = samples[(samples - HALF_PT >= 0) & (samples + HALF_PT < L)]
    if len(valid) == 0:
        return valid, np.empty((0, 1, POINT_WIN), dtype=np.float32)
    idx = valid[:, None] + np.arange(-HALF_PT, HALF_PT + 1)[None, :]
    batch = y_offset[idx].astype(np.float32)[:, None, :]
    return valid, batch


def _point_mask_vectorized(y_offset, samples, point_net):
    L = len(y_offset)
    mask = np.zeros(L, dtype=bool)
    valid, batch = _gather_point_windows(y_offset, samples)
    if len(valid) == 0:
        return mask
    with torch.no_grad():
        pred = point_net(torch.tensor(batch)).argmax(dim=1).numpy()
    mask[valid[pred == 1]] = True
    return mask


def _point_mask_naive(y_offset, samples, point_net):
    """Reference per-sample loop (MATLAB-style), for test equivalence only."""
    L = len(y_offset)
    mask = np.zeros(L, dtype=bool)
    for c in samples:
        if c - HALF_PT < 0 or c + HALF_PT >= L:
            continue
        seg = y_offset[c - HALF_PT:c + HALF_PT + 1].astype(np.float32)
        with torch.no_grad():
            p = point_net(torch.tensor(seg)[None, None, :]).argmax(dim=1).item()
        if p == 1:
            mask[c] = True
    return mask


def _merge_lick_points(mask, t):
    """Merge True samples within 20 ms into one lick; representative time = cluster center."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return np.array([])
    clusters = [[idx[0]]]
    for i in idx[1:]:
        if i - clusters[-1][-1] <= MERGE_SAMPLES:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    centers = [c[len(c) // 2] for c in clusters]
    return t[np.array(centers)]


def detect_licks(time_s, cap, bout_net, point_net):
    """
    Detect lick times from a raw (irregular) capacitance recording.

    Returns lick times in the ORIGINAL time base (seconds), obtained by resampling to 100 Hz,
    running the cascade, and mapping the 100 Hz cluster-center times back through the original
    recording start.
    """
    bout_net.eval(); point_net.eval()
    t, y = resample_to_100hz(time_s, cap)
    positive = _positive_bout_samples(y, bout_net)
    y_glob = offset_global(y)                              # point-net inference offset
    mask = _point_mask_vectorized(y_glob, positive, point_net)
    lick_t = _merge_lick_points(mask, t)                   # seconds relative to resampled t0
    # t already starts at time_s[0], so lick_t is in the original time base.
    return lick_t
```

- [ ] **Step 4: Export and run tests**

Add to `ml_detection/__init__.py`:
```python
from ml_detection.infer import detect_licks  # noqa: E402,F401
```
Run: `pytest tests/test_ml_infer.py -v`
Expected: PASS (both tests; the vectorized/naive equivalence is the key gate).

- [ ] **Step 5: Commit**

```bash
git add ml_detection/infer.py ml_detection/__init__.py tests/test_ml_infer.py
git commit -m "feat(ml): vectorized inference cascade (output-identical to MATLAB loop)"
```

---

## Task 6: Integrate into data_analysis.filter_data

**Files:**
- Modify: `data_analysis.py` (add `ml_algorithm`; extend `filter_data` dispatch at the `algorithm ==` chain, around line 161-164)
- Test: `tests/test_ml_integration.py`

**Interfaces:**
- Consumes: `detect_licks` (Task 5), `load_matlab_nets` or a checkpoint loader.
- Produces: `ml_algorithm(data_by_animal, filtered_h5f, logfile, checkpoint=None) -> bool` — mirrors `basic_algorithm`'s contract, writing `lick_times`, `lick_indices`, `num_licks` per animal via `save_filtered_data`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ml_integration.py
import numpy as np
import data_analysis as da


class _StubNet:
    def eval(self): return self


def test_ml_algorithm_writes_expected_datasets(monkeypatch, tmp_path):
    import h5py
    # Two animals with simple traces; stub detect_licks to return fixed times.
    data_by_animal = {
        "A1": {"cap_data": np.zeros(1000), "time_data": np.linspace(0, 10, 1000),
               "used_start_idx": 0, "used_stop_idx": 999},
    }
    monkeypatch.setattr(da, "_load_ml_nets", lambda ckpt: (_StubNet(), _StubNet()))
    monkeypatch.setattr(da, "detect_licks",
                        lambda t, c, b, p: np.array([1.0, 2.0, 3.0]))
    out = tmp_path / "filtered.h5"
    with h5py.File(out, "w") as f:
        missing = da.ml_algorithm(data_by_animal, f, str(tmp_path / "log.txt"))
    assert missing is False
    with h5py.File(out, "r") as f:
        assert f["A1"]["num_licks"][()] == 3
        assert np.allclose(f["A1"]["lick_times"][()], [1.0, 2.0, 3.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml_integration.py -v`
Expected: FAIL (`AttributeError: module 'data_analysis' has no attribute 'ml_algorithm'`).

- [ ] **Step 3: Write implementation in `data_analysis.py`**

Add near the other algorithms (after `hilbert_algorithm`):

```python
# --- ML lick detection (opt-in via algorithm='ml') --------------------------------------------
# Imported lazily so the heavy torch import is only paid when the ML path is actually used.
def _load_ml_nets(checkpoint=None):
    """Load the bout/point networks. Defaults to the fine-tuned checkpoint, falling back to the
    weights ported straight from MATLAB. Returns (bout_net, point_net) in eval mode."""
    import os
    import torch
    from ml_detection.nets import LickBoutNet, LickPointNet
    from ml_detection.weights_io import load_matlab_nets
    default_finetuned = os.path.join("ml_detection", "checkpoints", "finetuned.pt")
    ckpt = checkpoint or default_finetuned
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        bout, point = LickBoutNet(), LickPointNet()
        bout.load_state_dict(state["bout"]); point.load_state_dict(state["point"])
        bout.eval(); point.eval()
        return bout, point
    # Fall back to the raw MATLAB port (old-scale weights) if no fine-tuned checkpoint exists yet.
    return load_matlab_nets(os.path.join("ML Detection MATLAB Code", "lickNets.mat"))


# Module-level indirection so tests can monkeypatch `data_analysis.detect_licks`.
try:
    from ml_detection.infer import detect_licks
except Exception:      # torch/ml_detection not importable in some minimal contexts
    detect_licks = None


def ml_algorithm(data_by_animal, filtered_h5f, logfile, checkpoint=None):
    """Detect licks with the ML cascade. Mirrors basic_algorithm's I/O contract exactly:
    writes lick_times, lick_indices, num_licks per animal via save_filtered_data."""
    bout_net, point_net = _load_ml_nets(checkpoint)
    for animal, data in data_by_animal.items():
        cap = np.asarray(data["cap_data"])
        t = np.asarray(data["time_data"])
        if len(cap) == 0:
            data["lick_times"] = np.array([]); data["lick_indices"] = np.array([], dtype=int)
            data["num_licks"] = 0
            missing_data = save_filtered_data(data, animal, filtered_h5f, logfile)
            if missing_data: return missing_data
            continue
        lick_times = np.asarray(detect_licks(t, cap, bout_net, point_net))
        # Map lick TIMES back to indices in the (trimmed) original trace via nearest time sample.
        lick_indices = np.searchsorted(t, lick_times)
        lick_indices = np.clip(lick_indices, 0, len(t) - 1)
        data["lick_times"] = lick_times
        data["lick_indices"] = lick_indices
        data["num_licks"] = int(len(lick_times))
        print(f"Animal {animal} had {len(lick_times)} licks detected (ML)")
        missing_data = save_filtered_data(data, animal, filtered_h5f, logfile)
        if missing_data: return missing_data
    return False
```

Extend the dispatch in `filter_data` (currently ends at the `hilbert` branch):

```python
    elif algorithm == 'ml':
        missing_data = ml_algorithm(data_by_animal, filtered_h5f, logfile)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml_integration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_analysis.py tests/test_ml_integration.py
git commit -m "feat(ml): wire ML detector into filter_data(algorithm='ml')"
```

---

## Task 7: dataset.py + bootstrap.py (training-set build)

**Files:**
- Create: `ml_detection/dataset.py`, `ml_detection/bootstrap.py`
- Test: `tests/test_ml_dataset.py`

**Interfaces:**
- Consumes: `preprocess` (Task 4), `basic_algorithm`/threshold detection from `data_analysis`.
- Produces:
  - `save_training_h5(path, segments, times, lick_idx, labels_bout, meta)` and `load_training_h5(path) -> dict` with keys `samples [N,300]`, `t [N,300]`, `lick_idx (list)`, `labels_bout [N]`, `fs`, `win_sec`, `center_sec`, plus provenance.
  - `prepare_point_segments(training: dict, win_pt=21) -> tuple[np.ndarray, np.ndarray]` — port of MATLAB `preparePointSegments` (per-segment offset already applied in `samples`).
  - `bootstrap_segments(time_s, cap, threshold_lick_times, n_samples=200) -> dict` — build category-balanced 3 s windows (0 / 1-3 / >=4 licks in center) with seed lick indices.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ml_dataset.py
import numpy as np
from ml_detection.dataset import (
    save_training_h5, load_training_h5, prepare_point_segments,
)


def test_training_h5_roundtrip(tmp_path):
    N = 3
    segments = np.random.randn(N, 300).astype(np.float32)
    times = np.tile(np.arange(300) / 100.0, (N, 1))
    lick_idx = [np.array([150]), np.array([]), np.array([140, 160])]
    labels = np.array([1, 0, 1])
    meta = {"source": "unit-test"}
    p = tmp_path / "train.h5"
    save_training_h5(str(p), segments, times, lick_idx, labels, meta)
    d = load_training_h5(str(p))
    assert d["samples"].shape == (N, 300)
    assert list(d["lick_idx"][2]) == [140, 160]
    assert d["labels_bout"].tolist() == [1, 0, 1]


def test_prepare_point_segments_only_central_positive():
    # One positive segment with a lick at the exact center (index 150 in 300).
    training = {
        "samples": np.zeros((1, 300), dtype=np.float32),
        "lick_idx": [np.array([150])],
        "labels_bout": np.array([1]),
        "fs": 100, "win_sec": 3, "center_sec": 1,
    }
    X, y = prepare_point_segments(training, win_pt=21)
    # Central 1 s spans indices 100..199 (100 windows). Exactly one center (150) is a lick.
    assert X.shape[0] == 100
    assert X.shape[2] == 21 if X.ndim == 3 else True
    assert int(y.sum()) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml_dataset.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `dataset.py`**

```python
# ml_detection/dataset.py
"""
Training-set schema (HDF5) and point-window builder for the lick nets.

We store curated 3 s segments plus per-segment lick indices and bout labels. This mirrors the
MATLAB `training` struct but in HDF5 with our own field names. `prepare_point_segments` ports
MATLAB `preparePointSegments`: for each segment whose central 1 s contains >=1 lick, it emits one
21-sample window centered on every sample in the central 1 s, labeled by whether that center
sample is a lick.
"""
import h5py
import numpy as np

from ml_detection.preprocess import WIN_SAMPLES, CENTER_SAMPLES, POINT_WIN


def save_training_h5(path, segments, times, lick_idx, labels_bout, meta):
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=np.asarray(segments, dtype=np.float32))
        f.create_dataset("t", data=np.asarray(times, dtype=np.float64))
        f.create_dataset("labels_bout", data=np.asarray(labels_bout, dtype=np.int64))
        # Ragged lick indices: store one variable-length dataset per segment under a group.
        g = f.create_group("lick_idx")
        for i, li in enumerate(lick_idx):
            g.create_dataset(str(i), data=np.asarray(li, dtype=np.int64))
        f.attrs["fs"] = 100
        f.attrs["win_sec"] = 3
        f.attrs["center_sec"] = 1
        for k, v in meta.items():
            f.attrs[f"meta_{k}"] = v


def load_training_h5(path):
    with h5py.File(path, "r") as f:
        n = f["samples"].shape[0]
        lick_idx = [np.asarray(f["lick_idx"][str(i)]) for i in range(n)]
        return {
            "samples": np.asarray(f["samples"]),
            "t": np.asarray(f["t"]),
            "lick_idx": lick_idx,
            "labels_bout": np.asarray(f["labels_bout"]),
            "fs": int(f.attrs["fs"]),
            "win_sec": int(f.attrs["win_sec"]),
            "center_sec": int(f.attrs["center_sec"]),
        }


def prepare_point_segments(training, win_pt=POINT_WIN):
    """
    Build point-level training windows. Returns (X [n,1,win_pt] float32, y [n] int64).

    Only segments with labels_bout == 1 contribute; only window centers inside the central 1 s are
    used; windows that would run past the segment edge are skipped. Matches MATLAB exactly.
    """
    half_pt = (win_pt - 1) // 2
    win_samples = WIN_SAMPLES
    center_samples = CENTER_SAMPLES
    center_start = round(win_samples / 2 - center_samples / 2)   # 0-based
    center_range = range(center_start, center_start + center_samples)

    all_x, all_y = [], []
    for seg, li, lab in zip(training["samples"], training["lick_idx"], training["labels_bout"]):
        if lab != 1:
            continue
        L = len(seg)
        lick_mask = np.zeros(L, dtype=bool)
        lick_mask[np.asarray(li, dtype=int)] = True
        for c in center_range:
            if c - half_pt < 0 or c + half_pt >= L:
                continue
            all_x.append(seg[c - half_pt:c + half_pt + 1])
            all_y.append(1 if lick_mask[c] else 0)
    X = np.asarray(all_x, dtype=np.float32)[:, None, :] if all_x else np.empty((0, 1, win_pt), np.float32)
    y = np.asarray(all_y, dtype=np.int64)
    return X, y
```

- [ ] **Step 4: Write `bootstrap.py`**

```python
# ml_detection/bootstrap.py
"""
Seed an initial training set for a new-scale recording.

Strategy (port of MATLAB `buildInitialTrainingSet`, but seeded by CLiQR's existing threshold
detector instead of findpeaks): resample to 100 Hz, run the threshold detector to get candidate
lick times, then sample category-balanced 3 s windows (0 / 1-3 / >=4 licks in the central 1 s).
Each window is offset per-window (training convention).
"""
import numpy as np

from ml_detection.preprocess import (
    resample_to_100hz, offset_window, WIN_SAMPLES, CENTER_SAMPLES, FS,
)


def bootstrap_segments(time_s, cap, threshold_lick_times, n_samples=200, seed=0):
    """
    Build initial labeled 3 s segments from one recording.

    Parameters
    ----------
    threshold_lick_times : np.ndarray
        Lick times (seconds, original base) from the existing basic_algorithm threshold detector.

    Returns a training dict compatible with save_training_h5 / prepare_point_segments.
    """
    rng = np.random.RandomState(seed)
    t, y = resample_to_100hz(time_s, cap)
    # Convert seeded lick times to 100 Hz sample indices.
    lick_samples = np.clip(np.searchsorted(t, threshold_lick_times), 0, len(t) - 1)
    lick_flags = np.zeros(len(t), dtype=bool)
    lick_flags[lick_samples] = True

    center_start = round(WIN_SAMPLES / 2 - CENTER_SAMPLES / 2)
    per_cat = round(n_samples / 3)
    segments, times, lick_idx, labels = [], [], [], []

    def count_center_licks(start):
        cs = start + center_start
        return int(lick_flags[cs:cs + CENTER_SAMPLES].sum())

    max_start = len(t) - WIN_SAMPLES
    if max_start <= 0:
        raise ValueError("Recording shorter than one 3 s window after resampling.")

    for cat in range(3):
        got = 0
        for _ in range(100000):
            if got >= per_cat:
                break
            s = rng.randint(0, max_start)
            n_c = count_center_licks(s)
            if cat == 0 and n_c != 0: continue
            if cat == 1 and not (1 <= n_c <= 3): continue
            if cat == 2 and n_c < 4: continue
            win = offset_window(y[s:s + WIN_SAMPLES])
            in_win = np.nonzero(lick_flags[s:s + WIN_SAMPLES])[0]
            segments.append(win.astype(np.float32))
            times.append((t[s:s + WIN_SAMPLES] - t[s]).astype(np.float64))
            lick_idx.append(in_win.astype(np.int64))
            labels.append(1 if count_center_licks(s) > 0 else 0)
            got += 1

    return {
        "samples": np.asarray(segments, dtype=np.float32),
        "t": np.asarray(times, dtype=np.float64),
        "lick_idx": lick_idx,
        "labels_bout": np.asarray(labels, dtype=np.int64),
        "fs": FS, "win_sec": 3, "center_sec": 1,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ml_dataset.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ml_detection/dataset.py ml_detection/bootstrap.py tests/test_ml_dataset.py
git commit -m "feat(ml): training-set HDF5 schema, point-window builder, threshold-seeded bootstrap"
```

---

## Task 8: Solara labeler app

**Files:**
- Create: `ml_detection/labeler/__init__.py`, `ml_detection/labeler/app.py`
- Test: `tests/test_ml_labeler.py` (import + pure-logic smoke; no browser)

**Interfaces:**
- Consumes: `load_training_h5`, `save_training_h5` (Task 7).
- Produces:
  - `recompute_label_bout(lick_idx: np.ndarray, win_samples=300, center_samples=100) -> int` — pure helper (unit-tested).
  - `add_or_select_lick(lick_idx, click_sample, fs) -> tuple[np.ndarray, int]` — pure helper.
  - Solara `Page` component wiring these to a plot + buttons (not unit-tested; run manually).

- [ ] **Step 1: Write the failing test (pure helpers only)**

```python
# tests/test_ml_labeler.py
import numpy as np
from ml_detection.labeler.app import recompute_label_bout, add_or_select_lick


def test_recompute_label_bout_central_only():
    # central 1 s = indices 100..199. A lick at 150 -> label 1; a lick at 50 -> label 0.
    assert recompute_label_bout(np.array([150])) == 1
    assert recompute_label_bout(np.array([50])) == 0
    assert recompute_label_bout(np.array([])) == 0


def test_add_lick_when_far_from_existing():
    lick_idx = np.array([10, 200])
    new_idx, selected = add_or_select_lick(lick_idx, click_sample=150, fs=100)
    assert 150 in new_idx.tolist()
    assert new_idx.tolist() == sorted(new_idx.tolist())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml_labeler.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `labeler/__init__.py` and `labeler/app.py`**

```python
# ml_detection/labeler/__init__.py
"""Solara-based manual labeler for lick-detection training segments."""
```

```python
# ml_detection/labeler/app.py
"""
Solara curation app for lick training segments, porting MATLAB `lickLabelerGUI`.

Pure label logic (unit-tested) is separated from the Solara UI (run manually):
    solara run ml_detection/labeler/app.py
Load a training HDF5 (from bootstrap), step through segments, click to add/select licks, nudge
with buttons, delete, and save a curated HDF5. `labels_bout` is recomputed on every edit.
"""
import numpy as np
import solara

from ml_detection.preprocess import WIN_SAMPLES, CENTER_SAMPLES
from ml_detection.dataset import load_training_h5, save_training_h5


def recompute_label_bout(lick_idx, win_samples=WIN_SAMPLES, center_samples=CENTER_SAMPLES):
    """Return 1 iff at least one lick index lies in the central `center_samples` of the window."""
    if len(lick_idx) == 0:
        return 0
    center_start = round(win_samples / 2 - center_samples / 2)
    center_end = center_start + center_samples - 1
    li = np.asarray(lick_idx)
    return int(np.any((li >= center_start) & (li <= center_end)))


def add_or_select_lick(lick_idx, click_sample, fs, select_tol_samples=2):
    """
    If the click is within tolerance of an existing lick, select it; otherwise add a new lick.
    Returns (updated_sorted_lick_idx, selected_position).
    """
    lick_idx = np.asarray(lick_idx, dtype=int)
    if len(lick_idx) > 0:
        dist = np.abs(lick_idx - click_sample)
        k = int(np.argmin(dist))
        if dist[k] <= select_tol_samples:
            return lick_idx, k
    new = np.sort(np.append(lick_idx, int(click_sample)))
    selected = int(np.nonzero(new == int(click_sample))[0][0])
    return new, selected


# ---- Solara UI (manual run; not unit-tested) -------------------------------------------------
@solara.component
def Page():
    training = solara.use_reactive(None)     # loaded dict
    idx = solara.use_reactive(0)
    path = solara.use_reactive("")

    def load():
        training.value = load_training_h5(path.value)
        idx.value = 0

    solara.InputText("Training HDF5 path", value=path)
    solara.Button("Load", on_click=load)
    if training.value is not None:
        d = training.value
        i = idx.value
        seg = d["samples"][i]
        fig = _segment_figure(seg, d["lick_idx"][i])
        solara.FigureMatplotlib(fig)
        solara.Text(f"Segment {i+1}/{len(d['samples'])}  labels_bout="
                    f"{recompute_label_bout(d['lick_idx'][i])}")
        solara.Button("Prev", on_click=lambda: idx.set(max(0, i - 1)))
        solara.Button("Next", on_click=lambda: idx.set(min(len(d['samples']) - 1, i + 1)))
        solara.Button("Save", on_click=lambda: save_training_h5(
            path.value.replace(".h5", "_curated.h5"),
            d["samples"], d["t"], d["lick_idx"], d["labels_bout"], {"curated": "true"}))


def _segment_figure(seg, lick_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(seg, "k-")
    if len(lick_idx):
        ax.plot(lick_idx, seg[np.asarray(lick_idx, dtype=int)], "ro")
    ax.set_xlabel("sample"); ax.set_ylabel("cap (offset)")
    return fig


Page  # module-level component for `solara run`
```

Note for the implementer: full click-to-add wiring on the matplotlib figure inside Solara is the manual-QA portion. The pure helpers above are what the tests cover; the click handler calls `add_or_select_lick` and then `recompute_label_bout`. Keep the helper calls exactly as named.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml_labeler.py -v`
Expected: PASS. Also verify the app imports: `python -c "import ml_detection.labeler.app"`.

- [ ] **Step 5: Commit**

```bash
git add ml_detection/labeler/__init__.py ml_detection/labeler/app.py tests/test_ml_labeler.py
git commit -m "feat(ml): Solara labeler with unit-tested label helpers"
```

---

## Task 9: train.py (transfer learning)

**Files:**
- Create: `ml_detection/train.py`
- Test: `tests/test_ml_train.py`

**Interfaces:**
- Consumes: `LickBoutNet`/`LickPointNet` (Task 1), `load_matlab_nets` (Task 2), `prepare_point_segments`/`load_training_h5` (Task 7).
- Produces:
  - `refit_zscore(net, segments: np.ndarray) -> None` — set `net.norm_mean`/`net.norm_std` from new data (single scalar each).
  - `session_split(session_ids: list[str], val_fraction=0.25, seed=0) -> tuple[set, set]` — hold out whole sessions.
  - `fine_tune(bout_net, point_net, training_files, out_path, epochs=..., lr=1e-4) -> dict` — refit norm, fine-tune all layers, save `{bout, point, meta}` to `out_path`; returns metrics.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ml_train.py
import numpy as np
import torch
from ml_detection.nets import LickBoutNet
from ml_detection.train import refit_zscore, session_split


def test_refit_zscore_sets_scalar_stats():
    net = LickBoutNet()
    segs = np.random.RandomState(0).randn(50, 300).astype(np.float32) * 3.0 + 7.0
    refit_zscore(net, segs)
    assert net.norm_mean.item() == pytest.approx(float(segs.mean()), abs=1e-3)
    assert net.norm_std.item() == pytest.approx(float(segs.std()), abs=1e-3)


def test_session_split_holds_out_whole_sessions():
    sessions = [f"s{i}" for i in range(8)]
    train, val = session_split(sessions, val_fraction=0.25, seed=0)
    assert train.isdisjoint(val)
    assert train | val == set(sessions)
    assert len(val) == 2
```

Add `import pytest` at the top.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml_train.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

```python
# ml_detection/train.py
"""
Transfer learning: adapt the MATLAB-ported nets to new high-CDT data.

Steps (per the spec): load ported weights, refit the scalar zscore normalization from the new
curated segments (the primary fix for the CDT magnitude change), split by SESSION to avoid leakage
between adjacent overlapping windows, then fine-tune ALL layers at a low learning rate with early
stopping. The bout net trains on 3 s segments; the point net trains on 21-sample central windows
(via prepare_point_segments).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ml_detection.nets import LickBoutNet, LickPointNet
from ml_detection.weights_io import load_matlab_nets
from ml_detection.dataset import load_training_h5, prepare_point_segments


def refit_zscore(net, segments):
    """Set net.norm_mean / net.norm_std to the global scalar mean/std of the new segments."""
    with torch.no_grad():
        net.norm_mean.fill_(float(np.mean(segments)))
        net.norm_std.fill_(float(np.std(segments)) or 1.0)


def session_split(session_ids, val_fraction=0.25, seed=0):
    """Partition unique session ids into disjoint (train, val) sets, holding out whole sessions."""
    uniq = sorted(set(session_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, round(len(uniq) * val_fraction))
    val = set(uniq[:n_val])
    train = set(uniq[n_val:])
    return train, val


def _train_one(net, X, y, Xval, yval, epochs, lr, batch_size):
    """Fine-tune a single net; return best val accuracy (early-stopped)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                        batch_size=batch_size, shuffle=True)
    best, best_state, patience, bad = 0.0, None, 5, 0
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(torch.tensor(Xval)).argmax(1) == torch.tensor(yval)).float().mean().item()
        if acc > best:
            best, best_state, bad = acc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return best


def fine_tune(training_files, out_path, epochs=100, lr=1e-4, batch_size=64,
              mat_path="ML Detection MATLAB Code/lickNets.mat"):
    """
    Fine-tune both nets on curated training HDF5 files (one per session).

    `training_files` is a dict {session_id: path}. Sessions are split into train/val; both nets are
    fine-tuned; the best checkpoint is saved to out_path as {'bout', 'point', 'meta'}.
    Returns a metrics dict.
    """
    bout, point = load_matlab_nets(mat_path)
    train_ids, val_ids = session_split(list(training_files))

    def gather(ids):
        bout_x, bout_y, pt_x, pt_y = [], [], [], []
        for sid in ids:
            d = load_training_h5(training_files[sid])
            bout_x.append(d["samples"]); bout_y.append(d["labels_bout"])
            X, y = prepare_point_segments(d)
            if len(y):
                pt_x.append(X[:, 0, :]); pt_y.append(y)
        BX = np.concatenate(bout_x)[:, None, :].astype(np.float32)
        BY = np.concatenate(bout_y).astype(np.int64)
        PX = np.concatenate(pt_x)[:, None, :].astype(np.float32)
        PY = np.concatenate(pt_y).astype(np.int64)
        return BX, BY, PX, PY

    BXt, BYt, PXt, PYt = gather(train_ids)
    BXv, BYv, PXv, PYv = gather(val_ids)

    # Refit normalization from TRAIN segments only (no val leakage).
    refit_zscore(bout, BXt[:, 0, :])
    refit_zscore(point, PXt[:, 0, :])

    bout_acc = _train_one(bout, BXt, BYt, BXv, BYv, epochs, lr, batch_size)
    point_acc = _train_one(point, PXt, PYt, PXv, PYv, epochs=20, lr=lr, batch_size=128)

    meta = {"fs": 100, "win_sec": 3, "point_win": 21,
            "train_sessions": sorted(train_ids), "val_sessions": sorted(val_ids)}
    torch.save({"bout": bout.state_dict(), "point": point.state_dict(), "meta": meta}, out_path)
    return {"bout_val_acc": bout_acc, "point_val_acc": point_acc, "meta": meta}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ml_train.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml_detection/train.py tests/test_ml_train.py
git commit -m "feat(ml): transfer-learning trainer (refit zscore, session split, fine-tune)"
```

---

## Task 10: Full-suite run + validation gates 2-4 (manual, documented)

**Files:**
- Create: `docs/ml_detection_validation.md` (record results)

**Interfaces:** none (uses everything above).

- [ ] **Step 1: Run the whole ML test suite**

Run: `pytest tests/test_ml_*.py -v`
Expected: all PASS (parity test #1 among them).

- [ ] **Step 2: Validation gate #2 — pre-fine-tune sanity on OLD-scale data**

Load the raw MATLAB port (no fine-tuning) and run `detect_licks` on a pre-2026-07-22 recording
(e.g. `Lickometry Data/results_combined_ACG-26-3_2026-07-21.h5`, or a raw ACG-26-3 sensor trace).
Confirm detections are reasonable (non-empty, lick-rate plausible) and consistent with the old
MATLAB behavior. Record counts in `docs/ml_detection_validation.md`.

- [ ] **Step 3: Bootstrap + label new-scale data**

For each new-scale session (`2026-07-22` onward): load the raw sensor trace, run `basic_algorithm`
to get seed lick times, call `bootstrap_segments`, save with `save_training_h5`, curate in the
Solara labeler (`solara run ml_detection/labeler/app.py`), save `_curated.h5`.

- [ ] **Step 4: Fine-tune and evaluate (gate #3)**

Run `fine_tune({session_id: curated_path, ...}, "ml_detection/checkpoints/finetuned.pt")`.
Record val accuracy and, on held-out sessions, precision/recall/F1 vs curated labels and vs the
threshold detector. Save to `docs/ml_detection_validation.md`.

- [ ] **Step 5: Validation gate #4 — timing spot-check**

Run `filter_data(..., algorithm='ml')` on a session with sync video; overlay ML lick times on the
trace and confirm no systematic offset for a few bouts. Note results.

- [ ] **Step 6: Commit validation record**

```bash
git add docs/ml_detection_validation.md
git commit -m "docs(ml): record validation gate results"
```

---

## Self-Review (completed during authoring)

- **Spec coverage:** nets (T1), weight port + flatten fix (T2), parity oracle (T3), preprocess/offsets (T4), vectorized cascade (T5), `filter_data` integration + output contract (T6), dataset/bootstrap/HDF5 + threshold seeding (T7), Solara labeler (T8), refit-norm + session-split + fine-tune-all (T9), validation gates 1-4 + torch dep already committed (T10). CDT cutoff enforced in T7/T10 and Global Constraints.
- **Faithful-first:** offset conventions preserved verbatim (T4/T5); improvements remain in the spec, not built here.
- **Type consistency:** `detect_licks(time_s, cap, bout_net, point_net)`, `prepare_point_segments(training, win_pt)`, `load_training_h5`/`save_training_h5`, `refit_zscore`, `session_split`, `_point_mask_vectorized/_naive`, `_merge_lick_points` used consistently across tasks.
- **Deferred to improvements (not tasks):** point-net offset-mismatch experiment, bout-gate voting, reuse of prior `checkpoints/best.pt`, ML+threshold fusion.
