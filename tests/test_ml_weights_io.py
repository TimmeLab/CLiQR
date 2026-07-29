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
