"""
Port trained weights from the MATLAB `lickNets.mat` SeriesNetworks into the PyTorch models.

MATLAB stores networks in HDF5 (v7.3). Conv/BN/FC parameters live under `#refs#`. The two
subtleties handled here:

1. Conv weight layout. MATLAB conv weights are `[FH, FW, inC, outC]` (here FH=1, FW=k);
   h5py reads MATLAB arrays with REVERSED dims, so a MATLAB `[1, k, inC, outC]` comes back as
   `[outC, inC, k, 1]` -- already PyTorch Conv1d order `[outC, inC, k]` once the trailing FH=1
   axis is dropped. Both frameworks CROSS-CORRELATE (no kernel flip), so we only drop/reorder
   axes, never reverse the kernel.

2. The flatten-order fix (permute_fc_for_flatten). MATLAB flattens the final conv feature map
   before the first fully-connected layer. The feature map is `[H=1, W, C]`; MATLAB uses
   column-major linear indexing, so element (h=0, w, c) lands at index `w + W*c = c*W + w`
   (channel-outer, width-inner). PyTorch `flatten` over a `(C, W)` tensor is row-major:
   index `c*W + w` (channel-outer, width-inner). Because the height dim is 1, THESE TWO ORDERS
   COINCIDE for this topology, so the correct reshape branch is the identity re-flatten
   ("interpret the flat input as (C, W)"). See permute_fc_for_flatten for details. Task 3's
   MATLAB `predict` parity vectors (1e-4) are the numerical gate that confirms this choice.

Discovered `#refs#` dataset paths (from the Step-1 explorer). NOTE: the norm-scalar group letters
do NOT cluster with the conv group letters -- each layer is mapped by shape (conv/fc) and the
input-norm is mapped by VALUE against the true `net.Layers(1).Mean/StandardDeviation` (see below),
so a mislabeled path fails loudly.

  CAUTION (found by Task 3's MATLAB parity gate): the norm-scalar refs #refs#/C and #refs#/c are
  swapped relative to the naive "uppercase clusters with bout, lowercase with point" guess -- i.e.
  #refs#/C is actually POINT's zscore constant and #refs#/c is actually BOUT's, confirmed directly
  against `S.netBout.Layers(1).Mean` / `S.netPoint.Layers(1).Mean` in MATLAB. The conv/bn/fc refs
  below were independently value-verified against `S.netBout.Layers(k).Weights` /
  `S.netPoint.Layers(k).Weights` and are NOT swapped.

  BOUT net (kernel 15, width 75 after pool):
    input zscore : #refs#/c/Normalization/Mean , #refs#/c/Normalization/Std   (mean ~ -4.9989)
    conv1        : #refs#/d/Weights/Value (16,1,15,1) , #refs#/d/Bias/Value (16,1,1)
    bn1          : #refs#/g/{TrainedMean,TrainedVariance,Scale/Value,Offset/Value,Epsilon} (16)
    conv2        : #refs#/j/Weights/Value (32,16,15,1) , #refs#/j/Bias/Value (32,1,1)
    bn2          : #refs#/m/{...} (32)
    fc1          : #refs#/o/Weights/Value (32,32,75,1) , #refs#/o/Bias/Value (32,1,1)
    fc2          : #refs#/s/Weights/Value (2,32,1,1) , #refs#/s/Bias/Value (2,1,1)

  POINT net (kernel 5, width 21, no pool):
    input zscore : #refs#/C/Normalization/Mean , #refs#/C/Normalization/Std   (mean ~ -6.8734)
    conv1        : #refs#/D/Weights/Value (16,1,5,1) , #refs#/D/Bias/Value (16,1,1)
    bn1          : #refs#/G/{...} (16)
    conv2        : #refs#/I/Weights/Value (32,16,5,1) , #refs#/I/Bias/Value (32,1,1)
    bn2          : #refs#/L/{...} (32)
    fc1          : #refs#/N/Weights/Value (32,32,21,1) , #refs#/N/Bias/Value (32,1,1)
    fc2          : #refs#/R/Weights/Value (2,32,1,1) , #refs#/R/Bias/Value (2,1,1)
"""
import h5py
import numpy as np
import torch

from ml_detection.nets import LickBoutNet, LickPointNet

# Expected input zscore scalars, asserted so a wrong norm ref path fails loudly.
# NOTE: these were swapped vs. an earlier version of this file -- verified against the real
# S.netBout.Layers(1).Mean/StandardDeviation and S.netPoint.Layers(1).Mean/StandardDeviation
# in MATLAB (Task 3 parity-gate debugging). #refs#/C belongs to POINT and #refs#/c to BOUT.
_EXPECTED_BOUT_MEAN = -4.9988613
_EXPECTED_BOUT_STD = 8.966024
_EXPECTED_POINT_MEAN = -6.8734145
_EXPECTED_POINT_STD = 8.215723


def _deref(f, name):
    """Read a dataset by name and return it as a float32 numpy array (MATLAB/h5py dim order)."""
    return np.array(f[name], dtype=np.float32)


def _scalar(f, name):
    """Read a 1-element dataset as a python float."""
    return float(_deref(f, name).ravel()[0])


def permute_fc_for_flatten(W_matlab, n_channels, width):
    """
    Reorder FC input columns from MATLAB's conv-feature-map flatten order to PyTorch's.

    The MATLAB feature map feeding the first FC layer has shape ``[H=1, W, C]`` and is flattened
    column-major, so its linear index is ``w + W*c = c*width + w`` -- channel-outer, width-inner.
    PyTorch flattens a ``(C, W)`` tensor row-major to ``c*width + w`` -- also channel-outer,
    width-inner. Because ``H == 1`` these orders are identical, so the correct transform is to
    interpret the flat input columns as a ``(C, W)`` grid and re-flatten it (an identity mapping
    on the column order for this topology). Had MATLAB stored the columns width-outer, we would
    instead reshape ``(width, n_channels)`` and transpose -- that alternate branch is kept in a
    comment below for reference. Task 3's MATLAB-parity vectors confirm this choice numerically.

    Parameters
    ----------
    W_matlab : np.ndarray, shape [n_out, n_channels * width]
        FC weight whose input axis follows MATLAB's `[1, W, C]` column-major flatten order
        (channel-outer, width-inner, i.e. column index = c*width + w).
    n_channels, width : int
        Feature-map channel count and temporal width feeding the FC layer.

    Returns
    -------
    np.ndarray, shape [n_out, n_channels * width]
        FC weight whose input axis follows PyTorch row-major `[C, W]` flatten.
    """
    n_out = W_matlab.shape[0]
    assert W_matlab.shape[1] == n_channels * width, (W_matlab.shape, n_channels, width)
    # Input columns are (C, W) channel-outer/width-inner; PyTorch wants the same -> identity re-flatten.
    reshaped = W_matlab.reshape(n_out, n_channels, width)   # interpret as (out, C, W)
    # Alternate branch if MATLAB had stored width-outer:
    #   reshaped = W_matlab.reshape(n_out, width, n_channels).transpose(0, 2, 1)
    return reshaped.reshape(n_out, n_channels * width)


def _conv_weight(f, weights_name, out_c, in_c, k):
    """Read a MATLAB conv weight and return PyTorch Conv1d order [outC, inC, k]."""
    W = _deref(f, weights_name)          # h5py: [outC, inC, k, FH=1]
    assert W.shape == (out_c, in_c, k, 1), (weights_name, W.shape)
    return W.reshape(out_c, in_c, k)     # drop trailing FH=1 axis


def _vec(f, name, n):
    """Read a MATLAB [n,1,1]/[n,1] vector as a length-n numpy vector."""
    v = _deref(f, name).ravel()
    assert v.shape == (n,), (name, v.shape)
    return v


def _load_conv(f, conv, weights_name, bias_name):
    out_c, in_c, k = conv.weight.shape
    w = _conv_weight(f, weights_name, out_c, in_c, k)
    b = _vec(f, bias_name, out_c)
    conv.weight.copy_(torch.tensor(w, dtype=torch.float32))
    conv.bias.copy_(torch.tensor(b, dtype=torch.float32))


def _load_bn(f, bn, group):
    n = bn.num_features
    mean = _vec(f, f"{group}/TrainedMean", n)
    var = _vec(f, f"{group}/TrainedVariance", n)
    scale = _vec(f, f"{group}/Scale/Value", n)
    offset = _vec(f, f"{group}/Offset/Value", n)
    eps = _scalar(f, f"{group}/Epsilon")
    bn.eps = eps
    bn.running_mean.copy_(torch.tensor(mean, dtype=torch.float32))
    bn.running_var.copy_(torch.tensor(var, dtype=torch.float32))
    bn.weight.copy_(torch.tensor(scale, dtype=torch.float32))
    bn.bias.copy_(torch.tensor(offset, dtype=torch.float32))


def _load_fc1(f, fc, weights_name, bias_name, n_channels, width):
    """First FC after the conv stack: needs the flatten-order fix."""
    out, in_feat = fc.weight.shape
    assert in_feat == n_channels * width, (fc.weight.shape, n_channels, width)
    raw = _deref(f, weights_name)                       # h5py: [out, C, W, 1]
    assert raw.shape == (out, n_channels, width, 1), (weights_name, raw.shape)
    raw = raw.reshape(out, n_channels * width)          # numpy row-major over (out, C, W) -> c*W + w
    w = permute_fc_for_flatten(raw, n_channels=n_channels, width=width)
    assert w.shape == (out, n_channels * width), w.shape
    fc.weight.copy_(torch.tensor(w, dtype=torch.float32))
    fc.bias.copy_(torch.tensor(_vec(f, bias_name, out), dtype=torch.float32))


def _load_fc_plain(f, fc, weights_name, bias_name):
    """Trailing FC (MATLAB weight [out, in, 1, 1]) -> Linear [out, in]."""
    out, in_feat = fc.weight.shape
    raw = _deref(f, weights_name)                       # h5py: [out, in, 1, 1]
    assert raw.shape == (out, in_feat, 1, 1), (weights_name, raw.shape)
    w = raw.reshape(out, in_feat)
    fc.weight.copy_(torch.tensor(w, dtype=torch.float32))
    fc.bias.copy_(torch.tensor(_vec(f, bias_name, out), dtype=torch.float32))


def load_matlab_nets(mat_path):
    """
    Build LickBoutNet and LickPointNet with weights ported from `lickNets.mat`.

    Returns both nets in eval() mode. Raises AssertionError if any tensor shape or norm scalar
    does not match the target layer (guards against a wrong `#refs#` path).
    """
    f = h5py.File(mat_path, "r")
    try:
        bout = LickBoutNet()
        point = LickPointNet()

        with torch.no_grad():
            # ---- BOUT net (kernel 15, width 75 after pool) ----
            b_mean = _scalar(f, "#refs#/c/Normalization/Mean")
            b_std = _scalar(f, "#refs#/c/Normalization/Std")
            assert abs(b_mean - _EXPECTED_BOUT_MEAN) < 1e-3, b_mean
            assert abs(b_std - _EXPECTED_BOUT_STD) < 1e-3, b_std
            bout.norm_mean.fill_(b_mean)
            bout.norm_std.fill_(b_std)
            _load_conv(f, bout.conv1, "#refs#/d/Weights/Value", "#refs#/d/Bias/Value")
            _load_bn(f, bout.bn1, "#refs#/g")
            _load_conv(f, bout.conv2, "#refs#/j/Weights/Value", "#refs#/j/Bias/Value")
            _load_bn(f, bout.bn2, "#refs#/m")
            _load_fc1(f, bout.fc1, "#refs#/o/Weights/Value", "#refs#/o/Bias/Value",
                      n_channels=32, width=75)
            _load_fc_plain(f, bout.fc2, "#refs#/s/Weights/Value", "#refs#/s/Bias/Value")

            # ---- POINT net (kernel 5, width 21, no pool) ----
            p_mean = _scalar(f, "#refs#/C/Normalization/Mean")
            p_std = _scalar(f, "#refs#/C/Normalization/Std")
            assert abs(p_mean - _EXPECTED_POINT_MEAN) < 1e-3, p_mean
            assert abs(p_std - _EXPECTED_POINT_STD) < 1e-3, p_std
            point.norm_mean.fill_(p_mean)
            point.norm_std.fill_(p_std)
            _load_conv(f, point.conv1, "#refs#/D/Weights/Value", "#refs#/D/Bias/Value")
            _load_bn(f, point.bn1, "#refs#/G")
            _load_conv(f, point.conv2, "#refs#/I/Weights/Value", "#refs#/I/Bias/Value")
            _load_bn(f, point.bn2, "#refs#/L")
            _load_fc1(f, point.fc1, "#refs#/N/Weights/Value", "#refs#/N/Bias/Value",
                      n_channels=32, width=21)
            _load_fc_plain(f, point.fc2, "#refs#/R/Weights/Value", "#refs#/R/Bias/Value")
    finally:
        f.close()

    bout.eval()
    point.eval()
    return bout, point
