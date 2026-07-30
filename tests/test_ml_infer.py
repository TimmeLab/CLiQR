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
