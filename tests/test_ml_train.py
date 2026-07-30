import numpy as np
import pytest
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
