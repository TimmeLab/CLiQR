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
