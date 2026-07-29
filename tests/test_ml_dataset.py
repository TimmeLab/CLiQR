import numpy as np
from ml_detection.bootstrap import bootstrap_segments
from ml_detection.dataset import (
    save_training_h5, load_training_h5, prepare_point_segments,
)
from ml_detection.preprocess import resample_to_100hz


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
    # Provenance (meta) must round-trip.
    assert d["meta"]["source"] == "unit-test"


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


def test_bootstrap_segments_category_balance(tmp_path):
    # Synthetic 60 s / 100 Hz-resolvable recording with two known, well-separated lick clusters:
    # cluster B (3 licks) should land segments in the 1-3 category, cluster C (5 licks) in the >=4
    # category. Elsewhere in the recording there are no licks, so category 0 (no center licks) is
    # trivially reachable. Lick times are picked directly from the resampled 100 Hz grid so the
    # 100 Hz sample indices are exact (no float-rounding ambiguity in np.searchsorted).
    time_s = np.arange(0, 60, 0.01)
    cap = np.zeros_like(time_s)
    t, _ = resample_to_100hz(time_s, cap)
    idx_b = [3000, 3003, 3006]              # 3 licks close together -> 1-3-lick category
    idx_c = [5000, 5002, 5004, 5006, 5008]  # 5 licks close together -> >=4-lick category
    threshold_lick_times = t[idx_b + idx_c]

    n_samples = 30
    d = bootstrap_segments(time_s, cap, threshold_lick_times, n_samples=n_samples, seed=0)

    # (a) expected keys and samples shape.
    for key in ("samples", "t", "lick_idx", "labels_bout", "fs", "win_sec", "center_sec"):
        assert key in d
    per_cat = round(n_samples / 3)
    assert d["samples"].shape == (3 * per_cat, 300)

    # (b) labels_bout is 1 exactly when the segment's central 1 s (indices 100..199) contains
    # >= 1 seeded lick. Segments are appended category-by-category (0 licks, then 1-3, then >=4),
    # so with this recording the first per_cat segments are label 0 and the rest are label 1.
    labels = d["labels_bout"]
    assert labels[:per_cat].tolist() == [0] * per_cat
    assert labels[per_cat:].tolist() == [1] * (2 * per_cat)
    # Cross-check a few segments directly against their stored lick_idx.
    for i in (0, per_cat, 2 * per_cat):
        n_central = int(np.sum((d["lick_idx"][i] >= 100) & (d["lick_idx"][i] < 200)))
        assert (n_central >= 1) == bool(labels[i])

    # (c) round-trips through save_training_h5 / load_training_h5 and feeds prepare_point_segments
    # without error.
    p = tmp_path / "bootstrap.h5"
    save_training_h5(str(p), d["samples"], d["t"], d["lick_idx"], d["labels_bout"],
                      {"source": "bootstrap_segments"})
    d2 = load_training_h5(str(p))
    assert d2["samples"].shape == d["samples"].shape
    X, y = prepare_point_segments(d2)
    assert X.shape[0] == y.shape[0]
