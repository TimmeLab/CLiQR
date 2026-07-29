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


def test_select_nearest_lick_when_within_tolerance():
    # Click at 201, within select_tol_samples (default 2) of the existing lick at 200 ->
    # selects that lick instead of adding a new one; the array is left unchanged.
    lick_idx = np.array([10, 200])
    new_idx, selected = add_or_select_lick(lick_idx, click_sample=201, fs=100)
    assert new_idx.tolist() == lick_idx.tolist()
    assert new_idx[selected] == 200
