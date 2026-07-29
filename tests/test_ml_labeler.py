"""Unit tests for the pure label-editing logic used by the lick labeler.

The Panel UI in ml_detection/labeler/app.py is exercised manually (`panel serve`); only the pure
logic in ml_detection/labeler/logic.py is unit-tested here.
"""
import numpy as np

from ml_detection.labeler.logic import (
    recompute_label_bout, add_or_select_lick, remove_lick, nudge_lick,
)


def test_recompute_label_bout_central_only():
    # Central 1 s = indices 100..199. A lick at 150 -> label 1; a lick at 50 -> label 0.
    assert recompute_label_bout(np.array([150])) == 1
    assert recompute_label_bout(np.array([50])) == 0
    assert recompute_label_bout(np.array([])) == 0


def test_add_lick_when_far_from_existing():
    lick_idx = np.array([10, 200])
    new_idx, selected = add_or_select_lick(lick_idx, click_sample=150)
    assert 150 in new_idx.tolist()
    assert new_idx.tolist() == sorted(new_idx.tolist())
    assert new_idx[selected] == 150


def test_select_nearest_lick_when_within_tolerance():
    # Click at 201, within select_tol_samples (default 2) of the existing lick at 200 ->
    # selects that lick instead of adding a new one; the array is left unchanged.
    lick_idx = np.array([10, 200])
    new_idx, selected = add_or_select_lick(lick_idx, click_sample=201)
    assert new_idx.tolist() == lick_idx.tolist()
    assert new_idx[selected] == 200


def test_remove_lick_updates_selection():
    lick_idx = np.array([10, 150, 260])
    new_idx, selected = remove_lick(lick_idx, position=1)
    assert new_idx.tolist() == [10, 260]
    assert selected == 1                        # selection clamps onto the neighbor


def test_remove_last_lick_yields_empty_and_no_selection():
    new_idx, selected = remove_lick(np.array([42]), position=0)
    assert new_idx.tolist() == []
    assert selected == -1


def test_remove_with_no_selection_is_noop():
    lick_idx = np.array([10, 20])
    new_idx, selected = remove_lick(lick_idx, position=-1)
    assert new_idx.tolist() == [10, 20]
    assert selected == -1


def test_nudge_moves_selected_and_clamps():
    lick_idx = np.array([10, 150, 260])
    new_idx, selected = nudge_lick(lick_idx, position=1, delta=5, max_index=299)
    assert new_idx.tolist() == [10, 155, 260]
    assert new_idx[selected] == 155             # same physical lick stays selected


def test_nudge_reorders_and_tracks_selection():
    # Moving the first lick past the second should re-sort; selection follows the moved lick.
    lick_idx = np.array([100, 105])
    new_idx, selected = nudge_lick(lick_idx, position=0, delta=10, max_index=299)
    assert new_idx.tolist() == [105, 110]
    assert new_idx[selected] == 110             # the moved lick (was 100 -> 110) stays selected


def test_nudge_clamps_at_zero():
    new_idx, selected = nudge_lick(np.array([3]), position=0, delta=-10, max_index=299)
    assert new_idx.tolist() == [0]
    assert selected == 0
