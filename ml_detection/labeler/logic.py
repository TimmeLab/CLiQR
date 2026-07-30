"""
Pure, framework-agnostic label-editing logic for the lick labeler.

These functions contain all the "what happens when the user edits a segment" behavior, with no UI
dependency, so they can be unit-tested in isolation and reused regardless of whether the front end
is Panel, Solara, or a script. The UI layer (app.py) only translates clicks/button presses into
calls to these functions and redraws the result.

A "segment" is one 3 s (300-sample) window. Its licks are stored as an array of integer sample
indices into that window. `labels_bout` for the segment is a derived quantity: 1 iff at least one
lick falls in the central 1 s region, recomputed after every edit.
"""
import numpy as np

from ml_detection.preprocess import WIN_SAMPLES, CENTER_SAMPLES


def recompute_label_bout(lick_idx, win_samples=WIN_SAMPLES, center_samples=CENTER_SAMPLES):
    """
    Return 1 iff at least one lick index lies in the central `center_samples` of the window.

    The central region is centered in the window: for a 300-sample window and a 100-sample center,
    it spans indices 100..199. This mirrors MATLAB's labelsBout definition exactly.
    """
    if len(lick_idx) == 0:
        return 0
    center_start = round(win_samples / 2 - center_samples / 2)
    center_end = center_start + center_samples - 1
    li = np.asarray(lick_idx)
    return int(np.any((li >= center_start) & (li <= center_end)))


def add_or_select_lick(lick_idx, click_sample, select_tol_samples=2):
    """
    Resolve a click on the trace into either selecting an existing lick or adding a new one.

    If the click lands within `select_tol_samples` of an existing lick, that lick is selected
    (nothing added). Otherwise a new lick is inserted at `click_sample` and the array is kept
    sorted.

    Returns
    -------
    (updated_lick_idx, selected_position) : (np.ndarray[int], int)
        The (possibly extended) sorted array of lick sample indices, and the index WITHIN that
        array of the lick that is now selected.
    """
    lick_idx = np.asarray(lick_idx, dtype=int)
    click_sample = int(click_sample)
    if len(lick_idx) > 0:
        distances = np.abs(lick_idx - click_sample)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= select_tol_samples:
            return lick_idx, nearest
    updated = np.sort(np.append(lick_idx, click_sample))
    selected = int(np.nonzero(updated == click_sample)[0][0])
    return updated, selected


def remove_lick(lick_idx, position):
    """
    Remove the lick at `position` in the array. Returns (new_lick_idx, new_selected_position).

    After removal the selection moves to the neighboring lick (clamped), or -1 if the array is now
    empty. A `position` of -1 or an out-of-range index is treated as "nothing selected" and the
    array is returned unchanged with selection -1.
    """
    lick_idx = np.asarray(lick_idx, dtype=int)
    if position is None or position < 0 or position >= len(lick_idx):
        return lick_idx, -1
    updated = np.delete(lick_idx, position)
    if len(updated) == 0:
        return updated, -1
    # Keep the selection on a valid neighbor (the lick that shifted into this slot, or the last).
    new_position = min(position, len(updated) - 1)
    return updated, new_position


def nudge_lick(lick_idx, position, delta, max_index=WIN_SAMPLES - 1):
    """
    Move the selected lick by `delta` samples, clamped to [0, max_index], keeping the array sorted.

    Returns (new_lick_idx, new_selected_position). The selected lick is tracked across the re-sort
    so the same physical lick stays selected even if the move changes its ordinal position.
    A `position` of -1/out-of-range is a no-op returning selection -1.
    """
    lick_idx = np.asarray(lick_idx, dtype=int)
    if position is None or position < 0 or position >= len(lick_idx):
        return lick_idx, -1
    moved_value = int(np.clip(lick_idx[position] + delta, 0, max_index))
    # Replace the value in place, then re-sort and find where the moved lick ended up. We tag the
    # moved element by identity via an argsort of the new values.
    new_values = lick_idx.copy()
    new_values[position] = moved_value
    order = np.argsort(new_values, kind="stable")
    sorted_values = new_values[order]
    # The moved element was at `position` before sorting; find its new location.
    new_position = int(np.nonzero(order == position)[0][0])
    return sorted_values, new_position
