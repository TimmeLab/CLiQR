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
