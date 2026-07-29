"""
Panel labeler for lick-detection training segments (port of MATLAB lickLabelerGUI).

Run it as a Panel app (Panel is already installed via the project's `panel` dependency, and Panel
runs a Bokeh server so the trace plot can receive real Python click callbacks — which a static
Solara/matplotlib image cannot):

    panel serve ml_detection/labeler/app.py --show

Workflow
--------
1. Browse to and select a training HDF5 (produced by the bootstrap step), then click "Load".
2. Step through segments with Prev/Next. The capacitance trace is shown with red lick markers;
   the currently selected lick is highlighted green.
3. Click the trace to ADD a lick at the nearest sample, or click near an existing lick to SELECT
   it. Use Remove to delete the selected lick and Nudge -1 / Nudge +1 to move it by one sample.
4. `labels_bout` (1 iff a lick sits in the central 1 s) is recomputed after every edit and shown.
5. Click "Save" to write a curated copy (`<input>_curated.h5`).

All editing behavior lives in `logic.py` (pure, unit-tested); this module only wires clicks and
buttons to those functions and redraws the Bokeh sources.
"""
import os
import sys

# `panel serve path/to/app.py` executes this file directly, WITHOUT the repository root on
# sys.path, so `import ml_detection` would fail (blank page). Make the app self-locating: this file
# is ml_detection/labeler/app.py, so the repo root is three directories up. Add it to sys.path
# before importing the package, so `panel serve ml_detection/labeler/app.py` works from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import panel as pn
from bokeh.events import Tap
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from ml_detection.preprocess import WIN_SAMPLES
from ml_detection.dataset import load_training_h5, save_training_h5
from ml_detection.labeler.logic import (
    recompute_label_bout, add_or_select_lick, remove_lick, nudge_lick,
)

pn.extension(sizing_mode="stretch_width")


class LabelerState:
    """Holds the loaded training set and the current editing cursor for one browser session.

    Panel executes this module once per session, so a module-level instance gives each user their
    own independent state without any cross-session bleed.
    """

    def __init__(self):
        self.training = None      # dict from load_training_h5, edited in place
        self.path = None          # source file path (for naming the curated output)
        self.index = 0            # which segment is displayed
        self.selected = -1        # position within the current segment's lick array (-1 = none)


state = LabelerState()

# ----------------------------------------------------------------------------------------------
# Bokeh figure and its data sources
# ----------------------------------------------------------------------------------------------
# One source for the trace line, one for all lick markers, one for the single selected lick.
trace_source = ColumnDataSource(data={"x": [], "y": []})
lick_source = ColumnDataSource(data={"x": [], "y": []})
selected_source = ColumnDataSource(data={"x": [], "y": []})

plot = figure(
    height=600, sizing_mode="stretch_width", title="Load a training file to begin",
    x_axis_label="sample (within 3 s segment)", y_axis_label="capacitance (offset)",
    tools="pan,box_zoom,wheel_zoom,reset",
)
plot.line("x", "y", source=trace_source, line_color="black")
plot.scatter("x", "y", source=lick_source, size=9, fill_color="red", line_color="red")
plot.scatter("x", "y", source=selected_source, size=13, fill_color="green", line_color="green")

status = pn.pane.Markdown("No file loaded.")


# ----------------------------------------------------------------------------------------------
# Redraw helpers
# ----------------------------------------------------------------------------------------------
def _current_segment():
    """Return (segment_y, lick_idx_array) for the displayed segment, or (None, None)."""
    if state.training is None:
        return None, None
    seg = state.training["samples"][state.index]
    lick_idx = np.asarray(state.training["lick_idx"][state.index], dtype=int)
    return seg, lick_idx


def redraw():
    """Push the current segment + licks + selection to the Bokeh sources and update the status."""
    seg, lick_idx = _current_segment()
    if seg is None:
        return
    x = np.arange(len(seg))
    trace_source.data = {"x": x, "y": np.asarray(seg, dtype=float)}

    if len(lick_idx):
        lick_source.data = {"x": lick_idx, "y": np.asarray(seg, dtype=float)[lick_idx]}
    else:
        lick_source.data = {"x": [], "y": []}

    if 0 <= state.selected < len(lick_idx):
        s = int(lick_idx[state.selected])
        selected_source.data = {"x": [s], "y": [float(seg[s])]}
    else:
        selected_source.data = {"x": [], "y": []}

    total = len(state.training["samples"])
    label = recompute_label_bout(lick_idx)
    plot.title.text = (
        f"Segment {state.index + 1} / {total}   |   {len(lick_idx)} licks   "
        f"|   labels_bout = {label}"
    )
    src = state.training.get("source")
    origin = f"   |   {src[state.index]}" if src is not None else ""
    status.object = (
        f"**Segment {state.index + 1} / {total}**{origin} — "
        f"{len(lick_idx)} licks, central-lick label = **{label}**. "
        f"Click trace to add/select; Remove/Nudge act on the green (selected) lick."
    )


def _select_default_lick():
    """Preselect the furthest-left lick of the current segment (index 0, since licks are sorted),
    or -1 if the segment has no licks. Used whenever a new segment is shown."""
    _, lick_idx = _current_segment()
    state.selected = 0 if lick_idx is not None and len(lick_idx) else -1


def _step_selection(delta):
    """Move the selection to the previous/next lick within the current segment (no wrap)."""
    if state.training is None:
        return
    _, lick_idx = _current_segment()
    if lick_idx is None or len(lick_idx) == 0:
        state.selected = -1
    elif state.selected < 0:
        # Nothing selected yet: land on the leftmost (delta>0) or rightmost (delta<0) lick.
        state.selected = 0 if delta > 0 else len(lick_idx) - 1
    else:
        state.selected = int(np.clip(state.selected + delta, 0, len(lick_idx) - 1))
    redraw()


def _commit_edit(new_lick_idx, new_selected):
    """Store an edited lick array back into the training dict, refresh its label, and redraw."""
    new_lick_idx = np.asarray(new_lick_idx, dtype=int)
    state.training["lick_idx"][state.index] = new_lick_idx
    state.training["labels_bout"][state.index] = recompute_label_bout(new_lick_idx)
    state.selected = new_selected
    redraw()


# ----------------------------------------------------------------------------------------------
# Event handlers
# ----------------------------------------------------------------------------------------------
def on_tap(event):
    """A click on the trace: add a lick at the nearest sample, or select a nearby existing lick."""
    if state.training is None or event.x is None:
        return
    click_sample = int(np.clip(round(event.x), 0, WIN_SAMPLES - 1))
    _, lick_idx = _current_segment()
    updated, selected = add_or_select_lick(lick_idx, click_sample)
    _commit_edit(updated, selected)


plot.on_event(Tap, on_tap)


def on_remove(_event=None):
    if state.training is None:
        return
    _, lick_idx = _current_segment()
    updated, selected = remove_lick(lick_idx, state.selected)
    _commit_edit(updated, selected)


def _nudge(delta):
    if state.training is None:
        return
    _, lick_idx = _current_segment()
    updated, selected = nudge_lick(lick_idx, state.selected, delta, max_index=WIN_SAMPLES - 1)
    _commit_edit(updated, selected)


def on_prev(_event=None):
    if state.training is None:
        return
    state.index = max(0, state.index - 1)
    _select_default_lick()
    redraw()


def on_next(_event=None):
    if state.training is None:
        return
    state.index = min(len(state.training["samples"]) - 1, state.index + 1)
    _select_default_lick()
    redraw()


def on_load(_event=None):
    """Load the file currently selected in the file browser."""
    selection = file_browser.value
    if not selection:
        status.object = "**Select a training .h5 file in the browser above, then click Load.**"
        return
    path = selection[0]
    state.path = path
    state.training = load_training_h5(path)
    # Ensure lick_idx entries are mutable numpy arrays we can reassign per segment.
    state.training["lick_idx"] = [np.asarray(li, dtype=int) for li in state.training["lick_idx"]]
    state.index = 0
    _select_default_lick()
    redraw()


def on_save(_event=None):
    if state.training is None:
        return
    out_path = os.path.splitext(state.path)[0] + "_curated.h5"
    meta = dict(state.training.get("meta", {}))
    meta["curated"] = "true"
    save_training_h5(
        out_path,
        state.training["samples"], state.training["t"],
        state.training["lick_idx"], state.training["labels_bout"], meta,
    )
    status.object = f"**Saved curated file to `{out_path}`** ({len(state.training['samples'])} segments)."


# ----------------------------------------------------------------------------------------------
# Widgets and layout
# ----------------------------------------------------------------------------------------------
# FileSelector browses the local filesystem (no upload) — appropriate for large training files.
file_browser = pn.widgets.FileSelector(
    directory=os.getcwd(), file_pattern="*.h5", only_files=True, name="Training file",
)

load_button = pn.widgets.Button(name="Load", button_type="primary")
prev_button = pn.widgets.Button(name="◀ Prev segment")
next_button = pn.widgets.Button(name="Next segment ▶")
prev_lick_button = pn.widgets.Button(name="◀ Lick")
next_lick_button = pn.widgets.Button(name="Lick ▶")
remove_button = pn.widgets.Button(name="Remove selected", button_type="warning")
nudge_left_button = pn.widgets.Button(name="Nudge −1")
nudge_right_button = pn.widgets.Button(name="Nudge +1")
save_button = pn.widgets.Button(name="Save curated", button_type="success")

load_button.on_click(on_load)
prev_button.on_click(on_prev)
next_button.on_click(on_next)
prev_lick_button.on_click(lambda e: _step_selection(-1))
next_lick_button.on_click(lambda e: _step_selection(1))
remove_button.on_click(on_remove)
nudge_left_button.on_click(lambda e: _nudge(-1))
nudge_right_button.on_click(lambda e: _nudge(1))
save_button.on_click(on_save)

controls = pn.Row(
    prev_button, next_button, prev_lick_button, next_lick_button,
    remove_button, nudge_left_button, nudge_right_button, save_button,
)

layout = pn.Column(
    pn.pane.Markdown("# CLiQR — Lick Labeler"),
    pn.pane.Markdown(
        "Browse to a bootstrap training `.h5`, click **Load**, then click the trace to "
        "add/select licks."
    ),
    file_browser,
    load_button,
    status,
    pn.pane.Bokeh(plot, sizing_mode="stretch_width"),
    controls,
)

layout.servable(title="CLiQR Lick Labeler")
