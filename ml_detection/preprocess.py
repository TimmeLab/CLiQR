"""
Signal preprocessing shared by training and inference, matching the MATLAB pipeline exactly.

Sampling: capacitance is recorded at an irregular rate; the MATLAB nets were trained at a uniform
100 Hz. We reproduce `resampleCapacitance` (linear interpolation with linear extrapolation).

Offsets: MATLAB offsets each analysis window so its maximum is 0 (`y - max(y)`), which places lick
deflections at negative values. Two named helpers make the *scope* of the offset explicit, since
MATLAB (faithfully preserved here) uses different scopes in different places:
  - `offset_window` : per-window scope (bout net input; point net TRAINING windows).
  - `offset_global` : whole-recording scope (point net INFERENCE windows).
The math is identical; the distinction is which array you pass in. See the spec's offset section.
"""
import numpy as np

FS = 100                    # target sampling rate (Hz)
WIN_SAMPLES = 300           # bout window = 3 s at 100 Hz
POINT_WIN = 21              # point window = 0.21 s
CENTER_SAMPLES = 100        # central 1 s labeling region
BOUT_STEP = 50              # 0.5 s bout slide step, in samples


def resample_to_100hz(time_s, cap):
    """
    Resample irregular (time, capacitance) onto a uniform 100 Hz grid.

    Uses linear interpolation via np.interp. The grid is constructed strictly within the source
    range [t0, t_end) using np.arange, so all query points fall inside [t0, t_end] and no
    extrapolation is needed — behavior matches MATLAB's linear interp1 over the covered range.

    Returns (t_uniform, y_uniform) as float64 arrays.
    """
    time_s = np.asarray(time_s, dtype=float)
    cap = np.asarray(cap, dtype=float)
    t_uniform = np.arange(time_s[0], time_s[-1], 1.0 / FS)
    y_uniform = np.interp(t_uniform, time_s, cap)
    return t_uniform, y_uniform


def offset_global(y):
    """Offset a whole recording so its maximum is 0 (point-net inference convention)."""
    y = np.asarray(y, dtype=float)
    return y - np.max(y)


def offset_window(y_window):
    """Offset a single window so its maximum is 0 (bout-net + point-net training convention)."""
    y_window = np.asarray(y_window, dtype=float)
    return y_window - np.max(y_window)
