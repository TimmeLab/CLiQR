"""
End-to-end validation of the PyTorch lick-detection port against the original MATLAB cascade.

Why this exists
---------------
The Task 3 parity gate proved that each network (netBout, netPoint) reproduces its MATLAB
counterpart to ~1e-9 on random inputs. That validates the two networks in isolation, but it does
NOT validate the full inference cascade: the sliding bout gate, the per-sample point pass over
positive spans, and the 20 ms merge that turns a point mask into lick times. This module closes
that gap by running BOTH cascades on the same real capacitance trace and comparing the lick times
they produce.

Running this on an OLD-scale recording (recorded before the 2026-07-22 CDT change) is validation
gate #2 from the design spec: the un-fine-tuned ported network should reproduce the original
MATLAB behavior, because the ported weights were trained on old-scale data. If Python and MATLAB
agree on an old-scale trace, the whole cascade — not just the two nets — is faithfully ported.

What "agreement" means
----------------------
The two cascades resample and threshold identically, so in the ideal case every MATLAB lick time
has a Python lick time within one sample (10 ms at 100 Hz). We match greedily within a tolerance
and report precision/recall plus timing offsets, so a near-miss (e.g. a single-sample rounding
difference) is visible rather than being scored as a hard disagreement.

Typical use (from the repository root, in the cliqr-gui environment)
-------------------------------------------------------------------
    from ml_detection.validate import validate_recording
    report = validate_recording(
        h5_path="Lickometry Data/Example/raw_data_2025-09-01_10-57-58.h5",
        board="board_FT232H0",
        sensor="sensor_1",
    )
    print(report)

or from the command line:

    python -m ml_detection.validate \
        "Lickometry Data/Example/raw_data_2025-09-01_10-57-58.h5" \
        board_FT232H0 sensor_1
"""

import os
import subprocess
import tempfile

import h5py
import numpy as np

# scipy provides well-tested MATLAB .mat readers/writers; we use them to hand a trace to MATLAB
# and read the resulting lick times back, avoiding any custom binary format.
import scipy.io

from ml_detection.infer import detect_licks
from ml_detection.preprocess import FS


# Default locations within the repository. The caller can override any of these.
DEFAULT_NET_FILE = os.path.join("ML Detection MATLAB Code", "lickNets.mat")
DEFAULT_MATLAB_SCRIPT = os.path.join("scripts", "matlab_detect_licks.m")
DEFAULT_MATLAB_BIN = "/Applications/MATLAB_R2025a.app/bin/matlab"


# ----------------------------------------------------------------------------------------------
# Load one sensor's raw trace
# ----------------------------------------------------------------------------------------------
def load_sensor_trace(h5_path, board, sensor):
    """
    Read one sensor's raw (time, capacitance) trace from a recording HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to a raw_data_*.h5 recording. Its structure is
        `<board>/<sensor>/{cap_data, time_data}` (see the recording GUI's HDF5 layout).
    board : str
        Board group name, e.g. "board_FT232H0".
    sensor : str
        Sensor group name, e.g. "sensor_1".

    Returns
    -------
    (time_s, cap) : tuple of np.ndarray
        Absolute sample times (seconds) and raw integer capacitance counts, as float64 arrays.

    Notes
    -----
    We deliberately return the FULL untrimmed trace. detectLicksFromRaw (and our Python
    detect_licks) both resample the whole trace internally, so no start/stop trimming is required
    for a faithful cascade-vs-cascade comparison.
    """
    with h5py.File(h5_path, "r") as f:
        group = f[board][sensor]
        # [()] materializes the full dataset into a numpy array.
        time_s = np.asarray(group["time_data"][()], dtype=float)
        cap = np.asarray(group["cap_data"][()], dtype=float)
    return time_s, cap


# ----------------------------------------------------------------------------------------------
# Run each cascade
# ----------------------------------------------------------------------------------------------
def run_python_cascade(time_s, cap, net_file=DEFAULT_NET_FILE):
    """
    Run the PyTorch cascade using the weights ported straight from MATLAB (no fine-tuning).

    We load the ported nets here rather than a fine-tuned checkpoint because this function's job
    is to reproduce the ORIGINAL MATLAB behavior on old-scale data, so it must use the same
    old-scale weights MATLAB uses.

    Returns lick times (seconds, original time base) as a sorted np.ndarray.
    """
    # Imported lazily so importing this module does not force the (multi-second) weight load until
    # a cascade is actually run.
    from ml_detection.weights_io import load_matlab_nets

    bout_net, point_net = load_matlab_nets(net_file)
    lick_times = np.asarray(detect_licks(time_s, cap, bout_net, point_net), dtype=float)
    return np.sort(lick_times)


def run_matlab_cascade(time_s, cap, net_file=DEFAULT_NET_FILE,
                       matlab_bin=DEFAULT_MATLAB_BIN, timeout_s=1200):
    """
    Run the original MATLAB detectLicksFromRaw cascade on the same trace, via headless MATLAB.

    The trace is written to a temporary .mat as `rawData = [time, cap]` (the shape
    detectLicksFromRaw expects), MATLAB is invoked with a SINGLE-LINE -batch command (newlines
    break -batch), and the resulting lick times are read back.

    Parameters
    ----------
    timeout_s : int
        The FIRST Deep Learning Toolbox predict() call is slow (JIT + toolbox load), so allow a
        generous timeout — a few minutes is normal on the first run.

    Returns lick times (seconds) as a sorted np.ndarray. Raises RuntimeError if MATLAB fails.
    """
    # rawData must be an [nSamples x 2] matrix: column 0 = time, column 1 = capacitance.
    raw_data = np.column_stack([time_s, cap])

    work_dir = tempfile.mkdtemp(prefix="ml_validate_")
    in_mat = os.path.join(work_dir, "trace.mat")
    out_mat = os.path.join(work_dir, "matlab_licks.mat")
    scipy.io.savemat(in_mat, {"rawData": raw_data})

    # `scripts/matlab_detect_licks.m` defines a function of the same name; MATLAB finds it on the
    # path because we run with the repository root as the working directory and the script adds
    # the MATLAB code directory itself. Escape single quotes are unnecessary since our paths have
    # none, but we keep the command on ONE line as -batch requires.
    matlab_cmd = (
        "addpath('scripts'); "
        f"matlab_detect_licks('{in_mat}', '{net_file}', '{out_mat}')"
    )
    result = subprocess.run(
        [matlab_bin, "-batch", matlab_cmd],
        cwd=os.getcwd(),               # repository root, so relative net_file/script paths resolve
        capture_output=True, text=True, timeout=timeout_s,
    )
    if not os.path.exists(out_mat):
        raise RuntimeError(
            "MATLAB did not produce output. stdout/stderr follow:\n"
            f"{result.stdout}\n{result.stderr}"
        )

    loaded = scipy.io.loadmat(out_mat)
    # A column vector round-trips as shape (n, 1); flatten to 1-D. An empty result may load as an
    # empty array of shape (0, 0), which .ravel() turns into an empty 1-D array.
    lick_times = np.asarray(loaded["lickTimes"], dtype=float).ravel()
    return np.sort(lick_times)


# ----------------------------------------------------------------------------------------------
# Compare two lick-time lists
# ----------------------------------------------------------------------------------------------
def compare_lick_times(python_times, matlab_times, tol_s=0.02):
    """
    Greedily match Python lick times to MATLAB lick times within a timing tolerance.

    Treating MATLAB as ground truth:
      - a MATLAB lick with a Python lick within `tol_s` is a true positive (matched),
      - a MATLAB lick with no nearby Python lick is a false negative (missed),
      - a Python lick with no nearby MATLAB lick is a false positive (extra).

    Matching is greedy and one-to-one: we walk the sorted MATLAB times and claim the nearest
    unused Python lick within tolerance. Because both lists are dense and nearly identical in the
    faithful case, greedy nearest-matching is unambiguous; a pathological many-to-one case would
    only understate agreement, never overstate it.

    Parameters
    ----------
    tol_s : float
        Match tolerance in seconds. The default 0.02 s (20 ms) equals the cascade's own
        lick-merge threshold, so two cascades that merged slightly differently near a boundary
        still count as agreeing.

    Returns
    -------
    dict with keys:
        n_python, n_matlab : counts
        n_matched          : matched pairs (true positives)
        n_missed           : MATLAB licks with no Python match (false negatives)
        n_extra            : Python licks with no MATLAB match (false positives)
        precision, recall  : standard, MATLAB-as-truth
        max_offset_s, mean_offset_s : timing offset over matched pairs (0.0 if none matched)
    """
    python_times = np.sort(np.asarray(python_times, dtype=float))
    matlab_times = np.sort(np.asarray(matlab_times, dtype=float))

    python_used = np.zeros(len(python_times), dtype=bool)
    matched_offsets = []

    for t_ml in matlab_times:
        if len(python_times) == 0:
            break
        # Distance from this MATLAB lick to every Python lick; ignore already-claimed ones.
        distances = np.abs(python_times - t_ml)
        distances[python_used] = np.inf
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tol_s:
            python_used[nearest] = True
            matched_offsets.append(distances[nearest])

    n_matched = len(matched_offsets)
    n_matlab = len(matlab_times)
    n_python = len(python_times)
    n_missed = n_matlab - n_matched            # MATLAB licks we failed to match
    n_extra = n_python - n_matched             # Python licks left unclaimed

    # Precision = fraction of Python licks that correspond to a real (MATLAB) lick.
    precision = n_matched / n_python if n_python else 0.0
    # Recall = fraction of MATLAB licks the Python cascade recovered.
    recall = n_matched / n_matlab if n_matlab else 0.0

    offsets = np.asarray(matched_offsets, dtype=float)
    return {
        "n_python": n_python,
        "n_matlab": n_matlab,
        "n_matched": n_matched,
        "n_missed": n_missed,
        "n_extra": n_extra,
        "precision": precision,
        "recall": recall,
        "max_offset_s": float(offsets.max()) if len(offsets) else 0.0,
        "mean_offset_s": float(offsets.mean()) if len(offsets) else 0.0,
    }


# ----------------------------------------------------------------------------------------------
# Top-level: validate one recording end to end
# ----------------------------------------------------------------------------------------------
def validate_recording(h5_path, board, sensor, net_file=DEFAULT_NET_FILE,
                       matlab_bin=DEFAULT_MATLAB_BIN, tol_s=0.02):
    """
    Run both cascades on one sensor's trace and return the agreement report.

    This is the convenience entry point for validation gate #2. Point it at an OLD-scale recording
    (pre-2026-07-22) so the ported old-scale weights are the right comparison. A faithful port
    should yield precision and recall at or very near 1.0 with sub-tolerance timing offsets.

    We zero-base the trace (subtract the first timestamp) before running EITHER cascade. This
    mirrors production exactly — `data_analysis.filter_data` subtracts the recorded start time, so
    the ML path always sees time relative to ~0. It also removes an absolute-epoch floating-point
    artifact from the comparison: raw recording timestamps are Unix epoch seconds (~1.7e9), and
    resampling at that magnitude drifts the grid by up to ~1 sample, which would otherwise flip a
    handful of borderline detections and understate the two cascades' true agreement.
    """
    time_s, cap = load_sensor_trace(h5_path, board, sensor)
    time_s = time_s - time_s[0]                 # zero-base, mirroring filter_data (see docstring)
    python_times = run_python_cascade(time_s, cap, net_file=net_file)
    matlab_times = run_matlab_cascade(time_s, cap, net_file=net_file, matlab_bin=matlab_bin)
    report = compare_lick_times(python_times, matlab_times, tol_s=tol_s)
    report["h5_path"] = h5_path
    report["board"] = board
    report["sensor"] = sensor
    return report


def _format_report(report):
    """Human-readable one-block summary for CLI output."""
    return (
        f"Recording : {report.get('h5_path')} [{report.get('board')}/{report.get('sensor')}]\n"
        f"Python licks : {report['n_python']}   MATLAB licks : {report['n_matlab']}\n"
        f"Matched      : {report['n_matched']}   Missed : {report['n_missed']}   "
        f"Extra : {report['n_extra']}\n"
        f"Precision    : {report['precision']:.4f}   Recall : {report['recall']:.4f}\n"
        f"Offset (s)   : max {report['max_offset_s']:.4f}   mean {report['mean_offset_s']:.4f}\n"
        f"Sample rate  : {FS} Hz"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate the PyTorch lick cascade against MATLAB detectLicksFromRaw "
                    "on one sensor of one recording (validation gate #2)."
    )
    parser.add_argument("h5_path", help="Path to a raw_data_*.h5 recording (old-scale for gate 2)")
    parser.add_argument("board", help="Board group name, e.g. board_FT232H0")
    parser.add_argument("sensor", help="Sensor group name, e.g. sensor_1")
    parser.add_argument("--net-file", default=DEFAULT_NET_FILE, help="Path to lickNets.mat")
    parser.add_argument("--tol-s", type=float, default=0.02, help="Match tolerance in seconds")
    args = parser.parse_args()

    report = validate_recording(
        args.h5_path, args.board, args.sensor,
        net_file=args.net_file, tol_s=args.tol_s,
    )
    print(_format_report(report))
