"""The shared MPR121 sampling loop and the Solara-free hardware layer.

run_sampling_loop is the one pacing/reading loop used both in-process
(ThreadSource) and inside the hardware process. The hardware process is
spawned, so everything it imports must stay free of Solara/pandas/utils.state
(slow to import, and the GUI app must never be re-imported in the child).
"""
import os
import subprocess
import sys
import threading
import time

import pytest

from hardware.sampling import run_sampling_loop

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(read, serials, seconds, hz=200, on_loop=None):
    loops, errors = [], []
    stop = threading.Event()
    threading.Timer(seconds, stop.set).start()
    run_sampling_loop(read, serials, hz, stop,
                      on_loop or loops.append,
                      lambda sn, msg: errors.append((sn, msg)))
    return loops, errors


def _ok_read(sn):
    return [time.time()] * 3, [1, 2, 3]


def test_loop_paces_and_reports_each_board():
    loops, errors = _run(_ok_read, ["B0", "B1"], 0.25, hz=200)
    assert errors == []
    assert 40 <= len(loops) <= 60
    assert all({sn for sn, _, _ in loop} == {"B0", "B1"} for loop in loops)


def test_failed_read_skips_only_that_board():
    calls = {"B1": 0}
    lock = threading.Lock()

    def read(sn):
        if sn == "B1":
            with lock:
                calls["B1"] += 1
                n = calls["B1"]
            if n % 3 == 0:
                raise RuntimeError("No answer from FTDI")
        return _ok_read(sn)

    loops, errors = _run(read, ["B0", "B1"], 0.25)
    assert errors and all(sn == "B1" for sn, _ in errors)
    b0 = sum(1 for loop in loops for sn, _, _ in loop if sn == "B0")
    b1 = sum(1 for loop in loops for sn, _, _ in loop if sn == "B1")
    assert b0 - b1 == len(errors)


def test_on_loop_exception_propagates():
    count = {"n": 0}

    def on_loop(results):
        count["n"] += 1
        if count["n"] == 5:
            raise KeyError("bad serial")

    stop = threading.Event()
    with pytest.raises(KeyError):
        run_sampling_loop(_ok_read, ["B0"], 200, stop, on_loop,
                          lambda sn, msg: None)


def test_backends_import_without_solara():
    code = ("import sys; import hardware.backends, hardware.fake_backends, "
            "hardware.sampling, utils.hw_constants; "
            "print('solara' in sys.modules, 'utils.state' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False False"
