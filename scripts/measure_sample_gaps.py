"""
Measure inter-sample gaps of the recorder without hardware.

Why
---
On the Windows rig every UI action stalled sampling on all boards (~125 ms per
TEST press). Cause: Solara re-renders are pure Python on the GUI thread, and a
pyftdi board read is ~20 short USB transfers that each release and re-acquire
the GIL, so a read stretched across the whole render (GIL convoy effect).
Sampling now runs in a separate hardware process with its own GIL
(docs/superpowers/specs/2026-11-10-sampler-process-design.md).

This script runs the real recorder (HDF5 writes at HISTORY_SIZE=500) against a
fake backend and reports the gaps between samples:

  --source thread   sample in this process (the pre-process design)
  --source process  sample in the hardware process (what the GUI uses)
  --busy-gui        120 ms of pure-Python work every 2 s in this process,
                    like a Solara re-render (reproduces the rig problem)
  --stall-ui        block an asyncio loop 0.3 s every 2 s (the older problem)

The default backend, PyftdiLikeBackend, makes each read 20 short GIL-releasing
sleeps, like a pyftdi read.

Usage
-----
    python scripts/measure_sample_gaps.py --source thread --busy-gui
    python scripts/measure_sample_gaps.py --source process --busy-gui --seconds 30
"""
import argparse
import asyncio
import importlib
import os
import sys
import tempfile
import time

import h5py
import numpy as np


def _backend(path):
    module, cls = path.split(":")
    return getattr(importlib.import_module(module), cls)()


class _BackendManager:
    """Adapts a backend to the in-process manager interface (ThreadSource)."""

    def __init__(self, backend, channels):
        self.backend = backend
        self.channels = channels

    def read_sensor_data(self, sn):
        times, caps = self.backend.read(sn, self.channels)
        return times, caps, sn


def _busy(seconds):
    end = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < end:  # holds the GIL like a Solara render
        x += 1


def run(source, backend_path, seconds, busy_gui, stall_ui):
    import utils.state
    from hardware.fake_backends import PYFTDI_LIKE_MAP
    from hardware.hw_client import HardwareClient
    from recording import recorder as recorder_mod

    board_map = PYFTDI_LIKE_MAP
    channels = list(range(1, 12, 2))[:len(next(iter(board_map.values())))]
    utils.state.SERIAL_NUMBER_SENSOR_MAP = board_map
    filename = os.path.join(tempfile.mkdtemp(prefix="gaps_"), "raw.h5")
    controllers = {sn: {} for sn in board_map}
    hz = recorder_mod.MAX_SAMPLE_HZ

    client = None
    if source == "process":
        client = HardwareClient(backend_path)
        client.start()
        rec = recorder_mod.SensorRecorder(None, filename, controllers,
                                          source=client.source(channels, hz))
    else:
        manager = _BackendManager(_backend(backend_path), channels)
        rec = recorder_mod.SensorRecorder(manager, filename, controllers)

    async def gui_loop():
        rec.start(log_callback=print)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            await asyncio.sleep(2.0)
            if busy_gui:
                _busy(0.120)
            if stall_ui:
                time.sleep(0.3)
        if not rec.stop():
            print("stop() reported unwritten data")

    try:
        asyncio.run(gui_loop())
    finally:
        if client is not None:
            client.shutdown()

    period = 1.0 / hz
    gaps = []
    with h5py.File(filename, "r") as h5f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("time_data"):
                gaps.append(np.diff(obj[:]))
        h5f.visititems(visit)
    gaps = np.concatenate(gaps)
    print(f"\nsource={source}, {len(board_map) * len(channels)} channels, "
          f"{seconds:.0f} s, busy-gui={'on' if busy_gui else 'off'}, "
          f"stall-ui={'on' if stall_ui else 'off'}  ({filename})")
    print(f"  rate achieved      {1.0 / np.median(gaps):7.1f} Hz (target {hz})")
    print(f"  max gap            {gaps.max() * 1000:7.1f} ms")
    print(f"  99.9th pct gap     {np.percentile(gaps, 99.9) * 1000:7.1f} ms")
    print(f"  gaps > 2 periods   {int((gaps > 2 * period).sum()):7d} of {gaps.size}")


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=("thread", "process"), default="process")
    ap.add_argument("--backend", default="hardware.fake_backends:PyftdiLikeBackend")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--busy-gui", action="store_true",
                    help="120 ms of pure-Python work every 2 s in this process")
    ap.add_argument("--stall-ui", action="store_true",
                    help="block the asyncio loop 0.3 s every 2 s")
    args = ap.parse_args()
    run(args.source, args.backend, args.seconds, args.busy_gui, args.stall_ui)


if __name__ == "__main__":
    main()
