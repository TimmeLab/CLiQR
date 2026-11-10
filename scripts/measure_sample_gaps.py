"""
Measure inter-sample gaps of the threaded recorder without hardware.

Why
---
Sampling used to run on Solara's event loop, so slow UI work or an HDF5 flush
delayed reads and punched gaps into every sensor's trace. The recorder now runs
sampling and HDF5 writes on their own threads
(docs/superpowers/specs/2026-11-10-recorder-thread-design.md). One limit
remains: h5py holds the GIL inside each HDF5 call, so while the writer is
mid-call the sampler can't run Python. This script measures what that costs at
the real HISTORY_SIZE (500) and MAX_SAMPLE_HZ (150), for the 24-channel
(1-bottle) and 48-channel (2-bottle) 4-board layouts.

A fake board stands in for the FT232H/MPR121: each read sleeps --read-ms (a USB
round-trip releases the GIL, as the real one does) and stamps time.time() per
channel, like hardware/mpr121.py. --stall-ui runs an asyncio loop that blocks
for 0.3 s every 2 s, reproducing the original failure (the TEST plot render).

Usage
-----
    python scripts/measure_sample_gaps.py --channels 24
    python scripts/measure_sample_gaps.py --channels 48 --stall-ui --seconds 30
"""
import argparse
import asyncio
import os
import sys
import tempfile
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.state  # noqa: E402
from recording import recorder as recorder_mod  # noqa: E402

# 4-board rack: 6 sensors per board (x2 channels in 2-bottle mode).
BOARD_MAP = {f"FAKE{b}": list(range(6 * b + 1, 6 * b + 7)) for b in range(4)}


class FakeBoards:
    def __init__(self, channels_per_board, read_ms):
        self.k = channels_per_board
        self.read_s = read_ms / 1000.0

    def read_sensor_data(self, sn):
        time.sleep(self.read_s)  # USB round-trip; releases the GIL
        now = time.time()
        return [now] * self.k, [1000] * self.k, sn


def run(channels, seconds, read_ms, stall_ui):
    two_bottle = channels == 48
    utils.state.SERIAL_NUMBER_SENSOR_MAP = BOARD_MAP
    out_dir = tempfile.mkdtemp(prefix="gaps_")
    rec = recorder_mod.SensorRecorder(
        mpr121_manager=FakeBoards(channels // len(BOARD_MAP), read_ms),
        filename=os.path.join(out_dir, "raw.h5"),
        controllers={sn: object() for sn in BOARD_MAP},
        two_bottle=two_bottle,
    )

    async def ui_loop():
        rec.start(log_callback=print)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            await asyncio.sleep(2.0)
            if stall_ui:
                time.sleep(0.3)  # a slow synchronous render on the UI loop
        ok = rec.stop()
        if not ok:
            print("stop() reported unwritten data")

    asyncio.run(ui_loop())

    period = 1.0 / recorder_mod.MAX_SAMPLE_HZ
    gaps = []
    with h5py.File(rec.filename, "r") as h5f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("time_data"):
                gaps.append(np.diff(obj[:]))
        h5f.visititems(visit)
    gaps = np.concatenate(gaps)
    print(f"\n{channels} channels, {seconds:.0f} s, read {read_ms} ms, "
          f"stall-ui={'on' if stall_ui else 'off'}  ({rec.filename})")
    print(f"  rate achieved      {1.0 / np.median(gaps):7.1f} Hz "
          f"(target {recorder_mod.MAX_SAMPLE_HZ})")
    print(f"  max gap            {gaps.max() * 1000:7.1f} ms")
    print(f"  99.9th pct gap     {np.percentile(gaps, 99.9) * 1000:7.1f} ms")
    print(f"  gaps > 2 periods   {int((gaps > 2 * period).sum()):7d} "
          f"of {gaps.size}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--channels", type=int, choices=(24, 48), default=24)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--read-ms", type=float, default=2.0,
                    help="simulated USB read time per board (default 2 ms)")
    ap.add_argument("--stall-ui", action="store_true",
                    help="block the asyncio loop 0.3 s every 2 s")
    args = ap.parse_args()
    run(args.channels, args.seconds, args.read_ms, args.stall_ui)


if __name__ == "__main__":
    main()
