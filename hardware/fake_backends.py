"""
Fake hardware backends for tests and scripts/measure_sample_gaps.py.

They run inside the real hardware process, so tests exercise the actual
process/IPC path without boards attached.
"""
import json
import os
import sys
import threading
import time

DEFAULT_COUNTER_MAP = {"B0": [1, 2, 3], "B1": [4, 5, 6]}
# 4-board rack shape: board b -> sensors 6b+1 .. 6b+6.
PYFTDI_LIKE_MAP = {f"P{b}": list(range(6 * b + 1, 6 * b + 7)) for b in range(4)}


class CounterBackend:
    """Caps count successful reads per board (1, 2, 3, ...), times are real.

    Board map from env CLIQR_FAKE_MAP (JSON) or DEFAULT_COUNTER_MAP. Any
    dropped, duplicated or reordered sample is then visible on disk.
    """

    def __init__(self):
        raw = os.environ.get("CLIQR_FAKE_MAP")
        self.map = json.loads(raw) if raw else DEFAULT_COUNTER_MAP
        self._n = {sn: 0 for sn in self.map}
        self._lock = threading.Lock()

    def init(self):
        return {sn: len(s) for sn, s in self.map.items()}, ["fake boards ready"], []

    def read(self, sn, channels):
        with self._lock:
            self._n[sn] += 1
            v = self._n[sn]
        return [time.time()] * len(channels), [v] * len(channels)

    def close(self):
        pass


class PyftdiLikeBackend(CounterBackend):
    """Like a pyftdi read: ~20 short USB transfers, each releasing the GIL."""

    def __init__(self):
        super().__init__()
        self.map = PYFTDI_LIKE_MAP
        self._n = {sn: 0 for sn in self.map}

    def read(self, sn, channels):
        for _ in range(20):
            time.sleep(0.0001)
        return super().read(sn, channels)


class CrashAfterBackend(CounterBackend):
    """Kills the hardware process on its 50th read (simulated crash)."""

    def __init__(self):
        super().__init__()
        self._reads = 0

    def read(self, sn, channels):
        with self._lock:
            self._reads += 1
            if self._reads >= 50:
                os._exit(3)
        return super().read(sn, channels)


class ImportProbeBackend(CounterBackend):
    """Reports (as an init error) if the GUI stack leaked into the process."""

    def init(self):
        info, messages, _ = super().init()
        leaked = [m for m in ("solara", "pandas", "utils.state") if m in sys.modules]
        return info, messages, [f"{m} imported" for m in leaked]
