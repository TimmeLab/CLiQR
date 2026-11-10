"""
Sample sources for SensorRecorder.

A source produces board reads and pushes them into a SampleSink (the
recorder). ThreadSource samples in this process (unit tests, scripts);
hardware.hw_client.ProcessSource samples in the hardware process so the GUI's
GIL use can't stall reads.
"""
import threading
from typing import Protocol

from hardware.sampling import run_sampling_loop


class SampleSink(Protocol):
    def on_loops(self, loops):
        """loops: list of loops; each loop is [(sn, times, caps)]."""

    def on_read_error(self, sn: str, message: str):
        """A board read failed after retries (that board skipped this loop)."""

    def on_fatal(self, message: str):
        """Sampling died; no more loops will arrive."""


class ThreadSource:
    """Samples in-process on thread 'cliqr-sampler'.

    manager.read_sensor_data(sn) -> (times, caps, sn), as MPR121Manager.
    """

    def __init__(self, manager, serials, hz):
        self._manager = manager
        self._serials = list(serials)
        self._hz = hz
        self._stop = threading.Event()
        self._thread = None

    def _read(self, sn):
        times, caps, _ = self._manager.read_sensor_data(sn)
        return times, caps

    def start(self, sink):
        def run():
            try:
                run_sampling_loop(self._read, self._serials, self._hz, self._stop,
                                  lambda loop: sink.on_loops([loop]),
                                  sink.on_read_error)
            except Exception as exc:
                sink.on_fatal(str(exc))

        self._thread = threading.Thread(target=run, name="cliqr-sampler", daemon=True)
        self._thread.start()

    def stop(self, timeout):
        """Stop sampling; True once the loop ended (all loops delivered)."""
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()
