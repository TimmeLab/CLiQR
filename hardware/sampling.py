"""
The MPR121 sampling loop, shared by the in-process ThreadSource and the
hardware process (hardware/hw_process.py).

Solara-free on purpose: the hardware process is spawned and must import fast
without pulling in the GUI.
"""
import time
from concurrent.futures import ThreadPoolExecutor


def run_sampling_loop(read, serials, hz, stop, on_loop, on_read_error):
    """Read every board at `hz` until `stop` is set.

    Args:
        read: callable(sn) -> (times, caps) for one board read.
        serials: board serial numbers, read in parallel each loop.
        hz: target loop rate. Paced with a running deadline so jitter doesn't
            accumulate into drift; when behind, the deadline resyncs instead of
            bursting to catch up.
        stop: threading.Event; the loop returns once it is set.
        on_loop: callable(results), once per loop, with [(sn, times, caps)]
            for the boards whose read succeeded. Exceptions propagate (fatal).
        on_read_error: callable(sn, message) for a board read that failed even
            after the read's own retries. That board is skipped for this loop
            only -- one transient USB error costs one sample on one board.
    """
    period = 1.0 / hz
    with ThreadPoolExecutor(max_workers=max(1, len(serials))) as executor:
        next_deadline = time.monotonic() + period
        while not stop.is_set():
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                # time.sleep is high resolution (also on Windows, 3.11+).
                time.sleep(sleep_for)
                next_deadline += period
            else:
                next_deadline = time.monotonic() + period
            if stop.is_set():
                break

            futures = [(sn, executor.submit(read, sn)) for sn in serials]
            results = []
            for sn, future in futures:
                try:
                    times, caps = future.result()
                except Exception as exc:
                    on_read_error(sn, str(exc))
                    continue
                results.append((sn, list(times), list(caps)))
            on_loop(results)
