"""
The hardware process: owns all FT232H/MPR121 USB I/O for the life of the GUI
server.

Why a process: Solara re-renders are pure Python on the GUI's main thread.
A pyftdi read is ~20 short USB transfers, each releasing and re-acquiring the
GIL; while the GUI thread is busy each re-acquire waits up to the 5 ms switch
interval, so one read stretched across a whole render (~125 ms gap on every
board per UI action on the Windows rig). A separate process has its own GIL.

Started with the "spawn" method by hardware/hw_client.HardwareClient; must not
import Solara, pandas, utils.state or the GUI.

Protocol (tuples):
  commands on cmd_conn -> reply
    ("init",)                         -> ("init", board_info, messages, errors)
    ("start", session, channels, hz)  -> ("started",)
    ("stop",)                         -> ("stopped",)   after ("end", session) is queued
    ("shutdown",)                     -> ("bye",)
  data on data_q (tagged with the session number)
    ("loops", session, loops)   loops = [[(sn, times, caps), ...], ...]
    ("read_error", session, sn, message)
    ("fatal", session, message)
    ("end", session)
If cmd_conn closes (the GUI server went away) the process stops sampling,
releases the boards and exits.
"""
import importlib
import threading

from hardware.sampling import run_sampling_loop

BATCH_SECONDS = 0.025


def _load_backend(path):
    module, cls = path.split(":")
    return getattr(importlib.import_module(module), cls)()


class _Sampler:
    """Runs the sampling loop on a thread and ships loops in small batches."""

    def __init__(self, backend, serials, session, channels, hz, data_q):
        self._backend = backend
        self._serials = serials
        self._session = session
        self._channels = list(channels)
        self._hz = hz
        self._q = data_q
        self._stop = threading.Event()
        self._done = threading.Event()
        self._buf = []
        self._lock = threading.Lock()
        self._loop_thread = threading.Thread(target=self._run, daemon=True)
        self._flush_thread = threading.Thread(target=self._flusher, daemon=True)

    def start(self):
        self._loop_thread.start()
        self._flush_thread.start()

    def _read(self, sn):
        return self._backend.read(sn, self._channels)

    def _on_loop(self, results):
        with self._lock:
            self._buf.append(results)

    def _on_read_error(self, sn, message):
        self._q.put(("read_error", self._session, sn, message))

    def _run(self):
        try:
            run_sampling_loop(self._read, self._serials, self._hz, self._stop,
                              self._on_loop, self._on_read_error)
        except Exception as exc:
            self._flush()
            self._q.put(("fatal", self._session, f"sampling failed: {exc}"))
        finally:
            self._done.set()

    def _flush(self):
        with self._lock:
            buf, self._buf = self._buf, []
        if buf:
            self._q.put(("loops", self._session, buf))

    def _flusher(self):
        while not self._done.wait(BATCH_SECONDS):
            self._flush()
        self._flush()

    def stop(self):
        self._stop.set()
        self._loop_thread.join()
        self._flush_thread.join()


def hardware_main(cmd_conn, data_q, backend_path):
    """Entry point of the hardware process."""
    backend = None
    serials = []
    sampler = None
    try:
        backend = _load_backend(backend_path)
        while True:
            try:
                if not cmd_conn.poll(0.1):
                    continue
                msg = cmd_conn.recv()
            except (EOFError, OSError):
                break  # GUI server gone
            kind = msg[0]
            if kind == "init":
                try:
                    info, messages, errors = backend.init()
                except Exception as exc:
                    info, messages, errors = {}, [], [f"hardware init failed: {exc}"]
                serials = list(info)
                cmd_conn.send(("init", info, messages, errors))
            elif kind == "start":
                _, session, channels, hz = msg
                if sampler is not None:
                    sampler.stop()
                sampler = _Sampler(backend, serials, session, channels, hz, data_q)
                sampler.start()
                cmd_conn.send(("started",))
            elif kind == "stop":
                session = sampler._session if sampler else None
                if sampler is not None:
                    sampler.stop()
                    sampler = None
                data_q.put(("end", session))
                cmd_conn.send(("stopped",))
            elif kind == "shutdown":
                if sampler is not None:
                    sampler.stop()
                    sampler = None
                try:
                    backend.close()
                finally:
                    backend = None
                cmd_conn.send(("bye",))
                break
    finally:
        if sampler is not None:
            sampler.stop()
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
