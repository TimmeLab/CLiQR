"""
GUI-side handle on the hardware process (hardware/hw_process.py).

One HardwareClient lives in the Solara server process (a plain module global,
so a browser refresh never touches it). ProcessSource plugs the process's
samples into SensorRecorder (see recording/sources.py).
"""
import itertools
import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass, field

from hardware.hw_process import hardware_main

_sessions = itertools.count(1)


@dataclass
class InitResult:
    board_info: dict = field(default_factory=dict)  # serial -> num sensors
    messages: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class HardwareClient:
    def __init__(self, backend_path: str):
        self.backend_path = backend_path
        self._proc = None
        self._cmd = None
        self._data = None
        self._lock = threading.Lock()  # one command/reply exchange at a time

    def start(self, timeout: float = 60.0) -> InitResult:
        """Spawn the hardware process and initialize the boards."""
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=True)
        self._data = ctx.Queue()
        self._proc = ctx.Process(target=hardware_main,
                                 args=(child, self._data, self.backend_path),
                                 name="cliqr-hardware", daemon=True)
        self._proc.start()
        child.close()  # so the child sees EOF if this process goes away
        self._cmd = parent
        _, info, messages, errors = self._request(("init",), timeout)
        return InitResult(info, messages, errors)

    def _request(self, msg, timeout):
        with self._lock:
            self._cmd.send(msg)
            if not self._cmd.poll(timeout):
                raise TimeoutError(f"hardware process did not answer {msg[0]!r}")
            return self._cmd.recv()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def source(self, channels, hz) -> "ProcessSource":
        return ProcessSource(self, list(channels), hz)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the process and release the boards. Idempotent."""
        if self._proc is None:
            return
        if self._proc.is_alive():
            try:
                self._request(("shutdown",), timeout)
            except Exception:
                pass
            self._proc.join(timeout)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(1.0)
        try:
            self._cmd.close()
        except Exception:
            pass


class ProcessSource:
    """Sample source backed by the hardware process."""

    def __init__(self, client: HardwareClient, channels, hz):
        self._client = client
        self._channels = channels
        self._hz = hz
        self._session = next(_sessions)
        self._receiver = None

    def start(self, sink):
        self._sink = sink
        self._client._request(("start", self._session, self._channels, self._hz), 10.0)
        self._receiver = threading.Thread(target=self._receive,
                                          name="cliqr-hw-receiver", daemon=True)
        self._receiver.start()

    def _receive(self):
        q, sink, client = self._client._data, self._sink, self._client
        while True:
            try:
                msg = q.get(timeout=0.5)
            except queue.Empty:
                if not client.is_alive():
                    sink.on_fatal("hardware process died "
                                  f"(exit code {client._proc.exitcode})")
                    return
                continue
            except (EOFError, OSError) as exc:
                sink.on_fatal(f"hardware process connection lost: {exc}")
                return
            if msg[1] != self._session:
                continue  # leftovers from an earlier session
            kind = msg[0]
            try:
                if kind == "loops":
                    sink.on_loops(msg[2])
                elif kind == "read_error":
                    sink.on_read_error(msg[2], msg[3])
                elif kind == "fatal":
                    sink.on_fatal(msg[2])
                    return
                elif kind == "end":
                    return
            except Exception as exc:
                sink.on_fatal(str(exc))
                return

    def stop(self, timeout: float) -> bool:
        """Stop sampling; True once every loop sampled has been delivered."""
        if self._receiver is None:
            return True
        ok = True
        if self._client.is_alive():
            try:
                self._client._request(("stop",), timeout)
            except Exception:
                ok = False
        else:
            ok = False
        self._receiver.join(timeout)
        return ok and not self._receiver.is_alive()
