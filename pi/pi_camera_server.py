"""TCP front-end for the Pi camera server.

Each connection handles a stream of newline-delimited JSON requests. Most
commands return one JSON response line. GET_FILE returns a JSON header line
({"ok": true, "size": N}) followed by exactly N raw file bytes.

Run on the Pi:  python -m pi.pi_camera_server --port 8770 --output-dir ~/clips
"""
import argparse
import logging
import socket
import socketserver
import sys
from pathlib import Path

from pi.server_core import CameraServer
from video import protocol

log = logging.getLogger(__name__)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        server: "_Server" = self.server  # type: ignore[assignment]
        core = server.core
        for raw in self.rfile:
            if not raw.strip():
                continue
            request = protocol.decode_message(raw)
            if request.get("cmd") == protocol.GET_FILE:
                self._send_file(server.output_dir / request.get("name", ""))
                continue
            response = core.handle(request)
            self.wfile.write(protocol.encode_message(response))
            self.wfile.flush()

    def _send_file(self, path: Path):
        if not path.is_file():
            self.wfile.write(protocol.encode_message(
                protocol.make_error(f"no such file: {path.name}")))
            self.wfile.flush()
            return
        size = path.stat().st_size
        self.wfile.write(protocol.encode_message(protocol.make_ok(size=size)))
        with open(path, "rb") as fh:
            while chunk := fh.read(65536):
                self.wfile.write(chunk)
        self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, core: CameraServer, output_dir: Path):
        self.core = core
        self.output_dir = output_dir
        super().__init__(server_address, _Handler)


def serve(core: CameraServer, host: str = "0.0.0.0", port: int = 8770,
          output_dir: str = ".") -> _Server:
    """Create (but do not run) a threaded TCP server bound to host:port."""
    return _Server((host, port), core, Path(output_dir))


def configure_logging(level=logging.INFO):
    """Timestamped logging to stderr.

    Every line is stamped so a failed session can be lined up against the
    desktop's log and the HDF5 timestamps. Run the server through
    pi/run_server.sh: unbuffered output is what makes these lines actually
    reach the log file while the process is still alive.
    """
    logging.basicConfig(
        level=level, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")


def log_network_interfaces():
    """Log every up interface and its addresses at startup.

    The camera link must run over the wired interface. On 2026-07-22 wlan0 came
    back after a reboot (a `rfkill`/UI wifi toggle does not reliably persist),
    associated with a weak guest SSID in the vivarium, and roamed between APs
    every few minutes — each roam tearing down DHCP. A bookmark was refused and
    the session lost its video<->trace anchor. Logging this makes a live wlan0
    obvious in the first line of the log instead of after a lost run.
    """
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       proto=socket.IPPROTO_TCP):
            log.info("local address: %s", info[4][0])
    except Exception as exc:  # never block startup on a diagnostic
        log.warning("could not enumerate local addresses: %s", exc)
    try:
        names = [name for _, name in socket.if_nameindex()]
    except Exception as exc:
        log.warning("could not enumerate interfaces: %s", exc)
        return
    log.info("network interfaces: %s", ", ".join(names))
    wireless = [n for n in names if n.startswith(("wlan", "wlp"))]
    if wireless:
        log.warning("wireless interface(s) present: %s. The camera link must "
                    "run over the wired interface; a roaming wifi association "
                    "will drop requests mid-session. Disable with "
                    "'dtoverlay=disable-wifi' in /boot/firmware/config.txt.",
                    ", ".join(wireless))


def main():
    parser = argparse.ArgumentParser(description="CLiQR Pi camera server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    configure_logging()
    from pi.camera_backend import Picamera2Backend
    core = CameraServer(Picamera2Backend(output_dir=args.output_dir))
    server = serve(core, host=args.host, port=args.port, output_dir=args.output_dir)
    # Logged AFTER the bind succeeds: if this line is absent from the log, the
    # socket never came up and every desktop attempt saw "connection refused".
    log.info("listening on %s:%s, output-dir=%s",
             args.host, args.port, args.output_dir)
    log_network_interfaces()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("shutting down")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
