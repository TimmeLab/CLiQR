"""TCP front-end for the Pi camera server.

Each connection handles a stream of newline-delimited JSON requests. Most
commands return one JSON response line. GET_FILE returns a JSON header line
({"ok": true, "size": N}) followed by exactly N raw file bytes.

Run on the Pi:  python -m pi.pi_camera_server --port 8770 --output-dir ~/clips
"""
import argparse
import socketserver
from pathlib import Path

from pi.server_core import CameraServer
from video import protocol


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


def main():
    parser = argparse.ArgumentParser(description="CLiQR Pi camera server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    from pi.camera_backend import Picamera2Backend
    core = CameraServer(Picamera2Backend(output_dir=args.output_dir))
    server = serve(core, host=args.host, port=args.port, output_dir=args.output_dir)
    print(f"CLiQR camera server listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
