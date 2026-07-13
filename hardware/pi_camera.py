"""Desktop-side TCP client for the Pi camera server.

Every method is best-effort: connection or protocol failures are caught and
returned as error dicts (or False for ping). The camera is auxiliary and must
never raise into the recording loop.
"""
import socket
from pathlib import Path

from video import protocol


class PiCameraClient:
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _request(self, msg: dict) -> dict:
        """Send one request, return the decoded response dict (or error dict)."""
        try:
            with socket.create_connection((self.host, self.port), self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(protocol.encode_message(msg))
                return protocol.decode_message(self._recv_line(sock))
        except Exception as exc:
            return protocol.make_error(str(exc))

    @staticmethod
    def _recv_line(sock: socket.socket) -> bytes:
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        return buf

    def ping(self) -> bool:
        return self._request(protocol.make_request(protocol.PING)).get("ok", False)

    def start_session(self, name: str) -> dict:
        return self._request(protocol.make_request(protocol.START_SESSION, name=name))

    def bookmark(self, sensor_id) -> dict:
        return self._request(protocol.make_request(protocol.BOOKMARK, sensor_id=sensor_id))

    def stop_session(self) -> dict:
        return self._request(protocol.make_request(protocol.STOP_SESSION))

    def snapshot(self) -> dict:
        """Grab one still JPEG (base64 in the reply) for an alignment check."""
        return self._request(protocol.make_request(protocol.SNAPSHOT))

    def fetch_files(self, names, dest_dir: str) -> list:
        """Download each named file via GET_FILE into dest_dir. Returns paths."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        try:
            with socket.create_connection((self.host, self.port), self.timeout) as sock:
                sock.settimeout(self.timeout)
                reader = sock.makefile("rb")
                for name in names:
                    sock.sendall(protocol.encode_message(
                        protocol.make_request(protocol.GET_FILE, name=name)))
                    header_line = reader.readline()
                    if not header_line:
                        break
                    header = protocol.decode_message(header_line)
                    if not header.get("ok"):
                        continue
                    saved.append(self._recv_file_buffered(reader, dest / name, header["size"]))
        except Exception:
            return [p for p in saved if p is not None]
        return [p for p in saved if p is not None]

    @staticmethod
    def _recv_file_buffered(reader, path: Path, size: int):
        received = 0
        with open(path, "wb") as fh:
            while received < size:
                chunk = reader.read(min(65536, size - received))
                if not chunk:
                    return None
                fh.write(chunk)
                received += len(chunk)
        return path
