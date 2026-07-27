"""Desktop-side TCP client for the Pi camera server.

Every method is best-effort: connection or protocol failures are caught and
returned as error dicts (or False for ping). The camera is auxiliary and must
never raise into the recording loop.
"""
import socket
from pathlib import Path

from video import protocol

# Per-address connect timeout. getaddrinfo('picamera0.local') lists a dead IPv6
# link-local before the wired IPv4; the working IPv4 answers in ~1-24 ms, so a
# 0.1 s cap abandons the dead address fast instead of burning the full timeout.
CONNECT_TIMEOUT_S = 0.1


class PiCameraClient:
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        # The getaddrinfo tuple that last connected, reused for the rest of the
        # session so only the first request pays resolution. None until resolved.
        self._addr = None

    def _connect_addr(self, info) -> socket.socket:
        """Open a connected socket to one getaddrinfo tuple. The connect phase
        uses the short CONNECT_TIMEOUT_S; the returned socket carries the full
        self.timeout for send/recv. Closes the socket and re-raises on failure."""
        family, socktype, proto, _canon, sockaddr = info
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(CONNECT_TIMEOUT_S)
            sock.connect(sockaddr)
        except OSError:
            sock.close()
            raise
        sock.settimeout(self.timeout)
        return sock

    def _connect(self) -> socket.socket:
        """Connect to the Pi, preferring the cached address; else resolve
        IPv4-first and cache whichever address answers.

        getaddrinfo lists a dead IPv6 link-local before the wired IPv4, so a
        fresh resolve+connect on every request wasted ~2-5 s (and its variance
        skewed the video<->trace bookmark slope). A stale cache -- e.g. the Pi's
        IP changed mid-session -- self-heals: the failed fast-path connect clears
        it and this same call re-resolves."""
        if self._addr is not None:
            try:
                return self._connect_addr(self._addr)
            except OSError:
                self._addr = None
        infos = socket.getaddrinfo(self.host, self.port,
                                   proto=socket.IPPROTO_TCP)
        infos.sort(key=lambda i: i[0] != socket.AF_INET)  # AF_INET (IPv4) first
        last_exc = None
        for info in infos:
            try:
                sock = self._connect_addr(info)
            except OSError as exc:
                last_exc = exc
                continue
            self._addr = info
            return sock
        raise last_exc if last_exc is not None else OSError(
            f"no addresses for {self.host}:{self.port}")

    def _request(self, msg: dict) -> dict:
        """Send one request, return the decoded response dict (or error dict)."""
        try:
            with self._connect() as sock:
                sock.sendall(protocol.encode_message(msg))
                return protocol.decode_message(self._recv_line(sock))
        except Exception as exc:
            # Transport is the wired link (wifi is disabled on the Pi). Name
            # resolution can list a dead IPv6 link-local before the wired IPv4,
            # so _connect caches whichever address actually answers; naming the
            # endpoint keeps a bare "[WinError 10061] ... refused" diagnosable.
            return protocol.make_error(f"{exc} [{self.host}:{self.port}]")

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
            with self._connect() as sock:
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
