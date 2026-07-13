import socket
import threading
from pathlib import Path

from pi.pi_camera_server import serve
from pi.server_core import CameraServer
from video import protocol


class FakeBackend:
    def __init__(self, tmp_path):
        self._active = False
        self.tmp_path = tmp_path

    @property
    def is_active(self):
        return self._active

    def start_session(self, name):
        self._active = True
        self.video = self.tmp_path / f"{name}.mp4"
        self.video.write_bytes(b"VIDEODATA")
        return self.video.name

    def bookmark(self, sensor_id):
        return {"frame_index": 7, "pts": 2.0, "pi_monotonic": 1.0}

    def stop_session(self):
        self._active = False
        return [{"name": self.video.name, "size": self.video.stat().st_size}]


def _send(sock, msg):
    sock.sendall(protocol.encode_message(msg))
    return protocol.decode_message(_recv_line(sock))


def _recv_line(sock):
    buf = b""
    while not buf.endswith(b"\n"):
        buf += sock.recv(1)
    return buf


def test_tcp_roundtrip(tmp_path):
    backend = FakeBackend(tmp_path)
    server = serve(CameraServer(backend), host="127.0.0.1", port=0, output_dir=str(tmp_path))
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with socket.create_connection((host, port)) as sock:
            assert _send(sock, protocol.make_request(protocol.PING))["ok"] is True
            start = _send(sock, protocol.make_request(protocol.START_SESSION, name="clip"))
            assert start["video_filename"] == "clip.mp4"
            mark = _send(sock, protocol.make_request(protocol.BOOKMARK, sensor_id=1))
            assert mark["frame_index"] == 7

            # GET_FILE: header line then raw bytes
            sock.sendall(protocol.encode_message(
                protocol.make_request(protocol.GET_FILE, name="clip.mp4")))
            header = protocol.decode_message(_recv_line(sock))
            assert header["ok"] is True
            payload = b""
            while len(payload) < header["size"]:
                payload += sock.recv(4096)
            assert payload == b"VIDEODATA"
    finally:
        server.shutdown()
        server.server_close()
