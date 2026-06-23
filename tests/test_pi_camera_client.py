import threading
from pathlib import Path

import pytest

from hardware.pi_camera import PiCameraClient
from pi.pi_camera_server import serve
from pi.server_core import CameraServer


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
        self.video.write_bytes(b"VID")
        self.txt = self.tmp_path / f"{name}.txt"
        self.txt.write_text("100\n200\n")
        return self.video.name

    def bookmark(self, sensor_id):
        return {"frame_index": 3, "pts": 4.5, "pi_monotonic": 1.0}

    def stop_session(self):
        self._active = False
        return [
            {"name": self.video.name, "size": self.video.stat().st_size},
            {"name": self.txt.name, "size": self.txt.stat().st_size},
        ]


@pytest.fixture
def client(tmp_path):
    backend = FakeBackend(tmp_path)
    server = serve(CameraServer(backend), host="127.0.0.1", port=0,
                   output_dir=str(tmp_path))
    host, port = server.server_address
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield PiCameraClient(host, port)
    server.shutdown()
    server.server_close()


def test_ping(client):
    assert client.ping() is True


def test_full_flow_and_fetch(client, tmp_path):
    start = client.start_session("clip")
    assert start["ok"] is True
    assert start["video_filename"] == "clip.mp4"

    mark = client.bookmark(1)
    assert mark["frame_index"] == 3
    assert mark["pts"] == 4.5

    stop = client.stop_session()
    names = [f["name"] for f in stop["files"]]

    dest = tmp_path / "desktop"
    fetched = client.fetch_files(names, str(dest))
    assert {p.name for p in fetched} == {"clip.mp4", "clip.txt"}
    assert (dest / "clip.mp4").read_bytes() == b"VID"


def test_ping_unreachable():
    # Nothing listening on this port -> graceful False, no exception.
    assert PiCameraClient("127.0.0.1", 1, timeout=0.2).ping() is False
