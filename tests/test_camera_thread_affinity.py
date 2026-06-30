"""Camera ops must run on one persistent thread, not throwaway request threads.

picamera2's encoder/FfmpegOutput pipeline is bound to the thread that called
start_recording; if that thread exits, the mp4 is never written. The desktop
client opens a new connection per request, so each lands on a different
ThreadingTCPServer handler thread that dies after responding. The server must
therefore execute all backend calls on a single long-lived thread.
"""
import socket
import threading

from pi.pi_camera_server import serve
from pi.server_core import CameraServer
from video import protocol


class ThreadRecordingBackend:
    def __init__(self):
        self._active = False
        self.start_thread = None
        self.start_thread_alive_at_stop = None
        self.stop_thread = None

    @property
    def is_active(self):
        return self._active

    def start_session(self, name):
        self.start_thread = threading.current_thread()
        self._active = True
        return f"{name}.mp4"

    def bookmark(self, sensor_id):
        return {"frame_index": 1, "pts": 0.0, "pi_monotonic": 0.0}

    def stop_session(self):
        # The thread that ran start_session (and owns the encoder/ffmpeg feed)
        # must still be alive here, else the mp4 never finalizes.
        self.start_thread_alive_at_stop = self.start_thread.is_alive()
        self.stop_thread = threading.current_thread()
        self._active = False
        return []


def _send_one(host, port, msg):
    """One request over its own connection, mirroring PiCameraClient._request."""
    with socket.create_connection((host, port)) as sock:
        sock.sendall(protocol.encode_message(msg))
        buf = b""
        while not buf.endswith(b"\n"):
            buf += sock.recv(4096)
        return protocol.decode_message(buf)


def test_camera_ops_share_one_thread(tmp_path):
    backend = ThreadRecordingBackend()
    server = serve(CameraServer(backend), host="127.0.0.1", port=0,
                   output_dir=str(tmp_path))
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        _send_one(host, port, protocol.make_request(protocol.START_SESSION, name="clip"))
        _send_one(host, port, protocol.make_request(protocol.BOOKMARK, sensor_id=1))
        _send_one(host, port, protocol.make_request(protocol.STOP_SESSION))
    finally:
        server.shutdown()
        server.server_close()

    # The thread that started recording must still be alive at stop, and stop
    # must run on that same persistent thread.
    assert backend.start_thread_alive_at_stop is True
    assert backend.stop_thread is backend.start_thread
