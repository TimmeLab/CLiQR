from hardware.pi_camera_mock import MockPiCameraClient


def test_mock_flow(tmp_path):
    client = MockPiCameraClient()
    assert client.ping() is True

    start = client.start_session("clip")
    assert start["ok"] is True
    assert start["video_filename"] == "clip.mp4"

    first = client.bookmark(1)
    second = client.bookmark(1)
    assert first["frame_index"] == 1
    assert second["frame_index"] == 2
    assert second["pts"] == 2 / 30.0

    stop = client.stop_session()
    names = [f["name"] for f in stop["files"]]
    fetched = client.fetch_files(names, str(tmp_path))
    assert {p.name for p in fetched} == {"clip.mp4", "clip.txt"}
    assert (tmp_path / "clip.mp4").exists()
