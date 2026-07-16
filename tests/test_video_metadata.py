import h5py
from recording.recorder import SensorRecorder
from utils.state import SERIAL_NUMBER_SENSOR_MAP


def _make_recorder(tmp_path):
    serial = next(iter(SERIAL_NUMBER_SENSOR_MAP))  # a real serial from the map
    controllers = {serial: object()}
    rec = SensorRecorder(mpr121_manager=None,
                         filename=str(tmp_path / "raw.h5"),
                         controllers=controllers)
    rec.initialize_hdf5_file()
    return rec, serial, SERIAL_NUMBER_SENSOR_MAP[serial][0]


def test_write_video_metadata_cycle0(tmp_path):
    rec, serial, sensor_id = _make_recorder(tmp_path)
    rec.write_video_metadata(sensor_id=sensor_id, frame_index=42, pts=1.25,
                             video_filename="clip.mp4", cycle=0)
    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert grp["video_frame_index"][()] == 42
        assert abs(grp["video_pts"][()] - 1.25) < 1e-9
        assert grp["video_filename"][()].decode() == "clip.mp4"


def test_write_video_metadata_cycle1_suffix(tmp_path):
    rec, serial, sensor_id = _make_recorder(tmp_path)
    rec.write_video_metadata(sensor_id=sensor_id, frame_index=7, pts=0.5,
                             video_filename="clip.mp4", cycle=1)
    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert grp["video_frame_index1"][()] == 7


def test_write_video_metadata_records_bookmark_latency(tmp_path):
    # The bookmark round-trip's host bracket + Pi clock let the true latency
    # (frame's host time - start_time) be recovered later, so no manual offset
    # is needed. Persist all three.
    rec, serial, sensor_id = _make_recorder(tmp_path)
    rec.write_video_metadata(sensor_id=sensor_id, frame_index=42, pts=1.25,
                             video_filename="clip.mp4", cycle=0,
                             pi_monotonic=987.5,
                             host_time_before=1000.0, host_time_after=1000.2)
    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert abs(grp["video_pi_monotonic"][()] - 987.5) < 1e-9
        assert abs(grp["video_bookmark_host_before"][()] - 1000.0) < 1e-9
        assert abs(grp["video_bookmark_host_after"][()] - 1000.2) < 1e-9


def test_write_video_metadata_omits_latency_when_absent(tmp_path):
    # Backward compatible: recordings that don't pass the new fields (older
    # callers) write only the original three datasets.
    rec, serial, sensor_id = _make_recorder(tmp_path)
    rec.write_video_metadata(sensor_id=sensor_id, frame_index=1, pts=0.0,
                             video_filename="clip.mp4", cycle=0)
    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert "video_pi_monotonic" not in grp
        assert "video_bookmark_host_before" not in grp
        assert "video_bookmark_host_after" not in grp


def test_write_video_metadata_writes_stop_bookmark_datasets(tmp_path):
    # The Stop bookmark is the second clock anchor: its frame/pts + host bracket
    # let the video<->cap clock-rate drift be fit across the session.
    rec, serial, sensor_id = _make_recorder(tmp_path)
    rec.write_video_metadata(
        sensor_id=sensor_id, frame_index=10, pts=1.0, video_filename="v.mp4",
        cycle=0,
        stop_frame_index=200, stop_pts=5.0,
        stop_host_before=1000.0, stop_host_after=1000.4)
    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert int(grp["video_stop_frame_index"][()]) == 200
        assert abs(grp["video_stop_pts"][()] - 5.0) < 1e-9
        assert abs(grp["video_stop_bookmark_host_before"][()] - 1000.0) < 1e-9
        assert abs(grp["video_stop_bookmark_host_after"][()] - 1000.4) < 1e-9


def test_write_video_metadata_omits_stop_datasets_when_none(tmp_path):
    rec, serial, sensor_id = _make_recorder(tmp_path)
    rec.write_video_metadata(sensor_id=sensor_id, frame_index=10, pts=1.0,
                             video_filename="v.mp4", cycle=0)
    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert "video_stop_frame_index" not in grp
        assert "video_stop_pts" not in grp
        assert "video_stop_bookmark_host_before" not in grp
        assert "video_stop_bookmark_host_after" not in grp
