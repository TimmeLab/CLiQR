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
