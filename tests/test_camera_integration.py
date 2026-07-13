import asyncio
import h5py

from utils import state
from utils.state import SERIAL_NUMBER_SENSOR_MAP
from recording.recorder import SensorRecorder
import components.session_controls as sc
import components.sensor_card as scard


def test_bookmark_written_on_sensor_start(tmp_path):
    serial = next(iter(SERIAL_NUMBER_SENSOR_MAP))
    sensor_id = SERIAL_NUMBER_SENSOR_MAP[serial][0]

    # Build a real recorder over a temp HDF5 file.
    rec = SensorRecorder(mpr121_manager=None,
                         filename=str(tmp_path / "raw.h5"),
                         controllers={serial: object()})
    rec.initialize_hdf5_file()
    sc.current_recorder = rec

    # Camera: mock + enabled, designate this sensor as the camera sensor.
    state.camera_mock.set(True)
    state.camera_enabled.set(True)
    state.camera_sensor_id.set(sensor_id)
    state.recording_all.set(True)

    # Simulate session start wiring (start_recording's camera block).
    sc.camera_client = state.make_camera_client()
    resp = sc.camera_client.start_session("clip")
    state.camera_video_filename.set(resp["video_filename"])

    # Drive the per-sensor Start inside a running loop (start_sensor uses create_task).
    async def _run():
        scard.start_sensor(sensor_id)
    asyncio.run(_run())

    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert grp["video_frame_index"][()] == 1
        assert grp["video_filename"][()].decode() == "clip.mp4"

    # Cleanup global state for other tests.
    state.camera_enabled.set(False)
    state.camera_mock.set(False)
    state.camera_sensor_id.set(None)
    state.recording_all.set(False)
    sc.current_recorder = None
    sc.camera_client = None
