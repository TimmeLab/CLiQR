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
    # The bookmark runs on its own thread; join it before reading the file.
    async def _run():
        return scard.start_sensor(sensor_id)
    thread = asyncio.run(_run())
    if thread is not None:
        thread.join(5.0)

    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert grp["video_frame_index"][()] == 1
        assert grp["video_filename"][()].decode() == "clip.mp4"
        # bookmark latency provenance is captured so alignment needs no manual
        # offset: Pi clock at bookmark + host wall-clock bracketing the call.
        assert "video_pi_monotonic" in grp
        hb = grp["video_bookmark_host_before"][()]
        ha = grp["video_bookmark_host_after"][()]
        assert ha >= hb

    # Cleanup global state for other tests.
    state.camera_enabled.set(False)
    state.camera_mock.set(False)
    state.camera_sensor_id.set(None)
    state.recording_all.set(False)
    sc.current_recorder = None
    sc.camera_client = None


def test_bookmark_does_not_block_the_caller(tmp_path):
    """The bookmark round-trip must run OFF the asyncio event loop: a synchronous
    bookmark here froze record_sensors for the whole round-trip, punching a
    multi-second gap into every sensor's data at session start."""
    import threading
    import time as _time
    from dataclasses import replace

    serial = next(iter(SERIAL_NUMBER_SENSOR_MAP))
    sensor_id = SERIAL_NUMBER_SENSOR_MAP[serial][0]

    rec = SensorRecorder(mpr121_manager=None,
                         filename=str(tmp_path / "raw.h5"),
                         controllers={serial: object()})
    rec.initialize_hdf5_file()
    sc.current_recorder = rec

    # A prior test may have left this sensor flagged recording; clear it so
    # start_sensor doesn't early-return.
    sensors = state.sensor_states.value.copy()
    sensors[sensor_id] = replace(sensors[sensor_id], is_recording=False)
    state.sensor_states.set(sensors)

    state.camera_enabled.set(True)
    state.camera_sensor_id.set(sensor_id)
    state.camera_video_filename.set("clip.mp4")
    state.recording_all.set(True)

    started = threading.Event()
    release = threading.Event()

    class _SlowClient:
        def bookmark(self, sensor_id):
            started.set()
            release.wait(5.0)  # hold the "round-trip" open until released
            return {"ok": True, "frame_index": 2, "pts": 0.02,
                    "pi_monotonic": 123.0}

    sc.camera_client = _SlowClient()

    async def _run():
        return scard.start_sensor(sensor_id)

    t0 = _time.perf_counter()
    thread = asyncio.run(_run())
    elapsed = _time.perf_counter() - t0

    # Caller returned while the bookmark is still blocking -> not on its thread.
    assert started.wait(1.0)
    assert elapsed < 1.0
    assert thread is not None and thread.is_alive()

    # Once the round-trip finishes, the metadata (incl. latency bracket) lands.
    release.set()
    thread.join(5.0)
    with h5py.File(rec.filename, "r") as h5f:
        grp = h5f[f"board_{serial}/sensor_{sensor_id}"]
        assert grp["video_frame_index"][()] == 2
        assert "video_pi_monotonic" in grp

    state.camera_enabled.set(False)
    state.camera_sensor_id.set(None)
    state.recording_all.set(False)
    sc.current_recorder = None
    sc.camera_client = None
