"""The recorder serializes all HDF5 access through one lock.

HDF5 built without --enable-threadsafe gives undefined behavior when two threads
write the same file at once. In this recorder the periodic buffer flush runs on
the recording task while metadata/bookmark writes fire from GUI event handlers on
another thread, so all file access must be serialized.

This test hammers the metadata writers from several threads at once (the same
r+ openers that collide with a flush in production) and asserts every write
lands and the file stays readable — the locked path must not error or lose data.
"""
import threading

import h5py

from recording.recorder import SensorRecorder
from utils.state import SERIAL_NUMBER_SENSOR_MAP


def test_concurrent_metadata_writes_are_serialized(tmp_path):
    serial = next(iter(SERIAL_NUMBER_SENSOR_MAP))
    sensors = SERIAL_NUMBER_SENSOR_MAP[serial]

    rec = SensorRecorder(mpr121_manager=None,
                         filename=str(tmp_path / "raw.h5"),
                         controllers={serial: object()})
    rec.initialize_hdf5_file()

    errors = []

    def writer(sensor_id):
        try:
            for i in range(50):
                rec.write_sensor_metadata(sensor_id=sensor_id, start_time=1000.0 + i)
                rec.write_video_metadata(sensor_id=sensor_id, frame_index=i,
                                         pts=float(i), video_filename="clip.mp4")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=writer, args=(s,)) for s in sensors]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors

    # File is intact and every sensor's writes are present.
    with h5py.File(rec.filename, "r") as h5f:
        for s in sensors:
            grp = h5f[f"board_{serial}/sensor_{s}"]
            assert "start_time" in grp
            assert grp["video_filename"][()].decode() == "clip.mp4"
