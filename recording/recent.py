"""
Recent samples for the TEST plot: the live recorder's buffers, else the
latest HDF5 file. GUI-side only (no USB; the boards live in the hardware
process).
"""


def get_last_reading(sensor_id: int, num_samples: int = 250, fluid=None) -> tuple:
    """
    Get the last N samples from a sensor for testing/plotting.

    Args:
        sensor_id: Sensor ID (1-24)
        num_samples: Number of samples to retrieve
        fluid: 1 or 2 for one bottle of a 2-bottle sensor; None otherwise

    Returns:
        Tuple of (cap_data, time_data) lists, or (None, None) if no data available
    """
    # Try to get data from the live recorder buffer first
    try:
        from components.session_controls import current_recorder
        if current_recorder is not None:
            cap_data, time_data = current_recorder.get_recent_samples(
                sensor_id, num_samples, fluid=fluid)
            if cap_data is not None:
                return cap_data, time_data
    except Exception:
        pass  # Fall through to HDF5 reading

    # If no live data, try reading from the most recent HDF5 file
    try:
        from utils import state
        import h5py
        from utils.state import SERIAL_NUMBER_SENSOR_MAP, channel_group_path

        filename = state.filename.value
        if not filename:
            return None, None

        # Find which board this sensor is on: walk the board->sensors map and
        # take the serial number whose sensor list contains this sensor.
        sn = None
        for serial_number, sensor_numbers in SERIAL_NUMBER_SENSOR_MAP.items():
            if sensor_id in sensor_numbers:
                sn = serial_number
                break
        if sn is None:
            return None, None

        with h5py.File(filename, "r") as h5f:
            key = (sensor_id, fluid) if fluid is not None else sensor_id
            group_path = f"board_{sn}/{channel_group_path(key)}"
            if group_path not in h5f:
                return None, None

            group = h5f[group_path]
            if "cap_data" not in group or "time_data" not in group:
                return None, None

            # Read the last N samples
            cap_data = group["cap_data"][:]
            time_data = group["time_data"][:]

            n = min(num_samples, len(cap_data))
            return list(cap_data[-n:]), list(time_data[-n:])

    except Exception:
        return None, None
