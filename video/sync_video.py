"""Trim recorded video to the capacitive recording window and render a
side-by-side synced MP4 (video left, sliding capacitive trace right).

Alignment: the video bookmark (frame PTS at sipper insertion) equals the
sensor start_time, so frame f's absolute Unix time is
    abs(f) = start_time_abs + (frame_offsets_ns[f] / 1e9 - bookmark_pts).
"""
import numpy as np


def frame_abs_times(frame_offsets_ns, bookmark_pts, start_time_abs):
    """Absolute Unix seconds for every video frame."""
    offsets = np.asarray(frame_offsets_ns, dtype=float)
    return start_time_abs + (offsets / 1e9 - bookmark_pts)


def compute_trim_frames(frame_offsets_ns, bookmark_pts, start_time_abs, stop_time_abs):
    """Inclusive (start_frame, stop_frame) covering [start_time_abs, stop_time_abs]."""
    abs_t = frame_abs_times(frame_offsets_ns, bookmark_pts, start_time_abs)
    mask = (abs_t >= start_time_abs) & (abs_t <= stop_time_abs)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("no video frames fall within the recording window")
    return int(idx[0]), int(idx[-1])


def trim_video(input_path, output_path, start_frame, stop_frame, fps=None):
    """Write frames [start_frame, stop_frame] (inclusive) to a new MP4."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(input_path)
    out_fps = fps or reader.get_meta_data().get("fps", 30)
    writer = imageio.get_writer(output_path, fps=out_fps)
    try:
        for i, frame in enumerate(reader):
            if i < start_frame:
                continue
            if i > stop_frame:
                break
            writer.append_data(frame)
    finally:
        writer.close()
        reader.close()
    return output_path


def render_synced_video(video_path, frame_offsets_ns, cap_time, cap_data, lick_times,
                        bookmark_pts, start_time_abs, stop_time_abs, output_path,
                        window_sec=1.0, fps=None):
    """Render an MP4: left = video frame, right = capacitive trace zoomed to
    ±window_sec around the current time (center fixed, window slides), with
    detected licks marked."""
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    abs_t = frame_abs_times(frame_offsets_ns, bookmark_pts, start_time_abs)
    start_frame, stop_frame = compute_trim_frames(
        frame_offsets_ns, bookmark_pts, start_time_abs, stop_time_abs)

    cap_time = np.asarray(cap_time, dtype=float)
    cap_data = np.asarray(cap_data, dtype=float)
    lick_times = np.asarray(lick_times, dtype=float) if lick_times is not None else np.array([])

    window_mask = (cap_time >= start_time_abs) & (cap_time <= stop_time_abs)
    if window_mask.any():
        ylim = (cap_data[window_mask].min(), cap_data[window_mask].max())
    else:
        ylim = (cap_data.min(), cap_data.max())

    reader = imageio.get_reader(video_path)
    out_fps = fps or reader.get_meta_data().get("fps", 30)

    fig, (ax_vid, ax_tr) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1]})
    ax_vid.axis("off")
    image = ax_vid.imshow(reader.get_data(start_frame))

    ax_tr.plot(cap_time, cap_data, color="steelblue", lw=0.8)
    if lick_times.size:
        ax_tr.scatter(lick_times, np.interp(lick_times, cap_time, cap_data),
                      color="red", s=20, zorder=5, label="detected licks")
        ax_tr.legend(loc="upper right", fontsize=8)
    center0 = abs_t[start_frame]
    vline = ax_tr.axvline(center0, color="k", lw=1)
    ax_tr.set_ylim(*ylim)
    ax_tr.set_xlabel("time (s, Unix)")
    ax_tr.set_ylabel("capacitance")

    def update(f):
        image.set_data(reader.get_data(f))
        center = abs_t[f]
        ax_tr.set_xlim(center - window_sec, center + window_sec)
        vline.set_xdata([center, center])
        return image, vline

    anim = FuncAnimation(fig, update, frames=range(start_frame, stop_frame + 1),
                         blit=False)
    try:
        anim.save(output_path, writer=FFMpegWriter(fps=out_fps))
    finally:
        reader.close()
        plt.close(fig)
    return output_path


def make_sync_video_from_hdf5(raw_h5, sensor_id, video_path, frame_offsets_path,
                              output_path, lick_times=None, cycle=0,
                              window_sec=1.0, fps=None):
    """Load bookmark, recording window, and trace for a sensor from the raw HDF5,
    then render the synced video. `lick_times` (absolute Unix seconds) come from
    the filtered-data analysis; pass them in to mark detected licks.

    NOTE: verify the HDF5 group path and lick-time source against the analysis
    notebook before relying on this in batch.
    """
    import h5py

    suffix = "" if cycle == 0 else str(cycle)
    frame_offsets_ns = np.loadtxt(frame_offsets_path)

    from utils.state import SERIAL_NUMBER_SENSOR_MAP
    sn = [s for s, sensors in SERIAL_NUMBER_SENSOR_MAP.items() if sensor_id in sensors][0]

    with h5py.File(raw_h5, "r") as h5f:
        grp = h5f[f"board_{sn}/sensor_{sensor_id}"]
        bookmark_pts = float(grp[f"video_pts{suffix}"][()])
        start_time_abs = float(grp[f"start_time{suffix}"][()])
        stop_time_abs = float(grp[f"stop_time{suffix}"][()])
        cap_time = grp["time_data"][()]
        cap_data = grp["cap_data"][()]

    return render_synced_video(
        video_path, frame_offsets_ns, cap_time, cap_data, lick_times,
        bookmark_pts, start_time_abs, stop_time_abs, output_path,
        window_sec=window_sec, fps=fps)
