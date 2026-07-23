#!/usr/bin/env python3
"""Pi-only probe: what does picamera2 hand to ``Output.outputframe`` per frame?

Run this on the Pi for a few seconds. It records with the SAME config/encoder/
CFR mux as production (``camera_backend`` + ``ffmpeg_output``), but wraps the
output so it captures, for the first N encoded frames, the ``timestamp`` argument
picamera2 passes to ``outputframe`` -- alongside the capture ``SensorTimestamp``
values logged in ``pre_callback`` (what today's ``.txt`` sidecar holds).

It answers the three questions that decide the drop-marking sidecar format:

  1. Is a per-frame ``timestamp`` actually passed (not ``None``)?
  2. Units -- nanoseconds or microseconds?
  3. Same clock as ``SensorTimestamp`` (so ``.encpts.txt`` lines up with the
     existing ``.txt`` sidecar and the bookmark anchor)?

It also prints encoded-vs-captured frame counts and the muxed mp4's frame count,
confirming the encoder drops frames (encoded < captured) and that the mp4 carries
exactly one frame per ``outputframe`` call.

    python3 -m pi.probe_encoded_timestamps            # ~5 s, /tmp/probe.mp4

Nothing here is imported off-Pi (picamera2 only loads in ``main``).
"""
import subprocess
import sys
import time
from pathlib import Path

from pi import ffmpeg_output
from pi.camera_backend import TARGET_FPS, encoder_kwargs, video_config_kwargs

N = 25            # how many encoded frames to print in detail
KEEP_CAP = 400    # how many capture SensorTimestamps to retain for matching
DURATION_S = 5.0
OUT_MP4 = "/tmp/probe.mp4"


def _make_probe_output(stdin):
    """A FileOutput that records the ``timestamp`` picamera2 passes per encoded
    frame. Built inside main so picamera2 imports stay Pi-only."""
    from picamera2.outputs import FileOutput

    class ProbeOutput(FileOutput):
        def __init__(self, fileobj):
            super().__init__(fileobj)
            self.stamps = []   # timestamp arg for the first N encoded frames
            self.count = 0     # total encoded frames (== container frames)

        def outputframe(self, frame, keyframe=True, timestamp=None,
                        *args, **kwargs):
            if self.count < N:
                self.stamps.append(timestamp)
            self.count += 1
            return super().outputframe(frame, keyframe, timestamp,
                                       *args, **kwargs)

    return ProbeOutput(stdin)


def _report(enc_stamps, enc_count, cap_stamps, cap_count, mp4_frames):
    print("\n================ PROBE RESULT ================")
    print(f"encoded (outputframe) frames : {enc_count}")
    print(f"captured (pre_callback) frames: {cap_count}")
    print(f"muxed mp4 frames             : {mp4_frames}")
    if cap_count:
        print(f"encoded/captured ratio       : {enc_count / cap_count:.4f} "
              f"(<1 => encoder dropped {cap_count - enc_count} frames)")

    print("\n--- first encoded-frame timestamps vs capture SensorTimestamps ---")
    ts0 = enc_stamps[0] if enc_stamps else None
    if ts0 is None:
        print("  timestamp is None -> outputframe carries NO per-frame time.")
        print("  => use the encoder-subclass route, not an output wrapper.")
    else:
        print(f"  timestamp type: {type(ts0).__name__}")
    for i in range(min(N, len(enc_stamps))):
        ets = enc_stamps[i]
        cts = cap_stamps[i] if i < len(cap_stamps) else None
        line = f"  [{i:2d}] outputframe={ets!r}"
        if cts is not None:
            line += f"   SensorTimestamp={cts}"
            if isinstance(ets, (int, float)) and ets:
                # unit/clock hints: ns match ~1.0, us match ~1000
                line += f"   cap/out={cts / float(ets):.3f}"
        print(line)

    # Unit/clock heuristic from the whole captured set (robust to ordering).
    nums = [t for t in enc_stamps if isinstance(t, (int, float)) and t]
    if nums and cap_stamps:
        import numpy as np
        e = np.array(nums, dtype=float)
        c = np.array(cap_stamps[:len(nums)], dtype=float)
        for name, scale in (("nanoseconds", 1.0), ("microseconds", 1e3),
                            ("milliseconds", 1e6), ("seconds", 1e9)):
            # does encoded*scale land on a capture SensorTimestamp (ns)?
            diffs = np.min(np.abs(c[:, None] - (e * scale)[None, :]), axis=1)
            med = float(np.median(diffs))
            print(f"  if outputframe is {name:12s}: median match error "
                  f"{med:.0f} ns")
        print("  (the unit with ~0 match error is picamera2's outputframe unit;\n"
              "   a good match also confirms it IS the SensorTimestamp clock.)")
    print("=============================================\n")


def main():
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder

    cap_stamps = []
    cap_count = [0]

    picam = Picamera2()
    picam.configure(picam.create_video_configuration(**video_config_kwargs()))

    def on_frame(request):
        cap_count[0] += 1
        if len(cap_stamps) < KEEP_CAP:
            md = request.get_metadata()
            cap_stamps.append(int(md.get("SensorTimestamp", 0)))

    picam.pre_callback = on_frame

    proc = subprocess.Popen(
        ffmpeg_output.ffmpeg_command(OUT_MP4, TARGET_FPS),
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    out = _make_probe_output(proc.stdin)
    enc = H264Encoder(**encoder_kwargs())

    picam.start_recording(enc, out)
    time.sleep(DURATION_S)
    picam.stop_recording()
    picam.close()
    ffmpeg_output.finish_muxer(proc)

    mp4_frames = -1
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", OUT_MP4],
            capture_output=True, text=True)
        mp4_frames = int(r.stdout.strip() or -1)
    except Exception as exc:  # noqa: BLE001 - probe, never fatal
        print(f"(ffprobe frame count failed: {exc})", file=sys.stderr)

    _report(out.stamps, out.count, cap_stamps, cap_count[0], mp4_frames)
    Path(OUT_MP4).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
