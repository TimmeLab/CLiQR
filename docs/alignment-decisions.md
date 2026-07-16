# Cap ↔ Video Alignment — Settled Decisions

Decisions we do **not** want to relitigate. Read before proposing any new
alignment scheme.

## Do NOT anchor on sipper insertion detected in the capacitance trace

**Decision (2026-07-16): rejected.** Anchoring the video clock to the sensor
clock by detecting the sipper-insertion event *in the capacitance trace* (the
`detect_sipper_step` / `establish_alignment` approach) is not accurate enough and
will not be pursued.

**Why.** Sipper insertion is a broad, slow capacitance disturbance while
personnel handle the sipper — a multi-second dip, not a sharp edge. You can see
the *general region* of time it happened, but there is no single sample that
unambiguously *is* "the insertion." Whatever point you pick (dip minimum, step
edge, etc.) cannot be matched to the same instant in the video to better than
~seconds, which defeats the purpose. The video-side sipper time, in turn, only
exists as a hand annotation. So the method pairs one fuzzy estimate with one
manual estimate — worse than the alternative below, and not automatable to the
precision we need.

**Do instead.** Anchor on the **video bookmark** recorded at the sensor Start
click (`video_frame_index` / `video_pts` + the host-time bracket), with the
`bookmark_latency` correction. That is a single, well-defined shared instant on
both clocks and needs no trace feature detection. See
`docs/video-sync-alignment-bugs.md` and `alignment_from_bookmark()`.

**Residual drift** over a long (~2 h) session is handled by a **second bookmark
at the sensor Stop click**, giving a two-point linear clock fit — NOT by a second
sipper (removal) detection, which has the same fuzziness problem.
