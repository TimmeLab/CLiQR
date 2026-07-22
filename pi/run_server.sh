#!/usr/bin/env bash
#
# Launch the CLiQR camera server with its output redirected to a log file.
#
# Never run the server with its stdout/stderr attached to an interactive
# terminal. picamera2 spawns ffmpeg as a child that inherits those handles, and
# on 2026-07-21 ffmpeg started rejecting every packet and emitting two error
# lines per frame (~240 lines/s at 120 fps). If the tty/ssh consumer stops
# draining that flood, ffmpeg blocks writing stderr, stops reading its stdin
# pipe, and the back-pressure reaches picamera2's request loop and stalls the
# camera -- silently, 44 min into a 2 h 19 min session. Writing to a file
# cannot block that way.
#
# Usage:  ./pi/run_server.sh [--port 8770] [--output-dir ~/cliqr_clips]
# Log:    ~/cliqr_camera_server.log  (override with CLIQR_LOG)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${CLIQR_LOG:-$HOME/cliqr_camera_server.log}"

# Keep one previous run's log so a crash is still diagnosable after a restart.
if [ -f "$LOG" ]; then
    mv -f "$LOG" "$LOG.1"
fi

cd "$REPO_DIR"
echo "CLiQR camera server starting $(date -Is); logging to $LOG"
exec python -m pi.pi_camera_server "$@" >>"$LOG" 2>&1
