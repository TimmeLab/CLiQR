#!/bin/bash
# Submit the labeled-window array, sizing --array from the CSV so the two never drift apart.
#
#   DLC_CONFIG=/N/lustre/project/proj-530/dlc_projects/CLiQR_Validation-parkecp-2026-07-27/config.yaml \
#   ./dlc_integration/submit_dlc_label_windows.sh dlc_windows.csv
#
# Extra arguments are forwarded to sbatch, e.g.:
#   ./dlc_integration/submit_dlc_label_windows.sh dlc_windows.csv --array=1-5 --time=00:20:00

set -euo pipefail

CSV="${1:-dlc_windows.csv}"
shift || true

if [[ ! -f "$CSV" ]]; then
    echo "no such windows CSV: $CSV (run dlc_integration/find_dlc_windows.py first)" >&2
    exit 1
fi
if [[ -z "${DLC_CONFIG:-}" ]]; then
    echo "set DLC_CONFIG to the DLC project config.yaml before submitting" >&2
    exit 2
fi

N=$(( $(wc -l < "$CSV") - 1 ))       # minus the header row
if (( N < 1 )); then
    echo "$CSV has no windows" >&2
    exit 1
fi

# %8 throttles to 8 concurrent tasks; raise/lower to taste.
echo "submitting $N windows from $CSV"
exec sbatch \
    --array="1-${N}%8" \
    --export=ALL,WINDOW_CSV="$CSV",DLC_CONFIG="$DLC_CONFIG",OUT_DIR="${OUT_DIR:-labeled_windows}",DLC_ENV="${DLC_ENV:-deeplabcut}",EXTRA_ARGS="${EXTRA_ARGS:-}",STAGE_DIR="${STAGE_DIR:-}" \
    "$@" \
    dlc_integration/slurm_dlc_label_windows.sbatch
