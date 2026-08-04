#!/bin/bash
# SLURM array job: one task renders one high-confidence window from dlc_windows.csv into a
# labeled video, reusing the existing DLC predictions (no clip cutting, no re-analysis).
#
#   python dlc_integration/find_dlc_windows.py "Lickometry Data/ACG-26-3" --csv dlc_windows.csv
#   sbatch --array=1-$(( $(wc -l < dlc_windows.csv) - 1 )) dlc_integration/slurm_dlc_label_windows.sbatch
#
# or just use dlc_integration/submit_dlc_label_windows.sh, which computes the array range for you.
#
# Configure via environment (all optional except DLC_CONFIG):
#   DLC_CONFIG   path to the project config.yaml            (required)
#   WINDOW_CSV   windows CSV                                (default dlc_windows.csv)
#   OUT_DIR      where labeled videos land                  (default labeled_windows)
#   DLC_ENV      conda env to activate                      (default deeplabcut)
#   STAGE_DIR    hardlink staging root, same FS as the video (default: beside it)
#   EXTRA_ARGS   passed through to dlc_label_window.py      (e.g. "--trailpoints 10")
#
# Note on the partition: create_labeled_video's Frames2plot path renders with matplotlib, so this
# is CPU/IO bound and does not really need the A100. gpu-a100 is used because that is where the
# DLC environment is; a CPU partition would work if one is available with the same env.

#SBATCH --job-name=dlc-label-window
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dlc_label_%A_%a.out
#SBATCH --error=logs/dlc_label_%A_%a.err
##SBATCH --account=CHANGE_ME
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=c.parker51@gmail.com

set -euo pipefail

WINDOW_CSV="${WINDOW_CSV:-dlc_windows.csv}"
OUT_DIR="${OUT_DIR:-labeled_windows}"
DLC_ENV="${DLC_ENV:-deeplabcut}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${DLC_CONFIG:-}" ]]; then
    echo "DLC_CONFIG is not set (path to the DLC project config.yaml)" >&2
    exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"
mkdir -p logs "$OUT_DIR"

# NOTE: staging cannot go on node-local scratch ($SLURM_TMPDIR). The video is hardlinked into the
# staging directory to give each window its own DLC output name, and a hardlink cannot cross
# filesystems -- so staging defaults to .dlc_label_staging beside the recording on Lustre. Set
# STAGE_DIR only to another path on that same filesystem.
STAGE_ARG=()
if [[ -n "${STAGE_DIR:-}" ]]; then
    STAGE_ARG=(--stage-dir "$STAGE_DIR")
fi

module load conda 2>/dev/null || module load miniconda 2>/dev/null || true
# conda's activate hooks are not written for `set -u`: the MKL one reads MKL_INTERFACE_LAYER
# before setting it, which aborts the job with "unbound variable". nounset is off for activation.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DLC_ENV"
set -u

export DLClight=True          # no GUI backend on compute nodes
export MPLBACKEND=Agg

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-1} csv=$WINDOW_CSV"

python dlc_integration/dlc_label_window.py \
    --config "$DLC_CONFIG" \
    --csv "$WINDOW_CSV" \
    --row "${SLURM_ARRAY_TASK_ID:-1}" \
    --out-dir "$OUT_DIR" \
    "${STAGE_ARG[@]}" \
    $EXTRA_ARGS
