#!/bin/bash
# Extract outlier frames from every analyzed video in the DLC project, in one (non-array) job.
#
#   DLC_CONFIG=/N/lustre/.../CLiQR_Validation-parkecp-2026-07-27/config.yaml \
#       sbatch dlc_integration/slurm_extract_outliers.sbatch
#
# Configure via environment (all optional except DLC_CONFIG):
#   DLC_CONFIG   path to the project config.yaml                 (required)
#   VIDEOS_DIR   videos to scan                                  (default <project>/videos)
#   NUM_FRAMES   frames per video per algorithm                  (default 20)
#   ALGORITHMS   comma-separated outlier algorithms              (default jump,uncertain)
#   DLC_ENV      conda env to activate                           (default deeplabcut)
#   EXTRA_ARGS   passed through to extract_outliers.py           (e.g. "--savelabeled")
#
# kmeans clustering of the flagged frames is CPU/IO bound and needs no GPU; this uses gpu-a100 only
# because that is where the DLC environment lives. Drop the --gres line for a CPU partition.

#SBATCH --job-name=dlc-outliers
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dlc_outliers_%j.out
#SBATCH --error=logs/dlc_outliers_%j.err
##SBATCH --account=CHANGE_ME
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=c.parker51@gmail.com

set -euo pipefail

DLC_ENV="${DLC_ENV:-deeplabcut}"
NUM_FRAMES="${NUM_FRAMES:-20}"
ALGORITHMS="${ALGORITHMS:-jump,uncertain}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${DLC_CONFIG:-}" ]]; then
    echo "DLC_CONFIG is not set (path to the DLC project config.yaml)" >&2
    exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"
mkdir -p logs

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DLC_ENV"

VIDEOS_ARG=()
if [[ -n "${VIDEOS_DIR:-}" ]]; then
    VIDEOS_ARG=(--videos-dir "$VIDEOS_DIR")
fi

# The script never prompts (automatic=True) -- do not add --interactive here, it would hang the
# job until the wall clock runs out.
python dlc_integration/extract_outliers.py \
    --config "$DLC_CONFIG" \
    --outlieralgorithm "$ALGORITHMS" \
    --numframes2pick "$NUM_FRAMES" \
    "${VIDEOS_ARG[@]}" \
    ${EXTRA_ARGS}
