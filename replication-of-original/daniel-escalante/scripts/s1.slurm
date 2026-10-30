#!/usr/bin/env bash
#SBATCH --job-name=cdm-s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=12:00:00
set -euo pipefail
export VLLM_USE_V1=0
# Submit with --chdir pointing at this folder; account/partition are submission flags.
srun --ntasks=1 bash scripts/stage.sh s1 "$@"
