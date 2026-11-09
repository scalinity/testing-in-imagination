#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
exec > "$TIM_WORK_DIR/private-logs/$SLURM_JOB_ID.log" 2>&1
# Everything below writes only to the private log above (SPEC 1.3).
set -euo pipefail
export VLLM_USE_V1=0 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
# The replication's stage launcher sets this allocator option for every GPU stage.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "${TIM_PYTHON:-$CDM_WORK_DIR/runtime-env/bin/python}" scripts/run_tim.py "$@"
