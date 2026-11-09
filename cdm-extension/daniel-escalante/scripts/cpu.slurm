#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
exec > "$TIM_WORK_DIR/private-logs/$SLURM_JOB_ID.log" 2>&1
# Everything below writes only to the private log above (SPEC 1.3).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec "${TIM_PYTHON:-$CDM_WORK_DIR/runtime-env/bin/python}" scripts/run_tim.py "$@"
