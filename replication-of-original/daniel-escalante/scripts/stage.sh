#!/usr/bin/env bash
set -euo pipefail
own="$(cd "$(dirname "$0")/.." && pwd)"
export VLLM_USE_V1=0
bootstrap="${CDM_PYTHON:-$own/.venv/bin/python}"
if [[ ! -x "$bootstrap" ]]; then
  echo "Create the CPU check environment first: uv sync --frozen --project \"$own\" --python 3.12" >&2
  exit 1
fi
runtime="$($bootstrap "$own/scripts/run_stage.py" python)"
exec "$runtime" "$own/scripts/run_stage.py" "$@"
