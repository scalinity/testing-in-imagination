#!/usr/bin/env bash
set -euo pipefail
own="$(cd "$(dirname "$0")/.." && pwd)"
export VLLM_USE_V1=0
bootstrap="${CDM_PYTHON:-$own/.venv/bin/python}"
if [[ ! -x "$bootstrap" ]]; then
  echo "Create the CPU check environment first: uv sync --frozen --project \"$own\" --python 3.12" >&2
  exit 1
fi
env_args=()
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == "--env-file" ]]; then
    env_args+=("--env-file" "${args[i+1]:?--env-file requires a path}")
  elif [[ "${args[i]}" == --env-file=* ]]; then
    env_args+=("${args[i]}")
  fi
done
if ((${#env_args[@]})); then
  runtime="$("$bootstrap" "$own/scripts/run_stage.py" python "${env_args[@]}")"
else
  runtime="$("$bootstrap" "$own/scripts/run_stage.py" python)"
fi
exec "$runtime" "$own/scripts/run_stage.py" "$@"
