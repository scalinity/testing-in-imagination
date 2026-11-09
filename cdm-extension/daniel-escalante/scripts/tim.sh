#!/usr/bin/env bash
# Entry point for every tim command. A command that may touch patient data runs with stdout
# and stderr redirected to a private log before Python starts (SPEC §1.3); the terminal sees
# only the exit status and where the log is.
set -euo pipefail
own="$(cd "$(dirname "$0")/.." && pwd)"
command="${1:?usage: scripts/tim.sh <command> [arguments]}"

# Literal KEY=value settings from the untracked .env; the environment takes precedence.
if [[ -f "$own/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*(#.*)?$ ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    [[ "$key" =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "invalid line in .env" >&2; exit 2; }
    value="${value%\"}"
    value="${value#\"}"
    [[ -z "${!key+x}" ]] && export "$key=$value"
  done < "$own/.env"
fi

venv="$own/.venv/bin/python"
runtime="${CDM_WORK_DIR:-}/runtime-env/bin/python"
python="${TIM_PYTHON:-$venv}"
if [[ -z "${TIM_PYTHON:-}" && "${CDM_DATA_SOURCE:-synthetic}" == mimic && -x "$runtime" ]]; then
  python="$runtime"
fi
export PYTHONDONTWRITEBYTECODE=1 VLLM_USE_V1=0

case "$command" in
  test) exec "$venv" -m pytest "$own/tests" -q "${@:2}" ;;
  check-git|approval-request|render|addendum-a) exec "$python" "$own/scripts/run_tim.py" "$@" ;;
esac

: "${TIM_WORK_DIR:?set TIM_WORK_DIR in the untracked .env}"
logs="$TIM_WORK_DIR/private-logs"
mkdir -p "$logs"
chmod 700 "$logs"
log="$logs/${SLURM_JOB_ID:-local}-$command-$(date +%Y%m%dT%H%M%S)-$$.log"
set +e
(umask 077; exec >"$log" 2>&1; exec "$python" "$own/scripts/run_tim.py" "$@")
status=$?
set -e
echo "$command: exit $status (private log: $log)"
exit "$status"
