#!/bin/bash
# watcher_cycle.sh -- one watcher cycle for study TII-CEE-1. Installed as <TII>/pipeline/watcher_cycle.sh
# and started every 15 minutes by the tagged scrontab entry (scrontab_entry.txt).
#
# Settings come from watcher.env beside this script (see watcher.env.example): TII_ROOT, TII_CODE,
# TII_PYTHON, and anything the jobs need in their environment. scrontab jobs do not run in a login
# shell, so nothing is assumed from a profile. All recurrences share one job ID and one log file,
# so this script writes its own timestamped lines.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/watcher.env" ]; then
    set -a; . "$HERE/watcher.env"; set +a
fi
: "${TII_ROOT:?set TII_ROOT in watcher.env}" "${TII_CODE:?set TII_CODE in watcher.env}"
export TII_ROOT TII_CODE
WATCHER_PYTHON="${TII_WATCHER_PYTHON:-/usr/bin/python3}"   # the compute node's system Python (3.9)

stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
echo "$(stamp) watcher cycle start job=${SLURM_JOB_ID:-none} host=$(hostname -s)"
cd "$TII_CODE/tii-cee" || { echo "$(stamp) code checkout not found"; exit 1; }
PYTHONDONTWRITEBYTECODE=1 "$WATCHER_PYTHON" -m tiicee.pipeline.watcher
rc=$?
echo "$(stamp) watcher cycle end rc=$rc"
exit "$rc"
