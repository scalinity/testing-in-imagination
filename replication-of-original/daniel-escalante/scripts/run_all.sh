#!/usr/bin/env bash
set -euo pipefail
scripts="$(cd "$(dirname "$0")" && pwd)"
for stage in s0 s1 s2 s3 s4 s5 s6; do
  bash "$scripts/stage.sh" "$stage" "$@"
done
