"""One generated sbatch script per submission, and the sbatch call.

Follows the project's existing submissions (cdm-extension/daniel-escalante/scripts/submit_slurm.py,
gpu.slurm, cpu.slurm): GPU jobs take one RTX PRO 6000 on hpg-rtx6000 and export
CUBLAS_WORKSPACE_CONFIG; CPU jobs name only an account and a QOS; the script ends in `exec`, so the
batch shell becomes the stage process and Slurm's warning signal reaches it directly.
Standard library only and Python 3.9 compatible.
"""
from __future__ import annotations

import shlex
import subprocess

from . import authorization
from .util import write_text


def wall(hours):
    minutes = int(round(hours * 60))
    return "%02d:%02d:00" % (minutes // 60, minutes % 60)


def script(paths, stage, pool, mem_gb, python):
    name = "tii-" + stage["id"]
    resources = stage["resources"]
    lines = ["#!/bin/bash",
             "#SBATCH --job-name=%s" % name,
             "#SBATCH --account=%s" % pool.account,
             "#SBATCH --qos=%s" % pool.qos]
    if pool.kind == "gpu":
        lines += ["#SBATCH --partition=%s" % authorization.GPU_PARTITION,
                  "#SBATCH --gres=%s" % authorization.GPU_GRES]
    lines += ["#SBATCH --nodes=1",
              "#SBATCH --ntasks=1",
              "#SBATCH --cpus-per-task=%d" % resources["cpus"],
              "#SBATCH --mem=%dG" % mem_gb,
              "#SBATCH --time=%s" % wall(resources["time_hours"]),
              "#SBATCH --signal=B:USR1@600",   # ten minutes before the limit, to the batch shell
              "#SBATCH --no-requeue",          # the watcher alone decides on resubmission
              "#SBATCH --output=%s/%s-%%j.log" % (paths.logs, name),
              "export TII_ROOT=%s TII_CODE=%s" % (shlex.quote(str(paths.root)), shlex.quote(str(paths.code))),
              "export PYTHONDONTWRITEBYTECODE=1"]
    if pool.kind == "gpu":
        lines.append("export CUBLAS_WORKSPACE_CONFIG=:4096:8")
    lines += ["cd %s || exit 1" % shlex.quote(str(paths.package)),
              "exec %s -m tiicee.pipeline.stage %s" % (shlex.quote(python), stage["id"])]
    return "\n".join(lines) + "\n"


def submit(paths, stage, pool, mem_gb, python, attempt):
    """Write the script and submit it. Returns (job id or None, sbatch's return code)."""
    paths.logs.mkdir(parents=True, exist_ok=True)
    path = write_text(paths.scripts / ("tii-%s.%d.sbatch" % (stage["id"], attempt)),
                      script(paths, stage, pool, mem_gb, python))
    try:
        result = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None, 127
    job_id = result.stdout.strip().split(";")[0]
    if result.returncode != 0 or not job_id.isdecimal():
        return None, result.returncode or 1
    return job_id, 0
