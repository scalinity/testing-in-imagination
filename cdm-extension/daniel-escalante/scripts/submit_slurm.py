#!/usr/bin/env python3
"""Submit one stage: each GPU job gets one RTX PRO 6000 and the jobs are spread across course accounts.

A GPU stage is refused without a matching committed approval (smoke within its cap excepted).
All SLURM output goes to the private log directory; the terminal sees job IDs only.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

OWN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OWN))

import tim.paths  # noqa: E402,F401
from replication.config import read_env  # noqa: E402
from tim import approvals  # noqa: E402
from tim.cli import CPU_STAGES, GPU_STAGES  # noqa: E402
from tim.runner import STAGE_SPECS  # noqa: E402
from tim.settings import Settings  # noqa: E402


def least_busy_first(accounts):
    """Order course accounts by how many jobs their whole class has queued or running now."""
    def load(account):
        out = subprocess.run(["squeue", "-h", "-A", account, "-o", "%i"], capture_output=True, text=True).stdout
        return len(out.split())
    return sorted(accounts, key=load)


def wall(hours: float) -> str:
    minutes = int(max(60, 2 * hours * 60 + 30))
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=(*GPU_STAGES, *CPU_STAGES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = Settings.load()
    if settings.synthetic:
        parser.error("SLURM submission is for credentialed runs; use scripts/tim.sh all on CPU")
    site = read_env(OWN / ".env") | dict(os.environ)
    logs = settings.work_dir / "private-logs"
    logs.mkdir(parents=True, exist_ok=True)
    logs.chmod(0o700)
    export = f"ALL,TIM_WORK_DIR={settings.work_dir},CDM_WORK_DIR={settings.baseline_dir}"
    if site.get("HF_HOME"):  # the model cache the baseline run used
        export += f",HF_HOME={site['HF_HOME']}"
    common = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", f"--chdir={OWN}", f"--export={export}",
              f"--output={logs}/slurm-%j.out", f"--error={logs}/slurm-%j.err"]
    submissions = []
    if args.stage in GPU_STAGES:
        approvals.check(settings, args.stage)
        approved = OWN / "approvals" / f"{args.stage}.json"
        hours = json.loads(approved.read_text())["gpu_hours"] if approved.exists() else {}
        smoke_left = None
        if args.stage == "smoke" and not approved.exists():
            # Unapproved smoke may not outlast what remains of its 0.5 GPU-hour allowance.
            smoke_left = int(approvals.SMOKE_CAP_SECONDS - approvals.smoke_used_seconds(settings)) // 60
        accounts = least_busy_first([a for a in site["SLURM_ACCOUNTS"].split(",") if a])
        for i, job in enumerate(STAGE_SPECS[args.stage]["jobs"]):
            account = accounts[i % len(accounts)]
            budget = hours.get(job, approvals.PROVISIONAL[args.stage])
            submissions.append((job, account, common + [
                f"--account={account}", f"--qos={account}", f"--partition={site['SLURM_PARTITION']}",
                f"--gres={site['SLURM_GRES']}", "--cpus-per-task=4", "--mem=16G",
                f"--time={wall(budget) if smoke_left is None else f'00:{max(smoke_left, 1):02d}:00'}",
                f"--job-name=tim-{args.stage}-{job}", str(OWN / "scripts" / "gpu.slurm"), args.stage, "--job", job]))
    else:
        account = site["SLURM_CPU_ACCOUNT"]
        submissions.append(("main", account, common + [
            f"--account={account}", f"--qos={site['SLURM_CPU_QOS']}", "--cpus-per-task=4", "--mem=32G",
            "--time=04:00:00", f"--job-name=tim-{args.stage}", str(OWN / "scripts" / "cpu.slurm"), args.stage]))
    for job, account, command in submissions:
        if args.dry_run:
            print(" ".join(command))
            continue
        job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not job_id.isdecimal():
            raise RuntimeError("scheduler did not return a numeric job id")
        print(f"{args.stage} {job}: job {job_id} on {account}")


if __name__ == "__main__":
    main()
