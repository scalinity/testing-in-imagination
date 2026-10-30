#!/usr/bin/env python3
"""Submit stage dependencies after local S0, without embedding private site settings."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

OWN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OWN))
from replication.config import Settings, read_env

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
settings = Settings.load()
site = read_env(OWN / ".env") | dict(os.environ)
account, partition = site.get("SLURM_ACCOUNT"), site.get("SLURM_PARTITION", "gpu")
if not account:
    parser.error("set SLURM_ACCOUNT in the private .env before submitting")
if settings.mock:
    parser.error("use run_all.sh for CPU synthetic runs; SLURM submission requires mimic mode")

# Log files may include restricted generations from upstream exceptions: keep them private.
logs = settings.work_dir / "slurm"
if not args.dry_run:
    subprocess.run(["bash", str(OWN / "scripts/stage.sh"), "s0"], check=True)
    logs.mkdir(parents=True, exist_ok=True)
previous = None
for stage in range(1, 7):
    command = ["sbatch", "--parsable", f"--account={account}", f"--chdir={OWN}",
               f"--output={logs}/s{stage}-%j.log", f"--error={logs}/s{stage}-%j.err"]
    if previous:
        command.append(f"--dependency=afterok:{previous}")
    if stage in (1, 3, 4, 5):
        command.extend([f"--partition={partition}", f"--gres={site.get('SLURM_GRES', 'gpu:a100:2')}", str(OWN / f"scripts/s{stage}.slurm")])
    else:
        command.extend(["--time=01:00:00", "--mem=16G", "--cpus-per-task=4", "--wrap=" + shlex.join(["bash", str(OWN / "scripts/stage.sh"), f"s{stage}"])])
    if args.dry_run:
        print(shlex.join(command))
        previous = f"JOB_S{stage}"
    else:
        previous = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not previous.isdecimal():
            raise RuntimeError("scheduler did not return a numeric job id")
        print(f"S{stage}: submitted job {previous}")
