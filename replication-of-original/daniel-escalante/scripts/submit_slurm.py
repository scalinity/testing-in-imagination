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
parser.add_argument("--parallel", action="store_true",
                    help="shard S1 summaries over GPUs and give every training and evaluation arm its own GPU")
parser.add_argument("--summary-shards", type=int, default=8)
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
qos = site.get("SLURM_QOS")
gres = site.get("SLURM_GRES", "gpu:a100:2")


def submit(name, stage_args, after=(), gpu=False, time="01:00:00", mem="16G", cpus=4):
    command = ["sbatch", "--parsable", f"--account={account}", f"--chdir={OWN}", f"--job-name=cdm-{name}",
               f"--output={logs}/{name}-%j.log", f"--error={logs}/{name}-%j.err",
               f"--time={time}", f"--mem={mem}", f"--cpus-per-task={cpus}"]
    if qos:
        command.append(f"--qos={qos}")
    if after:
        command.append("--dependency=afterok:" + ":".join(after))
    if gpu:
        command.extend([f"--partition={partition}", f"--gres={gres}"])
    command.append("--wrap=" + shlex.join(["bash", str(OWN / "scripts/stage.sh"), *stage_args]))
    if args.dry_run:
        print(shlex.join(command))
        return f"JOB_{name}"
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError("scheduler did not return a numeric job id")
    print(f"{name}: submitted job {job}")
    return job


if args.parallel:
    # Sized for a class allocation of 250 GB and 32 CPUs: eight shards fit at 2 CPUs/28 GB, and S3 plus
    # four training arms fit at 4 CPUs/40 GB. Finalizing stages only verify per-arm results on CPU.
    from replication.config import ARMS
    shards = [submit(f"summary-{i}", ["summarize-shard", "--shard", str(i), "--shards", str(args.summary_shards)],
                     gpu=True, time="12:00:00", mem="28G", cpus=2) for i in range(args.summary_shards)]
    s1 = submit("s1", ["s1"], shards, gpu=True, time="06:00:00", mem="40G")
    s2 = submit("s2", ["s2"], [s1])
    s3 = submit("s3", ["s3"], [s2], gpu=True, time="12:00:00", mem="40G")
    trained = [submit(f"train-{arm}", ["train-arm", "--arm", arm], [s2], gpu=True, time="6-00:00:00", mem="40G")
               for arm in ARMS]
    s4 = submit("s4", ["s4"], [s3, *trained])
    evaluated = [submit(f"evaluate-{arm}", ["evaluate-arm", "--arm", arm], [s4], gpu=True, time="12:00:00", mem="40G")
                 for arm in ARMS]
    s5 = submit("s5", ["s5"], evaluated)
    submit("s6", ["s6"], [s5], time="02:00:00")
    raise SystemExit(0)

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
