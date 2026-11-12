"""scripts/submit_slurm.py: a dry run prints a stage's sbatch commands without an approval; a real
submission still refuses without one, and neither path reaches sbatch unapproved."""
import os
from pathlib import Path
import subprocess
import sys

from tim.diagnostics import DOSE_RESPONSE_ARMS
from tim.paths import OWN


def submit(tmp_path, *args):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "squeue").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "sbatch").write_text(f"#!/bin/sh\ntouch {tmp_path / 'sbatch-called'}\necho 1\n")
    for tool in ("squeue", "sbatch"):
        (bin_dir / tool).chmod(0o755)
    dirs = {name: tmp_path / name for name in ("data", "baseline", "work", "admissions")}
    for path in dirs.values():
        path.mkdir(exist_ok=True)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CDM_DATA_SOURCE": "mimic",
           "CDM_DATA_DIR": str(dirs["data"]), "CDM_WORK_DIR": str(dirs["baseline"]),
           "TIM_WORK_DIR": str(dirs["work"]), "TIM_ADMISSION_DIR": str(dirs["admissions"]),
           "SLURM_ACCOUNTS": "cai6734", "SLURM_PARTITION": "hpg-rtx6000", "SLURM_GRES": "gpu:rtx_pro_6000:1",
           "SLURM_CPU_ACCOUNT": "cai6734", "SLURM_CPU_QOS": "cai6734-b"}
    return subprocess.run([sys.executable, str(OWN / "scripts" / "submit_slurm.py"), *args], env=env,
                          capture_output=True, text=True, timeout=120)


def test_a_dry_run_prints_every_job_of_an_unapproved_gpu_stage(tmp_path):
    result = submit(tmp_path, "--stage", "dose-val-b", "--dry-run")
    assert result.returncode == 0, result.stderr[-2000:]
    lines = result.stdout.splitlines()
    assert len(lines) == len(DOSE_RESPONSE_ARMS)
    for line, arm in zip(lines, DOSE_RESPONSE_ARMS):
        assert line.startswith("sbatch ") and "--account=cai6734 --qos=cai6734 " in line
        assert "--gres=gpu:rtx_pro_6000:1" in line and line.endswith(f"scripts/gpu.slurm dose-val-b --job {arm}")
    assert not (tmp_path / "sbatch-called").exists()


def test_a_real_submission_still_refuses_without_an_approval(tmp_path):
    result = submit(tmp_path, "--stage", "dose-val-b")
    assert result.returncode != 0 and "approval_missing" in result.stderr
    assert not (tmp_path / "sbatch-called").exists()
