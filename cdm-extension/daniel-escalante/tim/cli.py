"""Command dispatch for scripts/run_tim.py; scripts/tim.sh is the only caller."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tim.events import Bundle, TimError, guarded
from tim.paths import OWN, ROOT
from tim.settings import Settings

CANARY = "TIM-CANARY-7f3a"
GPU_STAGES = ("smoke", "bench-a", "val-a", "test-a")
CPU_STAGES = ("validate", "analyze-val-a", "analyze-a")
SYNTHETIC_ORDER = ("validate", "smoke", "bench-a", "val-a", "analyze-val-a", "test-a", "analyze-a")


def run_stage(settings, stage, job, body, **start):
    bundle = Bundle(settings.work_dir, stage, job)
    bundle.emit("job_start", stage=stage, source=settings.source, **start)
    with guarded(bundle):
        end = body(settings, bundle) or {}
    bundle.emit("job_end", status="pass", **end)
    return bundle


def canary_probe(settings, bundle):
    """Leak a marker through every channel; scripts/tim.sh must catch them all (SPEC §1.8)."""
    if not settings.synthetic:
        raise TimError("access_denied", "the canary probe runs on synthetic settings only")
    import pandas as pd
    import traceback
    print(f"multiline output\n{CANARY} print\nend of output")
    print(pd.DataFrame({"history": [f"{CANARY} pandas"]}))
    sys.stderr.write(f"{CANARY} stderr\n")
    subprocess.run([sys.executable, "-c", f"import sys; sys.stderr.write('{CANARY} subprocess\\n')"], check=True)
    try:
        assert False, f"{CANARY} assertion"
    except AssertionError:
        traceback.print_exc()
    bundle.emit("progress", completed=1, total=1)
    raise ValueError(f"{CANARY} exception")


def commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _clock(settings):
    return (lambda: 0.0) if settings.synthetic else time.monotonic


def _rows(settings, protocol_hash, run_stage_name, variant, reader):
    """Private rows of the latest completed run of (stage, variant), read through the access table."""
    from tim import access
    from tim.runner import CONFIG
    split = CONFIG["stages"][run_stage_name]["split"]
    access.mode(reader, split)
    prefix = f"{protocol_hash[:16]}:{run_stage_name}:{variant}:primary"
    keys = [e["key"] for e in access.Ledger(settings.admission_dir).entries()
            if e["event"] == "completed" and (e["key"] == prefix or e["key"].startswith(prefix + ":"))]
    if not keys:
        raise TimError("access_denied", f"no completed {run_stage_name} run for {variant}")
    marker = settings.admission_dir / "markers" / (keys[-1].replace(":", "_") + ".json")
    log = Path(json.loads(marker.read_text())["log"])
    return [json.loads(line)["row"] for line in log.read_text().splitlines()]


def stage_body(stage, job, constants, protocol_hash):
    from tim import access, analysis, runner

    def validate(settings, bundle):
        from replication.manifest import file_digest
        from replication.pipeline import _assert_runtime, _rows_path, prepared
        baseline = settings.baseline()
        _assert_runtime(baseline)
        digests = {name: file_digest(prepared(baseline) / name)
                   for name in ("train.csv", "val.csv", "test.csv", "lab_test_mapping.csv")}
        digests["official_rows"] = file_digest(_rows_path(baseline, "parity", 269, "primary", "test"))
        block = analysis.summarize(access.official_rows(settings, "validate"), access.baseline_cfg(settings))
        reference = constants["t5_reference"]
        bundle.aggregate("validate", {"source": settings.source, "digests": digests, "baseline": block,
                                      "t5_matches_reference": None if reference is None else block["metrics"]["T5"] == reference})

    def analyze_val_a(settings, bundle):
        rows = {v: _rows(settings, protocol_hash, "val-a", v, "analyze-val-a") for v in ("V0", "V0-shadow", "V2", "V3")}
        bundle.aggregate("validation", analysis.validation(
            rows, access.baseline_cfg(settings), settings.source, constants["test_class_sizes"],
            sims=constants["sensitivity_simulations"], minimum=constants["premise_minimum_nonanswers"],
            resamples=constants["bootstrap_resamples"]))

    def analyze_a(settings, bundle):
        rows = {v: _rows(settings, protocol_hash, "test-a", v, "analyze-a") for v in ("V0-rerun", "V2", "V3")}
        validation = json.loads((settings.work_dir / "staging/analyze-val-a/validation.json").read_text())
        bundle.aggregate("metrics", analysis.metrics(
            rows, access.official_rows(settings, "analyze-a"), access.baseline_cfg(settings), settings.source,
            constants["training_majority"], validation["sensitivity"], constants["t5_reference"],
            resamples=constants["bootstrap_resamples"]))

    def gpu_job(settings, bundle):
        return runner.run_job(settings, stage, job, bundle, protocol_hash, commit(), _clock(settings))

    return {"validate": validate, "analyze-val-a": analyze_val_a, "analyze-a": analyze_a}.get(stage, gpu_job)


def run_command(settings, stage, job=None):
    from tim import protocol
    from tim.runner import CONFIG
    constants, protocol_hash = protocol.load(settings)
    start = {} if settings.synthetic else {"commit": commit()[:12], "protocol": protocol_hash[:16]}
    if stage in CPU_STAGES:
        return [run_stage(settings, stage, "main", stage_body(stage, None, constants, protocol_hash), **start)]
    jobs = CONFIG["stages"][stage]["jobs"]
    if not settings.synthetic:
        from tim import approvals
        if not os.environ.get("SLURM_JOB_ID"):
            raise TimError("access_denied", "GPU stages run inside SLURM jobs: use scripts/submit_slurm.py")
        if job not in jobs:
            raise TimError("approval_mismatch", "unknown job for this stage")
        approvals.check(settings, stage)
        jobs = [job]
    return [run_stage(settings, stage, j, stage_body(stage, j, constants, protocol_hash),
                      variant=j, split=CONFIG["stages"][stage]["split"], **start) for j in jobs]


def results_root() -> Path:
    return Path(os.environ.get("TIM_RESULTS_DIR", OWN / "results"))


def render_phase_a(metrics_path: Path) -> Path:
    from tim import protocol, render
    disclosure = (json.loads(protocol.PATH.read_text())["disclosure"]["test_set_status"]
                  if protocol.PATH.exists() else "No protocol is committed for this synthetic run.")
    path = metrics_path.parent / "REPORT.md"
    path.write_text(render.render(json.loads(metrics_path.read_text()), disclosure))
    return path


def synthetic_all(settings):
    """Every Phase-A stage on a fresh fixture, published under results/synthetic/."""
    from tim import fixture
    from tim.publish import publish
    if not settings.synthetic:
        raise TimError("access_denied", "the synthetic run needs CDM_DATA_SOURCE=synthetic")
    root = settings.work_dir / "synthetic-all"
    if root.exists():
        shutil.rmtree(root)
    run = Settings(source="synthetic", data_dir=root / "data", baseline_dir=root / "baseline",
                   work_dir=root / "work", admission_dir=root / "admissions")
    fixture.build(run.baseline_dir)
    target = results_root() / "synthetic"
    if target.exists():
        shutil.rmtree(target)
    for stage in SYNTHETIC_ORDER:
        run_command(run, stage)
        publish(run, stage, results_root())
    render_phase_a(target / "phase-a" / "metrics.json")


def check_git_command():
    from tim.approvals import FOLDER
    from tim.publish import ATTRIBUTION, check_git
    violations = check_git(ROOT, str(FOLDER))
    messages = subprocess.run(["git", "-C", str(ROOT), "log", "origin/main..HEAD", "--format=%B"],
                              capture_output=True, text=True, check=True).stdout
    if ATTRIBUTION.search(messages):
        violations.append(("commit messages since origin/main", "attribution text"))
    for path, rule in violations:
        print(f"{rule}: {path}")
    if violations:
        raise SystemExit(1)
    print("check-git: PASS")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tim")
    parser.add_argument("command")
    parser.add_argument("target", nargs="?")
    parser.add_argument("--job", default=None)
    args = parser.parse_args(argv)
    settings = Settings.load()
    if args.command == "canary-probe":
        run_stage(settings, "canary-probe", "main", canary_probe)
    elif args.command == "all":
        synthetic_all(settings)
    elif args.command == "fixture":
        from tim import fixture
        if not settings.synthetic:
            raise TimError("access_denied", "the fixture is synthetic only")
        fixture.build(settings.baseline_dir)
    elif args.command == "publish":
        from tim.publish import publish
        print(f"published: {publish(settings, args.target, results_root())}")
    elif args.command == "render":
        print(f"rendered: {render_phase_a(results_root() / 'credentialed' / 'phase-a' / 'metrics.json')}")
    elif args.command == "check-git":
        check_git_command()
    elif args.command == "approval-request":
        from tim import approvals
        path = approvals.request(settings, args.target)
        body = json.loads(path.read_text())
        print(f"drafted {path.relative_to(ROOT)}: {len(body['jobs'])} jobs, {body['gpu_hours_total']} GPU-h, "
              f"about {body['expected_wall_hours']} h of wall time if the jobs start together")
    elif args.command in (*GPU_STAGES, *CPU_STAGES):
        run_command(settings, args.command, args.job)
    else:
        parser.error("unknown command")
