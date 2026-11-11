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
GPU_STAGES = ("smoke", "bench-a", "val-a", "test-a", "bench-b", "val-b", "test-b", "dose-val-b")
CPU_STAGES = ("validate", "analyze-val-a", "analyze-a", "hpi-audit", "wm-fit", "inventory", "leakage",
              "analyze-val-b", "analyze-b", "diagnose-val-b")
SYNTHETIC_ORDER = ("validate", "smoke", "bench-a", "val-a", "analyze-val-a", "test-a", "analyze-a",
                   "hpi-audit", "wm-fit", "inventory", "leakage", "bench-b", "val-b", "analyze-val-b",
                   "diagnose-val-b", "dose-val-b", "test-b", "analyze-b")
WM_FILES = ("learned-s0.pkl", "learned-s1.pkl", "learned-s2.pkl", "base_rates.json")


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
    from tim.runner import STAGE_SPECS
    split = STAGE_SPECS[run_stage_name]["split"]
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

    from tim.enums import CONFIGURED_TESTS, TEST_CODES
    tests = list(CONFIGURED_TESTS)

    def coded(value):
        """Replace test names used as keys with their public codes."""
        if isinstance(value, dict):
            return {TEST_CODES.get(k, k): coded(v) for k, v in value.items()}
        return value

    def items(settings, split):
        dataset = access.open_split(settings, stage, split)
        return [dict(dataset[i]) for i in range(len(dataset))]

    def hpi_audit(settings, bundle):
        from tim import audit
        bundle.aggregate("hpi_audit", coded(audit.hpi_audit({s: items(settings, s) for s in ("train", "val")}, tests)))

    def wm_fit(settings, bundle):
        import hashlib
        import pickle
        from tim import availability as av
        train, val = items(settings, "train"), items(settings, "val")
        folder = runner.wm_dir(settings)
        folder.mkdir(parents=True, exist_ok=True)
        c = av.choose_c(train, tests)
        for seed in (0, 1, 2):
            (folder / f"learned-s{seed}.pkl").write_bytes(pickle.dumps(av.fit_learned(train, tests, seed, c)))
        (folder / "base_rates.json").write_text(json.dumps(av.base_rates(train, tests), sort_keys=True))
        files = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in WM_FILES}
        bundle.aggregate("wm_quality", {"c": c, "files": files,
                                        "quality": {"val": analysis.arm_quality(settings, val, "val")}})

    def inventory_stage(settings, bundle):
        from tim import inventory
        source = "synthetic" if settings.synthetic else "mimic_real"
        by_split = {s: inventory.records(items(settings, s), tests, s, source) for s in ("train", "val")}
        folder = settings.work_dir / "private" / "inventory"
        folder.mkdir(parents=True, exist_ok=True)
        for split, rows in by_split.items():
            (folder / f"{split}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        bundle.aggregate("inventory_card", coded(inventory.card(by_split, tests)))

    def leakage_stage(settings, bundle):
        import pickle
        from tim import leakage
        fitted = leakage.fit_probes(items(settings, "train"), tests)
        runner.wm_dir(settings).mkdir(parents=True, exist_ok=True)
        (runner.wm_dir(settings) / "leakage.pkl").write_bytes(pickle.dumps(fitted))
        bundle.aggregate("leakage", {"val": leakage.evaluate(fitted, items(settings, "val"), tests)})

    def analyze_val_b(settings, bundle):
        from tim.runner import STAGE_SPECS
        rows = {arm: _rows(settings, protocol_hash, "val-b", arm, stage) for arm in STAGE_SPECS["val-b"]["jobs"]}
        p0 = _rows(settings, protocol_hash, "val-a", "V3", stage)
        bundle.aggregate("validation_b", analysis.validation_b(rows, p0, access.baseline_cfg(settings), settings.source,
                                                               resamples=constants["bootstrap_resamples"]))

    def diagnose_val_b(settings, bundle):
        from tim import diagnostics
        from tim.runner import STAGE_SPECS
        rows = {arm: _rows(settings, protocol_hash, "val-b", arm, stage) for arm in STAGE_SPECS["val-b"]["jobs"]}
        p0 = _rows(settings, protocol_hash, "val-a", "V3", stage)
        relevant = diagnostics.relevance(access.open_split(settings, stage, "val").lab_test_mapping)
        bundle.aggregate("diagnostics", diagnostics.diagnostics(
            rows, p0, access.baseline_cfg(settings), settings.source, relevant,
            resamples=constants["bootstrap_resamples"]))
        bundle.aggregate("hypothesis_check", diagnostics.hypothesis_check(rows, p0, relevant, settings.source))

    def analyze_b(settings, bundle):
        import pickle
        from tim import leakage
        from tim.runner import STAGE_SPECS
        rows = {arm: _rows(settings, protocol_hash, "test-b", arm, stage) for arm in STAGE_SPECS["test-b"]["jobs"]}
        p0 = _rows(settings, protocol_hash, "test-a", "V3", stage)
        test = items(settings, "test")
        staged = settings.work_dir / "staging"
        wm = json.loads((staged / "wm-fit" / "wm_quality.json").read_text())
        quality = {**wm["quality"], "test": analysis.arm_quality(settings, test, "test")}
        fitted = pickle.loads((runner.wm_dir(settings) / "leakage.pkl").read_bytes())
        leak = {**json.loads((staged / "leakage" / "leakage.json").read_text()),
                "test": leakage.evaluate(fitted, test, tests)}
        bundle.aggregate("metrics", analysis.metrics_b(rows, p0, access.baseline_cfg(settings), settings.source,
                                                       quality, leak, resamples=constants["bootstrap_resamples"]))
        bundle.aggregate("wm_quality", {**wm, "quality": quality})
        bundle.aggregate("leakage", leak)
        for name, source_stage in (("inventory_card", "inventory"), ("hpi_audit", "hpi-audit")):
            bundle.aggregate(name, json.loads((staged / source_stage / f"{name}.json").read_text()))

    def gpu_job(settings, bundle):
        return runner.run_job(settings, stage, job, bundle, protocol_hash, commit(), _clock(settings))

    return {"validate": validate, "analyze-val-a": analyze_val_a, "analyze-a": analyze_a, "hpi-audit": hpi_audit,
            "wm-fit": wm_fit, "inventory": inventory_stage, "leakage": leakage_stage,
            "analyze-val-b": analyze_val_b, "diagnose-val-b": diagnose_val_b,
            "analyze-b": analyze_b}.get(stage, gpu_job)


def run_command(settings, stage, job=None):
    from tim import protocol
    from tim.runner import STAGE_SPECS
    constants, protocol_hash = protocol.load(settings)
    start = {} if settings.synthetic else {"commit": commit()[:12], "protocol": protocol_hash[:16]}
    if stage in CPU_STAGES:
        return [run_stage(settings, stage, "main", stage_body(stage, None, constants, protocol_hash), **start)]
    jobs = STAGE_SPECS[stage]["jobs"]
    if not settings.synthetic:
        from tim import approvals
        if not os.environ.get("SLURM_JOB_ID"):
            raise TimError("access_denied", "GPU stages run inside SLURM jobs: use scripts/submit_slurm.py")
        if job not in jobs:
            raise TimError("approval_mismatch", "unknown job for this stage")
        approvals.check(settings, stage)
        jobs = [job]
    return [run_stage(settings, stage, j, stage_body(stage, j, constants, protocol_hash),
                      variant=j, split=STAGE_SPECS[stage]["split"], **start) for j in jobs]


def results_root() -> Path:
    return Path(os.environ.get("TIM_RESULTS_DIR", OWN / "results"))


def _disclosure() -> str:
    from tim import protocol
    return (json.loads(protocol.PATH.read_text())["disclosure"]["test_set_status"]
            if protocol.PATH.exists() else "No protocol is committed for this synthetic run.")


def render_phase(metrics_path: Path) -> Path:
    from tim import render
    disclosure = _disclosure()
    metrics = json.loads(metrics_path.read_text())
    path = metrics_path.parent / "REPORT.md"
    path.write_text(render.render_b(metrics, disclosure) if "family_b" in metrics else render.render(metrics, disclosure))
    return path


def render_summary(kind_dir: Path) -> list[Path]:
    """WRITEUP.md and SLIDE.md beside the phase folders, from both phases' metrics.json (SPEC §7, M9)."""
    from tim import render
    from tim.paths import REPLICATION
    a, b = (json.loads((kind_dir / phase / "metrics.json").read_text()) for phase in ("phase-a", "phase-b"))
    paper = json.loads((REPLICATION / "results" / "paper_values.json").read_text())
    writeup, slide = kind_dir / "WRITEUP.md", kind_dir / "SLIDE.md"
    writeup.write_text(render.render_writeup(a, b, paper, _disclosure()))
    slide.write_text(render.render_slide(a, b, paper))
    return [writeup, slide]


def render_diagnostics(kind_dir: Path) -> Path:
    """DIAGNOSTICS.md beside the phase folders (amendments A4, A5): the bundle, analyze-val-b's summaries, and
    the threshold and per-arm GPU budget committed in addenda A and B."""
    from tim import render
    from tim.diagnostics import DOSE_RESPONSE_ARMS
    diagnostics = json.loads((kind_dir / "diagnose-val-b" / "diagnostics.json").read_text())
    hypothesis = json.loads((kind_dir / "diagnose-val-b" / "hypothesis_check.json").read_text())
    val_b = json.loads((kind_dir / "analyze-val-b" / "validation_b.json").read_text())
    threshold = json.loads((OWN / "prereg" / "addendum-a.json").read_text())["sensitivity"]["V1-V0"]["mde_80"]
    budget = json.loads((OWN / "prereg" / "addendum-b.json").read_text())["test_b_budget"]["gpu_hours"]
    path = kind_dir / "DIAGNOSTICS.md"
    path.write_text(render.render_diagnostics(diagnostics, val_b, threshold,
                                              sum(budget[arm] for arm in DOSE_RESPONSE_ARMS), hypothesis))
    return path


def synthetic_all(settings):
    """Every Phase-A and Phase-B stage on a fresh fixture, published under results/synthetic/."""
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
    for phase in ("phase-a", "phase-b"):
        render_phase(target / phase / "metrics.json")
    render_summary(target)
    render_diagnostics(target)


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
        for phase in ("phase-a", "phase-b"):
            metrics = results_root() / "credentialed" / phase / "metrics.json"
            if metrics.exists():
                print(f"rendered: {render_phase(metrics)}")
        if all((results_root() / "credentialed" / phase / "metrics.json").exists() for phase in ("phase-a", "phase-b")):
            for path in render_summary(results_root() / "credentialed"):
                print(f"rendered: {path}")
        if (results_root() / "credentialed" / "diagnose-val-b" / "diagnostics.json").exists():
            print(f"rendered: {render_diagnostics(results_root() / 'credentialed')}")
    elif args.command in ("addendum-a", "addendum-b"):
        from tim import protocol
        write = protocol.addendum_a if args.command == "addendum-a" else protocol.addendum_b
        print(f"wrote {write().relative_to(ROOT)}")
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
