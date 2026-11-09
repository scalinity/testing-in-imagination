"""Approval requests and their binding (SPEC §2).

An approval binds the protocol, addendum and config digests, the code commit, the job list and
a GPU-hour budget. Daniel commits approvals/<stage>.json; a GPU stage refuses to start unless it
is committed, unmodified and matches every bound field, and the code is unchanged since.
"""
import hashlib
import json
from pathlib import Path
import subprocess

from tim.events import TimError
from tim.paths import OWN, ROOT

APPROVER = "Daniel Escalante"
FOLDER = OWN.relative_to(ROOT)
SMOKE_CAP_SECONDS = 0.5 * 3600
CODE_PATHS = [str(FOLDER / p) for p in ("tim", "scripts", "configs", "schemas", "pyproject.toml", "uv.lock")] + \
    ["replication-of-original/daniel-escalante/replication", "lacdm/upstream"]
ADDENDA = {"test-a": "addendum-a.json", "test-a2": "addendum-a.json", "test-b": "addendum-b.json"}
LISTED = {"smoke": 2, "bench-a": 24, "bench-b": 24}
SPLIT_SIZE = {"val": 240, "test": 240}
# Provisional GPU-hours per job from SPEC §9 until bench measurements exist.
PROVISIONAL = {"smoke": 0.25, "bench-a": 0.075, "val-a": 0.55, "test-a": 1.7 / 3,
               "bench-b": 0.075, "val-b": 1.7 / 3, "test-b": 7.7 / 14}
BENCH_PROXY = {"V0": "V3", "V0-rerun": "V3", "V2": "V3", "V3": "V3", "V0-shadow": "V0-shadow"}


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git(root, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def jobs(stage) -> list[dict]:
    from tim.runner import STAGE_SPECS
    spec = STAGE_SPECS[stage]
    patients = LISTED.get(stage, SPLIT_SIZE[spec["split"]])
    return [{"stage": stage, "variant": v, "arm": "primary", "split": spec["split"], "patients": patients}
            for v in spec["jobs"]]


def binding(settings, stage, root=ROOT) -> dict:
    addendum, folder = ADDENDA.get(stage), root / FOLDER
    return {
        "stage": stage,
        "protocol_sha256": _sha(folder / "prereg" / "protocol.json"),
        "addendum_sha256": _sha(folder / "prereg" / addendum) if addendum else None,
        "config_sha256": _sha(folder / "configs" / "variants.yaml"),
        "code_commit": _git(root, "rev-parse", "HEAD").stdout.strip(),
        "jobs": jobs(stage),
    }


def estimate_hours(stage, job, root=ROOT) -> float:
    """Setup plus per-patient time measured by bench-a; the SPEC's provisional figure before it."""
    results = root / FOLDER / "results" / "credentialed"
    phase = stage[-1]
    proxy = BENCH_PROXY.get(job["variant"], "V3") if phase == "a" else (
        "B-oracle" if job["variant"] == "B-oracle" else "B-learned-s0")
    bench = results / f"bench-{phase}" / f"bench.{proxy}.json"
    smoke = results / "smoke" / "events" / "V3.jsonl"
    if stage == "bench-a" and smoke.exists():
        # Upper bound: smoke's seconds per patient include model loading, scaled to the bench patients.
        end = [json.loads(line) for line in smoke.read_text().splitlines()][-1]
        return end["seconds"] / end["patients"] * job["patients"] / 3600
    if stage == "bench-b" and (results / "bench-a" / "bench.V3.json").exists():
        # The planner scores every test turn; twice V3's bench-a per-patient time is the upper bound.
        m = json.loads((results / "bench-a" / "bench.V3.json").read_text())
        per_patient = (m["seconds"] - m["setup_seconds"]) / max(m["patients"], 1)
        return (m["setup_seconds"] + 2 * per_patient * job["patients"]) / 3600
    if not stage.startswith("bench") and bench.exists():
        m = json.loads(bench.read_text())
        per_patient = (m["seconds"] - m["setup_seconds"]) / max(m["patients"], 1)
        return (m["setup_seconds"] + per_patient * job["patients"]) / 3600
    return PROVISIONAL[stage]


def request(settings, stage, root=ROOT, code_paths=CODE_PATHS) -> Path:
    bound = binding(settings, stage, root)
    hours = {job["variant"]: round(estimate_hours(stage, job, root), 3) for job in bound["jobs"]}
    accounts = [a for a in settings_accounts() if a]
    body = {**bound, "gpu_hours": hours, "gpu_hours_total": round(sum(hours.values()), 3),
            "expected_wall_hours": round(max(hours.values()), 3),
            "queue_notes": (f"{len(bound['jobs'])} one-GPU jobs on hpg-rtx6000 (RTX PRO 6000), submitted together and "
                            f"spread across {', '.join(accounts) or 'the course accounts'}; each course QOS caps "
                            "8 GPUs and 250 GB shared by its whole class, so jobs may wait for free GPUs.")}
    path = root / FOLDER / "approvals" / f"{stage}.request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


def settings_accounts() -> list[str]:
    import os
    return os.environ.get("SLURM_ACCOUNTS", "").split(",")


def _sacct_seconds(job):
    """Scheduler-reported elapsed seconds of a job, or None when unavailable."""
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", str(job), "--format=ElapsedRaw"],
                         capture_output=True, text=True) if str(job).isdecimal() else None
    try:
        return float(out.stdout.split()[0])
    except (AttributeError, IndexError, ValueError):
        return None


def smoke_used_seconds(settings, sacct=_sacct_seconds) -> float:
    """Smoke GPU time so far. An attempt that never recorded its time (killed, timed out) is
    charged its scheduler-reported elapsed time, or the whole cap when that is unavailable."""
    from tim.access import Ledger
    entries = Ledger(settings.admission_dir).entries()
    recorded = [e for e in entries if e["event"] == "smoke_seconds"]
    started = {e["job"] for e in entries if e["event"] == "smoke_started"}
    used = sum(e["seconds"] for e in recorded)
    for job in started - {e.get("job") for e in recorded}:
        elapsed = sacct(job)
        used += SMOKE_CAP_SECONDS if elapsed is None else elapsed
    return used


def _validation_stop(root) -> None:
    """SPEC §5.3 and §5.5: no test access unless the premise held and the shadow control passed
    (or a committed amendment explains its mismatch)."""
    addendum = json.loads((root / FOLDER / "prereg" / "addendum-a.json").read_text())
    shadow, amendment = addendum["shadow_control"], addendum["shadow_control"].get("amendment")
    explained = amendment and (root / FOLDER / "prereg" / "amendments" / amendment).is_file()
    if not addendum["premise"]["passed"] or not (shadow["passed"] or explained):
        raise TimError("validation_stop", "addendum A blocks test access")


def check(settings, stage, root=ROOT, code_paths=CODE_PATHS) -> None:
    if stage == "smoke" and smoke_used_seconds(settings) < SMOKE_CAP_SECONDS:
        return
    path = root / FOLDER / "approvals" / f"{stage}.json"
    relative = str(path.relative_to(root))
    tracked = _git(root, "ls-files", "--error-unmatch", relative).returncode == 0
    if not path.exists() or not tracked or _git(root, "diff", "--quiet", "HEAD", "--", relative).returncode:
        raise TimError("approval_missing", f"commit approvals/{stage}.json before running {stage}")
    approval, expected = json.loads(path.read_text()), binding(settings, stage, root)
    if stage in ADDENDA and expected["addendum_sha256"] is None:
        raise TimError("approval_missing", f"commit prereg/{ADDENDA[stage]} before {stage}")
    for key in ("stage", "protocol_sha256", "addendum_sha256", "config_sha256", "jobs"):
        if approval.get(key) != expected[key]:
            raise TimError("approval_mismatch", f"approval field {key} does not match")
    if approval.get("approved_by") != APPROVER or not isinstance(approval.get("gpu_hours_total"), (int, float)):
        raise TimError("approval_mismatch", "approval is not signed off with a GPU-hour budget")
    commit = approval.get("code_commit", "")
    if (_git(root, "merge-base", "--is-ancestor", commit, "HEAD").returncode
            or _git(root, "diff", "--quiet", commit, "HEAD", "--", *code_paths).returncode
            or _git(root, "diff", "--quiet", "HEAD", "--", *code_paths).returncode):
        raise TimError("approval_mismatch", "code changed since the approved commit")
    if stage in ("test-a", "test-a2"):
        _validation_stop(root)
