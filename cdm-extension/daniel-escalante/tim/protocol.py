"""The frozen protocol (prereg/protocol.json) and the synthetic stand-in constants."""
import hashlib
import json
from pathlib import Path

from tim.enums import DISEASES
from tim.paths import OWN, ROOT

PATH = OWN / "prereg" / "protocol.json"
# The synthetic run exercises the pipeline at fixture scale with fewer resamples; numbers are meaningless.
SYNTHETIC = {"t5_reference": None, "training_majority": "appendicitis",
             "test_class_sizes": {d: 6 for d in DISEASES}, "bootstrap_resamples": 200,
             "sensitivity_simulations": 300, "premise_minimum_nonanswers": 10}


def load(settings) -> tuple[dict, str]:
    """(constants, sha256 of protocol.json); credentialed stages require the committed file."""
    from tim.access import check_admission_path
    from tim.events import TimError
    if not PATH.exists():
        if settings.synthetic:
            return SYNTHETIC, "0" * 64
        raise TimError("approval_missing", "commit prereg/protocol.json before any credentialed stage")
    data = PATH.read_bytes()
    protocol = json.loads(data)
    check_admission_path(settings, protocol)
    constants = SYNTHETIC if settings.synthetic else protocol["analysis"]
    return constants, hashlib.sha256(data).hexdigest()


def addendum_a(root=None) -> Path:
    """Addendum A (SPEC §5.3) from committed bundles only: premise check, shadow control,
    sensitivity, bench measurements and the test-a budget they imply. Written once."""
    from tim import approvals
    root = root or ROOT
    folder = root / approvals.FOLDER
    results = folder / "results" / "credentialed"
    target = folder / "prereg" / "addendum-a.json"
    if target.exists():
        raise FileExistsError("addendum A is already written")
    validation_path = results / "analyze-val-a" / "validation.json"
    bench_paths = {v: results / "bench-a" / f"bench.{v}.json" for v in ("V0-shadow", "V3")}
    validation = json.loads(validation_path.read_text())
    hours = {job["variant"]: round(approvals.estimate_hours("test-a", job, root), 3)
             for job in approvals.jobs("test-a")}

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    addendum = {
        "addendum": "A",
        "protocol_sha256": sha(folder / "prereg" / "protocol.json"),
        "premise": validation["premise"],
        "shadow_control": {**validation["shadow"], "amendment": None},
        "go": validation["go"],
        "sensitivity": validation["sensitivity"],
        "bench": {variant: json.loads(path.read_text()) for variant, path in bench_paths.items()},
        "test_a_budget": {"gpu_hours": hours, "gpu_hours_total": round(sum(hours.values()), 3)},
        "sources": {str(p.relative_to(folder)): sha(p) for p in (validation_path, *bench_paths.values())},
    }
    target.write_text(json.dumps(addendum, indent=2, sort_keys=True) + "\n")
    return target
