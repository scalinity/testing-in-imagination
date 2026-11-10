import json

import pytest

from tim import approvals, protocol
from tim.events import TimError

FOLDER = approvals.FOLDER


def bundles(root, shadow_passed=True, premise_passed=True):
    results = root / FOLDER / "results" / "credentialed"
    (results / "analyze-val-a").mkdir(parents=True)
    (results / "bench-a").mkdir(parents=True)
    (root / FOLDER / "prereg").mkdir(parents=True)
    (root / FOLDER / "prereg" / "protocol.json").write_text("{}")
    sensitivity = {"V3-V0": {"val_effect": 3.0, "discordance": {"appendicitis": 0.1}, "power_at_val_effect": 0.4,
                             "mde_80": 6.1}}
    (results / "analyze-val-a" / "validation.json").write_text(json.dumps({
        "shadow": {"identical": 240 if shadow_passed else 239, "total": 240, "scorer_errors": 0, "passed": shadow_passed},
        "premise": {"v0_nonanswers": 31 if premise_passed else 4, "minimum": 10, "passed": premise_passed},
        "go": shadow_passed and premise_passed, "sensitivity": sensitivity}))
    for variant in ("V0-shadow", "V3"):
        (results / "bench-a" / f"bench.{variant}.json").write_text(json.dumps({
            "variant": variant, "patients": 24, "seconds": 600.0, "setup_seconds": 120.0, "decision_turns": 90,
            "scoring_calls": 90, "max_abs_fast_minus_reference": 2e-4, "fast_within_tolerance": True,
            "next_token_checks": 90, "next_token_agreement": 0.97}))


def test_addendum_a_composes_the_committed_validation_and_bench_evidence(tmp_path):
    bundles(tmp_path)
    path = protocol.addendum_a(root=tmp_path)
    addendum = json.loads(path.read_text())
    assert addendum["premise"]["passed"] and addendum["shadow_control"]["passed"]
    assert addendum["shadow_control"]["amendment"] is None
    assert addendum["sensitivity"]["V3-V0"]["mde_80"] == 6.1
    assert addendum["bench"]["V3"]["fast_within_tolerance"] is True
    # setup 120 s + (600 - 120) / 24 s per patient x 240 patients = 4920 s per test job
    assert addendum["test_a_budget"]["gpu_hours"] == {"V0-rerun": 1.367, "V2": 1.367, "V3": 1.367}
    assert len(addendum["sources"]) == 3


def test_a_failed_premise_is_recorded_and_blocks_test_access(tmp_path):
    bundles(tmp_path, premise_passed=False)
    protocol.addendum_a(root=tmp_path)
    with pytest.raises(TimError) as caught:
        approvals._validation_stop(tmp_path)
    assert caught.value.code == "validation_stop"


def test_the_addendum_refuses_to_overwrite_a_committed_one(tmp_path):
    bundles(tmp_path)
    protocol.addendum_a(root=tmp_path)
    with pytest.raises(FileExistsError):
        protocol.addendum_a(root=tmp_path)
