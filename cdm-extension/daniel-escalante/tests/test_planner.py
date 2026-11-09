import json

import pytest

from tim import access, fixture, runner
from tim.settings import Settings


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return fixture.build(tmp_path_factory.mktemp("fx") / "baseline")


@pytest.fixture
def settings(environ, baseline, tmp_path):
    return Settings.load(env_file=tmp_path / "none.env", environ={**environ, "CDM_WORK_DIR": str(baseline)})


def evaluate(settings, variant, predictor=None):
    from replication.shim import generation_settings
    dataset = access.open_split(settings, "val-a", "val")
    evaluator = runner.make_evaluator(variant, dataset, lambda cfg, tok: fixture.MockScorer(tok), None,
                                      predictor=predictor)
    return evaluator(runner.split_cfg(settings, "val"), None, fixture.ScriptedPolicy(), generation_settings(False))


def by_kind(rows):
    return {fixture.kind_of(i): row for i, row in enumerate(rows) if i < len(fixture.KINDS)}


def strip(rows):
    return [{k: v for k, v in r.items() if k != "variant"} for r in rows]


def test_planner_off_is_v3(settings):
    off = runner.Variant("B-off", g1=True, g2=True, g3=True, planner=None)
    assert strip(evaluate(settings, off)) == strip(evaluate(settings, runner.VARIANTS["V3"]))


def test_no_available_test_commits_to_a_diagnosis(settings):
    arm = runner.Variant("B-test", g1=True, g2=True, g3=True, planner={"kind": "uniform"})
    rows = by_kind(evaluate(settings, arm, predictor=lambda item: {t: 0.01 for t in fixture.TESTS}))
    first = rows["direct"]["events"][0]
    assert (first["t"], first["type"], first["from_kind"], first["to_kind"]) == (0, "wm_stop", "test", "diagnosis")
    assert rows["direct"]["termination"] == "diagnosis" and rows["direct"]["requested_tests"] == []


def test_availability_reweights_the_requested_test(settings):
    arm = runner.Variant("B-test", g1=True, g2=True, g3=True, planner={"kind": "uniform"})
    # The mock scorer ties every test, so the most available untried test wins.
    favour_mri = {t: (0.9 if t == "MRI" else 0.2) for t in fixture.TESTS}
    rows = by_kind(evaluate(settings, arm, predictor=lambda item: favour_mri))
    first = rows["direct"]["events"][0]
    assert (first["t"], first["type"], first["to_kind"]) == (0, "wm_rerank", "test")
    assert rows["direct"]["requested_tests"][0] == "mri"


def test_an_unchanged_choice_records_no_event(settings):
    arm = runner.Variant("B-test", g1=True, g2=True, g3=True, planner={"kind": "uniform"})
    favour_ct = {t: (0.9 if t == "CT" else 0.2) for t in fixture.TESTS}  # the direct kind proposes CT first
    direct = by_kind(evaluate(settings, arm, predictor=lambda item: favour_ct))["direct"]
    assert direct["requested_tests"][0] == "ct"
    assert not [e for e in direct["events"] if e["t"] == 0]


def test_phase_b_arms_are_configured():
    arms = {name: v for name, v in runner.VARIANTS.items() if v.planner}
    assert len(arms) == 14
    corruption = [v.planner for v in arms.values() if v.planner["kind"] == "corrupt"]
    assert sorted((c["alpha"], c["temperature"], c["noise_seed"]) for c in corruption) == sorted(
        (a, t, s) for a in (0.1, 0.3) for t in (1, 0.25) for s in (0, 1))
    assert {v.planner["kind"] for v in arms.values()} == {"uniform", "marginal", "learned", "oracle", "corrupt"}
    assert all(v.g1 and v.g2 and v.g3 for v in arms.values())
    assert json.dumps(runner.CONFIG["planner"]) == json.dumps({"theta_stop": 0.05, "beta": 1})
