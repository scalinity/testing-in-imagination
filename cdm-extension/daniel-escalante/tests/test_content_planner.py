"""The content planner inside the guarded environment (SPEC_CONTENT_WM.md §2.2-2.4): it acts only on
turns where V3 would request a lab panel, replaces that request with its choice, and leaves imaging,
the physical exam and every guard unchanged."""
import pytest

from tim import access, fixture, runner
from tim.settings import Settings


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return fixture.build(tmp_path_factory.mktemp("fx") / "baseline")


@pytest.fixture
def settings(environ, baseline, tmp_path):
    return Settings.load(env_file=tmp_path / "none.env", environ={**environ, "CDM_WORK_DIR": str(baseline)})


class Stub:
    """Always picks the last untried lab and records what it was shown."""

    def __init__(self, calls):
        self.calls = calls

    def choose(self, untried, provided, t):
        self.calls.append((t, tuple(untried), tuple(provided)))
        return untried[-1]


def evaluate(settings, variant, predictor=None):
    dataset = access.open_split(settings, "val-a", "val")
    evaluator = runner.make_evaluator(variant, dataset, lambda cfg, tok: fixture.MockScorer(tok), None,
                                      predictor=predictor)
    return evaluator(runner.split_cfg(settings, "val"), None, fixture.ScriptedPolicy(), fixture_generation())


def fixture_generation():
    from replication.shim import generation_settings
    return generation_settings(False)


def by_kind(rows):
    return {fixture.kind_of(i): row for i, row in enumerate(rows) if i < len(fixture.KINDS)}


ARM = runner.Variant("WM-test", g1=True, g2=True, g3=True, planner={"kind": "content", "model": "learned"})


def test_a_lab_request_is_replaced_by_the_planners_choice(settings):
    calls = []
    rows = by_kind(evaluate(settings, ARM, predictor=lambda item: Stub(calls)))
    exhaust = rows["exhaust"]  # proposes PE, CT, MRI, XR, US, then CBC at turn 5
    assert exhaust["requested_tests"][:6] == ["physical examination", "ct", "mri", "radiograph", "ultrasound",
                                              "electrolyte panel"]
    event = next(e for e in exhaust["events"] if e["type"] == "eig_rerank")
    assert (event["t"], event["from_kind"], event["to_kind"]) == (5, "test", "test")
    # Exhaust is fixture record 7: CT is charted, the exam is charted (7 % 5 != 4), ultrasound is not (7 % 3 != 0).
    _, untried, provided = next(c for c in calls if c[0] == 5)
    assert untried == fixture.LAB_TESTS and provided == ("Physical Examination", "CT")


def test_imaging_and_exam_requests_pass_through_unchanged(settings):
    calls = []
    v3 = by_kind(evaluate(settings, runner.VARIANTS["V3"]))
    arm = by_kind(evaluate(settings, ARM, predictor=lambda item: Stub(calls)))
    assert arm["direct"]["requested_tests"] == v3["direct"]["requested_tests"]  # CT, then the exam
    assert not [e for e in arm["direct"]["events"] if e["type"] == "eig_rerank"]
    assert all(t >= 5 for t, _, _ in calls)  # only lab proposals reach the planner


def test_choosing_the_proposed_panel_records_no_event(settings):
    class Same:
        def choose(self, untried, provided, t):
            return untried[0]  # the configured order puts the proposed CBC first
    exhaust = by_kind(evaluate(settings, ARM, predictor=lambda item: Same()))["exhaust"]
    assert exhaust["requested_tests"][5] == "complete blood count"
    assert not [e for e in exhaust["events"] if e["type"] == "eig_rerank" and e["t"] == 5]


def test_the_content_study_arms_and_stages_are_configured():
    arms = {name: v.planner for name, v in runner.VARIANTS.items() if v.planner and v.planner["kind"] == "content"}
    assert arms == {"WM-learned": {"kind": "content", "model": "learned"},
                    "WM-oracle": {"kind": "content", "model": "oracle"},
                    "WM-prior": {"kind": "content", "model": "prior"},
                    "WM-mislead-a0.5": {"kind": "content", "model": "mislead", "alpha": 0.5},
                    "WM-mislead-a1": {"kind": "content", "model": "mislead", "alpha": 1}}
    assert runner.STAGE_SPECS["bench-c"] == {"split": "val", "jobs": ["WM-learned", "WM-oracle"]}
    assert runner.STAGE_SPECS["val-c"] == {"split": "val", "jobs": list(arms)}
    assert "test-c" not in runner.STAGE_SPECS  # test access waits for the frozen protocol (§6)
