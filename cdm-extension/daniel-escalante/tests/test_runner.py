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


class FailingScorer:
    def score(self, *args, **kwargs):
        raise RuntimeError("scorer exploded")


def evaluate(settings, variant, scorer=None, log=None, policy=None):
    from replication.shim import generation_settings
    dataset = access.open_split(settings, "val-a", "val")
    evaluator = runner.make_evaluator(runner.VARIANTS[variant], dataset,
                                      lambda cfg, tokenizer: scorer or fixture.MockScorer(tokenizer), log)
    return evaluator(runner.split_cfg(settings, "val"), None, policy or fixture.ScriptedPolicy(), generation_settings(False))


def by_kind(rows):
    return {fixture.kind_of(i): row for i, row in enumerate(rows) if i < len(fixture.KINDS)}


def test_v0_rows_equal_the_replication_evaluator(settings):
    from replication.shim import _evaluate_with_model, generation_settings
    dataset = access.open_split(settings, "val-a", "val")
    reference = _evaluate_with_model(runner.split_cfg(settings, "val"), dataset, fixture.ScriptedPolicy(), generation_settings(False))
    reference = json.loads(json.dumps(reference))
    rows = evaluate(settings, "V0")
    assert [{k: row[k] for k in ref} for row, ref in zip(rows, reference)] == reference
    assert all(row["variant"] == "V0" and row["events"] == [] for row in rows)


def test_untriggered_patients_follow_v0_byte_for_byte(settings):
    v0, v3 = evaluate(settings, "V0"), evaluate(settings, "V3")
    untouched = [i for i, row in enumerate(v3) if not row["events"]]
    assert untouched
    base = [k for k in v0[0] if k not in ("variant",)]
    for i in untouched:
        assert {k: v3[i][k] for k in base if k not in ("prediction", "finalized_prediction", "prediction_source")} == \
               {k: v0[i][k] for k in base if k not in ("prediction", "finalized_prediction", "prediction_source")}


def first_event(row):
    event = row["events"][0]
    return event["t"], event["type"], event["from_kind"], event["to_kind"]


def test_decision_table_paths_end_to_end(settings):
    v0, v3 = by_kind(evaluate(settings, "V0")), by_kind(evaluate(settings, "V3"))
    assert v0["malformed"]["termination"] == "malformed" and v3["malformed"]["termination"] == "diagnosis"
    assert first_event(v3["malformed"]) == (0, "g1_replace", "malformed", "diagnosis")
    assert first_event(v3["repeat"])[:3] == (1, "g1_replace", "repeated_test")
    assert v0["repeat"]["termination"] == "step_limit"
    for kind in ("unknown_action", "unknown_test", "unknown_diagnosis"):
        assert first_event(v3[kind])[:3] == (0, "g1_replace", kind)
    assert first_event(v3["exhaust"]) == (12, "g1_replace", "repeated_test", "diagnosis")
    assert first_event(v3["overflow"]) == (0, "g2_commit", "overflow", "diagnosis")
    assert v0["overflow"]["termination"] == "length_completion"
    assert first_event(v3["truncated"])[:3] == (0, "g1_replace", "malformed")
    assert v3["obs_overflow"]["termination"] == "length_observation" and not v3["obs_overflow"]["events"]
    assert v3["obs_overflow"]["prediction_source"] == "hypothesis_fallback"
    assert v3["obs_overflow"]["raw_prediction"] is None
    assert first_event(v3["too_long"]) == (0, "replacement_failed", "overflow", "diagnosis")
    assert v3["too_long"]["termination"] == "length_completion"
    assert v3["too_long"]["replacement_failed_count"] == 1
    assert v3["too_long"]["finalized_prediction"] == v3["too_long"]["label"]


def test_g2_commits_at_the_last_step_without_g1(settings):
    v2 = by_kind(evaluate(settings, "V2"))
    assert first_event(v2["repeat"]) == (13, "g2_commit", "repeated_test", "diagnosis")
    assert v2["repeat"]["termination"] == "diagnosis"


def test_scorer_errors_execute_the_original_completion(settings):
    rows = by_kind(evaluate(settings, "V3", scorer=FailingScorer()))
    assert rows["malformed"]["termination"] == "malformed"
    assert rows["malformed"]["scorer_error_count"] == 1
    assert first_event(rows["malformed"])[1] == "scorer_error"


def test_shadow_scoring_leaves_every_generation_identical(settings):
    from replication.shim import MockTokenizer
    scorer = fixture.MockScorer(MockTokenizer())
    v0, shadow = evaluate(settings, "V0"), evaluate(settings, "V0-shadow", scorer=scorer)
    assert [r["generation_digest"] for r in v0] == [r["generation_digest"] for r in shadow]
    assert scorer.calls == sum(r["n_turns"] for r in shadow if r["termination"] != "step_limit") + \
        sum(r["n_turns"] - 1 for r in shadow if r["termination"] == "step_limit")


class Interrupting(fixture.ScriptedPolicy):
    """Counts patients by their first decision turn and interrupts at the stop_at-th one."""

    MARKER = "Here is some context on the patient:"

    def __init__(self, stop_at):
        super().__init__()
        self.patients, self.stop_at = 0, stop_at

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        for prompt in prompts:
            if self.MARKER in prompt and prompt.split(self.MARKER, 1)[1].count("\nAction: ") == 0:
                self.patients += 1
                if self.patients == self.stop_at:
                    raise KeyboardInterrupt
        return super().generate(prompts, sampling_params, use_tqdm)


def test_resume_continues_at_the_first_incomplete_patient(settings):
    def episodes():
        return access.Episodes(settings, "f" * 64, "val-a", "V3", "val", "c0ffee", fixture.SIZES["val"])
    log = episodes()
    log.open()
    with pytest.raises(KeyboardInterrupt):
        evaluate(settings, "V3", log=log, policy=Interrupting(stop_at=6))
    log.close()
    log = episodes()
    done = log.open()
    assert len(done) == 5
    policy = Interrupting(stop_at=0)
    rest = evaluate(settings, "V3", log=log, policy=policy)
    assert policy.patients == fixture.SIZES["val"] - 5
    assert done + rest == evaluate(settings, "V3")


class CalibrationSensitive(fixture.ScriptedPolicy):
    """Confidence-calibration answers depend on the prompt, as a real model's would."""

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        outputs = super().generate(prompts, sampling_params, use_tqdm)
        for prompt, output in zip(prompts, outputs):
            if prompt.endswith("Confidence: "):
                output.outputs[0].token_ids = [len(prompt) % 97]
        return outputs


def test_generation_digest_covers_only_hypothesis_and_decision_generations(settings):
    plain = evaluate(settings, "V0")
    sensitive = evaluate(settings, "V0", policy=CalibrationSensitive())
    assert [r["generation_digest"] for r in plain] == [r["generation_digest"] for r in sensitive]
