"""The rollout stage: every role's prompt is logged, tim's digest is reproduced, nothing runs twice;
the V1 helper; and the launch-check acceptance of a regenerated rollout."""
import hashlib
import json
import os
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_clinical_support import approved, context, cpu_stack, fake_evaluate, unset_gpu_environment
from tiicee.clinical import acceptance, rollout, sha256, use_prior_studies, v1
from tiicee.clinical.backbone import BACKBONES
from tiicee.pipeline.stages import StageInterrupted

use_prior_studies()

MARKERS = {"decision": "You decide. Context:", "hypothesis": "You hypothesise. Information:"}
HYPOTHESIS = "<system>fixed</system>You hypothesise. Information: invented history<assistant>"
DECISION = "<system>fixed</system>You decide. Context: invented history\n---\nThe current hypothesis is: appendicitis\n"
CONFIDENCE = HYPOTHESIS + "Hypothesis: appendicitis\nConfidence: "


class Engine:
    """vLLM's part in this: generate() returns, per prompt, its token ids and one completion."""

    llm_engine = "the engine"

    def __init__(self):
        self.calls = []

    def get_tokenizer(self):
        return "the tokenizer"

    def generate(self, prompts, sampling_params=None, use_tqdm=True):
        self.calls.append((list(prompts), sampling_params, use_tqdm))
        return [SimpleNamespace(prompt=prompt, prompt_token_ids=[11, 12, len(prompt)], outputs=[SimpleNamespace(
            text=f"completion {len(self.calls)}", token_ids=(5, 6, len(self.calls)), finish_reason="stop", stop_reason=None)])
            for prompt in prompts]


def episode(model):
    """The calls upstream's Environment.run makes for one admission with an error turn in the middle."""
    model.generate([HYPOTHESIS], sampling_params="greedy", use_tqdm=False)
    model.generate([DECISION], sampling_params="greedy", use_tqdm=False)
    model.generate([], sampling_params="greedy", use_tqdm=False)  # no hypothesis after an error turn
    model.generate([DECISION + "Thought: x\nAction: Test\nAction Input: Brain MRI\n"], sampling_params="greedy", use_tqdm=False)
    model.generate([CONFIDENCE], sampling_params="greedy", use_tqdm=False)


def test_the_prompt_log_captures_every_step_of_every_role():
    engine = Engine()
    log = rollout.PromptLog(engine, MARKERS)
    episode(log)
    assert [(step["index"], step["turn"], step["role"]) for step in log.steps] == \
        [(0, 0, "hypothesis"), (1, 0, "decision"), (2, 1, "decision"), (3, None, "confidence")]
    for step, (prompts, sampling, tqdm) in zip(log.steps, [call for call in engine.calls if call[0]]):
        assert step["prompt"] == prompts[0] and step["prompt_sha256"] == hashlib.sha256(prompts[0].encode()).hexdigest()
        assert step["prompt_tokens"] == 3 and (sampling, tqdm) == ("greedy", False)  # arguments reach the engine untouched
        assert step["prompt_ids_sha256"] == hashlib.sha256(f"11,12,{len(prompts[0])}".encode()).hexdigest()
    assert [step["token_ids"] for step in log.steps] == [[5, 6, 1], [5, 6, 2], [5, 6, 4], [5, 6, 5]]
    assert [step["text"] for step in log.steps] == ["completion 1", "completion 2", "completion 4", "completion 5"]
    assert log.get_tokenizer() == "the tokenizer" and log.llm_engine == "the engine"
    json.dumps(log.steps, allow_nan=False)  # what the stage's item log requires


def test_a_prompt_of_no_single_agent_is_refused():
    log = rollout.PromptLog(Engine(), MARKERS)
    with pytest.raises(ValueError, match="exactly one agent"):
        log.generate(["an unrecognised prompt"])
    with pytest.raises(ValueError, match="exactly one agent"):
        log.generate([HYPOTHESIS + DECISION])


def test_the_digest_covers_decisions_and_hypotheses_and_leaves_confidence_out():
    log = rollout.PromptLog(Engine(), MARKERS)
    episode(log)
    digest = rollout.generation_digest(log.steps)
    changed = [dict(step, token_ids=step["token_ids"] + [9]) if step["role"] == "confidence" else step for step in log.steps]
    assert rollout.generation_digest(changed) == digest
    changed = [dict(step, token_ids=step["token_ids"] + [9]) if step["index"] == 2 else step for step in log.steps]
    assert rollout.generation_digest(changed) != digest
    assert rollout.generation_digest(list(reversed(log.steps))) != digest


def test_a_completed_admission_is_stored_once_and_the_stop_flag_ends_the_run(tmp_path):
    ctx = context(tmp_path, "rollout")
    completed = rollout.Completed(ctx, {101: "print-a", 102: "print-b", 103: "print-c"})
    completed.prompts = rollout.PromptLog(Engine(), MARKERS)
    assert 101 not in completed.done

    episode(completed.prompts)
    steps = list(completed.prompts.steps)
    completed.append(101, {"patient_id": 101, "generation_digest": rollout.generation_digest(steps)})
    assert 101 in completed.done and 102 not in completed.done and completed.prompts.steps == []
    progress = json.loads(ctx.paths.progress_file("rollout").read_text())
    assert (progress["done"], progress["total"], progress["unit"]) == (1, 3, "admission")
    assert ctx.results()["print-a"] == {"row": {"patient_id": 101, "generation_digest": rollout.generation_digest(steps)}, "steps": steps}

    episode(completed.prompts)
    with pytest.raises(RuntimeError, match="generation digest"):  # a log that misses a generation is not stored
        completed.append(102, {"patient_id": 102, "generation_digest": "0" * 64})
    assert not ctx.is_done("print-b")
    with pytest.raises(ValueError, match="twice"):
        completed.append(101, {"patient_id": 101, "generation_digest": rollout.generation_digest([])})

    ctx.stop = True  # Slurm's warning: every remaining admission now reads as done
    assert 102 in completed.done and 103 in completed.done and not ctx.is_done("print-c")
    resumed = context(tmp_path, "rollout")  # the next part reads the item log
    assert resumed.is_done("print-a") and not resumed.is_done("print-b")


def test_markers_are_the_fixed_openings_of_the_configured_templates(tmp_path):
    (tmp_path / "decision.txt").write_text("You decide.\nContext:\n{patient_history}\n---\n{first_hypothesis}")
    (tmp_path / "hypothesis.txt").write_text("You hypothesise.\nInformation:\n{provided_test_results}")
    cfg = SimpleNamespace(environment=SimpleNamespace(prompt_template_file=str(tmp_path / "decision.txt"),
                                                      hypothesis_prompt_template_file=str(tmp_path / "hypothesis.txt")))
    assert rollout.markers(cfg) == {"decision": "You decide.\nContext:\n", "hypothesis": "You hypothesise.\nInformation:\n"}
    (tmp_path / "hypothesis.txt").write_text("{provided_test_results}")
    with pytest.raises(ValueError):
        rollout.markers(cfg)


def test_v1_is_tims_rule_applied_by_tims_function():
    from tim.analysis import v1_rows
    rows = [{"patient_id": 1, "label": "appendicitis", "prediction": None, "raw_prediction": None, "hypotheses": ["unknown", "appendicitis", "unknown"]},
            {"patient_id": 2, "label": "pancreatitis", "prediction": "cholecystitis", "raw_prediction": "cholecystitis", "hypotheses": ["pancreatitis"]},
            {"patient_id": 3, "label": "pancreatitis", "prediction": None, "raw_prediction": None, "hypotheses": ["unknown"]}]
    out = v1.v1_rows(rows)
    assert out == v1_rows(rows)
    assert [(row["prediction"], row["prediction_source"], row["variant"]) for row in out] == \
        [("appendicitis", "hypothesis_fallback", "V1"), ("cholecystitis", "diagnosis", "V1"), (None, "none", "V1")]
    assert rows[0]["prediction"] is None  # the V0 rows are left as they were


def rows_for(count, wrong=(), other_digest=()):
    return [{"patient_id": i, "prediction": "pancreatitis" if i in wrong else ("appendicitis" if i % 7 else None),
             "generation_digest": f"digest-{i}-{'x' if i in other_digest else 'a'}"} for i in range(count)]


def test_acceptance_needs_ninety_nine_percent_of_final_answers():
    stored = rows_for(240)
    assert acceptance.compare(stored, rows_for(240)) == {
        "admissions": 240, "regenerated": 240, "paired": 240, "final_answers_matching": 240,
        "digests_matching": 240, "required_percent": 99, "accepted": True}
    two_off = acceptance.compare(stored, rows_for(240, wrong={3, 14}, other_digest={3, 14, 15}))
    assert (two_off["final_answers_matching"], two_off["digests_matching"], two_off["accepted"]) == (238, 237, True)
    assert acceptance.compare(stored, rows_for(240, wrong={3, 14, 21}))["accepted"] is False  # 237 of 240 is 98.75%
    assert acceptance.compare(stored, rows_for(240, wrong={0}))["final_answers_matching"] == 239  # no answer against an answer
    missing = acceptance.compare(stored, rows_for(239))
    assert (missing["paired"], missing["final_answers_matching"], missing["accepted"]) == (239, 239, False)
    assert acceptance.compare([], [])["accepted"] is False
    with pytest.raises(ValueError):
        acceptance.compare(stored + stored[:1], rows_for(240))


def test_the_acceptance_stage_reads_tims_log_and_publishes_counts_only(tmp_path):
    stored, regenerated = tmp_path / "val-a_V0.jsonl", tmp_path / "rows.jsonl"
    stored.write_text("".join(json.dumps({"patient_id": row["patient_id"], "row": row, "sha256": "unused"}) + "\n"
                              for row in rows_for(240)))
    regenerated.write_text("".join(json.dumps(row) + "\n" for row in rows_for(240, wrong={5})))
    ctx = context(tmp_path, "E4_acceptance", bound={"stored": stored, "regenerated": regenerated})
    acceptance.run(ctx)
    report = json.loads((ctx.out_dir / "acceptance.json").read_text())
    assert (report["final_answers_matching"], report["accepted"]) == (239, True)
    assert approved(ctx) == {"stage": "E4_acceptance", **report}
    assert all(isinstance(value, (int, bool, str)) for value in approved(ctx).values())


# ---- the replication's and tim's real CPU code: the upstream environment, tim's evaluator and scripted policy


@pytest.fixture(autouse=True)
def gpu_environment(monkeypatch):
    unset_gpu_environment(monkeypatch)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    from tim import fixture
    return fixture.build(tmp_path_factory.mktemp("clinical") / "baseline") / "prepared"


@pytest.fixture
def adapter(tmp_path):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text("{}")
    return tmp_path / "adapter"


def rollout_stage(tmp_path, baseline, adapter, backbone="nemotron", stage_id="E8_rollout", split="val.csv"):
    roles = {"split": baseline / split, "lab_test_mapping": baseline / "lab_test_mapping.csv",
             "adapter": adapter, "train_csv": baseline / "train.csv"}
    if backbone == "granite":
        roles["backbone_overlay"] = "tii-cee/clinical_configs/backbone/granite_3_3_8b.yaml"  # as graph.json would name it
    return context(tmp_path, stage_id, ["--backbone", backbone], roles, {"policy": 269})


def tims_own_v0(baseline):
    """tim's V0 evaluator run directly, the way tim's val-a stage configures it (tim/runner.py, split_cfg)."""
    from replication.config import compose, stream_seed
    from replication.shim import generation_settings
    from src.dataset.mimic_cdm_dataset import MIMICCDMDataset
    from tim import fixture, runner
    cfg = compose("primary", baseline, None, 269)
    cfg.training.seed = stream_seed(stream_seed(269, "val", "policy"), "val", "evaluation")
    d = cfg.dataset
    dataset = MIMICCDMDataset(data_file=str(baseline / "val.csv"), lab_test_mapping_file=str(baseline / "lab_test_mapping.csv"),
                              lab_tests=list(d.lab_tests), imaging_tests=list(d.imaging_tests), other_tests=list(d.other_tests))
    evaluate = runner.make_evaluator(runner.VARIANTS["V0"], dataset, None, None, runner.Limits(**runner.CONFIG["limits"]))
    return evaluate(cfg, None, fixture.ScriptedPolicy(), generation_settings(False))


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@cpu_stack
def test_the_digest_equals_the_one_tims_recorder_computes():
    from tim.runner import Recorder
    log = rollout.PromptLog(Engine(), MARKERS)
    recorder = Recorder(log)
    episode(recorder)
    assert rollout.generation_digest(log.steps) == hashlib.sha256(",".join(recorder.digests).encode()).hexdigest()
    assert len(recorder.digests) == 3  # tim leaves the confidence generation out; the log keeps it


@cpu_stack
def test_a_rollout_logs_every_role_and_reproduces_tims_rows_and_digests(tmp_path, baseline, adapter, monkeypatch):
    from replication import gpu
    from tim import fixture
    monkeypatch.setattr(gpu, "evaluate", fake_evaluate(fixture.ScriptedPolicy()))
    ctx = rollout_stage(tmp_path, baseline, adapter)
    rollout.run(ctx)

    rows, expected = lines(ctx.out_dir / "rows.jsonl"), tims_own_v0(baseline)
    assert rows == expected and len(rows) == 24  # tim's V0 rows, generation_digest included
    assert {"patient_id", "label", "prediction", "provided_tests", "requested_tests", "unavailable_requests", "reward",
            "hypotheses", "confidences", "ece_triplets", "generation_digest"} <= set(rows[0])  # the replication's row format

    logged = lines(ctx.out_dir / "steps.jsonl")
    assert [entry["patient_id"] for entry in logged] == [row["patient_id"] for row in rows]
    roles = set()
    for entry, row in zip(logged, rows):
        steps = entry["steps"]
        assert rollout.generation_digest(steps) == row["generation_digest"]
        assert all(step["prompt_sha256"] == sha256(step["prompt"].encode()) for step in steps)
        assert all("detailed thinking off" in step["prompt"] for step in steps)  # Nemotron's switch, on every role
        assert sum(step["role"] == "decision" for step in steps) == row["n_turns"]
        assert sum(step["role"] == "hypothesis" for step in steps) == len(row["hypotheses"])
        assert [step["turn"] for step in steps if step["role"] == "decision"] == list(range(row["n_turns"]))
        roles.update(step["role"] for step in steps)
    assert roles == {"decision", "hypothesis", "confidence"}

    assert lines(ctx.out_dir / "v1_rows.jsonl") == v1.v1_rows(rows)
    record = json.loads((ctx.out_dir / "rollout.json").read_text())
    assert (record["recipe"], record["seed"], record["admissions"], record["evaluation_split_name"]) == ("primary", 269, 24, "val")
    assert record["split_sha256"] == sha256((baseline / "val.csv").read_bytes())
    assert record["cublas_workspace_config"] == os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert record["runtime"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"  # the launchers' default
    assert record["config"]["training"]["seed"] != 269  # the evaluation seed gpu.evaluate derived, as it ran
    assert record["tokenizer"]["system_message_sent"] and record["tokenizer"]["observation_stop_strips"]
    assert approved(ctx) == {"stage": "E8_rollout", "backbone": "nemotron", "seed": 269, "admissions": 24,
                             "answered": sum(row["prediction"] is not None for row in rows)}


@cpu_stack
def test_a_resumed_rollout_never_repeats_an_admission(tmp_path, baseline, adapter, monkeypatch):
    from replication import gpu
    from tim import fixture

    class Counting(fixture.ScriptedPolicy):
        episodes = 0

        def generate(self, prompts, sampling_params=None, use_tqdm=False):
            for prompt in prompts:  # an episode's first decision prompt has no earlier action in it
                if "Here is some context on the patient:" in prompt and "\nAction: " not in prompt.split("Here is some context on the patient:")[1]:
                    Counting.episodes += 1
                    if Counting.episodes == 10:
                        ctx.stop = True  # Slurm's warning arrives during the tenth admission
            return super().generate(prompts, sampling_params, use_tqdm)

    monkeypatch.setattr(gpu, "evaluate", fake_evaluate(Counting()))
    ctx = rollout_stage(tmp_path, baseline, adapter)
    with pytest.raises(StageInterrupted):
        rollout.run(ctx)
    assert len(ctx.results()) == 10 and not (ctx.out_dir / "rows.jsonl").exists()

    ctx = rollout_stage(tmp_path, baseline, adapter)  # the next part
    rollout.run(ctx)
    assert Counting.episodes == 24 and len(ctx.results()) == 24  # 10 + 14: none ran twice
    assert lines(ctx.out_dir / "rows.jsonl") == tims_own_v0(baseline)

    ctx = rollout_stage(tmp_path, baseline, adapter)  # everything complete: the engine is not started again
    monkeypatch.setattr(gpu, "evaluate", lambda *args, **kwargs: pytest.fail("nothing is left to run"))
    rollout.run(ctx)
    assert Counting.episodes == 24


@cpu_stack
def test_the_test_split_and_trained_on_admissions_are_never_rolled_out(tmp_path, baseline, adapter, monkeypatch):
    from replication import gpu
    monkeypatch.setattr(gpu, "evaluate", lambda *args, **kwargs: pytest.fail("the engine must not start"))
    with pytest.raises(ValueError, match="test split"):
        rollout.run(rollout_stage(tmp_path, baseline, adapter, stage_id="on_test", split="test.csv"))
    with pytest.raises(ValueError, match="trained on"):  # a policy never scores an admission it trained on
        rollout.run(rollout_stage(tmp_path, baseline, adapter, stage_id="on_train", split="train.csv"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS"):
        rollout.run(rollout_stage(tmp_path, baseline, adapter, stage_id="other_cublas"))


@cpu_stack
def test_a_granite_rollout_sends_the_fixed_system_message_on_every_role(tmp_path, baseline, adapter, monkeypatch):
    from replication import gpu, shim
    from tim import fixture
    transformers = ModuleType("transformers")
    transformers.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {"apply_chat_template": lambda self, *a, **k: ""})
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(gpu, "evaluate", fake_evaluate(fixture.ScriptedPolicy()))
    ctx = rollout_stage(tmp_path, baseline, adapter, backbone="granite", stage_id="E9_rollout")
    rollout.run(ctx)
    fixed = BACKBONES["granite"].overlay.read_text().split('system_prompt: "')[1].split('"')[0].replace("\\n", "\n")
    steps = [step for entry in lines(ctx.out_dir / "steps.jsonl") for step in entry["steps"]]
    assert {step["role"] for step in steps} == {"decision", "hypothesis", "confidence"}
    assert all(fixed in step["prompt"] and "detailed thinking off" not in step["prompt"] for step in steps)
    record = json.loads((ctx.out_dir / "rollout.json").read_text())
    assert (record["backbone"], record["revision"]) == (BACKBONES["granite"].model, BACKBONES["granite"].revision)
    assert shim.SYSTEM_MESSAGE == "detailed thinking off"


@cpu_stack
def test_the_regenerated_primary_rollout_is_accepted_against_tims_stored_log(tmp_path, baseline, adapter, monkeypatch):
    from replication import gpu
    from tim import access, fixture
    monkeypatch.setattr(gpu, "evaluate", fake_evaluate(fixture.ScriptedPolicy()))
    regenerated = rollout_stage(tmp_path, baseline, adapter, stage_id="P_rollout")
    rollout.run(regenerated)
    stored = tmp_path / "val-a_V0_primary.jsonl"  # the lines tim.access.Episodes.append writes
    stored.write_text("".join(json.dumps({"patient_id": row["patient_id"], "row": row, "sha256": access.digest(row)}, sort_keys=True) + "\n"
                              for row in tims_own_v0(baseline)))
    ctx = context(tmp_path, "launch_check_7_4a", bound={"stored": stored, "regenerated": regenerated.out_dir / "rows.jsonl"})
    acceptance.run(ctx)
    assert approved(ctx) == {"stage": "launch_check_7_4a", "admissions": 24, "regenerated": 24, "paired": 24,
                             "final_answers_matching": 24, "digests_matching": 24, "required_percent": 99, "accepted": True}
    assert not re.search(r"9000\d{4}", json.dumps(approved(ctx)))  # no admission identifier leaves
