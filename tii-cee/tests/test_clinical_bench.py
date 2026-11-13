"""The bench stage: what it counts, its two processes, and faults reported as results."""
import json

import pytest

from test_clinical_support import approved, context, cpu_stack, fake_evaluate, fake_train, prepared
from tiicee.clinical import admissions, bench, folds, use_prior_studies
from tiicee.pipeline import export_guard
from tiicee.pipeline.stages import StageInterrupted

use_prior_studies()

FACTS = {"observation_stop_ids": [41, 42], "observation_stop_strips": True, "eos_id": 2, "pad_is_eos": False,
         "label_ends_with_eos": True, "assistant_turn_ends_with_eos": True, "system_message_sent": True,
         "date_strings": False, "clock_changes": False, "thinking_changes": False, "fixed_is_default_without_date": True}
MEMORY = {"peak_allocated_gb": 61.5, "peak_reserved_gb": 70.25, "device_gb": 95.0}


def decision(text, ids, finish="stop", stop=None):
    return {"role": "decision", "text": text, "token_ids": ids, "finish_reason": finish, "stop_reason": stop}


def test_adherence_counts_format_stop_stripping_and_eos_from_the_logged_generations():
    test = "Thought: check\nAction: Test\nAction Input: CT\n"
    results = {
        "a": {"row": {"prediction": "appendicitis", "termination": "diagnosis", "hypotheses": ["appendicitis", "unknown"],
                      "confidences": [0.7, None]},
              "steps": [{"role": "hypothesis", "text": "Hypothesis: appendicitis\nConfidence: 7", "token_ids": [1, 2]},
                        decision(test, [7, 8, 41, 42], stop="Observation:"),              # stop ids present: strippable
                        decision(test, [7, 8, 99], stop="Observation:"),                  # the word encoded another way
                        decision("Thought: done\nAction: Diagnosis\nAction Input: appendicitis", [7, 2]),
                        {"role": "confidence", "text": "7", "token_ids": [5]}]},
        "b": {"row": {"prediction": None, "termination": "malformed", "hypotheses": ["pancreatitis"], "confidences": [0.4]},
              "steps": [decision("I am not sure what to do", [7, 9]),                     # ended, but not on EOS
                        decision("Thought: " + "word " * 300, [7] * 256, finish="length")]},
    }
    report = bench.adherence(results, FACTS)
    assert report["format"] == {"admissions": 2, "answered": 1, "ended": {"diagnosis": 1, "malformed": 1},
                                "decisions": 5, "decisions_formatted": 3, "decisions_truncated": 1,
                                "hypotheses": 3, "hypotheses_valid": 2, "confidences_valid": 2}
    assert report["observation_stop"] == {"ids": 2, "strips_after_newline": True, "stopped": 2, "stripped": 1}
    assert report["eos"] == {"pad_is_eos": False, "label_ends_with_eos": True, "assistant_turn_ends_with_eos": True,
                             "ended_naturally": 2, "ended_with_eos": 1}
    assert report["template"]["date_strings"] is False and report["template"]["fixed_is_default_without_date"] is True
    export_guard.check_aggregate({"stage": "bench", "evaluation": {"outcome": "ok", **report}})  # counts and flags only


def test_only_weight_sync_and_cuda_memory_faults_are_bench_results():
    assert bench.fault(RuntimeError("Policy-to-vLLM weight transfer failed: size mismatch")) == "weight_sync_failed"
    assert bench.fault(RuntimeError("Adapter merge left 2 inference parameters unloaded")) == "weight_sync_failed"
    assert bench.fault(type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory")) == "out_of_memory"
    assert bench.fault(RuntimeError("anything else")) is None


def test_the_bench_data_are_the_first_admissions_unchanged(tmp_path):
    source = prepared(tmp_path, with_test=False)
    bench.cut(source, tmp_path / "small", 10, 3)
    for name, count in (("train.csv", 10), ("val.csv", 3)):
        full, small = (admissions.read((folder / name).read_bytes()) for folder in (source, tmp_path / "small"))
        assert small.admissions == full.admissions[:count] and small.header == full.header
    assert (tmp_path / "small" / "lab_test_mapping.csv").read_bytes() == (source / "lab_test_mapping.csv").read_bytes()


@pytest.fixture
def bench_stage(tmp_path, monkeypatch):
    from tim import fixture
    monkeypatch.setenv("CDM_WORK_DIR", str(fixture.build(tmp_path / "baseline")))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(bench, "gpu_memory", lambda: dict(MEMORY))
    data = context(tmp_path, "data")
    folds.run_data(data)

    def stage(stage_id="E9_bench", pause_at="30"):  # 20 admissions: 40 steps, the pause inside the second epoch
        return context(tmp_path, stage_id, ["--backbone", "nemotron", "--train-admissions", "20", "--val-admissions", "5",
                                            "--pause-at", pause_at], {"data": data.out_dir}, {"training": 269})
    return stage


@cpu_stack
def test_the_bench_pauses_resumes_then_rolls_out_in_its_own_process_and_reports(bench_stage, monkeypatch):
    from replication import gpu
    from tim import fixture
    monkeypatch.setattr(gpu, "train", fake_train())
    monkeypatch.setattr(gpu, "evaluate", lambda *args, **kwargs: pytest.fail("no second engine in a training process"))
    first = bench_stage()
    with pytest.raises(StageInterrupted):  # the bench's own stand-in for Slurm's warning, inside epoch two
        bench.run(first)
    assert first.load_checkpoint() == {"saved_step": 30, "paused_at": [30]}

    second = bench_stage()
    with pytest.raises(StageInterrupted):  # training done: the stage asks for a fresh process
        bench.run(second)
    assert second.load_checkpoint()["training"] == {
        "outcome": "ok", "optimizer_steps": 40, "paused_at": [30], **MEMORY,
        "visits_min": 2, "visits_max": 2, "visits_as_recipe": True}
    assert len(admissions.read((second.dir / "data" / "train.csv").read_bytes()).admissions) == 20

    third = bench_stage()
    monkeypatch.setattr(gpu, "train", lambda *args, **kwargs: pytest.fail("training is not repeated"))
    monkeypatch.setattr(gpu, "evaluate", fake_evaluate(fixture.ScriptedPolicy()))
    bench.run(third)
    report = json.loads((third.out_dir / "bench.json").read_text())
    assert approved(third) == {"stage": "E9_bench", **report}
    assert (report["backbone"], report["train_admissions"], report["val_admissions"]) == ("nemotron", 20, 5)
    assert report["training"]["outcome"] == "ok" and report["evaluation"]["outcome"] == "ok"
    form = report["evaluation"]["format"]
    assert form["admissions"] == 5 and form["decisions"] >= 5 and sum(form["ended"].values()) == 5
    assert form["hypotheses_valid"] <= form["hypotheses"] and report["evaluation"]["template"]["system_message_sent"]
    assert len((third.out_dir / "rows.jsonl").read_text().splitlines()) == 5


@cpu_stack
def test_the_bench_reports_a_resume_that_loses_the_epoch_order(bench_stage, monkeypatch):
    from replication import gpu
    monkeypatch.setattr(gpu, "train", fake_train(replays_epochs_on_resume=False))
    with pytest.raises(StageInterrupted):
        bench.run(bench_stage())
    with pytest.raises(StageInterrupted):
        bench.run(bench_stage())
    training = bench_stage().load_checkpoint()["training"]
    assert training["visits_as_recipe"] is False and training["visits_min"] < 2 < training["visits_max"]


@cpu_stack
def test_a_weight_sync_fault_is_the_benchs_result_not_a_crash(bench_stage, monkeypatch):
    from replication import gpu

    def failing(cfg, output_dir, data_dir, seed):
        raise RuntimeError("Policy-to-vLLM weight transfer failed: tied embedding")

    monkeypatch.setattr(gpu, "train", failing)
    ctx = bench_stage()
    bench.run(ctx)
    assert approved(ctx)["training"] == {"outcome": "weight_sync_failed", "paused_at": [], **MEMORY}
    assert approved(ctx)["evaluation"] is None

    def broken(cfg, output_dir, data_dir, seed):
        raise RuntimeError("an unrelated failure")

    monkeypatch.setattr(gpu, "train", broken)
    with pytest.raises(RuntimeError, match="unrelated"):  # anything else fails the stage
        bench.run(bench_stage("other_bench"))


@cpu_stack
def test_an_evaluation_weight_sync_fault_is_reported(bench_stage, monkeypatch):
    from replication import gpu
    monkeypatch.setattr(gpu, "train", fake_train())
    with pytest.raises(StageInterrupted):
        bench.run(bench_stage(pause_at="0"))  # no forced pause: training completes in one part

    def unloaded(*args, **kwargs):
        raise RuntimeError("Adapter merge left 1 inference parameters unloaded")

    monkeypatch.setattr(gpu, "evaluate", unloaded)
    second = bench_stage(pause_at="0")
    bench.run(second)
    assert approved(second)["evaluation"] == {"outcome": "weight_sync_failed"} and approved(second)["training"]["outcome"] == "ok"
