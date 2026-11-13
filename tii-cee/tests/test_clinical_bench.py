"""The bench stage: what it counts, its five processes, the comparison of a training in parts with
the uninterrupted reference, and faults reported as results."""
import json
import os

import pytest

from test_clinical_support import (approved, context, cpu_stack, fake_evaluate, fake_train, prepared,
                                   training_environment, until_done)
from tiicee.clinical import admissions, bench, folds, plan, use_prior_studies
from tiicee.pipeline import export_guard
from tiicee.pipeline.stages import StageInterrupted

use_prior_studies()


@pytest.fixture(autouse=True)
def job_environment(monkeypatch):
    training_environment(monkeypatch)  # the bench's job environment; it sets tim's cuBLAS value itself to evaluate


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


def test_only_named_faults_are_bench_results():
    assert bench.fault(plan.Departed("optimizer step 3 is not on its planned admission")) == "departed_from_plan"
    assert bench.fault(RuntimeError("Policy-to-vLLM weight transfer failed: size mismatch")) == "weight_sync_failed"
    assert bench.fault(RuntimeError("Adapter merge left 2 inference parameters unloaded")) == "weight_sync_failed"
    assert bench.fault(type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory")) == "out_of_memory"
    assert bench.fault(RuntimeError("anything else")) is None


def test_each_backbones_defaults_fit_a_three_hour_bench():
    nemotron, granite = bench.options_of(["--backbone", "nemotron"]), bench.options_of(["--backbone", "granite"])
    assert (nemotron.train_admissions, nemotron.val_admissions, nemotron.pause_at, nemotron.reference) == (60, 8, 90, True)
    assert (granite.train_admissions, granite.val_admissions, granite.pause_at, granite.reference) == (110, 12, 150, False)
    for options in (nemotron, granite):  # the pause lies inside the second of the two epochs
        assert options.train_admissions < options.pause_at < 2 * options.train_admissions
    assert 2 * granite.train_admissions > 200  # Granite's one training reaches all three 100-step loss phases

    def gpu_hours(options):  # the docstring's assumptions: 24 s a step, 20 s an episode, 5 min a process
        runs, processes = (2, 5) if options.reference else (1, 4)
        steps, episodes = runs * 2 * options.train_admissions, (runs * 4 + 1) * options.val_admissions
        return (steps * 24 + episodes * 20 + processes * 300) / 3600

    assert round(gpu_hours(nemotron), 1) == 2.4 and round(gpu_hours(granite), 1) == 2.1  # as the docstring states
    explicit = bench.options_of(["--backbone", "granite", "--reference", "--train-admissions", "20", "--pause-at", "0"])
    assert (explicit.reference, explicit.train_admissions, explicit.pause_at, explicit.val_admissions) == (True, 20, 0, 12)
    assert bench.options_of(["--backbone", "nemotron", "--no-reference"]).reference is False


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
    monkeypatch.setattr(bench, "gpu_memory", lambda: dict(MEMORY))
    data = context(tmp_path, "data")
    folds.run_data(data)

    def stage(*extra, stage_id="E9_bench", pause_at="30"):  # 20 admissions: 40 steps, epochs of 20, a pause inside the second
        return context(tmp_path, stage_id, ["--backbone", "nemotron", "--train-admissions", "20", "--val-admissions", "5",
                                            "--pause-at", pause_at, *extra], {"data": data.out_dir}, {"training": 269})
    return stage


TRAINED = {"outcome": "ok", "optimizer_steps": 40, "follows_plan": True, "visits_min": 2, "visits_max": 2,
           "visits_as_recipe": True, **MEMORY}


@cpu_stack
def test_the_bench_trains_one_seed_uninterrupted_and_in_parts_and_finds_them_identical(bench_stage, monkeypatch):
    from replication import gpu
    from tim import fixture
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # a job script that exports it for every GPU job
    happened, cublas = [], []
    train_stand_in = fake_train(lambda step: cublas.append(os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
    evaluate_stand_in = fake_evaluate(fixture.ScriptedPolicy())
    monkeypatch.setattr(gpu, "train", lambda *args, **kwargs: happened.append("train") or train_stand_in(*args, **kwargs))
    monkeypatch.setattr(gpu, "evaluate", lambda *args, **kwargs: happened.append("evaluate") or evaluate_stand_in(*args, **kwargs))

    ctx, processes = until_done(lambda: happened.append("process") or bench_stage(), bench.run)
    assert processes == 5 and happened == ["process", "train"] * 4 + ["process", "evaluate"]  # one engine per process
    report = json.loads((ctx.out_dir / "bench.json").read_text())
    assert approved(ctx) == {"stage": "E9_bench", **report}
    assert (report["backbone"], report["seed"], report["train_admissions"], report["val_admissions"]) == ("nemotron", 269, 20, 5)
    assert report["reference"] == TRAINED                          # uninterrupted, the sampler untouched
    assert report["training"] == {**TRAINED, "paused_at": [20, 30]}  # the epoch boundary, then inside epoch two
    assert report["sequences_identical"] is True and report["visits_identical"] is True
    assert "sha256" not in json.dumps(approved(ctx))               # digests over admission identifiers stay private
    assert set(ctx.load_checkpoint()["training"]) - set(report["training"]) == {"sequence_sha256", "visits_sha256"}
    assert not (ctx.dir / "reference").exists()                    # only the reference's record is kept

    assert len(cublas) == 80 and set(cublas) == {None}             # both trainings ran without the cuBLAS setting
    assert json.loads((ctx.out_dir / "rollout.json").read_text())["cublas_workspace_config"] == ":4096:8"  # the rollout with it
    assert report["evaluation"]["outcome"] == "ok"
    form = report["evaluation"]["format"]
    assert form["admissions"] == 5 and form["decisions"] >= 5 and sum(form["ended"].values()) == 5
    assert form["hypotheses_valid"] <= form["hypotheses"] and report["evaluation"]["template"]["system_message_sent"]
    assert len((ctx.out_dir / "rows.jsonl").read_text().splitlines()) == 5
    assert len(admissions.read((ctx.dir / "data" / "train.csv").read_bytes()).admissions) == 20


@cpu_stack
def test_a_training_in_parts_that_leaves_the_plan_is_the_benchs_result(bench_stage, monkeypatch):
    from replication import gpu
    stand_ins = iter([fake_train(), fake_train(stray_at=25), fake_train(stray_at=25)])  # a sound reference; the parts stray

    def stage():
        monkeypatch.setattr(gpu, "train", next(stand_ins))
        return bench_stage()

    monkeypatch.setattr(gpu, "evaluate", lambda *args, **kwargs: pytest.fail("a departed training is not rolled out"))
    ctx, processes = until_done(stage, bench.run)
    report = approved(ctx)
    assert processes == 3  # the reference, epoch one, then the departure at step 25
    assert report["reference"] == TRAINED
    assert report["training"] == {"outcome": "departed_from_plan", **MEMORY, "paused_at": [20]}
    assert report["evaluation"] is None
    assert report["sequences_identical"] is False and report["visits_identical"] is False


@cpu_stack
def test_a_reference_that_does_not_train_on_the_plan_fails_the_comparison(bench_stage, monkeypatch):
    """The plan is drawn from fresh samplers; if the untouched trainer trains on anything else, the
    plan is not what an uninterrupted run consumes and the bench must say so."""
    from replication import gpu
    from tim import fixture
    stand_ins = iter([fake_train(stray_at=25)] + [fake_train()] * 3)

    def stage():
        monkeypatch.setattr(gpu, "train", next(stand_ins, None))
        return bench_stage()

    monkeypatch.setattr(gpu, "evaluate", fake_evaluate(fixture.ScriptedPolicy()))
    ctx, processes = until_done(stage, bench.run)
    report = approved(ctx)
    assert processes == 5 and report["reference"] == {"outcome": "departed_from_plan", **MEMORY}
    assert report["training"] == {**TRAINED, "paused_at": [20, 30]}
    assert report["sequences_identical"] is False and report["visits_identical"] is False


@cpu_stack
def test_two_runs_that_each_follow_a_different_plan_are_not_identical(bench_stage, monkeypatch):
    """Each run draws its own plan. Were the order to differ between processes of one seed, both
    runs would follow their plans and visit every admission twice, in different orders."""
    from replication import gpu
    from tim import fixture
    stand_ins = iter([fake_train()] + [fake_train(order_salt=1)] * 3)

    def stage():
        monkeypatch.setattr(gpu, "train", next(stand_ins, None))
        return bench_stage()

    monkeypatch.setattr(gpu, "evaluate", fake_evaluate(fixture.ScriptedPolicy()))
    report = approved(until_done(stage, bench.run)[0])
    assert report["reference"] == TRAINED and report["training"] == {**TRAINED, "paused_at": [20, 30]}
    assert report["sequences_identical"] is False and report["visits_identical"] is True


@cpu_stack
def test_the_reference_can_be_left_out(bench_stage, monkeypatch):
    from replication import gpu
    from tim import fixture
    monkeypatch.setattr(gpu, "train", fake_train())
    monkeypatch.setattr(gpu, "evaluate", fake_evaluate(fixture.ScriptedPolicy()))
    ctx, processes = until_done(lambda: bench_stage("--no-reference"), bench.run)
    report = approved(ctx)
    assert processes == 4 and report["reference"] is None and report["training"]["paused_at"] == [20, 30]
    assert report["sequences_identical"] is None and report["visits_identical"] is None


@cpu_stack
def test_a_killed_reference_starts_over_from_nothing(tmp_path, monkeypatch):
    from replication import gpu
    source = prepared(tmp_path)  # 96 admissions: 192 steps, so a 100-step checkpoint exists when the run dies
    monkeypatch.setenv("CDM_WORK_DIR", str(source.parent))
    monkeypatch.setattr(bench, "gpu_memory", lambda: dict(MEMORY))
    data = context(tmp_path, "data")
    folds.run_data(data)

    def stage():
        return context(tmp_path, "bench", ["--backbone", "nemotron", "--train-admissions", "96"], {"data": data.out_dir}, {"training": 269})

    def killed(step):
        if step == 150:
            raise KeyboardInterrupt  # the job dies; nothing of the stage runs after it

    first = stage()
    monkeypatch.setattr(gpu, "train", fake_train(killed))
    with pytest.raises(KeyboardInterrupt):
        bench.run(first)
    assert (first.dir / "reference" / "checkpoint-100").is_dir() and first.load_checkpoint() is None

    steps = []
    monkeypatch.setattr(gpu, "train", fake_train(steps.append))
    with pytest.raises(StageInterrupted):
        bench.run(stage())
    assert steps == list(range(1, 193))  # not resumed from step 100: an uninterrupted run, whole
    assert stage().load_checkpoint()["reference"]["optimizer_steps"] == 192


@cpu_stack
def test_a_weight_sync_fault_is_the_benchs_result_not_a_crash(bench_stage, monkeypatch):
    from replication import gpu

    def failing(cfg, output_dir, data_dir, seed):
        raise RuntimeError("Policy-to-vLLM weight transfer failed: tied embedding")

    monkeypatch.setattr(gpu, "train", failing)
    ctx, processes = until_done(bench_stage, bench.run)
    report = approved(ctx)
    assert processes == 2 and report["reference"] == {"outcome": "weight_sync_failed", **MEMORY}
    assert report["training"] == {"outcome": "weight_sync_failed", "paused_at": [], **MEMORY}
    assert report["evaluation"] is None and report["sequences_identical"] is False

    def broken(cfg, output_dir, data_dir, seed):
        raise RuntimeError("an unrelated failure")

    monkeypatch.setattr(gpu, "train", broken)
    with pytest.raises(RuntimeError, match="unrelated"):  # anything else fails the stage
        bench.run(bench_stage(stage_id="other_bench"))


@cpu_stack
def test_an_evaluation_weight_sync_fault_is_reported(bench_stage, monkeypatch):
    from replication import gpu

    def unloaded(*args, **kwargs):
        raise RuntimeError("Adapter merge left 1 inference parameters unloaded")

    monkeypatch.setattr(gpu, "train", fake_train())
    monkeypatch.setattr(gpu, "evaluate", unloaded)
    ctx, processes = until_done(lambda: bench_stage("--no-reference", pause_at="0"), bench.run)
    assert processes == 3  # epoch one, epoch two, then the evaluation
    assert approved(ctx)["evaluation"] == {"outcome": "weight_sync_failed"} and approved(ctx)["training"]["outcome"] == "ok"
