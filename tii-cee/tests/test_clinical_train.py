"""The training stage: how a part ends, resume, refusals, the environment it requires, and what it binds."""
import json
import random
from types import SimpleNamespace

import pytest

from test_clinical_support import (CallbackBase, approved, context, cpu_stack, fake_train, prepared, split_bytes,
                                   training_environment, until_done)
from tiicee.clinical import backbone, folds, plan, sha256, train, use_prior_studies
from tiicee.pipeline.stages import StageInterrupted
from tiicee.pipeline.util import tree_digest

use_prior_studies()
from replication import gpu  # noqa: E402  (imports with the standard library alone)

ARGS = dict(num_train_epochs=2)  # with max_steps 1000 below: epochs of 500 steps


@pytest.fixture(autouse=True)
def job_environment(monkeypatch):
    training_environment(monkeypatch)


def trainer_step(callback, output_dir, step, due=False):
    """One Trainer step as transformers orders it: on_step_end, then the checkpoint and on_save."""
    args, state = SimpleNamespace(output_dir=str(output_dir), **ARGS), SimpleNamespace(global_step=step, max_steps=1000)
    control = SimpleNamespace(should_save=due, should_training_stop=False)
    control = callback.on_step_end(args, state, control) or control
    if control.should_save:
        save(callback, output_dir, args, state, control)
    return control


def save(callback, output_dir, args, state, control):
    checkpoint = output_dir / f"checkpoint-{state.global_step}"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": state.global_step}))
    callback.on_save(args, state, control)


def test_the_stop_flag_forces_a_save_with_the_sidecar_and_stops(tmp_path):
    stop, made, saved = [False], [], []
    environment = SimpleNamespace(_rng=random.Random(5))
    build = train.stop_aware(gpu._environment_rng_callback, lambda step, total: stop[0], made, saved.append)
    callback = build(CallbackBase, environment, "mimic")
    assert made == [callback]

    control = trainer_step(callback, tmp_path, 41)
    assert (control.should_save, control.should_training_stop) == (False, False) and not list(tmp_path.iterdir())
    trainer_step(callback, tmp_path, 100, due=True)  # an ordinary 100-step checkpoint: sidecar, no pause
    assert (tmp_path / "checkpoint-100" / "environment-rng.json").is_file() and saved == [100]

    stop[0] = True
    environment._rng.random()
    with pytest.raises(train.Paused) as paused:
        trainer_step(callback, tmp_path, 137)
    assert paused.value.step == 137 and paused.value.checkpoint == tmp_path / "checkpoint-137"
    assert saved == [100, 137] and train.resumable(gpu, paused.value.checkpoint)

    restored = SimpleNamespace(_rng=random.Random())
    gpu._restore_environment_rng(restored, paused.value.checkpoint, "mimic")  # what gpu.train does on resume
    assert restored._rng.random() == environment._rng.random()


def test_the_stop_flag_sets_save_and_stop_on_the_trainer_control():
    build = train.stop_aware(gpu._environment_rng_callback, lambda step, total: step == 9, [])
    callback = build(CallbackBase, SimpleNamespace(), "mimic")
    for step, expected in ((8, False), (9, True)):
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        callback.on_step_end(SimpleNamespace(**ARGS), SimpleNamespace(global_step=step, max_steps=1000), control)
        assert control.should_save is expected and control.should_training_stop is expected


def test_a_part_ends_at_the_epoch_boundary_after_the_epochs_validation(tmp_path):
    """The planned part boundary: asked for at on_epoch_end, where the Trainer evaluates and then
    saves, so the pause comes after the validation pass and the resume starts the next epoch."""
    environment = SimpleNamespace(_rng=random.Random(5))
    callback = train.stop_aware(gpu._environment_rng_callback, lambda step, total: True, [])(CallbackBase, environment, "mimic")
    args, state = SimpleNamespace(output_dir=str(tmp_path), **ARGS), SimpleNamespace(global_step=500, max_steps=1000)

    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_step_end(args, state, control)  # the stop flag is up, but this is an epoch's last step
    assert (control.should_save, control.should_training_stop) == (False, False)  # no stop before the validation pass
    callback.on_epoch_end(args, state, control)
    assert control.should_save is True and control.should_training_stop is False
    with pytest.raises(train.Paused) as paused:  # the Trainer has evaluated; now it saves
        save(callback, tmp_path, args, state, control)
    assert paused.value.step == 500 and train.resumable(gpu, tmp_path / "checkpoint-500")

    state.global_step, control = 1000, SimpleNamespace(should_save=True, should_training_stop=True)
    callback.on_epoch_end(args, state, control)  # the last epoch: training simply ends
    save(callback, tmp_path, args, state, control)
    assert (tmp_path / "checkpoint-1000" / "environment-rng.json").is_file()


def test_the_uninterrupted_reference_has_no_part_boundary(tmp_path):
    build = train.stop_aware(gpu._environment_rng_callback, lambda step, total: False, [], pause_at_epochs=False)
    callback = build(CallbackBase, SimpleNamespace(_rng=random.Random(5)), "mimic")
    args, state = SimpleNamespace(output_dir=str(tmp_path), **ARGS), SimpleNamespace(global_step=500, max_steps=1000)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_step_end(args, state, control)
    callback.on_epoch_end(args, state, control)
    assert (control.should_save, control.should_training_stop) == (False, False)


def test_the_callback_records_where_training_ended():
    build = train.stop_aware(gpu._environment_rng_callback, lambda step, total: False, [])
    callback = build(CallbackBase, SimpleNamespace(), "mimic")
    callback.on_train_end(SimpleNamespace(**ARGS), SimpleNamespace(global_step=3840, max_steps=3840), SimpleNamespace())
    assert (callback.final_step, callback.max_steps) == (3840, 3840)


def checkpoint(directory, step, sidecar=True, source="mimic"):
    path = directory / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    if sidecar:
        (path / "environment-rng.json").write_text(json.dumps({"source": source, "rng_state": random.Random(step).getstate()}))
    return path


def test_a_checkpoint_gpu_train_would_refuse_is_not_resumable(tmp_path):
    assert train.resumable(gpu, checkpoint(tmp_path, 100))
    assert not train.resumable(gpu, checkpoint(tmp_path, 200, sidecar=False))
    assert not train.resumable(gpu, checkpoint(tmp_path, 300, source="synthetic"))
    torn = checkpoint(tmp_path, 400)
    (torn / "environment-rng.json").write_text('{"source": "mimic", "rng_st')
    assert not train.resumable(gpu, torn)
    (checkpoint(tmp_path, 500) / "trainer_state.json").unlink()
    assert not train.resumable(gpu, tmp_path / "checkpoint-500")


def test_checkpoints_torn_by_a_kill_are_set_aside_so_resume_finds_a_complete_one(tmp_path):
    assert train.last_checkpoint(gpu, tmp_path) is None
    checkpoint(tmp_path, 100), checkpoint(tmp_path, 200), checkpoint(tmp_path, 300, sidecar=False)
    checkpoint(tmp_path, 1000, sidecar=False)
    assert train.last_checkpoint(gpu, tmp_path) == tmp_path / "checkpoint-200"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint-100", "checkpoint-200", "torn-1000-0", "torn-300-1"]
    assert train.last_checkpoint(gpu, tmp_path) == tmp_path / "checkpoint-200"  # nothing more to set aside
    assert len(list(tmp_path.glob("torn-*"))) == 2


def test_a_stage_reports_progress_and_stops_on_its_flag_or_a_chosen_step(tmp_path):
    ctx = context(tmp_path, "E8_train")
    on_step, saved = train.hooks(ctx)
    assert on_step(40, 3840) is False
    assert json.loads(ctx.paths.progress_file("E8_train").read_text())["done"] == 40  # counts for the monitor
    saved(100)
    assert ctx.load_checkpoint() == {"saved_step": 100}  # the watcher's progress mark moves with each checkpoint
    ctx.stop = True
    assert on_step(41, 3840) is True
    paused, _ = train.hooks(context(tmp_path, "bench"), pause_at=150)
    assert [paused(step, 220) for step in (149, 150, 151)] == [False, True, False]


def test_the_training_environment_is_observed_without_changing_what_the_trainer_gets():
    class Environment:
        def construct_prompt(self, inputs):
            return [dict(item, prompt="built") for item in inputs]

    noted = []
    environment = plan.observed(lambda cfg: Environment(), SimpleNamespace(note=noted.append))("the configuration")
    group = [{"patient_id": 77, "patient_history": "invented"}] * 8  # one admission, eight rollouts
    assert environment.construct_prompt(group) == [dict(item, prompt="built") for item in group]
    environment.construct_prompt([{"patient_id": 78}, {"patient_id": 77}])
    assert noted == [[77], [77, 78]]  # a mixed batch reaches the schedule as it is, to be refused there


def test_visit_counts_are_set_against_the_recipes_epochs():
    ids = [1, 2, 3]
    assert train.visits([1, 2, 3, 3, 1, 2], ids, 2) == {"visits_min": 2, "visits_max": 2, "visits_as_recipe": True}
    assert train.visits([1, 2, 3, 1, 1, 2], ids, 2) == {"visits_min": 1, "visits_max": 3, "visits_as_recipe": False}
    assert train.visits([1, 2, 3, 3, 1, 2, 9, 9], ids, 2)["visits_as_recipe"] is False  # an admission from elsewhere


def test_training_refuses_a_job_environment_other_than_the_replications_launchers(monkeypatch):
    backbone.require_environment(backbone.TRAINING_ENVIRONMENT)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # tim's evaluation setting does not belong in training
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        train.fit(None, 269, None, None, None)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    for value in (None, "max_split_size_mb:512"):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        if value:
            monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", value)
        with pytest.raises(RuntimeError, match="PYTORCH_CUDA_ALLOC_CONF"):
            train.fit(None, 269, None, None, None)


def data_directory(tmp_path, monkeypatch, stage_id="E8_data"):
    source = prepared(tmp_path)
    monkeypatch.setenv("CDM_WORK_DIR", str(source.parent))
    ctx = context(tmp_path, stage_id)
    folds.run_data(ctx)
    return ctx.out_dir


def test_a_training_directory_holding_test_csv_is_refused(tmp_path, monkeypatch):
    data = data_directory(tmp_path, monkeypatch)
    assert train.inspect_data(data)["admissions"] == {"train": 96, "val": 24}
    (data / "test.csv").write_bytes(split_bytes(90000, ["appendicitis"]))
    with pytest.raises(ValueError, match="test.csv"):
        train.inspect_data(data)


def test_a_training_directory_that_trains_on_monitored_or_heldout_admissions_is_refused(tmp_path, monkeypatch):
    source = prepared(tmp_path)
    monkeypatch.setenv("CDM_WORK_DIR", str(source.parent))
    ctx = context(tmp_path, "E10_folds")
    folds.run_folds(ctx)
    fold = ctx.out_dir / "fold-0"
    assert set(train.inspect_data(fold)["admissions"]) == {"train", "val", "heldout"}
    (fold / "train.csv").write_bytes((fold / "train.csv").read_bytes() + (fold / "heldout.csv").read_bytes().split(b"\n", 1)[1])
    with pytest.raises(ValueError, match="more than once"):
        train.inspect_data(fold)


@cpu_stack
def test_training_in_parts_binds_the_policy_an_uninterrupted_run_would_reach(tmp_path, monkeypatch):
    data = data_directory(tmp_path, monkeypatch)  # 96 training admissions: 192 optimizer steps in two epochs

    def stage(stage_id="E8_train_271"):
        return context(tmp_path, stage_id, ["--backbone", "nemotron"], {"data": data}, {"training": 271})

    first = stage()
    monkeypatch.setattr(gpu, "train", fake_train())
    with pytest.raises(StageInterrupted):  # the planned boundary after epoch one; the runner exits 10
        train.run(first)
    model = first.out_dir / "model"
    assert train.resumable(gpu, model / "checkpoint-96") and not (first.out_dir / "policy.json").exists()
    assert first.load_checkpoint() == {"saved_step": 96, "paused_at": [96]}
    assert len(json.loads((model / "plan.json").read_text())["steps"]) == 192

    second = stage()  # the next Slurm part; this one meets Slurm's warning inside epoch two
    monkeypatch.setattr(gpu, "train", fake_train(lambda step: step == 150 and setattr(second, "stop", True)))
    with pytest.raises(StageInterrupted):
        train.run(second)
    assert second.load_checkpoint() == {"saved_step": 150, "paused_at": [96, 150]}
    assert len(json.loads((model / "checkpoint-150" / "sequence.json").read_text())) == 150

    third = stage()
    monkeypatch.setattr(gpu, "train", fake_train())
    train.run(third)
    policy = json.loads((third.out_dir / "policy.json").read_text())
    assert (policy["recipe"], policy["seed"], policy["optimizer_steps"], policy["paused_at"]) == ("primary", 271, 192, [96, 150])
    assert policy["backbone"] == "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    assert policy["revision"] == "54641c1611fcff44fa4865626462445e0a153fc7"
    assert policy["data"] == {name: sha256((data / name).read_bytes()) for name in ("train.csv", "val.csv", "lab_test_mapping.csv")}
    assert policy["admissions"] == {"train": 96, "val": 24} and "trained_ids" not in policy
    assert policy["adapter_sha256"] == tree_digest(third.out_dir / policy["adapter"])[0] and policy["adapter_files"] == 2
    assert policy["manifest_id"] == "manifest-E8_train_271" and policy["config"]["training"]["seed"] == 271
    assert policy["config"]["dataset"]["train_data_file"] == str(data / "train.csv")
    assert set(policy["templates"]) == {"decision_agent.txt", "hypothesis_agent.txt"}
    assert (policy["follows_plan"], policy["visits_min"], policy["visits_max"], policy["visits_as_recipe"]) == (True, 2, 2, True)
    assert policy["plan_sha256"] == sha256(json.dumps(json.loads((model / "plan.json").read_text())["steps"]).encode())
    assert policy["runtime"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert policy["cublas_workspace_config"] == "UNSET"
    training = json.loads((third.out_dir / "training.json").read_text())
    assert training["adapter_dir"] == "model/adapter" and training["validation_history"] == [96, 192]
    assert training["sampler"] == "Following"  # every real training takes its order from the plan
    public = approved(third)
    assert (public["parts"], public["optimizer_steps"], public["train"], public["monitored"]) == (3, 192, 96, 24)
    assert public["adapter_sha256"] == policy["adapter_sha256"] and public["follows_plan"] is True

    # The same seed, one process, the trainer's sampler untouched: the order and the validation
    # passes a training in parts must reproduce. The stand-in's adapter depends on both.
    whole = tmp_path / "uninterrupted"
    _, result, schedule = train.fit(backbone.BACKBONES["nemotron"], 271, data, whole, lambda step, total: False, reference=True)
    assert tree_digest(whole / "adapter")[0] == policy["adapter_sha256"]
    assert result["validation_history"] == [96, 192] and schedule.sequence == schedule.plan["steps"]
    assert result["sampler"] == "RepeatRandomSampler"  # the reference trains on the sampler as TRL built it
    assert not list(whole.glob("checkpoint-96"))  # no boundary checkpoint: nothing paused it


@cpu_stack
@pytest.mark.parametrize("replays", [False, True])
def test_a_resumed_part_continues_the_plan_whatever_the_trainer_does_with_finished_epochs(tmp_path, monkeypatch, replays):
    data = data_directory(tmp_path, monkeypatch)
    reference = tmp_path / "uninterrupted"
    monkeypatch.setattr(gpu, "train", fake_train())
    train.fit(backbone.BACKBONES["nemotron"], 273, data, reference, lambda step, total: False, reference=True)

    stops = iter([None, 130, 171, None, None])  # after the boundary: two warnings inside epoch two

    def stage():
        ctx = context(tmp_path, "E8_train_273", ["--backbone", "nemotron"], {"data": data}, {"training": 273})
        stop_at = next(stops)
        monkeypatch.setattr(gpu, "train", fake_train(lambda step: step == stop_at and setattr(ctx, "stop", True),
                                                     replays_epochs_on_resume=replays))
        return ctx

    ctx, processes = until_done(stage, train.run)
    policy = json.loads((ctx.out_dir / "policy.json").read_text())
    assert processes == 4 and policy["paused_at"] == [96, 130, 171]
    assert policy["adapter_sha256"] == tree_digest(reference / "adapter")[0]


@cpu_stack
def test_a_training_that_leaves_its_plan_is_stopped_and_never_published(tmp_path, monkeypatch):
    data = data_directory(tmp_path, monkeypatch)

    def stage():
        return context(tmp_path, "E8_train_274", ["--backbone", "nemotron"], {"data": data}, {"training": 274})

    monkeypatch.setattr(gpu, "train", fake_train(stray_at=120))  # one step trains on another admission
    with pytest.raises(StageInterrupted):
        train.run(stage())
    with pytest.raises(plan.Departed, match="optimizer step 120 "):
        train.run(stage())
    assert not (stage().out_dir / "policy.json").exists() and not (stage().out_dir / "model" / "adapter").exists()
    with pytest.raises(plan.Departed):  # and a retry meets the same refusal
        train.run(stage())


@cpu_stack
def test_a_training_whose_record_misses_a_step_is_not_published(tmp_path, monkeypatch):
    data = data_directory(tmp_path, monkeypatch)
    monkeypatch.setattr(gpu, "train", fake_train(unobserved_at=192))  # the last step leaves no record

    def stage():
        return context(tmp_path, "E8_train_275", ["--backbone", "nemotron"], {"data": data}, {"training": 275})

    with pytest.raises(StageInterrupted):
        train.run(stage())
    with pytest.raises(plan.Departed, match="does not match its plan"):
        train.run(stage())
    assert not (stage().out_dir / "policy.json").exists()


@cpu_stack
def test_training_that_stops_short_of_its_budget_is_not_published(tmp_path, monkeypatch):
    data = data_directory(tmp_path, monkeypatch)
    ctx = context(tmp_path, "E8_train_272", ["--backbone", "nemotron"], {"data": data}, {"training": 272})

    def short(cfg, output_dir, data_dir, seed):
        callback = gpu._environment_rng_callback(CallbackBase, SimpleNamespace(_rng=random.Random()), "mimic")
        callback.on_train_end(SimpleNamespace(**ARGS), SimpleNamespace(global_step=150, max_steps=192), SimpleNamespace())
        return {}

    monkeypatch.setattr(gpu, "train", short)
    with pytest.raises(RuntimeError, match="fixed budget"):
        train.run(ctx)
    assert not (ctx.out_dir / "policy.json").exists()
