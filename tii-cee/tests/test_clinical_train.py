"""The training stage: forced save and stop on the stage's flag, resume, refusals, and what it binds."""
import json
import random
from types import SimpleNamespace

import pytest

from test_clinical_support import CallbackBase, approved, context, cpu_stack, fake_train, prepared, split_bytes
from tiicee.clinical import folds, sha256, train, use_prior_studies
from tiicee.pipeline.stages import StageInterrupted
from tiicee.pipeline.util import tree_digest

use_prior_studies()
from replication import gpu  # noqa: E402  (imports with the standard library alone)


def trainer_step(callback, output_dir, step, due=False):
    """One Trainer step as transformers orders it: on_step_end, then the checkpoint and on_save."""
    args, state = SimpleNamespace(output_dir=str(output_dir)), SimpleNamespace(global_step=step, max_steps=1000)
    control = SimpleNamespace(should_save=due, should_training_stop=False)
    control = callback.on_step_end(args, state, control) or control
    if control.should_save:
        checkpoint = output_dir / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        callback.on_save(args, state, control)
    return control


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
        callback.on_step_end(SimpleNamespace(), SimpleNamespace(global_step=step, max_steps=20), control)
        assert control.should_save is expected and control.should_training_stop is expected


def test_the_callback_records_where_training_ended():
    build = train.stop_aware(gpu._environment_rng_callback, lambda step, total: False, [])
    callback = build(CallbackBase, SimpleNamespace(), "mimic")
    callback.on_train_end(SimpleNamespace(), SimpleNamespace(global_step=3840, max_steps=3840), SimpleNamespace())
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

    sequence = []
    environment = train.counting(lambda cfg: Environment(), sequence)("the configuration")
    group = [{"patient_id": 77, "patient_history": "invented"}] * 8  # one admission, eight rollouts
    assert environment.construct_prompt(group) == [dict(item, prompt="built") for item in group]
    environment.construct_prompt([{"patient_id": 78}])
    environment.construct_prompt([{"no id": 1}])  # the observer never raises into training
    assert sequence == [[77], [78], None]


def test_visit_counts_are_checked_against_the_recipes_epochs():
    ids = [1, 2, 3]
    twice = [[1], [2], [3], [3], [1], [2]]
    assert train.visits(twice, 6, ids, 2) == {"visits_min": 2, "visits_max": 2, "visits_as_recipe": True}
    skewed = [[1], [2], [3], [1], [1], [2]]  # what a resumed epoch in the wrong order looks like
    assert train.visits(skewed, 6, ids, 2) == {"visits_min": 1, "visits_max": 3, "visits_as_recipe": False}
    assert train.visits(twice + [[9]] * 2, 8, ids, 2)["visits_as_recipe"] is False  # an admission from elsewhere
    unverified = {"visits_min": None, "visits_max": None, "visits_as_recipe": None}
    assert train.visits(None, 6, ids, 2) == unverified                 # a checkpoint without its record
    assert train.visits(twice[:5], 6, ids, 2) == unverified            # the record misses a step
    assert train.visits(twice[:5] + [[1, 2]], 6, ids, 2) == unverified  # a step that mixed admissions
    assert train.visits(twice[:5] + [None], 6, ids, 2) == unverified


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
def test_training_pauses_on_the_flag_resumes_and_binds_the_policy(tmp_path, monkeypatch):
    data = data_directory(tmp_path, monkeypatch)  # 96 training admissions: 192 optimizer steps in two epochs

    def stage(stage_id):
        return context(tmp_path, stage_id, ["--backbone", "nemotron"], {"data": data}, {"training": 271})

    first = stage("E8_train_271")
    monkeypatch.setattr(gpu, "train", fake_train(lambda step: step == 150 and setattr(first, "stop", True)))
    with pytest.raises(StageInterrupted):  # the stage runner turns this into the continue code
        train.run(first)
    model = first.out_dir / "model"
    assert train.resumable(gpu, model / "checkpoint-150") and not (first.out_dir / "policy.json").exists()
    assert first.load_checkpoint() == {"saved_step": 150, "paused_at": [150]}
    assert len(json.loads((model / "checkpoint-150" / "visits.json").read_text())) == 150

    second = stage("E8_train_271")  # the next Slurm part: a new process over the same stage directory
    monkeypatch.setattr(gpu, "train", fake_train())
    train.run(second)
    policy = json.loads((second.out_dir / "policy.json").read_text())
    assert (policy["recipe"], policy["seed"], policy["optimizer_steps"], policy["paused_at"]) == ("primary", 271, 192, [150])
    assert policy["backbone"] == "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    assert policy["revision"] == "54641c1611fcff44fa4865626462445e0a153fc7"
    assert policy["data"] == {name: sha256((data / name).read_bytes()) for name in ("train.csv", "val.csv", "lab_test_mapping.csv")}
    assert policy["admissions"] == {"train": 96, "val": 24} and "trained_ids" not in policy
    assert policy["adapter_sha256"] == tree_digest(second.out_dir / policy["adapter"])[0] and policy["adapter_files"] == 2
    assert policy["manifest_id"] == "manifest-E8_train_271" and policy["config"]["training"]["seed"] == 271
    assert policy["config"]["dataset"]["train_data_file"] == str(data / "train.csv")
    assert set(policy["templates"]) == {"decision_agent.txt", "hypothesis_agent.txt"}
    assert (policy["visits_min"], policy["visits_max"], policy["visits_as_recipe"]) == (2, 2, True)
    assert json.loads((second.out_dir / "training.json").read_text())["adapter_dir"] == "model/adapter"
    public = approved(second)
    assert (public["parts"], public["optimizer_steps"], public["train"], public["monitored"]) == (2, 192, 96, 24)
    assert public["adapter_sha256"] == policy["adapter_sha256"] and public["visits_as_recipe"] is True

    whole = stage("E8_train_271_uninterrupted")
    train.run(whole)  # same seed, never paused: the resumed run must have reached the same policy
    assert json.loads((whole.out_dir / "policy.json").read_text())["adapter_sha256"] == policy["adapter_sha256"]


@cpu_stack
def test_a_resume_that_restarts_the_epoch_order_is_measured_and_reported(tmp_path, monkeypatch):
    data = data_directory(tmp_path, monkeypatch)
    first = context(tmp_path, "E8_train_273", ["--backbone", "nemotron"], {"data": data}, {"training": 273})
    lost_order = dict(replays_epochs_on_resume=False)  # the sampler's generator starts again and nothing replays it
    monkeypatch.setattr(gpu, "train", fake_train(lambda step: step == 150 and setattr(first, "stop", True), **lost_order))
    with pytest.raises(StageInterrupted):
        train.run(first)
    second = context(tmp_path, "E8_train_273", ["--backbone", "nemotron"], {"data": data}, {"training": 273})
    monkeypatch.setattr(gpu, "train", fake_train(**lost_order))
    train.run(second)
    public = approved(second)
    assert public["optimizer_steps"] == 192 and public["visits_as_recipe"] is False
    assert public["visits_min"] < 2 < public["visits_max"]  # some admissions trained on once, others three times


@cpu_stack
def test_training_that_stops_short_of_its_budget_is_not_published(tmp_path, monkeypatch):
    data = data_directory(tmp_path, monkeypatch)
    ctx = context(tmp_path, "E8_train_272", ["--backbone", "nemotron"], {"data": data}, {"training": 272})

    def short(cfg, output_dir, data_dir, seed):
        callback = gpu._environment_rng_callback(CallbackBase, SimpleNamespace(_rng=random.Random()), "mimic")
        callback.on_train_end(SimpleNamespace(), SimpleNamespace(global_step=150, max_steps=192), SimpleNamespace())
        return {}

    monkeypatch.setattr(gpu, "train", short)
    with pytest.raises(RuntimeError, match="fixed budget"):
        train.run(ctx)
    assert not (ctx.out_dir / "policy.json").exists()
