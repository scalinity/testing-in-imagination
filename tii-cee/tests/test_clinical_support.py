"""Shared stand-ins for the clinical tests. Everything here is invented; no patient data exists locally.

Two tiers of tests use these:
  - numpy-only tests run in tii-cee's own environment (`uv run pytest -q tests/test_clinical_*.py`);
  - tests marked `cpu_stack` drive the replication's and tim's real CPU code (Hydra composition,
    the upstream environment, tim's evaluator and scripted policy) and need their packages. They
    are skipped in tii-cee's environment and run with tim's:
        cd tii-cee && ../cdm-extension/daniel-escalante/.venv/bin/python -m pytest -q tests/test_clinical_*.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tiicee.clinical import ROOT
from tiicee.pipeline.paths import Paths
from tiicee.pipeline.stage import Context
from tiicee.pipeline.stages import StageInterrupted

CPU_STACK = ("hydra", "omegaconf", "pandas", "yaml", "torch")
cpu_stack = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in CPU_STACK),
    reason="needs the replication's CPU packages; run with cdm-extension/daniel-escalante/.venv/bin/python -m pytest")

COLUMNS = ["Patient ID", "Patient History", "Patient History Summary", "Physical Examination",
           "Laboratory Tests", "Radiology", "Label"]
CLASSES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
MIX = (957, 648, 257, 538)  # the cohort's published class sizes, used here only as proportions


def labels(count):
    """`count` labels in roughly the cohort's proportions, in a fixed shuffled order."""
    drawn = [CLASSES[i] for i in range(4) for _ in range(round(count * MIX[i] / sum(MIX)))]
    drawn = (drawn + [CLASSES[0]] * count)[:count]
    random.Random(7).shuffle(drawn)
    return drawn


def split_bytes(first_id, split_labels) -> bytes:
    """A split file as pandas writes one (minimal quoting, LF line ends), with awkward field contents."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for i, label in enumerate(split_labels):
        writer.writerow([first_id + i, f"Invented history {first_id + i}, with a comma",
                         f"Invented summary {first_id + i}\nsecond line \"quoted\" café \x0c   end",
                         "Invented examination.\r\n", "{50001: '1.0 invented units'}", "[]", label])
    return buffer.getvalue().encode("utf-8")


def prepared(tmp_path, train=96, val=24, with_test=True) -> Path:
    """A stand-in for the baseline work directory's prepared/ folder."""
    directory = tmp_path / "baseline" / "prepared"
    directory.mkdir(parents=True)
    (directory / "train.csv").write_bytes(split_bytes(1000, labels(train)))
    (directory / "val.csv").write_bytes(split_bytes(50000, labels(val)))
    (directory / "lab_test_mapping.csv").write_bytes(b"itemid,label,corresponding_ids\n50001,Invented measure,[50001]\n")
    if with_test:
        (directory / "test.csv").write_bytes(split_bytes(90000, labels(val)))
    return directory


def context(tmp_path, stage_id, args=(), bound=None, seeds=None) -> Context:
    """A real stage context over a temporary storage root, as run_stage would build it."""
    paths = Paths(root=tmp_path / "tii", code=ROOT)
    paths.stage_dir(stage_id).mkdir(parents=True, exist_ok=True)
    manifest = {"manifest_id": "manifest-" + stage_id, "seeds": dict(seeds or {}),
                "bound": {role: {"path": str(path)} for role, path in (bound or {}).items()}}
    ctx = Context(paths, {"id": stage_id, "command": ["stage", *args]}, manifest)
    ctx.out_dir.mkdir(exist_ok=True)
    return ctx


def _no_git(monkeypatch):
    """On the cluster a stage runs from a code export that is not a git repository
    (tiicee/pipeline/deploy.py), so no clinical stage may call git."""
    run = subprocess.run

    def refusing(args, *rest, **options):
        assert args[0] != "git", "a clinical stage called git"
        return run(args, *rest, **options)

    monkeypatch.setattr(subprocess, "run", refusing)


def training_environment(monkeypatch):
    """The job environment a training stage runs in (backbone.TRAINING_ENVIRONMENT), for one test."""
    _no_git(monkeypatch)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "recorded for restoring")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")


def rollout_environment(monkeypatch):
    """The job environment a rollout stage runs in (backbone.ROLLOUT_ENVIRONMENT), for one test."""
    _no_git(monkeypatch)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def approved(ctx) -> dict:
    return json.loads((ctx.paths.approved / (ctx.stage_id + ".json")).read_text())


def until_done(make_ctx, run, limit=10):
    """Run a stage as the watcher does: a new process, here a new context, after every continue
    code. Returns the finishing context and how many processes it took."""
    for processes in range(1, limit + 1):
        ctx = make_ctx()
        try:
            run(ctx)
        except StageInterrupted:
            continue
        return ctx, processes
    raise AssertionError("the stage never finished")


class CallbackBase:
    """transformers.TrainerCallback's shape: every event is a no-op."""

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def on_save(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        pass


class RepeatRandomSampler:
    """TRL 0.15.2's training sampler as far as it matters here, written from memory, not from its
    source: one generator seeded when the sampler is built, a new permutation at every iteration,
    each index repeated once per rollout of its group. The cluster bench checks the real one."""

    def __init__(self, data_source, repeat_count, seed):
        self.num_samples, self.repeat_count, self.generator = len(data_source), repeat_count, random.Random(seed)

    def __iter__(self):
        order = list(range(self.num_samples))
        self.generator.shuffle(order)
        return iter([index for index in order for _ in range(self.repeat_count)])

    def __len__(self):
        return self.num_samples * self.repeat_count


class TrainerBase:
    """transformers.Trainer's training loop with TRL's sampler and upstream's one-group batches,
    as far as the order of data, checkpoints, validation passes and callback events goes.

    A process builds its sampler once and iterates it once per epoch. On resume it skips the
    batches already trained in the current epoch; replays_epochs_on_resume says whether it first
    begins one iteration per finished epoch, as older Trainers did. Faults a test can plant:
    stray_at makes one step train on another admission than the sampler gave; unobserved_at makes
    one step train without prompting through the environment; order_salt gives the sampler another
    seed. The state carries the sampler's class name and a digest chained over the admissions
    trained on, so tests can see both through what the run leaves behind.
    """

    replays_epochs_on_resume, stray_at, unobserved_at, order_salt = False, None, None, 0

    def __init__(self, train_dataset, environment, callback, args, at_step=lambda step: None):
        self.train_dataset, self.environment, self.callback, self.args = train_dataset, environment, callback, args
        self.at_step = at_step
        self.state = SimpleNamespace(global_step=0, max_steps=0, validations=[], digest="", sampler=None)

    def _get_train_sampler(self):
        return RepeatRandomSampler(self.train_dataset, self.args.num_generations, self.args.seed + self.order_salt)

    def _save(self, control):
        if control.should_save:
            checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            checkpoint.mkdir(exist_ok=True)
            (checkpoint / "trainer_state.json").write_text(json.dumps(vars(self.state)))
            self.callback.on_save(self.args, self.state, control)

    def train(self, resume_from_checkpoint=None):
        sampler = self._get_train_sampler()  # get_train_dataloader, once per process
        group, epochs = self.args.num_generations, int(self.args.num_train_epochs)
        per_epoch = len(sampler) // group
        if resume_from_checkpoint:
            self.state = SimpleNamespace(**json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text()))
        self.state.max_steps, self.state.sampler = epochs * per_epoch, type(sampler).__name__
        first, skip = divmod(self.state.global_step, per_epoch)
        if self.replays_epochs_on_resume:
            for _ in range(first):
                iter(sampler)
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        for _ in range(first, epochs):
            order = list(sampler)  # iter(epoch_dataloader)
            batches, skip = [order[i:i + group] for i in range(0, len(order), group)][skip:], 0
            for batch in batches:
                control.should_save = False  # on_step_begin
                if self.state.global_step + 1 == self.stray_at:
                    batch = [(index + 1) % len(self.train_dataset) for index in batch]
                inputs = [dict(self.train_dataset[index]) for index in batch]
                if self.state.global_step + 1 != self.unobserved_at:
                    self.environment.construct_prompt(inputs)  # training_step, through _prepare_inputs
                self.environment._rng.random()
                self.state.digest = hashlib.sha256(f"{self.state.digest}{inputs[0]['patient_id']}".encode()).hexdigest()
                self.state.global_step += 1
                self.at_step(self.state.global_step)
                last = self.state.global_step == self.state.max_steps
                control.should_save = self.state.global_step % 100 == 0 or last  # DefaultFlowCallback
                control.should_training_stop = last
                control = self.callback.on_step_end(self.args, self.state, control) or control
                self._save(control)
                if control.should_training_stop:
                    break
            control = self.callback.on_epoch_end(self.args, self.state, control) or control
            self.state.validations.append(self.state.global_step)  # eval_strategy "epoch": evaluate, then save
            self._save(control)
            if control.should_training_stop:
                break
        self.callback.on_train_end(self.args, self.state, control)


def fake_train(at_step=lambda step: None, **behaviour):
    """A stand-in for replication.gpu.train with gpu.train's own structure: its configuration
    step, then the environment, the callback and the trainer class, each built through the module
    attribute gpu.train uses (prepare_environment, _environment_rng_callback, _trainer_class), the
    environment RNG restored from the last checkpoint, training, and an adapter saved at the end.

    The training loop is TrainerBase; behaviour sets its replays_epochs_on_resume or stray_at. The
    adapter is a function of the environment RNG and of the order of admissions trained on, so a
    run in parts only leaves the same adapter as an uninterrupted one if the sidecar was restored
    and the order was the same."""

    def train(cfg, output_dir, data_dir, seed):
        from replication import gpu
        from tiicee.clinical import admissions
        cfg = gpu._configuration(cfg, data_dir, seed, output_dir)  # the replication's own allowlist check
        assert not (Path(data_dir) / "test.csv").exists()
        output_dir = Path(output_dir)
        dataset = [{"patient_id": a.patient_id, "patient_history": "invented"}
                   for a in admissions.read((Path(data_dir) / "train.csv").read_bytes()).admissions]
        environment = gpu.prepare_environment(cfg)
        callback = gpu._environment_rng_callback(CallbackBase, environment, "mimic")
        args = SimpleNamespace(output_dir=str(output_dir), seed=seed, num_generations=cfg.training.num_generations,
                               num_train_epochs=cfg.training.num_train_epochs,
                               gradient_accumulation_steps=cfg.training.gradient_accumulation_steps)
        trainer = gpu._trainer_class(type("Base", (TrainerBase,), behaviour), cfg, seed)(
            dataset, environment, callback, args, at_step)
        found = sorted(int(path.name.split("-")[1]) for path in output_dir.glob("checkpoint-*"))
        checkpoint = output_dir / f"checkpoint-{found[-1]}" if found else None
        if checkpoint:
            gpu._restore_environment_rng(environment, checkpoint, "mimic")
        trainer.train(resume_from_checkpoint=checkpoint)
        adapter = output_dir / "adapter"
        adapter.mkdir(exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(
            (repr(environment._rng.getstate()) + trainer.state.digest).encode())
        (adapter / "adapter_config.json").write_text(json.dumps(
            {"base_model_name_or_path": cfg.model.model_name, "revision": cfg.model.revision}))
        return {"adapter_dir": str(adapter), "optimizer": "Adam", "nan_metric": float("nan"),
                "validation_history": list(trainer.state.validations), "sampler": trainer.state.sampler}

    return train


def fake_evaluate(policy):
    """A stand-in for replication.gpu.evaluate: its configuration step and split loading, then the
    module's _evaluate_with_model (which the rollout replaces) over a CPU policy."""

    def evaluate(cfg, data_dir, split, seed, adapter_dir=None, temperature=None):
        from replication import gpu
        from replication.config import stream_seed
        from replication.shim import generation_settings
        from src.dataset.mimic_cdm_dataset import MIMICCDMDataset
        assert sorted(path.name for path in Path(data_dir).iterdir()) == ["lab_test_mapping.csv", f"{split}.csv"]
        cfg = gpu._configuration(cfg, data_dir, stream_seed(seed, split, "evaluation"))
        d = cfg.dataset
        dataset = MIMICCDMDataset(data_file=d[f"{split}_data_file"], lab_test_mapping_file=d.lab_test_mapping_file,
                                  lab_tests=list(d.lab_tests), imaging_tests=list(d.imaging_tests),
                                  other_tests=list(d.other_tests))
        return gpu._evaluate_with_model(cfg, dataset, policy, generation_settings(False))

    return evaluate
