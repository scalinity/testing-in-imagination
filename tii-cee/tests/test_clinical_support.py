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
import importlib.util
import io
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from tiicee.clinical import ROOT
from tiicee.pipeline.paths import Paths
from tiicee.pipeline.stage import Context

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


def approved(ctx) -> dict:
    return json.loads((ctx.paths.approved / (ctx.stage_id + ".json")).read_text())


class CallbackBase:
    """transformers.TrainerCallback's shape: every event is a no-op."""

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_save(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        pass


def fake_train(at_step=lambda step: None, epochs=2, replays_epochs_on_resume=True):
    """A stand-in for replication.gpu.train that follows the Trainer's order of events: a step,
    on_step_end, the checkpoint when one is due or asked for, on_save, and on_train_end at the end.

    It builds its callback through gpu._environment_rng_callback and its environment through
    gpu.prepare_environment (the real upstream environment), resumes as gpu.train does, and trains
    on one admission per step in a seeded order per epoch. Like the upstream sampler, a new process
    starts its order generator again; replays_epochs_on_resume says whether the Trainer then
    replays the finished epochs' draws (so the order is right) or not (so a resumed epoch follows
    the first epoch's order). The adapter it writes is a function of the environment RNG, so a
    resumed run only matches an uninterrupted one if the sidecar was written and restored."""

    def train(cfg, output_dir, data_dir, seed):
        from replication import gpu
        from tiicee.clinical import admissions
        cfg = gpu._configuration(cfg, data_dir, seed, output_dir)  # the replication's own allowlist check
        assert not (Path(data_dir) / "test.csv").exists()
        output_dir = Path(output_dir)
        ids = [a.patient_id for a in admissions.read((Path(data_dir) / "train.csv").read_bytes()).admissions]
        environment = gpu.prepare_environment(cfg)
        callback = gpu._environment_rng_callback(CallbackBase, environment, "mimic")
        found = sorted(int(path.name.split("-")[1]) for path in output_dir.glob("checkpoint-*"))
        if found:
            gpu._restore_environment_rng(environment, output_dir / f"checkpoint-{found[-1]}", "mimic")
        args = SimpleNamespace(output_dir=str(output_dir))
        state = SimpleNamespace(global_step=found[-1] if found else 0, max_steps=epochs * len(ids))
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        first_epoch = state.global_step // len(ids)
        while state.global_step < state.max_steps:
            epoch, position = divmod(state.global_step, len(ids))
            draw = epoch if replays_epochs_on_resume else epoch - first_epoch
            patient = random.Random(f"{seed}:{draw}").sample(ids, len(ids))[position]
            environment.construct_prompt([{"patient_id": patient, "patient_history": "invented"} for _ in range(8)])
            environment._rng.random()
            state.global_step += 1
            at_step(state.global_step)
            control.should_save = state.global_step % 100 == 0 or state.global_step == state.max_steps
            control.should_training_stop = state.global_step == state.max_steps
            control = callback.on_step_end(args, state, control) or control
            if control.should_save:
                checkpoint = output_dir / f"checkpoint-{state.global_step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": state.global_step}))
                control = callback.on_save(args, state, control) or control
            if control.should_training_stop:
                break
        callback.on_train_end(args, state, control)
        adapter = output_dir / "adapter"
        adapter.mkdir(exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(repr(environment._rng.getstate()).encode())
        (adapter / "adapter_config.json").write_text(json.dumps(
            {"base_model_name_or_path": cfg.model.model_name, "revision": cfg.model.revision}))
        return {"adapter_dir": str(adapter), "optimizer": "Adam", "validation_history": [{"eval_ece": float("nan")}]}

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
