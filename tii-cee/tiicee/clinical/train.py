"""Training of one LA-CDM policy by the replication's own launcher, in resumable parts (E8, E9, E10).

Stage: clinical_train --backbone nemotron|granite
Manifest: role "data" (a directory holding train.csv, val.csv and lab_test_mapping.csv, and no
test.csv) and seed "training". Job environment: the replication's launcher's, verified at start
(backbone.TRAINING_ENVIRONMENT).

The work is replication.gpu.train, unchanged. It resumes by itself: it takes transformers'
get_last_checkpoint of the output directory and refuses a checkpoint that lacks the
environment-RNG sidecar. Three attributes of that module are replaced while it runs:

  gpu._environment_rng_callback (gpu.py:117-124, used at gpu.py:353), the function that builds the
    trainer's only callback. The replacement returns that same callback with additions. A part
    ends at a checkpoint in two cases: at the end of an epoch that is not the last, after the
    epoch's validation pass (the planned part boundary); and at the next step end once the stage's
    stop flag is set (Slurm's warning, when an epoch does not fit in a part). In both the
    checkpoint is saved, the inherited on_save writes the sidecar, and Paused is raised. The stage
    checks that gpu.train's own reader accepts the checkpoint and exits with the continue code;
    the next part calls gpu.train again and resumes from it.
  gpu._trainer_class and gpu.prepare_environment: see tiicee.clinical.plan. The order of
    admissions is fixed before the first step, every step is checked against it, and a resumed
    part continues it. A training that departs from its plan is not published.

Outputs, all private: model/ (plan.json, checkpoints and the final adapter), policy.json (recipe
arm, backbone and revision, seed, data-file digests, adapter tree digest, visit counts),
training.json (what gpu.train returned).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiicee.clinical import admissions, bound, remember, sha256, use_prior_studies
from tiicee.clinical.backbone import (ARM, BACKBONES, TRAINING_ENVIRONMENT, chosen, compose, recipe_record,
                                       require_environment, require_frozen_code, runtime)
from tiicee.clinical.plan import Schedule, observed, planned
from tiicee.pipeline.stages import StageInterrupted, register
from tiicee.pipeline.util import tree_digest

RNG_SOURCE = "mimic"  # the source name gpu.train gives its sidecar writer and reader


class Paused(Exception):
    """A part of training ended at a checkpoint to resume from."""

    def __init__(self, checkpoint, step):
        super().__init__(f"paused at step {step}")
        self.checkpoint, self.step = Path(checkpoint), step


def stop_aware(make_callback, on_step, made, saving=lambda step: None, pause_at_epochs=True):
    """gpu._environment_rng_callback, with the two ways a part ends (see the module docstring).
    on_step(step, total) is called at every step end and returns whether to stop; saving(step) is
    called for every checkpoint, before the sidecar that marks it complete."""

    def build(base, environment, source):
        class StopAware(type(make_callback(base, environment, source))):
            forced = final_step = max_steps = None

            def on_step_end(self, args, state, control, **kwargs):
                step, total = state.global_step, state.max_steps
                boundary = pause_at_epochs and step < total and step % (total // round(args.num_train_epochs)) == 0
                # At a boundary the epoch-end pause below follows the validation pass; stopping
                # here instead would end the part before that pass and the resume would skip it.
                if on_step(step, total) and not boundary:
                    control.should_save = control.should_training_stop = True
                    self.forced = state.global_step
                return control

            def on_epoch_end(self, args, state, control, **kwargs):
                if pause_at_epochs and state.global_step < state.max_steps:
                    control.should_save = True  # the Trainer evaluates first, then saves
                    self.forced = state.global_step
                return control

            def on_save(self, args, state, control, **kwargs):
                saving(state.global_step)
                control = super().on_save(args, state, control, **kwargs)  # writes environment-rng.json
                if self.forced == state.global_step:
                    # A mid-epoch part ends here rather than through should_training_stop: a stopped
                    # loop still runs the epoch-end validation pass, which outlasts Slurm's warning.
                    raise Paused(Path(args.output_dir) / f"checkpoint-{state.global_step}", state.global_step)
                return control

            def on_train_end(self, args, state, control, **kwargs):
                self.final_step, self.max_steps = state.global_step, state.max_steps
                return control

        made.append(StopAware())
        return made[-1]

    return build


def visits(sequence, trained_ids, epochs) -> dict:
    """How often each training admission was trained on, against the recipe's number of epochs."""
    counts = Counter(sequence)
    each = [counts[patient_id] for patient_id in trained_ids]
    return {"visits_min": min(each), "visits_max": max(each),
            "visits_as_recipe": set(each) == {epochs} and set(counts) <= set(trained_ids)}


def resumable(gpu, checkpoint) -> bool:
    """Whether gpu.train can resume from this checkpoint: the trainer state is readable and
    gpu.train's own sidecar reader accepts the environment RNG state."""
    try:
        json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        gpu._restore_environment_rng(SimpleNamespace(_rng=random.Random()), checkpoint, RNG_SOURCE)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    return True


def last_checkpoint(gpu, output_dir):
    """The checkpoint gpu.train will resume from, or None. Checkpoints that a kill left incomplete
    are renamed first, newest first: gpu.train always takes the newest, so a torn one would be
    refused on every retry."""
    found = sorted((int(path.name.split("-")[1]), path) for path in Path(output_dir).glob("checkpoint-*")
                   if path.is_dir() and path.name.split("-")[1].isdigit())
    for step, path in reversed(found):
        if resumable(gpu, path):
            return path
        path.rename(path.with_name(f"torn-{step}-{len(list(Path(output_dir).glob('torn-*')))}"))
    return None


def inspect_data(data_dir) -> dict:
    """Digests and counts of a training directory, after refusing one that could leak."""
    data_dir = Path(data_dir)
    if (data_dir / "test.csv").exists():
        raise ValueError("a training directory must not hold test.csv")
    parts = ["train", "val"] + (["heldout"] if (data_dir / "heldout.csv").exists() else [])
    names = (*(f"{part}.csv" for part in parts), "lab_test_mapping.csv")
    data = {name: (data_dir / name).read_bytes() for name in names}
    members = {part: admissions.read(data[f"{part}.csv"]).admissions for part in parts}
    # No admission is trained on and monitored, and none of a fold's held-out admissions is either.
    admissions.require_unique([a for part in parts for a in members[part]])
    return {"files": {name: sha256(content) for name, content in data.items()},
            "admissions": {part: len(members[part]) for part in parts},
            "trained_ids": [a.patient_id for a in members["train"]]}


def hooks(ctx, pause_at=0):
    """A stage's part in training: at every step, report progress and say whether to stop (the
    stage's stop flag, or the step a bench chose to pause at); at every checkpoint, move the
    progress mark the watcher reads."""

    def on_step(step, total):
        ctx.progress(int(step), int(total), "step")
        return ctx.stop or step == pause_at

    return on_step, lambda step: remember(ctx, saved_step=step)


def fit(backbone, seed, data_dir, output_dir, on_step, saved=lambda step: None, reference=False):
    """One part of training. Returns (configuration, gpu.train's result, schedule) when the fixed
    budget is complete and the whole training followed its plan; raises Paused when the part ended
    at a checkpoint gpu.train can resume from; raises plan.Departed when the training left its plan.

    reference=True is the bench's uninterrupted run: no part boundary, and the trainer's sampler
    left exactly as TRL builds it, so the plan is checked against the unmodified mechanics.
    """
    require_environment(TRAINING_ENVIRONMENT)
    use_prior_studies()
    from replication import gpu
    require_frozen_code()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = compose(backbone, data_dir, output_dir, seed)
    schedule = Schedule(output_dir, last_checkpoint(gpu, output_dir))

    def saving(step):
        schedule.save(output_dir / f"checkpoint-{step}")
        saved(step)

    made = []
    callback = stop_aware(gpu._environment_rng_callback, on_step, made, saving, pause_at_epochs=not reference)
    with runtime(backbone, cfg), patch.object(gpu, "_environment_rng_callback", callback), \
            patch.object(gpu, "prepare_environment", observed(gpu.prepare_environment, schedule)), \
            patch.object(gpu, "_trainer_class", planned(gpu._trainer_class, schedule, follow=not reference)):
        try:
            result = gpu.train(cfg, output_dir=output_dir, data_dir=data_dir, seed=seed)
        except Paused as paused:
            if not resumable(gpu, paused.checkpoint):
                raise RuntimeError("the checkpoint a part ended at cannot be resumed from") from None
            raise
    if not made or made[-1].final_step is None or made[-1].final_step != made[-1].max_steps:
        raise RuntimeError("training ended before its fixed budget")
    schedule.finish(made[-1].final_step)
    return cfg, result, schedule


def _json(path, value):
    """A private record. Metrics may be numpy scalars or NaN, and must never cost a finished training."""
    text = json.dumps(value, sort_keys=True, indent=2,
                      default=lambda item: float(item) if hasattr(item, "__float__") else str(item))
    Path(path).write_text(text + "\n")


@register("clinical_train")
def run(ctx):
    parser = argparse.ArgumentParser(prog="clinical_train")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    backbone = chosen(ctx, parser.parse_args(ctx.args).backbone)
    data_dir, seed = bound(ctx, "data"), ctx.manifest["seeds"]["training"]
    data = inspect_data(data_dir)
    paused_at = (ctx.load_checkpoint() or {}).get("paused_at", [])
    try:
        cfg, result, schedule = fit(backbone, seed, data_dir, ctx.out_dir / "model", *hooks(ctx))
    except Paused as paused:
        remember(ctx, paused_at=paused_at + [paused.step])
        raise StageInterrupted() from None
    digest, files = tree_digest(ctx.out_dir / "model" / "adapter")
    steps = len(schedule.sequence)
    followed = {"follows_plan": True,  # fit returns only for a training that matched its plan throughout
                **visits(schedule.sequence, data["trained_ids"], schedule.plan["epochs"])}
    _json(ctx.out_dir / "training.json", {**result, "adapter_dir": "model/adapter"})
    _json(ctx.out_dir / "policy.json", {
        **recipe_record(cfg, backbone, seed), "data": data["files"], "admissions": data["admissions"],
        "optimizer_steps": steps, "paused_at": paused_at, **followed,
        "plan_sha256": sha256(json.dumps(schedule.plan["steps"]).encode()), "adapter": "model/adapter",
        "adapter_sha256": digest, "adapter_files": files, "manifest_id": ctx.manifest["manifest_id"]})
    ctx.approve(ctx.stage_id + ".json", {
        "stage": ctx.stage_id, "recipe": ARM, "backbone": backbone.key, "seed": seed,
        "optimizer_steps": steps, "parts": len(paused_at) + 1, **followed, "adapter_sha256": digest,
        "train": data["admissions"]["train"], "monitored": data["admissions"]["val"]})
