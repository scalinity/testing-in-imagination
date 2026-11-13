"""Training of one LA-CDM policy by the replication's own launcher, in resumable parts (E8, E9, E10).

Stage: clinical_train --backbone nemotron|granite
Manifest: role "data" (a directory holding train.csv, val.csv and lab_test_mapping.csv, and no
test.csv) and seed "training".

The work is replication.gpu.train, unchanged. It resumes by itself: it takes transformers'
get_last_checkpoint of the output directory and refuses a checkpoint that lacks the
environment-RNG sidecar. Two attributes of that module are replaced while it runs:

  gpu._environment_rng_callback, the function that builds the trainer's only callback. The
    replacement returns that same callback with additions: at a step end with the stage's stop
    flag set it asks for a save and a stop, and once that save is on disk, sidecar included, it
    raises Paused. The stage checks that gpu.train's own reader accepts the forced checkpoint and
    exits with the continue code; the next part calls gpu.train again and resumes from it.
  gpu.prepare_environment, the function that builds the training environment. The replacement
    returns that same environment and notes the patient_id of every batch it is asked to prompt
    for, one batch per optimizer step. It changes nothing the trainer sees. The sequence is saved
    inside each checkpoint (visits.json) and read back on resume, so the finished policy can state
    how often each admission was trained on. The recipe says num_train_epochs times; a resume in
    the middle of an epoch is where that could fail, because the upstream sampler draws each
    epoch's order from one generator that a new process starts again.

Outputs, all private: model/ (checkpoints and the final adapter), policy.json (recipe arm,
backbone and revision, seed, data-file digests, adapter tree digest, visit counts), training.json
(what gpu.train returned).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiicee.clinical import admissions, bound, remember, sha256, use_prior_studies
from tiicee.clinical.backbone import (ALLOCATOR, ARM, BACKBONES, chosen, compose, recipe_record, require_frozen_code,
                                       runtime)
from tiicee.pipeline.stages import StageInterrupted, register
from tiicee.pipeline.util import tree_digest

RNG_SOURCE = "mimic"  # the source name gpu.train gives its sidecar writer and reader
VISITS = "visits.json"


class Paused(Exception):
    """Training stopped on the stage's stop flag, leaving a checkpoint to resume from."""

    def __init__(self, checkpoint, step):
        super().__init__(f"paused at step {step}")
        self.checkpoint, self.step = Path(checkpoint), step


def stop_aware(make_callback, on_step, made, saving=lambda step: None):
    """gpu._environment_rng_callback, with a forced save and stop when on_step(step, total) is true.
    on_step is called at every step end; saving(step) for every checkpoint, before the sidecar that
    marks it complete."""

    def build(base, environment, source):
        class StopAware(type(make_callback(base, environment, source))):
            forced = final_step = max_steps = None

            def on_step_end(self, args, state, control, **kwargs):
                if on_step(state.global_step, state.max_steps):
                    control.should_save = control.should_training_stop = True
                    self.forced = state.global_step
                return control

            def on_save(self, args, state, control, **kwargs):
                saving(state.global_step)
                control = super().on_save(args, state, control, **kwargs)  # writes environment-rng.json
                if self.forced == state.global_step:
                    # The part ends here rather than through should_training_stop: a stopped loop
                    # still runs the epoch-end validation pass, which outlasts Slurm's warning.
                    raise Paused(Path(args.output_dir) / f"checkpoint-{state.global_step}", state.global_step)
                return control

            def on_train_end(self, args, state, control, **kwargs):
                self.final_step, self.max_steps = state.global_step, state.max_steps
                return control

        made.append(StopAware())
        return made[-1]

    return build


def counting(prepare_environment, sequence):
    """gpu.prepare_environment, with the patient_ids of every training batch appended to `sequence`."""

    def build(cfg):
        environment = prepare_environment(cfg)
        construct_prompt = environment.construct_prompt

        def noting(inputs):
            try:
                sequence.append(sorted({int(item["patient_id"]) for item in inputs}))
            except (KeyError, TypeError, ValueError):
                sequence.append(None)  # an observer never interrupts training; the count goes unverified
            return construct_prompt(inputs)

        environment.construct_prompt = noting
        return environment

    return build


def visits(sequence, steps, trained_ids, epochs) -> dict:
    """How often each training admission was trained on, against the recipe's number of epochs.
    All None when the record does not cover every optimizer step with one admission each."""
    if sequence is None or len(sequence) != steps or any(batch is None or len(batch) != 1 for batch in sequence):
        return {"visits_min": None, "visits_max": None, "visits_as_recipe": None}
    counts = Counter(batch[0] for batch in sequence)
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
    progress mark."""

    def on_step(step, total):
        ctx.progress(int(step), int(total), "step")
        return ctx.stop or step == pause_at

    return on_step, lambda step: remember(ctx, saved_step=step)


def fit(backbone, seed, data_dir, output_dir, on_step, saved=lambda step: None):
    """One part of training. Returns (configuration, gpu.train's result, optimizer steps, visit
    sequence) when the fixed budget is complete; raises Paused when on_step(step, total) ended the
    part at a checkpoint gpu.train can resume from. saved(step) is called at every checkpoint."""
    os.environ.setdefault(*ALLOCATOR)
    use_prior_studies()
    from replication import gpu
    require_frozen_code()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = compose(backbone, data_dir, output_dir, seed)
    checkpoint = last_checkpoint(gpu, output_dir)
    sequence = [] if checkpoint is None else None
    if checkpoint is not None and (checkpoint / VISITS).is_file():
        sequence = json.loads((checkpoint / VISITS).read_text())

    def saving(step):
        if sequence is not None:
            (output_dir / f"checkpoint-{step}" / VISITS).write_text(json.dumps(sequence))
        saved(step)

    made = []
    callback = stop_aware(gpu._environment_rng_callback, on_step, made, saving)
    environment = counting(gpu.prepare_environment, [] if sequence is None else sequence)
    with runtime(backbone, cfg), patch.object(gpu, "_environment_rng_callback", callback), \
            patch.object(gpu, "prepare_environment", environment):
        try:
            result = gpu.train(cfg, output_dir=output_dir, data_dir=data_dir, seed=seed)
        except Paused as paused:
            if not resumable(gpu, paused.checkpoint):
                raise RuntimeError("the forced checkpoint cannot be resumed from") from None
            raise
    if not made or made[-1].final_step is None or made[-1].final_step != made[-1].max_steps:
        raise RuntimeError("training ended before its fixed budget")
    return cfg, result, made[-1].final_step, sequence


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
        cfg, result, steps, sequence = fit(backbone, seed, data_dir, ctx.out_dir / "model", *hooks(ctx))
    except Paused as paused:
        remember(ctx, paused_at=paused_at + [paused.step])
        raise StageInterrupted() from None
    digest, files = tree_digest(ctx.out_dir / "model" / "adapter")
    seen = visits(sequence, steps, data["trained_ids"], int(cfg.training.num_train_epochs))
    _json(ctx.out_dir / "training.json", {**result, "adapter_dir": "model/adapter"})
    _json(ctx.out_dir / "policy.json", {
        **recipe_record(cfg, backbone, seed), "data": data["files"], "admissions": data["admissions"],
        "optimizer_steps": steps, "paused_at": paused_at, **seen, "adapter": "model/adapter",
        "adapter_sha256": digest, "adapter_files": files, "manifest_id": ctx.manifest["manifest_id"]})
    ctx.approve(ctx.stage_id + ".json", {
        "stage": ctx.stage_id, "recipe": ARM, "backbone": backbone.key, "seed": seed,
        "optimizer_steps": steps, "parts": len(paused_at) + 1, **seen, "adapter_sha256": digest,
        "train": data["admissions"]["train"], "monitored": data["admissions"]["val"]})
