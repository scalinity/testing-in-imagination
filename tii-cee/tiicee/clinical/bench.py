"""Bench of a backbone under the frozen recipe, and of training in parts (EXPERIMENT_CARDS/E9.md,
"Bench and repairs"; RESEARCH_LEDGER.md, D-CLIN-RESUME).

Stage: clinical_bench --backbone nemotron|granite [--train-admissions N] [--val-admissions N]
                      [--pause-at STEP] [--reference | --no-reference]
Manifest: role "data" (as clinical_train) and seed "training". Job environment: the training
environment (backbone.TRAINING_ENVIRONMENT). The bench also evaluates, which runs with tim's cuBLAS
setting, so the stage puts that one variable in place itself for each of its processes, before
torch is loaded; train.fit and rollout.roll then verify the environment as they do in the real
stages.

The bench calls the same code as the real stages (train.fit, rollout.roll) on the first
admissions of train.csv and val.csv, with one seed throughout and two epochs as the recipe has.

Defaults per backbone, each sized to the specification's limit of 3 GPU-hours for a bench
(section 14.3). The estimates assume 24 s an optimizer step (E8's measured 25.6 h for 3,840 steps,
validation included), up to 20 s a validation or rollout episode (the replication's figure; tim
measured about 6 s) and five minutes to load the model in each process (assumed, not measured):
  nemotron  the resume mechanics, charged to the benches allowance. 60 training and 8 validation
            admissions, a pause at step 90, with the reference: two trainings of 120 steps
            (96 min), 64 validation episodes (21 min), five processes (25 min), the rollout
            (3 min). About 2.4 GPU-hours. A run reaches the da and ha_sft loss phases and one
            100-step checkpoint, which the resumed part crosses.
  granite   E9's bench of the backbone, charged to E9. 110 and 12 admissions, a pause at step 150,
            no reference (the resume mechanics do not depend on the backbone): one training of 220
            steps (88 min), passing through all three loss phases (100 da, 100 ha_sft, then
            ha_conf_cal), 48 validation episodes (16 min), four processes (20 min), the rollout
            (4 min). About 2.1 GPU-hours at Nemotron's speed; Granite's own speed is not known.

It runs as up to five processes, each ending with the continue code until the last, because no
two engines can share a process:
  1. the reference: the whole training, uninterrupted, with the trainer's sampler left exactly as
     TRL builds it and only observed (left out with --no-reference). A killed attempt starts over;
  2. the same training in parts, as clinical_train runs it: the first epoch, ending at the planned
     boundary after its validation pass;
  3. the second epoch as far as --pause-at, where the stage behaves as if Slurm's warning had
     arrived: forced checkpoint, exit (0 disables this pause);
  4. the rest of the training;
  5. the rollout of the bench adapter, and the report.
A weight-synchronization fault, a CUDA out-of-memory fault or a departure from the training plan
is the bench's result, recorded and reported, not a crash; any other error fails the stage.

Report (bench.json, also approved for export; counts and flags only):
  reference, training   outcome, optimizer steps, follows_plan, visits per admission, where the
                        training paused, peak GPU memory as torch's allocator saw it in the
                        process that finished
  sequences_identical   the training in parts trained on the same admission at every optimizer
                        step as the uninterrupted reference
  visits_identical      and on every admission the same number of times
  evaluation.format     decisions matching upstream's Thought / Action / Action Input pattern,
                        valid hypotheses and confidences, how episodes ended
  evaluation.observation_stop, .eos, .template   the facts of backbone.tokenizer_facts, with what
                        the generated token ids showed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter

from tiicee.clinical import admissions, bound, remember, sha256, use_prior_studies
from tiicee.clinical.backbone import BACKBONES, ROLLOUT_ENVIRONMENT, TRAINING_ENVIRONMENT, chosen
from tiicee.clinical.plan import Departed
from tiicee.clinical.rollout import roll
from tiicee.clinical.train import Paused, fit, hooks, inspect_data, visits
from tiicee.pipeline.stages import StageInterrupted, register

MAPPING = "lab_test_mapping.csv"
DEFAULTS = {  # see the module docstring for what each is sized for
    "nemotron": {"train_admissions": 60, "val_admissions": 8, "pause_at": 90, "reference": True},
    "granite": {"train_admissions": 110, "val_admissions": 12, "pause_at": 150, "reference": False},
}
PRIVATE = ("sequence_sha256", "visits_sha256")  # digests over admission identifiers stay in the stage


def cut(data_dir, out, train, val) -> None:
    """The bench's data: the first admissions of train.csv and of val.csv, and the lab mapping."""
    out.mkdir(exist_ok=True)
    for name, count in (("train.csv", train), ("val.csv", val)):
        table = admissions.read((data_dir / name).read_bytes())
        (out / name).write_bytes(table.subset(table.admissions[:count]))
    (out / MAPPING).write_bytes((data_dir / MAPPING).read_bytes())


def gpu_memory() -> dict:
    import torch
    gb = 1024 ** 3
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gb, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / gb, 2),
            "device_gb": round(torch.cuda.get_device_properties(0).total_memory / gb, 2)}


def fault(error):
    """The outcome name of a fault the bench reports as its result; None for anything else."""
    text = str(error)
    if isinstance(error, Departed):
        return "departed_from_plan"
    if text.startswith("Policy-to-vLLM weight transfer failed") or text.startswith("Adapter merge left"):
        return "weight_sync_failed"  # replication/gpu.py, _move_model_to_vllm and _load_merged_adapter
    if type(error).__name__ == "OutOfMemoryError":
        return "out_of_memory"
    return None


def trained(backbone, seed, data, directory, on_step, saved=lambda step: None, reference=False) -> dict:
    """One part of a bench training, as the record the bench keeps of it. Paused passes through."""
    for name, value in TRAINING_ENVIRONMENT.items():
        if value is None:
            os.environ.pop(name, None)
    try:
        _, _, schedule = fit(backbone, seed, data, directory, on_step, saved, reference)
    except RuntimeError as error:
        if fault(error) is None:
            raise
        return {"outcome": fault(error), **gpu_memory()}
    counts = sorted(Counter(schedule.sequence).items())
    return {"outcome": "ok", "optimizer_steps": len(schedule.sequence), "follows_plan": True,
            **visits(schedule.sequence, inspect_data(data)["trained_ids"], schedule.plan["epochs"]), **gpu_memory(),
            "sequence_sha256": sha256(json.dumps(schedule.sequence).encode()),
            "visits_sha256": sha256(json.dumps(counts).encode())}


def adherence(results, facts) -> dict:
    """Format adherence and token handling, counted over the rolled-out admissions."""
    use_prior_studies()
    from tim.actions import PATTERN  # upstream's decision pattern, verbatim
    stop, eos, count, ended = facts["observation_stop_ids"], facts["eos_id"], Counter(), Counter()
    for item in results.values():
        row = item["row"]
        ended[str(row["termination"])] += 1
        count["answered"] += row["prediction"] is not None
        count["hypotheses"] += len(row["hypotheses"])
        count["hypotheses_valid"] += sum(h != "unknown" for h in row["hypotheses"])
        count["confidences_valid"] += sum(c is not None for c in row["confidences"])
        for step in item["steps"]:
            if step["role"] != "decision":
                continue
            count["decisions"] += 1
            count["formatted"] += re.search(PATTERN, step["text"]) is not None
            count["truncated"] += step["finish_reason"] == "length"
            if step["stop_reason"] == "Observation:":
                count["stopped"] += 1
                count["stripped"] += bool(stop) and step["token_ids"][-len(stop):] == stop
            elif step["finish_reason"] == "stop":
                count["ended_naturally"] += 1
                count["ended_with_eos"] += step["token_ids"][-1:] == [eos]
    return {
        "format": {"admissions": len(results), "answered": count["answered"], "ended": dict(ended),
                   "decisions": count["decisions"], "decisions_formatted": count["formatted"],
                   "decisions_truncated": count["truncated"], "hypotheses": count["hypotheses"],
                   "hypotheses_valid": count["hypotheses_valid"], "confidences_valid": count["confidences_valid"]},
        "observation_stop": {"ids": len(stop), "strips_after_newline": facts["observation_stop_strips"],
                             "stopped": count["stopped"], "stripped": count["stripped"]},
        "eos": {"pad_is_eos": facts["pad_is_eos"], "label_ends_with_eos": facts["label_ends_with_eos"],
                "assistant_turn_ends_with_eos": facts["assistant_turn_ends_with_eos"],
                "ended_naturally": count["ended_naturally"], "ended_with_eos": count["ended_with_eos"]},
        "template": {key: facts[key] for key in ("system_message_sent", "date_strings", "clock_changes",
                                                 "thinking_changes", "fixed_is_default_without_date")},
    }


def options_of(args):
    """The stage's options, with what was not given taken from the backbone's defaults."""
    parser = argparse.ArgumentParser(prog="clinical_bench")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--train-admissions", type=int)
    parser.add_argument("--val-admissions", type=int)
    parser.add_argument("--pause-at", type=int)
    parser.add_argument("--reference", action=argparse.BooleanOptionalAction)
    options = parser.parse_args(args)
    for name, value in DEFAULTS[options.backbone].items():
        if getattr(options, name) is None:
            setattr(options, name, value)
    return options


@register("clinical_bench")
def run(ctx):
    options = options_of(ctx.args)
    backbone, seed = chosen(ctx, options.backbone), ctx.manifest["seeds"]["training"]
    data, small, model = bound(ctx, "data"), ctx.dir / "data", ctx.out_dir / "model"
    inspect_data(data)
    cut(data, small, options.train_admissions, options.val_admissions)
    state = ctx.load_checkpoint() or {}

    if options.reference and "reference" not in state:
        def watched(step, total):  # the reference reports progress and is never asked to stop
            ctx.progress(int(step), int(total), "step")
            return False

        shutil.rmtree(ctx.dir / "reference", ignore_errors=True)  # uninterrupted: a killed attempt is not resumed
        remember(ctx, reference=trained(backbone, seed, small, ctx.dir / "reference", watched, reference=True))
        shutil.rmtree(ctx.dir / "reference", ignore_errors=True)  # only its record is kept
        raise StageInterrupted()

    training, paused_at = state.get("training"), state.get("paused_at", [])
    if training is None:
        try:
            training = {**trained(backbone, seed, small, model, *hooks(ctx, options.pause_at)), "paused_at": paused_at}
        except Paused as paused:
            remember(ctx, paused_at=paused_at + [paused.step])
            raise StageInterrupted() from None
        remember(ctx, training=training)
        if training["outcome"] == "ok":
            raise StageInterrupted()  # the evaluation engine needs a process of its own

    evaluation = None
    if training["outcome"] == "ok":
        os.environ.update(ROLLOUT_ENVIRONMENT)
        try:
            rolled = roll(ctx, backbone, seed, small / "val.csv", small / MAPPING, model / "adapter",
                          small / "train.csv")
            evaluation = {"outcome": "ok", **adherence(rolled["results"], rolled["tokenizer"])}
        except RuntimeError as error:
            if fault(error) is None:
                raise
            evaluation = {"outcome": fault(error)}

    reference = state.get("reference")

    def same(key):
        if reference is None:
            return None
        return reference["outcome"] == training["outcome"] == "ok" and reference[key] == training[key]

    def public(record):
        return None if record is None else {key: value for key, value in record.items() if key not in PRIVATE}

    report = {"backbone": backbone.key, "seed": seed, "train_admissions": options.train_admissions,
              "val_admissions": options.val_admissions, "reference": public(reference), "training": public(training),
              "sequences_identical": same("sequence_sha256"), "visits_identical": same("visits_sha256"),
              "evaluation": evaluation}
    (ctx.out_dir / "bench.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    ctx.approve(ctx.stage_id + ".json", {"stage": ctx.stage_id, **report})
