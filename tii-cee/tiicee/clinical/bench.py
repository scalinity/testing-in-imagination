"""Bench of a backbone under the frozen recipe: a short training run, then a few validation
episodes (EXPERIMENT_CARDS/E9.md, "Bench and repairs").

Stage: clinical_bench --backbone nemotron|granite [--train-admissions 110] [--val-admissions 12] [--pause-at 150]
Manifest: role "data" (as clinical_train) and seed "training".

The bench calls the same code as the real stages (train.fit, rollout.roll) on the first
admissions of train.csv and val.csv. The defaults give 220 optimizer steps, enough to pass through
all three loss phases (100 da, 100 ha_sft, then ha_conf_cal) and two 100-step checkpoints.

It runs as three processes, each ending with the continue code until the last:
  1. trains until --pause-at, where it behaves as if Slurm's warning had arrived: forced
     checkpoint, exit. The default step lies inside the second epoch, the case every real training
     meets when its 24 hours run out (0 disables the pause);
  2. resumes through gpu.train's own path, finishes training, and exits because the evaluation
     engine cannot share a process with the training engine;
  3. rolls the bench adapter out and writes the report.
A weight-synchronization fault or a CUDA out-of-memory fault is the bench's result, recorded and
reported, not a crash; any other error fails the stage.

Report (bench.json, also approved for export; counts and flags only):
  training     outcome, optimizer steps, where it paused, how often each admission was trained on
               across the resume (visits_as_recipe: exactly num_train_epochs times each), and peak
               GPU memory as torch's allocator saw it in the part that finished training
  format       decisions matching upstream's Thought / Action / Action Input pattern, valid
               hypotheses and confidences, how episodes ended
  observation_stop, eos, template   the facts of backbone.tokenizer_facts, with what the generated
               token ids showed: trailing "Observation:" ids stripped, completions ending in EOS
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from tiicee.clinical import admissions, bound, remember, use_prior_studies
from tiicee.clinical.backbone import BACKBONES, chosen
from tiicee.clinical.rollout import roll
from tiicee.clinical.train import Paused, fit, hooks, inspect_data, visits
from tiicee.pipeline.stages import StageInterrupted, register

MAPPING = "lab_test_mapping.csv"


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
    if text.startswith("Policy-to-vLLM weight transfer failed") or text.startswith("Adapter merge left"):
        return "weight_sync_failed"  # replication/gpu.py, _move_model_to_vllm and _load_merged_adapter
    if type(error).__name__ == "OutOfMemoryError":
        return "out_of_memory"
    return None


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


@register("clinical_bench")
def run(ctx):
    parser = argparse.ArgumentParser(prog="clinical_bench")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--train-admissions", type=int, default=110)
    parser.add_argument("--val-admissions", type=int, default=12)
    parser.add_argument("--pause-at", type=int, default=150)
    options = parser.parse_args(ctx.args)
    backbone, seed = chosen(ctx, options.backbone), ctx.manifest["seeds"]["training"]
    data, small, model = bound(ctx, "data"), ctx.dir / "data", ctx.out_dir / "model"
    inspect_data(data)
    cut(data, small, options.train_admissions, options.val_admissions)

    state = ctx.load_checkpoint() or {}
    training, paused_at = state.get("training"), state.get("paused_at", [])
    if training is None:
        try:
            cfg, _, steps, sequence = fit(backbone, seed, small, model, *hooks(ctx, options.pause_at))
            trained_ids, epochs = inspect_data(small)["trained_ids"], int(cfg.training.num_train_epochs)
            training = {"outcome": "ok", "optimizer_steps": steps, "paused_at": paused_at, **gpu_memory(),
                        **visits(sequence, steps, trained_ids, epochs)}
        except Paused as paused:
            remember(ctx, paused_at=paused_at + [paused.step])
            raise StageInterrupted() from None
        except RuntimeError as error:
            if fault(error) is None:
                raise
            training = {"outcome": fault(error), "paused_at": paused_at, **gpu_memory()}
        remember(ctx, training=training)
        if training["outcome"] == "ok":
            raise StageInterrupted()  # the evaluation engine needs a process of its own

    evaluation = None
    if training["outcome"] == "ok":
        try:
            rolled = roll(ctx, backbone, seed, small / "val.csv", small / MAPPING, model / "adapter",
                          small / "train.csv")
            evaluation = {"outcome": "ok", **adherence(rolled["results"], rolled["tokenizer"])}
        except RuntimeError as error:
            if fault(error) is None:
                raise
            evaluation = {"outcome": fault(error)}
    report = {"backbone": backbone.key, "seed": seed, "train_admissions": options.train_admissions,
              "val_admissions": options.val_admissions, "training": training, "evaluation": evaluation}
    (ctx.out_dir / "bench.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    ctx.approve(ctx.stage_id + ".json", {"stage": ctx.stage_id, **report})
