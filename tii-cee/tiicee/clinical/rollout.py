"""Unguarded greedy rollout of one policy on one split file, with every prompt logged (E8, E9, E10,
launch check 7.4a).

Stage: clinical_rollout --backbone nemotron|granite
Manifest: roles "split" (the split file to roll out on; never test.csv), "lab_test_mapping",
"adapter" (the policy's adapter directory), "train_csv" (the file the policy was trained on) and
seed "policy" (the policy's training seed).

The work is replication.gpu.evaluate, unchanged: its engine, adapter merge and greedy sampling.
One attribute of that module is replaced while it runs, gpu._evaluate_with_model, as tim does
(tim/runner.py, _gpu_rows). The replacement is tim's own evaluator for its unguarded variant V0,
so each row is tim's V0 row (the replication's row format plus tim's fields) and its
generation_digest is computed by tim's code. Between tim's evaluator and the engine sits
PromptLog, which keeps, for every generation of every role, the exact prompt text, its SHA-256 and
the generated token ids. gpu.evaluate reads a split named train, val or test from a directory, so
the stage hands it a private directory holding the split file's bytes as val.csv; "val" is
therefore the split name in the evaluation seed stream, whatever the file was called.

An admission is complete when its row and prompt log are appended, in one line, to the stage's
item log under its fingerprint; a resumed part skips it. Imagined outcomes play no part here.

Outputs, all private: rows.jsonl (V0), v1_rows.jsonl (tim's hypothesis fallback over the same
rows), steps.jsonl (the prompt log), rollout.json (identity and tokenizer facts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from tiicee.clinical import admissions, bound, remember, sha256, use_prior_studies
from tiicee.clinical.backbone import (BACKBONES, chosen, compose, recipe_record, require_frozen_code, runtime,
                                       tokenizer_facts)
from tiicee.clinical.v1 import v1_rows
from tiicee.pipeline.stages import StageInterrupted, register
from tiicee.pipeline.util import tree_digest

CUBLAS = ("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cdm-extension/daniel-escalante/DECISIONS.md, every GPU job
SPLIT = "val"  # the name gpu.evaluate knows the rolled-out file by


def _ids_digest(ids) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def generation_digest(steps) -> str:
    """tim's generation_digest, recomputed from a prompt log: the SHA-256 of the comma-joined
    per-generation digests of every hypothesis and decision generation, in order, confidence
    generations left out (tim/runner.py, Recorder and make_evaluator)."""
    kept = [_ids_digest(step["token_ids"]) for step in steps if step["role"] != "confidence"]
    return hashlib.sha256(",".join(kept).encode()).hexdigest()


def markers(cfg) -> dict:
    """The fixed opening of each agent's prompt, filled in as upstream's environment fills its
    templates. It tells a decision prompt from a hypothesis prompt without relying on call order."""
    decision = Path(cfg.environment.prompt_template_file).read_text()
    hypothesis = Path(cfg.environment.hypothesis_prompt_template_file).read_text()
    found = {"decision": decision.format(patient_history="\0", first_hypothesis="").partition("\0")[0],
             "hypothesis": hypothesis.format(provided_test_results="\0").partition("\0")[0]}
    if not all(found.values()):
        raise ValueError("an agent's prompt template has no fixed opening")
    return found


class PromptLog:
    """Passes generation through and keeps every generation of the admission being rolled out."""

    def __init__(self, model, markers):
        self.model, self.markers, self.steps = model, markers, []

    def get_tokenizer(self):
        return self.model.get_tokenizer()

    def __getattr__(self, name):
        return getattr(self.model, name)

    def role(self, prompt) -> str:
        if prompt.endswith("Confidence: "):  # upstream's confidence-calibration prompt; tim's Recorder uses this test
            return "confidence"
        found = [name for name, marker in self.markers.items() if marker in prompt]
        if len(found) != 1:
            raise ValueError("a prompt does not belong to exactly one agent")
        return found[0]

    def generate(self, prompts, *args, **kwargs):
        outputs = self.model.generate(prompts, *args, **kwargs)
        for prompt, output in zip(prompts, outputs):
            completion, role = output.outputs[0], self.role(prompt)
            turn = sum(step["role"] == "decision" for step in self.steps)  # the decision turn this belongs to
            self.steps.append({
                "index": len(self.steps), "turn": None if role == "confidence" else turn, "role": role,
                "prompt": prompt, "prompt_sha256": sha256(prompt.encode()),
                "prompt_tokens": len(output.prompt_token_ids),
                "prompt_ids_sha256": _ids_digest(output.prompt_token_ids),
                "token_ids": [int(i) for i in completion.token_ids], "text": completion.text,
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None)})
        return outputs


class Completed:
    """What tim's evaluator is given as its log: which admissions to skip, and where a finished one goes.
    tim asks `patient_id in log.done` and calls `log.append(patient_id, row)`; `done` is this object."""

    def __init__(self, ctx, fingerprint_of):
        self.ctx, self.fingerprint_of, self.done, self.prompts = ctx, fingerprint_of, self, None

    def __contains__(self, patient_id):
        # After the stop flag every remaining admission reads as done, so tim's loop runs no more.
        return self.ctx.stop or self.ctx.is_done(self.fingerprint_of[patient_id])

    def append(self, patient_id, row):
        steps, self.prompts.steps = self.prompts.steps, []
        if generation_digest(steps) != row["generation_digest"]:
            raise RuntimeError("the prompt log does not reproduce tim's generation digest")
        self.ctx.mark_done(self.fingerprint_of[patient_id], {"row": row, "steps": steps})
        done = sum(self.ctx.is_done(fingerprint) for fingerprint in self.fingerprint_of.values())
        self.ctx.progress(done, len(self.fingerprint_of), "admission")


def evaluator(completed, identity):
    """A drop-in for replication.shim._evaluate_with_model: tim's V0 evaluator over a logging model."""

    def evaluate(cfg, dataset, model, sampling_params):
        from tim import runner
        completed.prompts = PromptLog(model, markers(cfg))
        remember(completed.ctx, rollout={
            "identity": identity, "tokenizer": tokenizer_facts(model.get_tokenizer(), cfg.environment.system_prompt)})
        inner = runner.make_evaluator(runner.VARIANTS["V0"], dataset, None, completed,
                                      runner.Limits(**runner.CONFIG["limits"]))
        return inner(cfg, dataset, completed.prompts, sampling_params)

    return evaluate


def roll(ctx, backbone, seed, split, mapping, adapter, trained_on) -> dict:
    """Roll the policy out on every admission of `split` not yet complete, then write the outputs."""
    if Path(split).name == "test.csv":
        raise ValueError("the test split is never rolled out here")
    data = Path(split).read_bytes()
    members = admissions.read(data).admissions
    admissions.require_unique(members)
    if not members:
        raise ValueError("the split holds no admissions")
    trained = {a.fingerprint for a in admissions.read(Path(trained_on).read_bytes()).admissions}
    if any(a.fingerprint in trained for a in members):
        raise ValueError("the policy was trained on an admission of this split")

    if not all(ctx.is_done(a.fingerprint) for a in members):
        if os.environ.setdefault(*CUBLAS) != CUBLAS[1]:
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from the project's setting")
        use_prior_studies()
        from replication import gpu
        from replication.config import stream_seed
        require_frozen_code()
        input_dir = ctx.dir / "input"
        input_dir.mkdir(exist_ok=True)
        (input_dir / f"{SPLIT}.csv").write_bytes(data)
        (input_dir / "lab_test_mapping.csv").write_bytes(Path(mapping).read_bytes())
        cfg = compose(backbone, input_dir, None, seed)
        completed = Completed(ctx, {a.patient_id: a.fingerprint for a in members})
        with runtime(backbone, cfg), \
                patch.object(gpu, "_evaluate_with_model", evaluator(completed, recipe_record(cfg, backbone, seed))):
            gpu.evaluate(cfg, input_dir, split=SPLIT, seed=stream_seed(seed, SPLIT, "policy"), adapter_dir=adapter)
        if not all(ctx.is_done(a.fingerprint) for a in members):
            if not ctx.stop:
                raise RuntimeError("the evaluator returned before every admission was complete")
            raise StageInterrupted()

    state, results = ctx.load_checkpoint()["rollout"], ctx.results()
    rows = [results[a.fingerprint]["row"] for a in members]

    def lines(name, values):
        (ctx.out_dir / name).write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))

    lines("rows.jsonl", rows)
    lines("v1_rows.jsonl", v1_rows(rows))
    lines("steps.jsonl", [{"fingerprint": a.fingerprint, "patient_id": a.patient_id,
                           "steps": results[a.fingerprint]["steps"]} for a in members])
    record = {**state["identity"], "tokenizer": state["tokenizer"], "admissions": len(members),
              "split_sha256": sha256(data), "evaluation_split_name": SPLIT,
              "adapter_sha256": tree_digest(adapter)[0], "manifest_id": ctx.manifest["manifest_id"]}
    (ctx.out_dir / "rollout.json").write_text(json.dumps(record, sort_keys=True, indent=2, default=str) + "\n")
    return {"rows": rows, "results": results, "tokenizer": state["tokenizer"]}


@register("clinical_rollout")
def run(ctx):
    parser = argparse.ArgumentParser(prog="clinical_rollout")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    backbone = chosen(ctx, parser.parse_args(ctx.args).backbone)
    seed = ctx.manifest["seeds"]["policy"]
    rows = roll(ctx, backbone, seed, bound(ctx, "split"), bound(ctx, "lab_test_mapping"), bound(ctx, "adapter"),
                bound(ctx, "train_csv"))["rows"]
    ctx.approve(ctx.stage_id + ".json", {
        "stage": ctx.stage_id, "backbone": backbone.key, "seed": seed,
        "admissions": len(rows), "answered": sum(row["prediction"] is not None for row in rows)})
