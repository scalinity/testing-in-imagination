"""Reader probe conformance of one policy on its own unguarded rollout (specification 10.1;
RESEARCH_LEDGER.md, D-PROBE).

Stage: clinical_probe --backbone nemotron|granite
Manifest: the roles the policy's rollout bound, "split", "lab_test_mapping" and "adapter" (and
"backbone_overlay" for Granite), plus "steps" and "v0_rows", the rollout's steps.jsonl and
rows.jsonl; seed "policy". Job environment: the rollout's (backbone.ROLLOUT_ENVIRONMENT), and
gpu.evaluate is called with the rollout's split name and seed, so the engine, the merged adapter,
the tokenizer and the environment's configuration are the ones the rollout ran with.

Two checks, with nothing substituted:
  Replay   tim's V0 evaluator runs again over every admission of the split, with a stand-in model
           that returns, in order, the outputs the rollout logged for that admission instead of
           generating, and keeps the prompts the environment builds. Each rebuilt prompt must equal
           the logged one byte for byte, the token ids derived from it must reproduce the logged
           prompt_ids_sha256, and each admission's replayed V0 row must equal the stored row.
           A replay that asks for more generations than were logged is an error.
  Reading  at each hypothesis step, the log probability of each canonical continuation
           "Hypothesis: d", d one of the four diseases, after the logged prompt's token ids, from
           the engine's prompt log probabilities. The most likely d must match the stored greedy
           hypothesis, parsed as the environment parses it, on at least 99% of aligned steps
           (steps whose stored hypothesis is one of the four).
Outcome "conforms" when every prompt and token digest reproduces, every row is identical and the
rate is at least 0.99; otherwise "investigate". The probe reads the rollout's outputs and writes
only its own: scores.jsonl (private: per step, the stored hypothesis and the four log
probabilities) and the counts, rate and outcome, approved for export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiicee.clinical import admissions, bound, use_prior_studies
from tiicee.clinical.backbone import BACKBONES, ROLLOUT_ENVIRONMENT, chosen, compose, require_environment, runtime
from tiicee.clinical.rollout import SPLIT
from tiicee.pipeline.stages import register

RATE = 0.99  # specification 10.1
BATCH = 64  # hypothesis steps per scoring call, four requests each
HYPOTHESIS = re.compile(r"Hypothesis: (.*)")  # upstream's Environment._generate_hypothesis


def _ids_digest(ids) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def _lines(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stored_hypothesis(text, diseases):
    """The hypothesis the environment read from a generation, or None where it read "unknown"."""
    match = HYPOTHESIS.search(text)
    found = match.group(1).lower() if match else None
    return found if found in diseases else None


def prompt_ids(tokenizer, prompt, digest):
    """The prompt's token ids under whichever of the tokenizer's two encodings reproduces the
    logged digest, or None when neither does."""
    for special in (False, True):
        ids = list(tokenizer.encode(prompt, add_special_tokens=special))
        if _ids_digest(ids) == digest:
            return ids
    return None


class Replay:
    """A model that returns the logged outputs of one admission in order and keeps what it was asked."""

    def __init__(self, tokenizer):
        self.tokenizer, self.steps, self.position = tokenizer, [], 0
        self.prompts = self.digests = self.calls = 0

    def get_tokenizer(self):
        return self.tokenizer

    def load(self, steps):
        self.steps, self.position = steps, 0

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        outputs = []
        for prompt in prompts:
            if self.position >= len(self.steps):
                raise RuntimeError("the replay asked for more generations than the rollout logged")
            step = self.steps[self.position]
            self.position += 1
            self.calls += 1
            self.prompts += prompt == step["prompt"]
            ids = prompt_ids(self.tokenizer, prompt, step["prompt_ids_sha256"])
            self.digests += ids is not None
            outputs.append(SimpleNamespace(
                prompt=prompt, prompt_token_ids=ids if ids is not None else list(self.tokenizer.encode(prompt)),
                outputs=[SimpleNamespace(text=step["text"], token_ids=tuple(step["token_ids"]),
                                         finish_reason=step["finish_reason"], stop_reason=step["stop_reason"])]))
        return outputs


class ReplayLog:
    """tim's evaluator log: before an admission it loads that admission's logged steps; after it,
    it compares the replayed row with the stored one and requires every logged step used."""

    def __init__(self, replay, logged, stored):
        self.replay, self.logged, self.stored, self.done = replay, logged, stored, self
        self.admissions = self.rows = self.unused = 0

    def __contains__(self, patient_id):
        self.replay.load(self.logged[patient_id])
        return False

    def append(self, patient_id, row):
        self.admissions += 1
        self.rows += row == self.stored[patient_id]
        self.unused += self.replay.position != len(self.replay.steps)


def score(model, logged, diseases, sampling_params, limit, batch=BATCH):
    """Per hypothesis step: the stored hypothesis and the log probability of each canonical
    continuation after the logged prompt. Steps whose prompt ids cannot be reproduced are not scored,
    nor are steps whose prompt with its longest continuation and one generated token would exceed the
    engine's context `limit` (marked too_long): the prompt is never cut, since that would change it.
    The engine is asked for `batch` steps at a time; each call's outputs, which hold a log
    probability for every prompt position, are reduced to the continuation sums and dropped."""
    tokenizer = model.get_tokenizer()
    continuation = {d: list(tokenizer.encode(f"Hypothesis: {d}", add_special_tokens=False)) for d in diseases}
    longest = max(len(ids) for ids in continuation.values())
    records, pending = [], []
    for patient_id, entries in logged.items():
        for step in entries:
            if step["role"] != "hypothesis":
                continue
            stored = stored_hypothesis(step["text"], diseases)
            ids = prompt_ids(tokenizer, step["prompt"], step["prompt_ids_sha256"])
            record = {"patient_id": patient_id, "index": step["index"], "stored": stored, "logprobs": None,
                      "canonical": stored is not None and list(step["token_ids"][:len(continuation[stored])]) == continuation[stored],
                      "too_long": ids is not None and len(ids) + longest + 1 > limit}
            records.append(record)
            if ids is not None and not record["too_long"]:
                pending.append((record, ids))
    params = type(sampling_params)(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    for first in range(0, len(pending), batch):
        chunk = pending[first:first + batch]
        requests = [{"prompt_token_ids": ids + continuation[d]} for _, ids in chunk for d in diseases]
        outputs = model.generate(requests, sampling_params=params, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError("the engine returned a different number of outputs than it was asked for")
        for i, (record, ids) in enumerate(chunk):
            record["logprobs"] = {d: sum(outputs[i * len(diseases) + j].prompt_logprobs[len(ids) + k][token].logprob
                                         for k, token in enumerate(continuation[d]))
                                  for j, d in enumerate(diseases)}
        del outputs
    return records


def evaluator(found, logged, stored):
    """A drop-in for replication.shim._evaluate_with_model: the replay over tim's V0 evaluator, then
    the reading on the engine gpu.evaluate built."""

    def evaluate(cfg, dataset, model, sampling_params):
        from tim import runner
        replay = Replay(model.get_tokenizer())
        log = ReplayLog(replay, logged, stored)
        runner.make_evaluator(runner.VARIANTS["V0"], dataset, None, log,
                              runner.Limits(**runner.CONFIG["limits"]))(cfg, dataset, replay, sampling_params)
        diseases = [d.lower() for d in cfg.dataset.disease_list]
        limit = cfg.environment.max_length + cfg.training.max_completion_length  # gpu.evaluate's max_model_len
        found.update(replay=replay, log=log, steps=score(model, logged, diseases, sampling_params, limit))
        return []

    return evaluate


def report(found, total_steps) -> dict:
    replay, log, steps = found["replay"], found["log"], found["steps"]
    aligned = [s for s in steps if s["stored"] is not None]
    matched = sum(s["logprobs"] is not None and max(s["logprobs"], key=s["logprobs"].get) == s["stored"]
                  for s in aligned)
    rate = matched / len(aligned) if aligned else None
    reproduced = (replay.calls == total_steps and replay.prompts == total_steps and replay.digests == total_steps
                  and log.rows == log.admissions and log.unused == 0)
    return {"admissions": log.admissions, "steps": total_steps, "replayed": replay.calls,
            "rebuilt_identical": replay.prompts, "token_digests_identical": replay.digests,
            "v0_identical": log.rows, "admissions_with_unused_steps": log.unused,
            "hypothesis_steps": len(steps), "aligned": len(aligned), "matched": matched, "rate": rate,
            "canonical_tokens": sum(s["canonical"] for s in aligned), "unscoreable_length": sum(s["too_long"] for s in steps),
            "outcome": "conforms" if reproduced and rate is not None and rate >= RATE else "investigate"}


def probe(ctx, backbone, seed, split, mapping, adapter, steps_path, rows_path) -> dict:
    require_environment(ROLLOUT_ENVIRONMENT)
    if Path(split).name == "test.csv":
        raise ValueError("the test split is never probed here")
    data = Path(split).read_bytes()
    members = admissions.read(data).admissions
    by_fingerprint = {entry["fingerprint"]: entry["steps"] for entry in _lines(steps_path)}
    if set(by_fingerprint) != {a.fingerprint for a in members}:
        raise ValueError("the prompt log does not cover exactly the split's admissions")
    logged = {a.patient_id: by_fingerprint[a.fingerprint] for a in members}
    stored = {row["patient_id"]: row for row in _lines(rows_path)}
    if set(stored) != set(logged):
        raise ValueError("the V0 results do not cover exactly the split's admissions")

    use_prior_studies()
    from replication import gpu
    from replication.config import stream_seed
    input_dir = ctx.dir / "input"
    input_dir.mkdir(exist_ok=True)
    (input_dir / f"{SPLIT}.csv").write_bytes(data)
    (input_dir / "lab_test_mapping.csv").write_bytes(Path(mapping).read_bytes())
    cfg = compose(backbone, input_dir, None, seed)
    found = {}
    with runtime(backbone, cfg), patch.object(gpu, "_evaluate_with_model", evaluator(found, logged, stored)):
        gpu.evaluate(cfg, input_dir, split=SPLIT, seed=stream_seed(seed, SPLIT, "policy"), adapter_dir=adapter)
    (ctx.out_dir / "scores.jsonl").write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in found["steps"]))
    return report(found, sum(len(steps) for steps in logged.values()))


@register("clinical_probe")
def run(ctx):
    parser = argparse.ArgumentParser(prog="clinical_probe")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    backbone = chosen(ctx, parser.parse_args(ctx.args).backbone)
    seed = ctx.manifest["seeds"]["policy"]
    found = probe(ctx, backbone, seed, bound(ctx, "split"), bound(ctx, "lab_test_mapping"), bound(ctx, "adapter"),
                  bound(ctx, "steps"), bound(ctx, "v0_rows"))
    ctx.approve(ctx.stage_id + ".json", {"stage": ctx.stage_id, "backbone": backbone.key, "seed": seed, **found})
