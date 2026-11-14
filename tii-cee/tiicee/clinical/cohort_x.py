"""E10's cohort-X integrity check, over the outputs the fold stages actually wrote (EXPERIMENT_CARDS/E10.md,
decision rule).

Stage: clinical_cohort_x
Manifest: role "folds" (e10-folds' output directory) and, for each fold k, roles "policy_fk"
(e10-train-fk's output directory) and "rollout_fk" (e10-rollout-fk's). CPU only, standard library.

Integrity only, by fingerprints: every admission of cohort X is scored by exactly one fold policy,
and that policy's training and monitoring sets do not contain it. folds.check_folds verified the
plan when the folds were cut; this stage verifies what ran:
  fold_partition      the fold directories still partition the admissions (folds.check_folds);
  fk:policy_data      policy k recorded training and monitoring on fold k's files: its digests of
                      train.csv, val.csv, heldout.csv and the lab mapping are those of fold k's;
  fk:rollout_split    rollout k rolled out fold k's heldout.csv;
  fk:rollout_adapter  with policy k's adapter;
  fk:duplicate        rollout k scored an admission twice;
  fk:outside_heldout  rollout k scored an admission that is not in fold k's held-out file;
  fk:missing          rollout k left one of fold k's held-out admissions unscored;
  fk:leak             rollout k scored an admission policy k trained on or monitored;
  fk:v0_count         rollout k's V0 results do not number its scored admissions;
  not_once            across the five rollouts, an admission is not scored exactly once.
A failed check is the result, outcome "killed" with its codes: leakage kills cohort X. A missing or
unreadable output, a malformed V0 result among them, is an error, not a result.

Outputs: integrity.json, private, and the same counts, flags and codes approved for export.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tiicee.clinical import admissions, bound, folds, sha256
from tiicee.pipeline.stages import register

FILES = (*(f"{part}.csv" for part in folds.PARTS), folds.MAPPING)


def _json(path):
    return json.loads(Path(path).read_text())


def _fingerprints(steps):
    """The fingerprint of each admission a rollout scored, in its order, read one line at a time."""
    with open(steps) as handle:
        return [json.loads(line)["fingerprint"] for line in handle if line.strip()]


def _v0_results(path):
    """How many V0 results a rollout wrote. Each must be a JSON object: a malformed or truncated
    record raises, since what reads cohort X downstream could not read it either."""
    count = 0
    with open(path) as handle:
        for line in handle:
            if line.strip():
                if not isinstance(json.loads(line), dict):
                    raise ValueError("a V0 result is not a JSON object")
                count += 1
    return count


def check(folds_dir, policies, rollouts) -> dict:
    """The integrity report. policies and rollouts list fold k's output directories at index k."""
    folds_dir = Path(folds_dir)
    parts = [{part: admissions.read((folds_dir / f"fold-{k}" / f"{part}.csv").read_bytes()).admissions
              for part in folds.PARTS} for k in range(folds.FOLDS)]
    union = [a for fold in parts for a in fold["heldout"]]
    everyone = {a.fingerprint for a in union}
    reasons, rows, scored_anywhere = [], [], Counter()
    try:
        folds.check_folds(folds_dir, union, (folds_dir / "fold-0" / folds.MAPPING).read_bytes())
    except ValueError:
        reasons.append("fold_partition")

    for k in range(folds.FOLDS):
        directory = folds_dir / f"fold-{k}"
        digests = {name: sha256((directory / name).read_bytes()) for name in FILES}
        policy, rollout = _json(Path(policies[k]) / "policy.json"), _json(Path(rollouts[k]) / "rollout.json")
        scored = _fingerprints(Path(rollouts[k]) / "steps.jsonl")
        scored_anywhere.update(scored)
        held = {a.fingerprint for a in parts[k]["heldout"]}
        seen = {a.fingerprint for part in ("train", "val") for a in parts[k][part]}
        found = {
            "policy_data": {name: policy["data"].get(name) for name in FILES} == digests,
            "rollout_split": rollout["split_sha256"] == digests["heldout.csv"],
            "rollout_adapter": rollout["adapter_sha256"] == policy["adapter_sha256"],
        }
        leaks = len(set(scored) & seen)
        v0_results = _v0_results(Path(rollouts[k]) / "rows.jsonl")
        failed = [name for name, ok in found.items() if not ok]
        failed += [name for name, bad in (("duplicate", len(set(scored)) != len(scored)),
                                          ("outside_heldout", bool(set(scored) - held)),
                                          ("missing", bool(held - set(scored))),
                                          ("leak", leaks > 0),
                                          ("v0_count", v0_results != len(scored))) if bad]
        reasons += [f"f{k}:{name}" for name in failed]
        rows.append({"fold": k, "heldout": len(held), "scored": len(scored), "v0_results": v0_results,
                     "leaks": leaks, **found})

    scored_once = sum(scored_anywhere[f] == 1 for f in everyone)
    if scored_once != len(everyone) or set(scored_anywhere) - everyone:
        reasons.append("not_once")
    return {"outcome": "killed" if reasons else "intact", "reasons": reasons, "admissions": len(everyone),
            "scored_once": scored_once, "folds": rows}


@register("clinical_cohort_x")
def run(ctx):
    report = {"stage": ctx.stage_id, **check(bound(ctx, "folds"),
                                             [bound(ctx, f"policy_f{k}") for k in range(folds.FOLDS)],
                                             [bound(ctx, f"rollout_f{k}") for k in range(folds.FOLDS)])}
    (ctx.out_dir / "integrity.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    ctx.approve(ctx.stage_id + ".json", report)
