"""Launch check 7.4a: accept the primary policy's regenerated validation rollout, or not.

Stage: clinical_acceptance (CPU)
Manifest: roles "stored" (tim's val-a V0 episode log, one {"patient_id", "row", "sha256"} line per
admission; it holds outcomes but no prompts) and "regenerated" (rows.jsonl of the clinical_rollout
that re-ran the same policy on the same split with full prompt logging).

The regenerated trajectories are accepted only if their final answers match the stored outcomes
on at least 99% of admissions (SPEC_RESEARCH_ADVANCEMENT.md section 7.4a). Agreement of tim's
generation_digest is reported beside it. The output is counts only.
"""
from __future__ import annotations

import json
from pathlib import Path

from tiicee.clinical import bound
from tiicee.pipeline.stages import register

PERCENT = 99


def compare(stored, regenerated) -> dict:
    """Counts over admissions paired by patient_id; an admission missing on either side is never a match."""
    old, new = ({row["patient_id"]: row for row in rows} for rows in (stored, regenerated))
    if len(old) != len(stored) or len(new) != len(regenerated):
        raise ValueError("an admission appears more than once")
    paired = old.keys() & new.keys()
    answers = sum(old[i]["prediction"] == new[i]["prediction"] for i in paired)
    return {"admissions": len(old), "regenerated": len(new), "paired": len(paired),
            "final_answers_matching": answers,
            "digests_matching": sum(old[i]["generation_digest"] == new[i]["generation_digest"] for i in paired),
            "required_percent": PERCENT,
            "accepted": len(paired) == len(old) == len(new) and len(old) > 0 and 100 * answers >= PERCENT * len(old)}


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@register("clinical_acceptance")
def run(ctx):
    report = compare([line["row"] for line in _lines(bound(ctx, "stored"))], _lines(bound(ctx, "regenerated")))
    (ctx.out_dir / "acceptance.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    ctx.approve(ctx.stage_id + ".json", {"stage": ctx.stage_id, **report})
