"""Clinical adapters of study TII-CEE-1: the training track (E8, E9, E10) over the replication.

The replication (replication-of-original/daniel-escalante), tim (cdm-extension/daniel-escalante)
and the pinned upstream (lacdm/upstream) are used as they stand, by import. Where their code has
no seam, a module attribute is replaced inside this process for the length of one call; each such
replacement is named where it is made. That their code is unchanged is the pipeline's check, not
this package's: a stage runs from a read-only export of one commit, re-digested at every start,
with the upstream submodule required to sit at its pin (tiicee/pipeline/manifest.py, verify).
Nothing here calls git or writes under the code directory. Nothing here imports torch, vLLM,
transformers, hydra or pandas at module level, so `tiicee` stays importable with numpy alone.

Stages (tiicee.pipeline.stages registry; list these modules under "stage_modules" in graph.json):
  clinical_data, clinical_folds   tiicee.clinical.folds       CPU
  clinical_train                  tiicee.clinical.train       GPU
  clinical_rollout                tiicee.clinical.rollout     GPU
  clinical_bench                  tiicee.clinical.bench       GPU
  clinical_acceptance             tiicee.clinical.acceptance  CPU

A stage takes every input file or directory from a role its manifest binds, so what it consumes
is what the pipeline hashed before the stage began (SPEC_RESEARCH_ADVANCEMENT.md section 8.3).
GPU stages verify the job environment decided for their kind and refuse any other (backbone.py);
a training follows a plan of admissions fixed before its first step (plan.py).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPLICATION = ROOT / "replication-of-original" / "daniel-escalante"
TIM = ROOT / "cdm-extension" / "daniel-escalante"
CONFIGS = ROOT / "tii-cee" / "clinical_configs"


def use_prior_studies():
    """Put tim, the replication and the pinned upstream on the import path; nothing is installed."""
    if str(TIM) not in sys.path:
        sys.path.insert(0, str(TIM))
    import tim.paths  # noqa: F401  (adds the replication and lacdm/upstream)


def bound(ctx, role):
    """Path of an input the stage's manifest binds under this role."""
    from tiicee.pipeline.manifest import resolve
    return resolve(ctx.paths, ctx.manifest["bound"][role]["path"])


def remember(ctx, **values):
    """Merge values into the stage's checkpoint record. The watcher reads that file's modification
    time as the stage's progress mark, so a part killed at its time limit counts as progress."""
    state = ctx.load_checkpoint() or {}
    state.update(values)
    ctx.save_checkpoint(state)
    return state


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
