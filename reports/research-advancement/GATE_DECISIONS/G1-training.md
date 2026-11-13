# Gate G1, training part: repairs needed before any credentialed GPU run

Evaluated November 13, 2026, about 6:40 AM EST. Immutable once committed.

## Rule (specification 8.3, 8.4, 14.1)

Before any credentialed GPU run, including training: one shared preflight; an immutable manifest
binding the policy or recipe, reader-probe configuration, fitted artifacts, calibration, budgets,
the experiment card, the protocol and the SHA-256 of AUTHORIZATION.md, verified at every start and
resume, a mismatch being a refusal that is never retried; and the information-boundary tests the
stages need.

## Evidence

| Requirement | Verified by | Commit |
|---|---|---|
| Immutable manifest binding code export, every file digest, bound inputs, env map, seeds, authorization and addendum digests; refusal (exit 20/30) on mismatch, never retried | tests/test_pipeline_manifest.py, test_pipeline_stage.py, test_pipeline_watcher.py (193 tests, 20 consecutive clean runs under Python 3.12 and 3.9) | 838a819 |
| Code identity: stages run from read-only per-commit exports; the pinned upstream verified at every stage start | tests/test_pipeline_deploy.py; live deployment of 462600e | 838a819 |
| The watcher on the real cluster: authorization check, scheduler calls, gate evaluation, completion and self-disable | Live no-op graph, GATE_DECISIONS/G0-addendum-7.4c.md | 462600e |
| Boundaries for the training track: test.csv never opened or listed (audit hook with a planted file); fold disjointness and coverage; a fold policy never trains or is monitored on its held-out admissions; resumed training follows the uninterrupted admission sequence, else refused (D-CLIN-RESUME) | tests/test_clinical_folds.py, test_clinical_plan.py, test_clinical_train.py (89 tests under the project's CPU environment) | 5105e89, fafe643, 309c84c |
| Per-stage environment bound in the manifest; training without CUBLAS_WORKSPACE_CONFIG, evaluation with it | tests/test_pipeline_watcher.py, test_clinical_backbone.py | 838a819, 5105e89 |

Not yet verified, by design: anything on a GPU, a real tokenizer or real data. The training-track
graph's first GPU stages are the benches that verify those (gate G_resume holds every training
stage until the Nemotron bench shows faithful resume).

## Result

MET for the training track's first graph (tii-cee/tiicee/clinical/graph_training_m1.json).

## Path taken

Deploy 838a819 or later and install the training-track graph: data copy, folds, the Nemotron and
Granite benches, the primary policy's prompt-logged validation rollout and launch check 7.4a's
acceptance. No training stage is added until G_resume is decided.
