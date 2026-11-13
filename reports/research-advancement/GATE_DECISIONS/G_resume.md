# Gate G_resume: training in parts follows the uninterrupted admission sequence

Evaluated November 13, 2026, 2:13 PM EST. Immutable once committed.

## Rule (RESEARCH_LEDGER.md, D-CLIN-RESUME and D-GATE-RESUME)

tii-cee/gates/resume_gate.py, committed in 309c84c before its input existed. It is met only when
both the uninterrupted reference and the training in parts end "ok" and follow the drawn plan,
train on the same admission at every optimizer step, and visit every admission the same number of
times. Every E8, E9 and E10 training stage waits on it.

## Evidence

- Rule: run on HiPerGator from the deployed export of 351b786
  (code/351b786…/tii-cee/gates/resume_gate.py). SHA-256 b2207a1e…1a68, identical to the committed
  file at 309c84c.
- Input: export/approved/bench-nemotron.json, SHA-256 65546677…474b. It was written by the Nemotron
  bench (stage bench-nemotron) over five attempts. Attempts 1 to 4 ended with exit 10, a planned
  continuation; attempt 5 (job 51191830) completed with exit 0 at 1:44 PM EST. The bench used
  0.051 GPU-hours in attempt 5 and 3.335 in all five attempts.
- Bench size: 60 training and 8 validation admissions, seed 269.

| Check | Value |
|---|---|
| report_present | 1 |
| reference_ok | 1 |
| training_ok | 1 |
| reference_follows_plan | 1 |
| training_follows_plan | 1 |
| sequences_identical | 1 |
| visits_identical | 1 |

| Part | Optimizer steps | Paused at steps | Visits per admission | Peak allocated | Peak reserved | Device |
|---|---|---|---|---|---|---|
| Reference, uninterrupted | 120 | none | 2 (as the recipe) | 88.99 GB | 90.68 GB | 94.97 GB |
| Training in parts | 120 | 60 (epoch boundary), 90 (inside epoch two) | 2 (as the recipe) | 89.04 GB | 90.46 GB | 94.97 GB |

The evaluation step of the bench also ended "ok". Peak reserved memory leaves about 4.3 GB of the
RTX PRO 6000's 94.97 GB free at the bench's settings.

## Result

MET (rule output: path "training_track_proceeds").

## Path taken

The training track proceeds. The E8 and E10 training stages, on the Nemotron backbone, are added
to the graph by a redeploy. E9's training stages also wait on the Granite bench (e9-bench-granite,
job 51202779, running since 1:49 PM EST), which sets E9's backbone sizes.
