# Launch checks, specification 7.4

Each check ends in a recorded answer or a BLOCKED item. November 13, 2026.

## a. Logged trajectories (DEFAULT D5): ANSWERED

From code only (no private file was read). The unguarded validation and test trajectories store one
row per admission, not per step: prediction, provided and requested tests, hypotheses, confidences,
termination and turn counts, and tim's generation_digest, a SHA-256 over the generated token ids of
every hypothesis and decision generation. They store no prompt text, no prompt digest, and no
thought, action or observation text (tim/runner.py:212-222; replication/shim.py:298-303;
runner.py:156-176). Decision prompts embed earlier completions, including the thought, so prompts
cannot be rebuilt from the rows; neither route of 7.4a's first two options applies.

Path taken, as 7.4a pre-registers: the primary policy's unguarded validation trajectories are
regenerated once with full prompt logging from the same checkpoint and seeds, and accepted only if
final answers match the stored outcomes on at least 99% of admissions; generation_digest agreement is
reported beside it (greedy decoding was bit-reproducible before: V0-shadow matched V0 on 240 of 240
digests). New policies (E8, E9, E10) log full prompts from their first rollout. Test trajectories are
never regenerated outside E6.

## b. Data-use compliance: ANSWERED

DATA_USE_CHECK.md: F1 passes, F2 passes. Pima is excluded from E7 for an unverifiable license.

## c. UF Research Computing policy: PARTLY ANSWERED, finished by the live no-op cycle

Re-read November 13, 2026 (saved copies under sources/):
- scrontab page (docs.rc.ufl.edu/scheduler/scrontab): #SCRON lines act as #SBATCH; jobs must request
  resources; `--dependency=singleton` is the documented pattern; jobs do not run in a login shell;
  the command runs via bash; all recurrences share one job ID; modifying the scrontab cancels and
  resubmits the job. The page does not say whether a scrontab job may call sbatch, squeue or sacct.
- Usage policies: login nodes for job and file management only; "Processes that frequently poll
  the scheduler for queue or job status are not allowed"; no burst QOS for GPU jobs; more than
  10,000 jobs at once needs approval.
- SSH multiplexing page: ControlMaster, ControlPersist 8h, ControlPath ~/.ssh/control/%C on macOS;
  nothing on session length or keepalives. Consistent with the keeper as installed.
- Changed since WATCHER_DESIGN.md: docs.rc.ufl.edu/services/non_batch (the daemon server, the
  fallback host for the watcher) returns 404 today and is gone from the site index; its last
  archived copy (May 11, 2026) still describes the daemon server and cron. The fallback's existence
  is therefore unconfirmed.
Remaining: run one no-op watcher cycle under scrontab and confirm it can call sbatch, squeue and sacct.
If it cannot, and the daemon server cannot be confirmed, that becomes a BLOCKED item; until then
the design's primary path stands.

## d. Environment: ANSWERED

Every UNVERIFIED field of AUTHORIZATION.md section 3 matches the live cluster (STATUS.md section 3).
The study root was created owner-only. Model cache verified; panel models download at pinned
revisions on first use.

## e. Recipe's use of validation: ANSWERED

The validation split is monitored only: evaluated before training, at each epoch and at the end;
no early stopping, no checkpoint selection (`load_best_model_at_end` false and refused,
`metric_for_best_model` None), the final fixed-budget checkpoint is the policy
(lacdm/upstream/src/train.py:20-50; replication/config.py:155-159; replication/gpu.py:325-326,
374-377). E10's inner split therefore holds out a fixed stratified tenth for monitoring only
(EXPERIMENT_CARDS/E10.md).

## f. Cohorts in approved storage: ANSWERED

Only mimic-iv-ext-cdm (top-level names read, nothing inside). E11 is BLOCKED (BLOCKED-002).

## g. Venue: ANSWERED

ICML 2027 pages read November 13, 2026 (saved under sources/). Published: Rio de Janeiro, July 31
to August 6, 2027; abstract January 14 and full paper January 22, 2027, 11:59 PM AoE; first-stage
rejections February 23; decisions May 3; two-stage reviewing; 8 pages plus unlimited references,
impact statement and appendices; double-blind, anonymized submissions, preprints allowed, no links
to non-anonymized repositories, anonymized code as supplementary material; no parallel submission
of substantially similar work; a mandatory AI Disclosure Statement (not counted in the page limit),
with methodology-level use of generative AI described in the main paper; AI systems cannot be
authors; at most 20 submissions per author; every submission designates a reciprocal reviewer, and
each author may be on at most one submission with no eligible reciprocal reviewer (eligibility: an
accepted paper at a listed venue by January 14, 2027); a contribution type must be chosen,
including "Negative Results". Not yet published: process updates promised "by late November" and the
call for position papers. DEFAULT D12's assumed 2026 rules are replaced by these.
