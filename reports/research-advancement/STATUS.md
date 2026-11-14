# STATUS, study TII-CEE-1

Updated November 14, 2026, about 4:25 PM EST. Times are US Eastern.

## 1. Where the study is

- Milestone: M0 COMPLETE. M1 IN PROGRESS. M2 IN PROGRESS. M3 to M5 NOT STARTED.
- M0: CLOSED (MILESTONES/M0.md). The live no-op graph ran end to end on HiPerGator and the watcher
  removed its own entry at 5:45 AM; launch check 7.4c is answered and G0 is met in full.
- M1: the twin repairs (REVIEW-002, -006, -011) are verified (GATE_DECISIONS/G1-twin.md); the
  section 8.3 preflight and manifests are built (tii-cee/tiicee/pipeline); the training-track
  adapters, the resume guarantee (D-CLIN-RESUME) and gate G_resume are committed; the training
  track's first graph is installed on HiPerGator (G1 training part met). G_resume is MET
  (GATE_DECISIONS/G_resume.md, 2:13 PM EST): the Nemotron bench trained in parts on the same
  admission at every step as the uninterrupted run. Launch check 7.4a is
  accepted: 240 of 240 final answers and digests match (LAUNCH_CHECKS.md). REVIEW-001 (for E3) and
  REVIEW-008 (for E5) are not started.
- M2: E0 COMPLETED, PASS. E1 COMPLETED at a4f34bd except the E1-C interval: the Mac stopped it at
  about 9:08 PM EST on November 13 for low memory, and it runs again on HiPerGator since 10:18 AM EST
  (job 51313205, cai6734-b, from an exact export of a4f34bd; about 20 hours or more). G2b MET (projection repairs drift); G2c NOT MET (CEE-Scale stays in the twin;
  charts get projection only, stateful track). G2a waits for E1-C. D-REGIME's coordinate (tau_B at
  the first decision point) is VALIDATED in the twin under W0 and W2 (RESEARCH_LEDGER.md, "D-REGIME
  validation"); for the twin's readers the ordering is an identity. The two open questions are
  decided ("D-REGIME decisions"): E2's scale call is read against R0 with a band of log(1.01), and
  E2, E4 and E7 are amended. The extension with result-dependent readers is SUPPORTED under W0 and
  W2 ("D-REGIME extension"), with a residual dependence on outcome frequencies reported.
- Experiments: E0 COMPLETED; E1 COMPLETED except E1-C; E8, E9 and E10 IN PROGRESS: all twelve
  trainings finished their first parts overnight; their second parts failed at import from 4:18 AM
  EST because graph m4 listed a module their code exports lack, and they stood idle until the fix
  (graph_training_m6.json, 04513dc) and the reinstalled watcher resubmitted them at 1:45 PM EST
  (RESEARCH_LEDGER.md, "Operations, November 14"); five run, seven wait on the class pools;
  E11 SKIPPED (accepted gap, not run); all others PLANNED.
- Scientific outcome: NOT EVALUATED for every experiment.
- Open BLOCKED items: none. Accepted gap: BLOCKED-002, no independent cohort, so E11 does not run
  (accepted by Daniel November 13, 2026). BLOCKED-001 is closed.
- AI workers: the Cloudflare 403 of 4:47 AM was cleared by a fresh login. Fable 5.2 workers are at
  their weekly limit until November 18, 5 AM EST; the lead session and the HiPerGator watcher are
  unaffected.
- Next executable actions: the twelve second parts of training run (about 13 to 15 hours each),
  then each policy rolls out, is probed (section 10.1; P's probe conforms under D-PROBE amendment
  A1, 610/613), and E10's cohort-X check runs after the five fold rollouts; when E1-C ends, its
  c_intervals.json comes back, the rules and manifest are re-run and G2a is evaluated; E2's analysis
  code (the relative scale call, the W4 rules and the per-result-kind slopes of amendment A1);
  REVIEW-001 and REVIEW-008.

## 2. Authorization

- AUTHORIZATION.md SHA-256 72ccc89c…21ae equals the value quoted in SIGNATURE.md (commit b1a3614).
  Valid until March 31, 2027.
- AUTHORIZATION_ADDENDUM_1.md (SHA-256 f59cd6fa…5658, commit c5902ea): authorized by Daniel's
  direct statements of November 13, 2026, recorded verbatim in SIGNATURE.md (commit 7a5b3c1). GPU
  jobs may use accounts cai6734 and cai5724, 16 GPUs at once; CPU jobs the burst and CPU QOS the
  addendum lists. Every GPU job stays on one RTX PRO 6000.

## 3. Setup items from the launch prompt

1. Connection keeper: installed November 13, 2026 at 2:45 AM EST, following KEEPER_INSTALL.md's
   section for an existing master. A master for daniel.escalante@hpg.rc.ufl.edu was already
   running, so none was started. The existing `Host hpg.rc.ufl.edu` entry in ~/.ssh/config was
   kept with its ControlPath; only `ServerAliveCountMax 5` and `TCPKeepAlive yes` were added
   (backup at ~/.ssh/config.bak-tii-20261113). keeper.conf sets TII_HPG_HOST to
   daniel.escalante@hpg.rc.ufl.edu and TII_REMOTE_EVENTS to
   /blue/cai6734/daniel.escalante/private/tii-cee-1/state/events.tsv. Both plists passed
   `plutil -lint`; both launch agents are bootstrapped; the first keeper cycle logged "ok";
   caffeinate is running. If the session drops, the one command to run is `sshhiper` (or
   `ssh -fN daniel.escalante@hpg.rc.ufl.edu`).
2. Compute fields of AUTHORIZATION.md section 3, checked read-only at about 2:46 AM EST:

   | Field | Recorded value | Live value | Result |
   |---|---|---|---|
   | Slurm account | cai6734 | association cai6734 with QOS cai6734, cai6734-b | Match |
   | GPU QOS | cai6734: 8 GPUs, 32 CPUs, 250 GB | cpu=32, gres/gpu=8, mem=250G; MaxWall 31 days | Match |
   | CPU QOS | cai6734-b: CPUs and memory, no GPUs | cpu=288, gres/gpu=0, mem=2250G; MaxWall 4 days | Match |
   | GPU partition | hpg-rtx6000, gpu:rtx_pro_6000 | hpg-rtx6000 up, 45 nodes of 8 × rtx_pro_6000; partition MaxTime 14 days | Match |
   | Most GPUs at once | 8 | QOS gres/gpu=8 (16 with the addendum's cai5724) | Match |
   | Private data location | …/private/mimic-iv-ext-cdm/1.1/ | present (existence only; not listed) | Match |
   | Study storage root | …/private/tii-cee-1 | absent, then created with owner-only mode drwx--S--- | Created |
   | Model cache | HF_HOME from the private .env, pattern /blue/<group>/<user>/cache/hf | /blue/cai6734/daniel.escalante/cache/hf; Nemotron at revision 54641c16…e153fc7 present | Match |

   Free space on /blue/cai6734: 746 GB of 2.0 TB, shared by the class.
3. Model access: `hf auth whoami` reports no Hugging Face login, and no token file exists (checked
   by existence only; no token was read). None of the nine panel models has a shared copy under
   /data/ai/models; the READMEs there list CLIP, NLP checkpoints (Megatron, UF models, GatorTron,
   Llama, LLaVA-Med, Meditron), NVIDIA NGC models and protein-folding models. All nine panel models
   are ungated per AUTHORIZATION.md section 5, so no login is needed to download them.

## 4. DEFAULT items (specification 2.7)

| # | Decision in force | Status |
|---|---|---|
| D1 | Course checkpoint Friday, December 4, 2026 | Confirmed |
| D2 | GPU-hours unconstrained November 13 to about December 13, 2026, at least 8 GPUs at once | Confirmed; 16 at once with the addendum |
| D3 | Jobs at most 24 hours, split into resumable parts | Confirmed |
| D4 | Account cai6734, GPU QOS cai6734, CPU QOS cai6734-b, partition hpg-rtx6000, study root …/tii-cee-1 | Verified live November 13, 2026 (section 3) |
| D5 | Whether logged trajectories store full prompts | Answered (7.4a): no prompts stored; the primary policy's validation rollout is regenerated once with prompt logging, accepted at ≥ 99% answer agreement |
| D6 | E6 is the one additional, disclosed use of the 240 test admissions | Confirmed (stricter) |
| D7 | Reader panel of nine ungated US-origin models; second backbone granite-3.3-8b-instruct | Confirmed |
| D8 | Public datasets downloadable from official sources | Confirmed; CKD, HCV, DDXPlus are CC BY 4.0; Pima excluded, license unverifiable (E7 amendment A1) |
| D9 | No other credentialed cohort; E11 blocked | Confirmed; accepted as a gap by Daniel November 13, 2026: E11 skipped, not run (BLOCKED-002) |
| D10 | No push | Confirmed |
| D11 | PhysioNet Credentialed Health Data Use Agreement 1.5.0 | Confirmed; DATA_USE_CHECK.md: F1 and F2 pass |
| D12 | ICML 2027 target | The 2027 call is published (7.4g); its rules replace the assumed 2026 ones |
| D13 | 15-minute watcher cycle and held SSH session within UF policy as documented | Re-read (7.4c); the daemon-server page is gone (404), so the fallback is unconfirmed; the live no-op cycle settles whether scrontab can call Slurm |
| D14 | Method plus mechanism framing | Confirmed |
| D15 | AUTHORIZATION.md valid until March 31, 2027 | Confirmed |

## 5. Running and queued jobs

| Stage | Job ID | Account | State | Since |
|---|---|---|---|---|
| e8-train-s270, part 2 | 51335545 | cai6734 | RUNNING | 1:47 PM EST |
| e8-train-s271, part 2 | 51335546 | cai6734 | RUNNING | 2:05 PM EST |
| e8-train-s272, part 2 | 51335547 | cai6734 | RUNNING | 2:47 PM EST |
| e8-train-s273, part 2 | 51335548 | cai6734 | RUNNING | 4:18 PM EST |
| e10-train-f0, part 2 | 51335549 | cai5724 | RUNNING (moved from cai6734) | 4:23 PM EST |
| e10-train-f1, part 2 | 51335550 | cai5724 | RUNNING (moved from cai6734) | 4:23 PM EST |
| e10-train-f2, part 2 | 51335551 | cai5724 | RUNNING (moved from cai6734) | 4:23 PM EST |
| e10-train-f3, part 2 | 51335552 | cai6734 | PENDING (QOSGrpGRES) | 1:45 PM EST |
| e10-train-f4, part 2 | 51335553 | cai5724 | RUNNING | 1:47 PM EST |
| e9-train-s269, part 2 | 51335554 | cai5724 | RUNNING | 1:47 PM EST |
| e9-train-s270, part 2 | 51335561 | cai5724 | RUNNING | 1:51 PM EST |
| e9-train-s271, part 2 | 51335563 | cai5724 | RUNNING | 3:24 PM EST |
| E1-C interval (outside the graph) | 51313205 | cai6734-b | RUNNING | 10:18 AM EST |

Every attempt so far: RESEARCH_LEDGER.md section 5.
Second parts resume from their epoch checkpoints; the eight that started by 4:18 PM EST step (at 4:22 PM
EST, 1,985 of 3,110 for fold 4, 1,921 to 2,526 of 3,840 for the seeds). Folds 0 to 2 started at 4:23 PM
EST on cai5724, where 3 GPUs were free (RESEARCH_LEDGER.md, "three E10 folds moved"); they begin with
the silent initial validation pass. Trainings
that wait are held by the class QOS's shared limits of 32 CPUs, 250 GB and 8 GPUs per account, with
classmates' jobs queued ahead. At 5:20 PM EST cai6734 had 29 CPUs and 180 GB in use (ours 24 and 144), cai5724 28 CPUs and 236 GB
(ours 8 and 80). Each running training uses about 1 CPU on average (sstat AveCPU against elapsed
time) and 6.4 to 6.8 GB, against a request of 4 CPUs and 16 GB (40 for the four first parts that
started before the memory change). Every later part of a training asks for 2 CPUs, and E9's for
14 GB (chosen by the study lead at 5:23 PM EST). Fold 2's first attempt was cancelled in error at
6:19 PM EST during the silent initial validation pass every training runs before its first step
(37 to 83 minutes); its retry runs on cai5724 since 6:35 PM EST (RESEARCH_LEDGER.md, "Operations,
evening", and its correction).

## 6. GPU-hours by experiment

From the runlog's ended jobs at 3:30 PM EST, November 14; a training is charged as each part ends, so
the running second parts are not yet counted.

| Experiment | Spent | Of which | Estimate | Allowance |
|---|---|---|---|---|
| Benches and probe conformance | 3.789 | p-rollout-val 0.372; Nemotron bench 3.335 over five attempts; P probe 0.081 | 10 | 30 |
| E8 | 57.402 | four first parts 57.392; eight import failures 0.010 | 105 | 210 |
| E9 | 38.459 | Granite bench 2.306; three first parts 36.146; six import failures 0.008 | 120 | 300 |
| E10 | 57.720 | five first parts 56.825; fold 2's cancelled attempt 0.884; ten import failures 0.011 | 133 | 270 |
| Total | 157.370 | | | |

Second parts run about as long as first parts, so E8 will close near 115 GPU-hours before its four
rollouts: above its estimate, within its allowance.
Estimates and allowances: AUTHORIZATION.md section 4.
