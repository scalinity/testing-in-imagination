# STATUS, study TII-CEE-1

Updated November 15, 2026, about 4:05 AM EST. Times are US Eastern.

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
  (RESEARCH_LEDGER.md, "Operations, November 14"); all twelve completed theirs from 9:54 PM to 3:52 AM
  EST; ten probes conform under D-PROBE amendment A1 (all three of
  E9's, E8 seeds 270 to 272, E10 folds 4, 2, 0 and 1), and the last two rollouts run;
  E11 SKIPPED (accepted gap, not run); all others PLANNED.
- Scientific outcome: NOT EVALUATED for every experiment.
- Open BLOCKED items: none. Accepted gap: BLOCKED-002, no independent cohort, so E11 does not run
  (accepted by Daniel November 13, 2026). BLOCKED-001 is closed.
- AI workers: the Cloudflare 403 of 4:47 AM was cleared by a fresh login. Fable 5.2 workers are at
  their weekly limit until November 18, 5 AM EST; the lead session and the HiPerGator watcher are
  unaffected.
- Next executable actions: the last two rollouts (E8 seed 273, E10 fold 3) and their
  probes run (section 10.1; P's probe conforms under D-PROBE amendment
  A1, 610/613, and so do the first ten policies'), and E10's cohort-X check runs after fold 3's rollout, about 5:30 AM EST; when E1-C ends, its
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
| e8-rollout-s273 | 51421837 | cai6734 | RUNNING | 3:31 AM EST |
| e10-rollout-f3 | 51423564 | cai5724 | RUNNING (moved from cai6734) | 4:04 AM EST |
| E1-C interval (outside the graph) | 51313205 | cai6734-b | RUNNING | 10:18 AM EST |

Every attempt so far: RESEARCH_LEDGER.md section 5.
All twelve second parts have completed, from 9:54 PM to 3:52 AM EST, each following its plan. Each
policy's rollout (1 GPU, 4 CPUs and 16 GB for up to 2 hours; 15 to 76 minutes) and probe (3 to 9
minutes) follow, and ten probes conform under D-PROBE amendment A1 (RESEARCH_LEDGER.md, "Operations,
November 15, 2026", 12:50 to 4:05 AM EST). The last two rollouts run: E8 seed 273's on cai6734 since
3:31 AM EST, nearly done, and E10 fold 3's on cai5724 since 4:04 AM EST, about 75 minutes. Their
probes follow, and E10's cohort-X check runs once fold 3's rollout is done, about 5:30 AM EST. The
watcher picks an account only at submission, counting only this user's jobs, so it has sent fifteen
GPU jobs to cai6734 while classmates held its GPUs or CPUs; each was moved to cai5724 under its lock
and started there as soon as a GPU was free, every one since midnight within 3 minutes of its move.
Jobs that wait are held by the class QOS's shared limits of 32 CPUs, 250 GB and 8 GPUs per account,
with classmates' jobs queued ahead.

## 6. GPU-hours by experiment

From the runlog's ended jobs at 3:52 AM EST, November 15; a training is charged as each part ends, so
the running second parts are not yet counted.

| Experiment | Spent | Of which | Estimate | Allowance |
|---|---|---|---|---|
| Benches and probe conformance | 4.783 | p-rollout-val 0.372; Nemotron bench 3.335 over five attempts; probes: P 0.081, E9 seeds 269 to 271 0.055, 0.050 and 0.056, E8 seeds 270 to 272 0.076, 0.083 and 0.090, E10 folds 4, 2, 0 and 1 0.141, 0.148, 0.147 and 0.146 | 10 | 30 |
| E8 | 102.522 | four first parts 57.392; four second parts 43.534; rollouts of seeds 270 to 272 0.439, 0.598 and 0.548; eight import failures 0.010 | 105 | 210 |
| E9 | 66.171 | Granite bench 2.306; three first parts 36.146; three second parts 26.841; three rollouts 0.870; six import failures 0.008 | 120 | 300 |
| E10 | 110.371 | five first parts 56.825; five second parts 47.903; rollouts of folds 4, 2, 0 and 1 4.748; fold 2's cancelled attempt 0.884; ten import failures 0.011 | 133 | 270 |
| Total | 283.846 | | | |

Second parts run shorter than first parts (E8's four took 9.677 to 11.454 GPU-hours, against
14.35 on average for its first parts), so E8 closes near 103 GPU-hours with its last rollout, within
its estimate of 105. Probes are charged to the benches allowance (D-PROBE), rollouts to their experiment.
Estimates and allowances: AUTHORIZATION.md section 4.
