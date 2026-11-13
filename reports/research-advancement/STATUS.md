# STATUS, study TII-CEE-1

Updated November 13, 2026, about 8:05 AM EST. Times are US Eastern.

## 1. Where the study is

- Milestone: M0 COMPLETE. M1 IN PROGRESS. M2 IN PROGRESS. M3 to M5 NOT STARTED.
- M0: CLOSED (MILESTONES/M0.md). The live no-op graph ran end to end on HiPerGator and the watcher
  removed its own entry at 5:45 AM; launch check 7.4c is answered and G0 is met in full.
- M1: the twin repairs (REVIEW-002, -006, -011) are verified (GATE_DECISIONS/G1-twin.md); the
  section 8.3 preflight and manifests are built (tii-cee/tiicee/pipeline); the training-track
  adapters, the resume guarantee (D-CLIN-RESUME) and gate G_resume are committed; the training
  track's first graph is installed on HiPerGator (G1 training part met). Launch check 7.4a is
  accepted: 240 of 240 final answers and digests match (LAUNCH_CHECKS.md). REVIEW-001 (for E3) and
  REVIEW-008 (for E5) are not started.
- M2: E0 COMPLETED, PASS. E1 COMPLETED at a4f34bd except the E1-C interval, running locally
  (about 20 hours). G2b MET (projection repairs drift); G2c NOT MET (CEE-Scale stays in the twin;
  charts get projection only, stateful track). G2a waits for E1-C.
- Experiments: E0 COMPLETED; E1 COMPLETED except E1-C; E8 and E10 IN PROGRESS (data copy and folds
  done; training waits on G_resume); E11 SKIPPED (accepted gap, not run); all others PLANNED.
- Scientific outcome: NOT EVALUATED for every experiment.
- Open BLOCKED items: none. Accepted gap: BLOCKED-002, no independent cohort, so E11 does not run
  (accepted by Daniel November 13, 2026). BLOCKED-001 is closed.
- AI workers: the Cloudflare 403 of 4:47 AM was cleared by a fresh login. Fable 5.2 workers are at
  their weekly limit until November 18, 5 AM EST; the lead session and the HiPerGator watcher are
  unaffected.
- Next executable actions: the Nemotron resume bench decides gate G_resume; then the Granite bench
  and the E8 to E10 training stages, with a redeploy; the E1-C interval finishes locally (about 20
  hours from 5:58 AM), then the rules and manifest are re-run and G2a is evaluated; D-REGIME's
  validation in the twin; REVIEW-001 and REVIEW-008.

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
| bench-nemotron | 51161948 | cai6734 | RUNNING | 7:05 AM EST |

Every attempt so far: RESEARCH_LEDGER.md section 5.

## 6. GPU-hours by experiment

Benches: 1.03 spent of an estimate of 10 and an allowance of 30 (p-rollout-val 0.372; the rest is
the running Nemotron bench), as of 8:00 AM EST from state/status.json. E8, E9, E10: none spent.
Estimates and allowances: AUTHORIZATION.md section 4.
