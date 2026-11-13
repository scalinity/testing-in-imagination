# STATUS, study TII-CEE-1

Updated November 13, 2026, about 3:20 AM EST. Times are US Eastern.

## 1. Where the study is

- Milestone: M0 IN PROGRESS. M1 to M5 NOT STARTED.
- Experiments: all PLANNED except E11, BLOCKED (BLOCKED-002). No job has been submitted.
- Scientific outcome: NOT EVALUATED for every experiment.
- Open BLOCKED items: BLOCKED-002 (no independent cohort for E11; only E11 waits). BLOCKED-001 is
  closed.
- Next executable actions: finish the M0 documents and launch checks; build and test the CEE
  library (E0), the twin, and the pipeline; install the watcher after a no-op graph passes end to
  end; then E0, the M1 repairs, and the training track.

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
| D9 | No other credentialed cohort; E11 blocked | Confirmed; BLOCKED-002 |
| D10 | No push | Confirmed |
| D11 | PhysioNet Credentialed Health Data Use Agreement 1.5.0 | Confirmed; DATA_USE_CHECK.md: F1 and F2 pass |
| D12 | ICML 2027 target | The 2027 call is published (7.4g); its rules replace the assumed 2026 ones |
| D13 | 15-minute watcher cycle and held SSH session within UF policy as documented | Re-read (7.4c); the daemon-server page is gone (404), so the fallback is unconfirmed; the live no-op cycle settles whether scrontab can call Slurm |
| D14 | Method plus mechanism framing | Confirmed |
| D15 | AUTHORIZATION.md valid until March 31, 2027 | Confirmed |

## 5. Running and queued jobs

None.

## 6. GPU-hours by experiment

None spent. Estimates and allowances: AUTHORIZATION.md section 4.
