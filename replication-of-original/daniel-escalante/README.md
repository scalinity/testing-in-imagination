# LA-CDM replication

A **re-implementation on a different backbone** of LA-CDM, using the pinned upstream
implementation for data parsing, episodes, rewards, scheduling and metrics.
The experiment follows the [claims ledger](../../REPRODUCTION.md) and
[group rules](../README.md). See [PLAN.md](PLAN.md) for the architecture and deviations.

The approved policy and HPI summarizer are
`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`, pinned to
`54641c1611fcff44fa4865626462445e0a153fc7`. The primary reward uses the shipped
`incorrect_reward=0.0`; −1.0 is a separate arm. No result is labelled as a reproduced
paper table. CPU synthetic runs test the plumbing; they do not train weights.

## Run now on synthetic data

From the repository root:

```sh
cd replication-of-original/daniel-escalante
uv sync --frozen --python 3.12
cp -n .env.example .env
bash scripts/run_all.sh --dry-run
bash scripts/run_all.sh
bash scripts/stage.sh validate
make test
```

The default synthetic population has 2,400 invented admissions with the
957/648/257/538 class mix, missing tests and repeated subjects. It is never derived
from patient data. The default working directory is private temporary storage;
set `CDM_WORK_DIR` in `.env` to keep artifacts on durable storage. Working and input
directories must be outside the repository. Repeating the command verifies output
hashes and resumes completed stages. A changed manifest fails explicitly so previous
results remain intact. Use a new work directory for a different experiment.

To shorten a local smoke run, set `CDM_SYNTHETIC_N=120` in `.env`. The bootstrap
still uses 10,000 paired resamples. `--fake-summarizer` is accepted only in synthetic
mode. No full-stack installation, model download or GPU is needed for CPU checks.

## Day one: credentialed MIMIC data on HiPerGator

Use an approved HiPerGator checkout with the submodule already present. Put the
credentialed files in private storage, never in this repository.

The PhysioNet v1.1 release ships CSVs, while the pinned splitter reads four condition
pickles. Upstream defers that step to Hager et al.'s `ConvertPhysionet.py`
(`paulhager/MIMIC-Clinical-Decision-Making-Dataset@4d33f98`, MIT); this script builds the
same dictionaries without its unused spaCy import and never writes to the raw directory:

```sh
.venv/bin/python -I scripts/build_condition_pickles.py \
  --raw-dir "<private v1.1 download>" --out-dir "<new private directory>"
```

It checks every digest in `SHA256SUMS.txt` itself: PhysioNet separates digest and name
with one space, which `shasum -c` rejects while still exiting 0. The output's
`.conversion.json` records the checksum-manifest digest and every output digest, so a
second machine can confirm it built identical pickles. Point `CDM_DATA_DIR` at the
output directory. Keep all actual paths and site settings in the ignored `.env`:

```dotenv
CDM_DATA_SOURCE=mimic
CDM_DATA_DIR="<private directory containing the four condition pickle files>"
CDM_WORK_DIR="<new private durable working directory outside the repository>"
CDM_TRAINING_SEEDS=269
CDM_RUNTIME=blackwell
SLURM_ACCOUNT="<HiPerGator account>"
SLURM_QOS="<HiPerGator account>"
SLURM_PARTITION=hpg-rtx6000
SLURM_GRES=gpu:rtx_pro_6000:1
```

This is the configuration of the completed credentialed run. HiPerGator's GPU
partitions are `hpg-turin` (L4, 24 GB), `hpg-rtx6000` (RTX PRO 6000, 96 GB) and
`hpg-b200` (B200); none has A100s. The pinned torch 2.6.0+cu124 build contains kernels up
to `sm_90` only, so the pinned stack runs on L4 but not on the Blackwell partitions, and
an L4 cannot hold the 16 GB model at upstream's vLLM memory setting. Partition and GRES
are submission flags, which override the scripts' `#SBATCH` defaults.

A class QOS caps the whole class at 8 GPUs, 32 CPUs and 250 GB of memory. Its burst
QOS (`<account>-b`) adds 2,250 GB and 288 CPUs but no GPUs, so the CPU-only stages S4,
S5 and S6 can run there (`scontrol update JobId=<id> QOS=<account>-b` on a queued job)
when other class jobs fill the memory cap. Host memory peaked at 6.8 GB per GPU job, so
16 GB requests suffice.

`CDM_RUNTIME=blackwell` selects `runtime-blackwell/`, which keeps every upstream package
and changes 28 pins to the CUDA 12.8 set of the course's `lacdm-bw` environment: torch
2.7.0+cu128, vLLM 0.9.2, xgrammar 0.1.19 and transformers 4.51.1, with TRL 0.15.2 and
numpy unchanged. One 96 GB RTX PRO 6000 then holds the policy and vLLM engine at the
upstream memory settings, as the authors' single 48 GB A40 did. S0 records every
substituted version, and every manifest names the deviation. The default
`CDM_RUNTIME=pinned` keeps the exact upstream pins. Changing it needs a new work
directory.

Only `CDM_DATA_SOURCE` changes execution mode. Both modes use the same stages,
data contract, config composition and reporting. Set input and output locations
once for the machine. Do not reuse a synthetic work directory for MIMIC.

```sh
cd replication-of-original/daniel-escalante
uv sync --frozen --python 3.12
${EDITOR:-vi} .env
# Validate before requesting GPU time. No model is loaded by this command.
bash scripts/stage.sh validate
# Inspect the complete SLURM dependency chain before submission.
.venv/bin/python scripts/submit_slurm.py --dry-run
# S0 runs on the submission host, then S1–S6 run with afterok dependencies.
.venv/bin/python scripts/submit_slurm.py
```

S0 synchronizes the local CUDA runtime lock with `uv sync --frozen`; later scripts
automatically select that private runtime environment. The upstream pin ships no
lockfile. `runtime/pyproject.toml` retains its exact dependency declarations, while
`runtime/uv.lock` resolves them. One package source override builds `xgrammar==0.1.16`
from the publisher's immutable same-version source because the package index no
longer serves that release. This packaging difference is recorded in S0. Installation
of that source build and actual CUDA execution must be qualified on HiPerGator.

For a preallocated two-GPU session, the sequential equivalent is:

```sh
bash scripts/run_all.sh
```

Individual scripts `s0_env.sh` through `s6_report.sh` accept `--dry-run`.
S1/S3/S4/S5 have corresponding SLURM files. Account, partition, GPU type and private
log destinations are submission settings. GPU capacity must be checked on the
selected HiPerGator nodes. No cloud resources are provisioned by these scripts.

After S6 succeeds, export only aggregate metrics and the rendered report:

```sh
bash scripts/stage.sh export
bash scripts/stage.sh check-git
git status --short
```

`results/mimic/metrics.json` and `results/mimic/REPORT.md` are the intended reviewable
outputs. Record T1–T8 and baseline values from that aggregate file in the run-log
table below, together with the exact run configuration and any W&B run IDs. Never
copy private manifests, split CSVs, episode rows, generations, weights or W&B dumps
into `results/`. Synthetic exports go into `results/synthetic/` and carry a banner.

The aggregate provenance includes the code commit, a content fingerprint, stage
manifest identifiers, the runtime lock digest and W&B run IDs. Commit metadata and
live hardware observations are recorded but do not change a content-identical run's
resume identity. Updating this run log or committing aggregate exports therefore
does not authorize another test evaluation.

## What each stage checks

| Stage | Behavior | Private outputs |
|---|---|---|
| S0 | Python, exact upstream SHA, model pin, composed config, dependencies, hardware | environment record, manifest |
| S1 | raw schema, seed-269 parity split, mapping, summaries, grouped sensitivity | raw/split files, schema and leakage audits |
| S2 | approved model, system-message and configuration wiring | resolved configs for every arm/seed/protocol |
| S3 | zero-shot baseline, same test admissions as trained arms | episode rows and immutable evaluation receipt |
| S4 | independent primary and three ablations, train/val only | checkpoints, selection records, failed/successful seed ledger |
| S5 | once-only final-model evaluation, identical test admissions | episode rows and evaluation receipts |
| S6 | upstream metrics, paired intervals, significance and honest write-up | aggregate JSON and REPORT.md |

Every stage has a manifest. Separate training/evaluation manifests capture each
arm's actual config. Completed artifacts are hash checked before resume; an
interrupted evaluation after admission is left for inspection, because blindly
replaying it would violate the once-only test protocol. An interrupted training
run resumes from its owned checkpoints. Failures stay in the ledger.
Each checkpoint includes the environment's separate confidence-pair RNG state.
Continuation refuses a checkpoint without that state rather than restarting its
sampling stream.
Training generation derives a separate seed for each optimizer step, generation
call and rollout position. Eight identical GRPO prompts therefore receive eight
independent request streams; continuation reconstructs them from the saved step.
Kernel advisory locks release ownership after scheduler termination; no stale PID
file needs to be removed to resume training. Evaluation admission markers remain
separate and preserve the once-only test rule after interruption.

The parity split reuses upstream's `RANDOM_STATE=269`. D-4 in the shared ledger says
there is no canonical seed; the executable pin contradicts it. Subject overlap is
reported for the parity split. If subject IDs exist, a separate grouped split runs
the full arm matrix as a sensitivity analysis. Without subject IDs, subject leakage
is unassessable and the report says so. Admission overlap always fails on load.
The v1.1 release has no `subject_id`, so on real data the grouped split does not run
and subject leakage of the parity split is unassessable.

Two properties of the real release differ from the synthetic population. Its lab mapping
lists the seven panels and 111 other test groups as rows with a blank `itemid`; validation
resolves them by label, as the upstream parser does. Of its 5,960 radiology reports, 373
use modalities outside the four configured imaging tests (MRCP, ERCP, CTU and others).
The upstream parser skips those reports, so validation counts them in the private audit
rather than rejecting the release.

## Interpretation and open checks

Results come from one credentialed run (graph `7fcd9e5`, parity split, training seed 269,
240 test admissions) on a different 8B backbone and summarizer, the CUDA 12.8 Blackwell
runtime, and greedy evaluation decoding. They are a replication attempt, not a
reproduction of the paper's numbers: no tolerance was agreed and every deviation below
changes what the model sees or how it decodes. `REPRODUCTION.md` gives each claim's
status against our 95% interval with its reason.

Metric definitions. T1–T4 are per-class recall; T5 is their unweighted mean. Pooled
accuracy is also reported. T6/T7 use the upstream micro/macro-F1 implementation, in which
an episode without a valid diagnosis counts as wrong. T8 charges **provided tests only**;
requested-test cost and unavailable requests are reported separately. All comparisons
pair identical patients: 10,000 class-stratified paired resamples give 95% intervals, and
an exact McNemar test with Holm adjustment (family of four) covers pooled accuracy.

What the counts show (`results/mimic/REPORT.md`, supplementary section):

- **Undiagnosed episodes drive most errors.** An episode that ends with a malformed
  action, the 14-action budget or the 4,096-token trajectory limit has no diagnosis.
  Greedy zero-shot left 194 of 240 test episodes undiagnosed (613 repeated test
  requests); the trained primary arm left 29 (121 repeated requests).
- **Cholecystitis and pancreatitis.** Primary called 10 of 65 cholecystitis patients
  pancreatitis and left 6 undiagnosed; it left 14 of 54 pancreatitis patients
  undiagnosed and called 7 cholecystitis.
- **Cost reward (C3).** `TestCostReward` is part of the primary training reward on all
  3,840 steps (the logged total equals the sum of its components), but it is small: its
  mean was −0.09 to −0.11 per rollout against 0.17–0.53 for the diagnosis reward and
  0.26–0.69 for the format reward, and it acts only in the decision-agent phase (100 of
  every 300 updates). Unavailable requests cost nothing (F-2). The no-cost arm was not
  cheaper in our run.
- **Decoding (D-6).** Upstream evaluates at its training temperature, 1.0; this run
  pre-specified greedy decoding. A post-hoc check re-evaluated every locked model once on
  the same test set at 1.0 (greedy stays primary). Every arm scored lower: zero-shot
  mean recall 12.71 → 9.19 (paired −3.52 [−9.38, 2.37]) and primary 73.09 → 59.30
  (−13.79 [−21.70, −6.24]). Sampling removed most repeated requests (zero-shot 613 → 44)
  but left as many or more episodes undiagnosed (zero-shot 194 → 202, primary 29 → 55),
  with far fewer valid test requests (zero-shot 1,521 → 406). Decoding therefore does not
  explain the B1 gap. The code audit found no bug, so the backbone and summarizer (D-5,
  D-3) remain the untested candidates. C4 holds under both decodings (primary −
  decision_only T5 +18.66 [9.94, 27.21], Holm p 0.000049); the D-1 ablation difference
  does not (+3.55 [−5.68, 12.76], Holm p 0.474).

Open checks and honest gaps:

- **One training seed.** Between-seed uncertainty is unassessable; every interval covers
  test-patient sampling only.
- **No `subject_id` in v1.1.** Subject leakage of the parity split and the
  subject-grouped sensitivity split are unassessable, never reported as zero.
- **B2 ReAct** has no implementation at the upstream pin; no substitute is labelled ReAct.
- **Radiology count.** v1.1 has 5,960 radiology reports where upstream states 5,959; the
  one-report difference is unexplained.
- **Deviations named in every report.** D-1 keeps the shipped incorrect reward 0 as
  primary (−1 is the `incorrect_minus1` ablation). D-3 and D-5 replace the summarizer and
  the policy with `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` at `54641c16`. The Blackwell
  runtime changes 28 upstream pins (torch 2.7.0+cu128, vLLM 0.9.2). D-6 evaluates greedily
  where upstream samples at temperature 1.0. The local compatibility repairs are listed
  below and in every manifest.

### Running an independent second replication

A teammate (for example Hugo or Edwin) can repeat the run only from their own
credentialed access; no data, private directory, model output or work directory may be
shared, and nothing patient-level may be sent to a model service.

1. Complete PhysioNet credentialing, CITI training and the MIMIC-IV-Ext-CDM v1.1 data
   use agreement, then download the release with their own account and keep it
   owner-only outside every repository.
2. Clone this repository with its submodule (`lacdm/upstream` at `3f435a10`) at the commit
   named in the run log, and build the condition pickles with
   `scripts/build_condition_pickles.py` (all 12 digests must match).
3. Write their own untracked `.env` as in *Day one*, with a fresh `CDM_WORK_DIR`. Keep
   seed 269 to compare like with like, or declare a new training seed to measure
   between-seed variation.
4. Run `bash scripts/stage.sh validate`, then `scripts/submit_slurm.py --parallel` after
   the course staff approve the GPU budget below.
5. After S6, run `stage.sh export` and `check-git`, then compare their aggregate
   `results/mimic/metrics.json` with this run's committed one.

GPU budget from the logged durations of this run (RTX PRO 6000):

| Work | GPU-hours | Wall time |
|---|---|---|
| HPI summaries, 8 shards (once per summarizer) | 3.3 | 25 min on 8 GPUs |
| Zero-shot baseline (once per backbone) | 1.5 | 1 h 33 min |
| Training, four arms × 3,840 steps (per training seed) | 103.6 | 28 h on 4 GPUs (arms 24 h 53 min – 27 h 58 min) |
| Test evaluation, four arms (per training seed) | 1.8 | 24–33 min each |

One more training seed therefore costs about 105 GPU-hours (about 29 h on four GPUs).
Two more seeds, the minimum for a between-seed interval with three seeds, cost about 211.
A different backbone (for example the D-5 fallback `meta-llama/Llama-3.1-8B-Instruct`)
also regenerates the summaries and the zero-shot baseline: about 110 GPU-hours. Class
jobs share 8 GPUs and 250 GB, so queue time adds to every estimate; multi-GPU or
over-24-hour jobs need the course staff's approval first.

Local compatibility repairs, explicit in every manifest and report:
Singleton validation/evaluation prevents upstream's first-patient label from being
reused across a batch. Hypothesis SFT examples are batched independently of the
eight decision rollouts, so deduplication cannot discard them; each batch-two
forward is backpropagated immediately. The hypothesis objective is token weighted
over the complete group and retains the upstream sum-of-microbatch gradient scale.
CPU tests compare actual accumulated gradients with a direct reference for 1, 3,
8 and 11 hypothesis examples. Each optimizer update uses one phase, giving the
requested 100/100/100 updates; upstream changes phase on the last microbatch of a
boundary update. The phase derives from saved optimizer-step count on resume.
Inactive objectives do not run extra forwards, which changes dropout RNG use from
the broken upstream batching path. Weight-transfer failures propagate explicitly.
Validation runs once per epoch with best-metric tracking off, and the first-hypothesis
placeholder is filled as a literal substitution (both found on real data; see the run
log). Physical-examination strings recognized as missing by the pinned CSV parser
(including empty strings, `N/A` and `NULL`) become the existing unavailable-test
sentinel, preventing a non-string NaN from being treated as test text. Finite
out-of-range model confidences stay in hypothesis observations and are counted;
upstream excludes them from ECE without dropping patients or clamping values.
A trained adapter is evaluated as upstream's `evaluate.py` and training validation do:
loaded with upstream's `load_model`, merged into the base weights and copied into a vLLM
engine without LoRA support. A copy that leaves any engine parameter unloaded fails.
vLLM's own LoRA kernels are not used; their Triton 3.3.0 build fails on RTX PRO 6000
(`sm_120`) GPUs. These changes preserve F-1–F-4 and the shipped primary reward.

No upstream license is listed. This folder imports the pinned submodule and does
not vendor its source. The backbone has its own NVIDIA and Llama license terms.

## Run log

Only completed aggregate runs belong here. A synthetic row records an execution
check, never a clinical or paper-matching result. A real-data row without a metric
records how far the credentialed run has progressed.

| Date | Person | Upstream commit | Model | Config | Seed | Dataset version | Metric | Value | Paper value | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, grouped | 269 | synthetic (2,400 raw) | T5: Mean of four class recalls | 25 | 81.3 | SYNTHETIC mock; 95% CI [25.000000, 25.000000]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, grouped | 269 | synthetic (2,400 raw) | T8: Average provided-test cost | 979.367089 | 1295.61 | SYNTHETIC mock; 95% CI [894.620234, 1064.108022]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity | 269 | synthetic (2,400 raw) | T1: Appendicitis recall | 100 | 93.1 | SYNTHETIC mock; 95% CI [100.000000, 100.000000]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity | 269 | synthetic (2,400 raw) | T2: Cholecystitis recall | 0 | 83.6 | SYNTHETIC mock; 95% CI [0.000000, 0.000000]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity | 269 | synthetic (2,400 raw) | T3: Diverticulitis recall | 0 | 75.0 | SYNTHETIC mock; 95% CI [0.000000, 0.000000]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity | 269 | synthetic (2,400 raw) | T4: Pancreatitis recall | 0 | 73.5 | SYNTHETIC mock; 95% CI [0.000000, 0.000000]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity | 269 | synthetic (2,400 raw) | T5: Mean of four class recalls | 25 | 81.3 | SYNTHETIC mock; 95% CI [25.000000, 25.000000]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity | 269 | synthetic (2,400 raw) | T6: Micro F1 | 40 | 84.1 | SYNTHETIC mock; 95% CI [39.506173, 40.336134]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity | 269 | synthetic (2,400 raw) | T7: Macro F1 | 14.285714 | 81.3 | SYNTHETIC mock; 95% CI [14.159292, 14.371257]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity | 269 | synthetic (2,400 raw) | T8: Average provided-test cost | 1037.866667 | 1295.61 | SYNTHETIC mock; 95% CI [954.023237, 1117.478992]; no weight updates |
| 2026-10-30 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | zero_shot, parity | 269 | synthetic (2,400 raw) | B1: mean class recall | 25.0 | 64.5 | SYNTHETIC mock; no trained baseline or learning claim |
| 2026-11-05 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | all arms, parity | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | Raw schema and cohort | 957/648/257/538 | 957/648/257/538 | REAL DATA; validation PASS on macOS and HiPerGator; 373 reports in unconfigured modalities; subject_id absent |
| 2026-11-05 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | all arms, parity | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | S0 environment | PASS | — | REAL DATA; HiPerGator `login7`, commit `8f4e3c2`, receipt `92a05616`; no metric |
| 2026-11-05 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | split preflight, parity | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | Train/val/test admissions | 1,920/240/240 | — | REAL DATA CPU preflight outside stage receipts; zero admission overlap; upstream parser read all 2,400 |
| 2026-11-05 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | all arms, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | S0 environment | PASS | — | REAL DATA; commit `e65cb03`, CUDA 12.8 runtime (28 substituted pins); RTX PRO 6000 smoke job 49872040 passed summarize, train and both evaluate paths; no metric |
| 2026-11-05 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | all arms, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | S0 environment | PASS | — | REAL DATA; commit `b7190c5`, receipt `ba0e3fde`; the sequential S1–S3 jobs 49872927/49872929/49872930 were cancelled before writing any receipt; no metric |
| 2026-11-05 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | S1–S3, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | S1–S3 stages | PASS | — | REAL DATA; `b7190c5` graph: shards 49874606–49874613, S1 49874614 (30 s, all 2,400 summaries cached), S2 49874615, S3 49874616 (1 h 32 min, receipt `70fc5c16`). Train arms 49874617–49874620 and S4–S6 cancelled: each arm's first scheduled validation would raise `KeyError` (no `eval_loss`); no metric exported |
| 2026-11-06 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | S0–S3, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | S0–S3 stages | PASS | — | REAL DATA; `4d22c70` graph: S0 PASS, smoke 49889947, shards 49889983–49889991, S1 49889992 (2,400 summaries cached), S2 49889993, S3 49889994 (1 h 33 min, receipt `17f05b4b`). Training cancelled: decision_only 49889996 hit CUDA OOM at step 333 (12.31 GiB needed, 12.27 GiB free, 4.70 GiB fragmented; resumed as 49924900), and primary 49889995, no_cost 49889997 and incorrect_minus1 49889998 all raised `KeyError` at step 1,340 (upstream `str.format` on episode text with braces); no metric exported |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | all arms, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | S0–S6 stages | PASS | — | REAL DATA; `7fcd9e5` graph (literal first-hypothesis injection, expandable allocator segments in every stage), all 22 jobs COMPLETED 0:0: S0 PASS 4:04 PM EST Nov 6; GPU smoke 50010187 PASS; shards 50010230–50010237, S1 50010238, S2 50010239, S3 50010240 (receipt `f323a460`); train arms 50010241–50010244 (primary 25 h 5 min, decision_only 27 h 58 min, no_cost 25 h 42 min, incorrect_minus1 24 h 53 min; host memory peak 6.8 GB); S4 50010245 (receipt `05601581`); evaluate arms 50010246–50010249 (24–33 min each); S5 50010250 (receipt `a03cb630`); S6 50010251 (receipt `800c18a6`). Host-memory requests of incorrect_minus1 and the four evaluate arms were lowered from 40G to 16G with `scontrol` while queued. S4–S6 (CPU only) moved to the burst QOS `cai6734-b` after other jobs filled the class QOS's 250 GB memory cap; the evaluate arms waited 6 h 40 min to 7 h 20 min for class memory |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T1: Appendicitis recall | 90.62 | 93.1 | REAL DATA; 240 test admissions; 95% CI [84.38, 95.83]; train 50010241, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T2: Cholecystitis recall | 72.31 | 83.6 | REAL DATA; 95% CI [61.54, 83.08]; train 50010241, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T3: Diverticulitis recall | 72.00 | 75.0 | REAL DATA; 95% CI [56.00, 88.00]; train 50010241, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T4: Pancreatitis recall | 57.41 | 73.5 | REAL DATA; 95% CI [44.44, 70.37]; train 50010241, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T5: Mean of four class recalls | 73.09 | 81.3 | REAL DATA; 95% CI [66.65, 79.19]; delta −8.21 [−14.65, −2.11]; train 50010241, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T6: Micro F1 | 81.15 | 84.1 | REAL DATA; 95% CI [76.47, 85.53]; train 50010241, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T7: Macro F1 | 78.20 | 81.3 | REAL DATA; 95% CI [72.32, 83.44]; train 50010241, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T8: Average provided-test cost | 1439.58 | 1295.61 | REAL DATA; USD; 95% CI [1341.50, 1537.65]; train 50010241, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | zero_shot, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | B1: mean class recall | 12.71 | 64.5 | REAL DATA; 95% CI [8.04, 17.86]; S3 50010240, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | zero_shot → primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | C1: hypothesis accuracy | 29.59 → 81.67 | 75.7 → 81.9 | REAL DATA; held-out endpoints, not a training curve; 622 primary triplets; S3 50010240, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | zero_shot → primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | C2: ECE | 0.3022 → 0.1281 | 0.069 → 0.037 | REAL DATA; 622 ECE-eligible primary triplets; S3 50010240, evaluate 50010246, S6 50010251 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | no_cost → primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | C3: cost USD; mean recall | $1321.33 → $1439.58; 70.81 → 73.09 | $1427.85 → $1295.61; 82.3 → 81.3 | REAL DATA; paired T8 difference 118.25 [−0.25, 239.86]; McNemar Holm p 0.551; train 50010243/50010241, evaluate 50010248/50010246 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | decision_only → primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | C4: mean class recall | 57.49 → 73.09 | — | REAL DATA; paper Table 3 values absent from REPRODUCTION.md; paired T5 difference 15.59 [6.34, 24.60]; McNemar Holm p 0.00056; train 50010242/50010241, evaluate 50010247/50010246 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | incorrect_minus1 (D-1 ablation), parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T5: Mean of four class recalls | 59.38 | — | REAL DATA; T8 $1619.27; primary − ablation T5 13.71 [5.32, 21.93]; McNemar Holm p 0.0031; train 50010244, evaluate 50010249 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | zero_shot, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | B1 gap: code audit against upstream | no code bug; one decoding difference (D-6) | 64.5 | REAL DATA, counts only; prompt, chat template, parser, action names, stop string and 256-token budget match `3f435a10`; system message differs as D-5 requires; upstream evaluates at its training temperature 1.0, this run greedily. Greedy zero-shot: 194/240 undiagnosed (by valid requests before the episode ended: 0: 23, 1–3: 27, 4–7: 66, 8–13: 56, 14+: 22), 613 repeated test requests, 27 of 46 given diagnoses correct, 904 of 1,095 hypotheses cholecystitis |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | C3: cost reward wired into training | yes | — | REAL DATA; composed configs: primary and decision_only include `TestCostReward`, no_cost omits it; logged total reward equals the sum of its components on every logged step (max residual 0.0). Primary per 960-step block: cost −0.087, −0.099, −0.104, −0.106; diagnosis 0.166, 0.296, 0.453, 0.535; format 0.259, 0.416, 0.594, 0.692 |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | all arms (greedy), parity, blackwell runtime | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | 4×4 confusion counts with an undiagnosed column | added to `REPORT.md` | — | REAL DATA, counts only, from the saved S5 test episodes after S6 (no model run); primary undiagnosed 29, zero_shot 194; code `4b79683`; CPU report job 50282958 (burst QOS, 1:28 PM EST) |
| 2026-11-08 | Daniel Escalante | — | — | HiPerGator work directories | — | — | Superseded work directories deleted | 4 | — | `-4d22c70` (5.3 GB unique), `-b7190c5` (187 MB), `-e65cb03` (158 MB), `-d434552` (80 KB) removed 1:30 PM EST after zero references or symlinks from `-7fcd9e5` were confirmed; about 5.6 GB freed (hard-linked runtime files are shared with the kept 15 GB directory); `-7fcd9e5` untouched |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | zero_shot, parity, blackwell runtime, temperature 1.0 (D-6 check) | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | B1: mean class recall | 9.19 | 64.5 | REAL DATA, post-hoc decoding check, greedy stays primary; 95% CI [5.55, 13.44]; minus greedy −3.52 [−9.38, 2.37]; 202/240 undiagnosed, 44 repeated requests; job 50282940 (code `4b79683`, 1:29–2:11 PM EST); decoding does not explain the B1 gap |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary, parity, blackwell runtime, temperature 1.0 (D-6 check) | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | T5: Mean of four class recalls | 59.30 | 81.3 | REAL DATA, post-hoc; 95% CI [52.23, 66.19]; minus greedy −13.79 [−21.70, −6.24]; T1–T4 65.62/66.15/48.00/57.41, T6 70.12, T7 66.91, T8 $961.71; 55/240 undiagnosed; hypothesis accuracy 75.41%; job 50282941 (code `d3ef9de`, 16 min) |
| 2026-11-08 | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary vs ablations, parity, blackwell runtime, temperature 1.0 (D-6 check) | 269 | MIMIC-IV-Ext-CDM v1.1 (2,400 raw) | Paired comparisons (Holm, family of four) | C4 holds; C3 and D-1 not observed | — | REAL DATA, post-hoc; primary − decision_only T5 +18.66 [9.94, 27.21], Holm p 0.000049; primary − no_cost cost +$82.94 [−30.33, 195.84], Holm p 0.538; primary − incorrect_minus1 T5 +3.55 [−5.68, 12.76], Holm p 0.474; T5 at 1.0: decision_only 40.64, no_cost 61.46, incorrect_minus1 55.75; jobs 50282942–50282944 (16–21 min), report 50282959 (burst QOS, 2:46–2:58 PM EST) |

Verification: **264 tests passed** at `d3ef9de` (181 here and all 83 in `lacdm/tests`).

Credentialed aggregates (run `7fcd9e5`; no patient rows or identifiers):
[report](results/mimic/REPORT.md), [full-precision metrics](results/mimic/metrics.json) and
[supplementary counts and decoding check](results/mimic/supplementary.json). A narrow
`.gitignore` exception tracks exactly these three files; `check-git` scans them.

The default-population synthetic run completed S0–S6 on committed code
`e56e0de83444b0c5925dc3f1ed436323d2540024`. Both protocols ran all five arms with
10,000 paired resamples. The parity test set has 240 admissions; the grouped test
set has 237. The parity audit found 175 subjects shared across at least two splits
(80 shared between train and test); the grouped audit found zero subject overlap.
No learning, CUDA execution or clinical result is established by this run.

Synthetic aggregate artifacts: [report](results/synthetic/REPORT.md) and
[full-precision metrics](results/synthetic/metrics.json).
