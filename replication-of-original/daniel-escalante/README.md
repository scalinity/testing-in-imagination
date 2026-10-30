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
credentialed files in private storage, never in this repository. Keep all actual
paths and site settings in the ignored `.env`:

```dotenv
CDM_DATA_SOURCE=mimic
CDM_DATA_DIR="<private directory containing the four condition pickle files>"
CDM_WORK_DIR="<new private durable working directory outside the repository>"
CDM_TRAINING_SEEDS=269
SLURM_ACCOUNT="<HiPerGator account>"
SLURM_PARTITION=gpu
SLURM_GRES=gpu:a100:2
```

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

## Interpretation and open checks

T1–T4 use per-class recall; T5 is their unweighted mean. Pooled accuracy is also
reported. T6/T7 use the upstream micro/macro-F1 implementation, including its handling
of abstentions. T8 charges **provided tests only**. Requested-test cost and unavailable
request counts are separate. The normalized twelve-test cost sum includes physical
exam. F-1–F-4 are retained and characterized by tests, including the stale confidence
scale and the 14-action budget.

All comparisons pair identical patients. Patient uncertainty uses 10,000 paired
resamples and 95% intervals; accuracy discordance uses an exact binomial test.
Between-seed uncertainty is distinct. Optional extra training seeds must be declared
before running, for example `CDM_TRAINING_SEEDS=269,270,271`; every attempted seed stays
in the ledger, including failures. A single seed cannot establish training stability.

The pinned upstream has no ReAct implementation, so B2 remains an explicit gap.
There is no substitute straw-man baseline or claimed oracle bound. C1/C2 require
real training and validation dynamics; mock behavior cannot establish improvement.
Clinical validity, actual v1.1 counts and schema variation, CUDA memory/driver
compatibility, numerical learning behavior and final real-data outcomes remain
unverified until the corresponding runs finish.

Several local compatibility repairs are explicit in every manifest and report.
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
Physical-examination strings recognized as missing by the pinned CSV parser
(including empty strings, `N/A` and `NULL`) become the existing unavailable-test
sentinel, preventing a non-string NaN from being treated as test text. Finite
out-of-range model confidences stay in hypothesis observations and are counted;
upstream excludes them from ECE without dropping patients or clamping values.
These changes preserve F-1–F-4 and the shipped primary reward. CUDA numerical
qualification of the repaired training path remains outstanding.

No upstream license is listed. This folder imports the pinned submodule and does
not vendor its source. The backbone has its own NVIDIA and Llama license terms.

## Run log

Only completed aggregate runs belong here. A synthetic row records an execution
check, never a clinical or paper-matching result.

| Date | Person | Upstream commit | Model | Config | Seed | Dataset version | Metric | Value | Paper value | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Pending qualification | Daniel Escalante | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | primary + ablations, parity + grouped | 269 | synthetic | default-population S0–S6 | NOT_RUN | n/a | 120-admission shell integration and 232 tests passed; no learned weights |
