# AUTHORIZATION for study TII-CEE-1

Prepared November 13, 2026. Every decision in this file was confirmed by Daniel Escalante on November 13, 2026.

This file is the single standing authorization for the executing agent under `reports/research-advancement/SPEC_RESEARCH_ADVANCEMENT.md` version 3.1. The agent may read it. The agent may never sign, edit, extend, or reinterpret it. Anything not written here is not authorized: the agent records it in BLOCKED.md with the exact request and continues independent work.

This file is never edited after it is hashed. It carries no signature. Section 12 says how it is signed.

## 1. Scope

- Study: TII-CEE-1, specification version 3.1, kickoff version 3.1.
- Repository: scalinity/testing-in-imagination, which is public. Task branch: `tii-cee-1/2026-11-13`, created from `main` at `e5ad93fad5345f211cedc05c3dc491b101c63fa5`.
- Study directory: `reports/research-advancement/`.
- Valid from the date of the signature statement until **March 31, 2027**, or until this file is removed, renamed, or changed, whichever comes first. Removing or renaming it is how to revoke it.

## 2. What this file cannot do

It cannot waive the PhysioNet data-use terms or the project's rules for approved private storage. The two new data flows of specification section 20.4 run only if DATA_USE_CHECK.md confirms from the text of those documents that they fit. It cannot authorize contact with anyone, a merge, a push, data acquisition beyond section 7, or a submission.

## 3. Compute on HiPerGator

The values below come from the repository's records, named in the third column. They were not checked against the live cluster when this file was prepared, so each is marked UNVERIFIED. Before the first job the launch session checks every UNVERIFIED value with read-only commands through my existing authenticated session, for example `sacctmgr show assoc user=$USER format=account,qos,partition`, `sacctmgr show qos cai6734,cai6734-b format=name,grptres,maxwall`, and `sinfo -p hpg-rtx6000`. A live value that differs from this table is a BLOCKED item; this file is not edited.

| Item | Value | Evidence in the repository | Status |
|---|---|---|---|
| Slurm account | `cai6734` | `cdm-extension/daniel-escalante/DECISIONS.md`: from 2026-11-11 every job runs on `cai6734`; the approvals under `cdm-extension/daniel-escalante/approvals/` | UNVERIFIED live |
| QOS for GPU jobs | `cai6734`. The class QOS: 8 GPUs, 32 CPUs, and 250 GB shared by the whole class | Same file; `replication-of-original/daniel-escalante/README.md` | UNVERIFIED live |
| QOS for CPU-only jobs | `cai6734-b`, the class burst QOS: CPUs and memory only, no GPUs. No GPU burst QOS is assumed to exist | Same files; `.env.example` (`SLURM_CPU_QOS`) | UNVERIFIED live |
| GPU partition | `hpg-rtx6000`, requested as `--gres=gpu:rtx_pro_6000:1` (RTX PRO 6000, 96 GB), one GPU per job. Every job of one comparison uses this GPU type, because greedy decoding is not bit-stable across GPU architectures | `DECISIONS.md`; `prereg/protocol.json`; `scripts/submit_slurm.py` | UNVERIFIED live |
| Other GPU partitions on record | `hpg-turin` (L4, 24 GB) and `hpg-b200` (B200). Not authorized for this study | `reports/week11_handoff_for_Imagination_and_Claude_Duet.md` | Recorded only |
| Other account on record | `cai5724`, used for some GPU jobs before 2026-11-11. Not authorized for this study | The approvals' queue notes | Recorded only |
| Most GPUs in use at once | 8, the class QOS limit. It is shared with the rest of the class, so fewer may be free. At the existing 4 CPUs and 16 GB per one-GPU job, 8 jobs use the whole 32-CPU limit | `DECISIONS.md`; the approvals' queue notes | UNVERIFIED live |
| Private data location | `/blue/cai6734/daniel.escalante/private/mimic-iv-ext-cdm/1.1/`. The agent never lists, prints, or copies it | `reports/week11_handoff_for_Imagination_and_Claude_Duet.md` | UNVERIFIED live |
| Storage root for this study | `/blue/cai6734/daniel.escalante/private/tii-cee-1`, a new directory beside the existing private work directories, created by the launch session with owner-only permissions | Derived from the line above and from `prereg/protocol.json` | UNVERIFIED live |
| Model cache | The `HF_HOME` already set in the project's private environment file on the cluster. Its recorded pattern is `/blue/<group>/<user>/cache/hf` | `cdm-extension/daniel-escalante/.env.example` | UNVERIFIED live |
| Longest single job | 24 hours; longer work is split into resumable parts. Multi-GPU jobs and jobs over 24 hours need course staff approval, which this file does not give | `replication-of-original/daniel-escalante/README.md` | Confirmed |
| Compute window | November 13 to about December 13, 2026. Work after that date is allowed under this file at whatever the account then permits | My statement of November 13, 2026 | Confirmed |

## 4. Experiments and GPU allowances

An allowance is a ceiling against runaway spending. It is about twice the estimate and does not shape any experiment. If the next submission would exceed one, the agent submits nothing for that experiment and raises a BLOCKED request with the numbers.

Measured bases: 0.458 GPU-hours per 240-admission rollout arm; 25 h 5 min to train the primary policy on one GPU. Other estimates rest on the assumptions in specification section 14.3 and are replaced by a bench before each main run.

| Experiment | Authorized | Estimate (GPU-hours) | Allowance (GPU-hours) |
|---|---|---|---|
| E0, E1 parts A to D, E3, reader-opportunity audit | Yes, CPU | 0 | 0 |
| E2 reader panel on twin text | Yes | 90 | 200 |
| E4 clinical evidence replay, cohorts P, S, X, B | Yes, subject to section 2 | 180 | 400 |
| E5 closed-loop arms on validation and cohort X | Yes, when gate G3 is met | 400 | 800 |
| E6 frozen one-shot test | Yes, under section 8 | 145 | 300 |
| E7 public datasets | Yes, for the datasets in section 7 | 150 | 300 |
| E8 four seeds of the primary recipe | Yes | 105 | 210 |
| E9 three seeds on the second backbone | Yes | 120 | 300 |
| E10 five cross-fitted fold policies and cohort X rollouts | Yes | 133 | 270 |
| Benches and probe conformance | Yes | 10 | 30 |
| E11 independent credentialed cohort | No | 0 | 0 |
| Total | | 1,333 | 2,810 |

Failed jobs and retries count against the allowance.

## 5. Permitted models

All run on HiPerGator. No model that reads patient text runs anywhere else, and no patient text is sent to any API or online service.

No model whose base originates in a country of concern is permitted on HiPerGator. `REPRODUCTION.md`, D-5, records the rule and the banned families: Qwen, DeepSeek, Yi, InternLM, and their derivatives.

Every model below has a US-origin base and was reported as not gated by the Hugging Face API on November 13, 2026, at the revision shown. The agent may download these from Hugging Face at exactly these revisions, into the model cache of section 3. If a model has become gated, or its license page no longer permits research use, it is a BLOCKED item and the rest of the panel proceeds.

| # | Model | Parameters | Family | License on record | Revision |
|---|---|---|---|---|---|
| 1 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7 B | SmolLM | Apache-2.0 | `31b70e2e869a7173562077fd711b654946d38674` |
| 2 | `ibm-granite/granite-3.3-2b-instruct` | about 2.5 B | Granite | Apache-2.0 | `707f574c62054322f6b5b04b6d075f0a8f05e0f0` |
| 3 | `microsoft/Phi-4-mini-instruct` | 3.8 B | Phi | MIT | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| 4 | `allenai/OLMo-2-1124-7B-Instruct` | 7 B | OLMo | Apache-2.0 | `470b1fba1ae01581f270116362ee4aa1b97f4c84` |
| 5 | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`, the policy backbone, without fine-tuning | 8 B | Llama, NVIDIA post-training | Listed as "other" on Hugging Face; the terms on its model card, as used for the replication | `54641c1611fcff44fa4865626462445e0a153fc7`, already in the model cache |
| 6 | `ibm-granite/granite-3.3-8b-instruct` | 8 B | Granite | Apache-2.0 | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |
| 7 | `allenai/OLMo-2-1124-13B-Instruct` | 13 B | OLMo | Apache-2.0 | `3a5c85baefbb1896a54d56fe2e76c0395627ddf4` |
| 8 | `microsoft/phi-4` | 14 B | Phi | MIT | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` |
| 9 | `allenai/OLMo-2-0325-32B-Instruct` | 32 B | OLMo | Apache-2.0 | `b96024342a77a69aa0dda815c3454a671f477463` |

Nine models, five families, sizes from 1.7 B to 32 B.

| Use | Models |
|---|---|
| Primary policy backbone, E4 to E6, E8, E10 | Model 5, at the revision the replication pinned |
| Second backbone, E9 | Model 6 |
| Reader panel, E2 | Models 1 to 9 |
| Readers for E7 | Models 1, 2, 4, 6, 8, 9 |
| Generative sampler W4 | Any of models 1 to 9; on patient text, model 5 unless the experiment card names another |

Gated models are not in the panel. The agent may run a login-status command that prints no token to see whether a Hugging Face login already exists on the cluster, and may read the README files of the shared model directories under `/data/ai/models`. It never reads, prints, copies, or creates a token, and it accepts no license on my behalf. A gated model becomes usable only if I add it in a new authorization.

## 6. Credentialed data and stage-by-split permissions

- Dataset: MIMIC-IV-Ext CDM v1.1, in approved private storage, under my PhysioNet credentialed access only.
- Existing stage-by-split permissions stay as recorded in the repository.
- New permissions for this study, each conditional on DATA_USE_CHECK.md passing:
  - F1: a validation-stage or test-stage job may read training-split result text at inference, for the retrieval world model.
  - F2: a job may score a trained policy on a real history with an imagined result substituted, for the reader probe.
- Training on the 2,160 training-plus-validation admissions is permitted for the E10 fold policies only. No policy trains on a test admission.
- Test rows are never used for fitting, retrieval, calibration, budgets, or thresholds.

## 7. Other data

Public datasets for E7. The agent may download each from the official source named. It records each license from the official page before use and stops on any that does not permit research use.

| Dataset | Official source |
|---|---|
| Chronic Kidney Disease | UCI Machine Learning Repository |
| Hepatitis C (HCV) | UCI Machine Learning Repository |
| Pima diabetes | The source ACTMED cites |
| DDXPlus | The authors' official release |

Other credentialed cohorts for E11: none. The agent never requests, acquires, or copies credentialed data.

## 8. The one-shot frozen test

Experiment E6 is the one additional use of the 240 test admissions, and it is disclosed as a use of a previously examined benchmark. There is no second run and no further use.

I authorize E6 to run once, automatically, when all of these hold:
1. gate G4 is met and its record is committed;
2. the test protocol of specification section 11.3 is frozen and committed, with the primary contrast A1 against V1 averaged over the five policies of the primary recipe;
3. every fitted artifact it uses is bound in the manifest and was fit without test rows;
4. the disclosure above is written into the protocol.

Inside E6, each policy's test rollout and its replay happen once. An interruption resumes under the same manifest and is the same run. A manifest mismatch is a refusal. There is no re-tuning and no new test key.

Outside E6, and after E6 has run, no job reads, scores, rolls out, or otherwise touches a test admission for any purpose under this authorization. That includes robustness checks, reviews, and the manuscript. If E6 does not run, the test admissions are not used at all.

## 9. Git

- Local commits on the task branch: yes.
- Push: **no**. The repository is public. The agent pushes nothing, to any branch, until I decide otherwise in a new authorization.
- Never: merge or close a pull request; force-push; edit or delete another person's branch; rewrite history; create a release or tag on the remote.
- No agent attribution in commits: no co-author trailers, no generated-by lines.

## 10. Watcher, local loop, notifications, and the connection keeper

- A scrontab entry on HiPerGator for the watcher, every 15 minutes, as designed in `reports/research-advancement/WATCHER_DESIGN.md`: yes. Fallback to the daemon server if scrontab jobs cannot call the scheduler: yes.
- A launchd job on my Mac for the local loop, using my existing authenticated SSH connection only: yes.
- macOS notifications: yes.
- The connection keeper: the launch session installs it once, exactly as `reports/research-advancement/keeper/KEEPER_INSTALL.md` describes, with my GatorLink username `daniel.escalante`, merging with any existing HiPerGator entry in `~/.ssh/config` instead of duplicating it, and with `TII_REMOTE_EVENTS` set to `/blue/cai6734/daniel.escalante/private/tii-cee-1/state/events.tsv`. After that, no agent changes or removes it.
- No agent ever starts a master connection, or stores, reads, types, or replays my password or a Duo code, or automates the Duo prompt. If no master connection is running, the agent tells me the one command to run and waits for the connection, while everything that needs no connection continues.

## 11. Never authorized

Contact with Professor Liu, teammates, course staff, UF Research Computing, PhysioNet, or anyone else. Drafting messages for me to send to teammates. Any change to this file. Any push. Any use of test rows outside section 8. Export of protected data. Acquisition of credentialed data. Registration, upload, or submission of coursework or a paper. Any clinical deployment or safety claim.

## 12. Binding and signature

This file binds these two documents by their SHA-256:

- `reports/research-advancement/SPEC_RESEARCH_ADVANCEMENT.md`, version 3.1: `8f48329862060ff9b63325b5f45dc15bf610912e0955ff87614d7b48262f9ca6`
- `reports/research-advancement/KICKOFF_RESEARCH_ADVANCEMENT.txt`, version 3.1: `e8c4b25657cf45925cba06809c5b4e97f706eb1b393a30f6fa02b717f2d1199a`

I sign this file by a statement, in the launch prompt I paste, that quotes this file's path, the commit that contains it, and its SHA-256. The launch session verifies the three hashes, records my statement verbatim in `reports/research-advancement/SIGNATURE.md`, commits it, and binds this file's SHA-256 into every manifest. Without a SIGNATURE.md that quotes this file's SHA-256, this file authorizes nothing. If any of the three files changes, the hashes no longer match and the authorization is void until I sign again.

## 13. Open item: the author list

The author list is undecided until I have spoken with Professor Liu. It is due before abstract registration on January 14, 2027. It does not block any experiment or the manuscript draft, which is written with an anonymous author block. Edwin Salcedo's authorship of the synthetic study is credited in the text regardless of the author list.
