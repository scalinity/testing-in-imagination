# Protocol: parity; training seed: 269

# Replication report

**MIMIC source — credentialed-run results require manifests and completed CUDA checks.**

Every observed result below is a **re-implementation on a different backbone** (mimic). Paper references come only from `REPRODUCTION.md`; no matching tolerance or reproduction verdict is inferred. Reported values alone do not verify data access, model revision, training completion or CUDA acceptance.

## Setup and deviations

Source: **mimic**. Configured CUDA policy and HPI summarizer: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. Revision: `54641c1611fcff44fa4865626462445e0a153fc7`. Code commit: `7fcd9e512fb80a7576404b7c7daa1de1f9cad72d`. Upstream: `3f435a10f569c3255172c1ef8fe7880c6edb7fee`.

D-5 is a backbone deviation and D-3 replaces the paper's summarizer with that same pinned backbone. D-1 keeps incorrect reward 0 in the primary run and −1 only in its named ablation. D-2 includes the $300 physical exam in the twelve-test inventory. Configured common budgets are 14 actions, 256 tokens per completion and 4,096 trajectory tokens; training uses eight sampled rollouts and batch-two forwards, while validation and test generation are greedy. All local compatibility repairs are enumerated under Honest gaps. CUDA training/evaluation were requested; inspect the recorded hardware and successful stage receipts before accepting these as completed runs.

## T1–T8: held-out metrics

| Claim | Metric | Paper | Observed | 95% interval | Observed − paper | 95% delta interval | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | Appendicitis recall | 93.1 % | 90.62 | [84.38, 95.83] | -2.47 | [-8.72, 2.73] | observed; mimic; re-implementation on a different backbone |
| T2 | Cholecystitis recall | 83.6 % | 72.31 | [61.54, 83.08] | -11.29 | [-22.06, -0.52] | observed; mimic; re-implementation on a different backbone |
| T3 | Diverticulitis recall | 75.0 % | 72.00 | [56.00, 88.00] | -3.00 | [-19.00, 13.00] | observed; mimic; re-implementation on a different backbone |
| T4 | Pancreatitis recall | 73.5 % | 57.41 | [44.44, 70.37] | -16.09 | [-29.06, -3.13] | observed; mimic; re-implementation on a different backbone |
| T5 | Mean of four class recalls | 81.3 % | 73.09 | [66.65, 79.19] | -8.21 | [-14.65, -2.11] | observed; mimic; re-implementation on a different backbone |
| T6 | Micro F1 | 84.1 % | 81.15 | [76.47, 85.53] | -2.95 | [-7.63, 1.43] | observed; mimic; re-implementation on a different backbone |
| T7 | Macro F1 | 81.3 % | 78.20 | [72.32, 83.44] | -3.10 | [-8.98, 2.14] | observed; mimic; re-implementation on a different backbone |
| T8 | Average provided-test cost | 1295.61 USD | 1439.58 | [1341.50, 1537.65] | 143.97 | [45.89, 242.04] | observed; mimic; re-implementation on a different backbone |

T1–T4 are class recalls (%); T5 is their unweighted mean (%), rather than pooled accuracy. T6/T7 are upstream all/micro_f1 and all/macro_f1 (%); None diagnoses count as wrong. T8 is mean USD for tests that actually returned data.

Admissions: 240. Pooled accuracy: 76.25% (mimic; re-implementation on a different backbone). Missing classes and missing hypothesis observations are unassessable. Raw upstream empty-triplet zeros do not establish zero ECE.

## B1–B2: baselines

| Claim | Method | Paper mean recall (%) | Observed mean recall (%) | 95% interval | Observed − paper | Status |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | Zero-shot | 64.5 | 12.71 | [8.04, 17.86] | -51.79 | mimic; re-implementation on a different backbone |
| B2 | ReAct | 74.9 | unassessable | unassessable | unassessable | gap: ReAct implementation absent at pinned upstream |

## C1–C4: training dynamics and mechanism comparisons

| Claim | Paper | Observed comparison | Status |
| --- | --- | --- | --- |
| C1 | 75.7 → 81.9% | zero-shot → primary hypothesis accuracy: 29.59 → 81.67% | mimic; re-implementation on a different backbone |
| C2 | 0.069 → 0.037 | zero-shot → primary ECE: 0.3022 → 0.1281 | mimic; re-implementation on a different backbone |
| C3 | $1427.85 → $1295.61; recall 82.3 → 81.3% | no_cost → primary: $1321.33 → $1439.58; recall 70.81 → 73.09% | mimic; re-implementation on a different backbone |
| C4 | Table 3: exact values absent from ledger | decision_only → primary mean recall: 57.49 → 73.09% | mimic; re-implementation on a different backbone |

Hypothesis statistics use actual trajectory triplets, weighted by observation count. Unknown hypotheses contribute to hypothesis accuracy but are excluded from ECE by upstream. These held-out endpoints are not a measured training curve. C3 needs paired accuracy and cost evidence; a similar point accuracy does not establish equivalence.

Primary hypothesis observations: 622 actual triplets; 622 eligible for ECE; 0 finite out-of-range confidences retained for hypothesis accuracy and excluded from ECE by upstream (mimic; re-implementation on a different backbone).

Validation trajectory (never used test data):

| Validation observation | Hypothesis accuracy (%) | ECE | Optimizer step |
| --- | --- | --- | --- |
| 0 | 31.21 | 0.2712 | endpoint |
| 1 | unassessable | unassessable | 1920 |
| 2 | 77.58 | 0.0588 | 1920 |
| 3 | unassessable | unassessable | 3840 |
| 4 | 75.61 | 0.1947 | 3840 |
| 5 | 75.15 | 0.1936 | 3840 |
| 6 | 75.15 | 0.1936 | endpoint |

## Ablations

| Arm | Definition | Mean recall (%) | Provided cost (USD) | Status |
| --- | --- | --- | --- | --- |
| primary | incorrect reward 0 (shipped primary) | 73.09 | 1439.58 | mimic; re-implementation on a different backbone |
| decision_only | hypothesis/confidence agents disabled | 57.49 | 1593.22 | mimic; re-implementation on a different backbone |
| no_cost | cost reward disabled; USD costs still measured | 70.81 | 1321.33 | mimic; re-implementation on a different backbone |
| incorrect_minus1 | incorrect reward −1 (ablation) | 59.38 | 1619.27 | mimic; re-implementation on a different backbone |

Before test cost, primary correct/wrong well-formed/undiagnosed-invalid totals are 2/1/0; the −1 ablation gives 2/0/0. Do not silently replace primary reward 0 with −1. No inter-arm performance ordering or oracle baseline is assumed.

## Paired uncertainty and significance

Differences are primary minus the comparison arm. T1–T7 and pooled-accuracy differences are percentage points; T8 differences are USD. The paired bootstrap reuses identical patient draws in both arms. Intervals concern held-out sampling conditional on fitted models, not training-seed uncertainty.

| Comparison | Metric | Difference | 95% paired interval | Status |
| --- | --- | --- | --- | --- |
| primary_vs_zero_shot | T5 | 60.37 | [53.25, 67.20] | mimic; re-implementation on a different backbone |
| primary_vs_zero_shot | T6 | 62.27 | [55.25, 69.18] | mimic; re-implementation on a different backbone |
| primary_vs_zero_shot | T7 | 58.52 | [51.12, 66.20] | mimic; re-implementation on a different backbone |
| primary_vs_zero_shot | T8 | -300.99 | [-434.37, -164.64] | mimic; re-implementation on a different backbone |
| primary_vs_zero_shot | pooled_accuracy | 65.00 | [58.75, 70.83] | mimic; re-implementation on a different backbone |
| primary_vs_decision_only | T5 | 15.59 | [6.34, 24.60] | mimic; re-implementation on a different backbone |
| primary_vs_decision_only | T6 | 16.94 | [10.15, 23.59] | mimic; re-implementation on a different backbone |
| primary_vs_decision_only | T7 | 20.98 | [12.51, 29.34] | mimic; re-implementation on a different backbone |
| primary_vs_decision_only | T8 | -153.64 | [-254.43, -50.81] | mimic; re-implementation on a different backbone |
| primary_vs_decision_only | pooled_accuracy | 14.58 | [7.50, 21.67] | mimic; re-implementation on a different backbone |
| primary_vs_no_cost | T5 | 2.28 | [-4.64, 9.13] | mimic; re-implementation on a different backbone |
| primary_vs_no_cost | T6 | 1.87 | [-2.70, 6.48] | mimic; re-implementation on a different backbone |
| primary_vs_no_cost | T7 | 1.98 | [-3.61, 7.68] | mimic; re-implementation on a different backbone |
| primary_vs_no_cost | T8 | 118.25 | [-0.25, 239.86] | mimic; re-implementation on a different backbone |
| primary_vs_no_cost | pooled_accuracy | 2.08 | [-3.33, 7.50] | mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T5 | 13.71 | [5.32, 21.93] | mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T6 | 5.49 | [0.31, 10.71] | mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T7 | 8.33 | [1.10, 16.02] | mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T8 | -179.69 | [-281.00, -79.81] | mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | pooled_accuracy | 10.83 | [4.58, 17.08] | mimic; re-implementation on a different backbone |

| Planned pooled-accuracy test | Exact McNemar p | Holm p (family = 4) | Adjusted rejection |
| --- | --- | --- | --- |
| primary_vs_zero_shot | 0.000000 | 0.000000 | True |
| primary_vs_decision_only | 0.000187 | 0.000560 | True |
| primary_vs_no_cost | 0.551484 | 0.551484 | False |
| primary_vs_incorrect_minus1 | 0.001563 | 0.003126 | True |

McNemar tests paired correctness discordances only, not T5, F1, ECE or cost. No p-values are inferred from per-arm standard deviations or bootstrap interval exclusion. Other metrics have intervals only; adding metric tests requires declaring and correcting the entire family. Subject clustering is assessable only when subject IDs are supplied. Repeated subjects make the admission-level McNemar test unassessable because independent pairs are required. Class-stratified intervals condition on class composition; subject-cluster intervals condition on class-presence strata, which can be degenerate when strata contain few subjects.

At least one planned pooled-accuracy difference is significant after Holm adjustment.

primary_vs_zero_shot (mimic; re-implementation on a different backbone): class-stratified patient percentile bootstrap; resamples 10000; seed 971965967.

primary_vs_decision_only (mimic; re-implementation on a different backbone): class-stratified patient percentile bootstrap; resamples 10000; seed 971965967.

primary_vs_no_cost (mimic; re-implementation on a different backbone): class-stratified patient percentile bootstrap; resamples 10000; seed 971965967.

primary_vs_incorrect_minus1 (mimic; re-implementation on a different backbone): class-stratified patient percentile bootstrap; resamples 10000; seed 971965967.

Between-training-seed uncertainty: | Comparison | Completed/declared seeds | Mean paired difference | Student-t 95% interval | Status |
| --- | --- | --- | --- | --- |
| primary_vs_zero_shot/T5 | 1/1 | 60.37 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_zero_shot/T6 | 1/1 | 62.27 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_zero_shot/T7 | 1/1 | 58.52 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_zero_shot/T8 | 1/1 | -300.99 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_zero_shot/pooled_accuracy | 1/1 | 65.00 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_decision_only/T5 | 1/1 | 15.59 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_decision_only/T6 | 1/1 | 16.94 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_decision_only/T7 | 1/1 | 20.98 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_decision_only/T8 | 1/1 | -153.64 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_decision_only/pooled_accuracy | 1/1 | 14.58 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_no_cost/T5 | 1/1 | 2.28 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_no_cost/T6 | 1/1 | 1.87 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_no_cost/T7 | 1/1 | 1.98 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_no_cost/T8 | 1/1 | 118.25 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_no_cost/pooled_accuracy | 1/1 | 2.08 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T5 | 1/1 | 13.71 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T6 | 1/1 | 5.49 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T7 | 1/1 | 8.33 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T8 | 1/1 | -179.69 | unassessable | unassessable; mimic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/pooled_accuracy | 1/1 | 10.83 | unassessable | unassessable; mimic; re-implementation on a different backbone |

## Leakage and split sensitivity

| Audit | Observed |
| --- | --- |
| Admission-ID overlap across train/val/test | 0 |
| Subject overlap across train/val/test | unassessable |
| Subject-grouped sensitivity | unassessable: subject IDs absent |

Absent subject IDs mean unassessable subject leakage, never zero. The executable pinned splitter hardcodes seed 269. D-4 says no canonical seed: this ledger/code discrepancy is retained; use 269 for parity and declare additional split seeds before comparing them. Checkpoint selection must use validation only; final test evaluation follows model locking. Leakage audits here are supplied aggregate checks, not inferred from metric values.

## Provided versus requested cost

| Arm | Provided USD/admission | Requested USD/admission | Unavailable-request USD/admission | Repeated available request USD/admission | Missing requests | Source |
| --- | --- | --- | --- | --- | --- | --- |
| primary | 1439.58 | 3499.42 | 1851.61 | 208.22 | 267 | mimic; re-implementation on a different backbone |
| decision_only | 1593.22 | 4217.15 | 2437.65 | 186.27 | 297 | mimic; re-implementation on a different backbone |
| no_cost | 1321.33 | 3428.52 | 1793.50 | 313.69 | 259 | mimic; re-implementation on a different backbone |
| incorrect_minus1 | 1619.27 | 4976.47 | 2924.45 | 432.74 | 406 | mimic; re-implementation on a different backbone |
| zero_shot | 1740.58 | 8002.34 | 5109.94 | 1151.83 | 631 | mimic; re-implementation on a different backbone |

Provided cost is the T8 definition. Requested cost prices every recognized request at the same configured USD schedule; the difference includes unavailable and repeated available requests, separated above, and is not an observed charge. Repeated requests remain counted. Expected provided cost for one request is availability × cost; an unavailable request is free under the shipped provided-tests reward.

## F-1–F-4: environment development notes

- F-1: unavailable tests share the trajectory error flag with malformed actions and diagnoses.
- F-2: only provided tests incur reward cost; unavailable requests are free.
- F-3: unavailable requests skip hypothesis updates and consume an action; max_steps 13 allows 14 actions.
- F-4: stale confidence already divided by 10 is reused in a prompt otherwise showing a raw 0–10 value.

These pinned behaviors remain intact. Unit-level confirmation is recorded in the ledger; this report does not claim a new CUDA smoke episode.

## Configuration and implementation checks

| Claim | Check | Observed status |
| --- | --- | --- |
| I1 | Approved different backbone + LoRA; immutable revision | LoRA training completed on the blackwell runtime |
| I2 | Cyclic action GRPO / hypothesis SFT / confidence GRPO, 100/100/100 | 100/100/100 optimizer-update phases |
| I3 | Adam lr 1e-5, batch 2, seed 269 | Adam, lr 1e-5, batch 2, declared seed |
| I4 | Configured USD costs including physical examination | configured upstream USD schedule; provenance taken from paper ledger |
| I5 | Normalized costs sum to 1 including physical examination | twelve normalized costs sum to one (tested) |

Run seed: 269; declared seeds: [269]. Upstream SHA: 3f435a10f569c3255172c1ef8fe7880c6edb7fee; immutable backbone revision: 54641c1611fcff44fa4865626462445e0a153fc7. Training completion: True; verify against the run ledger.

## Honest gaps

- B2 ReAct is absent at the upstream pin; no substitute is labeled ReAct.
- Exact C4/Table 3 values are absent from REPRODUCTION.md and remain null in paper_values.json.
- Different backbone and summarizer prevent a bit-for-bit paper comparison; no numerical acceptance is implied.
- Local compatibility changes: Singleton validation/evaluation prevents upstream inputs[0] label reuse across patients.
- Local compatibility changes: HA SFT examples are batched independently, without dropping deduplicated examples; token-weighted group loss retains upstream sum-of-minibatch gradient scaling.
- Local compatibility changes: Loss phase is fixed for each optimizer update: exactly 100/100/100 updates; upstream switches on the final microbatch of the boundary update. Phase derives from saved global_step on resume.
- Local compatibility changes: Policy-to-inference weight transfer errors are propagated instead of suppressed upstream.
- Local compatibility changes: Inactive loss objectives skip unused forwards; dropout RNG consumption differs from the broken upstream minibatch path.
- Local compatibility changes: Checkpoint sidecars preserve the environment's independent confidence-pair RNG; standard framework RNG files do not contain it.
- Local compatibility changes: Training generation uses distinct deterministic request seeds by optimizer step, call and rollout; a single shared request seed would collapse identical GRPO prompts to identical draws.
- Local compatibility changes: Validation runs at each epoch end instead of every 100 steps (one singleton pass over 240 validation patients takes about 80 minutes); the final fixed-budget checkpoint is selected either way, so validation history has pre-training, per-epoch and final points.
- Local compatibility changes: The first-hypothesis placeholder is filled as a literal substitution: upstream applies str.format to the whole episode text, so any other brace field in a record or sampled completion raised KeyError during training.
- Local compatibility changes: Best-metric tracking is off (metric_for_best_model unset): singleton validation reports no eval_loss, so upstream's 'loss' setting raises KeyError at the first scheduled validation, and the final fixed-budget checkpoint is selected regardless.
- Packaging: no upstream lockfile; the CUDA 12.8 Blackwell runtime changes 28 upstream pins (torch 2.7.0+cu128, vLLM 0.9.2), listed in the S0 environment record.
- A paper delta treats the published value as fixed; its interval does not incorporate unknown paper-run uncertainty.
- Physical-examination CSV missing-value literals become the existing unavailable sentinel, avoiding non-string NaN observations.
- Primary parity admissions may share subjects with training; the grouped protocol tests sensitivity to that leakage.

<!-- supplementary aggregates -->

## Supplementary aggregates (added after S6)

Protocol: parity; training seed: 269.

Counts and metrics computed from the saved test episodes after S6; no patient row leaves the work directory. Rows are true diagnoses; columns are the episode's final diagnosis, or *undiagnosed* when the episode ended without a valid diagnosis (malformed action, step budget or trajectory length).

### Confusion: zero_shot (greedy decoding, pre-specified)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 11 | 10 | 1 | 0 | 74 |
| cholecystitis | 0 | 10 | 0 | 0 | 55 |
| diverticulitis | 0 | 0 | 6 | 0 | 19 |
| pancreatitis | 1 | 6 | 1 | 0 | 46 |

### Confusion: primary (greedy decoding, pre-specified)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 87 | 0 | 2 | 2 | 5 |
| cholecystitis | 2 | 47 | 0 | 10 | 6 |
| diverticulitis | 0 | 0 | 18 | 3 | 4 |
| pancreatitis | 1 | 7 | 1 | 31 | 14 |

### Confusion: decision_only (greedy decoding, pre-specified)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 81 | 8 | 4 | 0 | 3 |
| cholecystitis | 13 | 41 | 3 | 3 | 5 |
| diverticulitis | 3 | 4 | 16 | 0 | 2 |
| pancreatitis | 12 | 17 | 6 | 10 | 9 |

### Confusion: no_cost (greedy decoding, pre-specified)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 83 | 2 | 1 | 2 | 8 |
| cholecystitis | 4 | 50 | 0 | 6 | 5 |
| diverticulitis | 1 | 0 | 17 | 3 | 4 |
| pancreatitis | 1 | 9 | 2 | 28 | 14 |

### Confusion: incorrect_minus1 (greedy decoding, pre-specified)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 79 | 0 | 1 | 1 | 15 |
| cholecystitis | 0 | 41 | 1 | 5 | 18 |
| diverticulitis | 1 | 0 | 11 | 2 | 11 |
| pancreatitis | 1 | 6 | 0 | 26 | 21 |

### Confusion: zero_shot (temperature 1.0, upstream decoding)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 9 | 4 | 0 | 0 | 83 |
| cholecystitis | 2 | 9 | 0 | 0 | 54 |
| diverticulitis | 2 | 1 | 2 | 0 | 20 |
| pancreatitis | 1 | 5 | 0 | 3 | 45 |

### Confusion: primary (temperature 1.0, upstream decoding)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 63 | 2 | 5 | 4 | 22 |
| cholecystitis | 4 | 43 | 0 | 4 | 14 |
| diverticulitis | 2 | 0 | 12 | 3 | 8 |
| pancreatitis | 3 | 7 | 2 | 31 | 11 |

### Confusion: decision_only (temperature 1.0, upstream decoding)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 55 | 13 | 9 | 3 | 16 |
| cholecystitis | 9 | 30 | 4 | 6 | 16 |
| diverticulitis | 4 | 5 | 12 | 0 | 4 |
| pancreatitis | 9 | 19 | 6 | 6 | 14 |

### Confusion: no_cost (temperature 1.0, upstream decoding)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 74 | 2 | 3 | 2 | 15 |
| cholecystitis | 4 | 42 | 0 | 8 | 11 |
| diverticulitis | 5 | 0 | 14 | 1 | 5 |
| pancreatitis | 1 | 6 | 2 | 26 | 19 |

### Confusion: incorrect_minus1 (temperature 1.0, upstream decoding)

| True \ final | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
| --- | --- | --- | --- | --- | --- |
| appendicitis | 63 | 2 | 4 | 0 | 27 |
| cholecystitis | 4 | 32 | 1 | 5 | 23 |
| diverticulitis | 2 | 0 | 15 | 1 | 7 |
| pancreatitis | 0 | 5 | 2 | 26 | 21 |

### Episodes without a valid diagnosis

| Arm | Decoding | Undiagnosed | Valid test requests | Repeated requests |
| --- | --- | --- | --- | --- |
| zero_shot | greedy | 194 | 1521 | 613 |
| primary | greedy | 29 | 661 | 121 |
| decision_only | greedy | 19 | 747 | 108 |
| no_cost | greedy | 31 | 618 | 154 |
| incorrect_minus1 | greedy | 65 | 937 | 299 |
| zero_shot | temperature 1.0 | 202 | 406 | 44 |
| primary | temperature 1.0 | 55 | 400 | 26 |
| decision_only | temperature 1.0 | 50 | 479 | 26 |
| no_cost | temperature 1.0 | 50 | 372 | 25 |
| incorrect_minus1 | temperature 1.0 | 78 | 380 | 29 |

### Decoding check: upstream evaluation at temperature 1.0

Upstream evaluates with its training temperature (1.0); this replication pre-specified greedy decoding. Each locked model ran once more on the same 240 test patients, sampling with independent request seeds. Greedy remains the primary result. Intervals: class-stratified patient bootstrap; the difference is paired.

| Arm | Metric | Greedy | Temperature 1.0 | 95% interval | Difference | 95% paired interval |
| --- | --- | --- | --- | --- | --- | --- |
| zero_shot | T1 | 11.46 | 9.38 | [4.17, 15.62] | -2.08 | [-10.42, 6.25] |
| zero_shot | T2 | 15.38 | 13.85 | [6.15, 23.08] | -1.54 | [-12.31, 9.23] |
| zero_shot | T3 | 24.00 | 8.00 | [0.00, 20.00] | -16.00 | [-36.00, 0.00] |
| zero_shot | T4 | 0.00 | 5.56 | [0.00, 12.96] | 5.56 | [0.00, 12.96] |
| zero_shot | T5 | 12.71 | 9.19 | [5.55, 13.44] | -3.52 | [-9.38, 2.37] |
| zero_shot | T6 | 18.88 | 16.55 | [10.79, 22.46] | -2.33 | [-10.04, 5.32] |
| zero_shot | T7 | 19.68 | 15.78 | [9.53, 22.25] | -3.89 | [-11.92, 4.72] |
| zero_shot | T8 | 1,740.58 | 812.70 | [695.72, 930.98] | -927.87 | [-1,085.72, -764.27] |
| zero_shot | pooled_accuracy | 11.25 | 9.58 | [6.25, 13.33] | -1.67 | [-6.67, 3.33] |
| primary | T1 | 90.62 | 65.62 | [56.25, 75.00] | -25.00 | [-35.42, -14.58] |
| primary | T2 | 72.31 | 66.15 | [53.85, 76.92] | -6.15 | [-20.00, 7.69] |
| primary | T3 | 72.00 | 48.00 | [28.00, 68.00] | -24.00 | [-44.00, -4.00] |
| primary | T4 | 57.41 | 57.41 | [44.44, 70.37] | 0.00 | [-16.67, 16.67] |
| primary | T5 | 73.09 | 59.30 | [52.23, 66.19] | -13.79 | [-21.70, -6.24] |
| primary | T6 | 81.15 | 70.12 | [64.27, 75.63] | -11.04 | [-17.22, -5.21] |
| primary | T7 | 78.20 | 66.91 | [59.94, 73.16] | -11.29 | [-18.43, -4.52] |
| primary | T8 | 1,439.58 | 961.71 | [875.56, 1,046.85] | -477.87 | [-601.46, -355.99] |
| primary | pooled_accuracy | 76.25 | 62.08 | [55.83, 68.33] | -14.17 | [-21.25, -7.50] |
| decision_only | T1 | 84.38 | 57.29 | [47.92, 67.71] | -27.08 | [-37.50, -16.67] |
| decision_only | T2 | 63.08 | 46.15 | [33.85, 58.46] | -16.92 | [-32.31, -1.54] |
| decision_only | T3 | 64.00 | 48.00 | [28.00, 68.00] | -16.00 | [-40.00, 8.00] |
| decision_only | T4 | 18.52 | 11.11 | [3.70, 20.37] | -7.41 | [-18.52, 3.70] |
| decision_only | T5 | 57.49 | 40.64 | [33.86, 47.27] | -16.85 | [-25.05, -8.68] |
| decision_only | T6 | 64.21 | 47.91 | [41.78, 53.83] | -16.30 | [-23.45, -9.34] |
| decision_only | T7 | 57.22 | 42.32 | [35.79, 48.62] | -14.90 | [-22.88, -6.98] |
| decision_only | T8 | 1,593.22 | 1,248.03 | [1,152.74, 1,342.22] | -345.20 | [-460.37, -230.87] |
| decision_only | pooled_accuracy | 61.67 | 42.92 | [37.08, 48.75] | -18.75 | [-25.83, -11.67] |
| no_cost | T1 | 86.46 | 77.08 | [68.75, 85.42] | -9.38 | [-18.75, 0.00] |
| no_cost | T2 | 76.92 | 64.62 | [52.31, 75.38] | -12.31 | [-24.65, 1.54] |
| no_cost | T3 | 68.00 | 56.00 | [36.00, 76.00] | -12.00 | [-36.00, 12.00] |
| no_cost | T4 | 51.85 | 48.15 | [35.19, 61.11] | -3.70 | [-20.37, 12.96] |
| no_cost | T5 | 70.81 | 61.46 | [54.52, 68.28] | -9.35 | [-17.60, -1.02] |
| no_cost | T6 | 79.29 | 72.56 | [67.13, 77.78] | -6.73 | [-12.80, -0.72] |
| no_cost | T7 | 76.22 | 69.01 | [62.22, 75.07] | -7.21 | [-14.80, 0.37] |
| no_cost | T8 | 1,321.33 | 878.77 | [787.95, 969.84] | -442.56 | [-557.78, -328.08] |
| no_cost | pooled_accuracy | 74.17 | 65.00 | [59.17, 70.83] | -9.17 | [-15.83, -2.08] |
| incorrect_minus1 | T1 | 82.29 | 65.62 | [56.25, 75.00] | -16.67 | [-27.08, -6.25] |
| incorrect_minus1 | T2 | 63.08 | 49.23 | [36.92, 61.54] | -13.85 | [-27.69, 0.00] |
| incorrect_minus1 | T3 | 44.00 | 60.00 | [40.00, 80.00] | 16.00 | [-12.00, 44.00] |
| incorrect_minus1 | T4 | 48.15 | 48.15 | [35.19, 61.11] | 0.00 | [-14.81, 14.81] |
| incorrect_minus1 | T5 | 59.38 | 55.75 | [48.63, 62.62] | -3.63 | [-12.94, 5.66] |
| incorrect_minus1 | T6 | 75.66 | 67.66 | [61.81, 73.13] | -8.00 | [-14.44, -1.75] |
| incorrect_minus1 | T7 | 69.87 | 65.55 | [58.61, 71.62] | -4.32 | [-12.77, 4.48] |
| incorrect_minus1 | T8 | 1,619.27 | 926.87 | [826.09, 1,030.63] | -692.40 | [-814.75, -570.30] |
| incorrect_minus1 | pooled_accuracy | 65.42 | 56.67 | [50.42, 62.92] | -8.75 | [-15.84, -1.25] |

| Comparison at temperature 1.0 | T5 difference | 95% paired interval | Exact McNemar p | Holm p (family = 4) |
| --- | --- | --- | --- | --- |
| primary_vs_decision_only | 18.66 | [9.94, 27.21] | 0.000016 | 0.000049 |
| primary_vs_no_cost | -2.17 | [-10.74, 5.98] | 0.538394 | 0.538394 |
| primary_vs_incorrect_minus1 | 3.55 | [-5.68, 12.76] | 0.236902 | 0.473804 |
| primary_vs_zero_shot | 50.10 | [41.96, 57.98] | 0.000000 | 0.000000 |

