# Protocol: parity; training seed: 269

# Replication report

**SYNTHETIC — INVENTED DATA; MOCK UPDATES ARE NOT LEARNING. NO MIMIC RESULT OR PAPER REPRODUCTION IS ESTABLISHED.**

Every observed result below is a **re-implementation on a different backbone** (synthetic). Paper references come only from `REPRODUCTION.md`; no matching tolerance or reproduction verdict is inferred. Reported values alone do not verify data access, model revision, training completion or CUDA acceptance.

## Setup and deviations

Source: **synthetic**. Configured CUDA policy and HPI summarizer: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. Revision: `54641c1611fcff44fa4865626462445e0a153fc7`. Code commit: `e56e0de83444b0c5925dc3f1ed436323d2540024`. Upstream: `3f435a10f569c3255172c1ef8fe7880c6edb7fee`.

D-5 is a backbone deviation and D-3 replaces the paper's summarizer with that same pinned backbone. D-1 keeps incorrect reward 0 in the primary run and −1 only in its named ablation. D-2 includes the $300 physical exam in the twelve-test inventory. Configured common budgets are 14 actions, 256 tokens per completion and 4,096 trajectory tokens; training uses eight sampled rollouts and batch-two forwards, while validation and test generation are greedy. All local compatibility repairs are enumerated under Honest gaps. This run used an observation-only mock policy, fake HPI summaries and no weight updates.

## T1–T8: held-out metrics

| Claim | Metric | Paper | Observed | 95% interval | Observed − paper | 95% delta interval | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | Appendicitis recall | 93.1 % | 100.00 | [100.00, 100.00] | 6.90 | [6.90, 6.90] | observed; synthetic; re-implementation on a different backbone |
| T2 | Cholecystitis recall | 83.6 % | 0.00 | [0.00, 0.00] | -83.60 | [-83.60, -83.60] | observed; synthetic; re-implementation on a different backbone |
| T3 | Diverticulitis recall | 75.0 % | 0.00 | [0.00, 0.00] | -75.00 | [-75.00, -75.00] | observed; synthetic; re-implementation on a different backbone |
| T4 | Pancreatitis recall | 73.5 % | 0.00 | [0.00, 0.00] | -73.50 | [-73.50, -73.50] | observed; synthetic; re-implementation on a different backbone |
| T5 | Mean of four class recalls | 81.3 % | 25.00 | [25.00, 25.00] | -56.30 | [-56.30, -56.30] | observed; synthetic; re-implementation on a different backbone |
| T6 | Micro F1 | 84.1 % | 40.00 | [39.51, 40.34] | -44.10 | [-44.59, -43.76] | observed; synthetic; re-implementation on a different backbone |
| T7 | Macro F1 | 81.3 % | 14.29 | [14.16, 14.37] | -67.01 | [-67.14, -66.93] | observed; synthetic; re-implementation on a different backbone |
| T8 | Average provided-test cost | 1295.61 USD | 1037.87 | [954.02, 1117.48] | -257.74 | [-341.59, -178.13] | observed; synthetic; re-implementation on a different backbone |

T1–T4 are class recalls (%); T5 is their unweighted mean (%), rather than pooled accuracy. T6/T7 are upstream all/micro_f1 and all/macro_f1 (%); None diagnoses count as wrong. T8 is mean USD for tests that actually returned data.

Admissions: 240. Pooled accuracy: 40.00% (synthetic; re-implementation on a different backbone). Missing classes and missing hypothesis observations are unassessable. Raw upstream empty-triplet zeros do not establish zero ECE.

## B1–B2: baselines

| Claim | Method | Paper mean recall (%) | Observed mean recall (%) | 95% interval | Observed − paper | Status |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | Zero-shot | 64.5 | 25.00 | [25.00, 25.00] | -39.50 | synthetic; re-implementation on a different backbone |
| B2 | ReAct | 74.9 | unassessable | unassessable | unassessable | gap: ReAct implementation absent at pinned upstream |

## C1–C4: training dynamics and mechanism comparisons

| Claim | Paper | Observed comparison | Status |
| --- | --- | --- | --- |
| C1 | 75.7 → 81.9% | zero-shot → primary hypothesis accuracy: 40.59 → 40.59% | synthetic; re-implementation on a different backbone |
| C2 | 0.069 → 0.037 | zero-shot → primary ECE: 0.2941 → 0.2941 | synthetic; re-implementation on a different backbone |
| C3 | $1427.85 → $1295.61; recall 82.3 → 81.3% | no_cost → primary: $1037.87 → $1037.87; recall 25.00 → 25.00% | synthetic; re-implementation on a different backbone |
| C4 | Table 3: exact values absent from ledger | decision_only → primary mean recall: 25.00 → 25.00% | synthetic; re-implementation on a different backbone |

Hypothesis statistics use actual trajectory triplets, weighted by observation count. Unknown hypotheses contribute to hypothesis accuracy but are excluded from ECE by upstream. These held-out endpoints are not a measured training curve. C3 needs paired accuracy and cost evidence; a similar point accuracy does not establish equivalence.

Primary hypothesis observations: 574 actual triplets; 574 eligible for ECE; 0 finite out-of-range confidences retained for hypothesis accuracy and excluded from ECE by upstream (synthetic; re-implementation on a different backbone). Synthetic comparisons demonstrate wiring only; they do not establish training improvement or a scientific mechanism.

C1/C2 validation training curves are unassessable in this run.

## Ablations

| Arm | Definition | Mean recall (%) | Provided cost (USD) | Status |
| --- | --- | --- | --- | --- |
| primary | incorrect reward 0 (shipped primary) | 25.00 | 1037.87 | synthetic; re-implementation on a different backbone |
| decision_only | hypothesis/confidence agents disabled | 25.00 | 1037.87 | synthetic; re-implementation on a different backbone |
| no_cost | cost reward disabled; USD costs still measured | 25.00 | 1037.87 | synthetic; re-implementation on a different backbone |
| incorrect_minus1 | incorrect reward −1 (ablation) | 25.00 | 1037.87 | synthetic; re-implementation on a different backbone |

Before test cost, primary correct/wrong well-formed/undiagnosed-invalid totals are 2/1/0; the −1 ablation gives 2/0/0. Do not silently replace primary reward 0 with −1. No inter-arm performance ordering or oracle baseline is assumed.

## Paired uncertainty and significance

Differences are primary minus the comparison arm. T1–T7 and pooled-accuracy differences are percentage points; T8 differences are USD. The paired bootstrap reuses identical patient draws in both arms. Intervals concern held-out sampling conditional on fitted models, not training-seed uncertainty.

| Comparison | Metric | Difference | 95% paired interval | Status |
| --- | --- | --- | --- | --- |
| primary_vs_zero_shot | T5 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot | T6 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot | T7 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot | T8 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot | pooled_accuracy | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | T5 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | T6 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | T7 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | T8 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | pooled_accuracy | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | T5 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | T6 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | T7 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | T8 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | pooled_accuracy | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T5 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T6 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T7 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T8 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | pooled_accuracy | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |

| Planned pooled-accuracy test | Exact McNemar p | Holm p (family = 4) | Adjusted rejection |
| --- | --- | --- | --- |
| primary_vs_zero_shot | unassessable | unassessable | unassessable |
| primary_vs_decision_only | unassessable | unassessable | unassessable |
| primary_vs_no_cost | unassessable | unassessable | unassessable |
| primary_vs_incorrect_minus1 | unassessable | unassessable | unassessable |

McNemar tests paired correctness discordances only, not T5, F1, ECE or cost. No p-values are inferred from per-arm standard deviations or bootstrap interval exclusion. Other metrics have intervals only; adding metric tests requires declaring and correcting the entire family. Subject clustering is assessable only when subject IDs are supplied. Repeated subjects make the admission-level McNemar test unassessable because independent pairs are required. Class-stratified intervals condition on class composition; subject-cluster intervals condition on class-presence strata, which can be degenerate when strata contain few subjects.

No planned pooled-accuracy difference is established as significant after Holm adjustment; unassessable tests cannot support a significance claim.

primary_vs_zero_shot (synthetic; re-implementation on a different backbone): subject-cluster percentile bootstrap stratified by class-presence profile; resamples 10000; seed 971965967.

primary_vs_decision_only (synthetic; re-implementation on a different backbone): subject-cluster percentile bootstrap stratified by class-presence profile; resamples 10000; seed 971965967.

primary_vs_no_cost (synthetic; re-implementation on a different backbone): subject-cluster percentile bootstrap stratified by class-presence profile; resamples 10000; seed 971965967.

primary_vs_incorrect_minus1 (synthetic; re-implementation on a different backbone): subject-cluster percentile bootstrap stratified by class-presence profile; resamples 10000; seed 971965967.

Between-training-seed uncertainty: | Comparison | Completed/declared seeds | Mean paired difference | Student-t 95% interval | Status |
| --- | --- | --- | --- | --- |
| primary_vs_zero_shot/T5 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot/T6 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot/T7 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot/T8 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot/pooled_accuracy | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/T5 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/T6 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/T7 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/T8 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/pooled_accuracy | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/T5 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/T6 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/T7 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/T8 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/pooled_accuracy | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T5 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T6 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T7 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T8 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/pooled_accuracy | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |

## Leakage and split sensitivity

| Audit | Observed |
| --- | --- |
| Admission-ID overlap across train/val/test | 0 |
| Subject overlap across train/val/test | 175 |
| Subject-grouped sensitivity | full arm matrix reported separately below |

Absent subject IDs mean unassessable subject leakage, never zero. The executable pinned splitter hardcodes seed 269. D-4 says no canonical seed: this ledger/code discrepancy is retained; use 269 for parity and declare additional split seeds before comparing them. Checkpoint selection must use validation only; final test evaluation follows model locking. Leakage audits here are supplied aggregate checks, not inferred from metric values.

## Provided versus requested cost

| Arm | Provided USD/admission | Requested USD/admission | Unavailable-request USD/admission | Repeated available request USD/admission | Missing requests | Source |
| --- | --- | --- | --- | --- | --- | --- |
| primary | 1037.87 | 1606.00 | 568.13 | 0.00 | 146 | synthetic; re-implementation on a different backbone |
| decision_only | 1037.87 | 1606.00 | 568.13 | 0.00 | 146 | synthetic; re-implementation on a different backbone |
| no_cost | 1037.87 | 1606.00 | 568.13 | 0.00 | 146 | synthetic; re-implementation on a different backbone |
| incorrect_minus1 | 1037.87 | 1606.00 | 568.13 | 0.00 | 146 | synthetic; re-implementation on a different backbone |
| zero_shot | 1037.87 | 1606.00 | 568.13 | 0.00 | 146 | synthetic; re-implementation on a different backbone |

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
| I1 | Approved different backbone + LoRA; immutable revision | mock; no weights trained |
| I2 | Cyclic action GRPO / hypothesis SFT / confidence GRPO, 100/100/100 | configured; learning NOT_RUN |
| I3 | Adam lr 1e-5, batch 2, seed 269 | configured; Adam NOT_RUN |
| I4 | Configured USD costs including physical examination | configured upstream USD schedule; provenance taken from paper ledger |
| I5 | Normalized costs sum to 1 including physical examination | twelve normalized costs sum to one (tested) |

Run seed: 269; declared seeds: [269]. Upstream SHA: 3f435a10f569c3255172c1ef8fe7880c6edb7fee; immutable backbone revision: 54641c1611fcff44fa4865626462445e0a153fc7. Training completion: not established by synthetic/mock execution.

## Honest gaps

- B2 ReAct is absent at the upstream pin; no substitute is labeled ReAct.
- Exact C4/Table 3 values are absent from REPRODUCTION.md and remain null in paper_values.json.
- Different backbone and summarizer prevent a bit-for-bit paper comparison; no numerical acceptance is implied.
- MIMIC results, real learning, CUDA execution and paper reproduction remain unverified.
- Local compatibility changes: Singleton validation/evaluation prevents upstream inputs[0] label reuse across patients.
- Local compatibility changes: HA SFT examples are batched independently, without dropping deduplicated examples; token-weighted group loss retains upstream sum-of-minibatch gradient scaling.
- Local compatibility changes: Loss phase is fixed for each optimizer update: exactly 100/100/100 updates; upstream switches on the final microbatch of the boundary update. Phase derives from saved global_step on resume.
- Local compatibility changes: Policy-to-inference weight transfer errors are propagated instead of suppressed upstream.
- Local compatibility changes: Inactive loss objectives skip unused forwards; dropout RNG consumption differs from the broken upstream minibatch path.
- Local compatibility changes: Checkpoint sidecars preserve the environment's independent confidence-pair RNG; standard framework RNG files do not contain it.
- Local compatibility changes: Training generation uses distinct deterministic request seeds by optimizer step, call and rollout; a single shared request seed would collapse identical GRPO prompts to identical draws.
- Packaging: no upstream lockfile; same-version xgrammar source build is pinned in the local lock.
- A paper delta treats the published value as fixed; its interval does not incorporate unknown paper-run uncertainty.
- Physical-examination CSV missing-value literals become the existing unavailable sentinel, avoiding non-string NaN observations.
- Primary parity admissions may share subjects with training; the grouped protocol tests sensitivity to that leakage.


---

# Protocol: grouped; training seed: 269

# Replication report

**SYNTHETIC — INVENTED DATA; MOCK UPDATES ARE NOT LEARNING. NO MIMIC RESULT OR PAPER REPRODUCTION IS ESTABLISHED.**

Every observed result below is a **re-implementation on a different backbone** (synthetic). Paper references come only from `REPRODUCTION.md`; no matching tolerance or reproduction verdict is inferred. Reported values alone do not verify data access, model revision, training completion or CUDA acceptance.

## Setup and deviations

Source: **synthetic**. Configured CUDA policy and HPI summarizer: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. Revision: `54641c1611fcff44fa4865626462445e0a153fc7`. Code commit: `e56e0de83444b0c5925dc3f1ed436323d2540024`. Upstream: `3f435a10f569c3255172c1ef8fe7880c6edb7fee`.

D-5 is a backbone deviation and D-3 replaces the paper's summarizer with that same pinned backbone. D-1 keeps incorrect reward 0 in the primary run and −1 only in its named ablation. D-2 includes the $300 physical exam in the twelve-test inventory. Configured common budgets are 14 actions, 256 tokens per completion and 4,096 trajectory tokens; training uses eight sampled rollouts and batch-two forwards, while validation and test generation are greedy. All local compatibility repairs are enumerated under Honest gaps. This run used an observation-only mock policy, fake HPI summaries and no weight updates.

## T1–T8: held-out metrics

| Claim | Metric | Paper | Observed | 95% interval | Observed − paper | 95% delta interval | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | Appendicitis recall | 93.1 % | 100.00 | [100.00, 100.00] | 6.90 | [6.90, 6.90] | observed; synthetic; re-implementation on a different backbone |
| T2 | Cholecystitis recall | 83.6 % | 0.00 | [0.00, 0.00] | -83.60 | [-83.60, -83.60] | observed; synthetic; re-implementation on a different backbone |
| T3 | Diverticulitis recall | 75.0 % | 0.00 | [0.00, 0.00] | -75.00 | [-75.00, -75.00] | observed; synthetic; re-implementation on a different backbone |
| T4 | Pancreatitis recall | 73.5 % | 0.00 | [0.00, 0.00] | -73.50 | [-73.50, -73.50] | observed; synthetic; re-implementation on a different backbone |
| T5 | Mean of four class recalls | 81.3 % | 25.00 | [25.00, 25.00] | -56.30 | [-56.30, -56.30] | observed; synthetic; re-implementation on a different backbone |
| T6 | Micro F1 | 84.1 % | 40.51 | [38.14, 42.92] | -43.59 | [-45.96, -41.18] | observed; synthetic; re-implementation on a different backbone |
| T7 | Macro F1 | 81.3 % | 14.41 | [13.80, 15.02] | -66.89 | [-67.50, -66.28] | observed; synthetic; re-implementation on a different backbone |
| T8 | Average provided-test cost | 1295.61 USD | 979.37 | [894.62, 1064.11] | -316.24 | [-400.99, -231.50] | observed; synthetic; re-implementation on a different backbone |

T1–T4 are class recalls (%); T5 is their unweighted mean (%), rather than pooled accuracy. T6/T7 are upstream all/micro_f1 and all/macro_f1 (%); None diagnoses count as wrong. T8 is mean USD for tests that actually returned data.

Admissions: 237. Pooled accuracy: 40.51% (synthetic; re-implementation on a different backbone). Missing classes and missing hypothesis observations are unassessable. Raw upstream empty-triplet zeros do not establish zero ECE.

## B1–B2: baselines

| Claim | Method | Paper mean recall (%) | Observed mean recall (%) | 95% interval | Observed − paper | Status |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | Zero-shot | 64.5 | 25.00 | [25.00, 25.00] | -39.50 | synthetic; re-implementation on a different backbone |
| B2 | ReAct | 74.9 | unassessable | unassessable | unassessable | gap: ReAct implementation absent at pinned upstream |

## C1–C4: training dynamics and mechanism comparisons

| Claim | Paper | Observed comparison | Status |
| --- | --- | --- | --- |
| C1 | 75.7 → 81.9% | zero-shot → primary hypothesis accuracy: 40.50 → 40.50% | synthetic; re-implementation on a different backbone |
| C2 | 0.069 → 0.037 | zero-shot → primary ECE: 0.2950 → 0.2950 | synthetic; re-implementation on a different backbone |
| C3 | $1427.85 → $1295.61; recall 82.3 → 81.3% | no_cost → primary: $979.37 → $979.37; recall 25.00 → 25.00% | synthetic; re-implementation on a different backbone |
| C4 | Table 3: exact values absent from ledger | decision_only → primary mean recall: 25.00 → 25.00% | synthetic; re-implementation on a different backbone |

Hypothesis statistics use actual trajectory triplets, weighted by observation count. Unknown hypotheses contribute to hypothesis accuracy but are excluded from ECE by upstream. These held-out endpoints are not a measured training curve. C3 needs paired accuracy and cost evidence; a similar point accuracy does not establish equivalence.

Primary hypothesis observations: 558 actual triplets; 558 eligible for ECE; 0 finite out-of-range confidences retained for hypothesis accuracy and excluded from ECE by upstream (synthetic; re-implementation on a different backbone). Synthetic comparisons demonstrate wiring only; they do not establish training improvement or a scientific mechanism.

C1/C2 validation training curves are unassessable in this run.

## Ablations

| Arm | Definition | Mean recall (%) | Provided cost (USD) | Status |
| --- | --- | --- | --- | --- |
| primary | incorrect reward 0 (shipped primary) | 25.00 | 979.37 | synthetic; re-implementation on a different backbone |
| decision_only | hypothesis/confidence agents disabled | 25.00 | 979.37 | synthetic; re-implementation on a different backbone |
| no_cost | cost reward disabled; USD costs still measured | 25.00 | 979.37 | synthetic; re-implementation on a different backbone |
| incorrect_minus1 | incorrect reward −1 (ablation) | 25.00 | 979.37 | synthetic; re-implementation on a different backbone |

Before test cost, primary correct/wrong well-formed/undiagnosed-invalid totals are 2/1/0; the −1 ablation gives 2/0/0. Do not silently replace primary reward 0 with −1. No inter-arm performance ordering or oracle baseline is assumed.

## Paired uncertainty and significance

Differences are primary minus the comparison arm. T1–T7 and pooled-accuracy differences are percentage points; T8 differences are USD. The paired bootstrap reuses identical patient draws in both arms. Intervals concern held-out sampling conditional on fitted models, not training-seed uncertainty.

| Comparison | Metric | Difference | 95% paired interval | Status |
| --- | --- | --- | --- | --- |
| primary_vs_zero_shot | T5 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot | T6 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot | T7 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot | T8 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot | pooled_accuracy | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | T5 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | T6 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | T7 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | T8 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_decision_only | pooled_accuracy | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | T5 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | T6 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | T7 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | T8 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_no_cost | pooled_accuracy | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T5 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T6 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T7 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | T8 | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1 | pooled_accuracy | 0.00 | [0.00, 0.00] | synthetic; re-implementation on a different backbone |

| Planned pooled-accuracy test | Exact McNemar p | Holm p (family = 4) | Adjusted rejection |
| --- | --- | --- | --- |
| primary_vs_zero_shot | unassessable | unassessable | unassessable |
| primary_vs_decision_only | unassessable | unassessable | unassessable |
| primary_vs_no_cost | unassessable | unassessable | unassessable |
| primary_vs_incorrect_minus1 | unassessable | unassessable | unassessable |

McNemar tests paired correctness discordances only, not T5, F1, ECE or cost. No p-values are inferred from per-arm standard deviations or bootstrap interval exclusion. Other metrics have intervals only; adding metric tests requires declaring and correcting the entire family. Subject clustering is assessable only when subject IDs are supplied. Repeated subjects make the admission-level McNemar test unassessable because independent pairs are required. Class-stratified intervals condition on class composition; subject-cluster intervals condition on class-presence strata, which can be degenerate when strata contain few subjects.

No planned pooled-accuracy difference is established as significant after Holm adjustment; unassessable tests cannot support a significance claim.

primary_vs_zero_shot (synthetic; re-implementation on a different backbone): subject-cluster percentile bootstrap stratified by class-presence profile; resamples 10000; seed 971965967.

primary_vs_decision_only (synthetic; re-implementation on a different backbone): subject-cluster percentile bootstrap stratified by class-presence profile; resamples 10000; seed 971965967.

primary_vs_no_cost (synthetic; re-implementation on a different backbone): subject-cluster percentile bootstrap stratified by class-presence profile; resamples 10000; seed 971965967.

primary_vs_incorrect_minus1 (synthetic; re-implementation on a different backbone): subject-cluster percentile bootstrap stratified by class-presence profile; resamples 10000; seed 971965967.

Between-training-seed uncertainty: | Comparison | Completed/declared seeds | Mean paired difference | Student-t 95% interval | Status |
| --- | --- | --- | --- | --- |
| primary_vs_zero_shot/T5 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot/T6 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot/T7 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot/T8 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_zero_shot/pooled_accuracy | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/T5 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/T6 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/T7 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/T8 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_decision_only/pooled_accuracy | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/T5 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/T6 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/T7 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/T8 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_no_cost/pooled_accuracy | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T5 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T6 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T7 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/T8 | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |
| primary_vs_incorrect_minus1/pooled_accuracy | 1/1 | 0.00 | unassessable | unassessable; synthetic; re-implementation on a different backbone |

## Leakage and split sensitivity

| Audit | Observed |
| --- | --- |
| Admission-ID overlap across train/val/test | 0 |
| Subject overlap across train/val/test | 0 |
| Subject-grouped sensitivity | full arm matrix reported separately below |

Absent subject IDs mean unassessable subject leakage, never zero. The executable pinned splitter hardcodes seed 269. D-4 says no canonical seed: this ledger/code discrepancy is retained; use 269 for parity and declare additional split seeds before comparing them. Checkpoint selection must use validation only; final test evaluation follows model locking. Leakage audits here are supplied aggregate checks, not inferred from metric values.

## Provided versus requested cost

| Arm | Provided USD/admission | Requested USD/admission | Unavailable-request USD/admission | Repeated available request USD/admission | Missing requests | Source |
| --- | --- | --- | --- | --- | --- | --- |
| primary | 979.37 | 1606.00 | 626.63 | 0.00 | 153 | synthetic; re-implementation on a different backbone |
| decision_only | 979.37 | 1606.00 | 626.63 | 0.00 | 153 | synthetic; re-implementation on a different backbone |
| no_cost | 979.37 | 1606.00 | 626.63 | 0.00 | 153 | synthetic; re-implementation on a different backbone |
| incorrect_minus1 | 979.37 | 1606.00 | 626.63 | 0.00 | 153 | synthetic; re-implementation on a different backbone |
| zero_shot | 979.37 | 1606.00 | 626.63 | 0.00 | 153 | synthetic; re-implementation on a different backbone |

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
| I1 | Approved different backbone + LoRA; immutable revision | mock; no weights trained |
| I2 | Cyclic action GRPO / hypothesis SFT / confidence GRPO, 100/100/100 | configured; learning NOT_RUN |
| I3 | Adam lr 1e-5, batch 2, seed 269 | configured; Adam NOT_RUN |
| I4 | Configured USD costs including physical examination | configured upstream USD schedule; provenance taken from paper ledger |
| I5 | Normalized costs sum to 1 including physical examination | twelve normalized costs sum to one (tested) |

Run seed: 269; declared seeds: [269]. Upstream SHA: 3f435a10f569c3255172c1ef8fe7880c6edb7fee; immutable backbone revision: 54641c1611fcff44fa4865626462445e0a153fc7. Training completion: not established by synthetic/mock execution.

## Honest gaps

- B2 ReAct is absent at the upstream pin; no substitute is labeled ReAct.
- Exact C4/Table 3 values are absent from REPRODUCTION.md and remain null in paper_values.json.
- Different backbone and summarizer prevent a bit-for-bit paper comparison; no numerical acceptance is implied.
- MIMIC results, real learning, CUDA execution and paper reproduction remain unverified.
- Local compatibility changes: Singleton validation/evaluation prevents upstream inputs[0] label reuse across patients.
- Local compatibility changes: HA SFT examples are batched independently, without dropping deduplicated examples; token-weighted group loss retains upstream sum-of-minibatch gradient scaling.
- Local compatibility changes: Loss phase is fixed for each optimizer update: exactly 100/100/100 updates; upstream switches on the final microbatch of the boundary update. Phase derives from saved global_step on resume.
- Local compatibility changes: Policy-to-inference weight transfer errors are propagated instead of suppressed upstream.
- Local compatibility changes: Inactive loss objectives skip unused forwards; dropout RNG consumption differs from the broken upstream minibatch path.
- Local compatibility changes: Checkpoint sidecars preserve the environment's independent confidence-pair RNG; standard framework RNG files do not contain it.
- Local compatibility changes: Training generation uses distinct deterministic request seeds by optimizer step, call and rollout; a single shared request seed would collapse identical GRPO prompts to identical draws.
- Packaging: no upstream lockfile; same-version xgrammar source build is pinned in the local lock.
- A paper delta treats the published value as fixed; its interval does not incorporate unknown paper-run uncertainty.
- Physical-examination CSV missing-value literals become the existing unavailable sentinel, avoiding non-string NaN observations.
- Primary parity admissions may share subjects with training; the grouped protocol tests sensitivity to that leakage.
