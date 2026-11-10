
# Phase A: commit guard on the LA-CDM baseline

## Framing

Phase A is termination recovery and action validation under the frozen metric. Its fallback (G3) reuses the hypothesis agent's answer, so it is not a new diagnostic predictor, and it supports no clinical-safety claim. The comparator is the replication's trained Nemotron Nano 8B policy (run 7fcd9e5, seed 269, greedy); these results are not a reproduction of the paper.

Test-set status: a prospectively frozen follow-up analysis on a previously examined benchmark. These published test aggregates informed the design: per-arm confusion matrices with an undiagnosed column (greedy and temperature 1.0), undiagnosed / valid / repeated request counts, the cost decomposition (267 unavailable requests), ignore/* metrics (84.62), test hypothesis accuracy (81.67% over 622 triplets), and the zero-shot hypothesis distribution (904 of 1,095 cholecystitis) with its undiagnosed-by-request-count breakdown. No patient rows or trajectories were viewed. The hypothesis fallback (G3) has not been computed on test rows by anyone. Related test results already seen: the undiagnosed counts per arm and the C1 hypothesis accuracy.

## Baseline recomputation

T5 recomputed from the official rows: 73.0850 (reference 73.0850). V0-rerun reproduced 240/240 official predictions.

## Ladder

| Arm | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 ($) | Nonanswers |
|---|---|---|---|---|---|---|---|---|---|
| Official primary rows (comparator) | 90.62 | 72.31 | 72.00 | 57.41 | 73.09 | 81.15 | 78.20 | 1439.58 | 29 |
| V0-rerun (guard off) | 90.62 | 72.31 | 72.00 | 57.41 | 73.09 | 81.15 | 78.20 | 1439.58 | 29 |
| V1 = official + G3 | 94.79 | 81.54 | 80.00 | 81.48 | 84.45 | 86.67 | 84.56 | 1439.58 | 0 |
| V2 raw (G2) | 92.71 | 76.92 | 76.00 | 62.96 | 77.15 | 82.58 | 79.84 | 1439.58 | 15 |
| V2 = G2 + G3 | 94.79 | 80.00 | 80.00 | 79.63 | 83.61 | 85.83 | 83.72 | 1439.58 | 0 |
| V3 raw (G1 + G2) | 94.79 | 78.46 | 80.00 | 70.37 | 80.91 | 83.86 | 81.10 | 1514.87 | 3 |
| V3 = G1 + G2 + G3 (method) | 94.79 | 78.46 | 80.00 | 74.07 | 81.83 | 84.17 | 81.53 | 1514.87 | 0 |

Analytic references on the official nonanswers (expected T5 gains, not results): uniform random fallback +3.52, training-majority fallback +1.30, perfect recovery +14.09.

## Recovery and regressions

| Arm | Recovery yield | Newly answered | Accuracy among newly answered | Regressions |
|---|---|---|---|---|
| V1 = official + G3 | 0.862 (25/29) | 29 | 0.862 | 0 |
| V2 raw (G2) | 0.310 (9/29) | 14 | 0.643 | 0 |
| V2 = G2 + G3 | 0.793 (23/29) | 29 | 0.793 | 0 |
| V3 raw (G1 + G2) | 0.655 (19/29) | 26 | 0.731 | 2 |
| V3 = G1 + G2 + G3 (method) | 0.724 (21/29) | 29 | 0.724 | 2 |

## Termination, requests and cost

| Arm | Diagnosis | Step limit | Malformed | Completion overflow | Observation overflow | Requested | Unavailable | Repeated | Guard replacements | Scorer errors |
|---|---|---|---|---|---|---|---|---|---|---|
| V0-rerun (guard off) | 211 | 6 | 2 | 9 | 12 | 661 | 267 | 121 | 0 | 0 |
| V2 = G2 + G3 | 225 | 0 | 2 | 1 | 12 | 657 | 263 | 117 | 14 | 0 |
| V3 = G1 + G2 + G3 (method) | 237 | 0 | 0 | 0 | 3 | 598 | 208 | 0 | 97 | 0 |

## Primary result

ΔT5 (V3 − official) = +8.75 [4.67, 13.35] percentage points over 240 test admissions; two-sided paired sign-flip randomization p < 0.0001 (draws, 23 discordant). The difference is **statistically significant**. The test is exact for the exchangeability null, not for the weaker null ΔT5 = 0 when diseases move in opposite directions.

## Family A (Holm, α = 0.05)

| Comparison | ΔT5 [95% CI] | p | Holm p | Statement |
|---|---|---|---|---|
| V1 − official | +11.37 [7.28, 15.96] | < 0.0001 | < 0.0001 | statistically significant |
| V3 − V1 | -2.62 [-6.16, 0.76] | 0.1875 | 0.1875 | not statistically significant |

## Descriptive comparisons (intervals only)

| Comparison | ΔT5 [95% CI] |
|---|---|
| V2 − V1 | -0.85 [-2.16, 0.00] |
| V2 raw − official | +4.06 [1.57, 7.25] |
| V3 − V2 | -1.77 [-5.24, 1.61] |
| V3 raw − official | +7.82 [3.89, 12.34] |

## Per-class recall (%)

| Arm | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| Official primary rows (comparator) | 90.62 | 72.31 | 72.00 | 57.41 |
| V1 = official + G3 | 94.79 | 81.54 | 80.00 | 81.48 |
| V3 = G1 + G2 + G3 (method) | 94.79 | 78.46 | 80.00 | 74.07 |

## Sensitivity (validation-based simulation at the test class sizes)

- V1 − V0: validation effect +5.51 points, power at that effect 0.996, minimum detectable effect at 80% power 4.15 points.
- V3 − V0: validation effect +7.82 points, power at that effect 0.999, minimum detectable effect at 80% power 5.54 points.

A null result is read against these figures: an effect smaller than the minimum detectable effect would usually go undetected at this sample size.

## Interpretation (SPEC §11)

V1 − official is statistically significant but V3 − V1 is not: the hypothesis agent already held the missing answers, and the online guards added nothing measurable.

## Protocol notes

- `prereg/amendments/A1-code-fixes.md`: code fixes made before test access after an independent review, including the disclosure that the first smoke job's loader parsed the train and test files.
- `prereg/amendments/A2-reference-scoring.md`: decisions use the reference scoring path after bench-a found the fast path outside its tolerance.

## Limitations

- Previously examined benchmark (see the disclosure above).
- One policy seed: between-seed variation is unassessable.
- The scorer is the policy's own likelihood, not a calibrated posterior; verbalizer and length effects remain.
- No clinical-safety inference.
- Person-level overlap is unassessable: the release ships no subject_id.
- Baseline deviations D-1, D-3, D-5 and D-6 (REPRODUCTION.md) and the Blackwell runtime carry over.
