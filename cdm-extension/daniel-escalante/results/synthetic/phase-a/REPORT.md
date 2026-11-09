> **SYNTHETIC** — invented records from the CPU fixture at fixture scale. These numbers exercise the pipeline and say nothing about the method.


# Phase A: commit guard on the LA-CDM baseline

## Framing

Phase A is termination recovery and action validation under the frozen metric. Its fallback (G3) reuses the hypothesis agent's answer, so it is not a new diagnostic predictor, and it supports no clinical-safety claim. The comparator is the replication's trained Nemotron Nano 8B policy (run 7fcd9e5, seed 269, greedy); these results are not a reproduction of the paper.

Test-set status: a prospectively frozen follow-up analysis on a previously examined benchmark. These published test aggregates informed the design: per-arm confusion matrices with an undiagnosed column (greedy and temperature 1.0), undiagnosed / valid / repeated request counts, the cost decomposition (267 unavailable requests), ignore/* metrics (84.62), test hypothesis accuracy (81.67% over 622 triplets), and the zero-shot hypothesis distribution (904 of 1,095 cholecystitis) with its undiagnosed-by-request-count breakdown. No patient rows or trajectories were viewed. The hypothesis fallback (G3) has not been computed on test rows by anyone. Related test results already seen: the undiagnosed counts per arm and the C1 hypothesis accuracy.

## Baseline recomputation

T5 recomputed from the official rows: 8.3333 (reference —). V0-rerun reproduced 24/24 official predictions.

## Ladder

| Arm | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 ($) | Nonanswers |
|---|---|---|---|---|---|---|---|---|---|
| Official primary rows (comparator) | 16.67 | 16.67 | 0.00 | 0.00 | 8.33 | 14.29 | 14.29 | 714.79 | 20 |
| V0-rerun (guard off) | 16.67 | 16.67 | 0.00 | 0.00 | 8.33 | 14.29 | 14.29 | 714.79 | 20 |
| V1 = official + G3 | 83.33 | 83.33 | 83.33 | 100.00 | 87.50 | 89.36 | 89.36 | 714.79 | 1 |
| V2 raw (G2) | 83.33 | 66.67 | 33.33 | 50.00 | 58.33 | 70.00 | 68.84 | 714.79 | 8 |
| V2 = G2 + G3 | 83.33 | 83.33 | 83.33 | 100.00 | 87.50 | 89.36 | 89.36 | 714.79 | 1 |
| V3 raw (G1 + G2) | 100.00 | 66.67 | 66.67 | 83.33 | 79.17 | 84.44 | 84.02 | 714.79 | 3 |
| V3 = G1 + G2 + G3 (method) | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 91.67 | 91.64 | 714.79 | 0 |

Analytic references on the official nonanswers (expected T5 gains, not results): uniform random fallback +20.83, training-majority fallback +20.83, perfect recovery +83.33.

## Recovery and regressions

| Arm | Recovery yield | Newly answered | Accuracy among newly answered | Regressions |
|---|---|---|---|---|
| V1 = official + G3 | 0.950 (19/20) | 19 | 1.000 | 0 |
| V2 raw (G2) | 0.600 (12/20) | 12 | 1.000 | 0 |
| V2 = G2 + G3 | 0.950 (19/20) | 19 | 1.000 | 0 |
| V3 raw (G1 + G2) | 0.850 (17/20) | 17 | 1.000 | 0 |
| V3 = G1 + G2 + G3 (method) | 1.000 (20/20) | 20 | 1.000 | 0 |

## Termination, requests and cost

| Arm | Diagnosis | Step limit | Malformed | Completion overflow | Observation overflow | Requested | Unavailable | Repeated | Guard replacements | Scorer errors |
|---|---|---|---|---|---|---|---|---|---|---|
| V0-rerun (guard off) | 4 | 10 | 5 | 3 | 2 | 66 | 11 | 30 | 0 | 0 |
| V2 = G2 + G3 | 16 | 0 | 5 | 1 | 2 | 62 | 11 | 26 | 12 | 0 |
| V3 = G1 + G2 + G3 (method) | 21 | 0 | 0 | 1 | 2 | 36 | 11 | 0 | 17 | 0 |

## Primary result

ΔT5 (V3 − official) = +83.33 [66.67, 95.83] percentage points over 24 test admissions; two-sided paired sign-flip randomization p < 0.0001 (exact, 20 discordant). The difference is **statistically significant**. The test is exact for the exchangeability null, not for the weaker null ΔT5 = 0 when diseases move in opposite directions.

## Family A (Holm, α = 0.05)

| Comparison | ΔT5 [95% CI] | p | Holm p | Statement |
|---|---|---|---|---|
| V1 − official | +79.17 [62.50, 95.83] | < 0.0001 | < 0.0001 | statistically significant |
| V3 − V1 | +4.17 [0.00, 16.67] | 1.0000 | 1.0000 | not statistically significant |

## Descriptive comparisons (intervals only)

| Comparison | ΔT5 [95% CI] |
|---|---|
| V2 − V1 | +0.00 [0.00, 0.00] |
| V2 raw − official | +50.00 [29.17, 70.83] |
| V3 − V2 | +4.17 [0.00, 16.67] |
| V3 raw − official | +70.83 [54.06, 87.50] |

## Per-class recall (%)

| Arm | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| Official primary rows (comparator) | 16.67 | 16.67 | 0.00 | 0.00 |
| V1 = official + G3 | 83.33 | 83.33 | 83.33 | 100.00 |
| V3 = G1 + G2 + G3 (method) | 100.00 | 83.33 | 83.33 | 100.00 |

## Sensitivity (validation-based simulation at the test class sizes)

- V1 − V0: validation effect +79.17 points, power at that effect 1.000, minimum detectable effect at 80% power 49.25 points.
- V3 − V0: validation effect +83.33 points, power at that effect 1.000, minimum detectable effect at 80% power 52.24 points.

A null result is read against these figures: an effect smaller than the minimum detectable effect would usually go undetected at this sample size.

## Limitations

- Previously examined benchmark (see the disclosure above).
- One policy seed: between-seed variation is unassessable.
- The scorer is the policy's own likelihood, not a calibrated posterior; verbalizer and length effects remain.
- No clinical-safety inference.
- Person-level overlap is unassessable: the release ships no subject_id.
- Baseline deviations D-1, D-3, D-5 and D-6 (REPRODUCTION.md) and the Blackwell runtime carry over.
