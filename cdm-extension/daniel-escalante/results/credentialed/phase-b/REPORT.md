
# Phase B: retrospective chart-availability reranking

## Framing

Phase B estimates p(test present in chart | HPI) and reweights the policy's test requests on top of V3 (P0). It does not predict result content, does not update from revealed findings and does not evaluate imagined outcomes: it is retrospective chart-availability reranking, not full world-model planning. The oracle is a privileged chart-availability oracle; chart presence encodes clinicians' completed workups. These results are not a reproduction of the paper.

Test-set status: a prospectively frozen follow-up analysis on a previously examined benchmark. These published test aggregates informed the design: per-arm confusion matrices with an undiagnosed column (greedy and temperature 1.0), undiagnosed / valid / repeated request counts, the cost decomposition (267 unavailable requests), ignore/* metrics (84.62), test hypothesis accuracy (81.67% over 622 triplets), and the zero-shot hypothesis distribution (904 of 1,095 cholecystitis) with its undiagnosed-by-request-count breakdown. No patient rows or trajectories were viewed. The hypothesis fallback (G3) has not been computed on test rows by anyone. Related test results already seen: the undiagnosed counts per arm and the C1 hypothesis accuracy.

## Arms

| Arm | T1 | T2 | T3 | T4 | T5 | T8 ($) | Nonanswers | Reranks | Stops |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 94.79 | 78.46 | 80.00 | 74.07 | 81.83 | 1514.87 | 0 | 0 | 0 |
| B-uniform | 94.79 | 81.54 | 84.00 | 75.93 | 84.06 | 1502.25 | 0 | 60 | 0 |
| B-marginal | 94.79 | 76.92 | 80.00 | 72.22 | 80.98 | 1512.20 | 0 | 143 | 0 |
| B-learned-s0 | 94.79 | 76.92 | 84.00 | 75.93 | 82.91 | 1516.30 | 0 | 177 | 0 |
| B-learned-s1 | 94.79 | 78.46 | 88.00 | 75.93 | 84.29 | 1499.84 | 0 | 174 | 0 |
| B-learned-s2 | 94.79 | 80.00 | 84.00 | 75.93 | 83.68 | 1495.75 | 0 | 165 | 0 |
| B-oracle | 94.79 | 81.54 | 88.00 | 77.78 | 85.53 | 1593.34 | 0 | 259 | 3 |
| B-a0.1-T1-n0 | 94.79 | 83.08 | 84.00 | 81.48 | 85.84 | 1534.97 | 0 | 226 | 1 |
| B-a0.1-T1-n1 | 94.79 | 83.08 | 88.00 | 79.63 | 86.37 | 1505.22 | 0 | 217 | 0 |
| B-a0.1-T0.25-n0 | 94.79 | 76.92 | 84.00 | 79.63 | 83.84 | 1564.34 | 0 | 285 | 3 |
| B-a0.1-T0.25-n1 | 93.75 | 78.46 | 84.00 | 79.63 | 83.96 | 1512.38 | 0 | 281 | 3 |
| B-a0.3-T1-n0 | 94.79 | 81.54 | 76.00 | 72.22 | 81.14 | 1510.39 | 0 | 157 | 0 |
| B-a0.3-T1-n1 | 94.79 | 76.92 | 80.00 | 75.93 | 81.91 | 1527.37 | 0 | 161 | 0 |
| B-a0.3-T0.25-n0 | 94.79 | 76.92 | 84.00 | 70.37 | 81.52 | 1531.87 | 0 | 219 | 1 |
| B-a0.3-T0.25-n1 | 94.79 | 80.00 | 84.00 | 75.93 | 83.68 | 1519.10 | 0 | 230 | 1 |

## Family B (Holm, α = 0.05)

| Contrast | ΔT5 [95% CI] | p | Holm p | Statement |
|---|---|---|---|---|
| B-learned-s0 − B-marginal (patient-specific beyond test prevalence) | +1.93 [-0.46, 4.85] | 0.3125 | 0.9375 | not statistically significant |
| B-learned-s0 − B-uniform (information beyond the reranking mechanism) | -1.15 [-3.70, 1.47] | 0.4229 | 0.9375 | not statistically significant |
| B-oracle − B-uniform (privileged availability, same mechanism) | +1.46 [-1.71, 4.98] | 0.4292 | 0.9375 | not statistically significant |

## Every arm against P0 (descriptive, intervals only)

| Arm − P0 | ΔT5 [95% CI] |
|---|---|
| B-uniform − P0 | +2.23 [-0.28, 5.27] |
| B-marginal − P0 | -0.85 [-3.39, 1.62] |
| B-learned-s0 − P0 | +1.08 [-1.69, 4.23] |
| B-learned-s1 − P0 | +2.46 [-0.61, 6.16] |
| B-learned-s2 − P0 | +1.85 [-0.69, 4.92] |
| B-oracle − P0 | +3.70 [0.15, 7.77] |
| B-a0.1-T1-n0 − P0 | +4.01 [1.23, 7.32] |
| B-a0.1-T1-n1 − P0 | +4.54 [1.31, 8.31] |
| B-a0.1-T0.25-n0 − P0 | +2.00 [-2.00, 6.24] |
| B-a0.1-T0.25-n1 − P0 | +2.13 [-2.38, 6.79] |
| B-a0.3-T1-n0 − P0 | -0.69 [-3.54, 1.54] |
| B-a0.3-T1-n1 − P0 | +0.08 [-3.46, 3.69] |
| B-a0.3-T0.25-n0 − P0 | -0.31 [-3.47, 3.15] |
| B-a0.3-T0.25-n1 − P0 | +1.85 [-2.31, 6.31] |

## World-model quality

| Split | Arm | α | ECE | AUROC | Brier | Brier skill vs marginal |
|---|---|---|---|---|---|---|
| test | B-a0.1-T0.25-n0 | 0.1 | 0.031 | 0.962 | 0.034 | 0.569 |
| test | B-a0.1-T0.25-n1 | 0.1 | 0.036 | 0.959 | 0.038 | 0.512 |
| test | B-a0.1-T1-n0 | 0.1 | 0.010 | 0.989 | 0.029 | 0.625 |
| test | B-a0.1-T1-n1 | 0.1 | 0.004 | 0.987 | 0.033 | 0.576 |
| test | B-a0.3-T0.25-n0 | 0.3 | 0.066 | 0.950 | 0.079 | -0.001 |
| test | B-a0.3-T0.25-n1 | 0.3 | 0.069 | 0.943 | 0.082 | -0.046 |
| test | B-a0.3-T1-n0 | 0.3 | 0.010 | 0.959 | 0.066 | 0.159 |
| test | B-a0.3-T1-n1 | 0.3 | 0.006 | 0.956 | 0.068 | 0.130 |
| test | B-learned-s0 | — | 0.024 | 0.959 | 0.068 | 0.133 |
| test | B-learned-s1 | — | 0.022 | 0.959 | 0.068 | 0.137 |
| test | B-learned-s2 | — | 0.022 | 0.958 | 0.069 | 0.121 |
| test | B-marginal | — | 0.009 | 0.941 | 0.079 | -0.000 |
| test | B-oracle | — | 0.001 | 1.000 | 0.000 | 1.000 |
| test | B-uniform | — | 0.279 | 0.500 | 0.250 | -2.180 |
| val | B-a0.1-T0.25-n0 | 0.1 | 0.035 | 0.961 | 0.038 | 0.510 |
| val | B-a0.1-T0.25-n1 | 0.1 | 0.037 | 0.962 | 0.040 | 0.494 |
| val | B-a0.1-T1-n0 | 0.1 | 0.003 | 0.987 | 0.033 | 0.576 |
| val | B-a0.1-T1-n1 | 0.1 | 0.005 | 0.988 | 0.033 | 0.573 |
| val | B-a0.3-T0.25-n0 | 0.3 | 0.074 | 0.949 | 0.085 | -0.091 |
| val | B-a0.3-T0.25-n1 | 0.3 | 0.065 | 0.951 | 0.079 | -0.016 |
| val | B-a0.3-T1-n0 | 0.3 | 0.009 | 0.958 | 0.069 | 0.118 |
| val | B-a0.3-T1-n1 | 0.3 | 0.009 | 0.961 | 0.066 | 0.152 |
| val | B-learned-s0 | — | 0.030 | 0.957 | 0.070 | 0.101 |
| val | B-learned-s1 | — | 0.023 | 0.959 | 0.069 | 0.123 |
| val | B-learned-s2 | — | 0.028 | 0.958 | 0.070 | 0.108 |
| val | B-marginal | — | 0.011 | 0.944 | 0.078 | -0.000 |
| val | B-oracle | — | 0.001 | 1.000 | 0.000 | 1.000 |
| val | B-uniform | — | 0.274 | 0.500 | 0.250 | -2.198 |

Brier skill is a descriptive summary only; calibration is measured, not assumed.

## Leakage probes

| Split | Probe | T5 |
|---|---|---|
| test | Availability bits only (descriptive) | 43.13 |
| test | HPI TF-IDF | 62.22 |
| test | HPI TF-IDF + 12 availability bits | 64.20 |
| val | Availability bits only (descriptive) | 46.46 |
| val | HPI TF-IDF | 62.32 |
| val | HPI TF-IDF + 12 availability bits | 64.77 |

A weak probe cannot certify that leakage is absent.

## Interpretation (SPEC §4.5, §11)

No Family-B contrast is statistically significant. This is a finding about this fixed reranker at the measured world-model precision, with the corruption arms and the oracle contrast above, and no general claim about availability.
Conclusions are limited to this fixed planner at the measured precision. Perfect information does not make the heuristic optimal, so a non-positive oracle contrast does not show that availability has no value. Learned seeds 1 and 2 and the corruption arms are descriptive.

The inventory card and HPI audit are published beside this report (`inventory_card.json`, `hpi_audit.json`).

## Limitations

- Previously examined benchmark; one policy seed, so between-seed variation is unassessable.
- Retrospective: availability in the chart reflects completed workups, not what a test would show.
- No clinical-safety inference.
