> **SYNTHETIC** — invented records from the CPU fixture at fixture scale. These numbers exercise the pipeline and say nothing about the method.


# Phase B: retrospective chart-availability reranking

## Framing

Phase B estimates p(test present in chart | HPI) and reweights the policy's test requests on top of V3 (P0). It does not predict result content, does not update from revealed findings and does not evaluate imagined outcomes: it is retrospective chart-availability reranking, not full world-model planning. The oracle is a privileged chart-availability oracle; chart presence encodes clinicians' completed workups. These results are not a reproduction of the paper.

Test-set status: a prospectively frozen follow-up analysis on a previously examined benchmark. These published test aggregates informed the design: per-arm confusion matrices with an undiagnosed column (greedy and temperature 1.0), undiagnosed / valid / repeated request counts, the cost decomposition (267 unavailable requests), ignore/* metrics (84.62), test hypothesis accuracy (81.67% over 622 triplets), and the zero-shot hypothesis distribution (904 of 1,095 cholecystitis) with its undiagnosed-by-request-count breakdown. No patient rows or trajectories were viewed. The hypothesis fallback (G3) has not been computed on test rows by anyone. Related test results already seen: the undiagnosed counts per arm and the C1 hypothesis accuracy.

## Arms

| Arm | T1 | T2 | T3 | T4 | T5 | T8 ($) | Nonanswers | Reranks | Stops |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 714.79 | 0 | 0 | 0 |
| B-uniform | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 433.33 | 0 | 40 | 0 |
| B-marginal | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 581.67 | 0 | 2 | 0 |
| B-learned-s0 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 581.67 | 0 | 2 | 0 |
| B-learned-s1 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 581.67 | 0 | 2 | 0 |
| B-learned-s2 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 581.67 | 0 | 2 | 0 |
| B-oracle | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 536.08 | 0 | 35 | 0 |
| B-a0.1-T1-n0 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 584.54 | 0 | 4 | 0 |
| B-a0.1-T1-n1 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 584.62 | 0 | 3 | 0 |
| B-a0.1-T0.25-n0 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 520.21 | 0 | 34 | 0 |
| B-a0.1-T0.25-n1 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 579.46 | 0 | 33 | 0 |
| B-a0.3-T1-n0 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 575.00 | 0 | 5 | 0 |
| B-a0.3-T1-n1 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 575.08 | 0 | 4 | 0 |
| B-a0.3-T0.25-n0 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 575.00 | 0 | 5 | 0 |
| B-a0.3-T0.25-n1 | 100.00 | 83.33 | 83.33 | 100.00 | 91.67 | 575.08 | 0 | 4 | 0 |

## Family B (Holm, α = 0.05)

| Contrast | ΔT5 [95% CI] | p | Holm p | Statement |
|---|---|---|---|---|
| B-learned-s0 − B-marginal (patient-specific beyond test prevalence) | +0.00 [0.00, 0.00] | 1.0000 | 1.0000 | not statistically significant |
| B-learned-s0 − B-uniform (information beyond the reranking mechanism) | +0.00 [0.00, 0.00] | 1.0000 | 1.0000 | not statistically significant |
| B-oracle − B-uniform (privileged availability, same mechanism) | +0.00 [0.00, 0.00] | 1.0000 | 1.0000 | not statistically significant |

## Every arm against P0 (descriptive, intervals only)

| Arm − P0 | ΔT5 [95% CI] |
|---|---|
| B-uniform − P0 | +0.00 [0.00, 0.00] |
| B-marginal − P0 | +0.00 [0.00, 0.00] |
| B-learned-s0 − P0 | +0.00 [0.00, 0.00] |
| B-learned-s1 − P0 | +0.00 [0.00, 0.00] |
| B-learned-s2 − P0 | +0.00 [0.00, 0.00] |
| B-oracle − P0 | +0.00 [0.00, 0.00] |
| B-a0.1-T1-n0 − P0 | +0.00 [0.00, 0.00] |
| B-a0.1-T1-n1 − P0 | +0.00 [0.00, 0.00] |
| B-a0.1-T0.25-n0 − P0 | +0.00 [0.00, 0.00] |
| B-a0.1-T0.25-n1 − P0 | +0.00 [0.00, 0.00] |
| B-a0.3-T1-n0 − P0 | +0.00 [0.00, 0.00] |
| B-a0.3-T1-n1 − P0 | +0.00 [0.00, 0.00] |
| B-a0.3-T0.25-n0 − P0 | +0.00 [0.00, 0.00] |
| B-a0.3-T0.25-n1 − P0 | +0.00 [0.00, 0.00] |

## World-model quality

| Split | Arm | α | ECE | AUROC | Brier | Brier skill vs marginal |
|---|---|---|---|---|---|---|
| test | B-a0.1-T0.25-n0 | 0.1 | 0.071 | 0.957 | 0.072 | 0.549 |
| test | B-a0.1-T0.25-n1 | 0.1 | 0.074 | 0.959 | 0.075 | 0.528 |
| test | B-a0.1-T1-n0 | 0.1 | 0.013 | 0.962 | 0.061 | 0.620 |
| test | B-a0.1-T1-n1 | 0.1 | 0.017 | 0.963 | 0.062 | 0.610 |
| test | B-a0.3-T0.25-n0 | 0.3 | 0.102 | 0.878 | 0.150 | 0.062 |
| test | B-a0.3-T0.25-n1 | 0.3 | 0.097 | 0.875 | 0.147 | 0.080 |
| test | B-a0.3-T1-n0 | 0.3 | 0.019 | 0.878 | 0.133 | 0.165 |
| test | B-a0.3-T1-n1 | 0.3 | 0.030 | 0.875 | 0.134 | 0.159 |
| test | B-learned-s0 | — | 0.007 | 0.806 | 0.160 | -0.003 |
| test | B-learned-s1 | — | 0.047 | 0.806 | 0.164 | -0.029 |
| test | B-learned-s2 | — | 0.016 | 0.802 | 0.161 | -0.007 |
| test | B-marginal | — | 0.002 | 0.795 | 0.160 | -0.000 |
| test | B-oracle | — | 0.001 | 1.000 | 0.000 | 1.000 |
| test | B-uniform | — | 0.069 | 0.500 | 0.250 | -0.565 |
| val | B-a0.1-T0.25-n0 | 0.1 | 0.095 | 0.938 | 0.096 | 0.398 |
| val | B-a0.1-T0.25-n1 | 0.1 | 0.074 | 0.951 | 0.076 | 0.526 |
| val | B-a0.1-T1-n0 | 0.1 | 0.023 | 0.944 | 0.079 | 0.506 |
| val | B-a0.1-T1-n1 | 0.1 | 0.016 | 0.958 | 0.064 | 0.597 |
| val | B-a0.3-T0.25-n0 | 0.3 | 0.117 | 0.866 | 0.159 | 0.003 |
| val | B-a0.3-T0.25-n1 | 0.3 | 0.085 | 0.891 | 0.136 | 0.148 |
| val | B-a0.3-T1-n0 | 0.3 | 0.015 | 0.866 | 0.138 | 0.134 |
| val | B-a0.3-T1-n1 | 0.3 | 0.033 | 0.891 | 0.126 | 0.212 |
| val | B-learned-s0 | — | 0.007 | 0.806 | 0.160 | -0.003 |
| val | B-learned-s1 | — | 0.047 | 0.806 | 0.164 | -0.029 |
| val | B-learned-s2 | — | 0.016 | 0.802 | 0.161 | -0.007 |
| val | B-marginal | — | 0.002 | 0.795 | 0.160 | -0.000 |
| val | B-oracle | — | 0.001 | 1.000 | 0.000 | 1.000 |
| val | B-uniform | — | 0.069 | 0.500 | 0.250 | -0.565 |

Brier skill is a descriptive summary only; calibration is measured, not assumed.

## Leakage probes

| Split | Probe | T5 |
|---|---|---|
| test | Availability bits only (descriptive) | 25.00 |
| test | HPI TF-IDF | 91.67 |
| test | HPI TF-IDF + 12 availability bits | 91.67 |
| val | Availability bits only (descriptive) | 25.00 |
| val | HPI TF-IDF | 91.67 |
| val | HPI TF-IDF + 12 availability bits | 91.67 |

A weak probe cannot certify that leakage is absent.

## Interpretation (SPEC §4.5, §11)

No Family-B contrast is statistically significant. This is a finding about this fixed reranker at the measured world-model precision, with the corruption arms and the oracle contrast above, and no general claim about availability.
Conclusions are limited to this fixed planner at the measured precision. Perfect information does not make the heuristic optimal, so a non-positive oracle contrast does not show that availability has no value. Learned seeds 1 and 2 and the corruption arms are descriptive.

The inventory card and HPI audit are published beside this report (`inventory_card.json`, `hpi_audit.json`).

## Limitations

- Previously examined benchmark; one policy seed, so between-seed variation is unassessable.
- Retrospective: availability in the chart reflects completed workups, not what a test would show.
- No clinical-safety inference.
