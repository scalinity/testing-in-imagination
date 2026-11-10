
# Commit guard and chart-availability reranking on LA-CDM

**Summary.** On the 240 test admissions, the commit guard (V3) raises mean per-class recall (T5) from 73.09 to 81.83: ΔT5 = +8.75 [4.67, 13.35] percentage points, statistically significant (paired sign-flip randomization, p < 0.0001). That is a 12.0% relative increase, and the average share of each disease's patients that are missed falls from 26.91% to 18.17%, 32.5% fewer. The comparator is the replication's trained Nemotron Nano 8B policy; these results are not a reproduction of the paper.

## The question

The trained LA-CDM policy ends 29 of 240 test episodes without a diagnosis, and the frozen metric counts each of those as wrong. In the guard-off re-run (240/240 predictions identical to the official rows), 6 hit the step limit, 21 ran out of context (9 while generating, 12 after an observation) and 2 ended on a malformed action. Phase A asks whether guarding the policy's actions and committing to an answer recovers those admissions. Phase B asks whether reweighting the policy's test requests by how likely each test is to be in the chart, given the history of present illness (HPI), helps further.

## What was built

- **G1 validity guard.** When the policy proposes a malformed, unknown or repeated action, the guard replaces it with the candidate the policy itself scores highest (teacher-forced log-probability).
- **G2 budget commit.** At the last allowed step, or when the trajectory would overflow, the guard makes the policy commit to its highest-scoring diagnosis.
- **G3 hypothesis fallback.** If an episode still ends without a diagnosis, the hypothesis agent's latest valid hypothesis becomes the answer. It reuses an answer the system already produced, so it is not a new diagnostic predictor.
- **Phase B reranking.** A TF-IDF logistic regression per test estimates p(test present in chart | HPI); a fixed planner reweights the policy's test requests on top of V3. The prevalence-only, uniform, oracle and corrupted arms isolate where any change comes from.

## How it was tested

- The protocol (`prereg/protocol.json`) was frozen before any test access; amendments A1–A3 record code fixes made before the affected test runs. Test-set status: a prospectively frozen follow-up analysis on a previously examined benchmark. These published test aggregates informed the design: per-arm confusion matrices with an undiagnosed column (greedy and temperature 1.0), undiagnosed / valid / repeated request counts, the cost decomposition (267 unavailable requests), ignore/* metrics (84.62), test hypothesis accuracy (81.67% over 622 triplets), and the zero-shot hypothesis distribution (904 of 1,095 cholecystitis) with its undiagnosed-by-request-count breakdown. No patient rows or trajectories were viewed. The hypothesis fallback (G3) has not been computed on test rows by anyone. Related test results already seen: the undiagnosed counts per arm and the C1 hypothesis accuracy.
- Re-running the policy with every guard off reproduced 240/240 of the official test predictions.
- ΔT5 is tested with a two-sided paired sign-flip randomization test; intervals are 95% class-stratified paired bootstrap; comparisons within Family A and Family B are Holm-adjusted.

## Phase A: commit guard

| Arm | Mean recall (T5) | Micro F1 (T6) | Macro F1 (T7) | Avg test cost (T8) | Nonanswers |
|---|---|---|---|---|---|
| Trained policy, official rows (comparator) | 73.09 | 81.15 | 78.20 | $1,439.58 | 29/240 |
| V1: fallback only | 84.45 | 86.67 | 84.56 | $1,439.58 | 0/240 |
| V3: G1 + G2 + G3 (method) | 81.83 | 84.17 | 81.53 | $1,514.87 | 0/240 |
| Paper, as reported | 81.30 | 84.10 | 81.30 | $1,295.61 | — |

- **Recovery.** V3 answered 29 of the 29 admissions the trained policy left without a diagnosis, 21 of them correctly; 2 admissions the policy had right became wrong.
- **Where the gain comes from.** V1 − official (fallback only): +11.37 [7.28, 15.96], statistically significant (Holm-adjusted p < 0.0001). V3 − V1 (adding the online guards): -2.62 [-6.16, 0.76], not statistically significant (Holm-adjusted p = 0.1875).
- **Cost.** Average provided-test cost is $1,514.87 for V3, $1,439.58 for V1 and $1,439.58 for the trained policy.
- **The paper's figure.** The paper reports T5 81.30; V3 is 0.53 points above it. That figure comes from the authors' own run and has no interval here, so this comparison is descriptive, not a test.

| Arm | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| Trained policy | 90.62 | 72.31 | 72.00 | 57.41 |
| V1 | 94.79 | 81.54 | 80.00 | 81.48 |
| V3 | 94.79 | 78.46 | 80.00 | 74.07 |

Per-class recall (%) on the test admissions.

The fallback alone (V1) scores 84.45, and the online guards change that by an amount that is not statistically significant: the hypothesis agent already held the missing answers, and the online guards added nothing measurable.

## Phase B: chart-availability reranking

| Arm | Mean recall (T5) | ΔT5 vs P0 [95% CI], descriptive | Avg test cost (T8) | Nonanswers |
|---|---|---|---|---|
| P0 = V3, no reranking | 81.83 | — | $1,514.87 | 0/240 |
| Learned availability (seed 0) | 82.91 | +1.08 [-1.69, 4.23] | $1,516.30 | 0/240 |
| Test prevalence only | 80.98 | -0.85 [-3.39, 1.62] | $1,512.20 | 0/240 |
| Uniform availability | 84.06 | +2.23 [-0.28, 5.27] | $1,502.25 | 0/240 |
| Privileged oracle (true chart availability) | 85.53 | +3.70 [0.15, 7.77] | $1,593.34 | 0/240 |

| Family B contrast | ΔT5 [95% CI] | Holm-adjusted p | Statement |
|---|---|---|---|
| B-learned-s0 − B-marginal (patient-specific beyond test prevalence) | +1.93 [-0.46, 4.85] | 0.9375 | not statistically significant |
| B-learned-s0 − B-uniform (information beyond the reranking mechanism) | -1.15 [-3.70, 1.47] | 0.9375 | not statistically significant |
| B-oracle − B-uniform (privileged availability, same mechanism) | +1.46 [-1.71, 4.98] | 0.9375 | not statistically significant |

No Family-B contrast is statistically significant: at the measured world-model precision, patient-specific availability estimates did not measurably beat prevalence-only or uniform weights under this fixed reranker. That is a finding about this reranker, not a general claim about chart availability.

The learned model's test AUROC for chart presence is 0.959. The oracle arm sees which tests are actually in the chart, so its difference from P0 is descriptive only. It is privileged: the availability bits alone predict the diagnosis with T5 43.13 on test, against 25.00 by chance, because chart presence encodes the clinicians' completed workup.

## Limitations

- A previously examined benchmark (see the test-set status above) and one policy seed, so between-seed variation is unassessable.
- The guard scores candidates with the policy's own likelihood, not a calibrated posterior; verbalizer and length effects remain.
- Phase B is retrospective: chart availability reflects completed workups, not what a test would show.
- No clinical-safety inference; person-level overlap is unassessable because the release ships no subject_id.

## More detail

- Full reports: [`phase-a/REPORT.md`](phase-a/REPORT.md), [`phase-b/REPORT.md`](phase-b/REPORT.md).
- Specification, preregistration and decisions: [`SPEC.md`](../../SPEC.md), [`prereg/`](../../prereg/), [`DECISIONS.md`](../../DECISIONS.md).
