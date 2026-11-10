> **SYNTHETIC** — invented records from the CPU fixture at fixture scale. These numbers exercise the pipeline and say nothing about the method.


# Commit guard on LA-CDM: mean recall 8.33 → 91.67

| | Mean recall (T5) | Nonanswers | Avg test cost |
|---|---|---|---|
| Trained policy (comparator) | 8.33 | 20/24 | $714.79 |
| + hypothesis fallback (V1) | 87.50 | 1/24 | $714.79 |
| **V3: guards + fallback (method)** | 91.67 | 0/24 | $714.79 |
| V3 + learned availability reranking | 91.67 | 0/24 | $581.67 |
| Paper, as reported | 81.30 | — | $1,295.61 |

- **ΔT5 (V3 − comparator) = +83.33 [66.67, 95.83]** points, statistically significant (paired sign-flip p < 0.0001, 24 test admissions).
- The fallback carries the gain: V1 − comparator +79.17 [62.50, 95.83], statistically significant; V3 − V1 +4.17 [0.00, 16.67], not statistically significant (Holm).
- Availability reranking: 0 of 3 Family-B contrasts statistically significant (Holm); privileged oracle − V3 +0.00 [0.00, 0.00], descriptive.

These results are not a reproduction of the paper. One policy seed, a previously examined benchmark, no clinical-safety inference. Every number is rendered from `phase-a/metrics.json`, `phase-b/metrics.json` and the replication's `paper_values.json`.
