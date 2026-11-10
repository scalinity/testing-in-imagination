
# Commit guard on LA-CDM: mean recall 73.09 → 81.83

| | Mean recall (T5) | Nonanswers | Avg test cost |
|---|---|---|---|
| Trained policy (comparator) | 73.09 | 29/240 | $1,439.58 |
| + hypothesis fallback (V1) | 84.45 | 0/240 | $1,439.58 |
| **V3: guards + fallback (method)** | 81.83 | 0/240 | $1,514.87 |
| V3 + learned availability reranking | 82.91 | 0/240 | $1,516.30 |
| Paper, as reported | 81.30 | — | $1,295.61 |

- **ΔT5 (V3 − comparator) = +8.75 [4.67, 13.35]** points, statistically significant (paired sign-flip p < 0.0001, 240 test admissions).
- The fallback carries the gain: V1 − comparator +11.37 [7.28, 15.96], statistically significant; V3 − V1 -2.62 [-6.16, 0.76], not statistically significant (Holm).
- Availability reranking: 0 of 3 Family-B contrasts statistically significant (Holm); privileged oracle − V3 +3.70 [0.15, 7.77], descriptive.

These results are not a reproduction of the paper. One policy seed, a previously examined benchmark, no clinical-safety inference. Every number is rendered from `phase-a/metrics.json`, `phase-b/metrics.json` and the replication's `paper_values.json`.
