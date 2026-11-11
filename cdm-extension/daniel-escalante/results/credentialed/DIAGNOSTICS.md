
# Phase-B validation diagnostics: planner or signal?

**Verdict: planner bottleneck.** B-oracle − B-uniform is -0.11 [-2.56, 2.38] points, not statistically significant (Holm-adjusted p = 0.9736), and the interval's upper bound is below 4.15 points, the smallest effect the test design detects with 80% power: even perfect chart availability gives this fixed planner no detectable gain, so a better availability predictor cannot help it. The planner does act on the signal: unavailable requests fall from 203 under B-uniform to 66 under B-oracle. The corruption dose-response was not run (it needs GPU runs), so this rests on the oracle contrast alone.

## Scope

- Exploratory and validation only: the 240 validation admissions of val-a and val-b, analyzed under amendment A4 (`prereg/amendments/A4-validation-diagnostics.md`), committed before this stage read any row. No test admission was read; nothing was tuned or selected.
- analyze-val-b had already published every arm's validation T5, so these contrasts are not blind, and the verdict rule and its threshold were written after those point estimates were visible.
- Conclusions hold for this fixed planner (SPEC §4.4) and one policy seed. These results extend a replication attempt and are not a reproduction of the LA-CDM paper.

## Does perfect chart availability help this planner? (Family D)

| Contrast | ΔT5 [95% CI] | p | Holm p | Discordant | Result |
|---|---|---|---|---|---|
| B-oracle − B-uniform | -0.11 [-2.56, 2.38] | 0.9736 | 0.9736 | 11 | not statistically significant |
| B-oracle − P0 | +1.47 [-0.72, 3.83] | 0.2305 | 0.4609 | 9 | not statistically significant |

Two-sided paired sign-flip randomization tests, class-stratified paired bootstrap intervals, Holm within Family D at α = 0.05. B-oracle − P0's interval matches the published analyze-val-b interval for the same seed ([-0.72, 3.83]).

## Does the planner act on the signal?

Per-arm validation summaries from analyze-val-b:

| Arm | T5 | Requested | Provided | Unavailable | Reranks | Planner stops |
|---|---|---|---|---|---|---|
| P0 = V3, no reranking | 83.12 | 601 | 394 | 207 | 0 | 0 |
| B-uniform (no availability information) | 84.70 | 594 | 391 | 203 | 69 | 0 |
| B-learned-s0 (learned availability) | 82.92 | 588 | 435 | 153 | 159 | 0 |
| B-oracle (true chart availability) | 84.59 | 559 | 493 | 66 | 242 | 6 |

## Which tests did the reranker add or drop?

Each arm against P0, patient by patient, each test counted once. *Extra* tests are ones the arm requested and P0 did not; *dropped* tests the reverse. *Provided* means the result was in the chart. *Relevant* means diagnosis-relevant to the patient's labeled disease under amendment A4's rule: the guideline tests per pathology in Hager et al. (2024), the dataset's authors. Net relevant results = extra relevant and provided − dropped relevant and provided. Aggregate counts only.

| Arm | Tests | Extra | provided | relevant | relevant, provided | Dropped | provided | relevant | relevant, provided | Net relevant results |
|---|---|---|---|---|---|---|---|---|---|---|
| B-uniform | Lab panels | 23 | 22 | 13 | 13 | 28 | 28 | 17 | 17 | -4 |
| B-uniform | Imaging | 30 | 10 | 15 | 5 | 33 | 8 | 26 | 6 | -1 |
| B-uniform | Physical exam | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | +1 |
| B-uniform | All | 54 | 33 | 29 | 19 | 61 | 36 | 43 | 23 | -4 |
| B-learned-s0 | Lab panels | 81 | 78 | 25 | 25 | 35 | 34 | 14 | 14 | +11 |
| B-learned-s0 | Imaging | 24 | 10 | 10 | 5 | 85 | 15 | 66 | 12 | -7 |
| B-learned-s0 | Physical exam | 2 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | +2 |
| B-learned-s0 | All | 107 | 90 | 37 | 32 | 120 | 49 | 80 | 26 | +6 |
| B-oracle | Lab panels | 126 | 125 | 48 | 48 | 40 | 39 | 18 | 18 | +30 |
| B-oracle | Imaging | 24 | 14 | 10 | 5 | 160 | 9 | 116 | 6 | -1 |
| B-oracle | Physical exam | 8 | 8 | 8 | 8 | 0 | 0 | 0 | 0 | +8 |
| B-oracle | All | 158 | 147 | 66 | 61 | 200 | 48 | 134 | 24 | +37 |

By test (extra / dropped):

| Test | Relevant to | B-uniform | B-learned-s0 | B-oracle |
|---|---|---|---|---|
| CBC | all four | +8 / −6 | +14 / −6 | +26 / −6 |
| BMP | pancreatitis | +1 / −1 | +8 / −2 | +14 / −3 |
| CMP | cholecystitis, pancreatitis | +1 / −7 | +10 / −10 | +11 / −13 |
| RFP | pancreatitis | +2 / −3 | +8 / −4 | +12 / −4 |
| LFP | cholecystitis | +4 / −6 | +16 / −5 | +23 / −8 |
| UA | none | +4 / −3 | +18 / −6 | +28 / −4 |
| EP | pancreatitis | +3 / −2 | +7 / −2 | +12 / −2 |
| CT | appendicitis, diverticulitis, pancreatitis | +5 / −0 | +7 / −0 | +5 / −28 |
| MRI | appendicitis, cholecystitis, diverticulitis | +13 / −5 | +5 / −20 | +5 / −32 |
| XR | none | +9 / −7 | +9 / −10 | +12 / −17 |
| US | all four | +3 / −21 | +3 / −55 | +2 / −83 |
| PE | all four | +1 / −0 | +2 / −0 | +8 / −0 |

## Corruption dose-response: not run

The eight corruption arms and B-marginal were run only on test (SPEC §9); validation has only α = 0 (B-oracle). Amendment A4 names the slope test (T5 on α, class-stratified bootstrap interval), but it needs those 9 arms on validation: about 4.1 GPU-hours at addendum B's per-arm budget, an estimate rather than a measurement. It waits for an approval decision. Until then, whether recall falls as the availability signal degrades is untested on validation.

## Limitations

- 240 validation admissions and one policy seed; between-seed variation is unassessable.
- Relevance is guideline relevance of the requested action, not the value of the result it returned.
- A null oracle contrast concerns this fixed planner only: perfect information does not make the heuristic optimal (SPEC §4.5).
- Every number here is rendered from `diagnose-val-b/diagnostics.json`, `analyze-val-b/validation_b.json` and the committed addenda A and B.
