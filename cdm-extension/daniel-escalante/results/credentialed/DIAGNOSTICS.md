
# Phase-B validation diagnostics: planner or signal?

**Planner bottleneck.** This verdict combines one validation contrast with the validation dose-response (amendment A6); the oracle-contrast rule and its threshold were written after the validation point estimates were known. B-oracle − B-uniform is -0.11 [-2.56, 2.38] points, not statistically significant (Holm-adjusted p = 0.9736), and the interval's upper bound is below 4.15 points, the smallest effect the test design detects with 80% power: even perfect chart availability gives this fixed planner no detectable gain, so a better availability predictor cannot help it. The planner does act on the signal: unavailable requests fall from 203 under B-uniform to 66 under B-oracle. The corruption dose-response (T = 1) has slope -2.46 [-7.28, 2.26] T5 points per unit α: flat.

## Scope

- Exploratory and validation only: the 240 validation admissions of val-a and val-b, analyzed under amendment A4 (`prereg/amendments/A4-validation-diagnostics.md`), committed before this stage read any row, with A5's additions committed before the re-run that computed them. No test admission was read; nothing was tuned or selected.
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

## Sensitivity S1 (exploratory): urinalysis and radiographs counted as relevant

Amendment A5, defined after the counts above were seen. The guideline map stays the primary definition because it was fixed in advance; S1 counts urinalysis and radiographs as relevant to all four diseases, adding their per-test counts from the same bundle.

| Arm | Extra relevant | Extra relevant, provided | Dropped relevant | Dropped relevant, provided | Net relevant results (S1) | Net (primary) |
|---|---|---|---|---|---|---|
| B-uniform | 42 | 27 | 53 | 28 | -1 | -4 |
| B-learned-s0 | 64 | 52 | 96 | 35 | +17 | +6 |
| B-oracle | 106 | 97 | 155 | 31 | +66 | +37 |

## Do the added results repeat what the policy already knew? (exploratory)

An untested hypothesis, checked descriptively (amendment A5, check H; aggregate counts only). For each added test that was relevant and provided, the hypothesis agent's hypothesis when the test was requested is compared with the patient's label, and with the hypothesis stored after the result came back. If most added relevant results arrived while the policy already held the label and rarely changed its hypothesis, that is consistent with the hypothesis; these counts describe it and do not test it. Rows whose stored hypotheses cannot be aligned with their requests are excluded and counted.

Rows whose stored hypotheses align with their requests: B-learned-s0 239/240, B-oracle 240/240, B-uniform 240/240, P0 239/240.

Added by each arm (extra tests, relevant and provided):

| Arm | Tests | Unaligned | Already held the label | Held another disease | Held unknown | Hypothesis changed | Changed to the label | Changed away from the label |
|---|---|---|---|---|---|---|---|---|
| B-uniform | 19 | 0 | 17 | 2 | 0 | 1 | 0 | 1 |
| B-learned-s0 | 32 | 0 | 26 | 6 | 0 | 3 | 3 | 0 |
| B-oracle | 61 | 0 | 38 | 23 | 0 | 7 | 3 | 3 |

For comparison, dropped from P0 (P0's own rows; relevant and provided):

| Arm | Tests | Unaligned | Already held the label | Held another disease | Held unknown | Hypothesis changed | Changed to the label | Changed away from the label |
|---|---|---|---|---|---|---|---|---|
| B-uniform | 22 | 1 | 11 | 11 | 0 | 3 | 1 | 1 |
| B-learned-s0 | 26 | 0 | 17 | 9 | 0 | 3 | 1 | 1 |
| B-oracle | 23 | 1 | 13 | 10 | 0 | 3 | 1 | 2 |

## Corruption dose-response (D3)

Amendment A4's D3, with its code fixed before any dose-val-b run (amendment A6): the OLS slope of validation T5 on the corruption level α, one point per arm, with a 95% class-stratified percentile bootstrap that draws patients once per resample for every arm. Decreasing if the interval lies below 0, increasing if above, flat otherwise.

**Primary, T = 1.** Slope -2.46 [-7.28, 2.26] T5 points per unit α (flat; 10000 resamples). Mean T5 by level: α 0.0: 84.59, α 0.1: 85.12, α 0.3: 84.44, α 0.5: 83.65; the level means do not fall monotonically as α grows.

| Arm | α | T5 |
|---|---|---|
| B-oracle | 0.0 | 84.59 |
| B-a0.1-T1-n0 | 0.1 | 84.88 |
| B-a0.1-T1-n1 | 0.1 | 85.35 |
| B-a0.3-T1-n0 | 0.3 | 84.96 |
| B-a0.3-T1-n1 | 0.3 | 83.93 |
| B-marginal | 0.5 | 83.65 |

**Secondary, T = 0.25.** Slope -0.68 [-10.85, 9.79] T5 points per unit α (flat; 10000 resamples). Mean T5 by level: α 0.0: 84.59, α 0.1: 86.14, α 0.3: 85.08; the level means do not fall monotonically as α grows.

| Arm | α | T5 |
|---|---|---|
| B-oracle | 0.0 | 84.59 |
| B-a0.1-T0.25-n0 | 0.1 | 87.05 |
| B-a0.1-T0.25-n1 | 0.1 | 85.24 |
| B-a0.3-T0.25-n0 | 0.3 | 85.27 |
| B-a0.3-T0.25-n1 | 0.3 | 84.89 |

## Limitations

- 240 validation admissions and one policy seed; between-seed variation is unassessable.
- Relevance is guideline relevance of the requested action, not the value of the result it returned. The guideline map, fixed in advance, counts urinalysis and radiographs as relevant to none of the four diseases; sensitivity S1 (exploratory) counts them as relevant to all four.
- Check H reads the hypothesis agent's stored hypotheses, not the decision policy's internal belief, and says nothing about what each result showed.
- A null oracle contrast concerns this fixed planner only: perfect information does not make the heuristic optimal (SPEC §4.5).
- Every number here is rendered from `diagnose-val-b/diagnostics.json`, `diagnose-val-b/hypothesis_check.json`, `analyze-val-b/validation_b.json`, the committed addenda A and B, and `analyze-dose-val-b/dose_response.json`.
