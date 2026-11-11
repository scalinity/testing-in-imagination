> **SYNTHETIC** — invented records from the CPU fixture at fixture scale. These numbers exercise the pipeline and say nothing about the method.


# Phase-B validation diagnostics: planner or signal?

**Provisional: planner bottleneck.** This verdict rests on one validation contrast, the corruption dose-response that would confirm it has not run, and the verdict rule and its threshold were written after the validation point estimates were known. B-oracle − B-uniform is +0.00 [0.00, 0.00] points, not statistically significant (Holm-adjusted p = 1.0000), and the interval's upper bound is below 4.15 points, the smallest effect the test design detects with 80% power: even perfect chart availability gives this fixed planner no detectable gain, so a better availability predictor cannot help it. The planner does act on the signal: unavailable requests fall from 13 under B-uniform to 0 under B-oracle.

## Scope

- Exploratory and validation only: the 24 validation admissions of val-a and val-b, analyzed under amendment A4 (`prereg/amendments/A4-validation-diagnostics.md`), committed before this stage read any row, with A5's additions committed before the re-run that computed them. No test admission was read; nothing was tuned or selected.
- analyze-val-b had already published every arm's validation T5, so these contrasts are not blind, and the verdict rule and its threshold were written after those point estimates were visible.
- Conclusions hold for this fixed planner (SPEC §4.4) and one policy seed. These results extend a replication attempt and are not a reproduction of the LA-CDM paper.

## Does perfect chart availability help this planner? (Family D)

| Contrast | ΔT5 [95% CI] | p | Holm p | Discordant | Result |
|---|---|---|---|---|---|
| B-oracle − B-uniform | +0.00 [0.00, 0.00] | 1.0000 | 1.0000 | 0 | not statistically significant |
| B-oracle − P0 | +0.00 [0.00, 0.00] | 1.0000 | 1.0000 | 0 | not statistically significant |

Two-sided paired sign-flip randomization tests, class-stratified paired bootstrap intervals, Holm within Family D at α = 0.05. B-oracle − P0's interval matches the published analyze-val-b interval for the same seed ([0.00, 0.00]).

## Does the planner act on the signal?

Per-arm validation summaries from analyze-val-b:

| Arm | T5 | Requested | Provided | Unavailable | Reranks | Planner stops |
|---|---|---|---|---|---|---|
| P0 = V3, no reranking | 91.67 | 36 | 25 | 11 | 0 | 0 |
| B-uniform (no availability information) | 91.67 | 42 | 29 | 13 | 40 | 0 |
| B-learned-s0 (learned availability) | 91.67 | 14 | 13 | 1 | 2 | 0 |
| B-oracle (true chart availability) | 91.67 | 37 | 37 | 0 | 35 | 0 |

## Which tests did the reranker add or drop?

Each arm against P0, patient by patient, each test counted once. *Extra* tests are ones the arm requested and P0 did not; *dropped* tests the reverse. *Provided* means the result was in the chart. *Relevant* means diagnosis-relevant to the patient's labeled disease under amendment A4's rule: the guideline tests per pathology in Hager et al. (2024), the dataset's authors. Net relevant results = extra relevant and provided − dropped relevant and provided. Aggregate counts only.

| Arm | Tests | Extra | provided | relevant | relevant, provided | Dropped | provided | relevant | relevant, provided | Net relevant results |
|---|---|---|---|---|---|---|---|---|---|---|
| B-uniform | Lab panels | 30 | 20 | 0 | 0 | 4 | 3 | 0 | 0 | +0 |
| B-uniform | Imaging | 0 | 0 | 0 | 0 | 14 | 8 | 8 | 5 | -5 |
| B-uniform | Physical exam | 0 | 0 | 0 | 0 | 6 | 5 | 6 | 5 | -5 |
| B-uniform | All | 30 | 20 | 0 | 0 | 24 | 16 | 14 | 10 | -10 |
| B-learned-s0 | Lab panels | 0 | 0 | 0 | 0 | 14 | 10 | 0 | 0 | +0 |
| B-learned-s0 | Imaging | 0 | 0 | 0 | 0 | 6 | 0 | 3 | 0 | +0 |
| B-learned-s0 | Physical exam | 0 | 0 | 0 | 0 | 2 | 2 | 2 | 2 | -2 |
| B-learned-s0 | All | 0 | 0 | 0 | 0 | 22 | 12 | 5 | 2 | -2 |
| B-oracle | Lab panels | 25 | 25 | 0 | 0 | 4 | 0 | 0 | 0 | +0 |
| B-oracle | Imaging | 0 | 0 | 0 | 0 | 14 | 8 | 8 | 5 | -5 |
| B-oracle | Physical exam | 0 | 0 | 0 | 0 | 6 | 5 | 6 | 5 | -5 |
| B-oracle | All | 25 | 25 | 0 | 0 | 24 | 13 | 14 | 10 | -10 |

By test (extra / dropped):

| Test | Relevant to | B-uniform | B-learned-s0 | B-oracle |
|---|---|---|---|---|
| CBC | none | +8 / −0 | +0 / −2 | +5 / −0 |
| BMP | none | +8 / −0 | +0 / −2 | +7 / −1 |
| CMP | none | +4 / −0 | +0 / −2 | +4 / −1 |
| RFP | none | +4 / −0 | +0 / −2 | +2 / −0 |
| LFP | none | +2 / −0 | +0 / −2 | +4 / −1 |
| UA | none | +2 / −2 | +0 / −2 | +2 / −1 |
| EP | none | +2 / −2 | +0 / −2 | +1 / −0 |
| CT | appendicitis, diverticulitis, pancreatitis | +0 / −8 | +0 / −0 | +0 / −8 |
| MRI | appendicitis, cholecystitis, diverticulitis | +0 / −2 | +0 / −2 | +0 / −2 |
| XR | none | +0 / −2 | +0 / −2 | +0 / −2 |
| US | all four | +0 / −2 | +0 / −2 | +0 / −2 |
| PE | all four | +0 / −6 | +0 / −2 | +0 / −6 |

## Sensitivity S1 (exploratory): urinalysis and radiographs counted as relevant

Amendment A5, defined after the counts above were seen. The guideline map stays the primary definition because it was fixed in advance; S1 counts urinalysis and radiographs as relevant to all four diseases, adding their per-test counts from the same bundle.

| Arm | Extra relevant | Extra relevant, provided | Dropped relevant | Dropped relevant, provided | Net relevant results (S1) | Net (primary) |
|---|---|---|---|---|---|---|
| B-uniform | 2 | 1 | 18 | 11 | -10 | -10 |
| B-learned-s0 | 0 | 0 | 9 | 3 | -3 | -2 |
| B-oracle | 2 | 2 | 17 | 10 | -8 | -10 |

## Do the added results repeat what the policy already knew?

An untested hypothesis, checked descriptively (amendment A5, check H; aggregate counts only). For each added test that was relevant and provided, the hypothesis agent's hypothesis when the test was requested is compared with the patient's label, and with the hypothesis stored after the result came back. If most added relevant results arrived while the policy already held the label and rarely changed its hypothesis, that is consistent with the hypothesis; these counts describe it and do not test it. Rows whose stored hypotheses cannot be aligned with their requests are excluded and counted.

Rows whose stored hypotheses align with their requests: B-learned-s0 21/24, B-oracle 21/24, B-uniform 21/24, P0 21/24.

Added by each arm (extra tests, relevant and provided):

| Arm | Tests | Unaligned | Already held the label | Held another disease | Held unknown | Hypothesis changed | Changed to the label | Changed away from the label |
|---|---|---|---|---|---|---|---|---|
| B-uniform | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B-learned-s0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B-oracle | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

For comparison, dropped from P0 (P0's own rows; relevant and provided):

| Arm | Tests | Unaligned | Already held the label | Held another disease | Held unknown | Hypothesis changed | Changed to the label | Changed away from the label |
|---|---|---|---|---|---|---|---|---|
| B-uniform | 9 | 1 | 7 | 2 | 0 | 0 | 0 | 0 |
| B-learned-s0 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| B-oracle | 9 | 1 | 7 | 2 | 0 | 0 | 0 | 0 |

## Pending

- **Corruption dose-response, the confirmatory check for this verdict** (amendment A4, D3). The eight corruption arms and B-marginal were run only on test (SPEC §9); validation has only α = 0 (B-oracle). It needs 9 validation runs: about 4.1 GPU-hours at addendum B's per-arm budget, an estimate rather than a measurement. Stage `dose-val-b` is configured and its approval request drafted; nothing has been submitted. Until it runs, whether recall falls as the availability signal degrades is untested on validation.

## Limitations

- 24 validation admissions and one policy seed; between-seed variation is unassessable.
- Relevance is guideline relevance of the requested action, not the value of the result it returned. The guideline map, fixed in advance, counts urinalysis and radiographs as relevant to none of the four diseases; sensitivity S1 (exploratory) counts them as relevant to all four.
- Check H reads the hypothesis agent's stored hypotheses, not the decision policy's internal belief, and says nothing about what each result showed.
- A null oracle contrast concerns this fixed planner only: perfect information does not make the heuristic optimal (SPEC §4.5).
- Every number here is rendered from `diagnose-val-b/diagnostics.json`, `diagnose-val-b/hypothesis_check.json`, `analyze-val-b/validation_b.json` and the committed addenda A and B.
