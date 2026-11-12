# Amendment A4: exploratory Phase-B validation diagnostics

Date: 2026-11-11 (US Eastern). Written and committed before the `diagnose-val-b` stage reads any row.

## Status

- **Exploratory.** Nothing here changes the frozen protocol, the Phase-A primary result, Family A or Family B.
- **Validation split only:** the 240 validation admissions already used by val-a and val-b. No test admission is read. Nothing is tuned or selected.
- **CPU only.** No GPU job runs under this amendment.
- **Already seen.** analyze-val-b published, on validation (`results/credentialed/analyze-val-b/validation_b.json`):
  - per-arm T5, request counts and planner events;
  - descriptive intervals for B-uniform − P0, B-learned-s0 − P0 and B-oracle − P0.

  The point estimates behind the contrasts below were therefore visible before this rule was written, so the contrasts are not blind. The test-b results in `results/credentialed/phase-b/metrics.json` are not used.
- These are extensions of a replication attempt, not a reproduction of the LA-CDM paper.

## Question

Is Phase B limited by its planner, the fixed reranker of SPEC §4.4, or by the chart-availability signal that planner receives?

## D1. Oracle contrasts (Family D)

The contrasts:
- **B-oracle − B-uniform:** does perfect chart availability raise mean class recall over the same reranking mechanism given no availability information?
- **B-oracle − P0:** the oracle-guided planner's whole-system effect over V3. P0 is val-a's V3 run, as in analyze-val-b.

Method:
- **Statistic:** ΔT5 = Σ w_i d_i over the 240 validation admissions (SPEC §6).
- **Test:** two-sided paired sign-flip randomization test. It is exact when 20 or fewer patients are discordant; otherwise it uses 100,000 draws. Seed: `stream_seed(269, "val", "tim-permutation:<comparison>")`.
- **Interval:** 95% class-stratified paired percentile bootstrap, 10,000 resamples, seed `stream_seed(269, "val", "tim-bootstrap")`. This is analyze-val-b's seed, so B-oracle − P0's interval must equal the published one; the report states whether it does.
- **Multiplicity:** Holm at α = 0.05 within Family D only. The SPEC §6 wording rule applies.

## D2. Tests the reranker added or dropped

Each val-b arm (B-uniform, B-learned-s0, B-oracle) is compared with P0's row for the same patient.

Definitions:
- **Extra tests:** tests the arm requested that P0 did not.
- **Dropped tests:** tests P0 requested that the arm did not.
- Each test counts once per patient, even if requested more than once.
- **Why these differences count as the reranker's:** decoding is greedy, and untriggered turns are byte-identical. The shadow control showed separate jobs reproduce generations exactly. So the arm and P0 diverge only through planner decisions, and every difference in the requested set follows from the reranker, either directly or through the history it changed.
- **Provided:** the result was in the chart, meaning the test is in that run's `provided_tests`.
- **Diagnosis-relevant:** relevant to the patient's labeled disease under the rule below.

Reporting:
- **Counts:** extra, extra provided, extra relevant, extra relevant and provided; dropped, dropped provided, dropped relevant, dropped relevant and provided.
- **Breakdowns:** by test code, by category and in total. The categories are:
  - lab: CBC, BMP, CMP, RFP, LFP, UA, EP;
  - imaging: CT, MRI, XR, US;
  - exam: PE.
- **Net relevant results received:** extra relevant and provided, minus dropped relevant and provided.
- Aggregate counts only. These are descriptive; no test is run on them.

### Relevance rule

Source: Hager et al. (2024), the authors of MIMIC-IV-Ext-CDM, as implemented in their public evaluation framework: `github.com/paulhager/MIMIC-Clinical-Decision-Making-Framework`, commit `e1b4a5228531ef9fa77bf57189fc5828d27acdcb`, files `evaluators/*_evaluator.py` and `tools/utils.py`.

- **Laboratory panel.** A panel is relevant to disease d if its item IDs include any of d's required lab item IDs. Its item IDs are the ones listed in the release's `lab_test_mapping.csv`, plus the IDs each listed item maps to. The required IDs:
  - **All four diseases, inflammation:** 51301, 51755 and 51300 (white blood cells), 50889 (C-reactive protein), 51652.
  - **Cholecystitis adds** liver: 50861 (ALT) and 50878 (AST); gallbladder: 50883 (bilirubin) and 50927 (GGT).
  - **Pancreatitis adds** pancreas: 50867 (amylase) and 50956 (lipase); severity: 51480 (hematocrit), 50810, 51221, 51638, 51006 (urea nitrogen), 52647, 51000 (triglycerides), 50893 (calcium), 50824 (sodium), 52623, 50983, 52610 (potassium), 50971 and 50822.

  The panel × disease result is computed on the cluster from the mapping file and published as booleans. The mapping is a test dictionary and holds no patient data.
- **Imaging.** The abdominal modalities each evaluator scores:
  - appendicitis: US, CT, MRI;
  - cholecystitis: US, MRI;
  - diverticulitis: CT, US, MRI;
  - pancreatitis: US, CT.

  Radiograph is scored for none. HIDA and EUS are not actions here. An action names only a modality, so the imaging region is not checked.
- **Physical examination:** relevant to all four, because every evaluator scores whether it was requested.
- **Scope:** this is guideline relevance per action. It does not judge what a result shows.

## D3. Corruption dose-response (named now, not run)

Why it is not run:
- The eight corruption arms and B-marginal were run only on test (SPEC §9).
- On validation, only α = 0 (B-oracle) exists.
- Running the missing arms on validation takes 9 GPU runs, which this amendment does not authorize. They wait for a decision.

If those runs are approved and completed, the analysis is:
- **Primary, T = 1:**
  - α = 0: B-oracle;
  - α = 0.1: B-a0.1-T1-n0 and B-a0.1-T1-n1;
  - α = 0.3: B-a0.3-T1-n0 and B-a0.3-T1-n1;
  - α = 0.5: B-marginal, by the identity SPEC §4.3 tests.
- **Secondary, T = 0.25:** B-oracle, B-a0.1-T0.25-n0 and -n1, and B-a0.3-T0.25-n0 and -n1.
- **Statistic:** the OLS slope of validation T5 on α, in T5 points per unit α. Each arm is one point, and the two noise seeds are separate points.
- **Interval:** 95% class-stratified percentile bootstrap, 10,000 resamples, seed `stream_seed(269, "val", "tim-dose-response")`. Each resample draws patients once and applies the same draw to every arm.
- **Reading:**
  - decreasing if the interval's upper bound is below 0;
  - flat if the interval includes 0;
  - increasing if its lower bound is above 0.

  Whether the per-level means fall monotonically is reported descriptively.

## D4. Verdict

DIAGNOSTICS.md opens with a one-line verdict, chosen by this rule:

| Verdict | Condition | Reading |
|---|---|---|
| **Signal bottleneck** | B-oracle − B-uniform is statistically significant (Holm) and positive. | The planner turns perfect availability into recall, so the shortfall lies in the predicted signal. |
| **Planner bottleneck** | B-oracle − B-uniform is statistically significant and negative. | Perfect availability does not give this planner a gain the study could detect. A better availability predictor cannot help it; the limit is in how this fixed planner turns availability into choices. |
| **Planner bottleneck** | B-oracle − B-uniform is not statistically significant, and its 95% interval's upper bound is below 4.15 points. | Same reading as the row above. |
| **Inconclusive** | Anything else: the interval reaches 4.15 points. | — |

Where the 4.15-point threshold comes from:
- It is the smallest ΔT5 the 240-admission test design detects with 80% power at Phase A's V1 − V0 validation discordance (`prereg/addendum-a.json`, `sensitivity.V1-V0.mde_80`).
- It is the smaller of the two committed minimum detectable effects, so it is the stricter bar for a planner verdict.
- This threshold and the rule were written after the validation point estimates were visible.

What the verdict does not use:
- The D2 counts and the request counts are reported as evidence about the mechanism. They do not change the verdict.
- Without D3, the verdict rests on the oracle contrast alone, and DIAGNOSTICS.md says so.
