# Amendment C1 (content study): the diagnosis posterior's calibration

Date: 2026-11-11 (US Eastern). Written and committed before any code for these fixes exists and before they are computed. "C" amendments belong to the content world-model study (`SPEC_CONTENT_WM.md`); the A series belongs to the Phase A/B protocol.

## Scope

- **CPU only**, on QOS `cai6734-b`.
- **Data:** train for fitting, validation for selection. No test admission is read.
- **Outputs:** aggregates only.
- **Status:** validation-only development under the spec's §4. No GPU job.

## The problem

`wm-fit-c` (CPU job 50913048, `results/credentialed/wm-fit-c/content_wm.json`) found that the diagnosis posterior becomes overconfident once the charted labs are folded in:

| Validation, 240 admissions | HPI prior alone | After all charted labs |
|---|---|---|
| Log-loss | 0.690 | 2.485 |
| Top-label ECE | 0.046 | 0.295 |

The model multiplies one likelihood factor per panel item. Many of those factors are for "absent" items, which belong to a charted panel but were never reported. An unreported item says that the test was not done, not what the disease is.

## Candidate fixes

| Candidate | Definition |
|---|---|
| **none** | The current model. |
| **A** | In every update of the diagnosis belief, an item whose outcome is "absent" contributes no factor. This covers the posterior after observed panels, the imagined updates inside EIG, and the realized gain. |
| **A+B** | A, plus one scalar likelihood temperature τ: each observed panel's evidence (the product over its non-absent items) is raised to the power 1/τ. |

What none of the candidates changes:
- The content model's predictive distribution, p(item outcome | disease, panel), still includes "absent". Content log-loss and Brier still score the full outcome vector.
- The HPI prior keeps its own temperature.

How τ is chosen for A+B:
- **Grid:** τ ∈ {1, 1.5, 2, 3, 5, 8, 13, 20, 30, 50}.
- **Folds:** 5-fold cross-validation within the 240 validation admissions. Fold k holds the admissions with sha256("tim-tau-fold|val|<admission>") mod 5 = k.
- **Per fold:** τ is chosen on the other four folds, by the lowest posterior log-loss after all labs (ties go to the smaller τ). It is then scored on the held-out fold.
- **Reported score:** A+B's numbers below are these out-of-fold scores.
- **Stored τ:** chosen the same way on all 240 validation admissions.
- **Never per admission:** no admission gets its own τ.

## Metrics and choice

- **Primary:** validation diagnosis-posterior log-loss after all charted labs (natural log, mean over the 240 admissions).
- **Secondary:** top-label ECE of that posterior, 10 equal-width bins.
- **Fixed models:** none and A are fixed models, so their metrics are computed directly.
- **Rule:**
  1. Start from the candidate with the lowest log-loss. Every candidate within 0.01 nats of it counts as tied.
  2. Among tied candidates, the one with the lowest ECE wins.
  3. If tied candidates' ECEs are also within 0.005 of each other, the simplest wins, in the order none, then A, then A+B.

## After the choice

- The chosen candidate (and τ, for A+B) is stored in the private `content.pkl` and used by the planner.
- **Gate 2 is recomputed with it** (§5 of the spec): content log-loss skill > 0, and EIG-calibration Spearman > 0.
- If gate 2 fails, that is reported, and the study stops before any GPU request.
- `wm-fit-c` publishes the none / A / A+B table, the stored τ and the five fold-wise τ values. All of these are aggregates.
