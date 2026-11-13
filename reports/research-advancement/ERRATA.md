# Errata to earlier studies, recorded by study TII-CEE-1

Each erratum corrects wording only, names the historical file it qualifies, and changes no number.
The historical files are preserved unchanged.

## ERRATUM-001: scope of the availability null (REVIEW-010)

- Qualifies: cdm-extension/daniel-escalante/results/credentialed/DIAGNOSTICS.md as rendered at
  main e5ad93f, opening verdict, the clause "even perfect chart availability gives this fixed
  planner no detectable gain, so a better availability predictor cannot help it".
- Why: that clause combines a nonsignificant contrast with a minimum detectable effect (4.15
  points) borrowed from another contrast and used as if it were an equivalence margin, under a
  decision rule written after the validation point estimates were seen. Perfect availability
  supplied to one fixed heuristic does not establish anything about less perfect predictors or
  other planners.
- Scoped wording that replaces the clause for every later citation: "This fixed planner showed no
  detectable T5 benefit from true availability over the uniform control on this validation cohort;
  modest effects and benefits from other planners remain unresolved."
- Unchanged: B-oracle − B-uniform −0.11 [−2.56, 2.38] T5 points (Holm p = 0.9736); unavailable
  requests 203 under B-uniform and 66 under B-oracle; the dose-response slope −2.46 [−7.28, 2.26].
- Recorded: November 13, 2026 (specification 3.5 and 8.6).
