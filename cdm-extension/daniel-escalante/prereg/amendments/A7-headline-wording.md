# Amendment A7: headline wording after the dose-response

Date: 2026-11-11 (US Eastern). Written after the D3 results were published (`analyze-dose-val-b/dose_response.json`), at Daniel's request.

## What changes

Only the wording of DIAGNOSTICS.md. A6's rule and every number are unchanged.

| Situation | A6 headline | A7 headline |
|---|---|---|
| Oracle contrast reads "planner bottleneck" and the primary dose-response is **flat** | "Planner bottleneck." | **"Planner bottleneck (no detectable effect)."** |
| Any other combination of A6 | unchanged | unchanged |

A flat curve means the analysis detected no effect of corrupting the availability signal. The new headline says so.

## Added text

- **Verdict paragraph:** when the primary curve is flat and its 95% interval extends below 0, the paragraph states plainly that the interval cannot rule out a decline of a few T5 points over the tested range.
- **Dose-response section:** the primary block gains one line of derived arithmetic, the slope and its interval multiplied by the tested span of α. It is arithmetic on the published numbers, not a separate estimate, and the line says so.

## Status

The rule and the numbers were fixed before the runs (A4, A6). This wording was chosen after the results were known.
