# Amendment A6: the dose-response analysis is fixed in code before any run

Date: 2026-11-11 (US Eastern). Written after the code commit below and before any `dose-val-b` run.

## The commit

Amendment A4's D3 is implemented at commit **`e637745a0ac8c257d407f98bd45aadd49e505ed9`**:
- `tim/diagnostics.py`: `dose_response`, `DOSE_PRIMARY`, `DOSE_SECONDARY` and `DOSE_SEED_TAG`;
- the CPU stage `analyze-dose-val-b`;
- `schemas/dose_response.json`;
- tests in `tests/test_dose.py`.

## Evidence that no run preceded it

When this note was written, the admission ledger held no `dose-val-b` entry and no `dose-val-b` admission marker, and no job was queued. The check was a count on the cluster at 6:12 PM EST, 2026-11-11.

## The analysis, as A4 states it

- **Arms and levels:**

  | | T | Arms (α) |
  |---|---|---|
  | Primary | 1 | B-oracle (0), B-a0.1-T1-n0 and -n1 (0.1), B-a0.3-T1-n0 and -n1 (0.3), B-marginal (0.5) |
  | Secondary | 0.25 | B-oracle (0), B-a0.1-T0.25-n0 and -n1 (0.1), B-a0.3-T0.25-n0 and -n1 (0.3) |

  B-oracle's rows come from val-b; the other arms come from `dose-val-b`.
- **Statistic:** the OLS slope of validation T5 on α, one point per arm.
- **Interval:** 95% class-stratified percentile bootstrap, 10,000 resamples, seed `stream_seed(269, "val", "tim-dose-response")`. Each resample draws patients once within each class and applies the draw to every arm. Primary and secondary each start from that seed.
- **Reading:**
  - decreasing if the interval's upper bound is below 0;
  - increasing if its lower bound is above 0;
  - flat otherwise.

  A difference smaller than 1e-9 T5 points counts as zero (floating-point residue).
- **Descriptive:** whether the per-level mean T5 never rises as α grows.

## Reporting once it has run

Fixed now, before any run:

| Oracle contrast verdict (A4, D4) | Primary dose-response reading | DIAGNOSTICS.md headline |
|---|---|---|
| planner bottleneck | flat or increasing | "Planner bottleneck." |
| signal bottleneck | decreasing | "Signal bottleneck." |
| inconclusive | any | "Inconclusive." |
| any other combination | | "Inconclusive: the oracle contrast and the dose-response disagree." |

- The word "Provisional" is dropped once the dose-response has run.
- The paragraph still states that the oracle-contrast rule and its threshold were written after the validation point estimates were known.
- The Pending section is replaced by the dose-response results.

## Unchanged

- Launching `dose-val-b` needs Daniel's committed approval, which this amendment does not supply.
- Everything stays exploratory and validation-only.
