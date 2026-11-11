# Amendment A5: provisional verdict, two exploratory additions, and the dose-response stage

Date: 2026-11-11 (US Eastern). Written and committed before the `diagnose-val-b` re-run that computes check H.

Scope:
- No GPU job runs under this amendment.
- No test admission is read.
- Everything here is exploratory and on validation only, as A4 is.

## 1. Verdict wording

A4's rule (D4) is unchanged. Until the D3 dose-response has run:
- DIAGNOSTICS.md heads its verdict "Provisional:".
- Its first paragraph states three things:
  - the verdict rests on one validation contrast;
  - the dose-response has not run;
  - the rule and its threshold were written after the validation point estimates were known.
- The not-blind note in its Scope section stays.

## 2. Sensitivity S1: urinalysis and radiographs counted as relevant

- **Status:** exploratory, defined after D2's counts were seen.
- **Primary definition:** A4's guideline map stays primary, because it was fixed in advance. It counts urinalysis (UA) and radiograph (XR) as relevant to none of the four diseases.
- **What S1 changes:** it counts UA and XR as relevant to all four.
- **How it is computed:** the primary map gives these two tests no relevance. So S1's relevant counts are the primary counts plus UA's and XR's published per-test counts (extra, extra provided, dropped, dropped provided). This uses the committed bundle without reading any row. The renderer refuses if the published map ever marks UA or XR as relevant.
- **Where it is reported:** under its own heading, and named in the limitations.

## 3. Check H: do the added results repeat what the policy already knew?

### The hypothesis and the check

The hypothesis is untested: the results the reranker added repeat what the policy already knew.

Check H asks about the extra tests that were both relevant and provided (A4, D2). For each such test, it looks at two outputs of the hypothesis agent:
- **before:** the hypothesis held when the test was requested;
- **after:** the hypothesis generated once its result came back.

For comparison, the same is done for P0's dropped tests that were relevant and provided, read from P0's own rows.

### Alignment

- Upstream generates a hypothesis at the start of every step but stores none on the step after an unavailable result.
- So the first stored hypothesis precedes the first request, and each provided request is followed by exactly one more stored hypothesis.
- A row is aligned when it ended in a diagnosis turn and stores exactly 1 + (number of provided requests) hypotheses.
- Tests in rows that are not aligned are excluded and counted.

### Counts, per val-b arm (extra) and for P0 (dropped)

| Count | Meaning |
|---|---|
| tests | tests analyzed |
| unaligned | tests excluded because their row is not aligned |
| before_correct | before = the label: the policy already held the right answer |
| before_other | before = another disease |
| before_unknown | before = unknown |
| changed | after ≠ before |
| changed_to_correct | changed to the label |
| changed_from_correct | changed away from the label |

Per arm and for P0, the number of rows and of aligned rows is also reported.

### Reading and safety

- **Reading, descriptive only:** if most added relevant results arrived while the policy already held the label, and rarely changed its hypothesis, that is consistent with the hypothesis. It does not test it.
- **Safety:** stored hypotheses are one of the four disease names or "unknown", so only counts are published.
- **Where it is published:** a separate aggregate file, `diagnose-val-b/hypothesis_check.json`.
- **Integrity check:** the re-run recomputes `diagnostics.json` with unchanged code, and the result must be byte-identical to the committed bundle.

## 4. Dose-response stage: prepared, not run

- **What is added:** stage `dose-val-b`, the 9 validation arms of A4's D3 (B-marginal and the eight corruption arms), in `configs/variants.yaml`. An approval request is drafted for it.
- **What is not done:** nothing is submitted. Launching needs Daniel's approval.
- **The analysis is A4's D3,** as written there.
- **Dry runs:** `scripts/submit_slurm.py --dry-run` prints a stage's sbatch commands without an approval, so the jobs can be reviewed before one exists. A real submission still requires the committed approval.
