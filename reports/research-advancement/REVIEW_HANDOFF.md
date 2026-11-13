# REVIEW_HANDOFF.md
## Independent research and implementation review — Testing in Imagination

**Project:** `scalinity/testing-in-imagination`  
**Owner:** Daniel Escalante · **Advisor:** Professor Xuefeng Liu · **Course:** CAI6734, University of Florida  
**Review reference date:** November 11, 2026, America/New_York. Some GitHub timestamps are November 12 UTC for November 11 evening activity in Florida.  
**Method:** Read-only examination of supplied instructions, pinned repository code, PR discussions, public aggregate results, and primary research sources. This is an internal scientific review, not a conference decision or an independent experimental replication.

**Execution boundary:** No repository files were changed. No tests, benchmarks, training, inference, or cluster jobs were executed. No protected patient-level records or private fitted models were accessed. The sole newly authored artifact is this review, outside the repository. Proposed verifications and experiments below have **not** been performed.

## A. Executive verdict

### Specification and kickoff: READY WITH TARGETED REVISIONS

The new specification has the right strategic direction: repair measurement, identify the bottleneck, separate acquisition from interpretation, build one bounded improvement, and establish independent evidence. It already addresses many of the earlier concerns correctly. It should not be replaced by a larger-model project or an infrastructure rewrite.

Before handing it to an autonomous implementation agent, make the targeted revisions in Section F. The most important additions are:

1. Bring **already-confirmed synthetic split and cost defects** into the first integrity milestone rather than deferring synthetic scrutiny until the final robustness milestone.
2. Specify a **component-by-component control contract**, including availability, belief updating, final diagnosis, stopping, and information access—not merely a selector-by-reader table.
3. Bind scientific gates and fitted artifacts to **execution and resumption**, not just to a displayed report or code commit.
4. Resolve the M3/M4 dependency on an improved reader, distinguish the new study from the frozen content study, and make current authorization precedence explicit.

These are targeted revisions because the plan's core questions, architecture, privacy boundaries, strong-baseline emphasis, and teaching deliverables are appropriate.

### Existing implementation: NOT READY for credentialed content-planning evaluation or publication-level content-model claims

The content branch is a functioning CPU research prototype with reported runs, not a validated planning improvement. Its claimed model-quality pass is unsupported by the implemented comparison: learned joint-panel loss is compared with a sum of marginal item losses, and the reference distribution is not the stated pooled charted-data marginal. Content GPU execution also lacks the specified CPU preflight. These are concrete code findings, not objections to hypothetical future performance. See REVIEW-001 and REVIEW-003.

The synthetic study also needs repair before its predictive-quality and model-comparison claims can support the publication. Its “held-out” likelihood sample reproduces underlying training cases; its acquisition score misprices missing tests; and its model-free comparator is a narrow, model-dependent one-step baseline. These findings do **not** establish that every synthetic result is invalid: the separate main utility evaluation uses an offset seed, and its scope must be assessed separately.

### Completed clinical work: retain it, but narrow the interpretation

The Phase A fallback improvement and completed Phase B/dose-response experiments are useful evidence. This review found no reason to discard the replication or restart the completed studies. Their recorded execution is not independently reproduced here. The null availability results support a bounded statement about this fixed planner and cohort—not the universal conclusion that a better availability predictor cannot help.

**Most consequential issue:** the project currently lacks a trustworthy bridge from “better predictions or more confidence” to “better decisions.” Repair the measurement and isolate that bridge before allocating substantial additional training compute.

No verified system-wide P0 incident or unauthorized execution was established. The P1 findings below nevertheless block the affected scientific claims and the next content GPU study until repaired or explicitly scoped.

## B. Verified project status

### B1. Reviewed revisions and scope of visibility

The supplied advancement documents were read completely. They are conversation artifacts; they were not established as committed repository files.

| Input | Reviewed identity |
|---|---|
| `SPEC_RESEARCH_ADVANCEMENT.md` | SHA-256 `242277bec2f022e9b360604580864925a5ea380b02d0374bcdeb44b1da025a5f`; 381 lines |
| `KICKOFF_RESEARCH_ADVANCEMENT.txt` | SHA-256 `fa727a32c056a9e5601edb37d823e690f62695945adb88e88daecf60b63e7d7c`; 72 lines |
| Remote `main` | `e5ad93fad5345f211cedc05c3dc491b101c63fa5` |
| PR #9 / `feat/content-wm-m1` | `acf5254bb624ed881f864e84149fc4d983dc048a` |
| PR #7 / `feat/content-wm-spec` | `75c50a725655ad4cbe69d4ab79677c40ed4b309e` |
| PR #1 / `edwin-salcedo/synthetic-world-model` | `df467eadd07fe2bc761b618dafce8ed9f198641a` |
| PR #8 | Head `769858aa58c35e634c4aa7f0b8efc631e0f5eef5`; merged as `3a564ddee135277e7adf2caebfa560a4409e76f3` |
| PR #10 | Head `b1e93ac23e28800a968296f537285e7708f098e7`; merged as the reviewed `main` SHA |

The fetched remote branch list and main tree did not show execution artifacts for the new `reports/research-advancement/` workflow. Therefore its M0–M5 milestones are **NOT STARTED in the inspected remote evidence**, not FAILED. Unpushed work, an inaccessible local working tree, or private execution may exist; this review does not certify their absence. An open PR's GitHub merge-preview SHA is not evidence that it was merged.

### B2. Recorded results and actual status

| Area | Verified record and interpretation | Status |
|---|---|---|
| Original replication | Seed 269; documented backbone, summarizer, runtime, and decoding deviations; completed run graph `7fcd9e5`. [R0] | REPORTED COMPLETE; not rerun here |
| Phase A | Test T5: original 73.09%, V1 84.45%, V3 81.83%. V1−original +11.37 points [7.28, 15.96]; V3−V1 −2.62 [−6.16, 0.76], Holm p=0.1875. [R1] | Completed recorded follow-up on a previously examined benchmark |
| Phase B validation | Oracle−uniform −0.11 points [−2.56, 2.38], Holm p=0.9736; unavailable requests 203→66. [R2] | Exploratory null accuracy contrast with a behavioral change |
| PR #8 diagnostics | Merged; A4/A5 analyses and check H; recorded reproducibility checks. An earlier missing-bundle review finding was subsequently addressed in the published record. [P8] | Completed recorded diagnostics; not a new open rendering defect |
| PR #10 dose response | Merged; nine arms reportedly completed all 240 validation admissions; 4.384 GPU-hours. Primary slope −2.46 [−7.28, 2.26] T5 points per unit corruption. [P10, R3] | Completed recorded experiment; no detectable slope, not equivalence |
| PR #9 CPU content work | Open/unmerged. Premise 68/240 admissions, 158 lab requests. Current aggregate reports quality skill 0.04284 and gain-decile Spearman ≈1; validity limitations below. [P9, R4] | Implemented and evaluated on CPU; scientific quality acceptance unresolved |
| C1 correction | Prior NLL 0.68999/ECE 0.04554; uncorrected all-lab NLL 2.48502/ECE 0.29465; A+B NLL 0.63720/ECE 0.07046, tau=13 in all folds. [R4] | Promising development result; not independent evaluation or end-to-end improvement |
| Content GPU stages | `bench-c` and `val-c` requests exist, with no approved runs reported; `test-c` is not configured. [P9, C5] | PLANNED, not successful experiments |
| PR #7 design | Open/unmerged. Documents a lab-only study and additional future requirements. Some “no code/no runs” wording is stale relative to PR #9. [P7, D1] | Design, not a completion certificate |
| Edwin's synthetic study | Open PR #1; code and aggregate summaries present; main utility and predictive-quality evidence have different limitations. [P1, S1–S6] | Existing prototype; repairs and stronger controls required |

Do not mix test T5 in Phase A with validation T5 in Phase B. Do not describe V3 as significantly worse than V1. Conversely, an improvement over 73.09% alone is no longer sufficient to establish the value of a complex successor over the strongest simple system.

### B3. Source map and attribution

The stable links below are the evidence anchors used throughout this report. Code locations in findings identify symbols or sections at these pinned revisions; they do not depend on this chat's tool history.

**Repository and recorded results**

- [R0] [Replication claims ledger at main](https://github.com/scalinity/testing-in-imagination/blob/e5ad93fad5345f211cedc05c3dc491b101c63fa5/REPRODUCTION.md).
- [R1] [Phase A report: ladder, contrasts, limitations](https://github.com/scalinity/testing-in-imagination/blob/e5ad93fad5345f211cedc05c3dc491b101c63fa5/cdm-extension/daniel-escalante/results/credentialed/phase-a/REPORT.md).
- [R2] [Phase B validation diagnostics](https://github.com/scalinity/testing-in-imagination/blob/e5ad93fad5345f211cedc05c3dc491b101c63fa5/cdm-extension/daniel-escalante/results/credentialed/DIAGNOSTICS.md).
- [R3] [Dose-response aggregate](https://github.com/scalinity/testing-in-imagination/blob/e5ad93fad5345f211cedc05c3dc491b101c63fa5/cdm-extension/daniel-escalante/results/credentialed/analyze-dose-val-b/dose_response.json).
- [R4] [Content CPU aggregate, including C1](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/results/credentialed/wm-fit-c/content_wm.json).
- [P1] [PR #1 and discussion](https://github.com/scalinity/testing-in-imagination/pull/1); [P7] [PR #7](https://github.com/scalinity/testing-in-imagination/pull/7); [P8] [PR #8](https://github.com/scalinity/testing-in-imagination/pull/8); [P9] [PR #9](https://github.com/scalinity/testing-in-imagination/pull/9); [P10] [PR #10](https://github.com/scalinity/testing-in-imagination/pull/10).

**Clinical implementation and instructions**

- [C1] [`tim/content.py`](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/tim/content.py).
- [C2] [`tim/labs.py`](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/tim/labs.py).
- [C3] [`tim/runner.py`](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/tim/runner.py).
- [C4] [`tim/approvals.py`](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/tim/approvals.py).
- [C5] [`tim/cli.py`](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/tim/cli.py).
- [C6] [`tim/access.py`](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/tim/access.py).
- [C7] [`tests/test_content.py`](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/tests/test_content.py).
- [C8] [`tim/analysis.py`](https://github.com/scalinity/testing-in-imagination/blob/e5ad93fad5345f211cedc05c3dc491b101c63fa5/cdm-extension/daniel-escalante/tim/analysis.py).
- [D1] [Original content-model specification](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/SPEC_CONTENT_WM.md).
- [D2] [Decisions, including authorization history and C1 semantics](https://github.com/scalinity/testing-in-imagination/blob/acf5254bb624ed881f864e84149fc4d983dc048a/cdm-extension/daniel-escalante/DECISIONS.md).
- [D3] [Standing onboarding instructions](https://github.com/scalinity/testing-in-imagination/blob/e5ad93fad5345f211cedc05c3dc491b101c63fa5/onboarding/COMPREHENSIVE_ONBOARDING_PROMPT.md).
- [D4] [Week 11 report source](https://github.com/scalinity/testing-in-imagination/blob/e5ad93fad5345f211cedc05c3dc491b101c63fa5/reports/week11_individual_progress_Daniel_Escalante.tex).

**Synthetic implementation**

- [S1] [`env.py`](https://github.com/scalinity/testing-in-imagination/blob/df467eadd07fe2bc761b618dafce8ed9f198641a/extension/envs/synthetic_diagnosis/env.py).
- [S2] [`collect.py`](https://github.com/scalinity/testing-in-imagination/blob/df467eadd07fe2bc761b618dafce8ed9f198641a/extension/envs/synthetic_diagnosis/collect.py).
- [S3] [`run_matrix.py`](https://github.com/scalinity/testing-in-imagination/blob/df467eadd07fe2bc761b618dafce8ed9f198641a/extension/experiments/run_matrix.py).
- [S4] [`planners/agents.py`](https://github.com/scalinity/testing-in-imagination/blob/df467eadd07fe2bc761b618dafce8ed9f198641a/extension/planners/agents.py).
- [S5] [`results/synthetic/SUMMARY.md`](https://github.com/scalinity/testing-in-imagination/blob/df467eadd07fe2bc761b618dafce8ed9f198641a/results/synthetic/SUMMARY.md).
- [S6] [`matrix.json`, including recorded source SHA](https://github.com/scalinity/testing-in-imagination/blob/df467eadd07fe2bc761b618dafce8ed9f198641a/results/synthetic/matrix.json).

PR #1's synthetic implementation is Edwin's recorded contribution. Its split/cost/baseline concerns were already raised under Daniel's `scalinity` account; this review does not claim to have discovered them first. PR #9's likelihood, marginal, preflight, and thought findings originated in automated Codex reviews and were checked against the inspected implementation. The supplied advancement spec and kickoff were assistant-generated planning artifacts. No additional contribution is assigned to Hugo without a supporting record; that is a limit of attribution, not a claim that he did nothing.

## C. Highest-impact findings

| ID | Severity | Finding | Required disposition |
|---|---|---|---|
| REVIEW-001 | P1 | Content quality compares different likelihood estimands and uses the wrong stated marginal reference. | Repair before treating the CPU quality gate as passed. |
| REVIEW-002 | P1 | Synthetic “held-out” likelihood reuses underlying training cases. | Repair split generation before predictive-quality/error–gain claims. |
| REVIEW-003 | P1 | Content GPU preflight is unenforced; fitted files are not verified against an immutable run identity. | Enforce scientific gates and artifact identity at launch and resume. |
| REVIEW-004 | P1 | Oracle/prior controls change more components than their interpretation acknowledges. | Add explicit intervention contracts; do not infer pure content effects. |
| REVIEW-005 | P1 | Tempered updates break the claimed EIG identity; the calibration gate checks self-consistency, not diagnostic benefit. | Name the surrogate honestly and validate decision relevance, or restore consistency. |
| REVIEW-006 | P1 | Synthetic planner charges full cost despite free unavailable results. | Repair expected-cost accounting and reassess rankings. |
| REVIEW-007 | P2 | Synthetic model-free baseline has restricted state and model-dependent one-step targets. | Preserve it as a named weak baseline; add a matched-information comparator. |
| REVIEW-008 | P2 | Replaced lab actions can retain reasoning about the original action. | Repair serialized trajectories before clinical planning runs. |
| REVIEW-009 | P2 | C1 is out of fold only for likelihood-temperature selection, not the full tuned pipeline. | Keep developmental labeling; use independent/nested evaluation for generalization. |
| REVIEW-010 | P2 | The availability-null wording exceeds what its interval and post-hoc decision rule establish. | Add a scoped interpretation/erratum without changing historical numbers. |
| REVIEW-011 | P2 | Synthetic output's recorded Git SHA does not identify the code introduced by the PR. | Preserve historical output; regenerate with complete source provenance. |
| REVIEW-012 | P2 | The new plan lacks an explicit reader-probe dependency and successor-study execution freeze. | Clarify M2→M3→M4 and prevent protocol/result identity mixing. |
| REVIEW-013 | P2 | Historical approval delegation and obsolete milestone instructions need explicit precedence. | Keep current restrictions controlling; do not infer fresh spending permission. |
| REVIEW-014 | P3 | Kickoff reporting triggers can multiply immutable reports and PDFs for every finding. | Consolidate evidence once; refresh human-facing views at meaningful checkpoints. |

P1 here means a material validity/interpretation issue in the affected workflow. It does not mean all previous project results are disproven. Several findings are already anticipated in the new spec; retain those requirements rather than inventing a second redesign.

## D. Detailed findings

### REVIEW-001 — The content quality pass is not established

**Severity/type/confidence:** P1; confirmed mathematical/implementation defect; high confidence.  
**Location:** [C1] `fit_model()` and `quality()`; [C7] `lab_examples()` and the quality test; [R4] `quality`. Relevant automated reviews: [per-item scoring](https://github.com/scalinity/testing-in-imagination/pull/9#discussion_r4544469960) and [pooled marginal](https://github.com/scalinity/testing-in-imagination/pull/9#discussion_r4545434754).

**Evidence/problem:** Learned NLL accumulates `-logsumexp(log(belief) + log_lik(panel,outcome))`, a joint panel loss. Reference NLL accumulates one log loss per item. Both are divided by item count. The reference also uses `model.prior @ table`, where `prior` counts all admissions while each conditional table is estimated only from charted panels.

**Analytical counterexample, not an executed test:** Let two diseases be equally likely and two items independently have probability 0.9 of outcome 1 in one disease and 0.1 in the other. With belief equal to the prior, each item's learned and reference prediction is 0.5. For observed `(1,1)`, joint probability is 0.41. The implemented learned loss per item is `-log(0.41)/2`, versus `-log(0.5)` for the reference: approximately 0.446 versus 0.693, suggesting roughly 0.357 skill despite identical per-item predictions. This is not a claim that a joint model can never improve joint likelihood; it proves that the comparison does not measure the specified per-item skill.

Separately, equal population class proportions do not imply equal class proportions among charted panels. A panel charted 100 times in one class and once in each other class does not have a marginal given by equal class weights. Smoothing also needs to be defined consistently.

**Impact:** The reported positive NLL skill cannot establish Gate 2. The reference issue also affects Brier skill. Existing single-item quality fixtures cannot expose the joint-versus-item defect.

**Change:** Use matched per-item targets, conditioning, smoothing, and weighting. Fit a separately pooled training-only reference for the declared charted-data estimand. Keep joint scoring as a separately named diagnostic if needed. Preserve old results and publish a versioned correction.

**Verification:** Hand-computed multi-item examples; identical distributions yield zero skill; strongly imbalanced availability yields the correct reference; absent/unseen/rare outcomes stay well-defined. Regenerate authorized CPU aggregates. Do not forecast whether the corrected gate will pass.

### REVIEW-002 — Synthetic likelihood evaluation is not held out

**Severity/type/confidence:** P1; confirmed split-construction defect; high confidence.  
**Location:** [S1] `SyntheticCDMEnv.__init__`; [S2] `collect_rows`; [S3] `run_setting`; [S5] §3b–c. Previously identified by Daniel in [the split review](https://github.com/scalinity/testing-in-imagination/pull/1#discussion_r4419554073) and [likelihood review](https://github.com/scalinity/testing-in-imagination/pull/1#discussion_r4419554077).

**Evidence/problem:** Environment RNG initialization ignores `split`. `collect_rows` also uses the same behavior-policy seed for the same numerical seed. `run_setting` collects train and 500 validation episodes with the same seed, changing only `split`. Underlying patient draws, findings, and behavior choices therefore repeat the beginning of the training stream. Textual IDs include split labels, so this is not a claim of byte-identical JSON records.

**Impact:** “Held-out” likelihood and the claimed model-error/performance crossover are not independent predictive evidence. The default main utility evaluation uses `1000 + seed`; the demonstrated construction does **not** establish that those utility episodes overlap training. Do not throw out that table for the wrong reason.

**Change:** Namespace patient-generation and behavior RNG streams by purpose/split, while retaining matched evaluation cases across competing arms. Include a stable generated-case fingerprint independent of display IDs.

**Verification:** Train/validation/test generator streams differ under the same user seed; arms evaluated on the same test realization receive identical latent cases; fresh predictive likelihood is evaluated only on unseen cases. Regenerate the affected likelihood/error–gain outputs after approval for implementation, not during this review.

A learned model beating an oracle on a finite sample is not itself proof of leakage. The seed construction is the evidence here.

### REVIEW-003 — Content gates and fitted-artifact identity do not govern execution

**Severity/type/confidence:** P1; confirmed missing checks, with an unobserved mixed-artifact failure scenario; high confidence in code, unknown realized impact.  
**Location:** [C4] `check`; [C5] `run_command`, `wm_fit_c`; [C3] `_content_planner`, `wm_dir`, `run_job`; [C6] `run_key`/`Episodes`. Automated [preflight finding](https://github.com/scalinity/testing-in-imagination/pull/9#discussion_r4544469974).

**Evidence/problem:** GPU command execution calls `approvals.check()`, but its scientific validation-stop check applies only to `test-a`/`test-a2`. Content planner loading reads mutable `private/wm/content.pkl` and `learned-s0.pkl` without checking the CPU report's digest against those bytes. The CPU stage can replace `content.pkl`; episode identity does not directly bind that fitted content artifact.

**Impact:** A valid compute approval is not proof of passed content gates. Unchanged code could load a different model from the one evaluated or used before interruption. No actual unauthorized content run or mixed-model trajectory was established; current content GPU stages are reported unapproved/unrun.

**Change:** Add one shared content preflight for submit/launch, plus an immutable execution manifest checked before initial evaluation and resume. Bind content model, prior/bin representation, availability model, policy, study/protocol/config, and the compatible CPU-quality report. Prefer content-addressed artifacts or equivalent immutable run-local references.

**Verification:** Missing/failed/nonfinite/stale CPU evidence is rejected despite a valid compute approval. Replacing a fitted file without changing code is rejected. An unchanged interrupted run resumes; a different model cannot continue the same run.

Do not “fix” a test mismatch by silently generating a new admission key and rerunning completed test cases. Refuse and require the separately authorized study decision.

### REVIEW-004 — The oracle and population controls do not isolate content quality

**Severity/type/confidence:** P1; confirmed control/interpretation confounding; high confidence.  
**Location:** [D1] §§1.6, 3; [D2] content availability statements; [C1] `PatientPlanner.belief/score`; [S3] `run_setting`; new spec §8. Automated [availability-control finding](https://github.com/scalinity/testing-in-imagination/pull/7#discussion_r4543584403).

**Evidence/problem:** Learned clinical scores multiply by learned availability. The clinical oracle instead receives true availability and true outcomes. This contradicts the statement that all content arms differ only in predicted content. `WM-prior` resets the evaluation belief as well as the predictive mixture. In the synthetic matrix, the oracle changes both selection model and diagnosis belief model, while the corrupted arm changes selection with the learned diagnosis model retained.

**Impact:** These are interpretable system-level controls, but they cannot isolate pure content prediction effects. A true-outcome fixed-updater selector is not an optimal diagnostic policy or a guaranteed task-performance upper bound.

**Change:** For each arm freeze predictor, belief updater, final reader, availability information, eligible actions, stopping, representation, and budgets. Name current oracle controls jointly privileged. Use a content-only comparison with availability held fixed where the outcome support allows it. In chart-limited clinical data, treat oracle comparisons as restricted to observed support; use the synthetic environment for clean fully counterfactual separation.

**Verification:** Holding all declared fixed components constant changes only the intended component. Population-predictor-only and population-belief-policy controls are separately identified. Report paired contrasts and interaction terms for genuinely crossed experiments.

Do not retroactively reinterpret the original frozen oracle as a different arm. Introduce new controls prospectively.

### REVIEW-005 — The current gain score is an acknowledged surrogate, not established mutual information

**Severity/type/confidence:** P1 for the scientific interpretation; confirmed identity mismatch, unproven downstream harm; high confidence.  
**Location:** [C1] `log_lik`, `sample`, `posterior`, `eig`, `quality`; [D2] final C1 paragraph; [R4] gain deciles; new spec §7.3.

**Evidence/problem:** Hypothetical outcomes are sampled from the original class-conditional likelihood. Diagnosis updates may drop absent factors and divide evidence by tau. The update is therefore not generally the Bayes conditional of that same predictive joint distribution. The decision log explicitly acknowledges this; it is not an undisclosed discovery.

For coherent Bayes prediction, expected posterior belief equals the prior, and expected entropy reduction is mutual information. That identity is not supplied by an arbitrary normalized tempered update. “There is no stopping rule” does not make the distinction immaterial: rankings are the planner's main output.

The calibration check correlates ten bin-mean predicted gains with entropy changes produced by the same updater. A confidently wrong update can reduce entropy. High correlation therefore establishes neither probability calibration nor diagnostic usefulness. Negative estimated gain alone does not prove the mechanism: finite Monte Carlo noise and inconsistent updates must be distinguished.

**Change:** Either implement a justified coherent joint/update pair or retain and explicitly name the expected entropy-change surrogate. Add truth-referenced proper-score changes, action-ranking quality, and decision regret on known synthetic problems. Do not silently clip negative scores.

**Verification:** Exact enumeration in a tiny model distinguishes Monte Carlo error from the changed update semantics. A confidently-wrong example is not declared useful solely because entropy fell. Evaluate whether score rankings predict improvements for the actual diagnosis reader.

The new spec already anticipates this issue well; strengthen its acceptance check rather than mandating a wholesale architectural replacement.

### REVIEW-006 — Synthetic missing-test costs are systematically overstated

**Severity/type/confidence:** P1; confirmed objective/environment mismatch; high confidence.  
**Location:** [S4] `ImaginationSelector.value`; [S1] unavailable-test branch; [S3] lambda sweep. Previously raised by Daniel in [the cost review](https://github.com/scalinity/testing-in-imagination/pull/1#discussion_r4419554083).

**Evidence/problem:** The selector uses `p * IG - lambda * cost`, but unavailable tests incur no provided-test cost. Under the current diagnosis-independent missingness model and the stated one-step information utility, expected monetary penalty is `lambda * p * cost`.

**Impact:** Rarely available tests are overpenalized by `lambda * (1-p) * cost`. This can change choices, stopping, utility rankings, and the apparent advantage over a comparator whose observed targets already include free misses. It is not merely a display discrepancy.

**Change:** Correct expected cost under the existing utility, preserving a legacy arm if historical comparison is needed. Model the consumed step or a request penalty separately when the experiment calls for it; neither silently charge fictitious dollars nor treat free requests as consuming no horizon.

**Verification:** Enumerated available/unavailable outcomes match the expected score for p=0, p=1, and intermediate p. Test action ranking and stopping. Reassess the missingness and cost-weight curves; do not assume the old ranking survives.

### REVIEW-007 — The synthetic comparator cannot support a broad “planning beats model-free” claim

**Severity/type/confidence:** P2; confirmed limitation of the comparison; high confidence.  
**Location:** [S4] `ModelFreeSelector.fit/select`; [S3] model-free construction; [S5] headline findings; Daniel's [state critique](https://github.com/scalinity/testing-in-imagination/pull/1#discussion_r4419554085) and [target critique](https://github.com/scalinity/testing-in-imagination/pull/1#discussion_r4419554088).

**Evidence/problem:** The fitted action-value table observes a top diagnosis and confidence bin, not the full belief/history. Its training target is a one-step change in correct-label log probability under the learned belief model, minus observed cost. It is not a full-information terminal-return model-free learner.

**Impact:** The comparison mixes acquisition method with state compression, target choice, and dependence on the shared fitted model. The existing baseline remains useful, but the conclusion should be about this particular coarse one-step comparator.

**Change:** Preserve it with an accurate name. Add one matched-information non-planning value/policy baseline using the belief vector and acquisition mask or the same permitted history encoding, trained on appropriate observed task-return targets. For a pure selector comparison, keep updater and stopping fixed.

**Verification:** Document equality of training episodes, observable inputs, budgets, and final reader. Publish paired seed-level differences, not a claim of significance from a gap measured in unpaired seed standard deviations. Separate method gains from representational advantages.

### REVIEW-008 — Action replacement can preserve inconsistent reasoning

**Severity/type/confidence:** P2; confirmed trajectory-integrity defect; high confidence.  
**Location:** [C3] `_plan_content`; [D1] §2.2; automated [thought-replacement finding](https://github.com/scalinity/testing-in-imagination/pull/9#discussion_r4544469966).

**Evidence/problem:** `_plan_content` passes `proposal.thought` to `fit()` after selecting a different action. The original thought can remain when it fits the budget, contrary to the content specification's fixed neutral replacement thought.

**Impact:** Later policy context may contain reasoning about a different test. This confounds acquisition effects with inconsistent text history.

**Change:** Pass the required neutral thought explicitly for content replacements, or use an interface that cannot retain the mismatched proposal thought.

**Verification:** Inspect the complete serialized text, token mask, completion IDs, and subsequent prompt on a synthetic replacement case. Also preserve the no-replacement/no-planner identity. A unit test checking only the selected action is insufficient.

### REVIEW-009 — C1's out-of-fold label does not cover the complete tuning procedure

**Severity/type/confidence:** P2; confirmed evaluation-scope limitation, not a prohibition on validation tuning; high confidence.  
**Location:** [C1] `fit_prior`, `cv_tau`, `compare_fixes`, `choose_fix`; [C5] `wm_fit_c`; [R4]; new spec §7.4.

**Evidence/problem:** The HPI prior temperature is selected using all validation labels before likelihood-temperature cross-validation. C1 then selects among corrective candidates using development scores. Only the tau selection for A+B is out of fold in the described calculation.

**Impact:** The 0.637 result is useful developmental evidence, but not an independently evaluated full-pipeline generalization estimate. The scalar temperature selected in all five folds is evidence of selection stability on these cases, not independent replication.

**Change:** Keep the historical result, state exactly what was cross-validated, and use an outer evaluation split/cross-fitting procedure that isolates all applicable tuning for the successor estimate. Prefer train-internal model selection for the next validation comparison when feasible.

**Verification:** Every scored outer fold is excluded from all relevant tuning; prefixes from the same admission remain together. Report paired admission-level uncertainty and the separate prior/full-posterior endpoints. C1 improves NLL relative to the HPI prior, but its ECE of about 0.0705 remains above the prior's 0.0455; do not claim uniform improvement across metrics.

### REVIEW-010 — The null availability conclusion is too strong

**Severity/type/confidence:** P2; confirmed inferential overstatement; high confidence.  
**Location:** [R2] opening verdict; [D1] motivation/oracle interpretation; [R3]; new spec §2 already provides a more careful framing.

**Evidence/problem:** The diagnostics combine a nonsignificant oracle contrast with a confidence upper bound below a 4.15-point MDE borrowed from another contrast, then state that a better availability predictor cannot help. The rule was written after validation point estimates were visible. The dose-response interval permits modest decreases over the tested range.

**Impact:** An MDE is not a prespecified practical-equivalence margin. Perfect availability supplied to a fixed heuristic does not establish monotonic optimality over all less-perfect scores or different planners. These experiments do not prove universal uselessness of availability prediction.

**Change:** Preserve the results and document a wording erratum: “This fixed planner showed no detectable T5 benefit from true availability over the uniform control on this validation cohort; modest effects and benefits from other planners remain unresolved.” Keep operational request reductions separately reported.

**Verification:** Every conclusion names planner, cohort, seed, estimand, and uncertainty. No test of equality or clinical equivalence is implied. Future practical-equivalence claims need a prospectively chosen margin and appropriate analysis.

### REVIEW-011 — Synthetic result provenance does not identify the executed source state

**Severity/type/confidence:** P2; confirmed provenance gap; high confidence in the recorded mismatch, no claim about the exact unseen working-tree contents.  
**Location:** [S6] `config.git_sha`; [S3] `_git_sha`; [P1] commit history and Daniel's [provenance comment](https://github.com/scalinity/testing-in-imagination/pull/1#discussion_r4419554121).

**Evidence/problem:** The aggregate records base SHA `22d8a3229a7babdd639052f4c453767694ac3444`, while the experiment implementation is introduced by the PR at `df467eadd07fe2bc761b618dafce8ed9f198641a`. The manifest does not bind the missing working-tree changes.

**Impact:** Checking out the recorded SHA alone does not identify the implementation that generated the result. This prevents a clean reproduction claim from the artifact's provenance alone.

**Change:** Retain the historical artifact as incompletely bound. Generate successor evidence from a clean, recorded revision, or bind an explicit source snapshot/patch digest when a dirty run is intentionally allowed. Record dependencies and configuration as the advancement spec already requires.

**Verification:** The manifest identifies all executable source and inputs needed to reconstruct the run. Reporting regeneration, analysis rerun, and fresh experimental execution must be separate statuses.

### REVIEW-012 — M3 needs a reader before M4, and the successor study needs its own freeze

**Severity/type/confidence:** P2; confirmed planning ambiguity; medium-high confidence about execution risk.  
**Location:** Supplied spec §§6.4, 8, 9 and kickoff “EXECUTION ORDER”; [D1] §§4–6.

**Evidence/problem:** M3's matrix requires an improved calibrated reader, but its implementation is primarily described in M4. The new study also proposes different readers/objectives and additional controls, while the old content study has fixed arms, a V3 comparator, gates, and a repair-run allowance. The spec says to preserve history, but does not provide a concrete successor-study boundary before these experiments.

**Impact:** An agent may stall on the dependency, prematurely optimize the integrated system, or place changed-method results into a legacy stage identity. A loose “go/no-go” without a declared operational threshold can also become post-result discretion.

**Change:** Add a bounded CPU reader-probe step at the end of M2, explicitly permit that dependency before M3, and freeze the next experiment card before execution. Use a new study identity, output namespace, and approvals for changed estimands/arms. Keep legacy-conformance repairs separate from method changes.

**Verification:** Each M3 cell can be constructed from already available components; the card names what changes and what does not; thresholds and maximum runs are stated before outcomes; no new method is reported as an unchanged legacy arm. A negative or blocked milestone is distinguished from empirical success.

### REVIEW-013 — Authorization and milestone precedence should be explicit

**Severity/type/confidence:** P2; documented instruction-integration risk, not an observed permission violation; high confidence in the conflicting records.  
**Location:** [D2] “Approvals in practice”; [D3] old phase/first-action instructions; supplied spec §3 and kickoff “AUTHORITY AND DATA.”

**Evidence/problem:** The decision log records an older authorization for an agent to commit approval files on Daniel's behalf. The new task forbids inserting Daniel's signature or self-approving a new run. Older onboarding also tells the agent to start early reproduction phases that are now complete.

**Impact:** The newer explicit restrictions should already govern, but the documents do not state how to treat the historical delegation. An autonomous reader should not have to infer that a broad older grant covers a newly designed experiment.

**Change:** State that current task-specific restrictions govern over historical delegations and stale action lists. Existing approvals may be reused only when their exact scope and bindings still apply. New compute authorization is not inferred from the November 9 delegation. Record a precedence decision; do not delete the historical record.

**Verification:** A new unapproved stage remains blocked even when the old delegation text is present. The agent resumes from verified current status rather than restarting completed archaeology/training. This review remains strictly read-only regardless of permissions in implementation documents.

### REVIEW-014 — Reporting cadence is unnecessarily duplicative

**Severity/type/confidence:** P3; optional workflow improvement; medium confidence in its practical cost.  
**Location:** Supplied spec §4, §11, §12; kickoff “REPORT WHILE YOU WORK.”

**Evidence/problem:** The spec requires several maintained views plus immutable milestone and LaTeX reports. The kickoff asks for an immutable report and provisional LaTeX report after every milestone, important finding, or blocker. This is broader than the milestone-oriented source requirement.

**Impact:** Repeated derivative reports can consume attention, drift numerically, or label the same work as several accomplishments. The professor needs one current argument and inspectable backing evidence, not a stack of near-duplicates.

**Change:** Keep a canonical result/claim record; update STATUS and the current brief promptly; create immutable milestone reports at milestone closure or a genuinely decision-changing event. Compile the professor-facing PDF at meeting/session checkpoints. Preserve the explanatory guide and attribution.

**Verification:** Every displayed result resolves to one experiment identity; the brief, talk track, and technical report agree. A routine status change does not trigger a new “completed milestone.”

### Supported risks that are not established empirical defects

**Cross-panel duplicate evidence.** [C1–C2] multiply panel-indexed factors without explicit observation-identity deduplication. Actual overlap frequency and its contribution to overconfidence were not measured here. Audit privately and export aggregates; do not attribute the entire C1 problem to double counting without evidence. Deduplicate the same observation, not genuinely repeated measurements.

**Conditional outcome support.** Availability is defined by charted items, while independent item sampling can assign probability to combinations outside the intended “given available” support. Check all-absent and overlapping-item consistency in an enumerable fixture. Quantify the mass and consequence before escalating this to an empirical performance claim.

**History-copying screen.** Exact item names near a number are a limited heuristic. No favorable subgroup-loss difference is not proof of no leakage; the loss itself also needs REVIEW-001's repair. Audit timing/provenance and observation access, not just lexical coincidences.

**Broad privileged container.** Normal clinical planners receive an object containing labels and all outcomes, although the inspected normal-arm scoring path filters to provided observations and uses labels only for the misleading control. No hidden-label use by the learned arm was demonstrated. Capability-separated inputs and metamorphic tests are worthwhile safeguards.

**Distribution mismatch.** Eight chart-derived orders and pooled item metrics need not represent on-policy decision states. Repeated items, orders, and prefixes are not independent admissions. Evaluate action rankings and truth-referenced updates on permitted policy-like histories as a separately labeled diagnostic.

### Areas that are sound or already well handled

The following should be retained, not reopened without new evidence:

- The clinical Monte Carlo implementation seeds one generator and advances it across draws. The PR #7 concern applies to ambiguous “per-draw” wording; the inspected implementation does **not** reseed every draw identically.
- Class-balanced T5 weights, patient alignment checks, paired comparison structure, and within-family Holm correction are explicit in [C8]. The Phase A report acknowledges the stronger exchangeability null of sign-flipping. No new error in those inspected routines was established.
- The dose-response analysis uses a shared admission draw across arms and fixes its code before the new runs. Post-hoc wording and earlier exploratory contrast selection remain separately disclosed.
- Private stage-specific data loading, locks, episode digests, and export-boundary design are substantive safeguards. This review is not a complete security or privacy certification.
- The new spec explicitly preserves V1 as a strong baseline, rejects test-based retuning, distinguishes an oracle from a deployable method, and accepts negative results.
- C1 moves posterior computation to log space and records the uncorrected comparison. Its scientific evaluation limitations do not make that numerical repair inappropriate.

## E. Milestone acceptance and specification review

**Important:** M0–M5 below refer to the new advancement spec, not the unrelated older milestone numbering in onboarding or “content milestone 1” in PR #9. No advancement milestone is accepted as completed solely because this review discusses similar work.

| Milestone | Observed execution status | Assessment against its stated plan | Prospective improvement |
|---|---|---|---|
| M0 — evidence/novelty | NOT STARTED in inspected remote evidence | Good scope, attribution, source pinning, and initial professor packet. This review is input to M0, not its implementation. | Add input hashes, instruction precedence, and a compact arm-component registry. Time-bound the literature check to decision-relevant neighbors. |
| M1 — integrity repairs | NOT STARTED; legacy defects verified | Matched loss, reference cases, preflight, and information-boundary tests directly address real problems. | Add synthetic split/cost repairs now; bind actual fitted files and resume identity; require clean/source-bound manifests. |
| M2 — headroom | NOT STARTED | Correctly separates eligible errors, representation risks, gain semantics, and tuning. | Count actual pre-divergence intervention opportunities under guard precedence and time limits, not merely “had a lab request.” Add a bounded reader probe needed by M3. |
| M3 — crossed experiments | NOT STARTED | Replay→fixed-component selection→closed loop is a strong design and should be kept. | Freeze availability, updater, reader, input support, stopping, and budget for each contrast; separate joint privileged controls and interactions. |
| M4 — targeted improvement | NOT STARTED | Bounded candidate count, partial-history training, and delayed expensive training are appropriate. | Define the diagnostic-risk target and horizon explicitly. Use a scalar/vectorized prototype first; do not smuggle stopping or imaging expansion into a selector comparison. |
| M5 — robustness/publication | NOT STARTED | Independent evaluation, seed distinctions, corruption families, and publication humility are appropriate. | Separate “mechanism demonstrated,” “method benefit demonstrated,” and “external evidence available.” Do not call an unavailable external cohort a completed empirical validation. |

**Existing content CPU milestone:** Accept the narrow fact that code and CPU aggregates exist. Do not accept the quality-gate conclusion or full original content-study completion. The broader [D1] requirements include a discriminative alternative, per-prefix posterior reporting, and a pairwise co-occurrence check; these are not established by the inspected milestone artifacts. A narrower PR title/body does not silently discharge those broader criteria. Record omissions as remaining original requirements or explicitly narrow them prospectively.

**Existing clinical phases:** Preserve the recorded completed statuses and their original frozen comparisons. This review's proposed successor controls are not retroactive acceptance conditions for Phase A or PR #10.

**Highest-leverage simplification:** Keep the world model, belief update, selector, and final reader as small interchangeable components with one experiment card per comparison. Do not build a new orchestration platform. A shared selector is useful only if the adapters also document availability, cost, stopping, and observation semantics; common function names alone do not ensure the same scientific method.

## F. Proposed specification and kickoff revisions

These are precise amendments for later integration, not changes made by this review. Preserve the current full documents and incorporate only adjudicated changes.

### F1. Essential — add a synthetic integrity prerequisite to M1

**Target:** Spec §6, before §6.4; kickoff “EXECUTION ORDER.”  
**Addresses:** REVIEW-002, REVIEW-006, REVIEW-011; avoids postponing known defects to M5.

**Suggested insertion:**

> Before using synthetic predictive quality or utility comparisons to select the successor method, repair the split-namespace defect in the PR #1 collector and the missing-test expected-cost mismatch. Preserve historical outputs. Add case-fingerprint checks demonstrating train/validation separation and matched evaluation realizations across arms. Regenerate affected results from a source-bound revision. The old main utility evaluation and the old predictive-quality evaluation must be adjudicated separately; do not declare both leaked merely because one is.

Add the matched-information model-free baseline to the experiment register before making a general planning-superiority claim; the small baseline need not block basic deterministic integrity tests.

### F2. Essential — bind preflight and resumption to immutable fitted artifacts

**Target:** Spec §6.2 and §12; kickoff verification instructions.  
**Addresses:** REVIEW-003.

**Suggested replacement/addition:**

> Content preflight must verify a compatible passing CPU report and the exact digests of the loaded content model, prior/bin representation, availability model, and policy, alongside the study/protocol/configuration identity. Store these in an immutable run manifest before the first episode and verify them before every resume. A changed model under unchanged code is a different experimental treatment. Refuse a mismatch; do not reopen completed test admissions by automatically changing a run key. Resume unchanged runs and request separate authorization for genuinely new test studies.

### F3. Essential — require an arm-component contract

**Target:** Spec §8, before the matrix; §10 synthetic controls.  
**Addresses:** REVIEW-004, REVIEW-007, and evidence-parity risks.

**Suggested insertion:**

> Each arm must declare: outcome predictor; belief updater; final diagnosis reader; availability information; observation representation; action-trigger/eligibility rules; stopping and guards; monetary and action budgets; seeds; and all privileged inputs. A contrast described as changing only selection must hold the remaining components fixed. Name the existing clinical control “true-outcome and true-availability, fixed-updater control,” not an optimal ceiling. Add separate content-only and joint privileged comparisons where identifiable. Where missing chart outcomes prevent content-only identification, say so and establish that separation in synthetic data. Population-predictor and population-belief controls must not be conflated.

**Reader comparison condition:** A structured reader that excludes imaging/exam content cannot be called a reader-only replacement for a reader receiving those observations. Match permissible evidence or explicitly label the representation-plus-reader treatment.

### F4. Essential — define gain semantics and the decision-value target

**Target:** Spec §§7.3 and 9 “Decision-aligned selection.”  
**Addresses:** REVIEW-005 and underspecified execution choices.

**Suggested insertion:**

> Every selector must name the quantity it estimates: coherent mutual information, expected entropy-change surrogate, one-step task-risk reduction, or a specified finite-horizon return. For a risk-based selector, freeze the loss, class weighting, final decision rule, unavailable-outcome treatment, expected charged cost, horizon, and stopping convention before evaluation. The estimate must target the reader actually used in that arm. Hidden labels may train or score permitted development estimators but cannot enter action selection. A surrogate is not promoted to a mutual-information or safety guarantee by normalization, positive rank correlation, or ensemble disagreement.

A useful initial implementation is the expected loss of the current reader's committed decision, compared with its expected loss after one acquisition, at a fixed stopping convention. This is a proposed prototype, not a mandatory new loss or a claim of originality. Multi-step return remains a separate experiment.

### F5. Essential — remove the reader dependency cycle and freeze successor experiments

**Target:** Spec §§7.4, 8, 9 “Bounded development”; kickoff “EXECUTION ORDER.”  
**Addresses:** REVIEW-012.

**Suggested insertion:**

> At the end of M2, implement at most one bounded CPU reader-probe candidate using the repaired evidence representation. This prototype supplies M3's alternative-reader cell; it is not yet the integrated M4 method. M3 determines whether to retain it. M4 then integrates only the component changes supported by the crossed experiment. Before any successor run, commit an experiment card naming the study ID, versioned outputs, component contract, primary estimand, decision threshold or explicitly descriptive status, allowed candidate set, maximum runs, and applicable approval. Legacy conformance fixes and new research interventions must use distinct identities; the old four-repair allowance does not authorize arbitrary new arms.

The milestone can end with a negative result or an authorization blocker, but its status must distinguish engineering completion, experiment completion, and empirical outcome.

### F6. Essential — explicit authorization and instruction precedence

**Target:** Spec §3; kickoff “AUTHORITY AND DATA” and first-step instruction.  
**Addresses:** REVIEW-013 and stale onboarding.

**Suggested insertion:**

> Current task-specific restrictions govern over historical delegation and obsolete next-action lists. In particular, the older DECISIONS.md authorization to commit approval files on Daniel's behalf does not grant permission for new experiments under this task. Do not sign or manufacture approval. Existing approvals apply only when their scope and bindings still match; otherwise draft the exact request and continue independent permitted work. Retain historical instructions as records and document which provisions are superseded. Never restart a completed phase merely because old onboarding still calls it the next step.

For the present review, the stricter read-only instruction controls everything. These execution amendments are for a later explicitly authorized implementation session.

### F7. Essential before generalization claims — distinguish development, replay, and independent evaluation

**Target:** Spec §§7.4, 10, 12; kickoff evidence labels.  
**Addresses:** REVIEW-009 and provenance ambiguity.

**Suggested insertion:**

> Label whether a result is a fresh experimental execution, a rerun of analysis on fixed episodes, or rendering/reformatting of existing aggregates. Do not use “reproduced run” for all three. Record separately whether a number is in-sample, validation-tuned, partially out of fold, fully outer-fold evaluated, or independently held out. Group all derived observations from an admission together. A new freeze or split name cannot erase previous exposure to benchmark cases. Report missing subject identifiers as an unresolved grouping limitation, not evidence of zero person-level overlap.

The spec's existing independent-cohort requirement is sound; no additional institutional framework is necessary.

### F8. Optional — consolidate professor-facing reporting

**Target:** Spec §4/§11; kickoff “REPORT WHILE YOU WORK.”  
**Addresses:** REVIEW-014.

**Suggested replacement:**

> After each material finding or blocker, update the canonical claim/result ledger and STATUS. Refresh the current professor brief and plain-English guide when the scientific interpretation changes. Produce an immutable milestone report at milestone closure or a decision-changing event; compile the meeting-facing LaTeX/PDF at meeting or session checkpoints. All views must use the same experiment IDs and figures. Keep the brief about two pages, with the question, comparison, observation, uncertainty, and next decision visible on the first page.

### F9. Essential reporting correction; no numerical reanalysis implied

**Target:** Advancement spec §2/first M0 brief; an explicit historical interpretation erratum for [R2].  
**Addresses:** REVIEW-010.

Use the scoped wording in REVIEW-010. Preserve the original file/version as historical evidence; do not silently replace old conclusions while claiming the frozen report is unchanged. Do not use the 4.15-point MDE as an equivalence margin.

## G. Recommended next research experiments

The sequence below maximizes discriminating information before costly policy retraining. Compute descriptions are categories or historical proxies, not new measurements or approvals. All experiment execution is deferred to an authorized implementation session.

### E0 — Tiny reference calculations and integrity fixtures

**Question/hypothesis:** Do the implementations measure the declared quantities and enforce the intended boundaries? The known mismatches should be exposed by deterministic cases.

**Controls:** Identical per-item predictions; unequal chart availability by class; independent split streams; same observation repeated through two panels; changed unseen label/outcome; failed CPU report paired with a valid compute approval; changed model bytes during resume; available/unavailable synthetic outcomes.

**Primary measurement:** Agreement with hand-computed loss, marginal, expected cost, and information-flow invariants. No claim of clinical benefit.

**Compute:** Local CPU/synthetic tests and static reference calculations; no clinical inference. Runtime unmeasured.

**Decision:** Any discrepancy blocks use of the affected metric or path. Passing clears an engineering prerequisite, not a performance claim. Resolves whether current reported advantages could be measurement or execution artifacts.

### E1 — Corrected CPU quality, overlap, and eligible-error headroom

**Question/hypothesis:** Does the content model retain predictive skill after matched scoring, and can the current lab-only intervention correct enough baseline errors to matter?

**Controls:** Pooled charted-data marginal; repaired current model; unique-evidence versus legacy panel accumulation as a prospectively labeled diagnostic; unchanged HPI prior; fixed admission grouping. Keep all permitted patient computations private.

**Primary measurement:** Matched predictive NLL/Brier; truth-referenced posterior change; aggregate overlap; number and class of correctable baseline errors at actual content-trigger opportunities. Report the optimistic macro-recall bound and baseline-correct cases at risk of regression.

**Compute:** Credentialed CPU pipeline only, subject to current access/stage permissions; no new policy inference needed for aggregate audits using existing authorized private trajectories.

**Decision:** A failed quality gate stops the corresponding GPU path. Small eligible-error headroom argues against expecting a large gain from the unchanged lab-only intervention; it does not disprove a new reader or expanded future scope. Resolve prediction quality, redundancy, and reach before tuning a selector.

### E2 — Matched-evidence interpretation probe

**Question/hypothesis:** Can a calibrated alternative reader improve predictions from evidence already acquired by the baseline?

**Controls:** Frozen baseline trajectories; existing hypothesis/final reader; repaired current structured model; one modest regularized partial-evidence reader. Match observation access and state clearly when representation changes. Evaluate an all-available-evidence probe separately as privileged.

**Primary measurement:** Paired T5 and class-specific improvements/regressions, proper-score change, nonanswers, and calibration on fixed evidence. Distinguish endpoint replay from the evolving-reader/selector interaction.

**Compute:** CPU for structured readers; additional LLM replay could need separately approved GPU work. Do not assume an all-evidence LLM probe is free.

**Decision:** Improvement on identical evidence supports an interpretation intervention. Lack of improvement narrows that candidate, not every possible reader. The all-evidence probe is diagnostic headroom, not a deployable policy or guaranteed upper bound.

### E3 — Small crossed selector/updater study

**Question/hypothesis:** Is the limiting component prediction, selection, belief updating, or their interaction?

**Controls:** Strong simple acquisition, learned selector, and precisely named privileged selector crossed with fixed and alternative readers/updaters. Start synthetic; then run the smallest approved clinical matrix justified by E1/E2. Hold availability, stopping, evidence support, and budgets fixed for the intended contrasts. Include V1 and V3 as whole-system references without pretending they have identical acquisition mechanisms.

**Primary measurement:** Paired task utility and T5/cost; acquisition regret against an exact small-environment optimum; reader-by-selector interaction; selected-action quality. Predeclare the primary comparison and which cells are diagnostic.

**Compute:** Synthetic CPU first. Clinical content bench is unmeasured. Historical requests estimated 0.188 GPU-hours for two small bench arms and 2.29 for five full validation arms, but those figures omit measured content-planning overhead and do not cost the new factorial design. Benchmark and request a new budget.

**Decision:** Privileged selection helps but learned selection does not: investigate prediction/ranking. Reader helps on the same acquisitions: prioritize interpretation. Only the joint system helps: report an interaction rather than a selector-only effect. No arm helps: reconsider headroom, objective, and evidence support before larger training.

### E4 — Decision-relevant error and selective-intervention experiment

**Question/hypothesis:** Does action-value error explain planning harm better than pooled predictive loss, and can a conservative override improve the trade-off?

**Controls:** Content models with matched or tightly controlled aggregate loss but different errors on decision-critical outcomes; calibration-only and biased-error perturbations; shared reader/stopping; baseline action, mean-only override, and disagreement-aware override. Include redundant evidence and a held-out environment mechanism. A complementary-evidence task can reveal myopic limitations without automatically requiring a deep-search redesign.

**Primary measurement:** Decision regret, improvement/regression counts, cost, action-ranking error, and the relationship between those measures and predictive loss. Evaluate the override rule outside the data used to choose its threshold.

**Compute:** Predominantly synthetic CPU and small model ensembles initially. An approved clinical follow-up is conditional on the mechanism being demonstrated; no policy retraining is necessary to establish the first result.

**Decision:** If gains disappear against matched-information baselines or new perturbation families, do not claim general selective-planning robustness. If decision-relevant errors predict failure and the intervention consistently reduces it, proceed to broader confirmation. Ensemble disagreement alone does not supply a certified confidence bound.

### E5 — Independent confirmation and necessary seed replication

**Question/hypothesis:** Does the frozen method improve the predeclared accuracy–cost objective outside the development process?

**Controls:** Strongest prospectively selected simple baseline, repaired current method, targeted successor, and minimal mechanism ablations. Use independently trained policy seeds only where the claim concerns policy-training variability; keep corruption, fitting, and decoding seeds separately identified.

**Primary measurement:** Frozen primary contrast, confidence intervals, per-class behavior, cost frontier, and between-seed variability. Predeclare any practical noninferiority margin for a cost-saving claim.

**Compute:** Requires new authorization as applicable. The original approximately 105-GPU-hour “per seed” budget includes four training arms in the replication; do not automatically retrain all four for a successor that needs only the primary policy. Cost the actual necessary training and evaluation graph from measured runs. No new estimate is established here.

**Decision:** Independent benefit plus the mechanistic evidence supports a paper claim. An unavailable cohort remains a publication-evidence gap, not a failed software milestone. Another synthetic benchmark broadens evaluation but is not independent real-world clinical validation.

## H. ICML readiness assessment

### H1. Prior art materially narrows the novelty claim

This was a focused primary-source positioning check, not an exhaustive systematic review. Numerical superiority across papers was not assessed: differing splits, backbones, information access, and objectives prevent a direct comparison of headline scores.

| Primary source and verified status | Relevant overlap | Consequence for this project |
|---|---|---|
| [LA-CDM, arXiv v3, February 28, 2026](https://arxiv.org/abs/2506.13474v3) | Hypothesis-driven language agents trained for diagnosis, uncertainty, and efficient sequential decisions on four-disease MIMIC-CDM. | Baseline framing and uncertainty-aware diagnosis are inherited; backbone/runtime deviations remain disclosed. |
| [Li and Oliva, Active Feature Acquisition with Generative Surrogate Models, ICML 2021](https://proceedings.mlr.press/v139/li21p.html) | Learned generative dependencies support acquisition policies and information-gain-related guidance. | A generative predictor plus acquisition policy is established prior art. |
| [Valancius et al., Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition, ICML 2024](https://proceedings.mlr.press/v235/valancius24a.html) | Addresses nongreedy acquisition and limitations of common AFA approaches. | Include a serious acquisition comparator; myopic EIG is not the entire field. |
| [Kobayashi et al., Learning-To-Measure, v2 May 31, 2026](https://arxiv.org/abs/2510.12624v2), also listed in [ICML 2026 downloads](https://icml.cc/Downloads/2026) | Meta-AFA with uncertainty estimation and conditional-mutual-information acquisition under retrospective missingness. | Missingness-aware uncertainty-guided acquisition alone is not the differentiator. |
| [ACTMED, arXiv v5 September 10, 2026](https://arxiv.org/abs/2510.18988v5); arXiv records a NeurIPS 2025 journal reference | Bayesian experimental design with LLM outcome simulation for diagnostic test selection. | “Imagining tests before ordering” is already closely instantiated. |
| [LDTL, arXiv v2 October 22, 2026](https://arxiv.org/abs/2604.05116v2), preprint status in the inspected source | Planning and diagnosis agents trained around informative latent diagnostic trajectories on MIMIC-CDM. | Planner/reader coupling and sequential diagnostic performance need direct differentiation. |
| [CaRDI, arXiv v2 October 16, 2026](https://arxiv.org/abs/2608.27847v2), preprint status in the inspected source | Calibrated diagnostic-risk reduction guides information acquisition and stopping rather than entropy alone. | Replacing entropy by risk plus calibration is not sufficient novelty. |
| [Voelcker et al., Calibrated Value-Aware Model Learning with Probabilistic Environment Models, ICML 2025](https://proceedings.mlr.press/v267/voelcker25a.html) | Studies limitations and corrections of value-aware model-learning surrogates. | Decision-aware modeling has established theory; a generic value objective is not automatically a new contribution. |

This overlap does not make the project unworthy. It means that an isolated combination of calibrated beliefs, imagined outcomes, and cost-aware selection is not yet a sufficiently sharp claim.

### H2. Strongest potential contribution: a mechanism-and-repair study

The strongest direction is a falsifiable account of **which world-model errors alter decisions, how evidence-update inconsistencies amplify them, and whether a targeted intervention prevents harmful overrides**.

A credible paper would demonstrate, rather than merely assert, that models with comparable aggregate predictive quality can have different decision regret; identify a better decision-relevant diagnostic across controlled shifts; and show a targeted repair/selector rule improving task utility against strong matched-information alternatives.

This requires more than the unsurprising observation that a deliberately bad model can hurt. Use different error mechanisms, matching controls, coherent baselines, and held-out environment families. A simple bound relating uniform action-value error to regret can be useful explanatory scaffolding, but elementary ranking-error arguments should not be presented as novel theory.

A **second viable framing** is an evaluation contribution: a compact, reproducible protocol and shared implementation that separates prediction, belief updating, acquisition, and stopping, exposing reproducible failures of common quality proxies. It would need findings across more than this one implementation and convincing independent usefulness. A catalogue of this repository's bugs alone is not that contribution.

### H3. Current versus achievable readiness

| Dimension | Current assessment | Evidence still required |
|---|---|---|
| Soundness | Strong baseline documentation, but content quality and synthetic evaluation contain material defects. | Corrected metrics, execution bindings, coherent controls, independent reference cases. |
| Performance | Fallback recovery is demonstrated in recorded data; content-planning benefit is not. | Improvement against V1/other strong simple references at a matched or explicitly traded-off cost. |
| Mechanism | Useful negative observations and hypotheses, but selection versus interpretation remains unresolved. | Crossed controls, truth-referenced updates, decision-regret analysis, explanatory ablations. |
| Originality | General approach has close prior art; proposed sharper claim remains unproven. | A differentiating result that survives the closest conceptual and empirical alternatives. |
| Generalization | One policy seed, reused clinical test benchmark, unresolved subject grouping. | Independent evaluation and correctly identified seed variability, scoped to the intended claim. |
| Presentation | Detailed record exists; some historical status and conclusion wording is stale or too strong. | One coherent narrative with calibrated claims, provenance, limitations, and student-understandable explanations. |

The current package is **not yet a competitive completed ICML submission**. It is a credible research foundation with a feasible evidence path. That is not a prediction of acceptance or a requirement to invent a new neural architecture.

[ICML's 2026 reviewer instructions](https://icml.cc/Conferences/2026/ReviewerInstructions) distinguish soundness, presentation, significance, and originality and allow insight-oriented contributions. They do not require every experiment to be positive. This is a useful review framework, not a claim about an unverified future year's submission policy or deadline.

A rigorous null result can support publication when it establishes a generalizable mechanism or corrects a meaningful evaluation assumption. A single nonsignificant contrast on 240 previously examined admissions is not enough on its own.

## I. Professor discussion preparation

### Three findings Daniel should understand

**1. The largest established gain is answer recovery, not improved imagination.**

Technical meaning: the recorded simple fallback moves T5 from 73.09% to 84.45%; full guards reach 81.83%, with no significant V3−V1 difference. This makes fallback a mandatory strong reference for successor claims, not proof that every guard is harmful. [R1]

Plain English: “Sometimes the system had a useful diagnosis in its hypothesis component but never delivered it as its answer. Recovering that answer helped a lot. We now need to show that planning adds something beyond that simple fix.”

**2. A model can appear better because it was graded differently, or become more confident without becoming more correct.**

Technical meaning: joint and item likelihoods are mismatched in the quality comparison; gain calibration compares the updater with itself. C1 repairs a real overconfidence symptom, but the resulting development scores do not establish a better final diagnostic agent. [C1, R4]

Plain English: “One model was effectively being graded on the whole lab panel and the other one item at a time. We need a fair comparison. We also need to check whether new evidence improves the diagnosis, not just whether the model sounds more certain.”

**3. The decisive next question is whether the system chooses the wrong evidence or interprets useful evidence badly.**

Technical meaning: a component-controlled selector/reader comparison can distinguish those explanations; the current oracle changes availability as well as content and is not an optimal decision-maker. [C1, S3]

Plain English: “We will first give two diagnosis methods exactly the same evidence. Then we will change which tests they receive while holding interpretation fixed. That tells us whether to improve the test chooser, the interpreter, or both.”

### Three questions for Professor Liu

1. “Would the strongest contribution be a method that prevents harmful planning, or a controlled study showing which prediction and belief-update errors actually cause the harm? Which result would make the distinction convincing to you?”
2. “Given the fallback baseline and the lab-only intervention's limited reach, should we first prioritize the diagnosis reader and a controlled acquisition study rather than expanding the planner or retraining more large policies?”
3. “What genuinely independent, authorized evaluation would you consider sufficient for the clinical claim, and what accuracy–cost improvement would be scientifically meaningful enough to freeze before we run it?”

### Claims to avoid

Do not say the content world model has passed a valid quality gate, improved agent accuracy, or been independently validated. Do not say a null availability result proves no possible benefit, that the oracle is an optimal ceiling, that C1 is fully out of fold, or that every synthetic result leaked. Do not call a reused benchmark untouched, a simulated environment clinical validation, or a passing code review scientific proof.

Explain T5 as the average of the four disease-specific recalls, not simply the overall fraction correct. Distinguish diagnostic-test cost from GPU compute cost. Attribute Edwin's synthetic implementation, Daniel's recorded review/experiment work, automated findings, and agent-generated planning accurately.

## J. Recommended immediate actions

1. **Adjudicate and integrate the targeted instruction revisions before the next autonomous implementation session.** Add early synthetic integrity checks, the component contract, artifact-bound preflight/resume, the reader-probe dependency, successor-study identity, and explicit authorization precedence. Preserve the historical studies. This review does not itself authorize those edits; Daniel must start the separate implementation task.

2. **In that authorized implementation task, repair and verify measurement before requesting new clinical GPU work.** Implement the hand-computable scoring/reference cases, split-isolation and missing-cost fixes, content execution checks, and action/thought repair. Run permitted CPU/synthetic verifications and approved private aggregate audits, then produce the first corrected professor packet. Report whether quality survives correction; do not require a positive result to finish this work.

3. **Use corrected evidence to freeze the smallest decisive selector/reader experiment, then obtain its exact approval.** Measure eligible-error headroom and fixed-evidence reader behavior first. Request GPU time only for justified cells with bound artifacts and declared controls. Postpone broad policy retraining, new protected-data access, and fresh test evaluation until separately authorized and scientifically warranted.
