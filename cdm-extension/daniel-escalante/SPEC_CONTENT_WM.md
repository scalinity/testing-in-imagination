# SPEC: lab-result world model and imagined-outcome planner ("testing in imagination")

Project: Testing in Imagination (CAI 6734, UF) · Owner: Daniel Escalante · Draft, 2026-11-11
Folder: `cdm-extension/daniel-escalante/` · Status: design and preregistration plan only. No code, no runs and no GPU time exist under this spec.

## 0. Purpose and boundaries

### What it adds

This spec takes the next step on the ICML 2027 path (SPEC.md §12.1–12.2): a learned model of what a test would show, p_φ(o_{t+1} | h_t, a_t), and a planner that uses those imagined results to choose tests.

- **Phase B predicted whether a result is in the chart.** It never predicted what the result says.
- **This spec predicts lab content,** and asks whether imagining it helps the trained policy diagnose.

### Scope

- **Labs only:** the 7 lab panels CBC, BMP, CMP, RFP, LFP, UA and EP. Imaging and the physical exam keep their current behaviour; imaging content comes later (SPEC.md §12.1).
- **The policy, data, split, metric and V3 guards stay as they are.** The trained Nemotron Nano 8B policy (run `7fcd9e5`, seed 269, greedy), the MIMIC-IV-Ext-CDM splits, mean class recall (T5) and the V3 guards (G1, G2, G3) are unchanged.
- **The comparator is V3,** which scores T5 81.83 on test against the trained policy's 73.09 (`results/credentialed/phase-a/metrics.json`).
- **Wording:** these are extensions of a replication attempt and are never described as a reproduction of the LA-CDM paper.

### The data boundary of SPEC.md §1 applies unchanged

- Everything that touches patient data runs on HiPerGator.
- Only schema-valid aggregates become public.
- No row, HPI text, lab value, generation, ID or fitted model parameter leaves private storage. Fitted parameters are patient-derived, as Phase B's TF-IDF models are.

### The test split

The test split is a previously examined benchmark: Phase A and Phase B both evaluated on it. This study therefore treats test as one more prospectively frozen follow-up. All development uses train and validation only (§4, §5).

## 1. The world model p_φ(o_{t+1} | h_t, a_t)

### 1.1 Inputs: the history h_t

h_t is what the policy has seen by turn t:
- the HPI summary;
- the set of tests requested so far, each marked provided or unavailable;
- for provided lab panels, their binned item values (§1.2).

Imaging and exam results enter only as provided or unavailable. Their text is not modelled here.

### 1.2 Outputs: the predicted result o_{t+1} for a lab panel a_t

The outcome has two parts:

1. **Availability:** whether the panel is in the chart at all. This reuses Phase B's learned availability model, p_avail(a | HPI). Its validation quality for seed 0 is AUROC 0.957, ECE 0.030 and Brier skill 0.101 (`prereg/addendum-b.json`, `val_quality.B-learned-s0`).
2. **Content, given availability:** one outcome per item that the panel lists in the release's `lab_test_mapping.csv`.
   - **Numeric value:** one of 3 bins, cut at the train-split tertiles of that item, taken separately per unit when an item reports more than one unit.
   - **Non-numeric value:** a token from a train-split vocabulary of value tokens; tokens with fewer than 10 train occurrences become "other".
   - **Missing:** "absent" when the panel is available but this item is not reported.

Why bins rather than abnormal flags:
- The converted split files hold each lab result as an item-ID → "value unit" string. Reference ranges are not part of the split schema the replication validates (`replication/data.py`, `RAW_FIELDS`).
- So an abnormal flag needs either reference-range fields in the PhysioNet release, or fixed published adult reference intervals.
- Before fitting, one cluster check lists the release's column names, and only the names. If reference-range fields exist, an abnormal / normal / absent flag per item becomes a secondary output scored the same way. The bins stay primary, because they need no external table.

### 1.3 Model form: a latent-disease mixture

The disease y is the latent variable, as in the synthetic environment's discrete world model (§8). Three pieces:

- **Diagnosis prior from the HPI:** p_ψ0(y | HPI). A multinomial logistic regression on Phase B's HPI TF-IDF features, under the same frozen settings as SPEC.md §4.2. It is temperature-scaled on validation.
- **Class-conditional content:** p_θ(o | y, a). The per-item bin frequencies given the disease, with add-one smoothing, fit on train. Items are conditionally independent given y and the panel. This is a stated simplification, and it is checked on validation (§1.5).
- **Belief update after observed panels:** p(y | h_t) ∝ p_ψ0(y | HPI) · Π over provided panels b of p_θ(o_b | y, b).

The predictive distribution over a panel's content is then:

> p_φ(o | h_t, a) = Σ_y p(y | h_t) · p_θ(o | y, a)

This form gives the information gain over the diagnosis posterior directly (§2), and it matches the synthetic environment's interface (§8).

**One alternative, chosen offline by validation log-loss:** a discriminative per-item model given h_t, a multinomial logistic regression on HPI TF-IDF plus the observed bins. The choice is made on CPU from chart data, with no policy runs.

### 1.4 Training data: train split only

Lab results do not depend on the order in which tests are requested. So each of the 1,920 train admissions (SPEC.md §4.2) yields tuples (h_t, a_t, o_{t+1}) directly from its chart:
- draw 8 random orders of the admission's tests, with seeds fixed by sha256 as in SPEC.md §4.3;
- emit one tuple for every lab panel in each order.

The records use the tuple schema of `docs/env-and-tuples.md` §4 with `source: "mimic_real"`. They are private JSONL, as the Phase B inventory is.

These chart-derived histories differ in distribution from the histories the policy actually produces. An optional robustness fit uses train-split policy rollouts instead (§6, GPU). Validation and test admissions are never used for fitting.

### 1.5 Calibration and quality: validation only

The same 8-order construction on the 240 validation admissions gives the evaluation tuples. Reported per item and pooled, against the train marginal as the reference model:

| Quantity | Metrics |
|---|---|
| Content | Log-loss (NLL); multiclass Brier and Brier skill; top-label ECE (10 bins); reliability tables |
| Availability | AUROC and ECE, as in SPEC.md §4.6 |
| Diagnosis posterior p(y \| h_t) | Validation NLL and top-label ECE, before and after each observed panel |
| EIG calibration | Mean predicted information gain against mean realized gain, i.e. the entropy drop under the panel's true result, by decile of predicted gain |
| Conditional-independence check | Predicted against observed pairwise item co-occurrence for each panel, reported as one divergence per panel |

All of these are CPU computations on validation, published as aggregates.

### 1.6 Leakage audit (validation; counts and metrics only)

1. **Split hygiene.**
   - Every fitted object is fit on train only and its sha256 is published.
   - Validation is the only development split.
   - Test is opened once, under the frozen protocol (§6).
2. **HPI copying.**
   - The Phase B HPI audit counted panel names in validation summaries: CBC 1, UA 1, the other lab panels 0 (`results/credentialed/hpi-audit/hpi_audit.json`). Item names such as "lipase" or "WBC" were not counted.
   - The audit adds per-item counts of validation summaries that name a panel item within 5 words of a number.
   - It reports content log-loss separately for admissions with and without such a mention. A much lower log-loss in the mention group marks the model as copying the history rather than predicting.
3. **Workup leakage through availability.**
   - Availability bits alone predict the diagnosis well above chance: validation T5 46.46 for bits alone, 62.32 for HPI alone and 64.77 for both (`results/credentialed/leakage/leakage.json`).
   - So every planner arm uses the same availability head. Content arms then differ only in what they predict a result will show, and availability information cannot explain a content effect.
4. **Labels.**
   - Inputs exclude every diagnosis or discharge field.
   - The oracle and misleading arms (§3) use privileged information by design. They are controls, never deployable methods.
5. **Person-level overlap** between splits cannot be assessed: there is no `subject_id`. The limitation of SPEC.md §7 is stated again.

## 2. The planner: expected information gain per dollar

### 2.1 Score

For an untried lab panel a, at history h_t:

> EIG(a | h_t) = p_avail(a | HPI) · [ H(p(y | h_t)) − E_{o ~ p_φ(o | h_t, a)} H(p(y | h_t ⊕ o)) ]
>
> score(a) = EIG(a | h_t) / cost(a)

- An unavailable result contributes no gain.
- The expectation uses K = 64 Monte Carlo draws, with per-draw seeds sha256("tim-eig|split|patient|t|test").
- cost(a) is the replication's per-test cost in USD (`evaluation.test_costs_usd`).
- Entropy is in nats.

### 2.2 Where it acts

The planner hooks into the same place as Phase B's (SPEC.md §3.3 step 5), with the same precedence: G2 > G1 > planner.

- **When it acts:** on a decision turn with t < 13 where V3 would execute a request for a lab panel.
- **What it does:** it replaces that request with the untried lab panel of highest score. Ties go to the configured order.
- **Imaging and physical-exam requests** pass through unchanged.
- **The replacement** uses SPEC.md §3.3 step 6's replacement fit, with the fixed thought "I will choose the next step." It logs an `eig_rerank` event when the panel changes.

Invariants, each carried over from Phase B:
- planner off ≡ V3;
- an untriggered turn executes the greedy completion byte for byte.

### 2.3 Scoping decision for v1: labs only

- **The decision.** The world model predicts lab panels only, so v1's planner acts only on turns where V3 would request a lab panel. Imaging and physical-exam choices stay with the policy.
- **What it means for the result.** The policy still decides when to order a lab and when to order imaging; the planner decides only which lab panel. Any difference between a world-model arm and V3 is therefore attributable to lab selection alone.
- **What extends it.** Acting on imaging needs a content model of imaging reports (SPEC.md §12.1), which is outside v1.

### 2.4 Commit rule

The commit rule is V3's guards, unchanged:
- G1 validity;
- G2 commit at t = 13 or on completion overflow;
- G3 hypothesis fallback at finalization.

The primary design adds no information-based stopping rule. A stop-when-max-score-below-θ variant is a validation-only exploratory option. It is never on the test path unless a later protocol freezes it before test access.

### 2.5 Premise check (CPU, before any GPU run)

- **What it measures:** from val-a's V3 rows, the number of validation admissions in which V3 requested at least one lab panel, and the number of such turns. Aggregate counts only.
- **Why:** the planner can act only on those turns, so the count bounds how much it can change.
- **The threshold:** at least 24 admissions, set in §5 before the count is computed.

## 3. Arms and contrasts

All arms run on top of V3 and share the availability head.

| Arm | World model given to the planner | Purpose |
|---|---|---|
| V3 (model-free) | none; val-a and test-a V3 rows reused, no new run | comparator |
| WM-learned | p_φ as fit in §1 | the method |
| WM-oracle | the chart's true result when available, and true availability; EIG becomes the realized gain | privileged ceiling for this planner |
| WM-prior | p_φ with p(y \| h_t) replaced by the train class prior: imagined results ignore this patient | uninformative world model, i.e. population-level informativeness only |
| WM-mislead-α | (1 − α) · p_φ + α · p_θ(o \| y′, a), where y′ ≠ the label is chosen by sha256; α ∈ {0.5, 1} | a deliberately wrong world model: does a bad model hurt? |

**Primary contrast (test, α = 0.05, the only uncorrected test):** WM-learned − V3.

**Family W (test, Holm at α = 0.05):** every other arm against the same comparator, each two-sided:
- WM-oracle − V3: how much this planner could gain from perfect imagined results.
- WM-prior − V3: what imagined results give when they ignore the patient.
- WM-mislead-α0.5 − V3 and WM-mislead-α1 − V3: whether a bad model hurts.

**Descriptive only:**
- WM-learned − WM-prior: patient-specific prediction beyond population informativeness;
- the misleading-model dose-response, the slope of T5 on α over WM-learned (α = 0), α = 0.5 and α = 1, with a class-stratified bootstrap interval as in amendment A4's D3;
- T1–T8, requests and cost, per-class recall, and planner events.

Statistics follow SPEC.md §6 unchanged:
- **Estimand:** ΔT5 = Σ w_i d_i over the 240 test admissions.
- **Test:** two-sided paired sign-flip randomization.
- **Interval:** 95% class-stratified paired percentile bootstrap, 10,000 resamples.
- **Wording:** "not statistically significant" whenever p ≥ 0.05 (Holm-adjusted inside Family W) or the interval includes 0.
- **One policy seed:** between-seed variation is unassessable until §7's seeds plan runs, and every report says so.

### Minimum detectable effect

The 240-admission test design, at class sizes 96 / 65 / 25 / 54, detects these ΔT5 values with 80% power, at Phase A's validation discordance (`prereg/addendum-a.json`, `sensitivity.*.mde_80`):

| Discordance pattern | MDE at 80% power |
|---|---|
| V1 − V0 | 4.15 points |
| V3 − V0 | 5.54 points |

The planning figure is therefore an MDE of about 4–6 points. Before test access, an addendum recomputes the MDE from WM-learned − V3's own validation discordance, using SPEC.md §5.7's simulation, and null results are read against that figure.

## 4. Validation-only development

- **Offline on CPU, from chart tuples:** the model choice (mixture or discriminative), the temperature of p_ψ0, and the smoothing are all set by validation log-loss, with no policy runs.
- **Fixed now, not tuned:** the bin edges (train tertiles), K = 64, and the cost table.
- **Validation GPU runs:**
  - bench (24 validation patients; WM-learned and WM-oracle);
  - one validation run of each of the 5 new arms;
  - at most 4 further validation runs for repairs.

  Each run is logged in the README run log with its SLURM job ID. Validation decides go or no-go only. No arm or setting is selected by a validation T5 comparison, as in Phase A.
- **Failure diagnosis:** as SPEC.md §1.2 sets out, from error codes and the synthetic fixture. A failure that appears only on real data is read on the cluster by Daniel.

## 5. Validation decision rule (fixed before any run)

The rule is applied in order. The first two gates are CPU-only and come before any validation GPU run.

| Gate | When | Passes if | If it fails: the null we report |
|---|---|---|---|
| 1. Premise | CPU, before GPU | At least 24 of the 240 validation admissions had a lab-panel request by V3 (val-a V3 rows, §2.5). | **Premise null:** report the count; the lab-only planner has too little room; no GPU run. |
| 2. World-model quality | CPU, before GPU | Pooled validation content log-loss is below the train marginal's (skill > 0), and mean realized information gain rises across the 10 deciles of predicted gain (Spearman correlation > 0, §1.5). | **World-model null:** report the calibration results; no GPU run. |
| 3. Integrity | after the validation runs | Every arm completes all 240 validation admissions, with no scorer error or replacement failure left unexplained. | Repair within the 4-run margin (§4); if that fails, report the failure and stop. |
| 4. Direction | after the validation runs | WM-learned − V3 has a positive ΔT5 point estimate on validation. | **Validation null:** report the validation contrasts, intervals and per-class recall as the finding; no test run. |

Why these thresholds:
- **24 admissions (10% of 240) is a choice set here, not a measurement.** The planner can change an admission only where it acts. At the test class sizes 96 / 65 / 25 / 54 (`prereg/protocol.json`), one admission moves T5 by between 0.26 and 1.00 points (w = 100 / (4 · n_class), SPEC.md §6). Reaching the 4.15-point MDE (§3) therefore needs on the order of ten net-improved admissions, so a premise far below that leaves no room.
- **Gates 2 and 4 are directions, not effect sizes.** Validation decides go or no-go only; it never selects an arm or setting (§4).

**Go to the frozen test** only when all four gates pass, and then only after Daniel approves (§6).

**On test, after a go:** if the primary contrast is not statistically significant, the report states the null with its interval, read against the recomputed MDE, alongside Family W, as SPEC.md §11 does. There is no re-tuning and no re-run.

## 6. Frozen test protocol (written now; run only after validation development is complete and Daniel approves)

This section is the test protocol. It is not run until all four gates of §5 have passed on validation and Daniel has approved test access.

**What is fixed now, as written in this spec:**
- the arms (§3), their shared availability head and the planner (§2), with K = 64 and the cost table;
- the primary contrast WM-learned − V3, the only uncorrected test at α = 0.05;
- Family W, the four contrasts of §3, Holm at α = 0.05;
- the descriptive analyses of §3, labelled descriptive;
- statistics and wording as SPEC.md §6: the paired sign-flip randomization test, exact up to 20 discordant admissions and otherwise 100,000 draws, seeds `stream_seed(269, "test", "tim-permutation:<comparison>")`; 95% class-stratified paired percentile bootstrap, 10,000 resamples, seed `stream_seed(269, "test", "tim-bootstrap")`;
- the comparator: test-a's V3 rows, never re-run on test.

**What is filled in at freeze, from train and validation outputs only:**
- the model choice, temperature and smoothing (§4), and the sha256 digests of every fitted file;
- the bin edges (train tertiles, set before validation);
- the premise count and its pass (§5, gate 1);
- the MDE recomputed from WM-learned − V3's validation discordance (§3);
- bench timings and the test budget they imply;
- the test-set status disclosure.

All of the above is committed as `prereg/content-wm-protocol.json` before any test access.

How test access then works:
- **Approvals:** each GPU stage gets an approval bound to the protocol hash, as SPEC.md §2 already does.
- **Admissions:** test admissions are one-shot and keyed by protocol hash, stage and arm, so no code change can reopen them.
- **V3 rows:** test-a's V3 rows are reused as the comparator, so V3 is never re-run on test.
- **Amendments:** after the freeze, only code fixes may change, each by an amendment note written before test access, as in prereg/amendments A1–A3.
- **No test-based tuning:** nothing is re-tuned on test. Any analysis not frozen in the protocol is labelled exploratory.

## 7. Compute

### Estimate for this study (an estimate, not a measurement)

- **Per-arm basis:** each 240-admission arm is costed at addendum B's per-arm budget of 0.458 GPU-hours (`prereg/addendum-b.json`). The val-b arms actually ran in 27–28 minutes each (README run log).
- **Unmeasured overhead:** the EIG computation adds CPU work inside each GPU job (K = 64 draws × at most 7 panels per lab turn). bench measures it.

| Work | Runs | GPU-hours (estimate) |
|---|---|---|
| bench | 2 arms × 24 validation patients | about 0.2 (bench-b took 335–340 s per arm) |
| validation | 5 arms × 240 | about 2.3 |
| test | 5 arms × 240 | about 2.3 |
| repair margin (validation only) | up to 4 runs | up to about 1.8 |
| **Core total** | | **about 5, and about 7 with the margin** |
| CPU: tuple building, fitting, calibration, leakage audit | burst QOS | unmeasured; hours, not GPU time |
| Optional: train-split policy rollouts for the robustness fit | 1,920 admissions | about 3.7 (8 × 0.458) |

### Seeds plan: pending Daniel's approval and advisor feedback

- **The plan:** 3 policy seeds in total, so 2 more seeds beside seed 269: about 211 GPU-hours, the minimum for a three-seed between-seed interval (`replication-of-original/daniel-escalante/README.md`, GPU budget).
- **Source:** each extra seed costs about 105 GPU-hours from prior runs. Training took 103.6 GPU-hours per seed (four arms × 3,840 steps, 28 h on 4 GPUs) plus about 1.8 GPU-hours of test evaluation (same source).
- **Plus:** the 5 content-WM test arms on each new policy, about 2.3 GPU-hours per seed (estimate, §7), so about 4.6 for both.

### Llama-3.1-8B-Instruct backbone ablation: pending Daniel's approval and advisor feedback

- About 110 GPU-hours: one training seed plus regenerated HPI summaries and a zero-shot baseline (same source).
- Then the content-WM arms on that policy, about 2.3 GPU-hours per split.

### Constraints for both

- **Course staff approval:** multi-GPU jobs and jobs longer than 24 hours need the course staff's approval first (same source). Training a seed ran 28 h on 4 GPUs, so both the seeds plan and the backbone ablation need it, in addition to Daniel's approval.
- Class accounts share 8 GPUs and 250 GB.
- Queue time adds to every wall-clock figure.

### Job plan (prepared, not submitted)

**Status.** Nothing in this plan has run.
- No `content` planner kind exists in `tim` yet, so these jobs cannot run, or be dry-run through `scripts/submit_slurm.py`, until milestone 1's code adds the configuration below to `configs/variants.yaml`.
- The submitter, accounts and job shape are the ones already in use: `dose-val-b`'s dry run on 2026-11-11 printed the same form for the Nemotron policy.
- Nothing is submitted without Daniel's approval.

Configuration to add with the implementation (Nemotron policy, run `7fcd9e5`, seed 269, unchanged):

```yaml
arms_c:  # on top of V3; planner kinds named in SPEC_CONTENT_WM.md §3
  WM-learned:      {g1: true, g2: true, g3: true, planner: {kind: content, model: learned}}
  WM-oracle:       {g1: true, g2: true, g3: true, planner: {kind: content, model: oracle}}
  WM-prior:        {g1: true, g2: true, g3: true, planner: {kind: content, model: prior}}
  WM-mislead-a0.5: {g1: true, g2: true, g3: true, planner: {kind: content, model: mislead, alpha: 0.5}}
  WM-mislead-a1:   {g1: true, g2: true, g3: true, planner: {kind: content, model: mislead, alpha: 1}}
stages_c:  # names end in "c" so approval estimates come from bench-c
  bench-c: {split: val, jobs: [WM-learned, WM-oracle]}
  val-c:   {split: val, jobs: [WM-learned, WM-oracle, WM-prior, WM-mislead-a0.5, WM-mislead-a1]}
  test-c:  {split: test, jobs: [WM-learned, WM-oracle, WM-prior, WM-mislead-a0.5, WM-mislead-a1]}
```

| Stage | Jobs | GPU-hours per job | Source |
|---|---|---|---|
| bench-c | 2 | about 0.09 | bench-b measured 334.6 s and 340.0 s per arm (`results/credentialed/bench-b/`) |
| val-c | 5 | about 0.458 | addendum B's per-arm budget, an estimate; bench-c replaces it |
| test-c | 5 | about 0.458 | as above; runs only under §6 |
| CPU: tuples-c, wm-fit-c, analyze-val-c, analyze-c | 1 each | none (CPU) | burst QOS |

Commands, generated by the submitter:
- **GPU stages:** `SLURM_ACCOUNTS=cai6734 scripts/submit_slurm.py --stage <stage> --dry-run` prints one line per job:

  ```
  sbatch --parsable --nodes=1 --ntasks=1 --chdir=<worktree>/cdm-extension/daniel-escalante
    --export=ALL,TIM_WORK_DIR=<TIM_WORK_DIR>,CDM_WORK_DIR=<CDM_WORK_DIR>,HF_HOME=<HF_HOME>
    --output=<TIM_WORK_DIR>/private-logs/slurm-%j.out --error=<TIM_WORK_DIR>/private-logs/slurm-%j.err
    --account=cai6734 --qos=cai6734 --partition=hpg-rtx6000 --gres=gpu:rtx_pro_6000:1
    --cpus-per-task=4 --mem=16G --time=<limit> --job-name=tim-<stage>-<job> scripts/gpu.slurm <stage> --job <job>
  ```

  The time limit is the submitter's max(60 min, 2 × GPU-hours + 30 min): 01:24:00 for a 0.458-hour job and 01:00:00 for a bench job.
- **CPU stages:** `SLURM_CPU_ACCOUNT=cai6734 SLURM_CPU_QOS=cai6734-b scripts/submit_slurm.py --stage <stage> --dry-run` prints `... --account=cai6734 --qos=cai6734-b --cpus-per-task=4 --mem=32G --time=04:00:00 --job-name=tim-<stage> scripts/cpu.slurm <stage>`.

Outputs each GPU job writes:
- **Private:** the episode log `$TIM_WORK_DIR/private/episodes/<protocol16>_<stage>_<arm>_primary[_<commit12>].jsonl` (the commit suffix is on validation only); the admission marker and ledger entries in `$TIM_ADMISSION_DIR`; and `$TIM_WORK_DIR/private-logs/<job id>.log`.
- **Public, after `tim publish`:** `results/credentialed/<stage>/events/<arm>.jsonl`, plus `bench.<arm>.json` for bench-c.
- **CPU stages:** only their schema-valid aggregate JSON is public.

## 8. Proposed shared planner interface (proposal only)

### What exists today

Edwin's synthetic world-model study (PR #1, branch `edwin-salcedo/synthetic-world-model`; read, not edited) already defines:
- a `DiscreteWorldModel`, with `initial_belief`, `update`, `posterior_y`, `predictive`, `information_gain` and `p_available`;
- a `Selector`, with `reset`, `observe`, and `select(candidates, posterior_y)` returning a test or None.

Its imagination selector scores p_available · IG − λ · cost and stops when the best score is not positive.

### The proposal

The MIMIC planner above fits the same shape. A shared interface needs only these changes:

```python
Outcome = Hashable  # synthetic: a finding string; MIMIC: a tuple of item bins for one lab panel

class WorldModel(Protocol):
    def initial_belief(self, history: str) -> Belief: ...            # MIMIC conditions on the HPI
    def update(self, belief: Belief, test: str, outcome: Outcome | None) -> Belief: ...  # None = unavailable
    def posterior_y(self, belief: Belief) -> Mapping[str, float]: ...
    def predictive(self, belief: Belief, test: str) -> Mapping[Outcome, float]: ...      # given available
    def p_available(self, belief: Belief, test: str) -> float: ...   # patient-specific on MIMIC

class Selector(Protocol):
    def reset(self, history: str) -> None: ...
    def observe(self, test: str, outcome: Outcome | None) -> None: ...
    def select(self, candidates: Sequence[str], posterior_y: Mapping[str, float]) -> str | None: ...

Score = Callable[[WorldModel, Belief, str], float]  # ig_minus_cost(lam) (synthetic today) or ig_per_cost (§2.1)
```

What changes and what stays:
- **Only two signatures change.** `initial_belief` gains the history text, and `p_available` gains the belief. In the synthetic environment both can ignore the new argument.
- **Commit rules stay on the environment's side.** On MIMIC that is V3's guards; the synthetic environment keeps its own test limit.
- **What sharing buys.** One selector runs in both places. The synthetic environment's complete outcomes, with availability masks independent of the diagnosis, can then carry the causal claims that MIMIC's chart-limited oracle cannot (SPEC.md §12.3).
- **Process.** Adopting any of this is for Edwin to decide in his own code.

## 9. Milestones (none started)

1. CPU: tuple building, both model forms, calibration and leakage audit on validation, the premise count, and the synthetic-fixture end to end with canary tests.
2. bench → approval → validation runs → addendum (MDE, digests, bench timing).
3. Protocol freeze → approval → test runs → analysis → published report rendered from metrics.json.
4. Optional, each pending Daniel's approval: the robustness fit on policy rollouts, the seeds plan and the backbone ablation.

The ICML 2027 deadline is not announced; the planning estimate is around January 22, 2027 (SPEC.md §12.6).
