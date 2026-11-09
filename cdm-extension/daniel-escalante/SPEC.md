# SPEC v2: Commit guard and retrospective chart-availability reranking

Project: Testing in Imagination (CAI 6734, UF) · Owner: Daniel Escalante · v2, 2026-11-09 (supersedes v1)
Folder: cdm-extension/daniel-escalante/ · Branch: daniel/cdm-extension · Python package: tim

## 0. Purpose, framing and disclosure

Instructor asks (Dr. Xuefeng Liu)
- Sep 1: plan with a learned world model p_phi(o_{t+1} | h_t, a_t); compare a model-free agent, a learned-world-model planner and an oracle planner while varying world-model accuracy and calibration. Stage 1: reproduce LA-CDM, formalize the environment, design (h_t, a_t, o_{t+1}) examples. MIMIC validation later.
- Now: improve on the trained Nemotron model's mean class recall (73.09%) with a lightweight idea of our own.

Claims and non-claims
- Phase A is termination recovery and action validation under the frozen metric. Its fallback reuses the hypothesis agent's answer, so it is not a new diagnostic predictor, and it supports no clinical-safety claim.
- Phase B is retrospective chart-availability prediction and reranking. It estimates p(test present in chart | HPI) and reweights test requests. It does not predict result content, does not update from revealed findings and does not evaluate imagined outcomes. Its oracle is a privileged chart-availability oracle: chart presence encodes clinicians' completed workups. Phase B is the first narrow MIMIC step toward the thesis, not a test of full world-model planning.
- Test-set status: a prospectively frozen follow-up analysis on a previously examined benchmark. These published test aggregates informed the design: per-arm confusion matrices with an undiagnosed column (greedy and temperature 1.0), undiagnosed / valid / repeated request counts, the cost decomposition (267 unavailable requests), ignore/* metrics (84.62), test hypothesis accuracy (81.67% over 622 triplets), and the zero-shot hypothesis distribution (904 of 1,095 cholecystitis) with its undiagnosed-by-request-count breakdown. No patient rows or trajectories were viewed. Whether anyone has computed the fallback (G3) on test rows is recorded in prereg/protocol.json from Daniel's answer.

Baseline (run graph 7fcd9e5, seed 269, greedy; replication-of-original/daniel-escalante/results/mimic/)
- primary: T1–T4 90.62 / 72.31 / 72.00 / 57.41; T5 73.09 [66.65, 79.19]; T6 81.15; T7 78.20; T8 $1,439.58.
- 29/240 nonanswers: appendicitis 5/96, cholecystitis 6/65, diverticulitis 4/25, pancreatitis 14/54.
- Analytic references on those nonanswers (not results): uniform random fallback +3.52 T5 points expected; training-majority (appendicitis) fallback +1.30; perfect recovery +14.09.

Out of scope: policy retraining; any new backbone, data, split or metric; content prediction (§12); repository edits outside this folder except one row in the top-level README.md repo map.

## 1. Execution model and data boundary
1. The coding agent works in the repository checkout on Daniel's Mac and runs every stage itself, CPU and GPU, on HiPerGator, where the real MIMIC-IV-Ext-CDM data and the baseline work directory live. It reaches the cluster through the sshhiper alias (an SSH ControlMaster; Daniel renews the Duo login when it expires) and uses every course account and QOS Daniel holds.
2. The agent sees only what this section makes public: published bundles, SLURM job metadata (job id, state, exit code, elapsed time, GPU type) and the event stream's error codes. It never prints, lists, copies or reads private logs, rows, HPI text, generations, inventories or world-model files, because under the PhysioNet DUA the agent's conversation is a third-party service. A failure is diagnosed from its error code and exception class and reproduced on the synthetic fixture; a failure that reproduces only on real data is read on the cluster by Daniel.
3. Every credentialed job redirects stdout and stderr to a private file at the shell level, before Python starts: the first command after the #SBATCH headers is exec > "$TIM_WORK_DIR/private-logs/$SLURM_JOB_ID.log" 2>&1. SLURM --output/--error also point to private storage.
4. Python writes private artifacts (rows, models, inventories) plus a staging bundle. The bundle holds only (a) a fixed-schema event stream and (b) aggregate JSON, both validated on write against schemas/*.json. Validation is a positive allowlist: enumerated keys; numbers, integers and booleans; strings only from code-defined enumerations or the identifier pattern ^[A-Za-z0-9_.:-]{1,64}$. Exceptions are caught at the top level and emitted as an error-code enum plus an exception class name from an allowlist ("Other" otherwise). Messages and tracebacks stay in the private log.
5. tim publish <stage> re-validates the bundle, runs the canary scan and copies the bundle to results/credentialed/<stage>/ in the cluster checkout. The agent copies that directory to the local checkout, reviews the diff, commits and pushes. The agent reads only published bundles.
6. Never public: raw logs, rows, HPI text, generations, inventories, world-model files (TF-IDF vocabularies and coefficients are patient-derived), exception messages, any per-patient value.
7. replication.manifest.assert_aggregate rejects forbidden key names only and does not inspect values. It runs in addition to the schema check, never instead of it.
8. Canary tests (CPU, synthetic). Invented markers such as "TIM-CANARY-7f3a" are injected into HPI text, model outputs, exception messages, assertion messages, multiline prints, pandas reprs and stderr written from a subprocess. They must appear in the private log and in no file under the staging bundle or results/.
9. Repository edits stay inside this folder, plus the README row. Runtime artifacts go to $TIM_WORK_DIR and $TIM_ADMISSION_DIR: private, outside the repo, and never inside $CDM_WORK_DIR.
10. $CDM_WORK_DIR, the baseline, is read-only to this project. Use only replication functions that do not write: cfg_for, prepared, _rows_path, private_read, read_json, file_digest, data.load_dataset, gpu.evaluate, pipeline._assert_runtime. Never call stage_lock, private_write, _saved_evaluation or write_json on that directory. A CPU test runs the fixture with its fake baseline directory made read-only.

## 2. Stage × split access, admissions, approvals
All data loading, on CPU and GPU, goes through tim.access.open_split(split, stage). It is the only caller of the replication dataset loader and the only reader of official rows.

| Stage | train | val | test |
|---|---|---|---|
| validate | digests | digests | digests; recompute only the already-disclosed baseline aggregates (T1–T8, confusion, request counts) |
| smoke, bench-a, bench-b | – | listed patients | – |
| val-a (incl. shadow), analyze-val-a | – | yes | – |
| hpi-audit (counts only) | yes | yes | – |
| wm-fit | fit | quality | – |
| inventory | yes | yes | – |
| leakage | fit | evaluate | – |
| val-b, analyze-val-b | – | yes | – |
| test-a, analyze-a, test-a2, analyze-a2 | – | – | admitted variants; official rows |
| test-b, analyze-b | – | – | admitted variants; leakage evaluation on test |

- Admissions: an append-only ledger and exclusive-create markers in $TIM_ADMISSION_DIR, a fixed path recorded in prereg/protocol.json. A different path is refused, so changing TIM_WORK_DIR cannot reset admission. Each marker is keyed by protocol hash, stage, variant and arm.
- Interruption: episodes are independent singletons. Completed episodes go to an immutable per-patient log with digests. A resumed job continues at the first incomplete patient and never re-evaluates a completed one. The ledger records every interruption and resumption.
- Approvals:
  - approvals/<stage>.json, committed by Daniel, binds the protocol hash, addendum hash, config hash, code commit, job list (stage, variant, arm, split, patients) and GPU-hour budget. A GPU stage refuses to start unless all of them match.
  - tim approval-request <stage> drafts approvals/<stage>.request.json, with estimates taken from bench measurements.
  - Smoke has a cumulative cap of 0.5 GPU-hour, tracked in the ledger. Smoke beyond the cap needs approval.

## 3. Phase A: commit guard

3.1 Upstream behaviour (verified at 3f435a10, src/environment/environment.py)
- step() with step > max_steps (13): complete(invalid=True) before any parsing.
- Parsing fails when len(token_mask) > 4,096 (completion overflow) or the regex misses: complete(invalid=True).
- A Test action calls get_test_result, which updates request_count, requested_tests and provided_tests (unless the result is "not available.\n") and sets error for unavailable tests. Only then is the observation length checked; if it would exceed 4,096 tokens, the episode completes invalid without the observation, and the test has already been charged.
- Unknown action, test or diagnosis: an error observation is appended and the episode continues.
- The shim's step wrapper records request_events before calling upstream.
- Availability sentinel: the dataset writes "not available.\n" for missing labs and imaging, the replication maps missing physical exams to it, and get_test_result compares against it exactly.

3.2 Components
- G1 validity guard: a malformed proposal, unknown action, unknown or repeated test, or unknown diagnosis becomes the highest-scoring valid action.
- G2 budget commit: before the environment acts, at t = 13 or on completion overflow, the action must be a diagnosis. Observation overflow is not rescued; upstream terminates and G3 applies.
- G3 hypothesis fallback, at finalization only: if the episode ended without a diagnosis, the finalized prediction is the most recent hypothesis that is one of the four diseases.

3.3 Decision table for decision turn t with greedy completion c. L0 = trajectory tokens before c. Limits: MAX = 4,096 trajectory tokens, CTX = 4,352 model context, COMP = 256 completion tokens.
1. t > 13: upstream terminates. Unreachable when G2 is on.
2. Parse c with upstream's regex, giving proposal P or "malformed".
3. G2 (if on), when t = 13 or L0 + len(c) > MAX:
   - P is a valid Diagnosis and nothing overflows: execute c.
   - Otherwise candidates = the four diagnoses.
4. Else G1 (if on), when P is malformed, an unknown action, an unknown or already-requested test, or an unknown diagnosis:
   - P was a Diagnosis with an invalid label: candidates = the four diagnoses.
   - Otherwise: candidates = untried tests ∪ diagnoses (only diagnoses if no test is untried).
5. Else the Phase-B planner (if on), when the action is a Test and t < 13 (§4.4).
6. Replacement fit:
   - Use P's thought.
   - If the replacement breaks MAX, CTX or COMP, use the fixed thought instead: "I will choose the next step." for G1 and the planner, "I must now give my final diagnosis." for G2.
   - If it still does not fit, execute c unchanged and record replacement_failed.
7. Execute through upstream step() unchanged.
8. Finalization:
   - raw_prediction = upstream's diagnosis, or null.
   - If raw_prediction is null and G3 is on: finalized_prediction = the most recent valid hypothesis, or null.
   - Otherwise finalized_prediction = raw_prediction.
Precedence is G2 > G1 > planner. Invariant: on any turn where nothing triggers, the executed completion is the greedy completion byte for byte, so with identical histories V3 equals V0 until its first trigger.

3.4 Scorer: one definition for every candidate
- Score = the teacher-forced log-probability of the complete canonical continuation after "Action:", terminator included.
  - Test(x): " Test\nAction Input: {x}\nObservation:", with x exactly as written in prompts/decision_agent.txt.
  - Diagnosis(d): " Diagnosis\nAction Input: {d}" followed by <|eot_id|>, with d lowercase as listed in the prompt.
  - No aliases. Ties go to the fixed order: configured tests, then diseases.
- Context = the latest decision prompt's token ids exactly as vLLM saw them (trajectory.output.prompt_token_ids) + tokenizer("Thought: {thought}\nAction:", add_special_tokens=False). Each continuation is tokenized once with add_special_tokens=False.
- Backend: a separate Hugging Face instance with the same merged weights (upstream load_model, PeftModel, merge_and_unload; bf16), loaded after the vLLM engine. Settings: eval mode, torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8, float32 log-softmax on scored positions. Two code paths: full-sequence forward (the reference) and prefix KV reuse with batched continuations (fast). The vLLM engine is never used for scoring.
- Checks:
  - CPU: the scoring core equals brute force on a tiny torch model.
  - Bench (GPU): fast path = reference within 1e-3, and the agreement rate between the HF argmax next token and vLLM's first generated token is reported.
- Limitation: this is the policy's likelihood, not a calibrated posterior; verbalizer and length effects remain.
- Scorer errors: the turn executes c and an event scorer_error is logged. On validation, fix and re-run. On test, report them as protocol deviations and analyze the run as it is.

3.5 Variants and controls
- V0: guard off, instrumented. V1: V0 rows + G3 (CPU). V2: G2 + G3. V3: G1 + G2 + G3.
- The Phase-A method is V3, fixed in advance. Validation decides go / no-go only; nothing is selected on it.
- Rows keep raw and finalized predictions, so V2-raw and V3-raw (the online guards without G3) come from the same runs.
- Analytic controls on the comparator's nonanswers, from counts: uniform random fallback (expected value) and training-majority fallback.
- Reported separately: recovery yield (correct recoveries ÷ all baseline nonanswers), accuracy among newly answered episodes, and regressions (baseline-correct patients that become wrong; zero for V1 by construction, asserted).

3.6 Integration
- tim.runner calls replication.gpu.evaluate under unittest.mock.patch.object(replication.gpu, "_evaluate_with_model", guarded_evaluator), with seed stream_seed(269, split, "policy") exactly as replication.pipeline._evaluate passes it. Engine creation, adapter merge and sampling are the baseline's own.
- guarded_evaluator mirrors replication.shim._evaluate_with_model and builds env = shim.prepare_environment(cfg); env.__class__ = guarded_class(type(env), guard, planner, scorer).
- guarded_class overrides run() (stores the engine, scorer and current patient, then calls super().run) and step() (applies §3.3 before super().step and records the termination reason).
- Private rows add: variant, raw_prediction, finalized_prediction (the "prediction" used for metrics), prediction_source, termination (diagnosis | step_limit | malformed | length_completion | length_observation), n_turns, events [{t, type, from_kind, to_kind}], scorer_error_count, replacement_failed_count.

## 4. Phase B: retrospective chart-availability prediction and reranking

4.1 Labels: y(p, x) = 1 if test_results[x] != "not available.\n" (the retrieval rule exactly), else 0.

4.2 Models. All probabilities are clipped to [ε, 1−ε] with ε = 0.001.
- Learned:
  - Features: TF-IDF of the HPI summary the agent sees (word 1–2-grams, min_df 3, max_features 20,000, sublinear tf).
  - One L2 logistic regression per test. C is chosen once from {0.01, 0.1, 1, 10} by 5-fold CV log-loss on train (fold seed 0).
  - Then fit on bootstrap resamples of the 1,920 training admissions with seeds 0, 1 and 2.
  - A test with fewer than 10 positives or negatives uses its base rate.
  - Stored privately, never exported.
- Marginal: the train base rate π_x. Uniform: 0.5.
- Privileged chart-availability oracle: y itself, read through an accessor that exposes availability bits only.

4.3 Controlled error
- One uniform draw per (seed, split, patient, test): u = first 16 hex digits of sha256("tim-mask|seed|split|patient|test") / 16^16, shared across every α and T. ỹ = 1 − y if u < α, else y, so masks are nested across α.
- Nominal posterior q = (1−α)π / ((1−α)π + α(1−π)) if ỹ = 1, else απ / (απ + (1−α)(1−π)); clipped.
- Confidence: p_T = clip(σ(logit(q) / T)) with T = 1 (nominal) and T = 0.25 (overconfident). Rankings and decisions at the 0.5 threshold are identical across T.
- Grid: α ∈ {0.1, 0.3} × T ∈ {1, 0.25} × noise seeds {0, 1}.
- Tested identities: α = 0 gives the oracle for every T; α = 0.5 with T = 1 gives marginal.
- Nominal calibration rests on the train base rate; report measured calibration rather than assume it.

4.4 Planner (fixed now)
- Applies on top of V3 when the action is a Test and t < 13. U = untried tests.
- If max over U of p(x) < θ_stop = 0.05: commit to the highest-scoring diagnosis (event wm_stop).
- Otherwise: choose argmax over U of [score(Test(x)) + β·log p(x)], with β = 1, using the §3.4 scorer on test candidates (event wm_rerank when the test changes). Ties as in §3.4.

4.5 Arms and contrasts
- Arms: P0 = V3, B-uniform, B-marginal, B-learned-s0/s1/s2, B-oracle, and the 8 corruption arms.
- Family B (Holm, inferential):
  - B-learned-s0 − B-uniform: availability information beyond the reranking mechanism.
  - B-learned-s0 − B-marginal: patient-specific prediction beyond test prevalence.
  - B-oracle − B-uniform: privileged availability under the same mechanism.
- Descriptive only: every arm − P0 (whole-system effect), learned seeds 1–2, and the corruption arms (pointwise intervals).
- Conclusions are limited to this fixed planner at the measured precision. Perfect information does not make the heuristic optimal, so a non-positive oracle contrast does not show that availability has no value.

4.6 World-model quality is reported per arm and split on separate axes: corruption level α, ECE (10 bins), AUROC (discrimination), and Brier with Brier skill vs marginal as a descriptive summary only.

4.7 Leakage: two multinomial logistic regressions to the diagnosis under the same frozen protocol, one on HPI TF-IDF and one on HPI TF-IDF plus the 12 availability bits. Report T5 on val during Phase-B prep and on test only in analyze-b. A bits-only classifier is descriptive. A weak probe cannot certify that leakage is absent.

4.8 Initial-state action/outcome inventory (stage-1 item, narrowed): one record per patient and test at t = 0, with the HPI summary, test, result text or "UNAVAILABLE", label, source "mimic_real", plus the docs/env-and-tuples.md §4 fields. Private JSONL, train and val only. These are not sequential-history training tuples. The aggregate card holds counts by split × test × availability, per-class availability rates and word-count quantiles.

## 5. Preregistration and protocol
1. prereg/protocol.json is committed before bench-a. It freezes:
   - the primary method (V3) and comparator (the official primary rows);
   - every variant, arm, constant, seed, contrast, family, test and wording rule in §3–§6;
   - the admission path, the approver name ("Daniel Escalante") and the G3-on-test disclosure.
2. After the freeze, code fixes need an amendment note (what changed and why, before any test access). Scientific choices are fixed.
3. Addendum A, from validation only, before test-a:
   - premise check: if V0 has fewer than 10 val nonanswers, stop and report;
   - the shadow-control result;
   - the sensitivity simulation;
   - bench-a measurements and the updated budget.
4. Addendum B, before test-b: world-model file digests, val quality, bench-b measurements. Phase-B choices were frozen in step 1, so Phase-A test results cannot change them.
5. Shadow control, part of val-a:
   - Run V0 and V0+shadow on all 240 val patients. The shadow run calls the scorer on every decision turn with the full candidate set and discards the result.
   - Both runs use fresh, identically configured engines and identical request order.
   - Generated token ids must be identical for every hypothesis and decision generation. Any mismatch blocks test-a until an amendment explains it.
6. V0-rerun on test, part of test-a: 240/240 identical finalized predictions to the official rows permits an unqualified comparison. Otherwise report k/240, add V0-rerun as a second comparator and qualify every Phase-A claim.
7. Sensitivity simulation, in addendum A:
   - Use val per-disease counts of improved / regressed / unchanged patients (V3 vs V0 and V1 vs V0).
   - Simulate the primary test 10,000 times at the test class sizes 96 / 65 / 25 / 54.
   - Report power at the val effect and the minimum detectable effect at 80% power. For orientation: about 6.4 points at equal 10% discordance, about 9.1 at 20%.
   - Interpret null results against these figures.

## 6. Statistics
- Estimand: ΔT5 in percentage points over the 240 test admissions, conditional on the trained policy (seed 269). With w_i = 100 / (4 n_class(i)) and d_i = correct_method − correct_comparator, ΔT5 = Σ w_i d_i.
- Primary test, α = 0.05, the only uncorrected test: V3 − official, two-sided paired sign-flip randomization test.
  - Null: within each patient the two outcomes are exchangeable, i.e. each nonzero d_i is equally likely ±.
  - Statistic: S = Σ w_i d_i.
  - p-value: exact enumeration when ≤ 20 patients are discordant. Otherwise 100,000 draws, p = (1 + #{|S*| ≥ |S_obs|}) / (B + 1), seed stream_seed(269, "test", "tim-permutation:<comparison>"). Ties count as extreme.
  - This is exact for the exchangeability null but not for the weaker null ΔT5 = 0 when diseases move in opposite directions. Claims are worded accordingly.
- Interval: 95% class-stratified paired percentile bootstrap (replication.stats.paired_bootstrap; pairs resampled together; 10,000 resamples; seed stream_seed(269, "test", "tim-bootstrap")). A test asserts its point estimate equals Σ w_i d_i.
- Family A (Holm, α = 0.05): V1 − official; V3 − V1.
- Descriptive, intervals only: V2 − V1, V3 − V2, raw vs finalized, the analytic controls, per-class recalls, T6–T8, requests, termination counts, recovery yield, regressions, McNemar (never used for a decision).
- Family B: as in §4.5. The families are separate; no global "some variant wins" claim.
- Wording:
  - If p ≥ 0.05 (Holm-adjusted inside a family) or the interval includes 0, the report says "not statistically significant".
  - With one policy seed, between-seed variation is unassessable, and the report says so.

## 7. Outputs
- Phase A deliverable: results/credentialed/phase-a/{metrics.json, REPORT.md}. REPORT.md contains:
  - the framing and disclosure from §0, and the baseline recomputation;
  - the ladder table with raw and finalized predictions, plus the analytic controls;
  - recovery yield, regressions, termination breakdown, cost and requests;
  - the primary and Family-A results with plain significance statements, and the sensitivity analysis;
  - limitations: previously examined benchmark, one policy seed, verbalizer effects, no clinical-safety inference, person-level overlap unassessable (no subject_id), deviations D-1 / D-3 / D-5 / D-6, Blackwell runtime.
- Phase B: results/credentialed/phase-b/{metrics.json, REPORT.md, wm_quality.json, leakage.json, inventory_card.json, hpi_audit.json}.
- Deferred, optional after both phases: WRITEUP.md, SLIDE.md, results_table.tex, figures.
- Everything is rendered from metrics.json; no hand-typed numbers. results/synthetic/ mirrors this structure with a SYNTHETIC banner.

## 8. Layout and commands
    cdm-extension/daniel-escalante/
      SPEC.md README.md DECISIONS.md pyproject.toml uv.lock .env.example .gitignore
      configs/variants.yaml   schemas/*.json   prereg/ (protocol.json, addenda, amendments)
      approvals/   (requests drafted by tim; approvals committed by Daniel)
      tim/ settings access events publish actions scoring guard fallback planner
           availability leakage inventory audit runner analysis render fixture cli
      scripts/ run_tim.py tim.sh gpu.slurm cpu.slurm submit_slurm.py
      tests/   results/credentialed/   results/synthetic/
- Commands, run as scripts/tim.sh <command>:
  - CPU synthetic: test, fixture, all.
  - Phase A: validate, smoke, bench-a, val-a, analyze-val-a, addendum-a, test-a, analyze-a; optional test-a2, analyze-a2 (V3 on decision_only, no_cost, incorrect_minus1, zero_shot).
  - Phase B: hpi-audit, wm-fit, inventory, leakage, bench-b, val-b, analyze-val-b, addendum-b, test-b, analyze-b.
  - Always available: approval-request, publish, check-git.
- submit_slurm.py --stage <stage> [--dry-run] refuses a GPU stage without a matching approval (smoke excepted, within its cap). Every GPU job requests partition hpg-rtx6000, GRES gpu:rtx_pro_6000:1, 4 CPUs and 16G, so all runs share the baseline's hardware. A stage's jobs are submitted together and spread across the course accounts and QOS listed in SLURM_ACCOUNTS, as many at once as each QOS allows. CPU stages go to a burst QOS. All logs stay private.

## 9. Milestones and budget (Phase A is the critical path)
- M0: scaffold, schemas, settings.
- M1: Phase-A code, CPU tests first: access, events and publish with canaries, actions, scoring core, guard decision table, fallback, runner, analysis.
- M2: synthetic end to end for Phase A.
- M3: commit protocol.json; run validate and smoke.
- M4: bench-a → approval → val-a → analyze-val-a → addendum A.
- M5 (target Nov 20): approval → test-a → analyze-a → Phase-A REPORT published.
- M6: Phase-B code and CPU tests (availability, planner off ≡ V3, corruption identities, leakage, inventory, audit), then the synthetic end to end.
- M7: hpi-audit, wm-fit, inventory, leakage → bench-b → approval → val-b → addendum B.
- M8 (provisional Nov 30): approval → test-b → analyze-b → published.
- M9: optional deferred outputs, PR, merge.

Provisional GPU-hours on RTX PRO 6000. Unscored greedy evaluations logged 24–33 min per 240 admissions; scored runs are unmeasured until bench. A stage's jobs run concurrently, so its wall time is about that of its longest job.

| Stage | Runs | GPU-h |
|---|---|---|
| smoke | 2 val patients | ≤ 0.5 cumulative |
| bench-a, bench-b | 24 val patients (6 per class, ordered by sha256("bench|patient_id")) × 2 variants | 0.15 each |
| val-a | V0, V0+shadow, V2, V3 | 2.2 |
| test-a | V0-rerun, V2, V3 | 1.7 |
| test-a2 (optional) | V3 on 4 other arms | 2.7 |
| val-b | B-uniform, B-learned-s0, B-oracle | 1.7 |
| test-b | 14 arms | 7.7 |

Core total is about 14 GPU-h, or about 17 with test-a2. Approvals use bench measurements, and any margin is spent only on validation or infrastructure recovery, never on test reruns.

## 10. Acceptance criteria
Before merge:
1. scripts/tim.sh test passes on CPU, and the replication's tests pass unchanged. Required tests:
   - V0 rows identical to replication.shim._evaluate_with_model rows on the fixture.
   - Every decision-table path: t = 13; completion overflow; observation overflow → G3; malformed; unknown action, test and diagnosis; repeat; no untried tests; replacement_failed; 256-token truncation.
   - The untriggered-turn invariant; planner off ≡ V3; V1 never changes a non-null prediction.
   - Scoring core vs brute force on a tiny torch model, and ties.
   - Access-table enforcement on CPU paths, admission persistence across TIM_WORK_DIR changes, resume without re-evaluation, approval binding.
   - Corruption identities and nested masks.
   - Randomization test vs brute force; bootstrap point estimate = Σ w_i d_i.
   - Canary scan, read-only baseline, check-git.
2. validate reproduces T5 = 73.08502492877493 from the official rows.
3. Synthetic tim.sh all runs end to end, and a second run is byte-identical.
4. Every credentialed stage that ran has its receipt, ledger entries and a committed public bundle. The shadow control passed before test-a. V0-rerun agreement is reported.
5. analyze-* re-run on the cluster from the private rows reproduces metrics.json byte for byte.
6. The Phase-A and Phase-B REPORTs exist, are rendered from metrics.json and state significance plainly.
7. check-git passes, git holds only schema-valid aggregates, and there is no attribution text anywhere.

After merge:
8. main contains the merge commit, and the folder README run log is current.

## 11. Negative-result plan
- Premise fails on validation: stop Phase A before test access, publish the val evidence and move to Phase B.
- V3 − official not significant: report the interval, the ladder (raw and finalized), recovery yield against the analytic controls, regressions, termination changes and the sensitivity analysis, and name which component helped or hurt.
- V1 significant but V3 − V1 not: the hypothesis agent already held the missing answers, and the online guards added nothing measurable.
- Phase B shows no gain: report a finding about this fixed reranker at the measured world-model precision, with the corruption arms and the oracle contrast, and no general claim about availability.
- Never re-tune on test. Label any post-hoc analysis exploratory.

## 12. Next, not in this build (ICML 2027 path)
1. Content prediction p_phi(o | h, a): labs first, then imaging. Train on train-split data with sequential histories from train rollouts.
2. An imagined-outcome planner (expected information gain per dollar), compared on the same footing.
3. Causal claims in Edwin's synthetic environment, which has complete outcomes and availability masks generated independently of diagnosis.
4. An uncertainty-aware fallback.
5. More policy seeds (about 105 GPU-h each) and the Llama-3.1-8B-Instruct backbone (about 110 GPU-h).
6. ICML 2027 deadline: not announced; the planning estimate is around January 22, 2027.