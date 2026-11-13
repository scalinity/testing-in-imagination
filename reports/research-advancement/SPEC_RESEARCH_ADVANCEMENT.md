# Testing in Imagination: Research Advancement Specification

Version: 3.1 — single owner, unconstrained compute, autonomous execution through completion
Revision date: November 13, 2026
Repository: scalinity/testing-in-imagination
Owner: Daniel Escalante, working through one executing agent
Advisor: Professor Xuefeng Liu
Course: CAI6734, University of Florida
Study ID: TII-CEE-1
Study directory: reports/research-advancement/ on branch tii-cee-1/2026-11-13
Status: Canonical successor to versions 2.0 and 3.0. Execution plan, not a completion claim. Nothing in it has run.

Version 3.0 fixed one contribution: an agent that plans with imagined test results should conserve
expected evidence. Version 3.1 keeps that contribution and every validity protection, and changes
four things. Compute no longer shapes the design; budgets are estimates with their basis. The study
has one owner; nothing depends on Hugo Resendez or Edwin Salcedo, whose existing work is used as it
stands in the repository and credited. Quality comes before any deadline. And the executing agent
carries the study to completion under one signed authorization, without stopping at gates.

Items marked DEFAULT D1 to D15 began as assumptions. Daniel confirmed or changed each one on
November 13, 2026, and §2.7 records the result. Repository observations refer to read-only inspections on November 12
and 13, 2026 and must be re-checked when execution starts. Authoring this specification does not
execute it. A current read-only instruction always takes precedence.

## 1. Mission, thesis, and claims

### 1.1 Mission

Preserve the original project: a world model predicts what a diagnostic test would reveal, and the
agent uses those imagined outcomes before ordering real tests. Add the piece the project was missing:
imagined outcomes must also discipline how the agent reads the real result and when it commits.
Do not replace this with a larger-backbone project, an unrelated benchmark, or a new platform.

### 1.2 Thesis

An agent that plans with imagined test results should conserve expected evidence: its reactions to
the results it imagines must average to what it believes now. The violation can be measured without
labels at every decision point. It separates three ways an agent misuses evidence: drift, wrong
scale, and redundancy. Each has its own repair: a projection for drift, and a match to the test's
information budget for scale and redundancy. An exact identity says in advance how much a repair
is worth and which error blocks it.

Central question: "How much of a diagnostic agent's regret comes from choosing tests, from reading
results, and from committing; which reading regime are real agents in; and does enforcing
conservation of expected evidence with a world model repair the reading and commitment parts?"

### 1.3 Why this direction, from our own evidence

- The largest recorded gain is answer recovery: test T5 73.09 → 84.45 with the hypothesis fallback
  (V1). The online guards (V3, 81.83) did not improve on V1. V1 is the bar for any successor.
- Perfect availability information changed behavior (unavailable requests 203 → 66) with no
  detectable T5 change for the fixed planner on validation.
- Exploratory check H: of the 61 relevant, provided results the oracle arm added, 38 arrived while
  the hypothesis already held the label; the hypothesis changed 7 times, 3 toward the label.
- The content model's planning score is not mutual information: results are drawn from one model
  and beliefs updated with another. In the committed CPU aggregate, 4 of 10 predicted-gain deciles
  are negative, which a coherent Bayesian score cannot be.
- Lab panels moved the structured posterior from NLL 0.690 to 0.637 after correction: little
  information beyond the history for this four-way task.

These are recorded observations, not reproduced here. Together they say that test choice is not the
binding constraint for this agent and that the read-and-commit loop is where to look.

Sandbox evidence behind the design, not project evidence (PROOF_GAIN_IDENTITY.md, cee_check.py): in
a re-implementation of the synthetic twin's parameters with an oracle world model, the projection
repaired both drift readers, barely helped a reader that double counts correlated evidence, and
hurt over- and under-reactive readers; matching the true information budget brought all five
reader families close to the Bayes reference; a budget twice too large was harmful.

### 1.4 What is and is not claimed as new

Not new, and never to be presented as new:
- Imagined outcomes guiding acquisition (generative surrogate models, ACTMED, Uncertainty of Thoughts).
- Risk-based acquisition with calibrated stopping (CaRDI) and risk-constrained stopping (CROS, Wu and Rus).
- Weighting model error by its effect on decisions (value-aware model learning and its calibrated
  variant; objective mismatch in model-based reinforcement learning).
- Using a model only where it is reliable (masked model-based actor-critic, stochastic ensemble value
  expansion, safe policy improvement with baseline bootstrapping, Selective Foresight).
- That language models update beliefs incoherently and misuse foresight (martingale tests of
  in-context learning, the Martingale Score, Bayesian-inference audits, Qian et al.).
- Correcting a language model's probabilities toward the expected-evidence constraint
  (ArbitrageForecaster, Paleka et al.), and training a model to respect the martingale property
  (Kim et al.). Avoiding the incoherent pairing altogether (BED-LLM).
- The mathematics: minimum-KL projection onto marginal constraints, iterative proportional fitting,
  adjusting a classifier to a known class distribution, tempering, the chain rule for KL, and the
  classical result that incoherent forecasts are dominated.

Claimed, each subject to the status recorded in PRIOR_WORK_MATRIX.md:
1. The contract. Conservation of expected evidence (CEE) stated for a separate world model, a
   black-box reader, and a commit rule, with label-free measurements of drift, evidence scale,
   redundancy, and responsiveness at every decision point.
2. The accounting. An exact identity that splits the gain from a coherence correction into
   incoherence, prior calibration, and reader-weighted world-model error, estimable on real charts
   from factual outcomes only; and its multi-step form, which isolates accumulated drift.
3. The repairs. CEE-Read, a projection that removes evidence-independent drift for the "simulate a
   result, then re-query the reader" pairing, which is the practical one when results are rich text
   and the reader is a trained policy. CEE-Scale, a separately named component that sets the size
   of the reader's reactions by matching an information budget for the test, for readers whose
   reactions are too strong, too weak, or redundant. Coherent value of information and a commit
   rule follow from either.
4. The regime map. The audit run as a general diagnostic on any simulate-and-re-query agent: where
   language-model readers of different sizes and families, published recipes on public data, and a
   trained clinical agent sit in the drift-by-scale plane, and whether the label-free audit
   predicts the realized effect of each repair.
5. The validation. Exact counterfactual regret and component attribution in the synthetic
   environment; public complete-record datasets; MIMIC evidence replay with policy seeds, a
   cross-fitted cohort, and a second backbone; one disclosed one-shot test.

Scope fixed in advance. The projection repairs drift only; it is expected to hurt readers with the
wrong scale, and those cases stay in every report as negative controls. CEE-Scale is only as good as
its budget; an overestimated budget is harmful; it is never chosen by minimizing incoherence, which
selects an agent that never updates. Neither repair is guaranteed to improve accuracy.

### 1.5 Three outcomes kept separate

- Engineering validity: the code computes the declared quantities and enforces the protocol.
- Empirical benefit: the intervention improves a predeclared objective against strong controls.
- Publication evidence: the contribution is differentiated, explained, and supported beyond its
  development setting.

Passing tests establishes neither the second nor the third. A negative or inconclusive experiment
completes a milestone when it answers its question honestly. No accuracy gain or acceptance is
promised or required.

### 1.6 The paper under each outcome, and its headline

The paper is framed as method plus mechanism (DEFAULT), built so the mechanism stands alone.

| Outcome | Strongest headline the evidence would support |
|---|---|
| A. Drift dominates and projection helps in the twin, on public data, and in replay | Agents that imagine results do not conserve expected evidence; one vector of C numbers per decision restores it, and an exact identity predicts the gain. |
| B. Scale or redundancy dominates | Language-model readers mis-size evidence rather than drift: projection alone is harmful exactly where the audit says it will be, and matching an information budget repairs it. |
| C. Violations are present and no repair helps | The identity names the term that blocks the gain, prior calibration or world-model error: imagination fails for a measurable reason. |
| D. Readers barely respond | Imagined results cannot help a reader that does not move. Responsiveness near zero explains the availability null; committing early costs nothing. |
| E. Repairs work in the twin and on public data, not on charts | Scope: what chart-limited evidence can and cannot support, with the exact attribution as the main finding. |

Reported under every outcome: the regime map, the agreement between predicted and realized gain,
and the exact attribution of regret to selection, reading, and commitment.

### 1.7 The figure

Figure 1 is fixed now so that every experiment feeds it.

- Panel A, the regime map. Horizontal axis: evidence scale relative to the information budget,
  tau_B on a log axis, with 1 meaning the reader's reactions carry exactly the information the test
  has. Vertical axis: location incoherence in nats per decision after scaling. One marker per
  reader and setting: simulated families as open markers, language-model readers as filled markers
  sized by parameter count, the trained policy's seeds and second backbone as stars, public
  datasets by marker shape. From each marker an arrow whose length is the exact or realized change
  in log loss under the pre-registered repair. The region where projection alone is harmful, taken
  from the exact twin results, is shaded.
- Panel B, the accounting. Realized gain against the gain predicted by the identity, one point per
  reader-by-setting cell with intervals, and the diagonal.
- Panel C, the attribution. Share of exact regret from selection, reading, and commitment for the
  twin agents, raw and repaired.

No panel is drawn with data that do not exist. A panel whose experiment was blocked says so.

### 1.8 Out of scope for this version

Policy fine-tuning or distillation with the CEE objective; multi-step lookahead for selection; a
parametric model of imaging content; a new benchmark; any deployment or safety claim.

## 2. Canonical inputs, verified starting state, and changes

### 2.1 Inputs

The study directory is reports/research-advancement/ on branch tii-cee-1/2026-11-13. Read these
completely, all in that directory: SPEC_RESEARCH_ADVANCEMENT.md (this file),
KICKOFF_RESEARCH_ADVANCEMENT.txt, AUTHORIZATION.md, WATCHER_DESIGN.md, PRIOR_WORK_MATRIX.md,
PROOF_GAIN_IDENTITY.md with cee_check.py and cee_check_output.txt, REVIEW_AND_REDESIGN.md, and
keeper/KEEPER_INSTALL.md.

Also read the independent review, REVIEW_HANDOFF.md. It is not on this branch. Commit
91b21ae2b0127a42cfd406ec6a51339dc11c19a4 on origin/docs/research-advancement-v2 added it at
reports/research-advancement/REVIEW_HANDOFF.md; read it with
git show 91b21ae:reports/research-advancement/REVIEW_HANDOFF.md, fetching that branch first if the
commit is missing locally. Version 2.0 of this specification and its kickoff sit at the same commit
and directory and stay there as records. Version 3.0 was delivered to Daniel on November 12, 2026,
was never committed, and is superseded.

Also inspect: AGENTS.md files if present, ONBOARDING.md,
README.md, REPRODUCTION.md; cdm-extension/daniel-escalante/SPEC.md, SPEC_CONTENT_WM.md, DECISIONS.md,
prereg protocol and amendments A1–A7 and C1, approvals, schemas, results aggregates; tim/content.py,
labs.py, runner.py, scoring.py, guard.py, fallback.py, approvals.py, access.py, analysis.py and
tests; extension/ environment, world models, planners, experiments; PRs #1, #7, #8, #9, #10 and
their discussions; reports actually present.

### 2.2 Anchors observed on November 12, 2026

| Resource | Identity | State |
|---|---|---|
| main | e5ad93fad5345f211cedc05c3dc491b101c63fa5 | unchanged since the review |
| PR #9, content implementation | acf5254bb624ed881f864e84149fc4d983dc048a | open |
| PR #7, content design | 75c50a725655ad4cbe69d4ab79677c40ed4b309e | open |
| PR #1, Edwin Salcedo's synthetic study | df467eadd07fe2bc761b618dafce8ed9f198641a | open |
| docs/research-advancement-v2 | 91b21ae2b0127a42cfd406ec6a51339dc11c19a4 | branch, no PR seen |
| PR #8, PR #10 | merged as 3a564dd…, e5ad93f… | merged |
| tii-cee-1/2026-11-13 | task branch created November 13, 2026 from main at e5ad93fad5345f211cedc05c3dc491b101c63fa5 | local only, not pushed |

A local clone may lag the remote. Remote absence does not prove there is no unpushed work.
The synthetic parameters in extension/envs/synthetic_diagnosis/profiles.py at df467ea were re-read
on November 13 and match the check script.

### 2.3 Starting findings, not newly reproduced

| Observation | Required interpretation |
|---|---|
| Test T5: original 73.09, V1 84.45, V3 81.83; V3 − V1 = −2.62 [−6.16, 0.76] | V1 is the reference to beat. V3 is not shown worse than V1. |
| Validation oracle − uniform availability: −0.11 [−2.56, 2.38]; requests changed | No detectable benefit for this fixed planner and cohort. Not equivalence. |
| Dose response: 4.384 GPU-hours; slope −2.46 [−7.28, 2.26] | Completed; flat is not equality. |
| Check H counts (61 added results, 38 already held the label, 7 changes) | Exploratory; reads stored hypotheses, not the policy's belief. |
| Content CPU: premise 68/240 admissions, 158 lab requests; posterior NLL prior 0.68999, uncorrected 2.48502, A+B 0.63720, tau 13 in all folds; ECE 0.04554 prior, 0.07046 A+B | Developmental, partially out of fold. Quality gate unresolved (REVIEW-001). |
| Predicted-gain deciles: 4 of 10 negative | The score is an entropy-change surrogate under an incoherent pair. |
| Availability bits alone predict the diagnosis: T5 43.13 on test against 25 by chance | Chart presence encodes the completed workup; privileged and a benchmark-validity fact. |
| Compute logs: about 0.458 GPU-hours per 240-admission arm; primary policy training 25 h 5 min on one RTX PRO 6000; four training arms 103.6 GPU-hours | Basis for §14.3. |
| Clinical test split examined by Phases A and B; no subject identifiers | A new freeze does not create an untouched test set. |

Do not mix validation and test endpoints. Do not discard completed studies. Do not declare all
synthetic results leaked: the same-seed likelihood sample and the offset-seed utility sample are
adjudicated separately.

### 2.4 What changes from version 3.0

| Version 3.0 element | Disposition in 3.1 |
|---|---|
| GPU caps per tier that sized the design (K = 8, one policy, 240 admissions) | Removed. Design first, then cost: K = 32, five policies of the primary recipe, a cross-fitted cohort of 2,160 admissions, a second backbone (§10, §11, §14.3). |
| Approval STOP at G2a, G3, G4 | Replaced by one signed AUTHORIZATION.md. Gates are decision rules the agent evaluates and records (§14, §20). |
| Three owners and cross-lane interfaces | One owner. Interfaces kept as internal module boundaries (§16). |
| Prior-work verification as a teammate's task | Done by the specification's author on November 12 and 13 and delivered as PRIOR_WORK_MATRIX.md; maintained by the agent (§7.2). |
| Projection as the only repair; tau_hat as the scale diagnostic | CEE-Scale added as a separately named component; tau_B, the budget-implied scale, becomes the primary scale diagnostic (§5.5, §5.11). |
| Single-step guarantee only | Multi-step accounting identity added and tested exactly (§5.12, E1-D). |
| E2 unspecified models; E7 one unspecified dataset | Reader panel of at least eight models; four public datasets (§10.2, §10.4). |
| Seeds and second backbone deferred | Planned: E8, E9, E10 (§11.4). |
| Schedule "five weeks from November 12" | Real dates to a venue deadline with a go or no-go date (§17). |
| "Cros" written as an author | Corrected: CROS is the method; the authors are Wu and Rus. |

### 2.5 What changed from version 2.0, carried forward

Authority, protected data, frozen studies, artifacts, the arm-component contract, statistics, and
the integrity repairs are kept. The clinical selector-by-reader factorial, the open method search,
and the ensemble override stay cut; disagreement gating survives as a baseline. The legacy content
study's GPU arms (bench-c, val-c, test-c) stay deferred with their design preserved and are not run
under this study ID.

### 2.6 Prior-work positions that bound the claims

PRIOR_WORK_MATRIX.md is canonical. Positions that bind every report: ACTMED already simulates test
results with a language model and takes the posterior from the model; BED-LLM already names the
incoherence of that pairing and avoids it; Paleka et al. already patch a language model's forecasts
toward expected-evidence consistency and report their smallest consistency gain for that check and
no clear accuracy effect; Kim et al. already repair martingale
violations by fine-tuning; CaRDI already uses calibrated risk for question selection and stopping
with one structured model; CROS already controls stopping risk on an LA-CDM-style agent and reports
that continuing can raise error; LDTL already trains a planner on the reader's realized gain;
value-aware model learning already weights model error by decision impact. None of these may be
described as absent from the literature. The bounded search of November 13 found no paper that
projects imagined posteriors under a separate world model in sequential acquisition; that is a
search result, not a proof.

### 2.7 DEFAULT items as confirmed or changed by Daniel on November 13, 2026

| # | Decision in force | Status | Where it is used |
|---|---|---|---|
| D1 | Course checkpoint on Friday, December 4, 2026 | Confirmed | §17 |
| D2 | GPU-hours are unconstrained from November 13 to about December 13, 2026, with at least 8 GPUs at once; what is available afterward is unknown | Confirmed | §14.3, §17 |
| D3 | Jobs run at most 24 hours; longer work is split into resumable parts | Confirmed | §20.5 |
| D4 | Account cai6734, GPU QOS cai6734, CPU burst QOS cai6734-b, partition hpg-rtx6000, study root /blue/cai6734/daniel.escalante/private/tii-cee-1 | From the repository's records; UNVERIFIED against the live cluster until the launch check of §7.4d | AUTHORIZATION.md |
| D5 | Whether logged trajectories store full prompts is unknown; §7.4a settles it | Confirmed as an open check | §7.4, §10.1 |
| D6 | E6 is the one additional, disclosed use of the already examined 240 test admissions. No second run and no further use | Changed: stricter | §11.3 |
| D7 | The reader panel in AUTHORIZATION.md: nine ungated models with US-origin bases. Second backbone: ibm-granite/granite-3.3-8b-instruct | Changed: models with a base from a country of concern, Qwen among them, are not permitted on HiPerGator (§3.3) | §6.2, §11.4 |
| D8 | The agent may download the listed public datasets from their official sources | Confirmed | §10.4 |
| D9 | No other credentialed cohort is available; E11 stays blocked until one is listed | Confirmed | §12 |
| D10 | The agent does not push. The repository is public; work stays in local commits until Daniel decides | Changed: no push | §3.2 |
| D11 | The signed data-use terms are the PhysioNet Credentialed Health Data Use Agreement 1.5.0 | Confirmed | §20.4 |
| D12 | ICML 2027 is the target; its call for papers is not yet published, so page limit, anonymity, and dual-submission rules are taken from ICML 2026 until it is | Confirmed | §17.1 |
| D13 | A watcher cycle of 15 minutes and a held SSH session are within UF Research Computing policy as documented; neither is stated in writing by them | Confirmed | §20.6, §20.8 |
| D14 | Professor Liu's framing preference is unknown; method plus mechanism | Confirmed | §1.6 |
| D15 | AUTHORIZATION.md is valid until March 31, 2027 | Confirmed | §20.2 |

Open item, not a default: the author list. It is undecided until Daniel has spoken with Professor
Liu, is due before abstract registration on January 14, 2027, and blocks nothing (§3.6).

## 3. Execution authority, instruction precedence, and preservation

### 3.1 One authorization, and current instructions control

Standing authority for this study comes from one file, AUTHORIZATION.md, which Daniel signs once
by the statement recorded in SIGNATURE.md (§20.2). Inside it the agent acts without asking. Outside it the agent does not act:
it records a BLOCKED item with the exact request and continues independent work (§20.3).

Without AUTHORIZATION.md and a SIGNATURE.md that quotes its hash, only this is permitted: repository edits on a dedicated task
branch, local commits, documentation, local CPU and synthetic testing, and analysis of already
authorized aggregates.

Current task-specific restrictions govern over historical delegations and obsolete next-action
lists. The older DECISIONS.md authorization to commit approval files on Daniel's behalf does not
authorize signing, editing, or extending AUTHORIZATION.md or any approval for this study. Existing
approvals from earlier studies apply only if their exact scope, budget, code, and bindings still
match. Retain old instructions as records. Do not restart completed archaeology or training because
old onboarding calls it the next step. A current read-only request overrides this plan.

### 3.2 Repository changes

Confirm exactly scalinity/testing-in-imagination and inspect the working tree first. Preserve
unrelated uncommitted work. Work on the task branch tii-cee-1/2026-11-13, created from main at
e5ad93fad5345f211cedc05c3dc491b101c63fa5. The remote is public, so the agent does not push (DEFAULT
D10): work stays in local commits until Daniel decides otherwise, and a submission must not link to
a non-anonymized repository. Never push to main or to any branch, never merge or close a pull
request, never force-push, never edit another person's branch, never rewrite history, never create
a release or tag on the remote. Commits carry no agent attribution text or co-author trailers.

### 3.3 Credentialed execution and resources

Patient computations use the approved private workflow and declared stage-by-split permissions.
Adding a stage to code does not grant access. AUTHORIZATION.md names the experiments, the GPU
allowance for each with its basis, the Slurm account and partitions, the permitted models, the
public datasets, and the one-shot test protocol. An allowance is a ceiling against runaway spending,
not a target and not a design constraint; reaching one produces a BLOCKED request, never a silent
reduction of an experiment. Do not infer permission from available quota or from an authorization
for different code. Never insert Daniel's signature. Queued, launched, and completed are different
states. Retries and failed jobs stay in the ledger and count against the allowance.

The agent never acquires data. Public datasets are downloaded only if AUTHORIZATION.md lists them
with their official source. Credentialed data are used only where they already sit in approved
private storage under Daniel's access. Anything else is a BLOCKED request.

Models. No model whose base originates in a country of concern runs on HiPerGator. REPRODUCTION.md,
D-5, records the rule: Qwen, DeepSeek, Yi, InternLM, and their derivatives are banned, and a
fine-tune or re-release does not clear a banned base. The panel and the backbones in
AUTHORIZATION.md are ungated models with US-origin bases. A gated model is used only if
AUTHORIZATION.md lists it and access already works; the agent never reads, prints, or copies a
Hugging Face token.

### 3.4 Protected data

Patient-level records, identifiers, histories, measurements, trajectories, generated clinical text,
private logs, retrieval indexes, and patient-derived fitted parameters stay in approved private
storage on HiPerGator. They are never exposed to hosted coding agents, external services, public
artifacts, or synced folders, and they never cross the SSH channel to the Mac. PhysioNet states
that its agreement prohibits sharing access with third parties, including sending data through
APIs or online platforms, and recommends locally deployed models; every model that touches patient
text in this study runs on HiPerGator. The agent's inspection surface is code, invented fixtures,
approved aggregates, scheduler metadata, and approved error codes (§20.10). Export guards stay
intact. Synthetic and public datasets carry no patient restriction but are still not published
without authorization. Test-split rows are never used for fitting, retrieval, calibration, budget
estimation, or threshold selection.

### 3.5 Historical studies and successor identity

Preserve frozen comparisons, protocols, result identities, and outputs. Corrections get an explicit
erratum and a versioned result. Legacy-conformance repairs make old code compute what its study
specified; everything in §5 is a new intervention under study ID TII-CEE-1 with separate outputs,
prospective experiment cards, and its own authorization. The old repair-run allowance is not a
budget for this study. A new stage name or protocol hash must never reopen completed test
admissions.

For the availability null, retain the old record and add the linked clarification: "This fixed
planner showed no detectable T5 benefit from true availability over the uniform control on this
validation cohort; modest effects and benefits from other planners remain unresolved."

### 3.6 People

The agent does not contact Professor Liu, Hugo Resendez, Edwin Salcedo, course staff, UF Research
Computing, PhysioNet, or anyone else, and does not draft messages for Daniel to send to teammates.
Nothing in the plan waits on a teammate's work, review, or sign-off. Their existing work is used as
it stands in the repository: Edwin Salcedo's synthetic study through adapters, never by editing his
branch; Hugo Resendez's documented work as recorded. Authorship, review, run operation, automated
review, and agent assistance are attributed separately and accurately. Missing evidence of a
teammate's work is not evidence of no work.

The author list is open until Daniel has spoken with Professor Liu. It is due before abstract
registration on January 14, 2027. It blocks no experiment and no draft: the manuscript is drafted
with an anonymous author block. Edwin Salcedo's authorship of the synthetic study is credited in
the text whatever the author list becomes.

## 4. Artifacts, status, and reporting cadence

The study directory is reports/research-advancement/. Every file named in this specification lives
there unless a path is given. Results use a separate versioned output namespace consistent with
the repository's data boundary; record the chosen layout in M0. Maintain:

- AUTHORIZATION.md: read-only to the agent and never edited after it is hashed; its SHA-256 is
  recorded in the M0 manifest and checked at every watcher cycle.
- SIGNATURE.md: Daniel's signature statement, recorded verbatim by the launch session from the
  launch prompt and committed. It quotes AUTHORIZATION.md's SHA-256 and commit.
- STATUS.md: milestone, current evidence, blockers, next executable action, running and queued jobs.
- BLOCKED.md: every item only Daniel can resolve, each with the exact request (§20.3).
- RESEARCH_LEDGER.md: claim and evidence register, review dispositions, experiment register, every
  job attempt with its outcome and GPU-hours.
- GATE_DECISIONS/: one immutable record per gate evaluation.
- DATA_USE_CHECK.md: the compliance record of §20.4.
- PRIOR_WORK_MATRIX.md: the verified matrix with access dates and status.
- EXPERIMENT_CARDS/: one frozen card per experiment in §13, committed before its first run.
- M0 through M5 milestone snapshots, immutable once closed.
- PROFESSOR_BRIEF.md, PLAIN_ENGLISH_GUIDE.md, MEETING_TALK_TRACK.md as in §18.
- paper/: the manuscript, figures generated only from result manifests, and REVIEWS/ with each
  internal review round and its dispositions.

Machine-readable results and manifests are canonical for numbers; the ledger is canonical for
claims. Human-facing documents cite experiment IDs. Produce an immutable milestone report at
closure or a decision-changing event, not after every status change. Provide a dated provisional
individual progress report in the existing LaTeX style at meeting checkpoints; never invent the
reporting week. Compile PDFs when tools exist; otherwise state the missing verification.

Status dimensions stay separate. Milestone: NOT STARTED, IN PROGRESS, COMPLETE, BLOCKED.
Experiment: PLANNED, BLOCKED, QUEUED, RUNNING, INTERRUPTED, FAILED, COMPLETED, SKIPPED BY GATE.
Scientific outcome: NOT EVALUATED, BENEFIT DETECTED, NO DETECTABLE BENEFIT, INCONCLUSIVE, or a
precise statement.

Evidence labels: VERIFIED_CODE, REPORTED_RUN, EXECUTED_EXPERIMENT, REANALYZED_FIXED_DATA,
RENDERED_EXISTING_RESULTS, HYPOTHESIS, PLANNED, BLOCKED. Evaluation provenance: in-sample,
validation-tuned, cross-fitted, fully outer-fold, independently held out, exact (twin). Use
"reproduced" only with a target, a repeated operation, and a comparison.

## 5. Formal core: definitions, estimators, and the method

All logarithms are natural; information is in nats. Probabilities entering a logarithm are floored
at 1e-6 and renormalized; the floor is part of the method and is recorded. Derivations and the
numerical check are in PROOF_GAIN_IDENTITY.md and cee_check.py; E0 turns that script into fixtures.

### 5.1 Objects

- Diseases y in a closed set of C classes (C = 4 on MIMIC-CDM and in the twin).
- History h: everything the agent has legitimately observed. Test a. Outcome o, in a set that
  includes UNAVAILABLE.
- Reader R: any procedure mapping a history to a probability vector over diseases. b = R(h) is the
  pre-test belief. q_o = R(h + (a, o)) is the belief the reader would hold after outcome o.
- World model W: a distribution P_hat(o | h, a), given by enumeration (twin) or by K weighted
  samples (charts). UNAVAILABLE carries probability 1 − p_available(a | h).
- Truth, known only in the twin: P_star(o | h, a) and the Bayes posterior.

### 5.2 Conservation of expected evidence

For a Bayesian agent whose predictive distribution and update come from one joint model,

    sum over o of P(o | h, a) × q_o = b.

For any reader and world model define

    imagined mean    b_bar = sum_o P_hat(o) × q_o
    drift            d = b_bar − b
    responsiveness   r = sum_o P_hat(o) × KL(q_o || b_bar).

A reader with r = 0 does not move whatever the result says. A reader with d ≠ 0 moves in a
direction that does not depend on what the result says. Two probes need no world model and are
measured wherever possible: the reaction to UNAVAILABLE where missingness is known to be
uninformative, and the reaction to an exact re-exposure of evidence already in the history. Both
should be zero.

### 5.3 CEE-Read: the projection

Find positive class weights u, one per disease, such that the reweighted imagined posteriors

    q'_o(y) = u(y) × q_o(y) / Z_o,    Z_o = sum_y u(y) × q_o(y)

satisfy sum_o P_hat(o) × q'_o = b. This is the minimum-KL projection of the raw joint
P_hat(o) × q_o(y) onto joints whose marginals are b and P_hat, computed by iterative proportional
fitting: u ← u × b / (current imagined mean), rescaled, until the largest marginal violation is at
most 1e-8 or 500 iterations. Non-convergence is recorded and the raw reader is used.

When the real result arrives, the corrected belief is q'_real(y) = u(y) × q_real(y) / Z_real with
the same u. The whole correction at a decision point is one vector of C numbers.

Location incoherence:

    KL_inc = sum_y b(y) × log u(y) − sum_o P_hat(o) × log Z_o   ≥ 0,

zero exactly when the raw pair already conserves expected evidence. Verified properties, each a
required unit test: exact re-exposure with a point-mass predictive returns the prior; a pair that is
already Bayes-consistent is unchanged; coherent value of information (§5.6) is never negative.

### 5.4 The gain identity

Let b_star and P_star be the true label and outcome distributions given the information the
projection used. The expected improvement in log score from using q' instead of q on real results is

    Delta = KL_inc
          + sum_y (b_star(y) − b(y)) × log u(y)
          − sum_o (P_star(o) − P_hat(o)) × log Z_o.

The identity is exact for any truth and any positive class weights; the projection is what makes
its first term a divergence. Its three terms are incoherence, prior calibration, and world-model
error as seen by the reader. Consequences:

1. If the pre-test belief and the world model are calibrated, the gain equals KL_inc.
2. World-model error matters only through its inner product with log Z_o.
3. The per-case realized gain g = log u(y_true) − log Z_real is observable on labeled data from
   factual outcomes alone. Its mean estimates Delta without any counterfactual.
4. The guarantee is single-step. Over several steps the corrected belief becomes the next prior and
   need not be calibrated; §5.12 gives the multi-step accounting and E1-D tests it.

The identity is an application of the Pythagorean relation for minimum-KL projections. It is used
as an accounting device and is not claimed as new theory.

### 5.5 Reader-weighted world-model error and the scale diagnostics

Reader-weighted error for reading: E_W = sum_o (P_star(o) − P_hat(o)) × log Z_o. For decisions,
replace log Z_o by the committed utility phi(q'_o). Exact in the twin. On charts, estimated by test
type on calibration folds as the mean of log Z_real minus sum_o P_hat(o) × log Z_o. The matching
vector check is mean(q_real) − mean(b_bar) by test type.

Tempered family: q_o(tau) ∝ b × (q_o / b)^(1/tau). A tau above 1 flattens the reader's reactions; a
tau below 1 amplifies them.

Budget-implied scale, tau_B, the primary scale diagnostic. Given an information budget B for the
test (§5.11), tau_B is the tau at which the projected pair carries exactly B nats. tau_B above 1
means the reader reacts more than the test can justify; below 1, less. It is reported wherever a
budget exists, whether or not CEE-Scale is applied.

Budget-free scale, tau_ratio, a secondary diagnostic. The grid value minimizing KL_inc(tau) / r(tau)
over {0.5, 0.67, 1, 1.5, 2, 3, 5, 8}. In the sandbox it recovered pure scale errors and misbehaved
when drift or correlation was present. E1-B measures how well it tracks tau_B; it is never a repair.

Minimizing KL_inc(tau) alone is degenerate, because an agent that never updates is coherent, and
must not be used for anything.

### 5.6 Coherent value of information

With utility U(d, y) for committing to diagnosis d when the truth is y, let
phi(c) = max over d of sum_y c(y) × U(d, y). Then

    VOI(a | h) = sum_o P_hat(o) × phi(q'_o) − phi(b)   ≥ 0

because the projected posteriors average to b. Net value is VOI minus lambda times the expected
charged cost. Costs follow the environment: unavailable results are free and consume a step.
Utilities: in the twin, 1[correct], with lambda × normalized cost as in the existing study; on
charts, the recall-aligned utility U(d, y) = 1[d = y] / (C × pi_y) with pi the training class
frequencies, so that expected utility matches mean class recall. Primary lambda is 1 on normalized
cost in both settings; the frontier is reported at lambda in {0, 0.5, 1, 2}.

### 5.7 CEE-Commit and CEE-Select

CEE-Commit: commit to the diagnosis maximizing expected utility under the coherent belief when no
eligible test has positive net value, or when the step or token budget is reached. CEE-Select: among
eligible tests, request the one with the highest positive net value. Selection is one-step. Both
are named interventions with their own ablations; neither is hidden inside the other.

### 5.8 CEE-Gate

Apply the correction or an override at (h, a) only if

    KL_inc(h, a) > eps_prior(type of a) + eps_wm(type of a),

where the two bounds are upper 90% bootstrap bounds, by test type, of the absolute prior and
world-model terms of §5.4 estimated on calibration folds that exclude the scored case. Otherwise
the raw reader and the baseline action are used. A reader for which the estimated mean gain is not
positive on calibration data has CEE-Read disabled entirely. Beneficial and harmful applications
are counted and reported for both gated and ungated use.

### 5.9 Sequential tracking and evidence identity

Stateful track: c_0 = R(h_0) after any declared calibration. At step t with raw reader belief b_t,
form increments rho_o = q_o / b_t, proposals m_o ∝ c_t × rho_o, project the proposals onto mean c_t
to get u_t, and set c_(t+1) ∝ u_t × c_t × rho_real. Stateless track: c_(t+1) ∝ u_t × q_real with
u_t computed against b_t. Both run in E1. The variant used on charts is the one with lower exact
expected log loss averaged over the drift families under world model W2 in E1; ties go to the
stateless track. That selection uses synthetic data only.

Evidence identity lives in the world model. W never imagines a value that contradicts an identical
measurement already in the history; items shared between an observed panel and the candidate are
copied. Re-exposing identical evidence then yields a point-mass predictive and no belief change,
which is how duplicate evidence is discounted without a separate deduplication rule. Genuinely
repeated measurements at different times stay distinct. Where the data lack identity, say so.

### 5.10 What is guaranteed and what is not

Guaranteed: the identity, for any class weights and any truth; KL_inc non-negative once the
marginal constraint holds; non-negative coherent VOI; no change for an already coherent pair; no
update on exact re-exposure; the multi-step statements (i) to (iii) of §5.12. Not guaranteed:
improvement when the prior or world model is miscalibrated; improvement over several steps; any
accuracy gain when log loss improves; anything about CEE-Scale when its budget is wrong. The
projection changes location only. A reader whose reactions are too strong or too weak is flagged
by tau_B, is not repaired by the projection, and is the case CEE-Scale exists for.

### 5.11 CEE-Scale: matching an information budget

Purpose. Repair readers whose reactions have the wrong size, including reactions to evidence that
repeats what the history already says. It is a separately named component with its own ablation,
its own gate, and its own line in every arm's contract.

Information budget. B(h, a) ≥ 0 is an estimate of I(Y; O | h, a): how many nats the result of test
a can tell about the diagnosis, given the history. Sources, always declared:
- B-model: computed from a class-conditional world model's own likelihood,
  B = sum_o P_hat(o) × KL(post_W(· | o) || prior_W). Exact under the true model in the twin.
- B-held-out: the mean reduction in log loss of the reference reader R3 when the result is added,
  measured on folds that exclude the scored case, by test type and history stratum. Strata are
  fixed now: number of results already read (0, 1, 2 or more) crossed with whether R3's largest
  probability exceeds 0.8. The budget used is the one-sided lower 90% bootstrap bound of that mean,
  resampling admissions. A test type is eligible for CEE-Scale only if that bound does not exceed
  the held-out mean reduction measured on a disjoint fold.

Rule. For each tau in the allowed interval, temper the increments, project (§5.3), and compute the
responsiveness of the projected pair, r'(tau) = sum_o P_hat(o) × KL(q'_o(tau) || b), which is the
mutual information of the coherent joint. tau_B is the solution of r'(tau) = B by bisection on
log tau to 1% relative tolerance, clipped to the interval. The repaired posteriors are the
projected posteriors at tau_B, and the same tau_B and class weights are applied to the real result.

Variants. Shrink-only: tau in [1, 16]; reactions may be flattened, never amplified. Two-sided:
tau in [1/8, 16]. The variant used outside the twin is chosen in E1-D on synthetic data only: the
one with the lower worst-case exact expected log loss over the stress set {B, 0.5 B, 2 B, B with
independent lognormal noise of sigma 0.7}, averaged over reader families under world model W2. Ties
go to shrink-only.

Properties. If the reader's increments are a power distortion of a coherent reader's and B equals
that pair's mutual information, the rule recovers the coherent reader exactly. A never-updating
agent has r' = 0 and is not a solution unless B = 0, which is the exact re-exposure case. The rule
is not chosen by, and does not minimize, incoherence. Monotonicity of r' in tau is assumed, not
proven; E0 includes a monotonicity fixture and the code falls back to the largest tau with
r' ≥ B when it fails.

Known failure. An overestimated budget amplifies a reader past the evidence. In the sandbox a
budget twice the truth multiplied log loss several times over in the two-sided variant, and still
harmed an over-reactive reader in the shrink-only variant. Hence the conservative budget, the
eligibility test, the clipping, and the gate: CEE-Gate applies to the scale stage with its own
realized gain, the difference in log score at the real outcome between the scaled-and-projected
belief and the projected-only belief, estimated on excluded folds.

Labeled alternative. A scalar evidence temperature per test type fitted by log loss on calibration
folds is a baseline (§6.3) and the only route for under-reactive readers if shrink-only is selected.
It is always named as label-tuned.

### 5.12 Multi-step statements

Fix a deterministic selection policy. On the stateful track:

(i) Martingale by construction. E_{P_hat}[c_{t+1} | h_t] = c_t for any reader.

(ii) One joint model. M(o_{1:T}, y) = [prod_t P_hat(o_t | h_{t−1})] × c_T(y; h_T) has class marginal
c_0 and conditionals M(y | o_{1:t}) = c_t for every t. The corrected agent is Bayesian for one
joint model whose outcome marginal is the world model. A raw reader's beliefs are not the
conditionals of any such joint, and the class marginal it implies, m_0 = E_{P_hat}[r_T], drifts away
from r_0 as T grows.

(iii) Accounting over T steps. For any belief process,

    E_O[ KL(P_star(Y | O) || bel_T) ]
        = KL(b_star || m_0) + E_Y[ KL(P_star(O | Y) || M(O | Y)) ] − KL(P_star(O) || P_hat(O)).

The left side is the excess log loss of the final belief. For the corrected agent m_0 = c_0 at
every horizon. Coherence therefore removes accumulated drift and nothing else: the middle term, the
error of the likelihood the agent implicitly uses, is untouched, and the sum can go either way.

(iv) Under the truth the corrected belief is an approximate martingale with defect at most
2 × TV(P_star, P_hat) × max_o ||q'_o − c_t||_1 per step. This is elementary and is not a
contribution.

These are accounting statements, not a guarantee of improvement. E1-D reports the three terms of
(iii) exactly at every horizon for every reader family, raw and repaired; that is the tested
statement about multi-step behavior.

## 6. Components and the arm-component contract

### 6.1 Contract

Every arm declares all of the following before running:

| Component | Required declaration |
|---|---|
| Outcome predictor (world model) | Identity, conditioning, outcome support, K, evidence-identity rule |
| Reader | Identity, input information, belief readout, any output calibration |
| Coherence layer | None, CEE-Read stateless, or CEE-Read stateful; floors; convergence rule |
| Scale layer | None, CEE-Scale shrink-only or two-sided; budget source; bounds; eligibility result |
| Gate | None, CEE-Gate, or a named baseline gate; where its thresholds were fit |
| Acquisition | Baseline policy, raw score, or CEE-Select; eligible actions; tie rule |
| Commit and guards | Unchanged, confidence threshold, CEE-Commit, or reference; guard precedence |
| Utility and budgets | Utility table, lambda, cost semantics, step and token limits |
| Availability information | Learned, population, observed, or privileged |
| Information access | Every ordinary and privileged input |
| Policy identity | Recipe, backbone, seed, training folds, checkpoint digest |
| Pairing and provenance | Cases, seeds by purpose, source and fitted-artifact digests |

A comparison advertised as changing one component holds the others fixed. If changing the reader
also changes its input representation, name the joint treatment.

### 6.2 Registry

Readers:
- R0: Bayes reader under the true model. Twin only. Reference.
- R1: the trained policy as reader. Primary form: the hypothesis agent's distribution over the four
  diseases at a history, by teacher-forced scoring of canonical continuations. Secondary form: the
  decision agent's forced-commit distribution. Conformance and calibration in §10.1. One R1 per
  trained policy: the existing primary policy, the seeds of E8, the fold policies of E10, and the
  second-backbone policies of E9.
- R2: the structured content reader as stored (drop-absent with tau 13) and its untempered form.
- R3: a regularized partial-evidence classifier trained on training-split prefixes with randomized
  order, calibrated by cross-fitting. Reference reader and source of B-held-out.
- R4: simulated readers in the twin: class-conditional independence (double counting under
  correlation), entrenching drift, fixed-class drift, over-reactive, under-reactive, and mixtures
  of drift with scale error.
- R5: the language-model reader panel, zero-shot, on rendered text. At least eight open-weight
  models from at least four families spanning at least a tenfold range of parameter counts, and
  including the policy's backbone without fine-tuning. The list is in AUTHORIZATION.md (DEFAULT
  D7): ungated models with US-origin bases only (§3.3).
  Primary readout: logit scoring of the class continuations. Secondary readout on two models:
  verbalized probabilities.

World models:
- W0: the true generative model. Twin only.
- W1: class-conditional independent model (the twin's count model; the clinical content model).
- W2: history-conditioned outcome predictor that respects evidence identity; W2-cc is its
  class-conditional form, which supplies B-model. Twin: a latent phenotype model. Charts: per-item
  regularized multinomial regression on history features and already observed bins.
- W3: retrieval sampler for text outcomes on charts: K results for the requested test drawn from
  the nearest training admissions under declared features, excluding the evaluated admission;
  W3-cc is the class-conditional variant. Private, training split only. Requires data flow F1 (§20.4).
- W4: generative sampler: a language model on HiPerGator writes K plausible results for the
  requested test given the history. This is the published simulate-and-re-query recipe and needs no
  new data flow. On public data the sampler may be the reader itself or an empirical model.
- Controlled errors, twin only: mirror corruption at several strengths, temperature miscalibration,
  and a matched pair with equal exact KL from the truth, one perturbed in directions the reader
  ignores and one in the direction the reader responds to most.

K is 32 on charts and public data, with sensitivity at 8 and 16 by subsampling the same draws.
The twin uses exact enumeration and reports the error of K in {4, 8, 16, 32} against it.

### 6.3 Baselines and what each isolates

| Baseline | Isolates |
|---|---|
| Raw simulate-and-re-query scoring (ACTMED-style), no layer | The coherence and scale layers |
| Entropy-change score under the stored incoherent pair | The legacy planner as specified |
| Prior-times-likelihood coherent information gain (BED-LLM-style), where a likelihood is available | Repair versus avoidance of the pairing |
| Arbitrage-style patching with outcome weights supplied by the reader itself (after Paleka et al.) | A separate world model versus self-supplied outcome weights |
| Confidence-threshold commit (MediQ-style) | Coherent VOI commit versus raw confidence |
| Conformal risk stop with one count-based model (CaRDI-style) | Behavior under correlated evidence |
| Error-ranker stop (CROS-style, simplified) | Mechanism versus risk ranking |
| Outcome-uncertainty gate and value-disagreement gate | Reader-weighted gating versus generic model uncertainty |
| Content-free calibration and uniform-weight projection | Whether the world model's weights matter |
| Labeled evidence temperature, cross-fitted | Label-free budget versus labeled scale |
| Matched-information non-planning value baseline | Planning versus no planning at equal information |

Baselines implement the same interfaces and are evaluated on the same cases. Each is a
re-implementation at our budget and is named as an approximation of the cited idea, never as a
reproduction of the cited paper. A baseline that cannot be implemented faithfully is dropped with a
note.

## 7. M0 — Evidence, verification, freeze, and launch checks

### 7.1 Evidence register

Record input document hashes, repository refs, remote visibility, execution access, and instruction
precedence. Record that AUTHORIZATION.md's SHA-256 equals the one SIGNATURE.md quotes. Build the
claim-to-evidence register with source, evidence type, split, exposure, strongest competing
explanation, and next discriminating check. Adjudicate REVIEW-001 through REVIEW-014 against
current code; do not re-fix what is fixed. Copy §2.7 into STATUS.md with its status column.

### 7.2 Prior-work matrix

PRIOR_WORK_MATRIX.md is delivered with this specification, with a status for every row. The agent
maintains it: re-read rows marked U when their source is reachable; repeat the bounded search of
its last section against the final method description; add a citation-graph pass over ACTMED,
BED-LLM, Paleka et al., and Kim et al.; record dates and results. No report states novelty beyond
what the matrix supports, and none before the repeated search is recorded. Cross-paper scores are
never compared.

### 7.3 Freeze

Commit experiment cards E0 through E4 and E7 through E10 with hypotheses, estimands, arms under
§6.1, measurements, decision rules, candidate sets, maximum runs, estimates, and dependencies.
Cards E5 and E6 are frozen at their gates, by the agent, before their first run. A threshold is an
operational decision rule chosen before outcomes, not a significance claim.

### 7.4 Launch checks

Each check ends in a recorded answer or a BLOCKED item, never in a guess.

a. Logged trajectories (DEFAULT D5). From code and approved metadata only, determine whether the
   unguarded validation and test trajectories store the full prompt at each step, a digest of it,
   or neither. If prompts are stored, use them. If not, rebuild each prompt deterministically from
   the stored history through the same prompt-building code and checkpoint, and require the
   conformance of §10.1 on a bench: byte-for-byte equality where digests exist, otherwise agreement
   of the probe's greedy hypothesis with the stored one on at least 99% of aligned steps. If
   neither route conforms, the unguarded validation trajectories are regenerated once with full
   prompt logging from the same checkpoint and seeds, and accepted only if final answers match the
   stored outcomes on at least 99% of admissions. Test trajectories are never regenerated outside
   the one-shot protocol of §11.3.
b. Data-use compliance, §20.4.
c. UF Research Computing policy: re-read the pages cited in WATCHER_DESIGN.md, record the dates, run
   one no-op watcher cycle, and confirm that a scrontab job can call the scheduler commands.
d. Environment: the Slurm account, QOS, partition, GPU type, storage root, and model files named in
   AUTHORIZATION.md exist and are usable; nothing is assumed from quota. Every field that
   AUTHORIZATION.md marks UNVERIFIED is checked with read-only commands through the existing
   session. A live value that differs is a BLOCKED item, because AUTHORIZATION.md is never edited.
e. Recipe: how the training recipe uses the validation split, which fixes the inner split of E10.
f. Cohorts: the names of credentialed datasets already present in approved storage, for E11.
   Nothing is acquired.
g. Venue: re-read the official ICML 2027 pages and record what is published (§17.1).

Acceptance (gate G0): traceable status, review dispositions, matrix with statuses, frozen cards,
launch checks answered or blocked, first professor packet. No patient experiment is required.

## 8. M1 — Integrity repairs the claim depends on

Each repair names the experiment that needs it. A repair not needed by a planned experiment is
deferred with its dependency recorded, not dropped.

### 8.1 Synthetic environment (needed by E1, E2)

Work through adapters and a new namespace; Edwin Salcedo's branch is not edited.
REVIEW-002: namespace patient-generation and behavior RNG streams by purpose and split; verify
generated cases with fingerprints, not display IDs; keep identical evaluation realizations across
arms. REVIEW-006: with diagnosis-independent missingness and free unavailable results, expected
monetary cost is p_available × cost; test p = 0, p = 1, and intermediate values against enumeration.
REVIEW-011: record a clean source revision or an immutable snapshot digest. Preserve legacy outputs
and regenerate affected evidence in the new namespace. Adjudicate the old likelihood sample and the
old utility sample separately.

Additions for this study: a counterfactual interface returning every test's outcome and
availability for a patient; exact true predictive, posterior, and conditional mutual information;
a reduced six-test twin (physical examination, CT, ultrasound, complete blood count, liver function
panel, urinalysis) small enough for exact dynamic programming; a text renderer with three
templates; variants with no phenotype correlation, stronger correlation, a duplicated panel, and
informative missingness. The informative-missingness variant is held out: no method choice may use
it.

### 8.2 Clinical scoring (needed by E3)

REVIEW-001: learned and reference models score identical item targets with identical inclusion,
conditioning, smoothing, and weights; the reference is fit separately from pooled training outcomes
under the matching chart-availability condition; joint-panel likelihood is a separately named
metric. Hand-computed single- and multi-item fixtures, identical distributions giving zero skill,
diagnosis-dependent availability, rare and absent outcomes. Do not predict whether corrected skill
stays positive.

### 8.3 Preflight and artifact binding (needed before any credentialed GPU run: E4, E5, E6, E8, E9, E10)

REVIEW-003: one shared preflight at scheduling and at execution. Before the first episode or
training step an immutable manifest binds the policy checkpoint or training recipe, reader-probe
configuration, world-model artifacts including the retrieval index digest, availability model,
calibration parameters, budget tables, experiment card, protocol, evaluator, and the SHA-256 of
AUTHORIZATION.md. Verify the bytes consumed. Verify again on every resume. A mismatch is a refusal;
it never creates a new test key and is never retried.

### 8.4 Information boundaries (needed by every experiment)

Metamorphic tests with observations, artifacts, and RNG identity fixed: changing an unrevealed
result or the hidden diagnosis cannot change an ordinary arm's beliefs or actions; privileged arms
receive only their declared inputs; imagined outcomes are never written into the real history; the
retrieval model never returns rows from the evaluated admission or from validation or test; a fold
policy never scores an admission it trained on; re-exposure of identical evidence does not change
a CEE belief.

### 8.5 Trajectory consistency (needed before E5 arms that replace an action)

REVIEW-008: a replaced action carries the specified neutral thought; test the assembled text,
token mask, completion IDs, and next prompt. Preserve byte-identical behavior when no layer fires.

### 8.6 Wording erratum

REVIEW-010: add the scoped clarification in §3.5 as a linked erratum. No number changes.

Acceptance (gate G1): repairs needed by E0 through E3 and by the training track are verified or
specifically adjudicated; legacy outputs preserved; affected claims qualified. A repair that fails
blocks only the experiments that depend on it.

## 9. M2 — The CEE instrument and the exact synthetic study (CPU)

### 9.1 Instrument and fixtures (E0)

Implement the projection, identity terms, tau_B, tau_ratio, CEE-Scale, coherent VOI, commit, gate,
and both tracks as a small library with no dependency on either environment, and a command-line
audit that takes any reader and world model behind the interfaces of §16 and prints an audit card.
Fixtures are hand-computable or enumerated, starting from cee_check.py: the identity holds to 1e-9
for arbitrary truth and arbitrary class weights; calibrated marginals give gain equal to KL_inc;
point-mass predictive returns the prior; a Bayes-consistent pair is unchanged; coherent VOI is
non-negative; the power-distortion case is recovered exactly by CEE-Scale; r' is monotone in tau
on the fixture set; the three multi-step statements hold by enumeration; floors, clipping, and
non-convergence behave as specified.

### 9.2 Exact attribution of regret (E1, part A)

In the reduced twin compute by exact enumeration the value V(S, R, C) of an agent built from a
selector S, reader R, and commit rule C, and the Bayes-optimal value under the same observation
access and budget. Attribute regret by swapping one component at a time to its reference in the
pre-declared order selection, reading, commitment, and report the average over all six orders as a
sensitivity check. Exact search under known dynamics is not clairvoyance: references never see
unrevealed outcomes.

### 9.3 Audit, projection, and gate with simulated readers (E1, part B)

For every reader in R4 and R0, every world model in §6.2, and every environment variant, compute
exactly: drift, responsiveness, KL_inc, tau_B, tau_ratio, the two model-free probes, the identity
terms, expected log loss raw and corrected on both tracks, accuracy, one-step reader-relative regret
under raw and coherent scores, and episode utility under threshold commit and CEE-Commit. Pair
cases across arms. Use twenty environment seeds. Report how well tau_ratio tracks tau_B.

### 9.4 Reader-weighted error against likelihood (E1, part C)

Across the controlled world-model errors, relate per-history regret and log-loss change to exact
KL from the truth, calibration error, and reader-weighted error. The matched pair has equal KL
within 0.005 nats by construction.

### 9.5 Scale repair and multi-step behavior (E1, part D)

For every R4 family, including mixtures of drift and scale error, apply CEE-Scale in both variants
with four budgets: exact; the model budget of W1; the model budget of W2-cc; and a held-out budget
estimated from simulated training patients exactly as B-held-out is estimated on charts, with its
conservative bound and eligibility test. Run the stress set of §5.11. Report exact expected log
loss, accuracy, and utility; the selected variant by the rule of §5.11; how often the eligibility
test admits an overestimated budget; and, at every horizon up to the step limit, the three terms of
§5.12 (iii) for raw, projected, and scaled-and-projected beliefs.

### 9.6 Structured readers on charts (E3; CPU, credentialed)

Fit W2, W2-cc, and R3 on training data. On validation, with all fitted pieces cross-fitted by
admission in five folds and the existing eight chart orders, compute for R2 and R3 the audit
quantities, tau_B under B-held-out and B-model, the realized gain and its three estimated terms,
and posterior log loss and Brier score raw, projected, and scaled-and-projected. Report the budget
calibration by stratum and the mass of negative raw scores before and after projection. Export
aggregates only. Eight orders of one admission are not eight people: resample admissions.

### 9.7 Reader-opportunity audit (CPU, credentialed, existing rows)

From the logged unguarded validation trajectories, the ones V1 is computed from, count provided
results per admission and V1's errors with at least one provided result. Report the optimistic
bound on T5 from correcting the final belief, G_max = (100 / 4) × sum over classes k of E_k / N_k,
with E_k the V1 errors in class k that have at least one provided result and N_k the class size,
and report the V1-correct admissions exposed to regression. The bound applies only to reading
interventions on fixed evidence and is relative to V1.

Acceptance (gates G2a, G2b, G2c in §14): the instrument is validated and E1 has answered its
questions, whichever way. E3 completes or is precisely blocked.

## 10. M3 — Readers that matter: the panel, clinical replay, and public data

### 10.1 Reader probe conformance

For every trained policy, on the synthetic fixture and then on a bench, the R1 probe must reproduce
logged behavior: with no substitution, prompts are byte-identical to the logged or rebuilt ones
(§7.4a), and the probe's most likely hypothesis matches the stored greedy hypothesis on at least 99%
of aligned steps. A lower rate is investigated before any use, and a policy that cannot conform is
excluded with the reason recorded. Scorer likelihoods carry verbalizer and length effects; a
five-parameter output calibration of the initial belief is fit by cross-fitting and named as a
component. The probe never writes to a trajectory. Scoring substituted observations is data flow
F2 (§20.4).

### 10.2 The language-model reader panel in the twin (E2; GPU, no patient data)

Readers R5 on rendered twin observations. 1,000 twin patients, three renderings each, two fixed
selectors (random order, and greedy information gain under W0), exact enumeration of the five
findings and UNAVAILABLE for every remaining test at every step. Two world models per reader: W0,
and the reader itself as sampler (W4), which is the published recipe.

Per reader: the coordinates of Figure 1 with intervals, tau_B under the exact budget and location
incoherence at tau_B; responsiveness; the two model-free probes; exact change in expected log loss
under projection alone and under the scale variant selected in E1-D; both tracks; the share of
regret by component from one-step quantities and paired counterfactual replays; sensitivity to
rendering and prompt order; the error of K samples against enumeration.

Regime call, fixed now: scale regime if the 95% interval of log tau_B excludes 0; drift regime if
the interval of location incoherence at tau_B lies above the same quantity for R0 under identical
sampling; unresponsive if responsiveness is below 5% of the exact budget. More than one can hold.

### 10.3 Clinical evidence replay (E4; GPU, credentialed)

Fixed evidence, no new acquisition. Cohorts:
- P: the existing primary policy on its logged unguarded validation trajectories, 240 admissions.
- S: each E8 seed policy on its own unguarded validation trajectories, 240 admissions each.
- X: the cross-fitted cohort. Each of the 2,160 training-plus-validation admissions is rolled out
  unguarded and replayed by the E10 fold policy that never trained on it.
- B: each E9 second-backbone policy on its own unguarded validation trajectories.

At each step with a provided result, obtain R1's belief before the result, after the real result,
and after K = 32 imagined results from W3 if data flow F1 passed, and from W4 in every case, plus
the two model-free probes; for commitment, the same for every eligible test. Compute the audit
quantities, tau_B under B-held-out, the realized gain and its terms, and the corrected belief along
the trajectory, with every fitted piece cross-fitted by admission.

Arms evaluated here by replay, with no new rollout, because a layer that never writes to the
history follows a prefix of the logged trajectory: A1, read and commit with CEE-Read; A1s, the same
with CEE-Scale added, if G2c passed. Their stopping point, final answer, and provided-test cost are
compared with that policy's own V1 on identical evidence.

Primary endpoint: mean change in log loss of the final belief, A1 against raw, on cohort X,
resampling admissions. Secondary: the same on P, S, and B; T5 and cost of A1 and A1s against V1;
the forced-continuation curve of error against number of results read, raw and corrected; the
Figure 1 coordinates of every policy.

Layer rule, fixed now for everything downstream: the layer is scale-and-projection if G2c passed
and the scale stage's cross-fitted realized gain on cohort X has a 95% interval above 0; otherwise
projection only. No T5 comparison enters this choice.

### 10.4 Public complete-record datasets (E7; GPU, public data)

Datasets (DEFAULT D8, licenses verified from the official pages in M0 and recorded): Chronic Kidney
Disease, Hepatitis C, and Pima diabetes, the tabular sets ACTMED used, and DDXPlus with 2,000
sampled test patients. Readers: six models of the panel, the two smallest, two middle, and two
largest authorized. Two pairings per reader: the published one, where the reader samples the
outcomes itself, and a separate empirical world model fitted on a training split. Because records
are complete, the realized gain is computed for every test of every patient, not only the ones a
policy chose.

Per reader-by-dataset cell: Figure 1 coordinates, predicted and realized gain, log loss and
accuracy raw and repaired, tests used under each commit rule, and the baselines of §6.3. This
experiment exists to show whether the finding is specific to our agent and our environment.

Acceptance (gate G3 in §14): E2, E4, and E7 are completed and interpretable, or precisely blocked,
and the decision about closed-loop arms is recorded.

## 11. M4 — Closed-loop evaluation, the one-shot test, and the training track

### 11.1 Arms

Arms use an unguarded trained policy, the acquisition V1 uses, and the layer fixed by §10.3. The
CEE layer supplies the commit decision and the final answer, so no episode ends without a
diagnosis. Where an arm replaces an action, replacement follows §8.5. References need no new run:
each policy's V1, and A1 from E4.

| Arm | Acquisition | Reader and layer | Commit | Purpose |
|---|---|---|---|---|
| A1, A1s | Baseline policy | R1 + layer | CEE-Commit | Reading and commitment only; by replay in E4 |
| A2 | CEE-Select among the policy's proposal and two alternatives | R1 + layer | CEE-Commit | Adds coherent selection |
| A3 | Same as A2 with raw scores | R1, no layer | Raw-score commit | Isolates the layer |
| A4 | A2 with CEE-Gate | R1 + layer, gated | CEE-Commit | Run only if E4 shows at least 20% harmful applications |

Each new arm runs on validation with the primary policy and on cohort X with the fold policies,
2,400 admissions in all. At most two repair runs in total, on the 240 validation admissions only.
Candidate alternatives for A2 come from a CPU score so that reader calls stay bounded; the rule is
frozen on the card.

### 11.2 Validation rule (E5)

Closed-loop contrast: A2 − V1 on T5 over cohort X, paired by admission, with provided-test cost and
requests reported; the 240 validation admissions are reported separately. Family, Holm-adjusted:
A2 − A3 (the layer) and A2 − A1 (what selection adds to reading and commitment). The rule decides
eligibility only: integrity (every arm completes all admissions with no unexplained scorer or
replacement failure) and direction (A2 − V1 has a non-negative T5 point estimate on cohort X). No
arm or setting is selected by a T5 comparison: the test protocol's primary arm is fixed now as A1
with the layer of §10.3, and A2 joins the test family only if this rule holds for it.

### 11.3 Frozen one-shot test (E6)

The protocol is written and committed by the agent before any test access: arms; the primary
contrast, A1 − V1 on the 240 test admissions, averaged over the five policies of the primary
recipe and paired by admission, as the only uncorrected test; the family (A2 if eligible; the
second backbone; A1s if it is not the primary layer); statistics as in §15; the minimum detectable
effect recomputed from cohort X; digests of every fitted artifact; bench timings; and the
disclosure that the test split is a previously examined benchmark (DEFAULT D6).

Every fitted quantity is frozen from training and validation data. Each policy's unguarded test
rollout and its replay happen once, inside this experiment. The run starts automatically when gate
G4 passes, under the standing authorization, and never twice: an interruption resumes from its
checkpoint under the same manifest and counts as the same run; a manifest mismatch is a refusal;
there is no re-tuning and no second attempt. E6 is the one additional, disclosed use of the 240
test admissions: after it they are not read, scored, or rolled out again for any purpose in this
study. If the primary contrast is not significant, report the interval against the recomputed
minimum detectable effect.

### 11.4 The training track (E8, E9, E10)

Training does not depend on the CEE code and starts as soon as §8.3 is verified and the cards are
frozen. The recipe is frozen: no hyperparameter search, the documented settings only.

- E8, seeds: four additional policies of the primary recipe on the original training split, with
  new seeds. With the existing policy this gives five.
- E9, second backbone: three seeds of the same recipe on ibm-granite/granite-3.3-8b-instruct
  (DEFAULT D7), a model of the same size class from a different family. At most
  two recipe repairs, for template or memory faults only, each recorded. If it still does not
  train, report that and continue.
- E10, cross-fitted policies: five folds, stratified by class, over the 2,160 training-plus-
  validation admissions. Fold policy k trains on the other four folds; whatever role the
  validation split plays in the recipe (§7.4e) is played by a fixed stratified tenth of that
  training portion. The test split is untouched. Provenance label for cohort X: recipe tuned on
  the original validation split, policies cross-fitted.

Every policy gets its own V0 and V1 reference on validation or on its held-out fold, and its own
probe conformance. Each training job is split into resumable parts of at most 24 hours (DEFAULT D3).

Acceptance: frozen comparisons completed and interpretable, or a precise null or blocker.

## 12. M5 — Robustness, independent evidence, the paper, and the submission decision

Robustness, each as a declared sensitivity: K; world model (W3, W4, class-conditional variants);
stateful and stateless tracks; renderings; gate on and off; budget source; seeds; backbone; and
the held-out informative-missingness twin variant, used here for the first time.

Independent evidence. E7 and the reader panel are independent of our agent and environment. A
second credentialed cohort is used only if AUTHORIZATION.md lists one that already sits in approved
storage (E11, audit and replay only, its card frozen before any access). Otherwise E11 is a BLOCKED
item naming exactly what would be needed (DEFAULT D9). Another synthetic dataset is breadth, not
clinical validation. State closed-differential scope, retrospective availability, unknown
counterfactuals on charts, informative workup, missing subject identifiers, and backbone
deviations.

The paper. The agent drafts the full manuscript in paper/, every number and figure generated from
result manifests, with the claims limited to §1.4 and the matrix. It includes the AI Disclosure
Statement ICML 2027 requires, describing the agent's role in code, experiments, analysis, and
drafting, and how outputs were verified; an impact statement; and a limitations section. Three
internal review rounds, each by a fresh instance that has not seen the work being produced and
reads only the manuscript and supplement, using the objections in REVIEW_AND_REDESIGN.md as its
checklist. Every review comment gets a recorded disposition. Round two includes an independent
re-derivation of every headline number from the manifests.

Submission decision. Update the matrix against the final method. Give separate judgments for
measurement soundness, mechanism, method benefit, robustness, independent evidence, and
differentiation. Re-read the venue's official pages. Write a go or no-go recommendation against
§17.3. The agent does not register, upload, or submit anything; Daniel does.

Acceptance (gate G5): the recommendation, the manuscript, and the remaining requirements listed.

## 13. Experiment cards

Each card is committed before its first run with the fields of §7.3. Summary:

| ID | Question | Design | Primary measurement | Decision rule fixed in advance | Result that kills the claim | Compute |
|---|---|---|---|---|---|---|
| E0 | Does the code compute the declared quantities? | Enumerated fixtures from cee_check.py | Agreement with hand calculation | All fixtures within tolerance | Any identity or invariant failure | Local CPU |
| E1-A | Where does regret come from? | Exact component swaps, reduced twin | Share of regret by component | Descriptive; order sensitivity reported | Attribution unstable: order spread exceeds the effect | CPU |
| E1-B | Does projection repair drift, and only drift? | Simulated readers × world models × variants, twenty seeds | Exact change in expected log loss | Repair present if both drift families improve with the paired 95% interval excluding 0 under W0 and W2; no-harm if R0 with W0 changes by at most 1e-9; negative controls confirmed if both scale families worsen; gate valid if allowed applications have positive mean gain for every family | Drift families do not improve under W0 | CPU |
| E1-C | Does reader-weighted error explain loss better than likelihood? | Controlled model errors, matched pair | Rank correlation with per-history regret | Supported if the correlation exceeds that of KL by at least 0.20 with interval excluding 0 and the matched pair differs | Margin not met and matched pair does not differ | CPU |
| E1-D | Does budget matching repair scale and redundancy, and what happens over several steps? | Both variants × four budgets × stress set; terms of §5.12 (iii) at every horizon | Exact change in expected log loss against raw and against projection alone | Present if, with the exact budget under W0, the over-reactive, under-reactive, and independence families each improve on both comparators with the paired 95% interval excluding 0. Usable outside the twin if the same holds for the selected variant under W2 with the estimated conservative budget, and the eligibility test admits an overestimated budget in at most 10% of stress cases | No improvement even with the exact budget | CPU |
| E2 | Which regime are language-model readers in? | Panel of at least eight readers on twin text, exact enumeration | Figure 1 coordinates; exact change in log loss under each repair | Regime call by §10.2; repairs judged by the E1 rules | Every reader indistinguishable from R0, so there is nothing to audit | About 90 GPU-hours |
| E3 | Do structured readers on charts show the same accounting? | Cross-fitted replay, R2 and R3 with W1, W2 | Change in posterior log loss; identity terms; budget calibration | Benefit if the interval for R2-stored excludes 0; otherwise report the terms | Not applicable: measurement | CPU, credentialed |
| E4 | Which regime is the trained agent in, and does the layer help its final belief? | Fixed-evidence replay on cohorts P, S, X, B; arms A1 and A1s by replay | Change in final-belief log loss on cohort X | G3: the interval excludes 0 in the improving direction and A1's T5 is not below V1 in point estimate on cohort X | Interval excludes 0 in the harmful direction, or no measurable violation | About 180 GPU-hours |
| E5 | Does coherent selection add anything in the closed loop? | Arms A2, A3, and A4 if triggered, on validation and cohort X | T5 contrast A2 − V1 with cost | §11.2 | Integrity failure after two repairs, or negative direction | About 400 GPU-hours |
| E6 | Does it hold on the frozen test protocol? | §11.3, five policies, once | T5 contrast A1 − V1 | Single uncorrected test | Reported as found | About 145 GPU-hours |
| E7 | Is the finding specific to our agent and environment? | Four public complete-record datasets × six readers × two pairings | Agreement of predicted and realized gain across cells | The accounting transfers if the rank correlation has a 95% interval above 0 under a hierarchical bootstrap and the sign of the realized gain is predicted in at least 80% of cells whose predicted-gain interval excludes 0 | Interval includes 0, or sign agreement below 50% | About 150 GPU-hours |
| E8 | Is the agent's regime a property of the recipe or of one checkpoint? | Four new seeds of the primary recipe | Between-policy spread of the E4 and E6 quantities | Descriptive; feeds E4 and E6 | Not applicable | About 105 GPU-hours |
| E9 | Does it depend on the backbone? | Three seeds on a second backbone | Same quantities | Descriptive; secondary family | Not applicable | About 120 GPU-hours |
| E10 | Can clinical effects be estimated with adequate precision? | Five cross-fitted fold policies; cohort X | Coverage: every admission scored by a policy that never trained on it | Integrity only, checked by fingerprints | Any leakage between a fold policy and its held-out admissions | About 133 GPU-hours |
| E11 | Does the audit hold on an independent cohort? | Audit and replay only, if a cohort is listed | Figure 1 coordinates | Card frozen before access | Not applicable | Not estimated; blocked until listed |

Thresholds here are operational. The 0.20 margin, the 20% harmful-application trigger, the 99%
conformance rate, the 80% sign agreement, the 10% admission rate, and the 5% responsiveness cut
were chosen on November 12 and 13, 2026 before any run of this study; they are not borrowed from
another contrast and are not equivalence margins.

## 14. Gates and compute

### 14.1 Gates are decision rules

The agent evaluates each gate from recorded evidence, writes an immutable record in GATE_DECISIONS/
with the inputs' digests, the rule, the result, and the path taken, and continues. No gate waits
for a person.

| Gate | Rule | If met | If not met |
|---|---|---|---|
| G0 | M0 acceptance | M1 and M2 proceed | Work only on the missing items; launch checks that cannot be answered become BLOCKED items |
| G1 | Repairs needed by E0–E3 and by training verified | E1, E3, and the training track proceed | A failed repair blocks only its dependents |
| G2a | E0 passed; twin repairs verified; E1 completed and interpretable | Measurement tier runs: E2, E4, E7 | Fix and repeat E0 or E1; nothing downstream runs on an unvalidated instrument |
| G2b | E1-B says projection repair present | Method arms stay eligible | E5 is skipped; the measurement tier still runs; the paper is the mechanism paper |
| G2c | E1-D says scale repair usable outside the twin | CEE-Scale may be applied on charts and public data | CEE-Scale is reported from the twin as found and not applied elsewhere; tau_B is still reported |
| G3 | E4 rule met; §8.3 and §8.5 verified | E5 runs; A1 is eligible for the test protocol | E5 and E6 are skipped and the test split is left untouched; write up the replay result |
| G4 | G3 met; E5 completed or skipped by G2b; test protocol frozen and committed | E6 runs once | E6 does not run |
| G5 | M5 assessment | Recommendation recorded | Recommendation recorded |

### 14.2 What a gate never does

A gate never widens AUTHORIZATION.md, never substitutes for the data-use check, never selects an
arm by a T5 comparison, and never reopens test admissions. A skipped experiment is reported as
SKIPPED BY GATE with the record that skipped it.

### 14.3 GPU estimates tied to the experiment graph

Estimates, not caps on the science. Measured bases: 0.458 GPU-hours per 240-admission rollout arm,
and 25 h 5 min to train the primary policy on one GPU. Everything else rests on stated assumptions
about calls per admission and throughput, which a bench of at most 3 GPU-hours per experiment
replaces before the main run. The allowance in AUTHORIZATION.md is about twice the estimate.

| Experiment | What is counted | Estimate (GPU-hours) | Basis |
|---|---|---|---|
| E2 | 9 readers × 1,000 patients × 3 renderings × about 390 scored prompts | 90 | Assumed throughput by model size; no measurement yet |
| E4 | 4,320 admission-configurations: P twice, S four times, X once, B three times | 180 | About 150 GPU-seconds per admission: 3 result steps × 12 tests × 32 imagined results, written and scored |
| E5 | 3 arms × 2,400 admissions, plus two repair runs on 240 | 400 | About 180 GPU-seconds per admission per arm |
| E6 | Test replay for 8 policies; A2 for 5 if eligible; reference rollouts | 145 | E4 and E5 rates; 0.458 per rollout |
| E7 | 4 datasets × 6 readers × 2 pairings | 150 | Assumed; bench replaces it |
| E8 | 4 trainings and their validation references | 105 | 25.6 per training, measured |
| E9 | 3 trainings on a second backbone, bench, up to 2 repairs | 120 | Assumed near the primary recipe's cost |
| E10 | 5 trainings and 2,160 unguarded rollouts | 133 | 25.6 per training; 0.458 per 240 rollouts |
| Benches and conformance | | 10 | |
| Total | | 1,333 | About 167 hours of wall-clock with 8 GPUs fully used |

A training job exceeds 24 hours and is split into resumable parts (DEFAULT D3). GPU jobs have no
burst QOS on HiPerGator. Diagnostic-test dollars and GPU-hours are different quantities and are
never mixed. If compute after December 13 turns out to be limited (DEFAULT D2), nothing is cut:
the remaining runs are queued under whatever is available and the schedule of §17 moves.

## 15. Statistics and pre-registration rules

The unit is the admission on charts and the patient in the twin and on public data. Repeated
orders, prefixes, renderings, and items from one admission or patient are never treated as
independent. Intervals: 95% paired percentile bootstrap with 10,000 resamples, class-stratified for
T5. T5 contrasts use the existing two-sided paired sign-flip randomization test, and reports name
the null it tests. One primary contrast per experiment is uncorrected; other contrasts are grouped
into named families with Holm adjustment. Twin experiments report paired per-seed differences
across twenty environment seeds and keep generation, behavior, fitting, corruption, and Monte Carlo
seeds separate by purpose.

Several policies. A result averaged over policies resamples admissions and policies together; the
between-policy spread is reported beside it and is never used to shrink an interval. On cohort X
the fold policy is part of the treatment of each admission; intervals resample admissions within
folds. Across datasets and readers, intervals use a hierarchical bootstrap, datasets then patients.

Every fitted quantity that touches labels outside the training split (reader calibration, gate
thresholds, budgets, any temperature) is fit on folds that exclude the scored admission. A minimum
detectable effect is recomputed for each clinical contrast from its own discordance and is never
used as an equivalence margin. Nonsignificance is reported as no detectable effect at the stated
sensitivity. Log loss, Brier score, accuracy, T5, cost, requests, and runtime are reported
together; an improvement in one is not described as an improvement in another.

## 16. One owner, and the internal module boundaries

Daniel owns every part of the study and the executing agent does the work. The boundaries below
are internal: each module has conformance fixtures and is changed only with a recorded decision.

- Reader: belief(history) and read(history, test, outcome), each returning a probability vector
  over the disease list. Deterministic. No access to labels or unrevealed results.
- World model: outcomes(history, test, k, rng) returning outcomes with weights that sum to one and
  include UNAVAILABLE; it enforces evidence identity. Class-conditional models also expose
  budget(history, test).
- CEE library: project(prior, weights, posteriors) returning u, log Z, corrected posteriors, KL_inc,
  convergence status; gain_terms(...) returning the realized gain and the three terms;
  scale(prior, weights, posteriors, budget, variant) returning tau_B and the repaired posteriors;
  tau_ratio(...); voi(...); commit(...); gate(...); and the audit command of §9.1.
- Counterfactual environment, through adapters over the existing synthetic study:
  all_outcomes(patient), true_predictive(history, test), true_posterior(history),
  true_information(history, test), exact_value(selector, reader, commit), render(history, template),
  case fingerprints.
- Record schema, one row per decision point per arm: experiment and arm, policy identity, case key,
  step, test, K, prior, drift, responsiveness, KL_inc, tau_B, tau_ratio, budget and its source, raw
  and coherent scores, action and fallback, gate decision, realized fields, cost fields, seeds by
  purpose, component digests. Clinical rows are private; only schema-approved aggregates are
  exported.
- Statistics: paired_interval(records, metric, unit), t5_contrast(...), holm(...), mde(...),
  hierarchical_interval(...), consumed by every experiment so that none computes its own intervals.
- Pipeline: the stage contract, manifests, the watcher, and the notification queue of §20.

Existing code by teammates is reached only through thin adapters with conformance fixtures for
observation, cost, and stopping semantics; matching function names is not conformance. Their
branches are never edited, and their authorship is recorded wherever their code or results are used.

## 17. Venue and schedule

### 17.1 Venue

Read from the official pages on November 13, 2026:
- ICML 2027: abstract registration January 14, 2027 and full paper January 22, 2027, anywhere on
  earth; reviewing February 24 to March 16; decisions May 3. The call for papers and the author
  instructions are not yet published, so the page limit, anonymity, and dual-submission rules for
  2027 are unknown. ICML 2026 allowed 8 pages plus unlimited references, impact statement, and
  appendix, forbade parallel submission of substantially similar work, allowed preprints, and
  forbade links to non-anonymized repositories; these are assumed until the 2027 call appears
  (DEFAULT D12).
- Published for 2027: generative AI tools are permitted if disclosed in a mandatory AI Disclosure
  Statement; authors are fully responsible for all content; AI systems cannot be authors. No author
  may be on more than 20 submissions. Each author may be on at most one submission with no eligible
  reciprocal reviewer; if any author is eligible, one must be designated at abstract registration
  and must review.
- The 2027 calls of NeurIPS and UAI were not published. Third-party sites project their dates from
  2026; those projections are not used here.

Fit. The plan below completes the GPU critical path by December 12 and a full draft by January 8.
ICML 2027 fits if gate G3 is decided by about December 3 and nothing blocks E4. It does not fit if
the reader probe cannot be made to conform (§7.4a), if either data flow is blocked and its
alternative path also fails, or if compute stops on December 13 with runs outstanding. In those
cases the study continues to completion and goes to the next strong venue whose published call
allows it, decided by §17.3. No experiment is shortened for a date.

### 17.2 Dates

| Date | What |
|---|---|
| Fri Nov 13, 2026 | Version 3.1 committed on the task branch. Daniel signs by pasting the launch prompt; the launch session records SIGNATURE.md and installs the keeper. |
| Nov 14–19 | M0 and its launch checks; E0; twin repairs; §8.3 preflight; pipeline, watcher, benches. Training track (E8, E9, E10) submitted by Thu Nov 19. |
| Nov 20–26 | E1 parts A to D; E3. E2 and E7 running. Training finishes: about 360 GPU-hours, under two days on 8 GPUs plus queue time and 24-hour splits. Gates G2a, G2b, G2c by Wed Nov 25. |
| Nov 27–Dec 3 | Probe conformance for every policy; E4 on cohorts P, S, X, B; E2 and E7 finish. Gate G3 by Thu Dec 3. |
| Fri Dec 4 | Course checkpoint (DEFAULT D1): a report built from whatever has finished. It changes nothing in the plan. |
| Dec 4–10 | E5. Gate G4 by Thu Dec 10. |
| Dec 10–12 | E6, once. The GPU critical path ends Sat Dec 12, inside the window of DEFAULT D2. |
| Dec 13–Jan 2 | M5 robustness; analysis; figures from manifests; drafting. |
| Sun Jan 3, 2027 | First complete draft. Review round one, Jan 4–5. |
| Fri Jan 8 | Full draft, two weeks before the deadline. Review round two with independent re-derivation, Jan 8–10. |
| Mon Jan 11 | Go or no-go (§17.3). |
| Thu Jan 14 | Abstract registration, by Daniel, if go. |
| Jan 15–20 | Review round three; anonymization check; supplement; statements. |
| Fri Jan 22 | Submission, by Daniel, if go. |

GPU critical path with 8 GPUs fully used: training about 45 hours; E2 and E7 about 30; E4 about
23; E5 about 50; E6 about 18. About seven days of saturated use inside a four-week window, so queue
time and retries have room. With more than 8 GPUs, E2 and E7 run beside training.

### 17.3 Go or no-go on January 11, 2027

Go only if all of these hold:
1. Every experiment of §13 is COMPLETED, SKIPPED BY GATE, or BLOCKED on an item only Daniel can
   resolve. None is still running and none was shortened.
2. E1, E2, E4 on cohort X, and E7 are completed: the minimum for the mechanism paper. Closed-loop
   and test claims appear only if E5 and E6 completed.
3. Review round two has no open correctness finding, and every headline number was re-derived from
   the manifests by an instance that did not produce it.
4. The 2027 call is published and the manuscript conforms to it.
5. The author list and the reciprocal-reviewer designation are settled by Daniel and Professor Liu.

If any fails: no reduced submission. The study runs to completion, the agent re-reads the official
calls that are published by then, and the recommendation names the next venue and its confirmed
dates. Missing a deadline is an acceptable outcome; a weakened paper is not.

## 18. Teaching and professor discussion

The professor packet answers: what did we try, why, what was held fixed, what happened, what does
it imply and not prove, and what decision do we need. PROFESSOR_BRIEF is about two pages with the
question, two or three findings, uncertainty, and the next decisive experiment on the first page,
every number tied to an experiment ID. PLAIN_ENGLISH_GUIDE explains, with invented examples only:
world model, reader, conservation of expected evidence, drift, evidence scale, information budget,
redundancy, responsiveness, projection, value of information, commit rule, regret,
counterfactual, T5, log loss, calibration, cross-fitting, and why an exact environment is needed.
MEETING_TALK_TRACK holds a 90-second explanation, a five-minute outline, likely questions, three
questions for Professor Liu, and claims to avoid.

Distinctions to keep visible:
- The recorded large gain came from recovering answers the hypothesis component already held.
- A belief that becomes more confident has not thereby become more correct.
- Coherence is not correctness: a coherent agent can be coherently wrong.
- Lower log loss is not higher accuracy.
- Imagined results are predictions, never observations.
- No detectable benefit is not no possible benefit.

Do not claim a valid content-quality pass before repair, an optimal clinical oracle, a fully
out-of-fold C1, universal uselessness of availability, that all synthetic results leaked, that the
projection fixes double counting or wrong scale, that CEE-Scale works without a valid budget, that
sandbox numbers are project results, or that any of this supports clinical use.

## 19. Reproducibility, review dispositions, completion, and safeguards

Every experiment has a manifest tying together: study and experiment ID; legacy repair or new
intervention; source SHA and clean-tree status or snapshot digest; card, protocol, configuration,
analysis, dependency, and hardware identities; split provenance and exposure without private
identifiers; the full contract of §6.1 with digests of fitted inputs; seeds by purpose, pairing
unit, counts, weights, uncertainty method; authorization digest, allowance, command, job IDs with
every attempt, resource receipts, exit state; immutable input and result locations, resume history, approved aggregates. Identity is
bound before execution and verified on resume.

| Review findings | Disposition in 3.0 |
|---|---|
| 001: likelihood and reference | §8.2; needed by E3 |
| 002, 006, 011: synthetic split, cost, provenance | §8.1; needed by E1, E2 |
| 003: preflight, artifacts, resume | §8.3; needed before any credentialed GPU run |
| 004: controls | §6.1 contract; references in §9.2 |
| 005: gain semantics | Resolved by design: the legacy score is named an entropy-change surrogate; §5.6 defines the coherent quantity |
| 007: weak model-free comparator | §6.3 matched-information baseline |
| 008: replacement consistency | §8.5; needed before E5 |
| 009: tuning scope | §15 cross-fitting; legacy C1 stays labeled developmental |
| 010: null overstatement | §3.5 and §8.6 |
| 012: dependency and successor freeze | §7.3 cards; study ID TII-CEE-1 |
| 013: authorization precedence | §3 |
| 014: reporting duplication | §4 |

At milestone or session handoff state: what changed, what ran, what was only inspected, reanalyzed,
or rendered; the strongest supported conclusion and the most important open question; acceptance
items met and outstanding; experiment state; scientific outcome; paths of reports and manifests;
exact continuation commands; open BLOCKED items; and, independently, whether the
implementation is valid, whether a benefit is demonstrated, and whether publication evidence is
sufficient.

Do not mark future milestones complete because their plan exists, or unstarted work failed. Do not
turn analysis of old episodes into a claim of new replication.

Final safeguards: no protected-data export, fabricated attribution, hidden failed runs,
unauthorized compute or test access, rewritten historical study, merge, push beyond the agent's
own task branch, change to AUTHORIZATION.md, contact with the professor or teammates, or course or
conference submission. Deliver a useful professor packet even when the correct conclusion is
negative, incomplete, or blocked.

## 20. Autonomous execution and resilience

WATCHER_DESIGN.md gives the design in detail; this section is the contract it implements.

### 20.1 Principle and stop conditions

The executing agent carries the study from M0 through M5 and the manuscript without waiting for
routine approval and without depending on Daniel's terminal staying connected. It stops only when:
1. the study is complete (§20.11);
2. every remaining item is BLOCKED on something only Daniel can do; or
3. a safety or data-use rule stops it.
A dropped connection, a closed laptop, an expired session, a failed job, or a negative result is
not a reason to stop.

### 20.2 One upfront authorization

AUTHORIZATION.md is committed with this specification and is never edited after it is hashed.
Daniel signs it once, by a statement that quotes its SHA-256 and the commit that contains it. The
launch session records that statement verbatim in SIGNATURE.md, commits it, and binds the hash.
AUTHORIZATION.md names: the
experiments; a GPU allowance for each, with its basis; the Slurm account, QOS, and partitions; the
largest number of GPUs in use at once; the permitted models; the public datasets and whether the
agent may download them; any other credentialed cohort already in approved storage; the one-shot
frozen test protocol, to run once, automatically, when gate G4 is met and the protocol is frozen
and committed; whether the agent may push its own task branch; and the watcher, the local loop,
and notifications.

The agent never signs, edits, extends, or reinterprets AUTHORIZATION.md. Its SHA-256 is bound into
every manifest. If the file is missing, expired, or changed, or SIGNATURE.md is missing or quotes
a different hash, the watcher submits nothing
new that needs it, running jobs finish their current part and checkpoint, and a BLOCKED item is
raised. Work that needs no authorization continues.

### 20.3 Gates, and the BLOCKED protocol

Gates are scientific decision rules (§14.1). The agent evaluates them, records the decision with
its evidence, and takes the pre-registered path by itself.

Anything outside AUTHORIZATION.md, and anything only Daniel can supply, becomes an entry in
BLOCKED.md: an identifier; what is blocked and which experiments depend on it; the exact request,
written so Daniel can act on it in one step; the alternative path being taken meanwhile, if one is
pre-registered; and the date. A notification is sent once. Independent work continues. The agent
does not work around a BLOCKED item, does not ask again, and does not treat silence as consent.

### 20.4 Data-use compliance is not waivable

Two data flows are new in this study:
- F1: a validation-stage or test-stage job reads training-split result text at inference (the
  retrieval world model W3).
- F2: a job scores a trained policy on a real history with an imagined result substituted (the
  reader probe).

Neither Daniel, nor AUTHORIZATION.md, nor the agent can waive the data-use terms. A flow proceeds
only if the agent confirms, from the text of the terms and the project's storage rules, that it
fits. The confirmation is DATA_USE_CHECK.md, written in M0:
1. Identify the governing documents: the PhysioNet Credentialed Health Data Use Agreement in the
   version Daniel signed (DEFAULT D11: 1.5.0), the dataset's own license page, PhysioNet's notice
   on using its data with language models and online services, and the project's rules for
   approved private storage. Record title, version, URL, retrieval date, and a digest of the
   saved copy.
2. For each flow answer, with the clause quoted: Does any patient-level content leave approved
   private storage? Does any third party, service, or API receive it? Does any person other than
   the credentialed user gain access? Does it create a new copy, index, or derived artifact, and
   where is that stored? Does it raise the risk of identifying a person or institution? Is the
   purpose lawful scientific research under the agreement? Does it conflict with the project's
   own stage-by-split rules, and if so which pre-registered amendment covers it?
3. A flow passes only if every answer can be read from the documents. If one cannot, the flow is
   BLOCKED with that exact question, and the pre-registered alternative is taken:
   - F1 blocked: W4, the on-cluster generative sampler, replaces W3 everywhere.
   - F2 blocked: no experiment uses R1. E4, E5, and E6 are BLOCKED; E0 through E3 and E7 proceed;
     the paper is written without the trained-policy claims.

Standing consequences of the same terms: patient-level content never reaches the hosted coding
agent or the Mac; every model that reads patient text runs on HiPerGator; generated clinical text
and retrieval indexes are protected data.

### 20.5 Work lives on HiPerGator

Every experiment runs as Slurm jobs submitted from the repository's pipeline, never as a process
tied to a terminal.
- Stages. Each experiment is a graph of stages. A stage is idempotent: it finds its completion
  marker and stops, or finds a checkpoint and resumes, or starts. Outputs are written to a
  temporary name and renamed when complete.
- Identity. On every start and every resume a stage verifies its manifest (§8.3): code revision,
  card, fitted artifacts, inputs, and the authorization digest. A mismatch is a refusal with its
  own exit code; it is never retried and never creates a new test key.
- Checkpoints. Jobs request at most 24 hours (DEFAULT D3), trap the scheduler's warning signal,
  checkpoint, and exit with a "continue" code. Training resumes from the last saved step; rollouts
  and replays resume from the last completed admission.
- Accounting. GPU-hours come from the scheduler's accounting for every attempt, completed or not,
  and are charged to the experiment's allowance before the next submission.
- Login nodes run nothing but short management commands. No long-lived process runs there.

### 20.6 The watcher

A recurring Slurm job, scheduled with scrontab, which UF Research Computing documents for
recurring jobs, with a singleton dependency so cycles never overlap. If scrontab jobs cannot call
the scheduler, the fallback is cron on the daemon server that UF Research Computing provides for
small recurring tasks. Cycle: every 15 minutes (DEFAULT D13), one accounting query and one queue
query per cycle, because frequent polling of the scheduler is not allowed.

Each cycle the watcher:
1. takes a lock, and exits if another cycle holds it;
2. checks AUTHORIZATION.md's digest;
3. reads job states and classifies each ended job: completed, continue, timeout, preempted, node
   failure, out of memory, refusal, or other failure;
4. resubmits from the last checkpoint within a bounded retry count: three for timeout, preemption,
   and node failure; one for out of memory, with more memory; one for an unclassified failure;
   none for a refusal. Every attempt, with job ID, state, elapsed time, GPU-hours, and exit code,
   is appended to the run log and the ledger. A stage that exhausts its retries becomes FAILED
   with a notification, and only its dependents wait;
5. advances the graph: submits stages whose dependencies are complete, within the concurrent-GPU
   limit and the allowances;
6. evaluates any gate whose inputs are complete, by a deterministic script over approved
   aggregates, and writes the record;
7. rewrites STATUS.md and appends events to the notification queue, aggregate content only;
8. when no stage is pending, running, or retryable, writes the final status, queues the completion
   event, and disables its own schedule entry, leaving any other entry untouched.

The watcher runs no analysis and makes no scientific judgment beyond the scripted gate rules.

### 20.7 The local loop on the Mac

A launchd job on Daniel's Mac, built by the agent in M0. Every 10 minutes, through the existing
authenticated SSH connection only, it reads STATUS.md and the notification queue, copies newly
approved aggregates, and, when a stage has completed or a gate was decided, starts or resumes an
agent session in the repository with a fixed continuation prompt: read STATUS.md, the ledger, and
BLOCKED.md, then carry on with analysis, code, and writing. One session at a time, guarded by a
lock. If the connection is down, the loop does nothing and tries again later; work on HiPerGator is
unaffected. The loop never opens a new SSH connection by itself.

### 20.8 The connection keeper

Kept as ready-to-install files in keeper/, with keeper/KEEPER_INSTALL.md. Daniel authorized its
installation on November 13, 2026: the launch session installs it once, exactly as that guide
describes, and no agent changes or removes it afterward. It holds Daniel's existing authenticated
session open using only what OpenSSH and macOS provide and what UF Research Computing documents:
connection multiplexing, keepalives, a launchd agent that checks the master connection and sends a
trivial command through it, and caffeinate. It never automates, intercepts, replays, or bypasses
the Duo prompt; never stores a password or a second factor; never attempts a login; and never
weakens a UF security control. If the session drops, it tells Daniel once to sign in, the watcher
keeps everything running meanwhile, and the local loop resumes through the new connection.
UF Research Computing's multiplexing page states no maximum session length and is silent on
keepalives (DEFAULT D13).

### 20.9 Notifications

A macOS notification and a line in STATUS.md for each of: a stage completed; a gate decided; a
BLOCKED item; a failure that exhausted its retries; a reconnect needed; the study completed. The
text carries names and counts only, never data.

### 20.10 What may cross the SSH channel

Commands the agent or the loops run on HiPerGator return only: scheduler metadata, stage and gate
names, exit and error codes from an approved list, counts, timings, GPU-hours, and aggregates that
passed the export guard. The agent never prints, pages, greps, or copies patient-level files, and
writes tools that summarize on the cluster instead. A command whose output could contain protected
content is not run.

### 20.11 Completion

The study is complete when every experiment of §13 is COMPLETED, SKIPPED BY GATE, or BLOCKED; M5's
assessment and the manuscript with three recorded review rounds exist; and the watcher has
disabled itself. The agent then delivers what the kickoff lists under "Finish with evidence" and
stops. It does not submit anything anywhere.

### 20.12 Failure modes the design must survive

| Event | Required behavior |
|---|---|
| Laptop closed, network lost, session expired | HiPerGator work continues under the watcher; the local loop resumes when a connection exists |
| Job killed at its time limit | Checkpoint, "continue" exit, resubmission; no duplicated or lost admissions |
| Node failure or preemption | Requeue from checkpoint; attempt logged |
| Same stage started twice | Second instance finds the lock or the marker and exits |
| Code or artifact changed mid-run | Identity refusal; no retry; BLOCKED if the change was not a recorded decision |
| Allowance would be exceeded | No submission; BLOCKED with the estimate and the request |
| Authorization missing, expired, or changed | Nothing new that needs it is submitted; BLOCKED |
| Watcher itself fails | The next scheduled cycle starts clean from the state on disk |
| Test run interrupted | Resume under the same manifest, once; never a second run |
