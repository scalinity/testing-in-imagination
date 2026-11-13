# Research ledger, study TII-CEE-1

Canonical for claims, review dispositions, the experiment register, and every job attempt
(specification section 4). Machine-readable result manifests are canonical for numbers; this
ledger cites them. Times are US Eastern.

## 1. Evidence register (M0, specification 7.1)

### 1.1 Inputs, hashed November 13, 2026

| Input | SHA-256 |
|---|---|
| SPEC_RESEARCH_ADVANCEMENT.md, version 3.1 | 8f48329862060ff9b63325b5f45dc15bf610912e0955ff87614d7b48262f9ca6 |
| KICKOFF_RESEARCH_ADVANCEMENT.txt, version 3.1 | e8c4b25657cf45925cba06809c5b4e97f706eb1b393a30f6fa02b717f2d1199a |
| AUTHORIZATION.md | 72ccc89c8a825c82578c8efdf942025e3ce60d72f179b8b13489737ee37221ae |
| AUTHORIZATION_ADDENDUM_1.md | f59cd6fa638d00b2d1bc9bdc4848ba9ae412f9abecb7cdfcfa83cb4475e05658 |
| WATCHER_DESIGN.md | a36d6f6c066be170c400380f51102dc57c7a13f4372a02f0c03ffee30e4bcc04 |
| PRIOR_WORK_MATRIX.md (as delivered) | 8bfaef31951bcf9701e66f67a938daab0b9a70638fe3f077cc21c5d13aa795fe |
| PROOF_GAIN_IDENTITY.md | 5296a7a5268be08b9a46fdb64e3f048714eb3df32209f76f682396a7a4ac8b4f |
| cee_check.py | 5e29545ecc77b8d3522a7911da456ca9872dc1e193225314f112c1b7d416bb26 |
| cee_check_output.txt | e942fc7787b988d7e7233dfe908dcb72b93df463fdc467cc3524e145e403f049 |
| REVIEW_AND_REDESIGN.md | fde54e9737c5aa12ba446fe4c8bf2b806b6d1fa9503a9b1eb4a18f9644ea1c52 |
| keeper/KEEPER_INSTALL.md | a8e09e2eeeb29b0226e0e1714544cfa3ad5429e1c7cfdc6bb8220c099d0ce5bf |
| keeper/hpg_keeper.sh | d16ad75a2edc8b0e2b2ef65297f47946db51d3a20ae42bcebad597c5d03cfbb6 |
| REVIEW_HANDOFF.md at 91b21ae (read with git show) | 00e4dd01ed42f08ba851d21ebb676bae40ec4217439d48d2d05e7560dcab765a |

### 1.2 Repository references, read November 13, 2026

| Reference | Identity | Change since the specification's November 12 snapshot |
|---|---|---|
| Task branch tii-cee-1/2026-11-13 | started at 632bd18; local only | Study commits since: b1a3614 (signature), c5902ea (addendum), 7a5b3c1 (addendum authorization) and later |
| main (remote) | e5ad93fad5345f211cedc05c3dc491b101c63fa5 | None |
| PR #9, feat/content-wm-m1 | acf5254bb624ed881f864e84149fc4d983dc048a | None |
| PR #7, feat/content-wm-spec | 75c50a725655ad4cbe69d4ab79677c40ed4b309e | None |
| PR #1, edwin-salcedo/synthetic-world-model | df467eadd07fe2bc761b618dafce8ed9f198641a | None |
| docs/research-advancement-v2 | 91b21ae2b0127a42cfd406ec6a51339dc11c19a4 | None |
| Remote visibility | PUBLIC (`gh repo view`) | Nothing is pushed (AUTHORIZATION.md section 9) |

### 1.3 Execution access, November 13, 2026

- HiPerGator: an existing SSH master for daniel.escalante@hpg.rc.ufl.edu was running; every
  command used it with BatchMode=yes, ControlMaster=no and ProxyCommand=false. The connection
  keeper is installed (STATUS.md section 3).
- The compute fields of AUTHORIZATION.md section 3 were checked live: all match (STATUS.md
  section 4). The study root was created with owner-only permissions.

### 1.4 Instruction precedence

1. Current instructions from Daniel in the session, and any read-only restriction.
2. AUTHORIZATION.md, signed (SIGNATURE.md), with AUTHORIZATION_ADDENDUM_1.md, authorized by
   Daniel's direct statements recorded verbatim in SIGNATURE.md.
3. The kickoff, version 3.1, and the specification, version 3.1 (bound by hash).
4. WATCHER_DESIGN.md and keeper/KEEPER_INSTALL.md for the mechanisms they describe.
5. Historical records: ONBOARDING.md, onboarding/, cdm-extension/daniel-escalante/DECISIONS.md and
   SPEC.md, REPRODUCTION.md. ONBOARDING.md itself says that for this study the authorization and
   specification govern where they differ. The DECISIONS.md delegation to commit approval files on
   Daniel's behalf does not apply to this study (specification 3.1; REVIEW-013).

### 1.5 Authorization state

- AUTHORIZATION.md SHA-256 equals the value SIGNATURE.md quotes: yes.
- Valid until March 31, 2027.
- AUTHORIZATION_ADDENDUM_1.md: authorized November 13, 2026, 2:55 AM EST; adds GPU account
  cai5724 (16 GPUs at once in all) and the CPU QOS of the other associated accounts.

## 2. Claim-to-evidence register

Starting findings of specification 2.3. None was re-run in this study; each is labeled by what it is.

| Claim | Source | Evidence type | Split and exposure | Strongest competing explanation | Next discriminating check |
|---|---|---|---|---|---|
| Test T5: original 73.09, V1 84.45, V3 81.83; V3 − V1 −2.62 [−6.16, 0.76] | cdm-extension/daniel-escalante/results/credentialed/phase-a/REPORT.md | REPORTED_RUN | Test, previously examined | V1's gain is answer recovery from hypotheses already held, not better planning | E4 and E6 compare every arm with each policy's own V1 |
| Validation oracle − uniform availability −0.11 [−2.56, 2.38]; unavailable requests 203 → 66 | results/credentialed/DIAGNOSTICS.md | REPORTED_RUN | Validation | A modest effect below the design's sensitivity; other planners | None planned; wording erratum (REVIEW-010) |
| Dose response 4.384 GPU-hours; slope −2.46 [−7.28, 2.26] | results/credentialed/analyze-dose-val-b/dose_response.json | REPORTED_RUN | Validation | A decline of a few T5 points over the range | None planned |
| Check H: of 61 added relevant results, 38 arrived while the hypothesis held the label; 7 changes, 3 toward it | PR #8 hypothesis_check.json | REANALYZED_FIXED_DATA (exploratory) | Validation | Stored hypotheses are not the policy's belief | E4 reads R1's belief directly |
| Content CPU: posterior NLL prior 0.69, uncorrected 2.485, A+B 0.637, tau 13; ECE 0.0455 prior, 0.0705 A+B | PR #9 wm-fit-c/content_wm.json at acf5254 | REPORTED_RUN, developmental, partially out of fold | Validation | Tuning on validation labels (REVIEW-009); mismatched quality estimand (REVIEW-001) | E3 under cross-fitting with matched scoring |
| 4 of 10 predicted-gain deciles negative | Same | REPORTED_RUN | Validation | Monte Carlo noise rather than incoherence | E1 by exact enumeration; PROOF section 5 decomposition |
| Availability bits alone predict the diagnosis: test T5 43.13 | Phase B leakage probe | REPORTED_RUN | Test | None: privileged chart signal | Recorded as a benchmark-validity fact |
| 0.458 GPU-hours per 240-admission arm; training 25 h 5 min on one RTX PRO 6000 | Replication and tim run logs | REPORTED_RUN | Not applicable | Different arms or loads | Benches before each main run |
| Sandbox: projection repairs drift readers, hurts wrong-scale readers; budget matching repairs all five families with the true budget | cee_check.py, PROOF_GAIN_IDENTITY.md | HYPOTHESIS for the project (sandbox re-implementation, oracle world model) | Twin parameters only | Re-implementation errors; oracle world model | E0 and E1 |

## 3. Review dispositions, REVIEW-001 to REVIEW-014

Adjudicated November 13, 2026 against the code as it stands: main e5ad93f, PR #9 acf5254, PR #1
df467ea, all unchanged since the review.

| ID | Still present in current code? | Disposition in this study | Needed by |
|---|---|---|---|
| 001 likelihood estimand and reference | Yes, PR #9 tim/content.py at acf5254 | Repair in this study's namespace through an adapter (spec 8.2): matched per-item scoring, pooled training-only reference, joint likelihood as a separately named metric, hand-computed fixtures. PR #9 is not edited | E3 |
| 002 synthetic split | Yes, df467ea | Repaired in tii-cee/tiicee/twin adapters: streams namespaced by purpose and split, fingerprints. Edwin's branch untouched | E1, E2 |
| 003 preflight and artifact identity | Yes, PR #9 | Superseded for this study by the shared preflight and immutable manifests of tii-cee/tiicee/pipeline (spec 8.3) | Every credentialed GPU run |
| 004 controls | Design issue | The arm-component contract of spec 6.1 is declared on every card | Every arm |
| 005 gain semantics | Yes, legacy score | Resolved by design: the legacy score is named an entropy-change surrogate; coherent VOI is defined in spec 5.6 | Reporting |
| 006 synthetic cost | Yes, df467ea | Repaired in the twin adapter: expected charged cost = p_available × cost | E1, E2 |
| 007 weak model-free comparator | Yes, df467ea | Kept under its accurate name; the matched-information baseline of spec 6.3 is added | E1 baselines |
| 008 replacement thought | Yes, PR #9 `_plan_content` | Repair before any arm that replaces an action (spec 8.5); neutral thought, full-text and token-mask tests | E5 |
| 009 C1 tuning scope | Yes, record | Legacy C1 stays labeled developmental; this study cross-fits every fitted quantity (spec 15) | Reporting, E3 |
| 010 availability-null wording | Yes, DIAGNOSTICS.md on main still says "a better availability predictor cannot help it" | Linked erratum with the scoped wording of spec 3.5; no number changes; the historical file is preserved | M1 |
| 011 synthetic provenance | Yes, df467ea's matrix.json records base 22d8a32 | Historical artifact kept; this study binds the vendored snapshot's digest in every manifest | E1, E2 |
| 012 reader dependency and successor freeze | Plan issue | Resolved by spec 3.1: study ID TII-CEE-1, frozen cards, reader probe before E4 | M0, M3 |
| 013 authorization precedence | Record | Section 1.4 above | All |
| 014 reporting duplication | Process | One ledger, one STATUS, immutable reports only at milestones or decision-changing events (spec 4) | Reporting |

## 4. Experiment register

| ID | Status | Card | Notes |
|---|---|---|---|
| E0 | COMPLETED (VERIFIED_CODE, PASS) | EXPERIMENT_CARDS/E0.md | 166 fixtures pass at ef7c4ac; tii-cee/results/E0/manifest.json |
| E1 (A to D) | PLANNED | EXPERIMENT_CARDS/E1.md | Twin in construction |
| E2 | PLANNED | EXPERIMENT_CARDS/E2.md | Waits on G2a |
| E3 | PLANNED | EXPERIMENT_CARDS/E3.md | Waits on REVIEW-001 repair and G1 |
| E4 | PLANNED | EXPERIMENT_CARDS/E4.md | Waits on G2a, probe conformance, data-use check F2 |
| E5 | PLANNED | Frozen at G3 | Waits on G3 |
| E6 | PLANNED | Frozen at G4 | Waits on G4 |
| E7 | PLANNED | EXPERIMENT_CARDS/E7.md | Waits on G2a and dataset licenses |
| E8 | PLANNED | EXPERIMENT_CARDS/E8.md | Waits on the preflight of spec 8.3 |
| E9 | PLANNED | EXPERIMENT_CARDS/E9.md | Waits on the preflight of spec 8.3 |
| E10 | PLANNED | EXPERIMENT_CARDS/E10.md | Waits on the preflight of spec 8.3 |
| E11 | BLOCKED | None | No other credentialed cohort is listed (BLOCKED-002) |

## 5. Job ledger

Every attempt, completed or not, with job ID, state, elapsed time, GPU-hours from the scheduler and
exit code. The watcher's runlog.jsonl on HiPerGator is the primary record; this table mirrors it.

| Stage | Attempt | Job ID | Account | State | Exit | Elapsed | GPU-hours |
|---|---|---|---|---|---|---|---|
| (none yet) | | | | | | | |

## 6. Recorded design decisions

### D-TWIN, November 13, 2026: the counterfactual twin (tii-cee/tiicee/twin)

Built over a byte-exact snapshot of Edwin Salcedo's synthetic study (df467ea; snapshot digest
sha256:4d8ca818fd23b88624b9bd3eba53ec21d61f2af45d05f4fa1568f3386589ac6a). Conformance: the exact
true_information reproduces all twelve published info_gain_nats of results/synthetic/sanity.json to
1e-12; the parameters equal cee_check.py's (largest difference 1.1e-16). 154 tests; 41 of 41
injected faults caught in a mutation check.

1. W1, the legacy count model, has no evidence identity (registry: identity is W2's property); the
   interface docstring now says so.
2. "No phenotype correlation": each test reads its own phenotype draw, so single-test likelihoods
   equal the base variant's and tests are independent given the class (setting p_atypical = 0 would
   also change the marginals). The independence reader then equals R0 exactly.
3. Informative missingness (held out): availability of item i in class y is
   a_i + min(a_i, 1 − a_i) × (hit[i, y] − prior-weighted mean hit), marginal availability
   unchanged; refused unless allow_held_out=True.
4. Duplicated panel: the liver function panel repeated as a second test of the same item and cost.
   Stronger correlation: p_atypical 0.20, the sandbox's stress value.
5. The reduced twin keeps the study's cost normalization ($10,190 for twelve tests), so its six
   costs sum to 0.336 of the normalized scale.
6. Budget is the number of requests, unavailable ones included; each test once; commit forced at the
   budget; step limit as in the study (14 actions).
7. R0 reading an imagined outcome the true model rules out returns its belief unchanged (needed for
   R0 with W1 on the duplicated panel); an impossible real history raises.
8. R4 drift applies at every read, UNAVAILABLE included, so the UNAVAILABLE probe detects drift and
   not scale error.
9. Controlled errors keep P(UNAVAILABLE) at its true value; the matched pair is built by bisection
   to equal KL from the truth (0.020000 for both), "ignored" meaning the reader's imagined mean is
   unchanged and "responsive" its top response axis.
10. W2 is fitted by EM on a latent phenotype model with the twin's structure, all numbers unknown,
    from the fitting stream; deterministic given (model, seed, n_train, smoothing), which the
    manifest records with the snapshot digest and the code commit.
Reference: Bayes-optimal value on the reduced base twin at lambda 1 rises from 0.3988 with no test to
0.6819 with six; 0.8224 at lambda 0. These are VERIFIED_CODE reference values, not results.

### D-CEE, November 13, 2026: the CEE library (tii-cee/tiicee/cee)

166 fixture tests (E0), passing with numpy warnings as errors; 56 of 56 planted faults caught in a
mutation check.

1. The floor (1e-6, then renormalize) applies once, to reader vectors on entry; world-model weights
   are not floored (a zero weight is a point mass), and the tracked coherent belief is never floored
   (flooring it would break the martingale). Inputs more than 1e-6 from summing to one are rejected.
2. tau_B: r' is evaluated on a factor-2 grid of tau and bisected on log tau inside the crossing
   cell to the specified 1% relative tolerance; the result is the largest tau with r' ≥ B, so the
   monotonicity fallback of section 5.11 is a real check (a rise narrower than one grid step is not
   detected). r' was monotone in all 2,320 random and adversarial cases tried. Consequence of the 1%
   tolerance: a coherent reader given its exact budget gets tau_B within 0.27% of 1, not exactly 1;
   E1's 1e-9 no-harm rule concerns the projection, which is exact for a coherent pair, not the
   scale stage.
3. Non-convergence of IPF (tolerance 1e-8, 500 iterations) uses the raw reader as specified: the
   stateless track carries q_real, the stateful track the uncorrected proposal; the step records the
   fallback. At default limits IPF fails at tau = 1/8 in about 3% of random cases and never at
   tau ≥ 1/2, which E1-D reports for the two-sided variant.
4. The CEE-Gate bounds of section 5.8 are upper 90% bootstrap bounds of the absolute value of each
   estimated term (bootstrap of |mean|), resampling caller-supplied units.
5. tau_ratio is undefined (None) for a reader that does not move (r ≤ 1e-12 nats) and is reported
   only as a secondary diagnostic.
6. Names: coherent_voi, gate_allows, scaling.scale; realized_gain is separate from gain_terms.
