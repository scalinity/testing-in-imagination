# tiicee.cee: the CEE instrument

Conservation of expected evidence as a small library: numpy only, no dependency on the twin or on
chart data. It implements section 5 of `reports/research-advancement/SPEC_RESEARCH_ADVANCEMENT.md`
(SPEC below) and the derivations in `PROOF_GAIN_IDENTITY.md` (PROOF). The public surface is
`tiicee.cee.__all__`; everything else in these modules is internal.

All logarithms are natural and information is in nats.

## Functions and the sections they implement

| Function | Module | SPEC | Returns |
|---|---|---|---|
| `floor_norm(p, floor=1e-6)` | core | 5 | `p` floored, then renormalized along the last axis |
| `kl(p, q)`, `entropy(p)` | core | 5 | KL(p ‖ q) and Shannon entropy, 0 log 0 = 0; KL clipped at 0 |
| `project(prior, weights, posteriors, tol=1e-8, max_iter=500)` | core | 5.3 | `Projection`: `u`, `log_z`, `corrected`, `kl_inc`, `converged`, `iterations`, `max_violation`, and the floored `prior`, `posteriors` it used |
| `apply_to_real(u, q_real)` | core | 5.3 | corrected real belief, log Z_real |
| `imagined_mean(weights, posteriors)` | measures | 5.2 | b_bar |
| `drift(prior, weights, posteriors)` | measures | 5.2 | b_bar − b, a vector |
| `responsiveness(weights, posteriors)` | measures | 5.2 | r = Σ_o P_hat(o) KL(q_o ‖ b_bar) |
| `reaction_kl(prior, posterior)` | measures | 5.2 | KL(q ‖ b) for one result: the measurement behind both model-free probes |
| `entropy_change_surrogate(prior, weights, posteriors)` | measures | PROOF 5 | `EntropyChange`: the legacy score H(b) − Σ P_hat H(q_o), its reaction term and its drift term |
| `gain_terms(prior, weights, u, log_z, b_star, p_star)` | identity | 5.4 | `GainTerms`: `incoherence` F(u), `prior_calibration`, `world_model`, `total` |
| `realized_gain(u, log_z_real, y_true)` | identity | 5.4 | g = log u(y_true) − log Z_real |
| `temper(prior, posteriors, tau)` | scaling | 5.5 | b (q/b)^(1/tau), normalized |
| `projected_responsiveness(prior, weights, posteriors, tau)` | scaling | 5.11 | r'(tau) |
| `tau_budget(prior, weights, posteriors, budget, bounds, rtol=0.01)` | scaling | 5.5, 5.11 | `TauBudget`: `tau`, `path`, `monotone`, `converged`, the grid and r' on it |
| `scale(prior, weights, posteriors, budget, variant)` | scaling | 5.11 | `ScaleResult`: `tau_b`, `path`, `u`, `repaired`, `kl_inc`, `converged`, `projection` |
| `apply_scale_to_real(prior, q_real, tau, u)` | scaling | 5.11 | the same tau_B and class weights applied to the real result |
| `tau_ratio(prior, weights, posteriors, grid=...)` | scaling | 5.5 | `TauRatio`: the grid value and the table. Secondary diagnostic, never a repair |
| `phi(belief, utility)` | decision | 5.6 | max_d Σ_y c(y) U(d, y) |
| `coherent_voi(prior, weights, corrected, utility)` | decision | 5.6 | Σ_o P_hat(o) phi(q'_o) − phi(b), non-negative |
| `raw_voi_score(prior, weights, posteriors, utility)` | decision | 5.6 | the same expression on uncorrected posteriors, for the baseline; can be negative |
| `identity_utility(C)`, `recall_aligned_utility(class_freqs)` | decision | 5.6 | 1[d = y]; 1[d = y] / (C pi_y) |
| `expected_charged_cost(p_available, cost)`, `net_value(voi, expected_cost, lam=1.0)` | decision | 5.6, REVIEW-006 | p_available × cost; VOI − lambda × cost |
| `select(net_values, eligible=None)` | decision | 5.7 | index of the eligible test with the highest positive net value, or `None` |
| `commit(belief, utility, net_values, eligible=None, budget_reached=False)` | decision | 5.7 | `(True, diagnosis)` or `(False, None)` |
| `gate_allows(kl_inc, eps_prior, eps_wm)` | gate | 5.8 | KL_inc > eps_prior + eps_wm |
| `bootstrap_upper_bound(values, groups, rng, level=0.90, n_boot=10000, absolute=False)` | gate | 5.8, 15 | one-sided upper percentile bound of a mean, resampling groups |
| `stateful_step(c_t, b_t, weights, posteriors, real_index=None, scale=None, *, real_posterior=None)` | tracks | 5.9 | `Step`: `belief` c_{t+1}, `projection`, `scale`, `fallback` |
| `stateless_step(b_t, weights, posteriors, real_index=None, scale=None, *, real_posterior=None)` | tracks | 5.9 | `Step` |
| `multistep_accounting(paths, truth, path_probs, leaf_beliefs, prefix_beliefs)` | multistep | 5.12 | `MultiStepAccounting`: m_0, the three terms of (iii), martingale and tower residuals |
| `audit_record(point)`, `audit_point(reader, world_model, history, test, k, rng, budget)` | card | 9.1, 16 | one audit record |
| `python -m tiicee.cee.audit FILE.json [--json]` | audit | 9.1 | the audit card |

There is no function that minimizes KL_inc over tau. SPEC 5.5 calls that rule degenerate (an agent
that never updates is coherent) and says it must not be used for anything.

The reader-weighted world-model error of SPEC 5.5 is `−GainTerms.world_model`.

## Floors and tolerances in force

| Constant | Value | Source | Where |
|---|---|---|---|
| `FLOOR` | 1e-6, then renormalize | SPEC 5 | reader vectors, see below |
| `TOL` | 1e-8 on the largest class-marginal violation | SPEC 5.3 | `project` and everything that projects |
| `MAX_ITER` | 500 updates of u | SPEC 5.3 | same |
| `VARIANT_BOUNDS` | shrink-only [1, 16]; two-sided [1/8, 16] | SPEC 5.11 | `scale` |
| `RTOL` | 0.01: the bisection bracket on log tau is narrowed to 1% | SPEC 5.11 | `tau_budget`, `scale` |
| `TAU_RATIO_GRID` | 0.5, 0.67, 1, 1.5, 2, 3, 5, 8 | SPEC 5.5 | `tau_ratio` |
| bootstrap `level`, `N_BOOT` | 0.90 one-sided; 10,000 resamples | SPEC 5.8, 15 | `bootstrap_upper_bound` |
| `SUM_TOL` | 1e-6 | this library | a supplied prior, posterior or weight vector further than this from summing to one is rejected |
| `GRID_RATIO` | 2 | this library | spacing of the tau grid on which r' is checked for monotonicity |
| `MONOTONE_SLACK` | 100 × `tol` | this library | r' may rise by this much between grid points before it counts as a rise |
| `R_ZERO` | 1e-12 nats | this library | responsiveness at or below this is a reader that does not move; `tau_ratio` is then undefined |
| `DISPLAY_ZERO` | 1e-12 | this library | the markdown card prints smaller magnitudes as 0; the JSON keeps the number |

`tol`, `max_iter` and `rtol` are keyword arguments wherever they apply; the defaults are the
specification's.

## What is floored, and what is not

`floor_norm` is not idempotent: a floored entry becomes 1e-6 / (1 + excess), slightly below the
floor, so a second pass moves it again by a relative amount of about 1e-6. The floor is therefore
applied exactly once, to reader output, on entry:

- Floored: `prior`, `posteriors`, `q_real`, `b_t`, `real_posterior` wherever a public function takes them
  (`project`, `apply_to_real`, the audit quantities, `temper`, `projected_responsiveness`,
  `tau_budget`, `scale`, `apply_scale_to_real`, `tau_ratio`, `raw_voi_score`, both track steps).
  Pass raw reader output to these. The vectors actually used are on the result (`Projection.prior`,
  `Projection.posteriors`).
- Never floored: world-model `weights` (a zero weight is a legitimate point mass); class weights
  `u`; corrected, tempered and repaired posteriors; stateful proposals; the tracked belief `c_t`,
  which must stay exact for the martingale of SPEC 5.12 (i); the inputs of `gain_terms`,
  `realized_gain`, `coherent_voi`, `phi`, `select`, `commit`, `multistep_accounting`. Pass
  projection outputs to these, for instance `coherent_voi(p.prior, p.weights, p.corrected, U)`.

## Naming the real result to a track

Give exactly one of `real_index` (the position of the real outcome among the outcomes the world
model proposed) or `real_posterior` (the reader's belief after the real result, q_real). The
projection and tau_B depend only on the imagined outcomes, so both give the same u_t. With
`real_posterior` the step applies them to a result that was never imagined, which is the usual case
for a sampled world model:

- stateless: c_{t+1} ∝ u_t × q_real; with a scale stage, u_t × b_t (q_real / b_t)^(1/tau_B)
- stateful: c_{t+1} ∝ u_t × c_t × (q_real / b_t); with a scale stage, u_t × c_t (q_real / b_t)^(1/tau_B)

When q_real equals one of the imagined posteriors the two ways agree. `real_posterior` is a reader
vector and is floored. Giving both, or neither, is an error.

## Non-convergence

Never raised. `Projection.converged` is False and `max_violation` says by how much. KL_inc is a
divergence, and non-negative, only when the projection converged. The track steps then carry the
raw update forward and set `Step.fallback`; the audit card reports no coherent VOI or realized gain
for that point; any other caller must check the flag and use the raw reader (SPEC 5.3).

## How tau_B is found

r'(tau) is evaluated on a grid across the interval (factor 2 apart, plus the upper end), then the
crossing r' = B is bisected on log tau inside the grid cell that contains it. The value returned is
always the largest tau in the interval with r' ≥ B, or the lower end if there is none:

| `path` | Meaning |
|---|---|
| `solved` | r' was non-increasing on the grid; the crossing was bisected |
| `clipped_low` | r' is below B everywhere: even the most amplified allowed reader carries less. tau is the lower end |
| `clipped_high` | r' meets B at the upper end: even the flattest allowed reader carries at least B. tau is the upper end. A zero budget always lands here |
| `fallback` | r' rose somewhere on the grid; the last crossing was bisected |

Limits worth knowing:

- Monotonicity of r' is assumed by the specification, not proven. It held on every case tried,
  including an adversarial search with fewer outcomes than classes; the fallback path is exercised
  by a planted non-monotone r'. A rise narrower than one grid step would not be detected.
- With the specified 1% tolerance, a reader that is already exactly at its budget gets tau_B within
  about half a percent of 1, not 1. Pass a smaller `rtol` where that matters.
- The shrink-only result always equals the two-sided result clipped at 1, because the two searches
  share their grid.
- `tau_ratio` is uninformative for a pair that is coherent at every scale: its ratio is zero across
  the grid.

## Audit command

    uv run python -m tiicee.cee.audit FILE.json [--json]

`FILE.json` is a list of decision points:

    {"id": "...", "classes": [...], "prior": [...], "outcomes": [...], "weights": [...],
     "posteriors": [[...], ...],              one row per outcome
     "budget": 0.03,                          optional, nats
     "real_outcome": "...", "y_true": "...",  optional, an outcome and a class name
     "utility": [[...], ...]}                 optional, one row per diagnosis; 1[correct] if absent

Each record holds `drift_l1`, `drift_max`, `responsiveness`, `kl_inc`, `converged`, `iterations`,
`max_violation`, `budget`, `tau_b_shrink_only` and `tau_b_two_sided` with their paths, `tau_ratio`,
`utility`, `coherent_voi`, `realized_gain`. The two-sided tau_B is the regime-map diagnostic. The
summary averages scales geometrically and averages KL_inc, coherent VOI and realized gain over
converged points. The command exits with status 2 and names the point if one is malformed.

`audit_point(reader, world_model, history, test, k=None, rng=None, budget=None)` builds the same
record from any objects that satisfy `tiicee/interfaces.py`.

## Tests

    cd tii-cee && uv run pytest -q tests/test_cee_*.py

The E0 fixtures of SPEC 9.1. Expected values are worked by hand in the test docstrings or
enumerated inline; none comes from a library helper.
