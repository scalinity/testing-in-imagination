# The CEE projection and the gain identity: derivation and numerical check

Companion to SPEC_RESEARCH_ADVANCEMENT.md version 3.1, sections 5.3 to 5.6, 5.11 and 5.12.
Date: November 13, 2026. Check script: `cee_check.py` (numpy only). Recorded output: `cee_check_output.txt`.

Nothing here is new mathematics. The projection is a minimum-KL (I-) projection onto two marginal
constraints, computed by iterative proportional fitting; the first identity is the log-ratio of two
distributions that differ by a class reweighting; the sequential identity is the chain rule for KL.
The note exists so that every step is checkable and no claim leans on an unstated assumption.

## 1. Setup

Fix one decision point: a history h and a candidate test a. Logs are natural.

- Classes y in {1..C}. Outcomes o in a finite set that includes UNAVAILABLE.
- Reader: b(y) > 0 is the belief now; q_o(y) > 0 is the belief the reader reports after outcome o.
  Positivity is enforced by the floor of section 5 (1e-6, then renormalize).
- World model: weights P_hat(o) > 0 on the outcomes it proposes, summing to one.
- Raw joint: J(o, y) = P_hat(o) q_o(y). Its outcome marginal is P_hat. Its class marginal is the
  imagined mean b_bar(y) = sum_o P_hat(o) q_o(y). The pair is coherent when b_bar = b.
- Truth for this decision point: any distribution P_star(o, y) of the realized outcome and the label
  among cases that share the information the projection used. Its marginals are P_star(o) and
  b_star(y). Nothing is assumed about it.

## 2. The projection

Problem. Minimize KL(J' || J) over joints J' with outcome marginal P_hat and class marginal b.

Form of the solution. The Lagrangian is

    sum_{o,y} J' log(J'/J) + sum_o alpha_o (sum_y J'(o,y) − P_hat(o)) + sum_y beta_y (sum_o J'(o,y) − b(y)).

Setting the derivative in J'(o, y) to zero gives log(J'/J) + 1 + alpha_o + beta_y = 0, so
J'(o, y) = J(o, y) a(o) u(y) with a(o) = exp(−1 − alpha_o) and u(y) = exp(−beta_y) > 0.
The outcome constraint fixes a(o): P_hat(o) = a(o) P_hat(o) sum_y u(y) q_o(y), so a(o) = 1/Z_o with

    Z_o = sum_y u(y) q_o(y),        J'(o, y) = P_hat(o) q'_o(y),        q'_o(y) = u(y) q_o(y) / Z_o.

The class constraint is the conservation statement itself: sum_o P_hat(o) q'_o(y) = b(y).

Existence and uniqueness. The product P_hat(o) b(y) is feasible and J is strictly positive, so the
feasible set contains a point absolutely continuous with respect to J; the objective is strictly
convex on a compact convex set; hence the minimizer exists and is unique, and u is unique up to a
positive scale factor (which cancels in q'). Iterative proportional fitting converges to it for
strictly positive J. Without the floor a class with b(y) > 0 and q_o(y) = 0 for every o would make
the problem infeasible; that is what the floor prevents.

Iteration used. u ← u × b / (sum_o P_hat(o) q'_o), rescale, repeat. This alternates the two marginal
fits; the outcome fit is absorbed into Z_o.

## 3. KL_inc: formula and when it is non-negative

Using J'/J = u(y)/Z_o and both marginals of J',

    KL(J' || J) = sum_{o,y} J'(o,y) [log u(y) − log Z_o]
                = sum_y b(y) log u(y) − sum_o P_hat(o) log Z_o  =: KL_inc.

Conditions.
1. The class marginal of J' must equal b. If the iteration stops with a residual m − b, where m is
   the achieved imagined mean, the formula differs from the true divergence by
   sum_y (m(y) − b(y)) log u(y), at most ||m − b||_1 × max|log u|. With the tolerance of 1e-8 this is
   negligible but it is why non-convergence is recorded and falls back to the raw reader.
2. Given 1, KL_inc ≥ 0 because it is a KL divergence, with equality exactly when J' = J, that is,
   when the raw pair already satisfies b_bar = b.

Dual form. Define F(u) = sum_y b(y) log u(y) − sum_o P_hat(o) log sum_y u(y) q_o(y). In v = log u,
F is concave (a linear term minus a weighted log-sum-exp) and its gradient is
b(y) − sum_o P_hat(o) q'_o(y), which vanishes exactly at the projection. So KL_inc = max_u F(u):
the projected weights are the class reweighting with the largest expected log-score gain when the
agent's own two marginals are taken as the truth.

## 4. The gain identity

Claim. For any positive class weights u, with q'_o(y) = u(y) q_o(y)/Z_o, and any truth P_star,

    Delta := E_star[ log q'_O(Y) − log q_O(Y) ]
           = F(u) + sum_y (b_star(y) − b(y)) log u(y) − sum_o (P_star(o) − P_hat(o)) log Z_o.

Proof.
Step 1. The per-case log ratio separates: log q'_o(y) − log q_o(y) = log u(y) − log Z_o. One part
depends only on the class, the other only on the outcome.
Step 2. Take the expectation under P_star. The class part sees only the class marginal and the
outcome part only the outcome marginal:
Delta = sum_y b_star(y) log u(y) − sum_o P_star(o) log Z_o.
Step 3. Add and subtract the same two sums evaluated at the agent's own marginals b and P_hat:
Delta = [sum_y b log u − sum_o P_hat log Z_o] + sum_y (b_star − b) log u − sum_o (P_star − P_hat) log Z_o.
The bracket is F(u). ∎

With the projected u the bracket is KL_inc, which gives the three terms of section 5.4:

- incoherence, KL_inc ≥ 0: what the correction is worth if the agent's prior and the world model
  are both right;
- prior calibration, sum_y (b_star − b) log u: the correction moves mass toward classes with large
  u; this term is positive when the truth has more of those classes than the agent believed and
  negative when it has fewer;
- reader-weighted world-model error, −sum_o (P_star − P_hat) log Z_o: the world model's error
  enters only through its inner product with log Z_o. Outcomes on which the reweighting has no
  effect (Z_o equal across outcomes) contribute nothing however badly they are predicted.

What is exact and what is not.
- The identity is algebraic. It holds for every u, converged or not, and for every truth.
- Only the reading "first term is a divergence" needs the projection (section 3).
- Realized gain. g = log u(y_true) − log Z_{o_real} has expectation Delta under P_star, so its
  sample mean estimates Delta from factual outcomes and labels alone. On charts P_star is the
  distribution of results for tests that were actually performed in such cases; the estimate says
  nothing about tests that were never run.
- Elementary bound, used only for the gate. Because b_star − b and P_star − P_hat each sum to zero,
  |prior term| ≤ TV(b_star, b) × osc(log u) and |world-model term| ≤ TV(P_star, P_hat) × osc(log Z),
  where osc is max minus min. Hence Delta ≥ KL_inc − TV(b_star, b) osc(log u) − TV(P_star, P_hat) osc(log Z).

## 5. Corollaries used in section 5.3 and 5.6

Calibrated marginals. If b_star = b and P_star = P_hat then Delta = KL_inc ≥ 0. This is the
Pythagorean relation for I-projections, KL(P_star || J) = KL(P_star || J') + KL(J' || J), read as a
statement about log scores. It is a single-step statement.

Exact re-exposure. If the world model puts all mass on one outcome o0, the constraint reads
q'_{o0} = b, so the corrected belief after seeing o0 is the prior: duplicate evidence moves nothing.

Coherent pair unchanged. If b_bar = b already, u constant solves the constraint, so q' = q and KL_inc = 0.

Coherent value of information is non-negative. For a utility table U(d, y) let
phi(c) = max_d sum_y c(y) U(d, y). phi is convex, as a maximum of linear functions. Since
sum_o P_hat(o) q'_o = b, Jensen gives sum_o P_hat(o) phi(q'_o) ≥ phi(b). The condition is the
marginal constraint; with residual m − b the bound degrades by at most max|U| × ||m − b||_1.
For the raw pair the same expression is only bounded below by phi(b_bar) − phi(b), which can be
negative.

Why the legacy score went negative. For any pair,

    H(b) − sum_o P_hat(o) H(q_o) = sum_o P_hat(o) KL(q_o || b) + sum_y (b_bar(y) − b(y)) log b(y).

The first term is non-negative. The second vanishes for a coherent pair and has no sign otherwise.
An entropy-reduction score computed from an incoherent pair is therefore not an expected information
gain, which is what the 4 negative deciles of 10 in the content study show.

The degenerate scale rule. Temper the increments, q_o(tau) ∝ b (q_o/b)^(1/tau). As tau grows every
q_o(tau) tends to b, the pair becomes coherent and KL_inc tends to 0. Minimizing incoherence over
tau therefore selects an agent that never updates. Any scale rule must bring in something else.

## 6. Scale repair by information-budget matching (section 5.11)

Let B ≥ 0 be an estimate of the information the test carries about the diagnosis at this history,
I(Y; O | h, a). For each tau project the tempered pair and write
r'(tau) = sum_o P_hat(o) KL(q'_o(tau) || b), which is the mutual information of the coherent joint.
CEE-Scale picks tau_B in a declared interval with r'(tau_B) = B, clipped at the ends.

Proposition. Suppose the reader's increments are a power distortion of a coherent reader's:
q_o ∝ b (q°_o / b)^gamma with sum_o P_hat(o) q°_o = b, and B equals the mutual information of
(P_hat, q°). Then tau = gamma solves r'(tau) = B and the repaired posteriors equal q°.
Proof. At tau = gamma the tempered increments are those of q°; that pair is coherent, so the
projection leaves it unchanged, and r' is its mutual information, which is B. ∎

Not proven: that r' is monotone in tau, which would make the solution unique. It was monotone in
every case the script visits; the specification requires a monotonicity fixture and a fallback.
Not guaranteed: anything when B is wrong. Section 7 shows that an overestimated budget is harmful.

## 7. Sequential statements (section 5.12)

Fix a deterministic test-selection policy, so the test at step t is a function of the history
h_{t−1}. Stateful track: c_0 given; at each step the proposals m_o ∝ c_t × (q_o / b_t) are projected
onto mean c_t under P_hat(· | h_t, a_t), and c_{t+1} is the projected posterior at the real outcome.

(i) Martingale by construction. E_{P_hat}[c_{t+1} | h_t] = c_t, for any reader.

(ii) One joint model. Define M(o_{1:T}, y) = [prod_t P_hat(o_t | h_{t−1})] × c_T(y; h_T). By (i) and
the tower property, sum over later outcomes gives M(y | o_{1:t}) = c_t(y; h_t) for every t, and
sum over all outcomes gives the class marginal c_0. The corrected agent is therefore a Bayesian
agent for the single joint model M whose outcome marginal is the world model. A raw reader r_t
defines a joint the same way, but its intermediate beliefs are not the conditionals of that joint
and its class marginal m_0 = E_{P_hat}[r_T] is not r_0.

(iii) T-step identity. For any belief process with joint M and class marginal m_0, expanding
KL(P_star(Y, O) || M(Y, O)) with the chain rule in both orders gives

    E_O[ KL(P_star(Y | O) || bel_T) ]
        = KL(b_star || m_0) + E_Y[ KL(P_star(O | Y) || M(O | Y)) ] − KL(P_star(O) || P_hat(O)).

The left side is the excess log loss of the final belief. For the corrected agent m_0 = c_0 at
every horizon; for the raw reader m_0 is a drifted prior that can move further from r_0 at every
step. Coherence removes the accumulated-drift term. It does not control the middle term, the
error of the likelihood the agent implicitly uses, and so it does not guarantee a lower loss.

(iv) Elementary bound. Under the truth the corrected belief is an approximate martingale:
||E_star[c_{t+1} | h_t] − c_t||_1 ≤ 2 TV(P_star, P_hat) × max_o ||q'_o − c_t||_1.

## 8. What the script supports, and what it does not

Run of November 13, 2026, 407 seconds, all asserted checks passed.

| Claim in the specification | Script evidence | Status |
|---|---|---|
| 5.4 identity is exact | Parts 1: 3,000 random cases, worst error 1.9e-15; also for arbitrary weights | Supported |
| 5.3 KL_inc ≥ 0, zero for a coherent pair | Part 1 minimum 1.1e-7; Part 2b 2.2e-16 | Supported |
| 5.3 re-exposure returns the prior | Part 2a, error 3.3e-16 | Supported |
| 5.6 coherent VOI ≥ 0; raw score can be negative | Part 2c: minimum −4e-13; raw negative in 729 of 1,500 cases | Supported |
| 5.4 consequence 1, gain equals KL_inc when calibrated | Part 2d, error 1.6e-15 | Supported |
| 5.5 minimizing incoherence over scale is degenerate | Part 2e | Supported |
| 5.12 martingale, one joint model, T-step identity | Part 3: 200 random 3-step problems by enumeration, errors below 2e-13 | Supported |
| Projection repairs drift | Part 5: entrenching 0.797 → 0.545, fixed-class 0.766 → 0.545 log loss at p = 0.07. Part 4, the November 12 sample: 0.761 → 0.524 and 0.695 → 0.524 | Supported in this twin, with an oracle world model only |
| Projection barely helps double counting | Independence reader 0.581 → 0.545 log loss, accuracy 0.880 → 0.866 | Supported in this twin |
| Projection hurts wrong-scale readers | Over-reactive 1.029 → 1.705; under-reactive 0.498 → 0.663 | Supported in this twin |
| 5.11 scale repair works given the true budget | All five families reach 0.449 against Bayes 0.433 (p = 0.07) and 0.768 against 0.754 (p = 0.20) | Supported in this twin, true budget and oracle world model only |
| 5.11 overestimated budgets are harmful | Budget × 2, two-sided: 3.63 log loss. Shrink-only limits this to 0.469 for drift families but 1.425 for the over-reactive reader | Supported; it is the reason for the conservative budget and the gate |
| An independence budget fixes double counting | 0.581 → 0.537, no better than projection alone | Not supported |
| Shrink-only repairs under-reaction | 0.498 → 0.635 | Not supported; under-reaction needs the two-sided or the labeled variant |

Not touched by the script, and therefore still hypotheses: every statement about language-model
readers, charts, the trained policy, sampled rather than enumerated outcomes, learned world models,
budgets estimated from data, the gate, reader-weighted error against likelihood, exact regret
attribution, the stateless track, informative missingness, and any accuracy or T5 effect.

The numbers in Parts 4 and 5 are Monte Carlo estimates from my own re-implementation of the twin's
parameters, with an oracle world model. They are sandbox evidence for design decisions, not project
results, and they are not to be cited as such.
