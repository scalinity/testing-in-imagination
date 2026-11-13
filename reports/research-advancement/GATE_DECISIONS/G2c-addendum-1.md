# Gate G2c, addendum 1: stronger evidence for reason 2, and caveats

Written November 13, 2026, about 7:15 AM EST, after an independent re-derivation of E1's verdicts
from the committed tables. The result of G2c.md is unchanged: NOT MET.

1. Reason 2 has like-for-like evidence stronger than the W0 comparison G2c.md cites. Under W2 with
   the exact budget (shrink-only, stateful), the independence family improves by 0.040 nats against
   raw and 0.037 against projection (smallest seed 0.032), against −0.001 with the held-out budget.
   The held-out budget is a median 0.47 of the cell's exact mean information and is zero in 40% of
   cells. The conservative budget is too conservative to repair double counting: the estimate, not
   the scale rule, is what fails.
2. The variant selection is robust: the two-sided variant's worst case is the doubled budget in 18
   to 20 of 20 seeds for every family; its mean log loss there is 7.2 nats against 0.82 for
   shrink-only; 73% of its path mass is clipped at tau = 1/8; the projection fell back on 0.5% of
   points, so this is not a convergence artifact.
3. Coverage caveat (RESEARCH_LEDGER.md D-STATS item 2) applies to the table in G2c.md. One "yes" is
   marginal: over-reactive against raw, lower bound 0.005. The gate's result does not depend on it.
4. Reading R5: "scale repair present" (decided on the base variant) is False on the no-correlation
   variant, because the independence reader is exact there and has nothing to repair (−7e-6 nats
   against both comparators); the over- and under-reactive families improve there too.
