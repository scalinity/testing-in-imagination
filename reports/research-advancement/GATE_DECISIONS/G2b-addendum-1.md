# Gate G2b, addendum 1: context the record left out

Written November 13, 2026, about 7:15 AM EST, after an independent re-derivation of E1's verdicts
from the committed tables (byte-identical verdicts.json). The result of G2b.md is unchanged: MET.

1. The one part-B verdict that failed is not named in G2b.md: CEE-Gate is not valid. The
   applications the gate allows have negative mean gain for the independence, over-reactive and
   under-reactive families under each of W0, W1 and W2 (for example over-reactive under W0, −0.335
   nats on the 0.2% of points the gate allows; independence under W2, −0.012 on 0.6%); the two drift
   families and both mixtures are positive. The verdict is False on all four method-choice variants.
   G2b does not depend on it, but every later use of CEE-Gate must cite it.
2. Reading R6 applies to the W0 rows of G2b.md's table: under W0 the selector S_ig is exact and
   identical across seeds, so those intervals measure variation over S_rand's order, not sampling.
