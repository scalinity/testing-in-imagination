# Gate G2b: does E1-B find that the projection repairs drift?

Evaluated November 13, 2026, about 6:05 AM EST. Immutable once committed.

## Rule (specification 14.1; EXPERIMENT_CARDS/E1.md part B)

Met if E1-B says the projection repair is present: both drift families improve, with the paired 95%
interval over twenty seeds excluding 0, under W0 and under W2. If met, the method arms stay
eligible; if not, E5 is skipped and the paper is the mechanism paper.

## Evidence

tii-cee/results/E1/verdicts.json, run at commit a4f34bd (manifest tii-cee/results/E1/manifest.json),
base variant, stateful track (the track the pre-registered rule selected), exact expected log loss at
horizon 6 of the reduced twin, raw minus corrected:

| Reader family, world model | Estimate (nats) | 95% interval over 20 seeds |
|---|---|---|
| Entrenching drift, W0 | 0.191 | [0.188, 0.195] |
| Fixed-class drift, W0 | 0.179 | [0.178, 0.180] |
| Entrenching drift, W2 | 0.190 | [0.186, 0.194] |
| Fixed-class drift, W2 | 0.177 | [0.177, 0.178] |

Supporting verdicts from the same run: no harm to R0 under W0 (largest change 2.2e-16); negative
controls confirmed (projection alone worsens the over-reactive family by 0.48 nats and the
under-reactive by 0.11 under W0). The same verdict holds on the three other method-choice
variants. Caveat: the specified percentile bootstrap over twenty seeds has simulated coverage of
about 92% (RESEARCH_LEDGER.md D-STATS item 2); every interval above excludes 0 by a wide margin.

## Result

MET.

## Path taken

The method arms stay eligible (A2, A3 in E5, subject to G3).
