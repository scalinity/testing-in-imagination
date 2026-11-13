# Gate G2c: is CEE-Scale usable outside the twin?

Evaluated November 13, 2026, about 6:05 AM EST. Immutable once committed.

## Rule (specification 13, 14.1; EXPERIMENT_CARDS/E1.md part D, reading R1)

Met if, for the variant the pre-registered rule selects, under W2 with the estimated conservative
(held-out) budget, the over-reactive, under-reactive and independence families each improve on both
raw and projection-only with the paired 95% interval excluding 0, and the eligibility test admits
an overestimated budget in at most 10% of stress cases. If met, CEE-Scale may be applied on charts
and public data; if not, CEE-Scale is reported from the twin as found and not applied elsewhere, and
tau_B is still reported.

## Evidence

tii-cee/results/E1/verdicts.json at commit a4f34bd, base variant, stateful track.

Variant selection: shrink-only (two-sided minus shrink-only, worst case over the stress set, 6.46
nats [6.18, 6.79]: an overestimated budget makes the two-sided variant amplify reactions far past
the evidence).

Selected variant under W2 with the held-out budget, comparator minus scaled (positive = improvement):

| Family vs comparator | Estimate (nats) | 95% interval | Improves |
|---|---|---|---|
| Over-reactive vs raw | 0.064 | [0.005, 0.120] | yes |
| Over-reactive vs projection | 0.550 | [0.496, 0.598] | yes |
| Under-reactive vs raw | −0.111 | [−0.113, −0.109] | no |
| Under-reactive vs projection | −0.002 | [−0.003, −0.001] | no |
| Independence vs raw | 0.002 | [−0.002, 0.005] | no |
| Independence vs projection | −0.001 | [−0.005, 0.002] | no |

Eligibility: an overestimated budget was admitted in 37 of 2,396 stress cases (1.5%; 8.0% of the
461 overestimated cases), within the 10% limit; 72.6% of cells were eligible.

For comparison, with the exact budget under W0 the scale repair is present (two-sided): the three
families improve on both comparators (for example over-reactive 0.35 vs raw and 0.82 vs projection).

## Result

NOT MET. Two separate reasons:
1. Structural, recorded in advance (reading R1): the selected shrink-only variant cannot amplify an
   under-reactive reader.
2. Substantive: with the estimated conservative budget, the independence family (double counting)
   shows no detectable improvement against either comparator, although the exact budget repairs it.
   The estimated budget is what fails there, not the scale rule.

## Path taken

CEE-Scale is reported from the twin as found and is not applied on charts or public data. The layer
carried downstream is the projection only, on the stateful track (selected_layer in verdicts.json).
tau_B is still computed and reported everywhere a budget exists, including for the regime map.
The arms A1s in E4 and every scale-and-projection variant downstream are not run as treatments.
