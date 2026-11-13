# Testing in Imagination: brief for Professor Liu

Study TII-CEE-1, version 3.1. First brief, November 13, 2026. No experiment of this study has run
yet; every number below is a recorded result of earlier work, cited by its source.

## The question

Our agent imagines what a diagnostic test would show before ordering it. We ask how much of its
error comes from choosing tests, from reading the results, and from deciding when to stop; and
whether one principle repairs the reading and stopping parts.

The principle is conservation of expected evidence: averaged over the results an agent imagines, its
reactions should equal what it believes now. A reader that drifts toward one diagnosis whatever the
result says, or reacts too strongly or too weakly, violates it, and the violation can be measured at
every decision without labels.

## Why this direction, from our own record

1. The largest gain so far is answer recovery, not better planning: test T5 rose from 73.09 to
   84.45 when the final answer fell back to the agent's own hypothesis (V1); the full guards (V3,
   81.83) did not beat V1 (−2.62 [−6.16, 0.76]). Phase A report.
2. Perfect knowledge of which tests are available changed behavior (unavailable requests 203 → 66)
   but not validation T5 (−0.11 [−2.56, 2.38]) for this fixed planner. DIAGNOSTICS.md, with
   ERRATUM-001 narrowing its wording.
3. Our planning score went negative in 4 of 10 deciles, which a true information gain cannot do:
   results were imagined with one model and read with another. PR #9's content aggregate.
Together: test choice does not look like the binding constraint; reading and stopping do.

## What we will build and test

- An audit that measures, per decision, drift (a reaction that ignores the result), scale (reactions
  too large or small for what the test can tell), and responsiveness, without labels.
- Two repairs, kept separate: a projection that removes drift (one vector of four numbers per
  decision), and a scale component that sizes reactions to the test's information budget. The
  projection is expected to hurt readers whose scale is wrong; those cases are kept as negative
  controls.
- An exact identity that says in advance how much the projection is worth and which error blocks it.

## Evidence plan, in order

| Step | What it settles | Compute |
|---|---|---|
| E0 | The code computes the declared quantities (hand-checked fixtures) | CPU |
| E1 | In an exact synthetic environment: where regret comes from, and whether each repair works where it should and fails where it should | CPU |
| E2, E7 | Where nine open language models and four public datasets sit on the regime map | GPU, no patient data |
| E4 | The trained agent's regime on fixed real evidence, over five policy seeds, a cross-fitted cohort of 2,160 admissions, and a second backbone | GPU, MIMIC on HiPerGator |
| E5, E6 | Closed-loop arms, then one disclosed test-split run, only if E4's gate passes | GPU |

## Uncertainty, stated now

Every outcome is publishable as stated in advance: drift dominates and the projection helps; scale
dominates and only budget matching helps; violations exist but no repair helps, with the identity
naming why; or readers barely respond. Lower log loss is not higher accuracy, and a coherent agent
can be coherently wrong. The test split was examined before; E6 is disclosed as such.

## Next decisive experiment

E1 in the synthetic environment, which decides whether the method stays eligible (gates G2a to G2c)
before any clinical GPU time is spent on it. The training track for the policy seeds starts in
parallel because it does not depend on the method.

## Questions for you

1. Which result would make the distinction between drift and scale most convincing to you: the
   regime map across many readers, or exact regret attribution in the synthetic environment?
2. Is a mechanism paper (where agents fail, measured and explained) a strong enough contribution if
   the repairs do not improve accuracy on charts?
3. What evidence beyond MIMIC-CDM would you consider sufficient for the clinical claim, given that we
   hold no second credentialed cohort?

## What we are not claiming

That measuring the violation is new (DIVA, arXiv 2610.32181, measured it across 20 language models
and avoids it by construction; PRIOR_WORK_MATRIX.md row 17), that the content model passed a valid
quality check, that the availability null proves no possible benefit, that the projection fixes double counting or wrong scale, that sandbox numbers are project
results, or anything about clinical use.
