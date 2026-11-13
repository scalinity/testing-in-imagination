# Meeting talk track, study TII-CEE-1

First version, November 13, 2026. No experiment of this study has run yet.

## 90 seconds

Our agent imagines test results before ordering tests. The biggest gain we have so far did not come
from that: it came from recovering answers the agent already had in its hypothesis, 73 to 84 T5 on
test. And giving the planner perfect availability information changed what it ordered but not its
accuracy. So we think the problem is not which tests it picks; it is how it reads results and when
it stops.

There is a simple consistency rule an agent that imagines results should obey: averaged over the
results it imagines, its reactions should equal what it believes now. If they don't, it is either
drifting toward an answer regardless of the evidence, or reacting too strongly or too weakly. We can
measure that at every decision without labels.

We have two repairs, one for drift and one for scale, and an exact identity that predicts how much
the drift repair is worth. We test everything first in a synthetic environment where we can compute
regret exactly, then across nine open language models and four public datasets, then on the trained
agent with five seeds and a cross-fitted cohort of 2,160 admissions. Any outcome is reportable, and
we said in advance what each one would mean.

## Five-minute outline

1. The record: V1 is the bar; availability information changed behavior, not accuracy; the old
   planning score went negative, which a real information gain cannot.
2. The rule, with the invented example from the plain-English guide (60% → 90% or 30%).
3. The audit: drift, scale, responsiveness, two probes that need no world model.
4. The repairs, and where they are expected to fail: projection hurts wrong-scale readers; the scale
   repair is only as good as its budget.
5. The evidence ladder E0 → E1 → E2/E7 → E4, the gates between, and what each outcome would mean.

## Likely questions

- "Isn't this ArbitrageForecaster or BED-LLM?" Paleka et al. patch a model's own forecasts toward
  the same constraint and report the smallest gain for it; BED-LLM avoids the incoherent pairing. We
  repair it under a separate world model, account for the gain exactly, and add the scale repair.
  The mathematics is classical; we say so.
- "Will accuracy improve?" Unknown, and not promised. Lower log loss is not more correct answers.
- "Why should a 240-admission test move anything?" It may not; that is why the clinical sample is a
  cross-fitted cohort of 2,160 admissions and five policies, and why the test split is used once.

## Three questions for Professor Liu

The three questions at the end of PROFESSOR_BRIEF.md.

## Claims to avoid

That the content model passed a valid quality gate; that the availability null proves no possible
benefit; that the oracle was an optimal ceiling; that C1 was fully out of fold; that all synthetic
results leaked; that the projection fixes double counting or wrong scale; that CEE-Scale works
without a valid budget; that sandbox numbers are project results; anything about clinical use.
