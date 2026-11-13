# Plain-English guide to study TII-CEE-1

Every example below is invented. None comes from a patient or from a result of this study.

**Reader.** Whatever turns what has been seen so far into odds over the four diagnoses. Example: after
pain on the right side, a reader says appendicitis 60%, cholecystitis 20%, pancreatitis 10%,
diverticulitis 10%. The reader can be a trained agent, a language model, or a formula.

**World model.** Whatever predicts what a test would show if it were ordered. Example: "an ultrasound
here would show an inflamed appendix with chance 0.5, a normal study with 0.4, and would be
unavailable with 0.1."

**Imagining a result.** Asking the reader how it would react to each predicted result, before
ordering the test. The predictions are guesses, never observations, and never written into the
patient's record.

**Conservation of expected evidence.** If you average your reactions to every result you consider
possible, weighted by how likely you think each result is, you must land back on what you believe
now. Example: if appendicitis is 60% now, and a scan would push it to 90% half the time and to 30%
the other half, the average is 60%: consistent. If it would push it to 90% or to 70%, the average is
80%, and you are claiming to know something before seeing anything.

**Drift.** The part of a reader's reaction that does not depend on what the result says. Example: a
reader that adds a little to appendicitis after every test, normal or not.

**Evidence scale.** How large the reactions are. Too large: one mildly abnormal lab swings the odds
from 60% to 99%. Too small: a decisive scan barely moves them.

**Information budget.** How much a test can actually tell about the diagnosis, given what is already
known, measured in nats. A test that repeats an earlier one has a budget of zero.

**Redundancy.** Reacting again to evidence already counted. Example: two lab panels that share the
same white-cell count, and a reader that counts it twice.

**Responsiveness.** How much a reader moves at all across the possible results. Near zero means
imagined results cannot change anything it does.

**Projection (CEE-Read).** The drift repair: four numbers per decision, one per diagnosis, that
reweight every imagined reaction so they average back to the current belief; the same four numbers
then reweight the reaction to the real result. It removes drift and nothing else; it does not fix a
reader whose reactions are too big or too small.

**Scale repair (CEE-Scale).** Shrinks or enlarges a reader's reactions until they carry exactly the
test's information budget. It is only as good as that budget: a budget twice too large makes things
worse.

**Value of information.** How much better the final decision is expected to be after the test than
now. Computed from the repaired reactions, it can never be negative, which the old planning score
could be.

**Commit rule.** When to stop testing and name a diagnosis. Here: stop when no available test is
worth its cost.

**Regret.** How much worse an agent does than the best possible agent with the same information and
the same budget. Split into the part from choosing tests, the part from reading results, and the part
from stopping.

**Counterfactual.** What would have happened under a different choice. In the synthetic environment
we know every test's result for every patient, so we can compute it exactly; in real charts we
cannot.

**Why an exact synthetic environment.** In charts, tests that were never ordered have no result, so
"what if we had ordered the CT" has no answer. In the synthetic environment it always does, so
regret and the value of each repair can be computed exactly rather than estimated.

**T5.** The average of the four per-disease recall rates. Example: getting 90%, 80%, 70% and 60% of
each disease right gives T5 75, however many patients each disease has.

**Log loss.** How surprised a reader is by the true diagnosis: small when it gave the truth high
odds. Lower log loss is not the same as more correct answers; a reader can become better calibrated
without changing a single final answer.

**Calibration.** Whether stated odds match how often things happen: of all cases called 70%, about
70% should turn out true. More confidence is not more truth.

**Cross-fitting.** Fitting anything that looks at labels on some patients and using it only on
others, rotating through folds, so no patient is scored by something that learned from that patient.

**Coherent is not correct.** A reader can satisfy conservation of expected evidence and still be
consistently wrong. The repairs make readers consistent; whether that helps accuracy is what the
experiments measure.
