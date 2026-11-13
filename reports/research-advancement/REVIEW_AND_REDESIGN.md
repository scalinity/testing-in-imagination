# Review of version 3.0 as a top-venue reviewer, and the redesign into 3.1

November 13, 2026. Written by the specification's author against his own plan. The review assumes version 3.0 had been executed exactly as written and every experiment had come out in its favor.

## The review version 3.0 would get today

Summary. The paper states that an agent which imagines test results should conserve expected evidence, measures the violation, corrects it with a minimum-KL projection, and gives an identity for the gain. Evidence: a synthetic environment built by the authors, one trained agent on MIMIC-CDM, one public dataset.

Strengths. The principle is clean. The identity is exact and useful as an accounting device. The negative controls are stated before the experiments. The synthetic environment gives exact regret, which chart data cannot. The protocol is unusually careful.

Weaknesses.
1. Positioning. The nearest repairs are not discussed. ArbitrageForecaster (Paleka et al., ICLR 2025) already patches a language model's probabilities toward the expected-evidence constraint and reports its smallest consistency gain for that check, with no clear accuracy effect. Kim et al. (2026) already repair martingale violations by fine-tuning. BED-LLM names the incoherent pairing and avoids it. The mathematics is a textbook projection. What is left is a combination, and the paper does not say so precisely.
2. The repair covers one failure. By the authors' own probe the projection fixes drift, barely helps double counting, and hurts readers whose reactions are too strong or too weak. Those are the regimes one expects from language models. The paper offers a diagnostic for them that the authors admit is unreliable, and no repair.
3. Breadth. One environment, one agent, one backbone, one seed, 240 validation admissions, a test split already examined twice, an unspecified public dataset, unspecified language-model readers. The claim reads as general; the evidence is one case study.
4. Power. The closed-loop comparison cannot detect an effect smaller than about 3.5 T5 points on 240 admissions. A realistic gain from a reading correction is smaller than that. The test result will be an interval around zero.
5. Sequential behavior. The guarantee is one step. The agent takes several. Nothing is said about what accumulates.
6. Sampling. Eight imagined results per test on charts, with no account of the Monte Carlo error in the projection or the value of information.
7. Thresholds. The 0.20 margin and the gate bounds are not motivated.

Assessment. Sound and honest, but the method is incremental over cited and uncited work, it is validated where it is easy, and its own negative controls suggest it will not matter for real readers. Recommendation: weak reject. To change my mind: show the phenomenon across many readers and datasets, give a principled repair where the projection fails, position against the nearest repairs, and bring enough clinical evidence to see an effect of realistic size.

## What changes that verdict, and what 3.1 does

| Candidate addition | Does the claim need it? | Decision |
|---|---|---|
| Label-free audit as a general diagnostic | Yes. It is what turns one case study into a finding about a class of agents | In. The regime map (E2, E7, E4) is the paper's first result |
| More language-model readers | Yes, for the same reason | In. At least eight models, four families, tenfold size range |
| Several public datasets | Yes. One dataset cannot show the accounting transfers | In. Four complete-record datasets, two pairings each |
| A principled scale repair | Yes. Without it the method is silent in the likely regime | In, as CEE-Scale: match the test's information budget. Separately named, with its own ablation and gate |
| A sequential statement | Yes, modestly. A reviewer will ask; an honest accounting identity answers | In, as accounting, not as a guarantee: the corrected agent is Bayesian for one joint model, and the multi-step loss splits into prior, likelihood, and world-model terms |
| Seeds of the primary policy | Only for a closed-loop clinical claim, but they also show the agent's regime is a property of the recipe | In. Four new seeds |
| Cross-fitted policies (my addition) | Yes, if any clinical effect is to be detectable. It raises the clinical sample from 240 to 2,160 held-out admissions | In. Five fold policies |
| A second backbone | Not for the mechanism. It makes the clinical evidence less of a single-model story | In, as a secondary family, three seeds |
| An independent credentialed cohort | It would be the strongest clinical evidence. It cannot be planned without data you already hold | Blocked until you list one |
| Fine-tuning the policy with a coherence loss | No. It is a different paper and Kim et al. are already there | Out |

## What the sandbox says about the new component

From cee_check.py, an oracle world model, my re-implementation of the twin: matching the true information budget brought the independence, drift, over-reactive, and under-reactive readers to a log loss of 0.449 against a Bayes reference of 0.433, where projection alone gave 0.545 for the first three and made the last two worse. A budget twice too large was harmful. A budget from the independence model fixed over-reaction and nothing else. So the component is real in the easy case and its whole risk is the budget. Version 3.1 therefore tests the budget estimator before the repair, uses a conservative bound, and selects the variant by worst-case loss in the twin.

## The headline under each outcome

| Outcome | Headline |
|---|---|
| Drift dominates and projection helps | Agents that imagine results do not conserve expected evidence; one vector of C numbers per decision restores it, and an exact identity predicts the gain |
| Scale or redundancy dominates | Language-model readers mis-size evidence rather than drift: projection alone is harmful exactly where the audit says, and matching an information budget repairs it |
| Violations present, no repair helps | The identity names the term that blocks the gain: imagination fails for a measurable reason |
| Readers barely respond | Imagined results cannot help a reader that does not move; this explains the availability null, and committing early costs nothing |
| Works in the twin and on public data, not on charts | What chart-limited evidence can and cannot support, with exact attribution as the main finding |

The strongest of these is the second. It is also the one version 3.0 could not have produced.

## The one figure

Panel A, the regime map. Horizontal: evidence scale relative to the information budget, tau_B, log axis, 1 in the middle. Vertical: location incoherence in nats per decision after scaling. Every reader is a point: simulated families as open markers, language models as filled markers sized by parameter count, the trained policy's seeds and second backbone as stars, public datasets by marker shape. From each point an arrow whose length is the change in log loss under the pre-registered repair. The region where projection alone does harm, from the exact twin results, is shaded. A reader can see at a glance where real models live and which repair each needs.

Panel B. Realized gain against the gain the identity predicted, one point per reader-by-setting cell, with the diagonal.

Panel C. Exact regret split into selection, reading, and commitment.

## What remains weak after the redesign

- Novelty is still a combination: a known constraint, a known projection, a new use of an information budget, an accounting identity, and exact validation. The paper must say exactly that.
- If language-model readers turn out nearly coherent, or unresponsive, the method has nothing to repair and the paper is the mechanism paper.
- CEE-Scale depends on a budget estimated from data. On charts that estimate may be too noisy to pass its own eligibility test, in which case the scale result exists only in the twin and on public data.
- The clinical dataset is one closed four-way task, and its test split has been looked at before.
- The agent writes the manuscript. ICML 2027 permits that with disclosure; the authors remain responsible for every line.
