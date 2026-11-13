# Prior-work matrix for TII-CEE-1

Maintained with SPEC_RESEARCH_ADVANCEMENT.md version 3.1, section 7.2.
Verified by the specification's author on November 12 and 13, 2026. Owner from here: Daniel, through the executing agent.
Cross-paper scores are never compared; numbers are quoted only to identify a result.

## How to read the status column

- A: abstract and version metadata read directly on November 13, 2026.
- R: passages of the full text read directly on November 13, 2026.
- F: full text queried on November 13, 2026 with a fetch tool that answers from the page and returns short quotations. The quotations were not independently re-read. Pages longer than the tool's window were read to the stated point only.
- N12: read from the primary source on November 12, 2026 and not re-read on November 13.
- K: classical reference cited from background knowledge; not fetched.
- U: something in the row is unverified; the row says what.

## Core rows

| # | Work | Venue and version | What it does | Exact overlap with TII-CEE-1 | Experiment that separates us | Status |
|---|---|---|---|---|---|---|
| 1 | LA-CDM. Bani-Harouni, Pellegrini, Özsoy, Navab, Keicher | ICLR 2026 per N12; arXiv 2506.13474 v3, Feb 28 2026 | Hypothesis agent plus decision agent trained with supervised and reinforcement learning to request and interpret tests on MIMIC-CDM | The base agent. Its reading of results is the object we audit | None needed; it is the substrate. V1 (84.45 test T5) is the bar | A |
| 2 | ACTMED | NeurIPS 2025 per N12 (the text thanks NeurIPS reviewers); arXiv 2510.18988 | A language model samples ten outcomes per missing test, the "hypothetical posterior probability" is obtained "from the LLM", tests are chosen by expected KL per cost; stopping compares expected KL with a target | The simulate-and-re-query pairing we study. No coherence check, no correction, no world model separate from the reader. Its own appendix reports models that collapse or sample implausibly and cases where it falls below using all features | E7: audit this pairing on its own datasets, with and without the projection and scale layers; E1-B raw-pairing baseline | F to the end of Appendix F. U: the exact re-query prompt is not shown in the text |
| 3 | BED-LLM. Choudhury et al. | ICLR 2026 proceedings; arXiv 2508.21184 v3, Apr 20 2026 | Sequential experimental design with a language model. Builds the joint as prior times likelihood, filters sampled hypotheses against the history, and warns that the alternative pairing gives no joint model | Names the incoherence we measure and avoids it by construction. Does not measure its size, repair it, or decompose its cost. No medical task, no separate simulator | E1-B and E7: prior-times-likelihood baseline against repair of the re-query pairing, where a trained reader makes the likelihood pairing unavailable | R for the pairing argument; F for appendices B and D |
| 4 | Consistency checks for language-model forecasters. Paleka et al. | ICLR 2025 | Defines consistency checks, including expected evidence, scores violations by arbitrage under a log scoring rule, and builds ArbitrageForecaster, which patches a forecaster's answers toward consistency | Closest precedent for correcting a language model's probabilities toward the expected-evidence constraint. There the model supplies every probability, questions are forecasting questions, and the paper reports the smallest consistency gain for expected evidence and no clear accuracy effect | E7 baseline: arbitrage-style patching with model-supplied outcome probabilities against projection under a separate world model, scored by realized log loss | R |
| 5 | From Drift to Coherence. Kim, Lee, Fong, Lee, Lee | arXiv 2606.17832 v2, Jun 18 2026; venue not stated | Measures drift of a language model's predictive under its own resampled answers and fine-tunes with a loss that pulls early predictions toward late ones | A repair of martingale violations, by training, for self-generated data on question answering. No separate world model, no decisions, no closed-form correction, no accounting | E2: our inference-time projection on the same kind of drift without fine-tuning; cite as the training-time alternative | F to character 100,000 of 393,850 |
| 6 | CaRDI. Zhou et al. | arXiv 2608.27847 v2, Oct 16 2026, "Beyond Uncertainty Reduction: Calibrated Planning for Interactive Diagnosis" | Risk-sensitive planning for question asking on DDXPlus and MediQ: diagnostic, safety and masked-evidence beliefs, conformal calibration of prediction sets and severity-weighted risk, tree search over ask and commit | Calibrated risk for acquisition and stopping. One structured model predicts and updates, so reading is coherent by construction and correlated evidence is the open question | E1-B conformal-stop baseline under correlated evidence and duplicated panels | A for the abstract and version; N12 for internals. U: the full text could not be re-read on November 13 because the page was rate limited twice |
| 7 | CROS, "Safe to Stop?". Wu and Rus | arXiv 2609.09678 v1 | Risk-constrained stop, continue or defer layer for an LA-CDM-style agent on 1,834 MIMIC-IV-ED episodes; reports that error is 28.3% with the history alone and 34.3% after the full workup | Shows on a sibling agent that more evidence can raise error. Ranks risk; does not explain or correct the reading, never substitutes a test, uses no imagined outcomes | E4 forced-continuation curve raw and corrected; simplified error-ranker stop baseline | F |
| 8 | LDTL | arXiv 2604.05116 v2, Oct 22 2026 | Trains a planner to align with a frozen diagnostic agent's gain in log probability of the true diagnosis, using recorded outcomes; 93.4 accuracy on MIMIC-CDM against 84.8 for a random planner and 81.3 for LA-CDM as reported there | Uses the reader's realized gain as the training signal for selection. No simulator, no coherence check, binary confidence stop | E1-A attribution: how much regret selection can remove once reading is fixed | F. U: which reader is used in the random-planner baseline is not stated |
| 9 | Current Agents Fail to Leverage World Model as Tool for Foresight. Qian et al. | ACL 2026; arXiv 2601.03905 | Across control and visual question answering, agents rarely call a simulator, misuse about 15% of rollouts and can lose up to 5% when simulation is available or enforced; the authors' project page adds that the control tasks used exact simulators | Establishes that integrating foresight is the bottleneck. Offers a taxonomy, not a measurement of belief coherence or a correction | E2 and E4: the audit gives that bottleneck a number per reader | A |
| 10 | Learning-to-Measure | arXiv 2510.12624 v2, May 31 2026; ICML listed as keyword | In-context active feature acquisition with one pretrained sequence model that has a target head and a policy head | Imagination and reading come from one model. No external reader, no incoherence to measure | E1-B: a jointly trained reference shows what coherence by construction buys | F to character 100,000 of 532,554. U: venue; dataset names |
| 11 | Calibrated Value-Aware Model Learning. Voelcker et al. | ICML 2025 per N12; arXiv 2505.22772 v2 | Shows value-aware and MuZero-style model losses are uncalibrated surrogates and corrects them | Weighting model error by what the decision maker does with it. "Calibrated" there means loss calibration | E1-C: reader-weighted error against likelihood, with the matched pair of model errors | A |
| 12 | Uncertainty of Thoughts. Hu et al. | NeurIPS 2024 per N12; arXiv 2402.03271 v3 | Simulates possible futures, scores questions by information-gain rewards, propagates rewards to choose a question | Imagined outcomes guiding acquisition | Raw information-gain baseline in E7 | A |
| 13 | MediQ. Li et al. | NeurIPS 2024 per N12; arXiv 2406.00922 v3 | Interactive clinical benchmark; the expert abstains and asks when unconfident | Confidence-threshold stopping | Confidence-threshold commit baseline | A |
| 14 | Sequential Diagnosis with Language Models (SDBench, MAI-DxO). Nori et al. | arXiv 2506.22405 v2, Jul 2 2025 | 304 NEJM cases as stepwise encounters with a gatekeeper that reveals findings on request | Sequential diagnosis with test costs; findings are synthesized by a model when absent | Not a method baseline; cited for the setting | A. U: peer-reviewed venue not found |
| 15 | Generative surrogate models. Li and Oliva | ICML 2021 per N12; arXiv 2010.02433 v2 | Learns a generative model of unobserved features to guide acquisition | Imagination for acquisition, classical setting | None; lineage | A |
| 16 | Acquisition conditioned oracle. Valancius, Lennon, Oliva | ICML 2024 per N12; arXiv 2302.13960 | Nonparametric oracle for non-greedy acquisition | Retrieval in place of a parametric outcome model, as in our retrieval sampler | None; lineage for W3 | A |

## Audits of belief updating in language models

| Work | Version | What it shows | Relation | Status |
|---|---|---|---|---|
| Falck, Wang, Holmes, "Is In-Context Learning … Bayesian? A Martingale Perspective" | ICML 2024 per N12; arXiv 2406.00793 | In-context learning violates the martingale property | Motivates the audit; no repair | A |
| Martingale Score. He, Shirado, Qiu, Sap | NeurIPS 2025 per N12; arXiv 2512.02914 | Regression of belief change on prior belief across reasoning trajectories; a metric only, with a training objective left as future work | Aggregate entrenchment score over many items; ours is per decision point against an explicit outcome distribution | F |
| Filbry, Melo, Gal, "From Prompts to Posteriors" | arXiv 2610.29331 v1, Oct 21 2026 | Language models generally violate Bayes' rule in context; tools and reasoning help | Audit; no correction, no decisions | A |
| Pal et al. | arXiv 2511.13240, retitled in v3 | Incoherent beliefs and inconsistent actions | Audit | N12. U: later versions not re-read |
| BayesBench | arXiv 2606.30850 | Belief trajectories under multi-turn evidence | Audit | U: seen as a search result only |

## Selective use of a model, and decision-aware model error

Masked model-based actor-critic (NeurIPS 2020), stochastic ensemble value expansion (NeurIPS 2018), safe policy improvement with baseline bootstrapping (ICML 2019), Filos et al. (ICML 2022), WorldEvolver (arXiv 2606.30639), Lambert et al. on objective mismatch (L4DC 2020), decision calibration (NeurIPS 2021). All N12. They own "use the model where it is reliable" and "judge the model by its effect on decisions". Our gate and reader-weighted error are instances, not contributions.

## The mathematics is classical

Minimum-KL projection and its Pythagorean identity (Csiszár); iterative proportional fitting and raking; Jeffrey conditioning; adjusting classifier outputs to a known class distribution; contextual calibration (ICML 2021) and batch calibration (ICLR 2024) for prompt-level class bias; power posteriors and tempering; coherent forecasts dominate incoherent ones (de Finetti; Predd et al., IEEE Trans. Inf. Theory 2009); simulation-based calibration (arXiv 1804.06788). Status K, except the three with identifiers, which are N12.

## Bounded search for the specific method

Question: has anyone projected an agent's imagined posteriors onto "expected posterior equals prior" under a separate world model's outcome distribution, in sequential acquisition or diagnosis?

Searches on November 13, 2026: one keyword search and two semantic searches over 2023 to 2026 work on language-model experimental design, feature acquisition and diagnosis, phrased around Jeffrey conditioning, raking, iterative proportional fitting, minimum-KL projection, marginal matching, and "negative information gain" fixes. Nearest results: rows 3, 4 and 5 above. Not overlapping but surfaced: Belief-Based World Models (arXiv 2609.00455), Preregistered Belief Revision Contracts (arXiv 2604.15558).

Result: no such paper found. This is a bounded search, not proof of absence. The honest position for the paper is: the constraint, the projection and the audits each exist; the combination with a separate world model, the accounting of the gain, the scale layer tied to an information budget, and the exact counterfactual validation were not found.

Before any report states novelty the agent repeats the search with the final method description, adds a citation-graph pass over rows 2 to 5, and records both here.

## Corrections to earlier documents

- Version 3.0 wrote "Cros" as if it were an author. CROS is the method; the authors are Yuexin Wu and Vasile Rus.
- Rows 4 and 5 were listed among "also checked" on November 12. They are core rows: they are the nearest repairs.
