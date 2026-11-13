# Prior-work matrix for TII-CEE-1

Maintained with SPEC_RESEARCH_ADVANCEMENT.md version 3.1, section 7.2.
Verified by the specification's author on November 12 and 13, 2026. Owner from here: Daniel, through the executing agent.
Maintained by the executing agent: rows re-read, a citation-graph pass and the repeated bounded search on November 13, 2026 (section "Maintenance" below). Re-reads by the agent's research worker used each paper's own PDF text.
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
| 2 | ACTMED, "Timely Clinical Diagnosis through Active Test Selection" | NeurIPS 2025 (first-page header); arXiv 2510.18988 v5, Sep 10 2026 | A language model samples ten outcomes per missing test at temperature 1 with a "Sampling Prompt", then re-queries its unchanged "Risk Prediction Prompt" with the simulated value appended (Appendices C, D); tests are chosen by expected KL per cost; stopping compares expected KL with a target. Both prompts go to the same model | The simulate-and-re-query pairing we study. No coherence check, no correction, no world model separate from the reader. Its own appendix reports models that collapse or sample implausibly and cases where it falls below using all features. v5's Algorithm 1 still indexes the prior by sample while a v4 note says it is constant; cite by version | E7: audit this pairing on its own datasets, with and without the projection and scale layers; E1-B raw-pairing baseline | R (Algorithm 1, Appendices C, D, F); repository not opened |
| 3 | BED-LLM. Choudhury et al. | ICLR 2026 proceedings; arXiv 2508.21184 v3, Apr 20 2026 | Sequential experimental design with a language model. Builds the joint as prior times likelihood, filters sampled hypotheses against the history, and warns that the alternative pairing gives no joint model | Names the incoherence we measure and avoids it by construction. Does not measure its size, repair it, or decompose its cost. No medical task, no separate simulator | E1-B and E7: prior-times-likelihood baseline against repair of the re-query pairing, where a trained reader makes the likelihood pairing unavailable | R for the pairing argument; F for appendices B and D |
| 4 | Consistency checks for language-model forecasters. Paleka et al. | ICLR 2025 | Defines consistency checks, including expected evidence, scores violations by arbitrage under a log scoring rule, and builds ArbitrageForecaster, which patches a forecaster's answers toward consistency | Closest precedent for correcting a language model's probabilities toward the expected-evidence constraint. There the model supplies every probability, questions are forecasting questions, and the paper reports the smallest consistency gain for expected evidence and no clear accuracy effect | E7 baseline: arbitrage-style patching with model-supplied outcome probabilities against projection under a separate world model, scored by realized log loss | R |
| 5 | From Drift to Coherence. Kim, Lee, Fong, Lee, Lee | ICML 2026 (PMLR 306, the paper's own footer); arXiv 2606.17832 v2, Jun 18 2026 | Measures drift of a language model's predictive under its own resampled answers (a martingale posterior over repeated answers to one question) and repairs it two ways: an inference-time answer-seeding prompt (Prompted Predictive Resampling) and a training-time loss that amortizes it | A repair of martingale violations for self-generated data on question answering, at inference by prompting and at training by a loss. No separate world model, no decisions or acquisition, no closed-form correction against an explicit outcome distribution, no accounting | E2: our closed-form projection under a separate world model on the same kind of drift; cite both of their repairs as alternatives | R (whole text) |
| 6 | CaRDI. Zhou et al. | arXiv 2608.27847 v2, Oct 16 2026, "Beyond Uncertainty Reduction: Calibrated Planning for Interactive Diagnosis" | Risk-sensitive planning for question asking on DDXPlus and MediQ: diagnostic and safety beliefs, conformal calibration of realized clinical loss, tree search over ask and commit | Calibrated risk for acquisition and stopping. On DDXPlus one count-based model (conditional independence given the disease) both simulates outcomes and updates (Eqs. 2, 9), so reading is coherent by construction and correlated evidence is the open question. On MediQ the reader is a language model and search uses a separate empirical transition model over belief summaries (Appendix E.2) with no check that ties its mean to the current belief: nearer our setting, but a transition model, not a projection | E1-B conformal-stop baseline under correlated evidence and duplicated panels | R (sections 3.3-3.6, Appendices A and E, Limitations) |
| 7 | CROS, "Safe to Stop?". Wu and Rus | arXiv 2609.09678 v1 | Risk-constrained stop, continue or defer layer for an LA-CDM-style agent on 1,834 MIMIC-IV-ED episodes; reports that error is 28.3% with the history alone and 34.3% after the full workup | Shows on a sibling agent that more evidence can raise error. Ranks risk; does not explain or correct the reading, never substitutes a test, uses no imagined outcomes | E4 forced-continuation curve raw and corrected; simplified error-ranker stop baseline | F |
| 8 | LDTL | arXiv 2604.05116 v2, Oct 22 2026 | Trains a planner to align with a frozen diagnostic agent's gain in log probability of the true diagnosis, using recorded outcomes; 93.4 accuracy on MIMIC-CDM against 84.8 for a random planner and 81.3 for LA-CDM as reported there | Uses the reader's realized gain as the training signal for selection. No simulator, no coherence check, binary confidence stop | E1-A attribution: how much regret selection can remove once reading is fixed | R for the method and tables. U: the paper does not state which reader the random-planner baseline uses; the frozen Stage-1 agent is an inference, not a statement |
| 9 | Current Agents Fail to Leverage World Model as Tool for Foresight. Qian et al. | ACL 2026; arXiv 2601.03905 | Across control and visual question answering, agents rarely call a simulator, misuse about 15% of rollouts and can lose up to 5% when simulation is available or enforced; the authors' project page adds that the control tasks used exact simulators | Establishes that integrating foresight is the bottleneck. Offers a taxonomy, not a measurement of belief coherence or a correction | E2 and E4: the audit gives that bottleneck a number per reader | A |
| 10 | Learning-to-Measure | ICML 2026 (PMLR 306 footer); arXiv 2510.12624 v2, May 31 2026; datasets Metabric, MiniBooNE, MNIST, MIMIC-IV (three binary tasks) and synthetic GP and BNN tasks, real-data labels semi-synthetic | In-context active feature acquisition with one pretrained sequence model that has a target head and a policy head | Imagination and reading come from one model. No external reader, no incoherence to measure | E1-B: a jointly trained reference shows what coherence by construction buys | R (venue, Appendix A.7.1) |
| 11 | Calibrated Value-Aware Model Learning. Voelcker et al. | ICML 2025 per N12; arXiv 2505.22772 v2 | Shows value-aware and MuZero-style model losses are uncalibrated surrogates and corrects them | Weighting model error by what the decision maker does with it. "Calibrated" there means loss calibration | E1-C: reader-weighted error against likelihood, with the matched pair of model errors | A |
| 12 | Uncertainty of Thoughts. Hu et al. | NeurIPS 2024 per N12; arXiv 2402.03271 v3 | Simulates possible futures, scores questions by information-gain rewards, propagates rewards to choose a question | Imagined outcomes guiding acquisition | Raw information-gain baseline in E7 | A |
| 13 | MediQ. Li et al. | NeurIPS 2024 per N12; arXiv 2406.00922 v3 | Interactive clinical benchmark; the expert abstains and asks when unconfident | Confidence-threshold stopping | Confidence-threshold commit baseline | A |
| 14 | Sequential Diagnosis with Language Models (SDBench, MAI-DxO). Nori et al. | Science 394(6822):423-428, Oct 22 2026, DOI 10.1126/science.aeh7930; arXiv 2506.22405 v2, Jul 2 2025 (the two versions report different numbers; do not mix them) | 304 NEJM cases as stepwise encounters with a gatekeeper that reveals findings on request | Sequential diagnosis with test costs; findings are synthesized by a model when absent | Not a method baseline; cited for the setting | A (arXiv v2 text, Science abstract and excerpts) |
| 15 | Generative surrogate models. Li and Oliva | ICML 2021 per N12; arXiv 2010.02433 v2 | Learns a generative model of unobserved features to guide acquisition | Imagination for acquisition, classical setting | None; lineage | A |
| 16 | Acquisition conditioned oracle. Valancius, Lennon, Oliva | ICML 2024 per N12; arXiv 2302.13960 | Nonparametric oracle for non-greedy acquisition | Retrieval in place of a parametric outcome model, as in our retrieval sampler | None; lineage for W3 | A |
| 17 | DIVA, "Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs". Seunghan Lee et al., LG AI Research | arXiv 2610.32181 v1, Oct 23 2026 | Shows that raw imagined updates by language models are incoherent: across 20 models the outcome-weighted average of imagined beliefs, weighted by the exact predictive, differs from the current belief by 6.3 percentage points on average over 496 controlled probes of binary checks (section 3.2, Appendix D.1). Avoids it by construction: the model supplies a base rate and outcome likelihoods and Bayes' rule computes every update; evidence strength is shrunk by a factor tuned on development data; net-of-cost VOI decides asking and stopping | The measurement of the violation is not new: DIVA measures the same quantity as our drift, as an aggregate over controlled binary probes. It does not repair a given reader's posteriors (the model is replaced as the updater), has no separate reader and world model in the deployed method, no information-budget matching, no accounting identity, binary outcomes, no trained clinical agent | E1-B and E7: by-construction baseline beside BED-LLM; our per-decision, label-free audit against a separate world model for multi-class trained readers whose posteriors cannot be replaced | R (sections 3.2-3.3, Appendix D.1-D.2, the DIVA-PV ablation); rest keyword-screened; repository not opened |
| 18 | "Locally Coherent, Globally Incoherent: Bounding Compositional Incoherence in Multi-Component LLM Agents" | arXiv 2605.30335 v1, May 28 2026 | Projects a language model's probability quotes onto a coherent polytope (hierarchical Boyle-Dykstra), with a Pythagorean improvement guarantee and an anytime-valid monitor | Projection repair with a Pythagorean guarantee exists, for logical constraints among forecasts (conjunction, partition), not for expected evidence; no acquisition, no world model | Cite for the projection-with-guarantee lineage | R for the abstract and projection sections; proofs not read |
| 19 | MedClarify | arXiv 2602.17308 v1, Feb 19 2026 | For each candidate question the model is prompted twice, assuming a yes and a no answer, and a diagnostic expected information gain is computed from the two simulated posteriors | Another instance of the simulate-and-re-query pairing in diagnosis; no coherence check found | E7 raw-pairing baseline lineage | Passages and keyword screen only |

## Audits of belief updating in language models

| Work | Version | What it shows | Relation | Status |
|---|---|---|---|---|
| Falck, Wang, Holmes, "Is In-Context Learning … Bayesian? A Martingale Perspective" | ICML 2024 per N12; arXiv 2406.00793 | In-context learning violates the martingale property | Motivates the audit; no repair | A |
| Martingale Score. He, Shirado, Qiu, Sap | NeurIPS 2025 per N12; arXiv 2512.02914 | Regression of belief change on prior belief across reasoning trajectories; a metric only, with a training objective left as future work | Aggregate entrenchment score over many items; ours is per decision point against an explicit outcome distribution | F |
| Filbry, Melo, Gal, "From Prompts to Posteriors" | arXiv 2610.29331 v1, Oct 21 2026 | Language models generally violate Bayes' rule in context; tools and reasoning help | Audit; no correction, no decisions | A |
| Pal, Kitanovski, Liang, Potti, Goldblum, "Knowing What You Know Is Not Enough: Large Language Model Confidences Don't Align With Their Actions" | arXiv 2511.13240 v3, Feb 9 2026 | An action-belief gap: actions contradict elicited confidences in betting, tool-use and user-challenge settings | Audit of confidence against action, not of Bayesian coherence of updates | R (abstract, introduction) |
| BayesBench, Samanta et al. | arXiv 2606.30850 v1, Jun 29 2026 | Scores per-turn belief trajectories of seven open models (3B-70B) against Bayesian references on estimation, recommendation and latent-framed prediction tasks, including medical triage | Audit; no expected-evidence constraint, no repair | R (abstract, introduction) |
| "LLMs are not (consistently) Bayesian" (Apple) | arXiv 2605.06915 v2, May 26 2026 | Measures an information-processing gap from Bayes updates; non-Bayesian heuristic updates sometimes beat exact Bayes, which it attributes to misspecified world models | Audit; relevant to our world-model-error term | Abstract and key passages |
| "Latent Coherence: Do Frontier Language Models' Probability Judgments Obey the Law of Total Probability?" | arXiv 2610.31833 v1, Oct 22 2026 | Law-of-total-probability residuals with exact ground truth; the label-free residual flags unreliable answers (Spearman 0.40-0.62) | Audit; supports label-free flags; no repair, no sequential task | Abstract and key passages |
| "Partition, Prompt, Aggregate" | arXiv 2607.15277 v1, Jul 16 2026 | Law-of-total-probability audit over population partitions | Audit | Abstract |
| "Revealed Rationality" | arXiv 2608.05015 v1, Aug 5 2026 | Label-free de Finetti coherence penalties for evaluation and training | Lineage | Abstract |
| "Probabilistic Coherence of Stated Diagnostic Beliefs in Large Language Models" | Mach. Learn. Knowl. Extr. 8(11):369, Nov 7 2026 | Order-invariance audit of clinical beliefs on a synthetic corpus | Audit | Search snippet only (U) |

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

## Maintenance, November 13, 2026

Done by the executing agent's research worker; full report kept privately in the session's scratch
space. Row re-reads as recorded in the status column above.

Citation-graph pass: papers citing ACTMED (6 found), BED-LLM (35), Kim et al. (3) and Paleka et al.
(32) on Semantic Scholar, about 30 screened by full-text keyword search; lists probably incomplete
(Google Scholar and arXiv listings not queried). Result: DIVA (row 17) added as a core row; rows 18
and 19 and four audits added; the rest judged not overlapping.

Repeated bounded search with the final method description: five searches (aggregate web search,
semantic search) around conservation of expected evidence, imagined posteriors, projection, IPF and
Jeffrey conditioning in language-model belief updating, sequential diagnosis and feature
acquisition, 2024 onward.

Result, and what it changes:
- The measurement of the violation is not new. DIVA measures the outcome-weighted gap between
  imagined and current beliefs across 20 language models. Claim 1 of the specification (section
  1.4) stands only as narrowed: a per-decision, label-free audit against a separate world model's
  outcome distribution, for black-box multi-class readers including a trained clinical agent, that
  separates drift, scale and redundancy.
- Coherence by construction is a crowded design: BED-LLM, DIVA, CaRDI on DDXPlus, Learning-to-Measure,
  and others. Repairs of martingale drift exist by prompting and training (Kim et al.), and
  projection with a Pythagorean guarantee exists for other coherence constraints (row 18).
- Not found: a projection of an agent's imagined posteriors onto "expected posterior equals current
  belief" under a separate world model; a scale repair matched to a per-test information budget; an
  exact identity splitting the log-score gain of such a correction. This is a bounded search, not
  proof of absence. The honest position is the combination, with DIVA cited as the closest
  measurement and the closest by-construction alternative.
