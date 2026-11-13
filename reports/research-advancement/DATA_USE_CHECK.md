# DATA_USE_CHECK, study TII-CEE-1

Specification section 20.4 and launch check 7.4b. Written November 13, 2026. Neither Daniel,
AUTHORIZATION.md nor the agent can waive the data-use terms; a new data flow proceeds only if every
answer below can be read from the documents.

## 1. Governing documents

Copies were saved on November 13, 2026 under reports/research-advancement/sources/ (git-ignored;
third-party text is not committed). SHA-256 values are of the saved copies.

| Document | Version | URL | SHA-256 of the saved copy |
|---|---|---|---|
| PhysioNet Credentialed Health Data Use Agreement (served on the dataset's view-dua page; no standalone page exists) | 1.5.0 | https://physionet.org/content/mimic-iv-ext-cdm/view-dua/1.1/ | 57566d7737e767f6096e8d07af00d77c86e4fb59ee37c44f24c05cf23b823a88 |
| PhysioNet Credentialed Health Data License | 1.5.0 | https://physionet.org/content/mimic-iv-ext-cdm/view-license/1.1/ | 249561da735e879fb20dd93b9f316cbe29ca7d2e27100255960c0516edd095a1 |
| MIMIC-IV-Ext Clinical Decision Making, dataset page | 1.1 | https://physionet.org/content/mimic-iv-ext-cdm/1.1/ | 137fc1393ca7a572873d70f5993f458b96999cf0dc7312439054437e8646d5f0 |
| "Use of MIMIC Data with Large Language Models and Online Services" (current notice) | September 24, 2025 | https://physionet.org/news/post/llm-responsible-use/ | 4dcc63733e1712a6196409050813f13527baf75c093c9529618f58d3f204a307 |
| "Responsible use of MIMIC data with online services like GPT" (earlier notice; its list of services is not relied on) | April 18, 2023 | https://physionet.org/news/post/gpt-responsible-use/ | f22d5ffa8ea18ea2a203daf1545baaea44f6d9c5bdd254b00df46d84ff81458e |
| Project storage rules | as committed | SPEC_RESEARCH_ADVANCEMENT.md 3.4 and 20.4; cdm-extension/daniel-escalante/DECISIONS.md "Execution" and "Data boundary"; AUTHORIZATION.md sections 3 and 6 | in the repository |

Clauses relied on, quoted from the saved copies:

- C1, sharing: "I will not share access to PhysioNet restricted data with anyone else." (license:
  "The LICENSEE will not share access to PhysioNet restricted data with anyone else.")
- C2, security: "I will exercise all reasonable and prudent care to maintain the physical and
  electronic security of PhysioNet restricted data."
- C3, identification: "I will not attempt to identify any individual or institution referenced in
  PhysioNet restricted data." and "I will exercise all reasonable and prudent care to avoid
  disclosure of the identity of any individual or institution referenced in PhysioNet restricted
  data in any publication or other communication."
- C4, purpose: "I have requested access to PhysioNet restricted data for the sole purpose of lawful
  use in scientific research, and I will use my privilege of access, if it is granted, for this
  purpose and no other."
- C5, services: "The PhysioNet Credentialed Data Use Agreement explicitly prohibits sharing access
  to the data with third parties, including sending it through APIs or using it on online
  platforms." and "Strongly Recommended: Use locally deployed LLMs to maintain full control over the
  data." (notice of September 24, 2025)
- C6, publication: "If I openly disseminate my results, I will also contribute the code used to
  produce those results to a repository that is open to the research community."

The agreement has no clause on derived data, copies or indexes; for those the security clause C2
and the project's storage rules govern (specification 3.4: "retrieval indexes, and patient-derived
fitted parameters stay in approved private storage on HiPerGator").

## 2. Flow F1: a validation-stage or test-stage job reads training-split result text at inference (retrieval world model W3)

| Question | Answer | Basis |
|---|---|---|
| Does patient-level content leave approved private storage? | No. The job runs on a HiPerGator compute node, reads training-split result text from the private data directory, and writes the index and every imagined result under /blue/cai6734/daniel.escalante/private/tii-cee-1/private (owner-only, mode drwx--S---, verified November 13, 2026). | C2; specification 3.4 |
| Does any third party, service or API receive it? | No. W3 is a nearest-neighbor index with no model behind it; every model in the study runs on HiPerGator. The hosted coding agent receives only code, invented fixtures, scheduler metadata and export-guarded aggregates. | C1, C5 |
| Does any person other than the credentialed user gain access? | No. The directories are owner-only under Daniel's account. | C1 |
| Does it create a new copy, index or derived artifact, and where? | Yes: a retrieval index over training-split results and the imagined results drawn from it. Both are protected data stored only in the private directory above, never exported. | C2; specification 3.4, 20.4 |
| Does it raise the risk of identifying a person or institution? | No. It recombines content already in the dataset under the same access, adds no external linkage, and the dataset carries no subject identifiers (v1.1). Nothing patient-level is published or exported. | C3 |
| Is the purpose lawful scientific research under the agreement? | Yes: a world model for evaluating diagnostic agents. | C4 |
| Does it conflict with the project's stage-by-split rules, and what covers it? | The legacy tim access table lets a validation-stage job read only the validation split, so F1 is a new cross-split read. It is covered by AUTHORIZATION.md section 6 (F1) and implemented in this study's own stage table. The rules' purpose, no leakage into evaluation, is kept by the information-boundary tests of specification 8.4: W3 reads the training split only and never returns rows of the evaluated admission or of the validation or test split. | AUTHORIZATION.md 6; specification 8.4 |

F1 PASSES. Every answer is read from the documents above.

## 3. Flow F2: a job scores a trained policy on a real history with an imagined result substituted (reader probe R1)

| Question | Answer | Basis |
|---|---|---|
| Does patient-level content leave approved private storage? | No. The policy, the generative sampler W4 (model 5) and the probe run on HiPerGator; substituted histories, scores and generations stay in the private directory. | C2; specification 3.4 |
| Does any third party, service or API receive it? | No. Every model is a locally deployed open-weight model on HiPerGator, the arrangement the September 2025 notice strongly recommends. | C1, C5 |
| Does any person other than the credentialed user gain access? | No. | C1 |
| Does it create a new copy or derived artifact, and where? | Yes: substituted histories, imagined result text, and per-step beliefs. Generated clinical text is protected data, kept in the private directory, never exported; only schema-approved aggregates leave. | C2; specification 3.4, 20.4 |
| Does it raise the risk of identifying a person or institution? | No. A substituted history combines content already under the same access; nothing is linked to external data or published. | C3 |
| Is the purpose lawful scientific research? | Yes: measuring how a trained agent reacts to imagined evidence. | C4 |
| Does it conflict with the stage-by-split rules? | It scores histories only of the splits each experiment is authorized for: validation (cohorts P, S, B), the cross-fitted training-plus-validation cohort X (AUTHORIZATION.md section 6), and the test split only inside E6 (section 8). Covered by AUTHORIZATION.md section 6 (F2). | AUTHORIZATION.md 6, 8 |

F2 PASSES. Every answer is read from the documents above.

## 4. Standing consequences

- Patient-level content never reaches the hosted coding agent, the Mac, the monitor app or any
  synced folder. The agent never prints, pages, searches or copies patient-level files; tools
  summarize on the cluster. The September 2025 notice requires zero data retention by third-party
  LLM services; this study meets it by sending them nothing patient-level.
- Every model that reads patient text runs on HiPerGator.
- Retrieval indexes, generated clinical text and patient-derived fitted parameters are protected
  data in the private directory.
- Publication obligation (C6): if results are openly disseminated, the code that produced them is
  contributed to an open repository. For a double-blind venue the code goes as anonymized
  supplementary material at submission (ICML 2027 author instructions) and to an open repository
  when the paper is made public; Daniel decides when the repository is pushed.

## 5. Public datasets of E7 (AUTHORIZATION.md section 7)

| Dataset | License from the official page | Saved copy SHA-256 | Decision |
|---|---|---|---|
| Chronic Kidney Disease, UCI 336 | CC BY 4.0 | c03837118d2e5f23cd061dc3ffda501bc400afafa4924236f9e080d5ef8c86ff | Use, with attribution |
| HCV data, UCI 571 | CC BY 4.0 | 42d394d094746d73f053df74cb0817c8d7bee3e3b635b337119506823c4e5164 | Use, with attribution |
| DDXPlus, figshare 20043374 v15 | CC BY 4.0 (record license field; README: "We are releasing under the CC-BY licence") | 11047556d26824edff3b46da810c3268deb6c8c185e068c1039dfa27f76fd595 | Use, with attribution |
| Pima Indians Diabetes, the Kaggle page ACTMED cites | Not readable: the page returns 404 to anonymous requests and the Kaggle API returns 403. The original UCI page carries the donor's note: "The dataset is no longer available due to permission restrictions." | 914813bf8c43c5bd42447b88e2b70444131e09c6a067f5cfc7912baae5e80438 (UCI archive snapshot) | Not used. AUTHORIZATION.md section 7 requires the license to be recorded from the official page before use; it cannot be, and the original source withdrew the data for permission reasons. E7 runs on three datasets (amendment on EXPERIMENT_CARDS/E7.md) |
