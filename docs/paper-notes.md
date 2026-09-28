# LA-CDM paper notes

**Cite:** Bani-Harouni, Pellegrini, Özsoy, Keicher, Navab. *Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning.* ICLR 2026. arXiv:2506.13474.

**Code:** https://github.com/dharouni/LA-CDM (Python 3.11–3.12, CUDA, `uv sync`, Hydra, LoRA on Qwen-2.5-7B-Instruct, GRPO via cyclic objectives).

## Problem they solve

Most LLM clinical CDSS either (1) assume all patient info is available up front, or (2) use zero-shot / untrained interactive test-requesting. Real diagnosis is iterative differential diagnosis: form hypotheses, order tests, update beliefs until confident enough to diagnose.

## Method (two agents, shared LLM weights)

**Environment (retrospective MIMIC-CDM):**

- Patient = set of textual test results \( [t_i]_{i=1}^n \) (notes, imaging reports, labs).
- Observed state \( p_j \) = tests revealed so far; \( p_0 \) = history / HPI (always available).
- Action = request one named test **or** commit diagnosis \( y_{pred} \).
- Unavailable tests (not ordered in the chart): environment returns “unavailable”; agent must pick another action.
- Episode ends on diagnosis, token budget, or format violation.

**Hypothesis agent** \(\mathcal{H}: p_j \mapsto \{h_j, c_j\}\)

- Output format: `Hypothesis: <one of 4>, Confidence: <0–10>`
- Trained with SFT for correct \( h_j = y_{true} \) (confidence token ignored in CE).
- Uncertainty trained with GRPO using Stangel et al. betting-style reward on hypothesis correctness.

**Decision agent** \(\mathcal{D}: \{p_j,h_j,c_j\} \mapsto\) test \( t_j \) or diagnosis \( y_{pred} \)

- ReAct: Thought / Action / Action Input / Observation.
- GRPO with diagnosis reward \( R_{diag} \) (correct / wrong / invalid) plus optional cost penalty \( R_{cost} = -\sum c(t_j) \).

**Cyclic training (100 steps each):** (1) clinical action GRPO → (2) hypothesis SFT → (3) confidence GRPO. Adam \( 1e{-5} \), batch 2, LoRA, ~3 days on 1× A40.

## Dataset

- MIMIC-IV-Ext-CDM / MIMIC-CDM (Hager et al.): 2,400 patients; 4 abdominal dx: appendicitis, cholecystitis, diverticulitis, pancreatitis.
- Split (paper): 80/10/10 train/val/test (dataset originally for zero-shot eval; they introduce the split).
- Tests (12): Physical Examination, CT, MRI, Radiograph, Ultrasound, CBC, BMP, CMP, Renal, Liver, Urinalysis, Electrolyte Panel.
- Prep: PhysioNet → MIMIC-CDM processing repo → `scripts/prepare_data.py` (split + lab mapping + LLM HPI summaries).

## Headline claims (Table 1 — reproduce under logged config)

| Method | Mean Acc | Micro F1 | Macro F1 | Avg Test Cost |
|--------|----------|----------|----------|---------------|
| ReAct (ZS) | 74.9 | 79.1 | 74.8 | $1480.32 |
| LA-CDM (ZS) | 64.5 | 65.3 | 64.5 | $1521.73 |
| **LA-CDM** | **81.3** | **84.1** | **81.3** | **$1295.61** |
| SFT-all (oracle info) | 92.8 | 93.6 | 92.9 | $3792.79 |

Other reported deltas:

- Hypothesis accuracy 75.7% → 81.9% with training.
- ECE 0.069 → 0.037.
- Cost ablation: with cost reward, mean acc 81.3 vs 82.3 without, cost $1295 vs $1428.
- Hypothesis agent ablation helps all metrics (Table 3).

## Hard limitation (motivation for *our* paper)

> “The data we are training on is retrospective with different tests missing for different patients… The model can only explore a limited spread of testing pathways… Simulation of unavailable test data could open up a pathway for modeling a more holistic clinical decision-making environment.”

That is exactly the gap: **no true counterfactual observations** for tests never ordered. A world model \( p_\phi(o_{t+1}|h_t,a_t) \) is the natural extension.

## Repo layout (upstream)

```
configs/   # Hydra: dataset, environment, evaluation, experiment, model, reward, training, wandb
prompts/
scripts/   # prepare_data.py, split_dataset.py, create_data_with_summaries.py
src/
  dataset/
  environment/
  reward_function/
  trainer/
  utils/
  train.py
  evaluate.py
```

## Practical barriers for UF reproduction

1. PhysioNet / MIMIC credentialed access (MIMIC-IV-Ext-CDM).
2. GPU: paper used A40 48 GB; Qwen-2.5-7B + vLLM + GRPO is heavy. **Our HiPerGator backbone:** `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` (see `REPRODUCTION.md` D-5) — not Qwen.
3. Data prep needs LLM summarization GPU pass.
4. Seeds / exact Hydra overrides for Table 1 may need logging from our runs — not fully pinned in the paper beyond Appendix C.

## Mapping to our Stage 1 formalization

| LA-CDM symbol | Our working notation |
|---------------|----------------------|
| \( p_j \) observed patient state | history \( h_t \) (text context) |
| decision action (test or diagnose) | \( a_t \in \mathcal{A}_{\text{test}} \cup \mathcal{A}_{\text{diag}} \) |
| environment reply (test text or N/A) | observation \( o_{t+1} \) |
| world model (ours) | \( p_\phi(o_{t+1} \mid h_t, a_t) \) |

Model-free LA-CDM never learns \( p_\phi \); it only conditions on real returned observations.
