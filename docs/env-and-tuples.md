# Spec: Environment formalization & world-model tuples

**Status:** Draft v0 — Stage 1 design + code in `extension/envs/synthetic_diagnosis/`  
**Date:** 2026-09-22  
**Project:** Testing in Imagination

## 1. Why formalize before coding

LA-CDM’s env is informal prose + Hydra configs. For a world model we need a crisp interface so synthetic and MIMIC-backed envs share one tuple schema.

## 2. POMDP-style view (working)

- **Underlying patient** \( z \): full bag of latent test outcomes + true label \( y \).
- **History** \( h_t \): textual context so far (system prompts + revealed observations + prior actions). In LA-CDM this is roughly \( p_t \) plus dialogue formatting.
- **Action** \( a_t \in \mathcal{A} = \mathcal{A}_{\mathrm{test}} \cup \mathcal{A}_{\mathrm{diag}} \).
  - Example \( \mathcal{A}_{\mathrm{test}} \): the 12 LA-CDM tests (exact string names).
  - \( \mathcal{A}_{\mathrm{diag}} \): {appendicitis, cholecystitis, diverticulitis, pancreatitis}.
- **Observation** \( o_{t+1} \):
  - If \( a_t \in \mathcal{A}_{\mathrm{test}} \): test result text **or** `UNAVAILABLE`.
  - If \( a_t \in \mathcal{A}_{\mathrm{diag}} \): terminal signal (correct/incorrect) for RL; not used as WM target in planning loops.
- **Reward** (model-free baseline): as in LA-CDM \( R_{diag} + R_{cost} \). Planning variants may use imagined returns.

**Missingness (baseline):** \( o_{t+1} = \mathrm{UNAVAILABLE} \) if test not in chart. That is **not** a counterfactual of what the test *would have shown*.

## 3. World model objective (ours)

Learn \( p_\phi(o_{t+1} \mid h_t, a_t) \) for \( a_t \in \mathcal{A}_{\mathrm{test}} \).

Uses:

1. **Imagination rollouts:** planner proposes tests, samples/modes \( \hat{o} \sim p_\phi \), updates imagined \( \hat{h} \), scores diagnostic value / cost.
2. **Comparisons:** model-free vs WM-planner vs **oracle** WM (true \( z \) for requested tests, even if not in chart — only possible in synthetic or imputed sim).
3. **Failure analysis:** deliberately degraded \( \phi \) should be able to **hurt** planning vs model-free (publishable negative result).

## 4. Training tuple schema

Canonical record (JSONL / parquet later):

```json
{
  "episode_id": "string",
  "t": 0,
  "h_t": "string — serialized history / patient state text",
  "a_t": "string — exact test name",
  "o_t1": "string — observation text or UNAVAILABLE",
  "y_true": "string — diagnosis label",
  "source": "synthetic | mimic_real | mimic_imputed",
  "meta": {
    "split": "train|val|test",
    "available_tests_at_t": ["..."],
    "seed": 0
  }
}
```

**Filters for Stage 1 WM training:**

- Keep only rows where \( a_t \in \mathcal{A}_{\mathrm{test}} \).
- Optionally exclude `UNAVAILABLE` for a “predict real labs” model; **or** include them as a discrete outcome class (two-headed model: availability + content).

**Recommendation:** synthetic env first with **always-available** tests so every \( (h_t,a_t) \) has a true \( o_{t+1} \); add missingness as a second phase.

## 5. Synthetic / semi-synthetic env (preferred Stage 1)

Minimal design that still stresses planning:

1. **Latent disease** \( y \sim \mathrm{Cat}(4) \).
2. Each test \( a \) has a **template generator** \( g_a(y, \epsilon) \rightarrow \) observation text (or structured features rendered to text).
3. Optional noise / distractors so greedy one-test policies are suboptimal.
4. Cost vector reused from LA-CDM Table 4 (or scaled).
5. Oracle WM = call \( g_a(y,\cdot) \); learned WM = fit \( \phi \) on logged trajectories from a random or expert policy.

**Success criteria for synthetic:**

- Model-free agent and WM-planner both runnable end-to-end.
- Oracle planner ≥ model-free ≥ bad-WM planner (ordering, not absolute numbers).
- Tuple export matches schema above.

## 6. Bridge to MIMIC later

- **Real tuples:** from LA-CDM rollouts / dataset: only tests present in chart.
- **Imputed counterfactuals:** research risk — must be labeled `mimic_imputed` and never mixed silently with real.
- PhysioNet access is orthogonal to synthetic progress.

## 7. Experiment matrix (preview)

| Arm | Policy | Observation source |
|-----|--------|--------------------|
| MF | LA-CDM-style decision agent | Real env only |
| WM-plan | Lookahead with \( p_\phi \) | Imagined then optional real confirm |
| Oracle | Lookahead with true \( g \) / full \( z \) | True counterfactuals |
| Bad-WM | Same planner, corrupted \( \phi \) | Imagined |

Metrics: accuracy / F1, avg cost, ECE (if hyp agent kept), **regret vs oracle**, **Δ vs MF when WM is bad**.

## 8. Immediate next engineering tasks

1. Freeze string inventories for \( \mathcal{A}_{\mathrm{test}} \) and diagnoses (copy LA-CDM prompts).
2. Implement toy `SyntheticCDMEnv.step(h, a) -> o`.
3. Logger emitting JSONL tuples.
4. Tiny baseline: bag-of-words or small LM for \( p_\phi \) on synthetic.
5. Parallel track: PhysioNet application / data prep for LA-CDM ZS eval if credentials exist.
