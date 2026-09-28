# Testing in Imagination

> We extend LA-CDM from retrospective, model-free test selection to world-model–based “testing in imagination,” so an agent can evaluate counterfactual diagnostic actions — and we measure not only when imagination helps, but when a flawed world model makes planning worse.

**Course:** CAI6734 (UF) · **Group:** Daniel Escalante, Hugo Resendez, Edwin Salcedo · **Advisor:** Xuefeng Liu

**Baseline:** LA-CDM (Bani-Harouni et al., ICLR 2026) — [arXiv:2506.13474](https://arxiv.org/abs/2506.13474) · [code](https://github.com/dharouni/LA-CDM) · [OpenReview](https://openreview.net/forum?id=7vHUQCMAzG)

## Stage 1

1. Reproduce (honest partial OK) under logged configs/seeds/metrics — see `REPRODUCTION.md`.
2. Formalize env / action space.
3. Design \((h_t, a_t, o_{t+1})\) tuples for a world model.
4. Prefer **synthetic / semi-synthetic** env before full MIMIC-CDM.

## Repo map

| Path | Purpose |
|------|---------|
| `ONBOARDING.md` | Standing constitution (merged from comprehensive prompt) |
| `REPRODUCTION.md` | Claims ledger — GitHub ≠ reproduced |
| `docs/` | Paper notes, env/tuples spec, gated method-spec, open questions |
| `onboarding/` | Full comprehensive prompt + coding-agent brief |
| `baseline/` | Upstream LA-CDM as submodule (later); never vendor |
| `extension/` | Synthetic env, world model, planners |
| `extension/envs/synthetic_diagnosis/` | Stage 1 synthetic CDM env + JSONL tuple export |
| `tools/` | Offline verification scripts |
| `results/` | Logged metrics only |


## Default agent backbone (HiPerGator)

**`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`** — US base (Meta Llama 3.1 8B Instruct) + NVIDIA post-train. Upstream LA-CDM’s Qwen2.5-7B is **not** allowed on HiPerGator (UF CoC). See `REPRODUCTION.md` § D-5.

## Hard rules

- Never invent Table 1 / MIMIC numbers.
- Never commit PHI, PhysioNet extracts, or credentials.
- Do not email Liu or the group unless Daniel asks.
