# REPRODUCTION.md — LA-CDM claims checklist

**Rule:** GitHub clone ≠ reproduced. A claim is reproduced only under **logged config + seed + metric artifact**, or we mark an honest gap.

**Upstream:** https://github.com/dharouni/LA-CDM · commit pin: _fill after submodule_ · paper: arXiv:2506.13474 · OpenReview: https://openreview.net/forum?id=7vHUQCMAzG

**Milestone:** M0 (onboarding + ledger opened). Phase A archaeology extended below; no training run yet.

## Environment log (fill as you go)

| Field | Value |
|-------|-------|
| Date | |
| Operator | |
| Machine / GPU | |
| CUDA / driver | |
| Python | 3.11–3.12 (required) |
| Install | `uv sync` |
| Upstream SHA | |
| `VLLM_USE_V1` | `0` |
| Data source | PhysioNet MIMIC-IV-Ext-CDM path |
| Split seed | (none canonical — see D-4) |
| Hydra overrides | |
| W&B run ids | |

## Archaeology findings (preserve — do not redo from zero)

| ID | Finding | Status |
|----|---------|--------|
| **D-1** | Paper/code narrative uses `incorrect_reward = -1.0`, but shipped Hydra config sets `0.0`. Wrong diagnosis == undiagnosed reward under default. **Reproduce with shipped config; treat −1.0 vs 0.0 as later ablation — do not silent-fix.** | Confirmed (config read) |
| **D-2** | Physical examination cost missing from Appendix C Table 4 but present in config (−0.0294 ≈ $300). Twelve costs sum to 1.0 as claimed. | Confirmed; verify with `tools/verify_cost_normalization.py` when added |
| **D-3** | Paper summarizes HPI with Mixtral-8x7B; repo `prepare_data.py` defaults to Qwen2.5-7B-Instruct. Input-text divergence. | Confirmed |
| **D-4** | No canonical 80/10/10 split seed. Expect Table 1 mismatch; quantify with ≥3 split seeds. | Confirmed |
| **F-1** | Unavailable tests share the same `trajectory.error` flag as unknown action / bad test name / bad diagnosis — conflated. | Confirmed (code read) |
| **F-2** | Cost uses `provided_tests` only → unavailable requests are **free**; headline $1295.61 is cost of tests that returned data. | Confirmed |
| **F-3** | Unavailable path skips hypothesis update (stale hypothesis one turn) + burns a step. `max_steps: 13` vs 12 tests → little pressure to avoid unavailable requests. | Confirmed |
| **B-1** | Full stack needs CUDA (vLLM). Local Mac (Apple Silicon) cannot produce Table 1. Local OK for Phase E + synthetic. Prefer HiPerGator; cloud A40 ~$0.49/hr ≈ ~$35 paper-length — **ask before renting**. | Confirmed |

**Thesis implication:** baseline env makes unavailability cheap and uninformative; a world model changes the economics so every planned action yields an observation. Confirm F-1–F-3 on a smoke episode before write-up.

**Upstream license:** none listed. Submodule only; never vendor.

## Setup claims

| ID | Claim | Status | Evidence / gap |
|----|-------|--------|----------------|
| S1 | Repo installs with `uv sync` on Python 3.11–3.12 | ☐ not started | |
| S2 | `prepare_data.py` produces `train/val/test.csv` + `lab_test_mapping.csv` | ☐ | Needs PhysioNet + GPU for summaries |
| S3 | Zero-shot eval runs: `src.evaluate evaluation.base_model=true` | ☐ | Needs CUDA + data |
| S4 | Full train launches: `accelerate ... -m src.train` | ☐ | A40-class GPU |

## Table 1 / main result claims (test set)

Primary row — **LA-CDM** (fill Ours only from logged runs):

| ID | Metric | Paper | Ours | Match? | Log path |
|----|--------|-------|------|--------|----------|
| T1 | Appendicitis Acc | 93.1 | | ☐ | |
| T2 | Cholecystitis Acc | 83.6 | | ☐ | |
| T3 | Diverticulitis Acc | 75.0 | | ☐ | |
| T4 | Pancreatitis Acc | 73.5 | | ☐ | |
| T5 | Mean Acc | 81.3 | | ☐ | |
| T6 | Micro F1 | 84.1 | | ☐ | |
| T7 | Macro F1 | 81.3 | | ☐ | |
| T8 | Avg Test Cost (USD) | 1295.61 | | ☐ | |

Trainless sanity targets:

| ID | Method | Mean Acc (paper) | Ours | Match? |
|----|--------|------------------|------|--------|
| B1 | LA-CDM (ZS) | 64.5 | | ☐ |
| B2 | ReAct | 74.9 | | ☐ |

## Training-dynamics claims (paper §6)

| ID | Claim | Paper | Ours | Match? |
|----|-------|-------|------|--------|
| C1 | Hypothesis Acc improves with training | 75.7 → 81.9 | | ☐ |
| C2 | ECE improves | 0.069 → 0.037 | | ☐ |
| C3 | Cost reward lowers avg cost at similar Acc | $1427.85 → $1295.61; Acc 82.3 → 81.3 | | ☐ |
| C4 | Hypothesis agent helps vs DA-only | Table 3 | | ☐ |

## Implementation-detail claims (Appendix C)

| ID | Claim | Status |
|----|-------|--------|
| I1 | Backbone Qwen-2.5-7B-Instruct + LoRA | ☐ |
| I2 | Cyclic 100 steps: action GRPO → hyp SFT → conf GRPO | ☐ |
| I3 | Adam lr 1e-5, batch 2, seed **269** (code) | ☐ |
| I4 | Costs from BIDMC CMS table (Appendix Table 4) | ☐ (see D-2) |
| I5 | Cost normalized so “all tests” ≈ zero total reward vs correct dx | ☐ |

## Honest gaps (expected)

- Full Table 1 may be blocked by MIMIC access, GPU budget, or unpinned seeds — document rather than invent.
- OASST row is not directly comparable (paper footnote).
- SFT-all is an information upper bound, not an interactive baseline.

## Definition of done (pick one)

- **A (full):** T5–T8 within agreed tolerance under logged seed.
- **B (partial):** S1–S3 + ZS metrics logged; full train deferred; gaps written.
- **C (synthetic-first):** Formal env + toy tuples complete; MIMIC reproduction deferred with written rationale.

**Current recommendation:** Phase E unit tests now; synthetic-first (C) in parallel with whatever of B PhysioNet/CUDA allows.
