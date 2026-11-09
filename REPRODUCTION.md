# REPRODUCTION.md — LA-CDM claims checklist

**Rule:** GitHub clone ≠ reproduced. A claim is reproduced only under **logged config + seed + metric artifact**, or we mark an honest gap.

**Upstream:** https://github.com/dharouni/LA-CDM · commit pin: `3f435a10f569c3255172c1ef8fe7880c6edb7fee` · paper: arXiv:2506.13474 · OpenReview: https://openreview.net/forum?id=7vHUQCMAzG

**Milestone:** credentialed parity run complete (run graph `7fcd9e5`, seed 269, all S0–S6 receipts PASS on 2026-11-08); observed values are filled in below and logged in the replication README.

## Environment log (fill as you go)

| Field | Value |
|-------|-------|
| Date | 2026-11-05 (America/New_York) |
| Operator | Daniel Escalante |
| Machine / GPU | UF HiPerGator. S0 on login node `login7`; run graph at `7fcd9e5` (first run at `b7190c5`: summary shard 49874606 on `c0800a-s5`), NVIDIA RTX PRO 6000 Blackwell Server Edition (compute capability 12.0, 97,887 MiB). Pinned-stack probe: job 49867089 on an L4 (`c0603a-s14`) |
| CUDA / driver | Driver 580.178.04. `CDM_RUNTIME=blackwell`: torch 2.7.0+cu128, vLLM 0.9.2, TRL 0.15.2; arch list through `sm_120`. The pinned torch 2.6.0+cu124 stack stops at `sm_90` |
| Python | 3.12.14 (uv-managed) |
| Install | `uv sync --frozen` (CPU checks); S0 `uv sync --frozen --project runtime-blackwell` (lock sha256 `f12ce556…`, 28 substituted pins recorded in S0) |
| Upstream SHA | `3f435a10f569c3255172c1ef8fe7880c6edb7fee` (lacdm/upstream) |
| `VLLM_USE_V1` | `0` |
| Data source | PhysioNet MIMIC-IV-Ext-CDM v1.1: 12/12 SHA256 digests verified; condition pickles from `scripts/build_condition_pickles.py`, byte-identical on macOS and HiPerGator |
| Split seed | 269 (pinned splitter `RANDOM_STATE`; see D-4) |
| Hydra overrides | Composed by `replication-of-original/daniel-escalante/replication/config.py` and recorded in every stage manifest |
| W&B run ids | None: `wandb.mode=disabled` in every arm |

## Archaeology findings (preserve — do not redo from zero)

| ID | Finding | Status |
|----|---------|--------|
| **D-1** | Paper/code narrative uses `incorrect_reward = -1.0`, but shipped Hydra config sets `0.0`. Wrong diagnosis == undiagnosed reward under default. **At episode-total level** (Diagnosis+Format+Cost unit weights): shipped (`0.0`) scores Correct=2, Wrong(well-formed)=1, Undiagnosed(invalid)=0 — prefers wrong guess over timeout; paper (`−1.0`) makes Wrong and Undiagnosed **tie** at 0 (`lacdm/tests/test_rewards.py`). **Reproduce with shipped config; treat −1.0 vs 0.0 as later ablation — do not silent-fix.** | Confirmed (config + episode totals) |
| **D-2** | Physical examination cost missing from Appendix C Table 4 but present in config (−0.0294 ≈ $300). Twelve costs sum to 1.0 as claimed. | Confirmed; verify with `tools/verify_cost_normalization.py` when added |
| **D-3** | Paper summarizes HPI with Mixtral-8x7B. At pinned upstream `3f435a10`, `prepare_data.py` **defaults to** `mistralai/Mixtral-8x7B-Instruct-v0.1` (Qwen only in a docstring example / `create_data_with_summaries.py` help where `--model` is required). **Narrowed 2026-10-06:** risk is following the docstring, not the real default — pass `--model` explicitly and log it. **Our stack:** same Llama-Nemotron Nano 8B for summarizer + policy (CoC + one-stack simplicity). | Confirmed (Mixtral default at pin); choice logged under D-5 |
| **D-4** | No canonical 80/10/10 split seed. Expect Table 1 mismatch; quantify with ≥3 split seeds. | Confirmed |
| **F-1** | Unavailable tests share the same `trajectory.error` flag as unknown action / bad test name / bad diagnosis — conflated. | Confirmed (code read) |
| **F-2** | Cost uses `provided_tests` only → unavailable requests are **free**; headline $1295.61 is cost of tests that returned data. | Confirmed |
| **F-3** | Unavailable path skips hypothesis update (stale hypothesis one turn) + burns a step. Budget is **14 actions, not 13**: `step()` ends only when `step > max_steps` (`environment.py:311`), steps from 0; `max_steps: 13` vs 12 tests → little pressure to avoid unavailable requests. | Confirmed (code + unit tests) |
| **F-4** | Stale-turn confidence scale mismatch: fresh turns show raw 0–10 integer; on error turns `environment.py:376` reuses `trajectory.confidence[-1]` already ÷10, so the same belief appears as `(confidence: 0.7)` against a 0–10 prompt. Confirmed by `lacdm/tests/test_unavailable_tests.py`. | Confirmed (executed unit) |
| **B-1** | Full stack needs CUDA (vLLM). Local Mac (Apple Silicon) cannot produce Table 1. Local OK for Phase E + synthetic. Prefer HiPerGator; cloud A40 ~$0.49/hr ≈ ~$35 paper-length — **ask before renting**. | Confirmed |
| **D-5** | **Backbone swap (HiPerGator CoC):** upstream LA-CDM defaults to `Qwen/Qwen2.5-7B-Instruct` (PRC-origin → **not permitted** on HiPerGator). **Our default:** `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` (Meta Llama 3.1 8B Instruct **US base** + NVIDIA post-train for reasoning / tool-calling). UFIT RC confirmed NVIDIA models OK when the base is US. This is a documented reproduction deviation — do **not** claim Table 1 bit-for-bit weight match. Stock `meta-llama/Llama-3.1-8B-Instruct` remains the fallback ablation (same family, no Nemotron post-train). | Confirmed (Canvas CoC + RC; 2026-09-28) |
| **D-6** | **Evaluation decoding.** Upstream evaluates with its training temperature: `configure_grpo(eval_only=True)` passes `temperature=getattr(cfg.training, "temperature", 1.0)` and `CDMGRPOTrainer` samples with it in `_run_environment` for evaluation as well as training. This replication pre-specified greedy validation and test decoding (PLAN S2), so its primary results use temperature 0. A post-hoc check (`decode-arm`, `decode-report`) re-evaluates every locked model once at temperature 1.0; greedy stays the primary analysis. | Confirmed (code read, 2026-11-08). Measured 2026-11-08 (jobs 50282940–50282944, report 50282959): temperature 1.0 lowers every arm and does not explain B1 |

**Thesis implication:** baseline env makes unavailability cheap and uninformative; a world model changes the economics so every planned action yields an observation. F-1–F-4 executed at unit level against upstream `3f435a10` via `lacdm/tests` (2026-10-06); still confirm on a smoke episode (rung F2) before write-up.

**Upstream license:** none listed. Submodule only; never vendor.


## Backbone (our stack) — D-5

| Role | Upstream (paper/repo) | **Ours (HiPerGator)** | Notes |
|------|----------------------|------------------------|-------|
| Main agent / GRPO policy | `Qwen/Qwen2.5-7B-Instruct` | **`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`** | US base (Meta Llama 3.1 8B) + NVIDIA post-train. HF: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1 · release 2025-03-18 · NVIDIA Open Model License + Llama 3.1 Community |
| HPI summarizer | Mixtral-8x7B (paper) / Qwen (repo) | **Same Nemotron Nano 8B** | One US stack; log D-3. Mixtral OK under CoC (non-China) but not US — optional later |
| World model \(p_\phi\) | n/a (model-free paper) | Tiny bag-of-words / small classifier first | Do not default \(p_\phi\) to 8B/70B |
| Fallback ablation | — | `meta-llama/Llama-3.1-8B-Instruct` | Same architecture family without Nemotron post-train |
| Banned on HiPerGator | — | Qwen, DeepSeek, Yi, InternLM, other CoC-country bases + derivatives | LoRA / “re-release” does not fix a banned base |

### Hydra / job checklist (before any GRPO or vLLM run)

1. Override every Hydra/`prepare_data` default that still points at Qwen.
2. Pin exact HF revision (or HiPerGator local path) in configs; prefer RC mirrors when present.
3. For Nemotron Nano agent loops: start with **reasoning off** (`detailed thinking off` system prompt) for tool-style diagnose/order-test actions; treat reasoning-on as a later ablation.
4. Confirm checkpoint path with UFIT RC if leaving mirrored Llama paths.
5. Log this file’s D-5 row + HF revision in every W&B / results run.

## Setup claims

| ID | Claim | Status | Evidence / gap |
|----|-------|--------|----------------|
| S1 | Repo installs with `uv sync` on Python 3.11–3.12 | ☑ | Python 3.12.14; `uv sync --frozen` and the Blackwell runtime lock (S0 receipts) |
| S2 | `prepare_data.py` produces `train/val/test.csv` + `lab_test_mapping.csv` | ☑ | Run graph `7fcd9e5`: 2,400 summaries (shards 50010230–50010237), split 1,920/240/240 (S1 50010238, S2 50010239) |
| S3 | Zero-shot eval runs: `src.evaluate evaluation.base_model=true` | ☑ | S3 50010240 on an RTX PRO 6000, receipt `f323a460` |
| S4 | Full train launches: `accelerate ... -m src.train` | ☑ | Four arms × 3,840 steps, jobs 50010241–50010244, all COMPLETED 0:0; S4 receipt `05601581` |

## Table 1 / main result claims (test set)

Primary row — **LA-CDM** (fill Ours only from logged runs):

**Status** compares the paper's published value with our 95% interval. It is not a reproduction verdict: no tolerance was agreed, the backbone and summarizer differ (D-3, D-5), the runtime differs (28 pins), decoding differs (D-6) and there is one training seed.
*Within CI*: the paper value lies inside our interval. *Below CI* / *Above CI*: our whole interval lies below or above the paper value.

| ID | Metric | Paper | Ours (greedy, primary analysis) | Status | Reason |
|----|--------|-------|------|--------|----------|
| T1 | Appendicitis recall | 93.1 | 90.62 [84.38, 95.83] | within CI | 87/96 test patients correct |
| T2 | Cholecystitis recall | 83.6 | 72.31 [61.54, 83.08] | below CI | 10/65 called pancreatitis, 6/65 undiagnosed |
| T3 | Diverticulitis recall | 75.0 | 72.00 [56.00, 88.00] | within CI | only 25 test patients; wide interval |
| T4 | Pancreatitis recall | 73.5 | 57.41 [44.44, 70.37] | below CI | 14/54 undiagnosed, 7/54 called cholecystitis |
| T5 | Mean class recall | 81.3 | 73.09 [66.65, 79.19] | below CI | driven by T2 and T4 |
| T6 | Micro F1 | 84.1 | 81.15 [76.47, 85.53] | within CI | |
| T7 | Macro F1 | 81.3 | 78.20 [72.32, 83.44] | within CI | |
| T8 | Avg provided-test cost (USD) | 1295.61 | 1439.58 [1341.50, 1537.65] | above CI (costlier) | the cost reward did not lower cost here (C3) |

Ours: run graph `7fcd9e5`, parity split, seed 269, 240 test admissions; 95% class-stratified patient bootstrap (10,000 resamples). Counts in *Reason* come from the confusion matrices in `results/mimic/REPORT.md` (supplementary section). The aggregate export (`results/mimic/metrics.json`, `REPORT.md`, `supplementary.json`) is tracked through a narrow `.gitignore` exception; all other `mimic*` paths stay ignored.

Trainless sanity targets:

| ID | Method | Mean recall (paper) | Ours | Status | Reason |
|----|--------|------------------|------|--------|--------|
| B1 | LA-CDM (ZS) | 64.5 | 12.71 [8.04, 17.86] | below CI | 194/240 greedy zero-shot episodes ended without a valid diagnosis (613 repeated test requests); 27 of the 46 diagnoses it did give were correct. No code bug against `3f435a10`; upstream's temperature-1.0 decoding scores lower still (9.19 [5.55, 13.44]), so decoding does not explain the gap. The backbone and summarizer (D-5, D-3) remain the untested candidates |
| B2 | ReAct | 74.9 | — | gap | no ReAct implementation at the upstream pin |

### Decoding check (D-6, post-hoc)

Every locked model ran once more on the same 240 test patients at upstream's evaluation temperature, 1.0, each request with its own seed. Greedy stays the primary analysis; this table measures how much the decoding choice moves each result. Source: `results/mimic/REPORT.md` and `supplementary.json` (supplementary section).

| Arm | T5 greedy | T5 at 1.0 [95% CI] | Paired difference [95% CI] | Undiagnosed, greedy → 1.0 |
|-----|-----------|--------------------|----------------------------|---------------------------|
| zero_shot | 12.71 | 9.19 [5.55, 13.44] | −3.52 [−9.38, 2.37] | 194 → 202 |
| primary | 73.09 | 59.30 [52.23, 66.19] | −13.79 [−21.70, −6.24] | 29 → 55 |
| decision_only | 57.49 | 40.64 [33.86, 47.27] | −16.85 [−25.05, −8.68] | 19 → 50 |
| no_cost | 70.81 | 61.46 [54.52, 68.28] | −9.35 [−17.60, −1.02] | 31 → 50 |
| incorrect_minus1 | 59.38 | 55.75 [48.63, 62.62] | −3.63 [−12.94, 5.66] | 65 → 78 |

At 1.0 the primary arm's whole interval lies below the paper on each of T1–T7, and its provided-test cost falls to $961.71 [875.56, 1,046.85] with fewer valid test requests (661 → 400). Hypothesis accuracy is 28.49% → 75.41% (zero-shot → primary; greedy 29.59% → 81.67%). Comparisons at 1.0 (Holm family of four): C4 primary − decision_only T5 +18.66 [9.94, 27.21], Holm p 0.000049, holds; C3 primary − no_cost cost +$82.94 [−30.33, 195.84], Holm p 0.538, still not observed; primary − incorrect_minus1 T5 +3.55 [−5.68, 12.76], Holm p 0.474, so the D-1 ablation difference seen under greedy decoding does not hold under sampling.

## Training-dynamics claims (paper §6)

| ID | Claim | Paper | Ours | Status | Reason |
|----|-------|-------|------|--------|--------|
| C1 | Hypothesis accuracy improves with training | 75.7 → 81.9 | 29.59 → 81.67 [77.24, 85.89] (zero-shot → primary, held-out endpoints) | endpoint within CI; start below | improvement +52.08 points [46.40, 57.73]; our zero-shot start is far lower |
| C2 | ECE improves | 0.069 → 0.037 | 0.3022 → 0.1281 [0.0861, 0.1749] | improves; endpoint above paper (worse calibration) | difference −0.174 [−0.232, −0.114] |
| C3 | Cost reward lowers avg cost at similar accuracy | $1427.85 → $1295.61; recall 82.3 → 81.3 | no_cost → primary: $1321.33 → $1439.58; recall 70.81 → 73.09 | not observed | paired cost difference +118.25 [−0.25, 239.86]; McNemar Holm p 0.551. The cost reward is in the primary training reward (logged on all 3,840 steps) but small: −0.09 to −0.11 per rollout against 0.17–0.53 for the diagnosis reward |
| C4 | Hypothesis agent helps vs decision-agent only | Table 3 | decision_only → primary mean recall 57.49 → 73.09 | direction observed; paper value unassessable | paired +15.59 [6.34, 24.60], McNemar Holm p 0.00056; exact Table 3 values absent from this ledger |

## Implementation-detail claims (Appendix C)

| ID | Claim | Status |
|----|-------|--------|
| I1 | Upstream: Qwen-2.5-7B-Instruct + LoRA. **Ours:** `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` + LoRA (D-5) | ☑ LoRA training completed on the Blackwell runtime (run graph `7fcd9e5`) |
| I2 | Cyclic 100 steps: action GRPO → hyp SFT → conf GRPO | ☑ 100/100/100 optimizer-update phases |
| I3 | Adam lr 1e-5, batch 2, seed **269** (code) | ☑ Adam, lr 1e-5, batch 2, seed 269 |
| I4 | Costs from BIDMC CMS table (Appendix Table 4) | ☑ configured upstream USD schedule (see D-2) |
| I5 | Cost normalized so “all tests” ≈ zero total reward vs correct dx | ☑ twelve normalized costs sum to one (tested) |

## Honest gaps (expected)

- Full Table 1 may be blocked by MIMIC access, GPU budget, or unpinned seeds — document rather than invent.
- OASST row is not directly comparable (paper footnote).
- SFT-all is an information upper bound, not an interactive baseline.
- **B1 gap unexplained.** No code bug against `3f435a10` and decoding ruled out (D-6 check); the backbone and summarizer (D-5, D-3) are untested because a new backbone run was out of scope.
- **One training seed (269).** Between-seed uncertainty is unassessable; every interval covers test-patient sampling only.
- **No `subject_id` in v1.1.** Subject leakage and the subject-grouped split are unassessable, never reported as zero.
- **Radiology count** 5,960 in v1.1 vs 5,959 stated upstream; unexplained.
- **C4 paper value** (Table 3) is not in this ledger, so only the direction is assessed.

## Definition of done (pick one)

- **A (full):** T5–T8 within agreed tolerance under logged seed.
- **B (partial):** S1–S3 + ZS metrics logged; full train deferred; gaps written.
- **C (synthetic-first):** Formal env + toy tuples complete; MIMIC reproduction deferred with written rationale.

**Status (2026-11-08):** B is met and exceeded: S1–S4 ran on credentialed v1.1 and T1–T8, B1 and C1–C4 are logged from one full training run (graph `7fcd9e5`). A is unassessable: no tolerance was agreed, and T2, T4 and T5 sit below and T8 above our intervals. The results are a replication attempt on a different backbone, summarizer and runtime, not a reproduction.
