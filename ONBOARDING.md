# ONBOARDING — Publication constitution

**Full standing prompt:** [`onboarding/COMPREHENSIVE_ONBOARDING_PROMPT.md`](onboarding/COMPREHENSIVE_ONBOARDING_PROMPT.md) (§§0–14). Merge into existing artifacts; do not wipe.

## Mission

Extend LA-CDM with a world model \(p_\phi(o_{t+1}\mid h_t,a_t)\) for testing in imagination; compare model-free vs learned vs oracle, including when a bad world model hurts.

## Behavior

1. Artifact-first (`REPRODUCTION.md`, specs, matrices, drafts).
2. Never invent numbers or claim “reproduced” without logged run metadata.
3. Honest gaps are publishable findings.
4. Ask before: large GPU spend, PHI-touching cloud, changing the research question, emailing Liu/Hugo/Edwin, public data push.
5. If blocked on CUDA/PhysioNet → Phase E unit tests + synthetic design locally.
6. Scope: this publication only.

## Phase order (mandatory)

A archaeology → B env freeze (CUDA) → C data (PhysioNet) → D code comprehension → **E unit tests (do next, Mac OK)** → F baseline ladder (CUDA) → G extension gate.

## Preserve these findings in `REPRODUCTION.md`

D-1 (incorrect_reward config ≠ paper), D-2 (physical-exam cost), D-3 (Mixtral vs Qwen summarizer; **ours = Nemotron for both**), D-4 (no split seed), **D-5 (backbone: Qwen → `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`, HiPerGator CoC)**, F-1/F-2/F-3 (unavailable-test economics), B-1 (Mac has no CUDA for Table 1).

## First coding milestone

Phase E: unit tests for parsers, rewards, unavailable branch, stub metrics — then synthetic env under `extension/envs/`.
