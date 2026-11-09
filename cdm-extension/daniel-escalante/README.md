# cdm-extension: commit guard and chart-availability reranking

A research extension of the LA-CDM replication in `replication-of-original/daniel-escalante/` (CAI 6734, University of Florida). [SPEC.md](SPEC.md) is the source of truth; [DECISIONS.md](DECISIONS.md) records the choices it leaves open; [prereg/protocol.json](prereg/protocol.json) freezes the analysis.

- **Phase A, commit guard.** Termination recovery and action validation for the trained Nemotron Nano 8B policy under the replication's frozen metric. G1 replaces an invalid action with the highest-scoring valid one, G2 commits to a diagnosis at the step or length budget, and G3 falls back to the hypothesis agent's latest valid answer when an episode ends without one. G3 reuses the hypothesis agent's answer, so Phase A is not a new diagnostic predictor.
- **Phase B, retrospective chart-availability reranking.** Predicts whether a test is present in the chart from the history the agent sees, and reweights test requests. It does not predict result content and is not full world-model planning.

Nothing here is a reproduction of the LA-CDM paper; the comparator is the replication's own run (graph `7fcd9e5`, seed 269, greedy).

## Running it

CPU checks and the synthetic end to end (invented data; no credentials needed):

```sh
cd cdm-extension/daniel-escalante
uv sync --frozen
scripts/tim.sh test      # CPU test suite
scripts/tim.sh all       # every Phase-A stage on the synthetic fixture -> results/synthetic/
scripts/tim.sh check-git # tracked files: schema-valid aggregates only, no canaries, no attribution text
```

`.env` (git-ignored; template in `.env.example`) names the private directories. `CDM_WORK_DIR` is the replication's baseline work directory and is read-only here; runtime artifacts go to `TIM_WORK_DIR` and `TIM_ADMISSION_DIR`, both outside the repository.

Credentialed stages run on HiPerGator, one command per stage:

```sh
scripts/tim.sh approval-request <stage>          # drafts approvals/<stage>.request.json
# Daniel commits approvals/<stage>.json
scripts/submit_slurm.py --stage <stage>           # one RTX PRO 6000 per GPU job, across SLURM_ACCOUNTS
scripts/tim.sh publish <stage>                    # re-validate, canary-scan, copy to results/credentialed/
```

Phase-A order: `validate`, `smoke`, `bench-a`, `val-a`, `analyze-val-a`, addendum A, `test-a`, `analyze-a`. Smoke runs without an approval up to 0.5 GPU-hour in total.

## Data boundary

Every job redirects stdout and stderr to a private log before Python starts. Public output is a schema-validated event stream plus aggregate JSON; `tim publish` re-validates it, refuses any canary marker or loaded admission ID, and only then copies it under `results/`. Rows, histories, generations, logs, inventories and world-model files stay private. All patient data passes through `tim.access`, which enforces SPEC §2's stage × split table; test admissions are one-shot and interrupted runs resume without re-evaluating a completed patient.

## Layout

| Path | Contents |
|---|---|
| `tim/` | the package: `access`, `events`, `publish`, `actions`, `scoring`, `guard`, `fallback`, `runner`, `analysis`, `render`, `approvals`, `protocol`, `fixture`, `cli` |
| `configs/variants.yaml` | variants, limits and stage jobs |
| `schemas/` | the public allowlist for every event and aggregate |
| `prereg/` | the frozen protocol, addenda and amendment notes |
| `approvals/` | approval requests and Daniel's committed approvals |
| `scripts/` | `tim.sh` (capture wrapper), `run_tim.py`, `gpu.slurm`, `cpu.slurm`, `submit_slurm.py` |
| `results/credentialed/` | published credentialed bundles and the Phase reports |
| `results/synthetic/` | the same structure from the synthetic fixture |

## Run log

| Date (ET) | Stage | Jobs | Bundle | Outcome |
|---|---|---|---|---|
| 2026-11-09 | validate | CPU job 50442531 | `results/credentialed/validate/` | T5 recomputed 73.08502492877493 from the official rows; matches the reference: yes; 29 nonanswers |
| 2026-11-09 | smoke | GPU job 50442537 (V3, cai5724) | `results/credentialed/smoke/` | pass: 2 val patients in 186 s including model loading (0.052 of the 0.5 GPU-hour smoke cap) |
| 2026-11-09 | validate (re-run, amendment A1) | CPU job 50449864 | `results/credentialed/validate/` | T5 73.08502492877493, matches the reference: yes; repeated requests now 121 under the corrected definition |
| 2026-11-09 | smoke (re-run, amendment A1) | GPU job 50449871 (V3, cai5724) | `results/credentialed/smoke/` | pass: 2 val patients in 183 s on the split-local data path |
| 2026-11-09 | bench-a | GPU job 50451223 (V0-shadow, cai5724) | `results/credentialed/bench-a/` | 24 patients in 462 s: setup 177 s, 11.8 s per patient; fast vs reference max |diff| 0.3625 (tolerance 0.001: exceeded); next-token agreement 1.00 over 96 turns |
| 2026-11-09 | bench-a | GPU job 50451224 (V3, cai5724) | `results/credentialed/bench-a/` | 24 patients in 286 s: setup 179 s, 4.5 s per patient |
