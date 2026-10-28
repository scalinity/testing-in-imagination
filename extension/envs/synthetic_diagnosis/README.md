# Synthetic CDM environment (Stage 1)

Toy always-available abdominal diagnosis env for world-model tuple collection.
Latent disease \(y\) over `{appendicitis, cholecystitis, diverticulitis, pancreatitis}`;
12 LA-CDM test actions return template observations via \(g_a(y,\epsilon)\).

**No MIMIC / PHI.** Pure Python stdlib (+ pytest for tests).

## Layout

| File | Role |
|------|------|
| `actions.py` | Frozen `A_test` / `A_diag` + USD / normalized costs |
| `generators.py` | `g_a(y, eps)` observation templates |
| `env.py` | `SyntheticCDMEnv` (`reset` / `step` / `oracle_observe`) |
| `tuples.py` | `TupleRecord` + JSONL writer |
| `collect.py` | CLI: random / expert stub → JSONL |

## Quick start

From repo root (`testing-in-imagination/`):

```bash
# Import check
python -c "from extension.envs.synthetic_diagnosis import SyntheticCDMEnv; e=SyntheticCDMEnv(seed=0); print(e.reset()[:80])"

# Collect tuples
python -m extension.envs.synthetic_diagnosis.collect \\
  --n-episodes 20 --seed 0 --out /tmp/tuples.jsonl --policy random

# Tests
python -m pytest extension/tests -q
```

## Env API

```python
from extension.envs.synthetic_diagnosis import SyntheticCDMEnv

env = SyntheticCDMEnv(seed=0, split="train")
h = env.reset()                          # samples y; returns history text
o, reward, done, info = env.step("CT")   # test → observation text
o2 = env.oracle_observe("MRI")           # counterfactual, no advance
o3, r, done, info = env.step("appendicitis")  # diagnosis terminates
```

Rewards: `+1` / `-1` for correct / incorrect diagnosis; test steps get
`-COST_NORM[test]` (normalized costs sum to 1.0 over the inventory).

## v1: graded informativeness + missingness

`SyntheticCDMEnv(version="v1")` (v0 stays the default). Parameters live in `profiles.py`:

- y from the MIMIC-CDM class prior; a presenting phenotype e (atypical with p = 0.07 → e ≠ y).
- Each test returns one finding s ∈ {4 diseases, nonspecific} from `LIKELIHOOD[test][e]`, rendered
  as text with the v0 templates; `info["finding"]` carries s. CT/MRI strong, radiograph/UA weak.
- `missingness=True`: each test is in the chart with `AVAILABILITY[test]` (expected cost of all
  available tests ≈ $3.8k, the paper's SFT-all cost). A missing test returns UNAVAILABLE, is free
  and burns a step (F-2, F-3). `oracle_finding` still returns the counterfactual.
- Episodes end undiagnosed after 14 actions (F-3).

```bash
python -m extension.envs.synthetic_diagnosis.collect \
  --n-episodes 200 --seed 0 --out /tmp/v1.jsonl --env-version v1 --missingness
```

World models, agents and experiments built on v1: `extension/world_model/`,
`extension/planners/`, `extension/experiments/` (results in `results/synthetic/SUMMARY.md`).

## Policies in `collect`

- `random`: unused tests until N or random diagnose trigger.
- `expert`: disease-specific preferred tests then correct diagnosis (uses `y_true`; for logging only).

Only **test** steps are written to JSONL (diagnosis actions omitted), matching `docs/env-and-tuples.md` Stage 1 WM filter.
