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

## Policies in `collect`

- `random`: unused tests until N or random diagnose trigger.
- `expert`: disease-specific preferred tests then correct diagnosis (uses `y_true`; for logging only).

Only **test** steps are written to JSONL (diagnosis actions omitted), matching `docs/env-and-tuples.md` Stage 1 WM filter.
