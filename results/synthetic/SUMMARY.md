# Synthetic results — env v1 (first pass)

**Source:** synthetic only (no MIMIC, no PHI). Nothing here is a Table 1 reproduction;
paper numbers are calibration targets.
**Produced by:**

```sh
python -m extension.experiments.sanity     --out results/synthetic   # sanity.json
python -m extension.experiments.run_matrix --out results/synthetic   # matrix.json
```

Defaults: seeds 0/1/2, 2,000 random-policy training episodes, 1,000 held-out
eval episodes per arm (same patients for every arm), cost weight λ = 1, diagnose
at p(y|h) ≥ τ = 0.9, 14-action budget. ± is std over 3 seeds.
Utility = 1[correct] − λ·cost_norm (cost_norm: all 12 tests = 1.0, i.e. $10,190).

## 1. Does the env look like the paper's setting?

| Check | Synthetic | Paper / target | Verdict |
|---|---|---|---|
| Class prior (app / chole / div / panc) | .397 / .272 / .103 / .228 | MIMIC-CDM 957/648/257/538 → .399 / .270 / .107 / .224 | ✓ (verify counts against the dataset) |
| All tests, always available: mean acc | 91.3 | SFT-all 92.8 | ✓ close |
| All available tests, missingness: avg cost | $3,770 | SFT-all $3,792.79 | ✓ calibrated |
| All available tests, missingness: mean acc | 82.2 | SFT-all 92.8 | ✗ **gap — see §4.1** |
| Most → least informative test (info gain) | MRI, CT, US, LFT, PE, CMP, BMP, CBC, Renal, Lytes, UA, XR | imaging ≫ panels; radiograph nonspecific | ✓ plausible |
| Cost reward effect (WM, λ 0 → 1, always available) | 90.4 → 89.2 acc, $6,054 → $2,587 | 82.3 → 81.3, $1,428 → $1,296 (C3) | ✓ same direction |

## 2. Main arms (λ = 1)

**Missingness on** (closest to LA-CDM):

| Arm | Mean acc | Micro F1 | Macro F1 | Avg cost | Tests / requests | Utility | Regret vs oracle |
|---|---|---|---|---|---|---|---|
| Oracle WM | 79.8 ± 1.8 | 81.5 | 79.2 | $1,470 | 2.15 / 2.74 | **0.671** ± .009 | 0 |
| Learned WM | 79.2 ± 1.7 | 82.0 | 79.3 | $1,682 | 2.90 / 3.78 | **0.655** ± .013 | 0.016 |
| Model-free | 80.6 ± 2.1 | 82.5 | 80.0 | $1,971 | 2.99 / 4.48 | **0.631** ± .010 | 0.040 |
| Random | 80.2 ± 1.8 | 83.1 | 80.4 | $2,844 | 5.22 / 8.53 | 0.552 ± .010 | 0.119 |
| Bad-WM (α = 1) | 42.8 ± 6.1 | 52.0 | 42.8 | $731 | 3.76 / 5.71 | 0.449 ± .031 | 0.222 |

For scale, LA-CDM Table 1 is 81.3 mean acc at $1,295.61.

**Always available:** Oracle 0.689, WM 0.645 ± .007, MF 0.647 ± .036, Random 0.383, Bad-WM 0.392.
With 2,000 training episodes and every test in the chart, WM and MF tie.

Accuracy alone does not separate the arms (Random buys ~5 tests and gets similar accuracy);
cost does. Read utility and cost together, as the paper does.

## 3. The three findings

**(a) Ordering holds with missingness:** Oracle > learned WM > model-free > Random > Bad-WM on
utility. The WM–MF gap (0.024) is about 2 seed-stds, so it is real but small at 2,000 episodes.

**(b) The world model is more data-efficient.** Utility vs training episodes:

| Episodes | 10 | 30 | 100 | 300 | 1,000 | 3,000 |
|---|---|---|---|---|---|---|
| WM (missingness) | 0.21 | 0.40 | 0.56 | 0.62 | 0.65 | 0.66 |
| MF (missingness) | 0.04 | 0.29 | 0.47 | 0.58 | 0.63 | 0.64 |
| WM (always avail.) | 0.27 | 0.36 | 0.51 | 0.60 | 0.64 | 0.65 |
| MF (always avail.) | −0.02 | 0.16 | 0.42 | 0.50 | 0.59 | 0.66 |

WM wins at every size up to 1,000 episodes; with all tests available MF catches up by 3,000.
At small sizes WM's lead comes from spending less, not from higher accuracy.

**(c) A bad world model makes planning worse than not planning.** Bad-WM utility vs corruption α
(held-out log-lik of the corrupted p_φ in brackets):

| α | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| Missingness | 0.655 [−1.21] | 0.648 [−1.24] | **0.595** [−1.30] | 0.500 [−1.41] | 0.449 [−1.59] |
| Always avail. | 0.645 [−1.17] | 0.661 [−1.22] | **0.577** [−1.32] | 0.440 [−1.48] | 0.392 [−1.78] |

Model-free is 0.631 / 0.647, so the crossover sits between α = 0.25 and 0.5: about 0.1 nats/obs
of world-model error. From α ≥ 0.75 the planner is at or below Random. The corrupted model thinks
the informative tests are worthless, so it stops early and diagnoses at low confidence (accuracy
drops to ~43%, cost drops too). The α = 0.25 bump in the always-available setting is within ~2 stds.

Also: the learned p_φ matches the oracle's held-out log-lik from ~1,000 episodes (−1.17 vs −1.17)
yet keeps a 0.016–0.044 regret. The learned model ignores the atypical-presentation correlation
and is overconfident, so it keeps buying tests the oracle knows cannot push it past τ.

## 4. Gaps and caveats (honest)

1. **No history (HPI) observation.** LA-CDM agents always see p₀ (history / HPI), which carries a
   lot of signal; v1 starts from the prior. That's the likely reason "all available tests" reaches
   only 82.2 vs the paper's 92.8 under missingness. Next fix: an always-available HPI finding.
2. **Model-free ≠ LA-CDM.** MF here is a tabular one-step Q policy, not a GRPO-trained LLM. All
   arms except Oracle share the same learned diagnosis head, so the comparison isolates
   *test selection*.
3. **Idealized perception.** Agents read the structured finding (`info["finding"]`), not the text.
   On MIMIC, an LLM or text classifier has to do this.
4. **Missingness is independent of disease.** In MIMIC, tests are ordered *because* of suspicion
   (informative missingness). Not modeled yet.
5. **One-step lookahead, myopic stopping.** No multi-step planning yet.
6. Every likelihood / availability number is a synthetic design choice (`profiles.py`), not a
   MIMIC statistic.
