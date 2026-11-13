# E1: the exact synthetic study

Runners, decision rules and Figure 1 inputs for `reports/research-advancement/EXPERIMENT_CARDS/E1.md`,
which is frozen and authoritative. This file says how the code reads the card where the card leaves a
choice open. Nothing here changes a rule or a threshold of the card.

The twin is Edwin Salcedo's synthetic study (PR #1, commit df467ea), reached through
`tiicee.twin` and its vendored snapshot.

## Run order

    python -m tiicee.experiments.e1.run cells       --out results/E1 --scratch DIR --jobs N
    python -m tiicee.experiments.e1.run c-intervals --out results/E1 --scratch DIR
    python -m tiicee.experiments.e1.run rules       --out results/E1
    python -m tiicee.experiments.e1.run part-a      --out results/E1 --jobs N      # needs the selected layer
    python -m tiicee.experiments.e1.run harm        --out results/E1 --jobs N
    python -m tiicee.experiments.e1.run rules       --out results/E1               # now with part A
    python -m tiicee.experiments.e1.run figure      --out results/E1 --figure results/figure1
    python -m tiicee.experiments.e1.run manifest    --out results/E1 --figure results/figure1

`--config tiny` is for tests and timing and never produces a result. A finished cell leaves a marker
under `tables/_done` and is skipped on a later call unless `--force` is given. Part C's per-history
rows go to the scratch directory, not to the results: the results hold aggregates only.

## Where each quantity is computed

| Quantity | Module | Exactness |
|---|---|---|
| Log loss, accuracy, episode utility at horizons 0 to 6; the terms of 5.12 (iii) | `reduced.py` on the dense tree of `tree.py` | Exact over the latent case and every outcome path |
| Drift, responsiveness, KL_inc, tau_B, tau_ratio, probes, identity terms, one-step regrets, gate | `points.py` | Exact at each decision point, along the evaluation patients' trajectories |
| Episode log loss at horizon 8 on the full twin (`mc_*` columns, table `B_sandbox`) | `points.py` | Monte Carlo over patients, labeled so |
| Values V of part A and the Bayes-optimal references | `part_a.py`, `tiicee.twin.env` | Exact |
| Held-out budget and its eligibility test | `heldout.py` | Estimated from simulated training patients, as on charts |

The CEE library handles one decision point at a time, and E1 has tens of millions. `batch.py` and
`filters.py` are the library's and the twin's routines on arrays; `tests/test_e1_batch.py` and
`tests/test_e1_filters.py` hold each one to the library function or twin object it mirrors, and
`tests/test_e1_reduced.py`, `test_e1_points.py` and `test_e1_part_a.py` hold the runners to a
point-by-point recursion through the library's public functions.

## Tables (`results/E1/tables/<name>/seedNN_<variant>.csv.gz`)

Gzipped CSV, written with a fixed header so that the same rows give the same bytes. `D_horizon` is
large and goes to `results/E1/large/`, which git ignores; the manifest records the SHA-256 of every
table file, committed or not. `D_final` holds its final-horizon rows, which is all that the rules and
the figure step read of part D, so the committed tables are enough to re-evaluate every verdict.
`verdicts.json` also holds the same rules evaluated on each other variant (`other_variants`).

| Table | One row per |
|---|---|
| `B_reduced_horizon` | acquisition rule, reader, world model, arm (raw, stateless, stateful), horizon |
| `B_reduced_commit` | the same, per commit rule (threshold 0.7, 0.8, 0.9; CEE-Commit at each lambda) |
| `B_reduced_fallback` | acquisition rule, reader, world model: share of path mass whose projection did not converge |
| `B_points` | acquisition rule, reader, world model: means over the full twin's decision points |
| `B_sandbox` | seed and sandbox cell: Monte Carlo episode log loss in the sandbox's configuration |
| `C_summary` | acquisition rule, reader, controlled error; and one row across errors with the rank correlations |
| `D_horizon` (large), `D_final`, `D_commit`, `D_tau` | acquisition rule, reader, world model (W0, W2), track, scale variant, budget source |
| `D_heldout` | test and stratum: the held-out budget, its bound, the eligibility test, the stress cases |
| `A_values` | lambda, reader, agent (raw, repaired): V of the eight component combinations and V* |
| `F_harmful_grid` | grid reader: Figure 1 coordinates and exact harm of projection alone |

## Readings of the card

Scope
1. W2-cc is W2's fitted model read class-conditionally (ledger D-TWIN), so its predictive is W2's. Part B
   computes W0, W1 and W2; the W2 rows are also W2-cc's. W2-cc's own contribution is its budget, part D's `w2cc`.
2. Evaluation patients come from the validation split, because E1 selects the track and the scale variant.
3. Parts B, C and D run on the four method-choice variants. Part A and the harmful-region grid run on
   the base variant. Every rule is decided on the base variant; the others are reported.
4. Part D uses the two world models its rules name, W0 and W2, and seven budgets per scale variant and
   track: the exact budget; the stress set applied to it (half, double, lognormal noise of sigma 0.7,
   one independent factor per decision point); the B-model of W1; the B-model of W2-cc; B-held-out.
5. A seed's value in a rule is the exact expected log loss at the final horizon of the reduced twin,
   averaged over the two acquisition rules. S_rand is one random order per seed; S_ig requests the
   unrequested test with the largest exact information under W0, ties to the lowest index.
6. The gate's bounds come from 1,000 further patients of the fitting stream per seed (cases 5,000 to
   5,999), which no fit and no evaluation uses, with each term estimated from factual results as on charts.
7. The held-out budget: reference reader with L2 penalty 1.0; folds 0 to 3 estimate, fold 4 is kept
   apart; strata count results that returned a finding; the bound is floored at zero; a decision point
   whose cell is not eligible gets CEE-Read without CEE-Scale.
8. The matched pair sits at 0.02 nats from the truth.
9. The sandbox ran with every test available and a fresh order per patient. Its cross-check is run in
   that configuration (table `B_sandbox`); the card's own full-twin setting is reported in `B_points`.

Rules (each is a method of `rules.Evaluator`, with the card's sentence in its docstring)
10. Repair present and the negative controls are read on the track the track rule selects.
11. No-harm: the largest absolute change over both tracks, every horizon, both acquisition rules, every seed.
12. Negative controls: under W0, the mean over seeds of corrected minus raw is positive for both scale families.
13. Gate valid: for every R4 family and every world model, the exact expected gain averaged over the
    decision points the gate allows is positive. A gate that never opens for a cell decides nothing.
14. Part C: the regret of CEE-Select uses CEE-Read on the stateless track at the decision point.
    |E_W| for regret is the decision form of specification 5.5 (the committed utility in place of log Z).
    A history's KL, calibration error and |E_W| are means over its unrequested tests. Rows are pooled
    over readers, acquisition rules and controlled errors; patients are resampled within seeds.
    Calibration error at a point is the gap between the predicted and the true probability of the
    model's most likely outcome.
15. Variant rule: per family the worst case is the largest log loss over the four stress budgets; then
    the families are averaged; under W2, on the selected track.
16. Scale repair present is decided on the two-sided variant, the only one that can amplify an
    under-reactive reader; shrink-only is evaluated and reported.
17. Usable outside the twin: the scale-repair condition for the selected variant under W2 with the
    held-out budget, and an admission rate of at most 10%. A stress case is a (seed, test, stratum)
    cell times a member of the stress set applied to its held-out budget; it counts when the stressed
    budget exceeds the cell's exact mean information and still passes the eligibility test.
18. Attribution: values are averaged over seeds; a component's spread is its largest minus its smallest
    share over the six orders; an agent is unstable if a spread exceeds the share in the declared order.
19. The repaired agent of part A uses the selected track, with CEE-Scale (selected variant, held-out
    budget) only if rule 17 holds.

## Tests

    cd tii-cee && uv run pytest -q -W error tests/test_e1_*.py
