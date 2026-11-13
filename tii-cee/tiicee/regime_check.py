"""D-REGIME: the regime map's scale coordinate, validated in the twin before E2, E4 and E7 use it.

RESEARCH_LEDGER.md, D-REGIME (November 13, 2026). The coordinate is tau_B, the two-sided diagnostic
(bounds 1/8 to 16) under the exact budget, at the first decision point of each trajectory: the empty
history and the test the acquisition rule requests there. Location incoherence at tau_B is taken at
the same point and reported; the criteria concern the scale coordinate only.

Setting: E1's full twin, evaluation patients and acquisition rules (random order; greedy information
under W0), world models W0 and W2, the card's seeds, decided on the primary variant. A seed's value
is the mean of log tau_B over the patients' first decision points and the two acquisition rules; the
seed values are averaged, the scale geometrically, as Figure 1 averages them.

Criteria, fixed before the run (the decision's wording, and how this module reads it):
  1. Monotone ordering: under each world model, for each drift kind and size of the 81-reader grid,
     the seed mean of log tau_B rises strictly with the likelihood power over the nine powers.
  2. Correct side of 1: under each world model, each R4 family whose likelihood power is not 1
     (over, under, entrench+over, fixed+under) has its 95% seed interval of log tau_B wholly on the
     side of 0 its power is on. A family of power 1 (independence, entrench, fixed) has no scale
     error at the empty history, so it has no side: it passes when its interval, widened by the
     search's resolution log(1 + RTOL), contains 0, so that the regime call of specification 10.2
     would not place it in the scale regime.
  Validated only when both hold under W0 and under W2.

Under W0 the check is close to an identity. At the empty history a reader whose reactions are the
true likelihood raised to a power p, times a class factor that does not depend on the result (both
drift kinds there), becomes the true posterior when tempered at tau = p and projected; that pair
carries exactly the budget, so tau_B = p to the search's resolution. W2, whose predictive is fitted,
is the part of the check that is not an identity.

What the first decision point cannot see: an error that needs a history. The independence reader's
double counting of correlated evidence is one; at the empty history that reader is the Bayes reader.

Every row reports the shares of searches clipped, on the fallback path, and with an unconverged
projection (review F004: a result that rests on tau_B reports search_converged beside it).

    python -m tiicee.regime_check cells    --out results/D-REGIME [--config card|tiny] [--jobs N] [--force]
    python -m tiicee.regime_check evaluate --out results/D-REGIME [--config card|tiny]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tiicee.cee.scaling import RTOL
from tiicee.experiments.e1 import batch, config, figure1, filters, intervals, points
from tiicee.experiments.e1.context import Context
from tiicee.experiments.e1.manifest import code_identity, now

WORLD_MODELS = ("W0", "W2")
SCALE_VARIANT = "two-sided"
POWER = {"independence": 1.0, "entrench": 1.0, "fixed": 1.0, "over": 2.0, "under": 0.5,
         "entrench+over": 2.0, "fixed+under": 0.5}
RESOLUTION = float(np.log1p(RTOL))
KEY = 5000                  # bootstrap streams of this check: KEY + 100 x world model + reader


def grid_name(power: float, delta: float, drift: str) -> str:
    return filters.grid_name(power, delta, drift if delta > 0 else "none")


def reader_names() -> list:
    """R0, the R4 families, and the 81 distinct readers of Figure 1's grid."""
    grid = [grid_name(p, d, k) for p, d, k in figure1.grid_readers()]
    return ["r0", *config.R4, *dict.fromkeys(grid)]


def first_point(traj: points.Trajectories, reader_name: str, world_name: str) -> dict:
    """tau_B (two-sided, exact budget) and KL_inc at tau_B at each patient's first decision point, averaged."""
    pr = traj.pairs(0, reader_name)
    j = pr.taken
    weights = points.world_predictive(traj, 0, world_name, pr)[j]
    search, at_tau = batch.scale(pr.bf, weights, pr.q[j], batch.information(pr.joint[j]), SCALE_VARIANT)
    log_tau, ok = np.log(search.tau), at_tau.converged
    return {"log_tau_b": float(log_tau.mean()), "log_tau_b_range": float(np.ptp(log_tau)),
            "kl_inc_at_tau_b": float(at_tau.kl_inc[ok].mean()) if ok.any() else None,
            "clipped": float(np.isin(search.path, (1, 2)).mean()), "fallback": float((search.path == 3).mean()),
            "search_converged": float(search.converged.mean()), "n_points": int(j.size)}


def cell(cfg: config.Config, seed: int, variant: str) -> list:
    ctx = Context(cfg, seed, variant)
    rows = []
    for selector in points.SELECTORS:
        traj = points.Trajectories(ctx, selector)
        first = traj.steps[0].tests
        if not (first == first[0]).all():
            raise AssertionError("the first decision point should be shared by every patient")
        for world in WORLD_MODELS:
            for name in reader_names():
                rows.append({"seed": seed, "variant": variant, "selector": selector, "world_model": world, "reader": name,
                             "test": traj.model.tests[int(first[0])], **first_point(traj, name, world)})
    return rows


def seed_values(rows: list, world: str, reader: str, seeds) -> np.ndarray:
    """One value per seed: log tau_B averaged over the acquisition rules."""
    out = []
    for seed in seeds:
        here = [r["log_tau_b"] for r in rows if (r["seed"], r["world_model"], r["reader"]) == (seed, world, reader)]
        if len(here) != len(points.SELECTORS):
            raise ValueError(f"seed {seed}, {world}, {reader}: {len(here)} rows, expected {len(points.SELECTORS)}")
        out.append(np.mean(here))
    return np.array(out)


def ordering(rows: list, world: str, seeds) -> dict:
    """Criterion 1: the seed mean of log tau_B rises strictly with the power at every drift kind and size."""
    lines = []
    for drift, delta in dict.fromkeys((k if d > 0 else "none", d) for k in figure1.DRIFTS for d in figure1.DELTAS):
        values = np.stack([seed_values(rows, world, grid_name(p, delta, drift), seeds) for p in figure1.POWERS])
        means = values.mean(1)
        lines.append({"drift": drift, "delta": delta, "powers": list(figure1.POWERS),
                      "tau_b": [float(v) for v in np.exp(means)], "rises": bool((np.diff(means) > 0).all()),
                      "seeds_rising": float((np.diff(values, axis=0) > 0).all(0).mean())})
    return {"holds": all(line["rises"] for line in lines), "lines": lines}


def side(estimate: dict, power: float) -> bool:
    """Criterion 2 for one family, on the interval of log tau_B."""
    if power > 1:
        return estimate["lo"] > 0
    if power < 1:
        return estimate["hi"] < 0
    return estimate["lo"] - RESOLUTION <= 0 <= estimate["hi"] + RESOLUTION


def families(rows: list, world: str, cfg: config.Config, w: int) -> dict:
    out = {}
    for i, name in enumerate(["r0", *config.R4]):
        found = intervals.seed_interval(seed_values(rows, world, name, cfg.seeds), cfg, KEY + 100 * w + i)
        kl_inc = [r["kl_inc_at_tau_b"] for r in rows if (r["world_model"], r["reader"]) == (world, name) and r["kl_inc_at_tau_b"] is not None]
        entry = {"power": POWER.get(name, 1.0), "tau_b": float(np.exp(found["estimate"])), "tau_b_lo": float(np.exp(found["lo"])),
                 "tau_b_hi": float(np.exp(found["hi"])), "kl_inc_at_tau_b": float(np.mean(kl_inc)) if kl_inc else None,
                 "holds": side(found, POWER.get(name, 1.0))}
        if name == "r0":
            entry["note"] = "reported; not an R4 family, so not part of the decision"
        out[name] = entry
    return out


def evaluate_variant(rows: list, cfg: config.Config, variant: str) -> dict:
    rows = [r for r in rows if r["variant"] == variant]
    out = {}
    for w, world in enumerate(WORLD_MODELS):
        here = [r for r in rows if r["world_model"] == world]
        fams = families(here, world, cfg, w)
        order = ordering(here, world, cfg.seeds)
        out[world] = {
            "ordering": order, "families": fams,
            "sides_hold": all(entry["holds"] for name, entry in fams.items() if name != "r0"),
            "search": {"clipped_max": max(r["clipped"] for r in here), "fallback_max": max(r["fallback"] for r in here),
                       "search_converged_min": min(r["search_converged"] for r in here),
                       "log_tau_b_range_max": max(r["log_tau_b_range"] for r in here)},
            "first_tests": sorted({f'{r["selector"]}: {r["test"]}' for r in here}),
        }
    out["validated"] = all(out[w]["ordering"]["holds"] and out[w]["sides_hold"] for w in WORLD_MODELS)
    return out


def evaluate(rows: list, cfg: config.Config) -> dict:
    primary = evaluate_variant(rows, cfg, cfg.primary_variant)
    others = {v: evaluate_variant(rows, cfg, v) for v in cfg.variants if v != cfg.primary_variant and any(r["variant"] == v for r in rows)}
    return {"decision": "D-REGIME", "generated_at": now(), "config": cfg.name, "seeds": list(cfg.seeds),
            "variant": cfg.primary_variant, "statistic": "tau_B, two-sided, exact budget, first decision point",
            "resolution_log": RESOLUTION, "validated": primary["validated"], "primary": primary,
            "other_variants": {"note": "Reported, not decided on.", "results": others},
            "code": {**code_identity(), "regime_check_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def _rows_file(out: Path, seed: int, variant: str) -> Path:
    return Path(out) / "rows" / f"seed{seed:02d}_{variant}.json"


def _job(args):
    cfg, seed, variant, out = args
    path = _rows_file(out, seed, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cell(cfg, seed, variant)) + "\n")
    return seed, variant


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tiicee.regime_check")
    parser.add_argument("command", choices=("cells", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", choices=("card", "tiny"), default="card")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = config.card() if args.config == "card" else config.tiny()
    if args.command == "cells":
        variants = args.variants or [cfg.primary_variant]
        todo = [(cfg, s, v, args.out) for v in variants for s in cfg.seeds if args.force or not _rows_file(args.out, s, v).exists()]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for seed, variant in pool.map(_job, todo):
                print(f"{now()} seed {seed} {variant} done", flush=True)
        return
    rows = [r for path in sorted((args.out / "rows").glob("*.json")) for r in json.loads(path.read_text())]
    found = evaluate(rows, cfg)
    (args.out / "verdict.json").write_text(json.dumps(found, indent=1) + "\n")
    print(json.dumps({"validated": found["validated"], **{w: {"ordering": found["primary"][w]["ordering"]["holds"],
                                                                "sides": found["primary"][w]["sides_hold"]} for w in WORLD_MODELS}}))


if __name__ == "__main__":
    main()
