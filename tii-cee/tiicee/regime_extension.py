"""D-REGIME extension: readers whose distortion depends on the result, at the first decision point.

RESEARCH_LEDGER.md, "D-REGIME decisions" (November 13, 2026), items 1 to 4, fixed before this ran.
For the twin's R4 readers D-REGIME's ordering is an identity (tiicee/regime_check.py, review F002).
This check asks whether the coordinate still orders readers whose likelihood power depends on the
kind of result, where it is no identity, and whether reading it against R0 cancels the world
model's outcome frequencies for them.

Readers. At the empty history a reader reacts to outcome o of the requested test with a posterior
proportional to prior x P(o | class)^p(o), where p(o) = a on a finding that points to a class
("abnormal") and p(o) = b on the nonspecific finding and on UNAVAILABLE ("normal"); a and b in
{0.5, 1, 2}, nine readers. (1, 1) is R0 there; (2, 2) and (0.5, 0.5) are the over and under
families. R0, the twin's Bayes reader, is the reference.

World models: W0 and W2, judged; W0 with the probability of every class-pointing finding
multiplied by 2, and by 0.5, then renormalized, reported only.

Setting: E1's card configuration, its seeds, the base variant, both acquisition rules; tau_B
two-sided under the exact budget at the first decision point, which every patient of a rule
shares. A seed's value is the mean of log tau_B over the two rules, as in regime_check.

Criteria, judged under W0 and under W2:
  1. Side: every reader whose powers are on the same side of 1 (1 counting as either side) and
     are not both 1 has the seed interval of log tau_B - log tau_B(R0), paired by seed, wholly
     beyond log(1.01) on its side. Above: (2, 2), (2, 1), (1, 2). Below: (0.5, 0.5), (0.5, 1),
     (1, 0.5). log(1.01) is the error bound for a difference of two bisected values at RTOL = 0.01,
     the band of the amended scale call (EXPERIMENT_CARDS/E2.md, A1).
  2. Ordering: the seed mean of log tau_B rises strictly with a at each fixed b, and with b at
     each fixed a.
Reported, not decided on: the mixed readers (2, 0.5) and (0.5, 2); everything under the shifted
world models; absolute and relative to R0 throughout; each reader's two slopes (slopes()).

Outcome: "supported" when both criteria hold under both judged world models. Otherwise "scope
limit": the scale call is stated as validated for distortions that do not depend on the result,
with these findings beside it. Either way D-REGIME's VALIDATED stands.

    python -m tiicee.regime_extension cells    --out results/D-REGIME-extension [--config card|tiny] [--jobs N] [--force]
    python -m tiicee.regime_extension evaluate --out results/D-REGIME-extension [--config card|tiny]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tiicee import cee
from tiicee.experiments.e1 import config, intervals, points
from tiicee.experiments.e1.context import Context
from tiicee.experiments.e1.manifest import code_identity, now
from tiicee.interfaces import UNAVAILABLE
from tiicee.regime_check import SCALE_VARIANT, seed_values
from tiicee.twin import world_models as wm
from tiicee.twin.readers import BayesReader

POWERS = (0.5, 1.0, 2.0)
READERS = tuple((a, b) for a in POWERS for b in POWERS)
JUDGED = ("W0", "W2")
SHIFTS = {"W0-abnormal-x2": 2.0, "W0-abnormal-x0.5": 0.5}
WORLD_MODELS = JUDGED + tuple(SHIFTS)
BAND = float(np.log(1.01))
KEY = 6000                  # bootstrap streams of this check: KEY + 100 x world model + 2 x reader (+ 1 relative)


def reader_name(a: float, b: float) -> str:
    return f"a{a:g}:b{b:g}"


def expected_side(a: float, b: float) -> int:
    """+1 or -1 for a reader criterion 1 judges; 0 for R0's twin (1, 1) and the mixed readers."""
    if (a, b) == (1.0, 1.0):
        return 0
    if a >= 1 and b >= 1:
        return 1
    if a <= 1 and b <= 1:
        return -1
    return 0


def abnormal(model) -> np.ndarray:
    """Outcome mask of the class-pointing findings; the rest must be the nonspecific finding and UNAVAILABLE."""
    mask = np.array([o in model.classes for o in model.outcomes])
    rest = {o for o, m in zip(model.outcomes, mask) if not m}
    if rest != {"nonspecific", UNAVAILABLE}:
        raise ValueError(f"unexpected outcomes outside the classes: {sorted(rest)}")
    return mask


def reactions(model, t: int, powers: np.ndarray) -> np.ndarray:
    """[outcome, class]: prior x P(o | class)^p(o), normalized; an outcome no class allows leaves the prior."""
    q = model.prior[None, :] * model.class_lik()[t].T ** powers[:, None]
    total = q.sum(1, keepdims=True)
    return np.where(total > 0, q / np.where(total > 0, total, 1.0), model.prior[None, :])


def weights(ctx: Context, world: str, test: str) -> np.ndarray:
    """The world model's outcome probabilities at the empty history, over model.outcomes."""
    if world not in WORLD_MODELS:
        raise ValueError(f"unknown world model {world!r}")
    twin = ctx.twin("full")
    m = twin.model
    found = ctx.fitted("full", "W2") if world == "W2" else wm.w0(twin)
    predictive = found.outcomes((), test)
    w = np.zeros(len(m.outcomes))
    for outcome, p in zip(predictive.outcomes, predictive.weights):
        w[m.outcomes.index(outcome)] += p
    if world in SHIFTS:
        w = w * np.where(abnormal(m), SHIFTS[world], 1.0)
        w = w / w.sum()
    return w


def point(prior, w, posteriors, budget) -> dict:
    found = cee.tau_budget(prior, w, posteriors, budget, cee.VARIANT_BOUNDS[SCALE_VARIANT])
    return {"log_tau_b": float(np.log(found.tau)), "path": found.path, "converged": bool(found.converged)}


def slopes(model, t: int, posteriors, w0, mask) -> dict:
    """The temper by result kind (ledger, D-REGIME decisions, item 3): p_abnormal and p_normal.

    log posterior - log prior is fitted by W0-weighted least squares as p(kind of o) x log P(o | class)
    plus a class term shared by every outcome and a term per outcome. Pairs the truth rules out and
    outcomes of W0 probability zero are left out. A slope whose log likelihood is constant over the
    classes on every kept outcome of its kind is not identified (None).
    """
    lik = model.class_lik()[t].T                     # [outcome, class]
    keep = (lik > 0) & (posteriors > 0) & (w0[:, None] > 0)
    o_idx, c_idx = np.nonzero(keep)
    ell = np.log(lik[o_idx, c_idx])
    y = np.log(posteriors[o_idx, c_idx]) - np.log(model.prior[c_idx])
    n_out, n_cls = lik.shape
    kinds = {"p_abnormal": mask[o_idx], "p_normal": ~mask[o_idx]}
    columns = [np.where(kind, ell, 0.0) for kind in kinds.values()]
    columns += [(c_idx == c).astype(float) for c in range(1, n_cls)]
    columns += [(o_idx == o).astype(float) for o in range(n_out) if (o_idx == o).any()]
    x, root = np.stack(columns, 1), np.sqrt(w0[o_idx])
    coef = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
    out = {}
    for i, (name, kind) in enumerate(kinds.items()):
        varies = any(np.ptp(ell[kind & (o_idx == o)]) > 1e-9 for o in np.unique(o_idx[kind]))
        out[name] = float(coef[i]) if varies else None
    return out


def cell(cfg: config.Config, seed: int, variant: str) -> list:
    ctx = Context(cfg, seed, variant)
    twin = ctx.twin("full")
    m, mask, r0 = twin.model, abnormal(twin.model), BayesReader(twin)
    rows = []
    for selector in points.SELECTORS:
        first = points.Trajectories(ctx, selector).steps[0].tests
        if not (first == first[0]).all():
            raise AssertionError("the first decision point should be shared by every patient")
        t = int(first[0])
        test = m.tests[t]
        budget = twin.true_information((), test)
        readers = {"r0": np.array([r0.read((), test, o) for o in m.outcomes])}
        readers.update({reader_name(a, b): reactions(m, t, np.where(mask, a, b)) for a, b in READERS})
        w0 = weights(ctx, "W0", test)
        fitted = {name: slopes(m, t, post, w0, mask) for name, post in readers.items()}
        for world in WORLD_MODELS:
            w = weights(ctx, world, test)
            for name, post in readers.items():
                rows.append({"seed": seed, "variant": variant, "selector": selector, "world_model": world,
                             "reader": name, "test": test, "abnormal_share": float(w[mask].sum()),
                             **point(m.prior, w, post, budget), **fitted[name]})
    return rows


def _interval(values, cfg, key) -> dict:
    found = intervals.seed_interval(values, cfg, key)
    return {"log": [found["lo"], found["hi"]], "estimate": float(np.exp(found["estimate"])),
            "lo": float(np.exp(found["lo"])), "hi": float(np.exp(found["hi"]))}


def world_result(rows: list, cfg: config.Config, w: int, world: str) -> dict:
    here = [r for r in rows if r["world_model"] == world]
    r0 = seed_values(here, world, "r0", cfg.seeds)
    readers, means = {}, {}
    for i, (a, b) in enumerate(READERS):
        name = reader_name(a, b)
        values = seed_values(here, world, name, cfg.seeds)
        means[(a, b)] = values.mean()
        relative = _interval(values - r0, cfg, KEY + 100 * w + 2 * i + 1)
        side = expected_side(a, b)
        holds = None
        if side > 0:
            holds = relative["log"][0] > BAND
        elif side < 0:
            holds = relative["log"][1] < -BAND
        mine = [r for r in here if r["reader"] == name]
        readers[name] = {"a": a, "b": b, "expected_side": side, "tau_b": _interval(values, cfg, KEY + 100 * w + 2 * i),
                         "relative_to_r0": relative, "side_holds": holds,
                         "p_abnormal": _mean(r["p_abnormal"] for r in mine), "p_normal": _mean(r["p_normal"] for r in mine)}
    lines = []
    for vary, keys in (("a", [[(a, b) for a in POWERS] for b in POWERS]), ("b", [[(a, b) for b in POWERS] for a in POWERS])):
        for line in keys:
            logs = [means[key] for key in line]
            lines.append({"vary": vary, "fixed": line[0][1] if vary == "a" else line[0][0],
                          "tau_b": [float(np.exp(v)) for v in logs], "rises": bool((np.diff(logs) > 0).all())})
    return {"judged": world in JUDGED, "r0": _interval(r0, cfg, KEY + 100 * w + 99), "readers": readers,
            "ordering": {"holds": all(line["rises"] for line in lines), "lines": lines},
            "sides_hold": all(entry["side_holds"] for entry in readers.values() if entry["expected_side"]),
            "abnormal_share": sorted({f'{r["selector"]}: {r["test"]} {r["abnormal_share"]:.4f}' for r in here}),
            "search": {"paths": sorted({r["path"] for r in here}), "all_converged": all(r["converged"] for r in here)}}


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def evaluate(rows: list, cfg: config.Config) -> dict:
    rows = [r for r in rows if r["variant"] == cfg.primary_variant and r["seed"] in cfg.seeds]
    worlds = {world: world_result(rows, cfg, w, world) for w, world in enumerate(WORLD_MODELS)}
    supported = all(worlds[w]["ordering"]["holds"] and worlds[w]["sides_hold"] for w in JUDGED)
    return {"decision": "D-REGIME extension", "generated_at": now(), "config": cfg.name, "seeds": list(cfg.seeds),
            "variant": cfg.primary_variant, "statistic": "tau_B, two-sided, exact budget, first decision point",
            "band_log": BAND, "outcome": "supported" if supported else "scope limit", "world_models": worlds,
            "code": {**code_identity(), "regime_extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def _rows_file(out: Path, seed: int, variant: str) -> Path:
    return Path(out) / "rows" / f"seed{seed:02d}_{variant}.json"


def _job(args):
    cfg, seed, variant, out = args
    path = _rows_file(out, seed, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cell(cfg, seed, variant)) + "\n")
    return seed


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tiicee.regime_extension")
    parser.add_argument("command", choices=("cells", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", choices=("card", "tiny"), default="card")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = config.card() if args.config == "card" else config.tiny()
    if args.command == "cells":
        todo = [(cfg, s, cfg.primary_variant, args.out) for s in cfg.seeds
                if args.force or not _rows_file(args.out, s, cfg.primary_variant).exists()]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for seed in pool.map(_job, todo):
                print(f"{now()} seed {seed} done", flush=True)
        return
    rows = [r for path in sorted((args.out / "rows").glob("*.json")) for r in json.loads(path.read_text())]
    found = evaluate(rows, cfg)
    (args.out / "verdict.json").write_text(json.dumps(found, indent=1) + "\n")
    print(json.dumps({"outcome": found["outcome"], **{w: {"ordering": found["world_models"][w]["ordering"]["holds"],
                                                            "sides": found["world_models"][w]["sides_hold"]} for w in JUDGED}}))


if __name__ == "__main__":
    main()
