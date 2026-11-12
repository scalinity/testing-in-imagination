"""Synthetic experiment matrix: model-free vs world-model vs oracle vs bad-WM.

Usage (repo root):
  python -m extension.experiments.run_matrix --out results/synthetic

For each seed and each missingness setting it collects random-policy
training tuples, fits the learned world model and the model-free policy on
the *same* tuples, then evaluates every arm on the same held-out patients.

Arms (``docs/env-and-tuples.md`` §7):
  MF      model-free Q policy, learned belief
  WM      imagination planner with the learned p_phi
  Oracle  imagination planner + belief with the true env model
  Bad-WM  imagination planner with corrupt_model(learned, alpha)
  Random  uniform test choice, learned belief (floor)

Sweeps: Bad-WM alpha; training episodes for MF and WM (data efficiency);
cost weight lam for MF / WM / Oracle (the paper's cost-reward ablation, C3).
Outputs ``matrix.json`` (all numbers + config) to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from extension.envs.synthetic_diagnosis.collect import collect_rows
from extension.envs.synthetic_diagnosis.env import SyntheticCDMEnv
from extension.planners import (
    ImaginationSelector,
    ModelFreeSelector,
    RandomSelector,
    Selector,
    run_episode,
)
from extension.world_model import (
    DiscreteWorldModel,
    corrupt_model,
    fit_count_model,
    oracle_model,
)

from .metrics import aggregate, episode_metrics

EVAL_SEED_OFFSET = 1000


def evaluate(
    selector: Selector,
    belief_model: DiscreteWorldModel,
    env_kwargs: Dict[str, Any],
    seed: int,
    n_eval: int,
    tau: float,
    lam: float,
) -> Dict[str, float]:
    env = SyntheticCDMEnv(seed=EVAL_SEED_OFFSET + seed, split="test", **env_kwargs)
    results = [run_episode(env, selector, belief_model, tau) for _ in range(n_eval)]
    return episode_metrics(results, lam)


def run_setting(
    missingness: bool,
    seeds: Sequence[int],
    n_train: int,
    n_eval: int,
    lam: float,
    tau: float,
    alphas: Sequence[float],
    n_train_sweep: Sequence[int],
    lam_sweep: Sequence[float],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    env_kwargs = {"version": "v1", "missingness": missingness}
    oracle = oracle_model(missingness=missingness)
    # name -> list of per-seed metric dicts
    main: Dict[str, List[Dict[str, float]]] = {}
    bad: Dict[str, List[Dict[str, float]]] = {}
    data_eff: Dict[str, List[Dict[str, float]]] = {}
    lam_curve: Dict[str, List[Dict[str, float]]] = {}
    wm_fit: Dict[str, List[Dict[str, float]]] = {}

    def add(table: Dict[str, List[Dict[str, float]]], key: str, m: Dict[str, float]) -> None:
        table.setdefault(key, []).append(m)

    for seed in seeds:
        t0 = time.time()
        rows = collect_rows(max(n_train, *n_train_sweep), seed=seed, env_kwargs=env_kwargs)
        heldout = collect_rows(500, seed=seed, split="val", env_kwargs=env_kwargs)
        train = _first_episodes(rows, n_train)
        learned = fit_count_model(train)

        def ev(sel: Selector, bm: DiscreteWorldModel, lam_: float = lam) -> Dict[str, float]:
            return evaluate(sel, bm, env_kwargs, seed, n_eval, tau, lam_)

        oracle_m = ev(ImaginationSelector(oracle, lam), oracle)
        add(main, "Oracle", oracle_m)
        add(main, "WM", ev(ImaginationSelector(learned, lam), learned))
        add(main, "MF", ev(ModelFreeSelector.fit(train, learned, lam), learned))
        add(main, "Random", ev(RandomSelector(seed), learned))
        for alpha in alphas:
            add(bad, f"{alpha:g}", ev(ImaginationSelector(corrupt_model(learned, alpha), lam), learned))
            add(
                wm_fit,
                f"bad_alpha={alpha:g}",
                {"heldout_loglik": corrupt_model(learned, alpha).log_likelihood(heldout)},
            )
        add(main, "Bad-WM(alpha=1)", ev(ImaginationSelector(corrupt_model(learned, 1.0), lam), learned))

        for n in n_train_sweep:
            sub = _first_episodes(rows, n)
            wm_n = fit_count_model(sub)
            add(data_eff, f"WM/{n}", ev(ImaginationSelector(wm_n, lam), wm_n))
            add(data_eff, f"MF/{n}", ev(ModelFreeSelector.fit(sub, wm_n, lam), wm_n))
            add(wm_fit, f"learned_n={n}", {"heldout_loglik": wm_n.log_likelihood(heldout)})
        add(wm_fit, "oracle", {"heldout_loglik": oracle.log_likelihood(heldout)})

        for lam_ in lam_sweep:
            add(lam_curve, f"Oracle/{lam_:g}", ev(ImaginationSelector(oracle, lam_), oracle, lam_))
            add(lam_curve, f"WM/{lam_:g}", ev(ImaginationSelector(learned, lam_), learned, lam_))
            add(
                lam_curve,
                f"MF/{lam_:g}",
                ev(ModelFreeSelector.fit(train, learned, lam_), learned, lam_),
            )

        # Regret vs oracle on utility, per seed.
        for table in (main, bad):
            for runs in table.values():
                runs[-1]["regret_vs_oracle"] = oracle_m["utility"] - runs[-1]["utility"]
        log(f"  missingness={missingness} seed={seed} done in {time.time() - t0:.1f}s")

    def agg(table: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
        return {k: aggregate(v) for k, v in table.items()}

    return {
        "main": agg(main),
        "bad_wm_alpha": agg(bad),
        "data_efficiency": agg(data_eff),
        "lambda_sweep": agg(lam_curve),
        "world_model_fit": agg(wm_fit),
    }


def _first_episodes(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Rows of the first n episodes, in collection order."""
    keep: List[str] = []
    seen = set()
    for r in rows:
        if r["episode_id"] not in seen:
            seen.add(r["episode_id"])
            keep.append(r["episode_id"])
    allowed = set(keep[:n])
    return [r for r in rows if r["episode_id"] in allowed]


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=str, default="results/synthetic")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-train", type=int, default=2000, help="training episodes")
    p.add_argument("--n-eval", type=int, default=1000, help="eval episodes per arm")
    p.add_argument("--lam", type=float, default=1.0, help="cost weight")
    p.add_argument("--tau", type=float, default=0.9, help="diagnose confidence")
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    p.add_argument(
        "--n-train-sweep", type=int, nargs="+", default=[10, 30, 100, 300, 1000, 3000]
    )
    p.add_argument("--lam-sweep", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2, 4])
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    report: Dict[str, Any] = {
        "config": {**vars(args), "git_sha": _git_sha(), "env_version": "v1"},
        "settings": {},
    }
    for missingness in (False, True):
        key = "missingness" if missingness else "always_available"
        log(f"setting: {key}")
        report["settings"][key] = run_setting(
            missingness,
            args.seeds,
            args.n_train,
            args.n_eval,
            args.lam,
            args.tau,
            args.alphas,
            args.n_train_sweep,
            args.lam_sweep,
            log,
        )
    path = out / "matrix.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
