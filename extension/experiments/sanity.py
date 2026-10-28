"""Does env v1 behave like the paper's setting? Synthetic-vs-paper checks.

Usage (repo root):
  python -m extension.experiments.sanity --out results/synthetic

Paper numbers are LA-CDM Table 1 (arXiv:2506.13474). Matching them is a
*calibration* target for a synthetic env, not a reproduction claim.
Writes ``sanity.json`` to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from extension.envs.synthetic_diagnosis.actions import A_DIAG, A_TEST, COST_USD
from extension.envs.synthetic_diagnosis.env import SyntheticCDMEnv
from extension.envs.synthetic_diagnosis.profiles import (
    AVAILABILITY,
    P_ATYPICAL,
    PRIOR_MIMIC_CDM,
    expected_all_available_cost_usd,
)
from extension.planners.agents import EpisodeResult, argmax
from extension.world_model import oracle_model

from .metrics import episode_metrics

PAPER = {
    "sft_all_mean_acc": 92.8,
    "sft_all_avg_cost_usd": 3792.79,
    "la_cdm_mean_acc": 81.3,
    "la_cdm_avg_cost_usd": 1295.61,
}


def all_tests_episode(env: SyntheticCDMEnv, tests: List[str]) -> EpisodeResult:
    """Order ``tests`` (skipping nothing), then diagnose by oracle posterior."""
    wm = oracle_model(missingness=env.missingness)
    env.reset()
    b = wm.initial_belief()
    for a in tests:
        _, _, _, info = env.step(a)
        b = wm.update(b, a, info.get("finding"))
    y_pred = argmax(wm.posterior_y(b))
    env.step(y_pred)
    taken = tuple(env.tests_taken)
    assert env.y_true is not None
    return EpisodeResult(
        env.y_true, y_pred, sum(COST_USD[a] for a in taken), 0.0,
        len(taken), len(tests), taken,
    )


def run(n: int, seed: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"paper": PAPER, "n_episodes": n, "seed": seed}

    env = SyntheticCDMEnv(seed=seed, split="test", version="v1")
    counts: Counter = Counter()
    for _ in range(n):
        env.reset()
        counts[env.y_true] += 1
    out["class_prior"] = {
        y: {"target_mimic_cdm": PRIOR_MIMIC_CDM[y], "sampled": counts[y] / n}
        for y in A_DIAG
    }

    # SFT-all analogue: every test the chart has, then the Bayes decision.
    for miss in (False, True):
        env = SyntheticCDMEnv(seed=seed, split="test", version="v1", missingness=miss)
        res = [all_tests_episode(env, list(A_TEST)) for _ in range(n)]
        m = episode_metrics(res, lam=0.0)
        out[f"all_tests_{'missingness' if miss else 'always_available'}"] = {
            "mean_acc": m["mean_acc"],
            "overall_acc": m["overall_acc"],
            "avg_cost_usd": m["avg_cost_usd"],
            **{f"acc_{y}": m[f"acc_{y}"] for y in A_DIAG},
        }
    out["bayes_cap_overall_acc_approx"] = 1 - P_ATYPICAL
    out["expected_all_available_cost_usd"] = expected_all_available_cost_usd()

    # One test alone: how much does each test reveal? (no missingness)
    wm = oracle_model()
    b0 = wm.initial_belief()
    env = SyntheticCDMEnv(seed=seed, split="test", version="v1")
    single: Dict[str, Dict[str, float]] = {}
    for a in A_TEST:
        res = [all_tests_episode(env, [a]) for _ in range(n)]
        m = episode_metrics(res, lam=0.0)
        single[a] = {
            "cost_usd": COST_USD[a],
            "availability": AVAILABILITY[a],
            "info_gain_nats": wm.information_gain(b0, a),
            "mean_acc_alone": m["mean_acc"],
            "overall_acc_alone": m["overall_acc"],
        }
    out["single_test"] = dict(
        sorted(single.items(), key=lambda kv: -kv[1]["info_gain_nats"])
    )
    majority = argmax(PRIOR_MIMIC_CDM)
    out["no_test_majority_baseline"] = {
        "predict": majority,
        "overall_acc": PRIOR_MIMIC_CDM[majority],
        "mean_acc": 1 / len(A_DIAG),
    }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=str, default="results/synthetic")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sanity.json"
    path.write_text(json.dumps(run(args.n, args.seed), indent=2) + "\n")
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
