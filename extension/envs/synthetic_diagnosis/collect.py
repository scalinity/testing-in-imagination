"""Collect (h_t, a_t, o_{t+1}) tuples from synthetic CDM rollouts → JSONL.

Usage:
  python -m extension.envs.synthetic_diagnosis.collect \\
      --n-episodes 20 --seed 0 --out /tmp/tuples.jsonl --policy random
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import List, Optional

from .actions import A_DIAG, A_TEST
from .env import SyntheticCDMEnv
from .tuples import JSONLWriter, TupleRecord

# Preferred first-line tests per disease (scripted expert stub)
_EXPERT_FIRST_TEST = {
    "appendicitis": "Physical Examination",
    "cholecystitis": "Ultrasound",
    "diverticulitis": "CT",
    "pancreatitis": "Comprehensive Metabolic Panel",
}
_EXPERT_SECOND_TEST = {
    "appendicitis": "CT",
    "cholecystitis": "Liver Function Panel",
    "diverticulitis": "Complete Blood Count",
    "pancreatitis": "CT",
}


def _random_policy_action(
    env: SyntheticCDMEnv,
    rng: random.Random,
    *,
    diagnose_prob: float = 0.15,
    max_tests_before_diag: int = 5,
) -> str:
    available = env.available_tests
    n_taken = len(env.tests_taken)
    force_diag = n_taken >= max_tests_before_diag or not available
    if force_diag or (n_taken > 0 and rng.random() < diagnose_prob):
        return rng.choice(list(A_DIAG))
    if not available:
        return rng.choice(list(A_DIAG))
    return rng.choice(available)


def _expert_policy_action(env: SyntheticCDMEnv, rng: random.Random) -> str:
    """Disease-aware stub: preferred tests then correct diagnosis.

    Uses env.y_true (oracle) — for logging expert trajectories only, not
    a fair agent. Still useful for warm-start WM data.
    """
    assert env.y_true is not None
    y = env.y_true
    taken = set(env.tests_taken)
    first = _EXPERT_FIRST_TEST[y]
    second = _EXPERT_SECOND_TEST[y]
    if first not in taken:
        return first
    if second not in taken:
        return second
    # Optional cheap confirmatory lab
    if "Complete Blood Count" not in taken and rng.random() < 0.5:
        return "Complete Blood Count"
    return y  # correct diagnosis


def rollout_episode(
    env: SyntheticCDMEnv,
    policy: str,
    rng: random.Random,
    *,
    diagnose_prob: float = 0.15,
    max_tests_before_diag: int = 5,
) -> List[TupleRecord]:
    """Run one episode; return tuples for test actions only."""
    h = env.reset()
    assert env.y_true is not None
    y_true = env.y_true
    episode_id = env.episode_id
    assert episode_id is not None
    records: List[TupleRecord] = []

    while not env.done:
        t = env.t
        h_t = env.history
        available = env.available_tests
        if policy == "random":
            a = _random_policy_action(
                env,
                rng,
                diagnose_prob=diagnose_prob,
                max_tests_before_diag=max_tests_before_diag,
            )
        elif policy == "expert":
            a = _expert_policy_action(env, rng)
        else:
            raise ValueError(f"Unknown policy: {policy!r}")

        o, _reward, done, info = env.step(a)

        # Only emit tuples for test actions (not diagnosis)
        if a in A_TEST and not info.get("invalid") and not info.get("duplicate_test"):
            rec = TupleRecord(
                episode_id=episode_id,
                t=t,
                h_t=h_t,
                a_t=a,
                o_t1=o,
                y_true=y_true,
                source="synthetic",
                meta={
                    "split": env.split,
                    "available_tests_at_t": available,
                    "seed": env.seed,
                    "patient_seed": env.patient_seed,
                    "policy": policy,
                },
            )
            records.append(rec)

        if done:
            break
        # Safety: prevent infinite loops on pathological policies
        if env.t > 30:
            break

    return records


def collect(
    n_episodes: int,
    seed: int,
    out_path: str,
    policy: str = "random",
    split: str = "train",
    diagnose_prob: float = 0.15,
    max_tests_before_diag: int = 5,
) -> int:
    env = SyntheticCDMEnv(seed=seed, split=split)
    rng = random.Random(seed + 17)
    total = 0
    with JSONLWriter(out_path) as writer:
        for _ in range(n_episodes):
            recs = rollout_episode(
                env,
                policy,
                rng,
                diagnose_prob=diagnose_prob,
                max_tests_before_diag=max_tests_before_diag,
            )
            writer.write_many(recs)
            total += len(recs)
    return total


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect synthetic CDM world-model tuples to JSONL."
    )
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=("random", "expert"),
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "test"),
    )
    p.add_argument("--diagnose-prob", type=float, default=0.15)
    p.add_argument("--max-tests-before-diag", type=int, default=5)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    n = collect(
        n_episodes=args.n_episodes,
        seed=args.seed,
        out_path=args.out,
        policy=args.policy,
        split=args.split,
        diagnose_prob=args.diagnose_prob,
        max_tests_before_diag=args.max_tests_before_diag,
    )
    print(f"Wrote {n} tuples → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
