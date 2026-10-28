"""Agents for env v1: imagination planner, model-free policy, random.

Every agent shares one loop (``run_episode``):

1. A *belief model* turns the real findings so far into p(y | h). The agent
   diagnoses argmax y once max p(y | h) >= ``tau``, when no untried test is
   worth its cost, or on the last step of the budget.
2. Otherwise a *selector* picks the next test.

Arms differ only in the selector (and, for the oracle, the belief model):

- ``ImaginationSelector``: testing in imagination. For each untried test it
  imagines findings s ~ p_phi(s | h, a) under its own world model and scores
  P(available) * IG(a) - lam * cost(a). Planning quality depends on p_phi.
- ``ModelFreeSelector``: never predicts observations. It learns Q(state, a),
  the mean realized gain in log p(y_true | h) minus cost, from real logged
  transitions, where state = (top hypothesis, confidence bin).
- ``RandomSelector``: uniform over untried tests (floor).

Perception is idealized: agents read the structured finding from
``info["finding"]`` rather than parsing the observation text. On MIMIC this
is the job of an LLM or a text classifier.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from extension.envs.synthetic_diagnosis.actions import A_TEST, COST_NORM, COST_USD
from extension.envs.synthetic_diagnosis.env import SyntheticCDMEnv
from extension.world_model.discrete import Belief, DiscreteWorldModel

CONF_BINS: Tuple[float, ...] = (0.5, 0.7, 0.85)


def argmax(d: Mapping[str, float]) -> str:
    return max(d, key=d.get)


def conf_bin(conf: float) -> int:
    return sum(conf >= edge for edge in CONF_BINS)


class Selector:
    """Picks the next test, or None to stop testing and diagnose."""

    def reset(self) -> None:
        pass

    def observe(self, test: str, finding: Optional[str]) -> None:
        pass

    def select(
        self, candidates: Sequence[str], posterior_y: Mapping[str, float]
    ) -> Optional[str]:
        raise NotImplementedError


class ImaginationSelector(Selector):
    def __init__(self, model: DiscreteWorldModel, lam: float) -> None:
        self.model = model
        self.lam = lam
        self.belief: Belief = model.initial_belief()

    def reset(self) -> None:
        self.belief = self.model.initial_belief()

    def observe(self, test: str, finding: Optional[str]) -> None:
        self.belief = self.model.update(self.belief, test, finding)

    def value(self, test: str) -> float:
        ig = self.model.information_gain(self.belief, test)
        return self.model.p_available(test) * ig - self.lam * COST_NORM[test]

    def select(self, candidates, posterior_y):
        if not candidates:
            return None
        best = max(candidates, key=self.value)
        return best if self.value(best) > 0 else None


class ModelFreeSelector(Selector):
    def __init__(
        self, q: Dict[Tuple[str, int], Dict[str, float]], q_global: Dict[str, float]
    ) -> None:
        self.q = q
        self.q_global = q_global

    @classmethod
    def fit(
        cls,
        rows: Iterable[Mapping],
        belief_model: DiscreteWorldModel,
        lam: float,
        min_count: int = 5,
    ) -> "ModelFreeSelector":
        """Monte-Carlo Q from logged transitions (one-step realized return).

        A transition's return is log p(y_true | h_{t+1}) - log p(y_true | h_t)
        under the belief model, minus lam * cost if the test was provided.
        UNAVAILABLE transitions return 0, so the policy learns from
        experience that rarely-charted tests rarely pay off.
        """
        sums: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        g_sum: Dict[str, float] = defaultdict(float)
        g_cnt: Dict[str, int] = defaultdict(int)
        by_ep: Dict[str, List[Mapping]] = defaultdict(list)
        for r in rows:
            by_ep[r["episode_id"]].append(r)
        for ep_rows in by_ep.values():
            b = belief_model.initial_belief()
            for r in sorted(ep_rows, key=lambda r: r["t"]):
                a, s, y = r["a_t"], r["meta"].get("finding"), r["y_true"]
                post = belief_model.posterior_y(b)
                state = (argmax(post), conf_bin(max(post.values())))
                b_next = belief_model.update(b, a, s)
                gain = math.log(max(belief_model.posterior_y(b_next)[y], 1e-12)) - math.log(
                    max(post[y], 1e-12)
                )
                ret = gain - (lam * COST_NORM[a] if s is not None else 0.0)
                sums[state][a] += ret
                counts[state][a] += 1
                g_sum[a] += ret
                g_cnt[a] += 1
                b = b_next
        q_global = {a: g_sum[a] / g_cnt[a] for a in g_cnt}
        q = {
            state: {a: sums[state][a] / n for a, n in c.items() if n >= min_count}
            for state, c in counts.items()
        }
        return cls(q, q_global)

    def select(self, candidates, posterior_y):
        state = (argmax(posterior_y), conf_bin(max(posterior_y.values())))
        table = self.q.get(state, {})

        def score(a: str) -> float:
            return table.get(a, self.q_global.get(a, 0.0))

        if not candidates:
            return None
        best = max(candidates, key=score)
        return best if score(best) > 0 else None


class RandomSelector(Selector):
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select(self, candidates, posterior_y):
        return self.rng.choice(list(candidates)) if candidates else None


@dataclass
class EpisodeResult:
    y_true: str
    y_pred: Optional[str]  # None = undiagnosed (budget exhausted)
    cost_usd: float
    cost_norm: float
    n_tests: int  # tests that returned data (cost basis, F-2)
    n_requests: int  # all test requests incl. UNAVAILABLE
    tests: Tuple[str, ...]

    @property
    def correct(self) -> bool:
        return self.y_pred == self.y_true


def run_episode(
    env: SyntheticCDMEnv,
    selector: Selector,
    belief_model: DiscreteWorldModel,
    tau: float,
) -> EpisodeResult:
    if env.version != "v1":
        raise ValueError("agents need env version v1 (structured findings)")
    env.reset()
    selector.reset()
    b = belief_model.initial_belief()
    y_pred: Optional[str] = None
    requests = 0
    while not env.done:
        post = belief_model.posterior_y(b)
        last_step = env.max_steps is not None and env.t >= env.max_steps - 1
        action: Optional[str] = None
        if max(post.values()) < tau and not last_step:
            action = selector.select(env.available_tests, post)
        if action is None:
            action = argmax(post)
            y_pred = action
            env.step(action)
            break
        requests += 1
        _, _, _, info = env.step(action)
        finding = info.get("finding")
        b = belief_model.update(b, action, finding)
        selector.observe(action, finding)
    tests = tuple(env.tests_taken)
    assert env.y_true is not None
    return EpisodeResult(
        y_true=env.y_true,
        y_pred=y_pred,
        cost_usd=sum(COST_USD[a] for a in tests),
        cost_norm=sum(COST_NORM[a] for a in tests),
        n_tests=len(tests),
        n_requests=requests,
        tests=tests,
    )
