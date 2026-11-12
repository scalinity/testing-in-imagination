"""Discrete world model p_phi(o_{t+1} | h_t, a_t) over v1 findings.

The model has a latent z (the disease for learned models, the presenting
phenotype for the oracle) and factorizes the history through a belief b_h:

    p(s | h, a) = sum_z b_h(z) * lik[a][z][s]
    p(y | h)    = sum_z b_h(z) * y_given_z[z][y]

Beliefs are plain dicts z -> prob, updated by Bayes rule on real or
imagined findings. A finding of None (UNAVAILABLE) carries no information.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Mapping, Optional

from extension.envs.synthetic_diagnosis.actions import A_DIAG, A_TEST
from extension.envs.synthetic_diagnosis.profiles import FINDINGS, sample_categorical

Belief = Dict[str, float]
Likelihood = Dict[str, Dict[str, Dict[str, float]]]  # test -> z -> finding -> p


def entropy(dist: Mapping[str, float]) -> float:
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


class DiscreteWorldModel:
    def __init__(
        self,
        prior_z: Mapping[str, float],
        lik: Likelihood,
        y_given_z: Optional[Mapping[str, Mapping[str, float]]] = None,
        availability: Optional[Mapping[str, float]] = None,
        name: str = "wm",
    ) -> None:
        self.prior_z = dict(prior_z)
        self.lik = lik
        if y_given_z is None:
            # z is the diagnosis itself.
            y_given_z = {z: {y: float(y == z) for y in A_DIAG} for z in prior_z}
        self.y_given_z = {z: dict(row) for z, row in y_given_z.items()}
        self.availability = dict(availability or {a: 1.0 for a in A_TEST})
        self.name = name

    # ----------------------------------------------------------- beliefs

    def initial_belief(self) -> Belief:
        return dict(self.prior_z)

    def update(self, belief: Belief, test: str, finding: Optional[str]) -> Belief:
        if finding is None:
            return dict(belief)
        post = {z: b * self.lik[test][z][finding] for z, b in belief.items()}
        total = sum(post.values())
        if total <= 0:
            # The model gives this finding zero probability; keep the belief.
            return dict(belief)
        return {z: p / total for z, p in post.items()}

    def posterior_y(self, belief: Belief) -> Dict[str, float]:
        out = {y: 0.0 for y in A_DIAG}
        for z, b in belief.items():
            for y, p in self.y_given_z[z].items():
                out[y] += b * p
        return out

    # ------------------------------------------------------- imagination

    def predictive(self, belief: Belief, test: str) -> Dict[str, float]:
        """p(o_{t+1} | h_t, a_t) for a test that returns a finding."""
        out = {s: 0.0 for s in FINDINGS}
        for z, b in belief.items():
            for s, p in self.lik[test][z].items():
                out[s] += b * p
        return out

    def sample(self, belief: Belief, test: str, rng: random.Random) -> str:
        return sample_categorical(self.predictive(belief, test), rng)

    def p_available(self, test: str) -> float:
        return self.availability.get(test, 1.0)

    def information_gain(self, belief: Belief, test: str) -> float:
        """Expected reduction in entropy of p(y | h) from imagining ``test``."""
        h0 = entropy(self.posterior_y(belief))
        expected = 0.0
        for s, p in self.predictive(belief, test).items():
            if p > 0:
                expected += p * entropy(self.posterior_y(self.update(belief, test, s)))
        return max(h0 - expected, 0.0)

    def log_likelihood(self, rows) -> float:
        """Mean log p(s | h, a) over tuple dicts, filtering each episode.

        Rows are grouped by episode and ordered by t. UNAVAILABLE rows are
        skipped. Used to score world-model fit on held-out tuples.
        """
        by_ep: Dict[str, list] = {}
        for r in rows:
            by_ep.setdefault(r["episode_id"], []).append(r)
        total, n = 0.0, 0
        for ep_rows in by_ep.values():
            b = self.initial_belief()
            for r in sorted(ep_rows, key=lambda r: r["t"]):
                s = r["meta"].get("finding")
                if s is None:
                    continue
                p = self.predictive(b, r["a_t"])[s]
                total += math.log(max(p, 1e-12))
                n += 1
                b = self.update(b, r["a_t"], s)
        return total / max(n, 1)
