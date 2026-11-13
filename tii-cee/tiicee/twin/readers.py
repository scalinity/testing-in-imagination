"""Readers R0 and R4 of the registry (specification section 6.2), for the twin.

A reader maps a history to a belief over the classes and sees nothing else. read(history, test,
outcome) is the belief the reader would hold after one more result; it builds a new history and
leaves the one it was given untouched.

R0 is the posterior of the true model. The R4 family are simulated readers with a known fault. Each
starts from the true prior and folds the history one result at a time through

    q(y) proportional to b(y) x P(outcome | test, y) ** power,   then   q(k) <- q(k) x exp(delta),

where P(outcome | test, y) is the true single-test likelihood given the class (the phenotype summed
out) and k is the drifting class. Treating results as independent given the class double counts
whatever the shared phenotype, or a repeated measurement, already said. Defaults follow
cee_check.py: delta 0.4, powers 2 and 0.5, fixed class index 1.

    independence     power 1, no drift
    over, under      power 2, power 0.5
    entrench         drift toward the argmax of the belief before the update
    fixed            drift toward one fixed class
    mixtures         a drift together with a power

The drift is applied at every read, whatever the result says, including UNAVAILABLE. A result the
reader's likelihood rules out for every class carries no evidence.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tiicee.twin.env import ImpossibleHistory, Twin


class BayesReader:
    """R0: the Bayes reader under the true model."""

    name = "R0"

    def __init__(self, twin: Twin):
        self.twin = twin
        self.classes = twin.model.classes

    def belief(self, history) -> np.ndarray:
        return self.twin.true_posterior(history)

    def read(self, history, test, outcome) -> np.ndarray:
        """The posterior after one more result.

        A result already in the history changes nothing. An imagined result that the true model
        rules out (a world model without evidence identity can propose one) carries no update.
        """
        belief = self.twin.true_posterior(history)
        try:
            return self.twin.true_posterior(tuple(history) + ((test, outcome),))
        except ImpossibleHistory:
            self.twin.index(test, outcome)
            return belief


class SimulatedReader:
    """One member of the R4 family."""

    def __init__(self, twin: Twin, power: float = 1.0, drift: Optional[str] = None, delta: float = 0.4,
                 fixed_class: int = 1, name: Optional[str] = None):
        if drift not in (None, "entrench", "fixed"):
            raise ValueError(f"unknown drift {drift!r}")
        if not power > 0:
            raise ValueError("power must be positive")
        if not 0 <= fixed_class < len(twin.model.classes):
            raise ValueError("fixed_class is not a class index")
        self.twin = twin
        self.classes = twin.model.classes
        self.power, self.drift, self.delta, self.fixed_class = float(power), drift, float(delta), int(fixed_class)
        self.name = name or "R4"
        self._lik = twin.model.class_lik()          # [test, class, outcome]

    def _react(self, belief: np.ndarray, test, outcome) -> np.ndarray:
        t, o = self.twin.index(test, outcome)
        q = belief * self._lik[t, :, o] ** self.power
        if not q.sum() > 0:                          # no class allows this result: no evidence
            q = belief.copy()
        if self.drift is not None:
            k = int(np.argmax(belief)) if self.drift == "entrench" else self.fixed_class
            q[k] *= np.exp(self.delta)
        return q / q.sum()

    def belief(self, history) -> np.ndarray:
        belief = np.array(self.twin.model.prior)
        for test, outcome in history:
            belief = self._react(belief, test, outcome)
        return belief

    def read(self, history, test, outcome) -> np.ndarray:
        return self._react(self.belief(history), test, outcome)


def r4_family(twin: Twin, delta: float = 0.4, fixed_class: int = 1) -> dict:
    """The simulated readers of the registry, by name."""
    drifts = {"entrench": dict(drift="entrench", delta=delta),
              "fixed": dict(drift="fixed", delta=delta, fixed_class=fixed_class)}
    scales = {"over": 2.0, "under": 0.5}
    family = {"independence": SimulatedReader(twin, name="R4/independence")}
    family.update({name: SimulatedReader(twin, name=f"R4/{name}", **options) for name, options in drifts.items()})
    family.update({name: SimulatedReader(twin, power=power, name=f"R4/{name}") for name, power in scales.items()})
    for drift, options in drifts.items():
        for scale, power in scales.items():
            family[f"{drift}+{scale}"] = SimulatedReader(twin, power=power, name=f"R4/{drift}+{scale}", **options)
    return family
