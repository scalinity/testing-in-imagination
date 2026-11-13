"""Module boundaries of SPEC_RESEARCH_ADVANCEMENT.md section 16, as typing protocols.

A history is opaque to everything except the environment that produced it. An outcome is any
hashable token; UNAVAILABLE is the one shared token. Probability vectors are numpy arrays over the
reader's class list, in that list's order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Optional, Protocol, Sequence

import numpy as np

UNAVAILABLE = "UNAVAILABLE"


class Reader(Protocol):
    """Maps a history to a belief over the classes. Deterministic; sees no labels or unrevealed results."""

    classes: Sequence[str]

    def belief(self, history: Any) -> np.ndarray:
        """b = R(h)."""

    def read(self, history: Any, test: Hashable, outcome: Hashable) -> np.ndarray:
        """q_o = R(h + (test, outcome)). Never writes the imagined outcome into the history."""


@dataclass(frozen=True)
class Predictive:
    """A world model's distribution over one test's outcomes at one history.

    outcomes may include UNAVAILABLE; weights are positive and sum to one. Sampled predictives
    (k draws) carry equal weights per draw, with repeated outcomes merged or kept as the producer
    declares in `merged`.
    """

    outcomes: tuple
    weights: np.ndarray
    merged: bool = True


class WorldModel(Protocol):
    """P_hat(o | h, a). Enforces evidence identity: never contradicts an identical measurement in h."""

    def outcomes(self, history: Any, test: Hashable, k: Optional[int], rng: Optional[np.random.Generator]) -> Predictive:
        """k=None means exact enumeration; otherwise k draws from rng."""


class ClassConditionalWorldModel(WorldModel, Protocol):
    def budget(self, history: Any, test: Hashable) -> float:
        """B-model: sum_o P_hat(o) KL(post_W(. | o) || prior_W), in nats."""
