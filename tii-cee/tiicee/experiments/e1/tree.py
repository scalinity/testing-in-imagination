"""The reduced twin's outcome tree as dense arrays.

A policy that requests one test per step visits, after t steps, one history per outcome path. With
O possible outcomes per request the paths of length t are numbered in base O: the children of node
n are n * O + o, in outcome order. Every path is kept, including those the truth rules out (mass
zero), because a world model whose support is wider than the truth's gives them probability and
the multi-step accounting of specification 5.12 sums over the world model's paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tiicee.experiments.e1 import batch, filters
from tiicee.twin.env import Twin


def flat(children: np.ndarray) -> np.ndarray:
    """[n, outcome, ...] -> [n * outcome, ...]: the next level, in node order."""
    return children.reshape((children.shape[0] * children.shape[1],) + children.shape[2:])


@dataclass
class Level:
    t: int
    seen: np.ndarray            # [n, item] outcome recorded per item, -1 if unread
    answers: np.ndarray         # [n, test] outcome returned to each requested test, -1 if not requested
    mass: np.ndarray            # [n, class, phenotype] P*(class, phenotype, history); zero where the truth rules the path out
    cost: np.ndarray            # [n] charged normalized cost along the path
    tests: Optional[np.ndarray] = None      # [n] the test requested here; None at the horizon

    @property
    def n(self) -> int:
        return len(self.cost)

    @property
    def asked(self) -> np.ndarray:
        return self.answers >= 0

    @property
    def class_mass(self) -> np.ndarray:
        return self.mass.sum(2)

    def histories(self, model) -> list:
        """Each node's history as the tuple the twin's objects take, in the order the tests were requested."""
        order = np.argsort(np.where(self.asked, self._step, np.iinfo(np.int64).max), axis=1)[:, :self.t]
        return [tuple((model.tests[t], model.outcomes[self.answers[i, t]]) for t in row) for i, row in enumerate(order)]

    _step: np.ndarray = None    # [n, test] the step at which each test was requested, -1 if not


def root(twin: Twin) -> Level:
    m = twin.model
    truth = filters.LatentFilter(m)
    blank = np.full((1, len(m.tests)), -1, dtype=np.int64)
    return Level(0, filters.new_seen(1, m), blank.copy(), truth.start(1), np.zeros(1), None, blank.copy())


def expand(level: Level, twin: Twin, tests: np.ndarray) -> Level:
    """The next level after every node requests its test."""
    m = twin.model
    truth = filters.LatentFilter(m)
    n, n_out = level.n, len(m.outcomes)
    rows = np.repeat(np.arange(n), n_out)
    outcomes = np.tile(np.arange(n_out), n)
    child_tests = tests[rows]
    answers, step = level.answers[rows].copy(), level._step[rows].copy()
    answers[np.arange(n * n_out), child_tests] = outcomes
    step[np.arange(n * n_out), child_tests] = level.t
    charged = np.where(outcomes != n_out - 1, m.cost_norm[child_tests], 0.0)
    return Level(
        level.t + 1,
        filters.record(level.seen[rows], truth.item_of, child_tests, outcomes),
        answers,
        flat(truth.expand(level.mass, level.seen, tests, keep_impossible=False)),
        level.cost[rows] + charged,
        None,
        step,
    )


def choose_order(level: Level, order: np.ndarray) -> np.ndarray:
    """S_rand: the seed's fixed order."""
    return np.full(level.n, order[level.t])


def choose_information(level: Level, twin: Twin) -> np.ndarray:
    """S_ig: the unrequested test with the largest exact information under W0; ties to the lowest index."""
    m = twin.model
    joint = filters.LatentFilter(m).joint_all(level.mass, level.seen)
    info = batch.information(joint.reshape(-1, len(m.classes), len(m.outcomes))).reshape(level.n, len(m.tests))
    return np.argmax(np.where(level.asked, -1.0, info), axis=1)


def skeleton(twin: Twin, horizon: int, choose) -> list:
    """Levels 0..horizon of the tree of a selector that depends on the history only."""
    levels = [root(twin)]
    for _ in range(horizon):
        levels[-1].tests = choose(levels[-1])
        levels.append(expand(levels[-1], twin, levels[-1].tests))
    return levels


# ---------------------------------------------------------------------------------------------
# Exact expectations over a level
# ---------------------------------------------------------------------------------------------
def log_loss(level: Level, belief: np.ndarray) -> float:
    """Exact expected log loss of a belief process at this horizon: -sum over paths and classes of P* log bel."""
    mass = level.class_mass
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.where(mass > 0, mass * np.log(belief), 0.0).sum())


def accuracy(level: Level, belief: np.ndarray) -> float:
    return float(level.class_mass[np.arange(level.n), np.argmax(belief, axis=1)].sum())


def accounting(level: Level, path_prob: np.ndarray, belief: np.ndarray) -> dict:
    """The terms of specification 5.12 (iii) at this horizon (tiicee.cee.multistep_accounting, on arrays).

    path_prob is the world model's probability of each path; belief the process's belief at its end.
    """
    truth = level.class_mass
    joint = path_prob[:, None] * belief
    m0 = joint.sum(0)
    p_star, b_star = truth.sum(1), truth.sum(0)
    seen, labeled = p_star > 0, b_star > 0
    excess = float(p_star[seen] @ batch.kl(truth[seen] / p_star[seen, None], belief[seen]))
    likelihood = float(b_star[labeled] @ batch.kl((truth[:, labeled] / b_star[labeled]).T, (joint[:, labeled] / m0[labeled]).T))
    prior_kl, world_model_kl = float(batch.kl(b_star, m0)), float(batch.kl(p_star, path_prob))
    return {"excess": excess, "prior_kl": prior_kl, "likelihood_kl": likelihood, "world_model_kl": world_model_kl,
            "residual": excess - (prior_kl + likelihood - world_model_kl)}


class Stopper:
    """The exact value of committing the first time a rule says stop along each path."""

    def __init__(self):
        self.alive = np.ones(1, dtype=bool)
        self.accuracy = self.cost = self.requests = 0.0

    def step(self, level: Level, belief: np.ndarray, stop: np.ndarray, n_outcomes: int) -> None:
        """Commit to argmax belief at the live nodes where stop is true (everywhere at the horizon)."""
        halt = self.alive & stop
        mass = level.class_mass[halt]
        self.accuracy += float(mass[np.arange(len(mass)), np.argmax(belief[halt], axis=1)].sum())
        self.cost += float((level.cost[halt] * mass.sum(1)).sum())
        self.requests += float(level.t * mass.sum())
        self.alive = np.repeat(self.alive & ~halt, n_outcomes)

    def value(self, lam: float) -> float:
        return self.accuracy - lam * self.cost
