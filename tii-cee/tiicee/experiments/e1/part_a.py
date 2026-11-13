"""Part A: exact values of the raw and the repaired agent with each component swapped to its reference.

An agent is three maps on histories: which test to request (S), what to believe (R), when to stop (C).

    raw agent        S_raw: the unrequested test with the largest entropy-change surrogate of the
                     reader under W2, minus lambda x expected charged cost. R: the reader. C: stop
                     when the belief it is given reaches the threshold.
    repaired agent   S: CEE-Select under W2, on the reader behind the chosen CEE layer. R: the
                     layer's tracked belief. C: CEE-Commit, stop when no unrequested test has
                     positive net value.
    references       S*: the Bayes-optimal request. R*: R0. C*: Bayes-optimal stopping.

A swap replaces one map and leaves the other two as they are: S keeps using the agent's own reader
and layer when R is swapped, and a threshold C reads whichever belief it is handed. Every agent
commits to the class its belief ranks first and is forced to commit at the step budget. V of the
eight combinations is exact, by the dense tree of tiicee.experiments.e1.tree.
"""
from __future__ import annotations

import itertools
from typing import NamedTuple, Optional

import numpy as np

from tiicee.cee import entropy, floor_norm
from tiicee.experiments.e1 import filters, tree
from tiicee.experiments.e1.context import Context
from tiicee.experiments.e1.reduced import _scaled_step

COMPONENTS = ("selection", "reading", "commitment")
COMBINATIONS = tuple(itertools.product((False, True), repeat=3))        # (S is reference, R is reference, C is reference)


class Layer(NamedTuple):
    """The CEE layer of the repaired agent: a track, and optionally CEE-Scale with a variant and a budget source."""

    track: str = "stateless"
    scale: Optional[str] = None
    budget: Optional[str] = None

    def label(self) -> str:
        return self.track if self.scale is None else f"{self.track}+{self.scale}({self.budget})"


def key(combination) -> str:
    return "v_" + "".join("*" if ref else letter for ref, letter in zip(combination, "SRC"))


def _walk(ctx: Context, reader_name: str, agent: str, layer: Layer, lam: float, optimal, heldout, reference_selection: bool) -> dict:
    cfg, twin = ctx.cfg, ctx.twin("reduced")
    m = twin.model
    horizon, n_out = ctx.horizon("reduced"), len(m.outcomes)
    reader, r0 = filters.reader(m, reader_name), filters.reader(m, "r0")
    world = ctx.world_filter("reduced", "W2")
    level = tree.root(twin)
    r_state, r0_state, w_state = reader.start(1), r0.start(1), world.start(1)
    tracked = None
    stoppers = {(r, c): tree.Stopper() for r in (False, True) for c in (False, True)}
    for t in range(horizon + 1):
        n, last = level.n, t == horizon
        bf = floor_norm(reader.belief(r_state))
        if tracked is None:
            tracked = bf
        own_belief = bf if agent == "raw" else tracked
        histories = level.histories(m)
        stop_reference = np.array([optimal.should_stop(h) for h in histories])
        own_stop = None
        if not last:
            rows, tests = np.nonzero(~level.asked)
            k = len(m.tests) - t
            p = filters.predictive(world.joint_all(w_state, level.seen))[rows, tests]
            q = floor_norm(reader.reactions_all(r_state, level.seen)[rows, tests])
            cost = (1.0 - p[:, -1]) * m.cost_norm[tests]
            if agent == "raw":
                score = entropy(bf[rows]) - (p * entropy(q)).sum(1) - lam * cost
            else:
                if layer.scale is None:
                    budget, eligible = None, np.zeros(len(rows), dtype=bool)
                else:
                    budget, eligible = heldout.lookup(level.answers[rows], tests)
                target = bf[rows] if layer.track == "stateless" else tracked[rows]
                imagined = _scaled_step(layer.track, tracked[rows], bf[rows], p, q, budget, eligible, layer.scale)[0]
                score = (p * imagined.max(2)).sum(1) - target.max(1) - lam * cost
                own_stop = score.reshape(n, k).max(1) <= 0
            best = np.argmax(score.reshape(n, k), axis=1)
            if reference_selection:
                names = [optimal.best_test(h) for h in histories]
                chosen = np.array([m.tests.index(name) for name in names])
                position = np.argmax(tests.reshape(n, k) == chosen[:, None], axis=1)
            else:
                position, chosen = best, tests.reshape(n, k)[np.arange(n), best]
        for (r_ref, c_ref), stopper in stoppers.items():
            belief = r0.belief(r0_state) if r_ref else own_belief
            if last:
                stop = np.ones(n, dtype=bool)
            elif c_ref:
                stop = stop_reference
            else:
                stop = belief.max(1) >= cfg.threshold if agent == "raw" else own_stop
            stopper.step(level, belief, stop, n_out)
        if last:
            break
        if agent == "repaired":
            tracked = tree.flat(imagined[np.arange(n) * k + position])
        r_state = tree.flat(reader.expand(r_state, level.seen, chosen))
        r0_state = tree.flat(r0.expand(r0_state, level.seen, chosen))
        w_state = tree.flat(world.expand(w_state, level.seen, chosen, keep_impossible=True))
        level = tree.expand(level, twin, chosen)
    return {c: (s.value(lam), s.accuracy, s.cost, s.requests) for c, s in stoppers.items()}


def agent_values(ctx: Context, reader_name: str, agent: str, layer: Layer, lam: float, heldout=None) -> dict:
    """V of the eight combinations for one agent, the Bayes-optimal value, and the agent's own accuracy, cost and requests."""
    twin = ctx.twin("reduced")
    optimal = twin.bayes_optimal(budget=ctx.horizon("reduced"), lam=lam)
    row = {"v_star": optimal.value}
    for reference_selection in (False, True):
        values = _walk(ctx, reader_name, agent, layer, lam, optimal, heldout, reference_selection)
        for (r_ref, c_ref), (value, accuracy, cost, requests) in values.items():
            row[key((reference_selection, r_ref, c_ref))] = value
            if not (reference_selection or r_ref or c_ref):
                row.update(accuracy=accuracy, cost=cost, requests=requests)
    return row


def attribution(row: dict) -> dict:
    """Increments of V as the components are swapped to their references, for each of the six orders.

    Returns {"regret": V* - V, "orders": {order: {component: increment}}}; an order is a tuple of
    component names and its three increments sum to the regret.
    """
    out = {"regret": row["v_star"] - row[key((False, False, False))], "orders": {}}
    for order in itertools.permutations(range(3)):
        state, increments = [False, False, False], {}
        for component in order:
            before = row[key(tuple(state))]
            state[component] = True
            increments[COMPONENTS[component]] = row[key(tuple(state))] - before
        out["orders"][tuple(COMPONENTS[c] for c in order)] = increments
    return out
