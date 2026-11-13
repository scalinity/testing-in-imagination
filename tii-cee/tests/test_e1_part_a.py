"""Part A: the eight exact values of an agent against Twin.exact_value on agents rebuilt from the library's functions."""
from __future__ import annotations

import numpy as np
import pytest

from tiicee import cee
from tiicee.experiments.e1 import heldout, part_a
from tiicee.experiments.e1.config import tiny
from tiicee.experiments.e1.context import Context
from tiicee.twin.readers import BayesReader, SimulatedReader


def dense(model, predictive):
    out = np.zeros(len(model.outcomes))
    for outcome, weight in zip(predictive.outcomes, predictive.weights):
        out[model.outcomes.index(outcome)] += weight
    return out


class Agent:
    """The raw or the repaired agent, one history at a time, from tiicee.cee's public functions."""

    def __init__(self, ctx, reader, kind, layer, lam, budget_of=None):
        self.twin = ctx.twin("reduced")
        self.model = self.twin.model
        self.world = ctx.fitted("reduced", "W2")
        self.reader, self.kind, self.layer, self.lam, self.budget_of = reader, kind, layer, lam, budget_of
        self.threshold = ctx.cfg.threshold
        self.utility = cee.identity_utility(len(self.model.classes))
        self.classes = self.model.classes
        self._beliefs = {}

    def _pair(self, history, test):
        weights = dense(self.model, self.world.outcomes(history, test, None, None))
        return weights, [self.reader.read(history, test, o) for o in self.model.outcomes]

    def _step(self, history, test, outcome_index):
        weights, posteriors = self._pair(history, test)
        scaling = None
        if self.layer.scale is not None:
            budget, eligible = self.budget_of(history, test)
            scaling = (budget, self.layer.scale) if eligible else None
        if self.layer.track == "stateless":
            return cee.stateless_step(self.reader.belief(history), weights, posteriors, outcome_index, scaling)
        return cee.stateful_step(self.belief(history), self.reader.belief(history), weights, posteriors, outcome_index, scaling)

    def belief(self, history):
        """R: the reader, or the layer's tracked belief."""
        if self.kind == "raw":
            return cee.floor_norm(self.reader.belief(history))
        if history not in self._beliefs:
            if not history:
                self._beliefs[history] = cee.floor_norm(self.reader.belief(()))
            else:
                test, outcome = history[-1]
                self._beliefs[history] = self._step(history[:-1], test, self.model.outcomes.index(outcome)).belief
        return self._beliefs[history]

    def scores(self, history):
        """Net score of every eligible test, in the model's order."""
        out = []
        for test in self.twin.eligible(history):
            weights, posteriors = self._pair(history, test)
            cost = cee.expected_charged_cost(1.0 - weights[-1], float(self.model.cost_norm[self.model.tests.index(test)]))
            if self.kind == "raw":
                out.append(cee.entropy_change_surrogate(self.reader.belief(history), weights, posteriors).score - self.lam * cost)
                continue
            target = cee.floor_norm(self.reader.belief(history)) if self.layer.track == "stateless" else self.belief(history)
            imagined = np.array([self._step(history, test, o).belief for o in range(len(self.model.outcomes))])
            out.append(cee.net_value(cee.coherent_voi(target, weights, imagined, self.utility), cost, self.lam))
        return out

    def select(self, history, belief):
        return self.twin.eligible(history)[int(np.argmax(self.scores(history)))]

    def commit(self, history, belief, forced):
        if self.kind == "raw":
            stop = forced or belief.max() >= self.threshold
        else:
            stop = forced or cee.select(self.scores(history)) is None
        return int(np.argmax(belief)) if stop else None


def reference_values(ctx, agent, lam):
    twin = ctx.twin("reduced")
    optimal = twin.bayes_optimal(budget=ctx.horizon("reduced"), lam=lam)
    r0 = BayesReader(twin)
    out = {"v_star": optimal.value}
    for s_ref, r_ref, c_ref in part_a.COMBINATIONS:
        selector = (lambda h, b: optimal.best_test(h)) if s_ref else agent.select
        reader = r0 if r_ref else agent
        if c_ref:
            commit = lambda h, b, forced: int(np.argmax(b)) if forced or optimal.should_stop(h) else None
        else:
            commit = agent.commit
        out[part_a.key((s_ref, r_ref, c_ref))] = twin.exact_value(selector, reader, commit, budget=ctx.horizon("reduced"), lam=lam).utility
    return out


@pytest.fixture(scope="module")
def ctx():
    return Context(tiny(n_train=400, n_boot=300), 1, "base")


@pytest.mark.parametrize("reader_name,options", [("entrench", {"drift": "entrench"}), ("over", {"power": 2.0}), ("fixed+under", {"drift": "fixed", "power": 0.5})])
@pytest.mark.parametrize("lam", [1.0, 0.0])
def test_raw_agent_values_match_exact_value(ctx, reader_name, options, lam):
    twin = ctx.twin("reduced")
    got = part_a.agent_values(ctx, reader_name, "raw", part_a.Layer(), lam)
    expected = reference_values(ctx, Agent(ctx, SimulatedReader(twin, **options), "raw", part_a.Layer(), lam), lam)
    for name, value in expected.items():
        assert got[name] == pytest.approx(value, abs=1e-10), name
    assert got["v_***"] == pytest.approx(got["v_star"], abs=1e-12)          # all three references together are the optimum
    assert all(got[part_a.key(c)] <= got["v_star"] + 1e-12 for c in part_a.COMBINATIONS)


@pytest.mark.parametrize("track", ["stateless", "stateful"])
def test_repaired_agent_values_match_exact_value(ctx, track):
    twin = ctx.twin("reduced")
    layer = part_a.Layer(track)
    for reader_name, options in (("entrench", {"drift": "entrench"}), ("independence", {})):
        got = part_a.agent_values(ctx, reader_name, "repaired", layer, 1.0)
        expected = reference_values(ctx, Agent(ctx, SimulatedReader(twin, **options), "repaired", layer, 1.0), 1.0)
        for name, value in expected.items():
            assert got[name] == pytest.approx(value, abs=1e-9), (reader_name, name)


def test_repaired_agent_with_the_held_out_scale_layer_matches_exact_value(ctx):
    twin = ctx.twin("reduced")
    m = twin.model
    estimate = heldout.HeldOut(ctx)
    assert estimate.eligible.any()
    layer = part_a.Layer("stateful", "two-sided", "heldout")
    assert layer.label() == "stateful+two-sided(heldout)" and part_a.Layer().label() == "stateless"

    def budget_of(history, test):
        answers = np.full((1, len(m.tests)), -1)
        for name, outcome in history:
            answers[0, m.tests.index(name)] = m.outcomes.index(outcome)
        budget, eligible = estimate.lookup(answers, np.array([m.tests.index(test)]))
        return float(budget[0]), bool(eligible[0])

    got = part_a.agent_values(ctx, "over", "repaired", layer, 1.0, heldout=estimate)
    agent = Agent(ctx, SimulatedReader(twin, power=2.0), "repaired", layer, 1.0, budget_of)
    expected = reference_values(ctx, agent, 1.0)
    for name, value in expected.items():
        assert got[name] == pytest.approx(value, abs=1e-9), name


def test_attribution_increments_sum_to_the_regret_in_every_order(ctx):
    row = part_a.agent_values(ctx, "fixed", "raw", part_a.Layer(), 1.0)
    result = part_a.attribution(row)
    assert result["regret"] == pytest.approx(row["v_star"] - row["v_SRC"]) and result["regret"] > 0
    assert len(result["orders"]) == 6 and ("selection", "reading", "commitment") in result["orders"]
    for order, increments in result["orders"].items():
        assert set(increments) == set(part_a.COMPONENTS)
        assert sum(increments.values()) == pytest.approx(result["regret"], abs=1e-12)
    declared = result["orders"][("selection", "reading", "commitment")]
    assert declared["selection"] == pytest.approx(row["v_*RC"] - row["v_SRC"])
    assert declared["reading"] == pytest.approx(row["v_**C"] - row["v_*RC"])
    assert declared["commitment"] == pytest.approx(row["v_***"] - row["v_**C"])
    assert [part_a.key(c) for c in part_a.COMBINATIONS] == ["v_SRC", "v_SR*", "v_S*C", "v_S**", "v_*RC", "v_*R*", "v_**C", "v_***"]
