"""Brute-force references shared by the twin tests, and checks of the references themselves.

Nothing here calls the twin's inference code. `Joint` holds the probability of every latent case of
a model in one explicit array, p[class, phenotype, outcome of test 1, ..., outcome of test T], and
every reference quantity is a slice and a sum of that array.
"""
from __future__ import annotations

import functools
import itertools
import pathlib

import numpy as np
import pytest

from tiicee.interfaces import UNAVAILABLE
from tiicee.twin.model import TwinModel

REPO = pathlib.Path(__file__).resolve().parents[2]
CEE_CHECK = REPO / "reports" / "research-advancement" / "cee_check.py"


@functools.lru_cache(maxsize=None)
def load_cee_check() -> dict:
    """The definitions of cee_check.py, executed from source so that no bytecode is written beside it."""
    if not CEE_CHECK.exists():
        pytest.skip(f"cross-check source not present: {CEE_CHECK}")
    namespace = {"__name__": "cee_check"}
    exec(compile(CEE_CHECK.read_text(), str(CEE_CHECK), "exec"), namespace)
    return namespace


def toy_model(availability=(1.0, 0.3, 0.0)) -> TwinModel:
    """Two classes, two phenotypes, three tests with two findings each. Small enough to check by hand.

    availability is one value per test, or one row per test with a value per class.
    """
    by_phenotype = np.array([
        [[0.9, 0.1], [0.2, 0.8]],
        [[0.7, 0.3], [0.4, 0.6]],
        [[0.5, 0.5], [0.1, 0.9]],
    ])
    avail = np.array(availability, dtype=float)
    if avail.ndim == 1:
        avail = np.repeat(avail[:, None], 2, axis=1)
    return TwinModel(
        name="toy", variant="toy", held_out=False,
        classes=("a", "b"), phenotypes=("p", "q"), findings=("pos", "neg"),
        tests=("t0", "t1", "t2"), items=("t0", "t1", "t2"), item_of=(0, 1, 2),
        prior=np.array([0.6, 0.4]),
        phenotype_given_class=np.array([[0.8, 0.2], [0.3, 0.7]]),
        finding_lik=np.repeat(by_phenotype[:, None], 2, axis=1),
        availability=avail,
        cost_usd=np.array([100.0, 250.0, 50.0]),
        cost_norm=np.array([0.25, 0.625, 0.125]),
        step_limit=4, source="toy",
    )


class Joint:
    """Every latent case of a model, enumerated."""

    def __init__(self, model: TwinModel):
        self.model = model
        self.outcomes = tuple(model.findings) + (UNAVAILABLE,)
        n_class, n_pheno = len(model.classes), len(model.phenotypes)
        n_out, n_items, n_tests = len(self.outcomes), len(model.items), len(model.tests)
        n_find = len(model.findings)
        by_item = model.prior[:, None] * model.phenotype_given_class
        for i in range(n_items):
            lik = np.empty((n_class, n_pheno, n_out))
            lik[:, :, :n_find] = model.availability[i][:, None, None] * model.finding_lik[i]
            lik[:, :, n_find] = (1.0 - model.availability[i])[:, None]
            by_item = by_item[..., None] * lik.reshape((n_class, n_pheno) + (1,) * i + (n_out,))
        grid = np.indices((n_out,) * n_tests)
        first = [model.item_of.index(i) for i in range(n_items)]
        consistent = np.ones((n_out,) * n_tests, dtype=bool)
        for t in range(n_tests):
            consistent &= grid[t] == grid[first[model.item_of[t]]]
        self.p = by_item[(slice(None), slice(None)) + tuple(grid[first[i]] for i in range(n_items))] * consistent

    def mass(self, history) -> np.ndarray:
        """P(class, phenotype, the observations in history), as an array over (class, phenotype)."""
        seen = {}
        for test, outcome in history:
            if seen.setdefault(test, outcome) != outcome:
                return np.zeros(self.p.shape[:2])
        index = [slice(None), slice(None)]
        for test in self.model.tests:
            index.append(self.outcomes.index(seen[test]) if test in seen else slice(None))
        sub = self.p[tuple(index)]
        return sub.reshape(sub.shape[0], sub.shape[1], -1).sum(-1)

    def posterior(self, history) -> np.ndarray:
        m = self.mass(history).sum(1)
        return m / m.sum()

    def predictive(self, history, test) -> dict:
        total = self.mass(history).sum()
        return {o: self.mass(tuple(history) + ((test, o),)).sum() / total for o in self.outcomes}

    def information(self, history, test) -> float:
        """I(Y; O | h, a) from its definition, sum p(y, o) log[p(y, o) / (p(y) p(o))]."""
        total = self.mass(history).sum()
        joint = np.stack([self.mass(tuple(history) + ((test, o),)).sum(1) for o in self.outcomes], axis=1) / total
        p_y, p_o = joint.sum(1), joint.sum(0)
        return float(sum(
            joint[y, o] * np.log(joint[y, o] / (p_y[y] * p_o[o]))
            for y in range(joint.shape[0]) for o in range(joint.shape[1]) if joint[y, o] > 0
        ))

    def outcome_tuples(self):
        """(outcomes of every test, P(class, those outcomes)) for every tuple with positive probability."""
        by_class = self.p.sum(1)
        for index in map(tuple, np.argwhere(by_class.sum(0) > 0)):
            yield dict(zip(self.model.tests, (self.outcomes[o] for o in index))), by_class[(slice(None),) + index]


def play(model, outcome_of, selector, reader, commit, budget, memo=None):
    """One episode on one fully specified patient. Returns (diagnosis index, charged normalized cost, requests).

    memo caches the agent's decision at each history; the agent is a deterministic function of it.
    """
    memo = {} if memo is None else memo
    history, cost = (), 0.0
    while True:
        if history not in memo:
            belief = reader.belief(history)
            requested = {test for test, _ in history}
            forced = len(history) >= budget or all(test in requested for test in model.tests)
            diagnosis = commit(history, belief, forced)
            memo[history] = (diagnosis, None if diagnosis is not None else selector(history, belief))
        diagnosis, test = memo[history]
        if diagnosis is not None:
            return diagnosis, cost, len(history)
        outcome = outcome_of[test]
        if outcome != UNAVAILABLE:
            cost += float(model.cost_norm[model.tests.index(test)])
        history = history + ((test, outcome),)


def brute_value(joint: Joint, selector, reader, commit, budget, lam) -> dict:
    """Expected accuracy, cost, requests and utility by playing the agent on every latent case."""
    accuracy = cost = requests = 0.0
    memo = {}
    for outcome_of, p_y in joint.outcome_tuples():
        diagnosis, charged, n = play(joint.model, outcome_of, selector, reader, commit, budget, memo)
        accuracy += p_y[diagnosis]
        cost += charged * p_y.sum()
        requests += n * p_y.sum()
    return {"accuracy": accuracy, "cost": cost, "requests": requests, "utility": accuracy - lam * cost}


def expectimax(joint: Joint, history, budget, lam) -> float:
    """Best achievable expected utility from history: plain recursion over ordered histories."""
    best = float(joint.posterior(history).max())
    requested = {test for test, _ in history}
    if len(history) >= budget:
        return best
    for t, test in enumerate(joint.model.tests):
        if test in requested:
            continue
        value = 0.0
        for outcome, p in joint.predictive(history, test).items():
            if p > 0:
                charge = 0.0 if outcome == UNAVAILABLE else lam * float(joint.model.cost_norm[t])
                value += p * (expectimax(joint, history + ((test, outcome),), budget, lam) - charge)
        best = max(best, value)
    return best


def sample_histories(twin, n=25, seed=5):
    """Histories that can occur: the empty one and random prefixes of random test orders on sampled patients."""
    rng = np.random.default_rng(seed)
    tests = twin.model.tests
    histories = [()]
    for patient in twin.sample_patients(n, seed=seed, split="test"):
        order = rng.permutation(len(tests))
        k = int(rng.integers(1, len(tests) + 1))
        histories.append(tuple((tests[t], twin.observe(patient, tests[t])) for t in order[:k]))
    return histories


def kl(p, q) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(sum(pi * np.log(pi / qi) for pi, qi in zip(p, q) if pi > 0))


# ---------------------------------------------------------------------------------------------
# The references checked by hand and against a loop over every atom
# ---------------------------------------------------------------------------------------------
def test_joint_matches_hand_arithmetic_on_the_toy_model():
    joint = Joint(toy_model())
    assert joint.p.sum() == pytest.approx(1.0, abs=1e-15)
    # P(t0 = pos) = 0.6 (0.8*0.9 + 0.2*0.2) + 0.4 (0.3*0.9 + 0.7*0.2) = 0.456 + 0.164
    assert joint.predictive((), "t0")["pos"] == pytest.approx(0.62, abs=1e-15)
    assert joint.posterior((("t0", "pos"),))[0] == pytest.approx(0.456 / 0.62, abs=1e-15)
    # t1 is in the chart with probability 0.3 whatever the class, so its absence says nothing
    assert joint.predictive((), "t1")[UNAVAILABLE] == pytest.approx(0.7, abs=1e-15)
    assert joint.posterior((("t1", UNAVAILABLE),)) == pytest.approx([0.6, 0.4], abs=1e-15)
    # t2 is never in the chart
    assert joint.predictive((), "t2") == pytest.approx({"pos": 0.0, "neg": 0.0, UNAVAILABLE: 1.0})


def test_joint_matches_a_loop_over_every_atom():
    model = toy_model(availability=[[1.0, 1.0], [0.3, 0.6], [0.2, 0.2]])
    joint = Joint(model)
    outcomes = ("pos", "neg", UNAVAILABLE)
    for y, e, o0, o1, o2 in itertools.product(range(2), range(2), range(3), range(3), range(3)):
        p = model.prior[y] * model.phenotype_given_class[y, e]
        for i, o in enumerate((o0, o1, o2)):
            a = model.availability[i, y]
            p *= (1 - a) if outcomes[o] == UNAVAILABLE else a * model.finding_lik[i, y, e, o]
        assert joint.p[y, e, o0, o1, o2] == pytest.approx(p, abs=1e-15)


def test_joint_copies_a_repeated_measurement():
    base = toy_model(availability=(1.0, 0.3, 0.5))
    repeated = TwinModel(
        name="toy-dup", variant="toy", held_out=False, classes=base.classes, phenotypes=base.phenotypes,
        findings=base.findings, tests=base.tests + ("t1 again",), items=base.items, item_of=(0, 1, 2, 1),
        prior=base.prior, phenotype_given_class=base.phenotype_given_class, finding_lik=base.finding_lik,
        availability=base.availability, cost_usd=np.append(base.cost_usd, 250.0),
        cost_norm=np.append(base.cost_norm, 0.625), step_limit=5, source="toy",
    )
    joint = Joint(repeated)
    assert joint.p.sum() == pytest.approx(1.0, abs=1e-15)
    assert joint.predictive((("t1", "neg"),), "t1 again") == pytest.approx({"pos": 0.0, "neg": 1.0, UNAVAILABLE: 0.0})
    assert joint.predictive((), "t1 again") == pytest.approx(Joint(base).predictive((), "t1"))
