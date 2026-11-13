"""The counterfactual environment of specification section 16, over a TwinModel.

A history is a tuple of (test, outcome) pairs in the order they were read. An outcome is a finding
or UNAVAILABLE. Every exact quantity is a function of the model and the history alone: nothing here
reads a patient's unrevealed results except all_outcomes, observe and run_episode, which are the
environment's side of an episode.

Conventions the rest of the study relies on:
- A request returns UNAVAILABLE when the item is not in the chart. That result is free and uses a
  step (REVIEW-006), so the expected charged cost of a request is p_available(test | history) x cost.
- Each test may be requested once. A test that reads an item already in the history is still
  eligible when it is a different test (the duplicated panel); its result is then certain.
- Utility is 1[correct] - lam x charged normalized cost, as in the existing study. A budget is the
  largest number of requests before the agent must commit.
- An agent is three callables. reader.belief(history) gives the belief; commit(history, belief,
  forced) returns the index of a class to commit to, or None to continue (not allowed when forced);
  selector(history, belief) returns an eligible test. They receive nothing else.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from tiicee.interfaces import UNAVAILABLE, Predictive
from tiicee.twin import source
from tiicee.twin.model import TwinModel

PURPOSES = ("generation", "behavior", "fitting", "corruption", "montecarlo")
SPLITS = ("train", "validation", "test")
TEMPLATES = ("structured", "narrative", "clinical")
_LARGEST_TABLE = 20_000_000       # entries in the largest array of the dynamic program


def stream(seed: int, purpose: str, split: str, *key: int) -> np.random.Generator:
    """The random stream of one (user seed, purpose, split), optionally of one numbered unit within it.

    Streams with different purposes or splits are independent under the same user seed (REVIEW-002).
    """
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}; known: {PURPOSES}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; known: {SPLITS}")
    spawn_key = (PURPOSES.index(purpose), SPLITS.index(split)) + tuple(int(k) for k in key)
    return np.random.default_rng(np.random.SeedSequence(entropy=int(seed), spawn_key=spawn_key))


class ImpossibleHistory(ValueError):
    """The history contradicts itself or has probability zero under the model."""


@dataclass(frozen=True)
class Patient:
    """One latent case. findings holds every test's finding, whether or not the test is in the chart."""

    display_id: str
    diagnosis: str
    phenotype: str
    tests: tuple
    findings: tuple
    available: tuple

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the latent content. The display id is not part of it."""
        content = {
            "diagnosis": self.diagnosis,
            "phenotype": self.phenotype,
            "results": [[t, f, bool(a)] for t, f, a in zip(self.tests, self.findings, self.available)],
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Leaf:
    """A history at which the agent commits, reached with the given probability."""

    history: tuple
    probability: float
    posterior: np.ndarray       # the true posterior at this history
    diagnosis: int              # index of the committed class
    cost: float                 # charged normalized cost along the history
    cost_usd: float


@dataclass(frozen=True)
class EpisodeValue:
    utility: float              # accuracy - lam x cost
    accuracy: float
    cost: float                 # expected charged normalized cost
    cost_usd: float
    requests: float             # expected number of requests, unavailable ones included


@dataclass(frozen=True)
class Episode:
    history: tuple
    diagnosis: str
    correct: bool
    cost: float
    cost_usd: float
    requests: int
    utility: float


class _NoDistractor:
    """Stands in for the snapshot's random source, so its text wrapper adds no distractor and no overlap note."""

    def choice(self, options):
        return ""

    def random(self):
        return 1.0


def _draw(probabilities: np.ndarray, u: float) -> int:
    """Inverse-CDF draw that can never land on an outcome of probability zero."""
    support = np.flatnonzero(probabilities > 0)
    k = int(np.searchsorted(np.cumsum(probabilities[support]), u, side="right"))
    return int(support[min(k, len(support) - 1)])


class Twin:
    def __init__(self, model: TwinModel):
        self.model = model
        self._test = {name: t for t, name in enumerate(model.tests)}
        self._outcome = {name: o for o, name in enumerate(model.outcomes)}
        self._lik = model.outcome_lik()                                     # [item, class, phenotype, outcome]
        self._joint = model.prior[:, None] * model.phenotype_given_class    # [class, phenotype]
        self._absent = len(model.findings)

    # ------------------------------------------------------------------------------ patients
    def sample_patients(self, n: int, seed: int, split: str, purpose: str = "generation") -> list:
        """Cases 0..n-1 of the (seed, purpose, split) stream. Case i is the same whatever n is."""
        m = self.model
        patients = []
        for index in range(n):
            u = stream(seed, purpose, split, index).random(2 + 2 * len(m.items))
            y = _draw(m.prior, u[0])
            e = _draw(m.phenotype_given_class[y], u[1])
            finding = [_draw(m.finding_lik[i, y, e], u[2 + 2 * i]) for i in range(len(m.items))]
            present = [bool(u[3 + 2 * i] < m.availability[i, y]) for i in range(len(m.items))]
            patients.append(Patient(
                display_id=f"{purpose}-{split}-s{seed}-{index:06d}",
                diagnosis=m.classes[y],
                phenotype=m.phenotypes[e],
                tests=m.tests,
                findings=tuple(m.findings[finding[i]] for i in m.item_of),
                available=tuple(present[i] for i in m.item_of),
            ))
        return patients

    def all_outcomes(self, patient: Patient) -> dict:
        """Every test's (finding, available) for this patient: the counterfactual view."""
        if patient.tests != self.model.tests:
            raise ValueError("patient was generated by a model with different tests")
        return {t: (f, bool(a)) for t, f, a in zip(patient.tests, patient.findings, patient.available)}

    def observe(self, patient: Patient, test: str):
        """What a request for this test returns."""
        finding, available = self.all_outcomes(patient)[test]
        return finding if available else UNAVAILABLE

    # ------------------------------------------------------------------------------ exact inference
    def index(self, test, outcome=None):
        """Position of a test, or of a (test, outcome) pair, in the model's lists. ValueError for an unknown name."""
        if test not in self._test:
            raise ValueError(f"{test!r} is not a test of {self.model.name}")
        if outcome is None:
            return self._test[test]
        if outcome not in self._outcome:
            raise ValueError(f"{outcome!r} is not an outcome of {self.model.name}")
        return self._test[test], self._outcome[outcome]

    def _latent(self, history):
        """P(class, phenotype, history) and the outcome index recorded for each item read."""
        weights, seen = self._joint, {}
        for test, outcome in history:
            t, o = self.index(test, outcome)
            item = self.model.item_of[t]
            if item in seen:
                if seen[item] != o:
                    raise ImpossibleHistory(f"history contradicts itself on {self.model.items[item]!r}")
                continue
            seen[item] = o
            weights = weights * self._lik[item, :, :, o]
        if not weights.sum() > 0:
            raise ImpossibleHistory("history has probability zero under the model")
        return weights, seen

    def _joint_with(self, history, test):
        """P(class, outcome of test, history) as [class, outcome], and the outcome index if it is already known."""
        t = self.index(test)
        weights, seen = self._latent(history)
        item = self.model.item_of[t]
        if item in seen:                                    # evidence identity: the result is already in the history
            joint = np.zeros((len(self.model.classes), len(self.model.outcomes)))
            joint[:, seen[item]] = weights.sum(1)
            return joint, seen[item]
        return np.einsum("ye,yeo->yo", weights, self._lik[item]), None

    def true_posterior(self, history) -> np.ndarray:
        weights, _ = self._latent(history)
        by_class = weights.sum(1)
        return by_class / by_class.sum()

    def true_predictive(self, history, test) -> Predictive:
        """P*(o | h, a) over the findings and UNAVAILABLE; outcomes of probability zero are left out."""
        joint, known = self._joint_with(history, test)
        if known is not None:
            return Predictive((self.model.outcomes[known],), np.array([1.0]))
        p = joint.sum(0)
        keep = p > 0
        return Predictive(tuple(o for o, k in zip(self.model.outcomes, keep) if k), p[keep] / p[keep].sum())

    def class_predictive(self, history, test):
        """(outcomes, P(o | class, h, a) as [class, outcome]) for the outcomes possible at this history.

        A class of posterior probability zero has no conditional; its row is the pooled predictive.
        """
        joint, known = self._joint_with(history, test)
        if known is not None:
            return (self.model.outcomes[known],), np.ones((len(self.model.classes), 1))
        keep = joint.sum(0) > 0
        by_class = joint.sum(1, keepdims=True)
        pooled = np.broadcast_to(joint.sum(0) / joint.sum(), joint.shape)
        lik = np.where(by_class > 0, joint / np.where(by_class > 0, by_class, 1.0), pooled)
        return tuple(o for o, k in zip(self.model.outcomes, keep) if k), lik[:, keep]

    def true_information(self, history, test) -> float:
        """I(Y; O | h, a) in nats. Zero for a result that is already in the history."""
        joint, known = self._joint_with(history, test)
        if known is not None:
            return 0.0
        joint = joint / joint.sum()
        independent = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
        positive = joint > 0
        return float((joint[positive] * np.log(joint[positive] / independent[positive])).sum())

    def p_available(self, history, test) -> float:
        joint, _ = self._joint_with(history, test)
        found, missing = joint[:, :self._absent].sum(), joint[:, self._absent].sum()
        return float(found / (found + missing))

    def charged_cost(self, test, outcome, usd: bool = False) -> float:
        """What a returned outcome costs: the test's cost for a finding, nothing for UNAVAILABLE."""
        t, o = self.index(test, outcome)
        cost = self.model.cost_usd if usd else self.model.cost_norm
        return 0.0 if o == self._absent else float(cost[t])

    def expected_charged_cost(self, history, test, usd: bool = False) -> float:
        """p_available(test | history) x cost: unavailable results are free."""
        cost = self.model.cost_usd if usd else self.model.cost_norm
        return self.p_available(history, test) * float(cost[self.index(test)])

    def eligible(self, history) -> tuple:
        """Tests not yet requested, in the model's order."""
        requested = {test for test, _ in history}
        return tuple(test for test in self.model.tests if test not in requested)

    # ------------------------------------------------------------------------------ text
    def render(self, history, template: str) -> str:
        """The history as text for a language-model reader. Deterministic; three templates.

        "clinical" uses the finding text of the snapshot's generators, without their random
        distractor sentence and overlap note, so the text carries the finding and nothing else.
        """
        if template not in TEMPLATES:
            raise ValueError(f"unknown template {template!r}; known: {TEMPLATES}")
        m = self.model
        steps = [(test, outcome, m.items[m.item_of[self.index(test, outcome)[0]]]) for test, outcome in history]
        if template == "structured":
            lines = ["Presenting complaint: acute abdominal pain", "Candidate diagnoses: " + "; ".join(m.classes), "Results:"]
            lines += [f"{n}. {test}: {outcome}" for n, (test, outcome, _) in enumerate(steps, 1)] or ["(none yet)"]
            return "\n".join(lines)
        if template == "narrative":
            text = ("A patient presents with acute abdominal pain. The diagnoses under consideration are "
                    + ", ".join(m.classes[:-1]) + " and " + m.classes[-1] + ".")
            if not steps:
                return text + " No test has been requested yet."
            for n, (test, outcome, _) in enumerate(steps):
                lead = "First" if n == 0 else "Next"
                if outcome == UNAVAILABLE:
                    text += f" {lead}, the {test} was requested, but it had not been performed for this patient and no result is available."
                elif outcome in m.classes:
                    text += f" {lead}, the {test} was requested and its result pointed to {outcome}."
                else:
                    text += f" {lead}, the {test} was requested and its result was {outcome}."
            return text
        lines = ["Chief complaint: acute abdominal pain. Differential under consideration: " + ", ".join(m.classes) + "."]
        for test, outcome, item in steps:
            if outcome == UNAVAILABLE:
                observation = f"[{test}] UNAVAILABLE: this test was not performed for this patient."
            else:
                observation = source.generators.render_finding(item, outcome, _NoDistractor())
            lines += [f"Action: {test}", f"Observation: {observation}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------------------ episodes
    def _budget(self, budget: Optional[int]) -> int:
        if budget is None:
            return min(len(self.model.tests), self.model.step_limit - 1)
        if int(budget) != budget or budget < 0:
            raise ValueError("budget is a non-negative number of requests")
        return int(budget)

    def _decide(self, history, selector, reader, commit, budget):
        belief = reader.belief(history)
        eligible = self.eligible(history)
        forced = len(history) >= budget or not eligible
        diagnosis = commit(history, belief, forced)
        if diagnosis is not None:
            if not 0 <= int(diagnosis) < len(self.model.classes):
                raise ValueError(f"commit returned {diagnosis!r}, which is not a class index")
            return int(diagnosis), None
        if forced:
            raise ValueError("commit returned no diagnosis at a forced decision")
        test = selector(history, belief)
        if test not in eligible:
            raise ValueError(f"selector chose {test!r}, which is not an eligible test at this history")
        return None, test

    def reachable(self, selector, reader, commit, budget: Optional[int] = None) -> Iterator[Leaf]:
        """Every history at which the agent commits, with its exact probability."""
        m = self.model
        budget = self._budget(budget)
        stack = [((), self._joint, {}, 0.0, 0.0)]
        while stack:
            history, weights, seen, cost, cost_usd = stack.pop()
            diagnosis, test = self._decide(history, selector, reader, commit, budget)
            if diagnosis is not None:
                mass = weights.sum()
                yield Leaf(history, float(mass), weights.sum(1) / mass, diagnosis, cost, cost_usd)
                continue
            t = self._test[test]
            item = m.item_of[t]
            outcomes = [seen[item]] if item in seen else range(len(m.outcomes))
            for o in outcomes:
                child = weights if item in seen else weights * self._lik[item, :, :, o]
                if child.sum() > 0:
                    charged = o != self._absent
                    stack.append((
                        history + ((test, m.outcomes[o]),), child, {**seen, item: o},
                        cost + (float(m.cost_norm[t]) if charged else 0.0),
                        cost_usd + (float(m.cost_usd[t]) if charged else 0.0),
                    ))

    def exact_value(self, selector, reader, commit, budget: Optional[int] = None, lam: float = 1.0) -> EpisodeValue:
        """V(S, R, C): the agent's exact expected utility, by enumeration of every reachable history."""
        accuracy = cost = cost_usd = requests = 0.0
        for leaf in self.reachable(selector, reader, commit, budget):
            accuracy += leaf.probability * leaf.posterior[leaf.diagnosis]
            cost += leaf.probability * leaf.cost
            cost_usd += leaf.probability * leaf.cost_usd
            requests += leaf.probability * len(leaf.history)
        return EpisodeValue(float(accuracy - lam * cost), float(accuracy), cost, cost_usd, requests)

    def run_episode(self, patient: Patient, selector, reader, commit, budget: Optional[int] = None, lam: float = 1.0) -> Episode:
        """One realized episode. The agent's callables see the history and the belief, never the patient."""
        m = self.model
        budget = self._budget(budget)
        history, cost, cost_usd = (), 0.0, 0.0
        while True:
            diagnosis, test = self._decide(history, selector, reader, commit, budget)
            if diagnosis is not None:
                correct = m.classes[diagnosis] == patient.diagnosis
                return Episode(history, m.classes[diagnosis], correct, cost, cost_usd, len(history), float(correct) - lam * cost)
            outcome = self.observe(patient, test)
            cost += self.charged_cost(test, outcome)
            cost_usd += self.charged_cost(test, outcome, usd=True)
            history = history + ((test, outcome),)

    # ------------------------------------------------------------------------------ Bayes-optimal reference
    def _subset_weights(self, subset) -> np.ndarray:
        """P(class, phenotype, outcomes of the tests in subset): shape (outcome,) * len(subset) + (class, phenotype)."""
        m = self.model
        k, n_out = len(subset), len(m.outcomes)
        weights = self._joint.reshape((1,) * k + self._joint.shape)
        axis_of = {}
        for axis, t in enumerate(subset):
            item = m.item_of[t]
            shape = [1] * k
            shape[axis] = n_out
            if item in axis_of:                              # a second test of the same item copies the first
                shape[axis_of[item]] = n_out
                weights = weights * np.eye(n_out).reshape(shape + [1, 1])
            else:
                axis_of[item] = axis
                weights = weights * np.moveaxis(self._lik[item], -1, 0).reshape(shape + list(self._joint.shape))
        return weights

    def bayes_optimal(self, budget: Optional[int] = None, lam: float = 1.0) -> "BayesOptimal":
        """The best achievable expected utility with the same observation access and budget.

        Dynamic programming over what has been observed. The policy searches the known model; it
        never sees an unrevealed outcome, so it is a reference and not a clairvoyant bound.
        """
        m = self.model
        n_tests, n_out = len(m.tests), len(m.outcomes)
        depth = min(self._budget(budget), n_tests)
        if n_out ** depth * self._joint.size > _LARGEST_TABLE:
            raise ValueError(
                f"{n_tests} tests with budget {depth} are too many for exact dynamic programming; use the reduced twin")
        value, mass, stop, best = {}, {}, {}, {}
        for size in range(depth, -1, -1):
            for subset in itertools.combinations(range(n_tests), size):
                weights = self._subset_weights(subset)
                mass[subset] = weights.sum((-1, -2))                    # P(history)
                commit = weights.sum(-1).max(-1)                        # P(history) x largest posterior probability
                q_best = np.full(commit.shape, -np.inf)
                choice = np.full(commit.shape, -1)
                if size < depth:
                    for t in range(n_tests):
                        if t in subset:
                            continue
                        child = tuple(sorted(subset + (t,)))
                        axis = child.index(t)
                        charged = mass[child].sum(axis) - np.take(mass[child], self._absent, axis=axis)
                        q = value[child].sum(axis) - lam * m.cost_norm[t] * charged
                        better = q > q_best
                        q_best = np.where(better, q, q_best)
                        choice = np.where(better, t, choice)
                stop[subset] = commit >= q_best                         # ties stop
                value[subset] = np.maximum(commit, q_best)
                best[subset] = choice
        return BayesOptimal(float(value[()]), depth, float(lam), self, stop, best)

    def bayes_optimal_value(self, budget: Optional[int] = None, lam: float = 1.0) -> float:
        return self.bayes_optimal(budget, lam).value


@dataclass(frozen=True)
class BayesOptimal:
    """The Bayes-optimal policy: when to stop and which test to request, as functions of the history."""

    value: float
    budget: int
    lam: float
    _twin: Twin = field(repr=False)
    _stop: dict = field(repr=False)
    _best: dict = field(repr=False)

    def _locate(self, history):
        indexed = sorted(self._twin.index(test, outcome) for test, outcome in history)
        subset = tuple(t for t, _ in indexed)
        if len(set(subset)) != len(subset):
            raise ValueError("a test appears twice in the history")
        if subset not in self._stop:
            raise ValueError("the history is longer than the budget this policy was computed for")
        return subset, tuple(o for _, o in indexed)

    def should_stop(self, history) -> bool:
        """True when committing now is at least as good as any further request."""
        subset, index = self._locate(history)
        return bool(self._stop[subset][index])

    def best_test(self, history) -> Optional[str]:
        """The request with the highest value if the agent continues; None when no request is allowed."""
        subset, index = self._locate(history)
        t = int(self._best[subset][index])
        return None if t < 0 else self._twin.model.tests[t]
