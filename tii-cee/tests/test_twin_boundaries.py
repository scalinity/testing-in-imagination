"""Information boundaries of specification section 8.4, as metamorphic tests on the twin.

Observations, fitted artifacts, and random streams are held fixed; only what the agent has not seen changes.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tiicee.interfaces import UNAVAILABLE
from tiicee.twin import model as tm
from tiicee.twin import world_models as wm
from tiicee.twin.env import Twin
from tiicee.twin.readers import BayesReader, r4_family


@pytest.fixture(scope="module")
def twin():
    return Twin(tm.build("duplicated_panel", reduced=True))


def imagining_agent(twin, reader, world_model, log):
    """Selects by imagined results: expected largest posterior probability after each candidate, net of cost."""
    def selector(history, belief):
        scores = {}
        for test in twin.eligible(history):
            predictive = world_model.outcomes(history, test, None, None)
            imagined = [reader.read(history, test, outcome) for outcome in predictive.outcomes]
            log.append((history, test, predictive.outcomes))
            scores[test] = float(predictive.weights @ [q.max() for q in imagined]) - twin.expected_charged_cost(history, test)
        return max(scores, key=scores.get)

    def commit(history, belief, forced):
        return int(np.argmax(belief)) if forced or belief.max() >= 0.9 else None

    return selector, commit


def arms(twin):
    w1 = wm.fit_w1(twin, seed=0, n_train=500)
    readers = dict(r4_family(twin), r0=BayesReader(twin))
    return [(name, reader, world_model) for name, reader in readers.items() for world_model in (wm.w0(twin), w1)]


def test_unrevealed_results_and_the_hidden_diagnosis_cannot_change_beliefs_or_actions(twin):
    model = twin.model
    patients = twin.sample_patients(6, seed=21, split="test")
    for name, reader, world_model in arms(twin):
        selector, commit = imagining_agent(twin, reader, world_model, [])
        for patient in patients:
            episode = twin.run_episode(patient, selector, reader, commit, budget=4)
            beliefs = [reader.belief(episode.history[:k]) for k in range(len(episode.history) + 1)]
            revealed = {test for test, _ in episode.history}
            revealed_items = {model.item_of[model.tests.index(test)] for test in revealed}
            hidden = [t for t, test in enumerate(model.tests) if model.item_of[t] not in revealed_items]

            # a different hidden diagnosis and phenotype
            other = dataclasses.replace(
                patient,
                diagnosis=next(c for c in model.classes if c != patient.diagnosis),
                phenotype=next(c for c in model.phenotypes if c != patient.phenotype),
            )
            # every unrevealed result changed, and its availability flipped where that is possible
            findings, available = list(patient.findings), list(patient.available)
            for t in hidden:
                findings[t] = "nonspecific" if findings[t] != "nonspecific" else "pancreatitis"
                if model.tests[t] != "Physical Examination":
                    available[t] = not available[t]
            unseen = dataclasses.replace(patient, findings=tuple(findings), available=tuple(available))

            for changed in (other, unseen):
                assert changed.fingerprint != patient.fingerprint or not hidden
                replay = twin.run_episode(changed, selector, reader, commit, budget=4)
                assert replay.history == episode.history, name
                assert replay.diagnosis == episode.diagnosis and replay.cost == episode.cost, name
                for k, belief in enumerate(beliefs):
                    np.testing.assert_array_equal(reader.belief(replay.history[:k]), belief)
            # only the hidden label moved, so only correctness may differ
            assert twin.run_episode(other, selector, reader, commit, budget=4).correct == (episode.diagnosis == other.diagnosis)


def test_imagined_outcomes_are_never_written_into_the_real_history(twin):
    patients = twin.sample_patients(5, seed=22, split="test")
    for name, reader, world_model in arms(twin):
        log = []
        selector, commit = imagining_agent(twin, reader, world_model, log)
        for patient in patients:
            episode = twin.run_episode(patient, selector, reader, commit, budget=4)
            real = tuple((test, twin.observe(patient, test)) for test, _ in episode.history)
            assert episode.history == real, name                    # one real observation per request, in order
            assert len({test for test, _ in episode.history}) == len(episode.history)
        assert log and all(len(outcomes) >= 1 for _, _, outcomes in log)
        imagined_more_than_observed = any(len(outcomes) > 1 for _, _, outcomes in log)
        assert imagined_more_than_observed


def test_reading_and_imagining_leave_the_history_and_later_beliefs_untouched(twin):
    history = (("CT", "appendicitis"), ("Liver Function Panel", UNAVAILABLE))
    frozen = tuple(history)
    for name, reader, world_model in arms(twin):
        before = reader.belief(history)
        for test in twin.model.tests:
            predictive = world_model.outcomes(history, test, None, None)
            world_model.outcomes(history, test, 8, np.random.default_rng(0))
            for outcome in twin.model.outcomes:
                reader.read(history, test, outcome)
            assert predictive.outcomes
        assert history == frozen
        np.testing.assert_array_equal(reader.belief(history), before)


def test_agents_are_given_the_history_and_the_belief_and_nothing_else(twin):
    seen = []

    def selector(*args, **kwargs):
        seen.append(("selector", args, kwargs))
        return twin.eligible(args[0])[0]

    def commit(*args, **kwargs):
        seen.append(("commit", args, kwargs))
        return 0 if args[2] else None

    reader = BayesReader(twin)
    patient = twin.sample_patients(1, seed=5, split="test")[0]
    twin.run_episode(patient, selector, reader, commit, budget=2)
    list(twin.reachable(selector, reader, commit, budget=1))
    for kind, args, kwargs in seen:
        assert not kwargs
        assert len(args) == (2 if kind == "selector" else 3)
        history, belief = args[0], args[1]
        assert isinstance(history, tuple) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in history)
        assert isinstance(belief, np.ndarray) and belief.shape == (4,)
        if kind == "commit":
            assert isinstance(args[2], bool)
        hash(history)


def test_references_never_see_unrevealed_outcomes(twin):
    """The Bayes-optimal policy and the true model are functions of the model and the history alone."""
    optimal = twin.bayes_optimal(budget=3, lam=1.0)
    reader = BayesReader(twin)

    def commit(history, belief, forced):
        return int(np.argmax(belief)) if forced or optimal.should_stop(history) else None

    def selector(history, belief):
        return optimal.best_test(history)

    for patient in twin.sample_patients(15, seed=23, split="test"):
        episode = twin.run_episode(patient, selector, reader, commit, budget=3)
        revealed = {test for test, _ in episode.history}
        findings = tuple(f if twin.model.tests[t] in revealed or twin.model.tests[t].startswith("Liver") else "nonspecific"
                         for t, f in enumerate(patient.findings))
        changed = dataclasses.replace(patient, findings=findings, diagnosis="pancreatitis")
        replay = twin.run_episode(changed, selector, reader, commit, budget=3)
        assert replay.history == episode.history and replay.diagnosis == episode.diagnosis
