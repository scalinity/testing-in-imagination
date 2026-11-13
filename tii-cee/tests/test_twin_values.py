"""Exact episode values and the Bayes-optimal reference, against enumeration of every latent case."""
from __future__ import annotations

import numpy as np
import pytest

from tiicee.interfaces import UNAVAILABLE
from tiicee.twin import model as tm
from tiicee.twin.env import Twin
from tiicee.twin.readers import BayesReader, SimulatedReader

from test_twin_support import Joint, brute_value, expectimax, play, toy_model


# --- simple agents, defined here so that the references do not lean on library code -----------------
def fixed_order(order):
    def selector(history, belief):
        requested = {test for test, _ in history}
        return next(test for test in order if test not in requested)
    return selector


def most_informative(twin):
    def selector(history, belief):
        return max(twin.eligible(history), key=lambda test: twin.true_information(history, test))
    return selector


def threshold(tau):
    def commit(history, belief, forced):
        return int(np.argmax(belief)) if forced or belief.max() >= tau else None
    return commit


def at_the_limit(history, belief, forced):
    return int(np.argmax(belief)) if forced else None


def check_against_enumeration(twin, joint, selector, reader, commit, budget, lam):
    value = twin.exact_value(selector, reader, commit, budget=budget, lam=lam)
    reference = brute_value(joint, selector, reader, commit, budget, lam)
    assert value.accuracy == pytest.approx(reference["accuracy"], abs=1e-12)
    assert value.cost == pytest.approx(reference["cost"], abs=1e-12)
    assert value.requests == pytest.approx(reference["requests"], abs=1e-12)
    assert value.utility == pytest.approx(reference["utility"], abs=1e-12)
    assert value.utility == pytest.approx(value.accuracy - lam * value.cost, abs=1e-15)
    return value


# --- exact_value --------------------------------------------------------------------------------------
@pytest.mark.parametrize("budget", [0, 1, 2, 3])
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_exact_value_on_the_toy_model(budget, lam):
    model = toy_model(availability=[[1.0, 1.0], [0.3, 0.6], [0.5, 0.5]])
    twin, joint = Twin(model), Joint(model)
    agents = [
        (fixed_order(("t1", "t0", "t2")), BayesReader(twin), threshold(0.8)),
        (fixed_order(("t2", "t1", "t0")), SimulatedReader(twin, power=2.0), at_the_limit),
        (most_informative(twin), SimulatedReader(twin, drift="entrench"), threshold(0.7)),
    ]
    for selector, reader, commit in agents:
        check_against_enumeration(twin, joint, selector, reader, commit, budget, lam)


@pytest.mark.parametrize("variant", ["base", "duplicated_panel", "informative_missingness"])
def test_exact_value_on_the_reduced_twin(variant):
    model = tm.build(variant, reduced=True, allow_held_out=True)
    twin, joint = Twin(model), Joint(model)
    order = tuple(reversed(model.tests))               # the duplicate comes first, then its original
    check_against_enumeration(twin, joint, fixed_order(order), SimulatedReader(twin), threshold(0.9), 3, 1.0)
    check_against_enumeration(twin, joint, most_informative(twin), BayesReader(twin), at_the_limit, 2, 0.5)


def test_exact_value_reports_cost_in_both_units_and_uses_the_step_limit_by_default():
    twin = Twin(tm.build("base", reduced=True))
    one_test = twin.exact_value(fixed_order(("CT",)), BayesReader(twin), at_the_limit, budget=1)
    assert one_test.cost == pytest.approx(0.75 * 1306 / 10190, abs=1e-15)       # p_available x cost
    assert one_test.cost_usd == pytest.approx(0.75 * 1306, abs=1e-12)
    assert one_test.requests == pytest.approx(1.0)
    none = twin.exact_value(fixed_order(twin.model.tests), BayesReader(twin), at_the_limit, budget=0)
    assert none.accuracy == pytest.approx(957 / 2400) and none.cost == 0.0 and none.requests == 0.0
    # default budget: every test once, which the fourteen-action limit of the snapshot allows
    everything = twin.exact_value(fixed_order(twin.model.tests), BayesReader(twin), at_the_limit)
    assert everything.requests == pytest.approx(6.0)
    assert everything.cost_usd == pytest.approx(sum(
        twin.model.availability[t, 0] * twin.model.cost_usd[t] for t in range(6)), abs=1e-9)


def test_reachable_leaves_carry_their_probability_and_the_true_posterior():
    model = tm.build("duplicated_panel", reduced=True)
    twin, joint = Twin(model), Joint(model)
    leaves = list(twin.reachable(most_informative(twin), BayesReader(twin), threshold(0.9), budget=3))
    assert sum(leaf.probability for leaf in leaves) == pytest.approx(1.0, abs=1e-12)
    assert len({leaf.history for leaf in leaves}) == len(leaves)
    for leaf in leaves[:40]:
        assert leaf.probability == pytest.approx(joint.mass(leaf.history).sum(), abs=1e-14)
        np.testing.assert_allclose(leaf.posterior, joint.posterior(leaf.history), atol=1e-12)
        assert leaf.cost == pytest.approx(sum(twin.charged_cost(t, o) for t, o in leaf.history))
        assert len(leaf.history) <= 3


def test_agents_that_break_the_protocol_are_stopped():
    twin = Twin(tm.build("base", reduced=True))
    reader = BayesReader(twin)
    with pytest.raises(ValueError, match="eligible"):
        twin.exact_value(lambda h, b: "CT", reader, at_the_limit, budget=2)       # asks for CT twice
    with pytest.raises(ValueError, match="eligible"):
        twin.exact_value(lambda h, b: "MRI", reader, at_the_limit, budget=2)
    with pytest.raises(ValueError, match="commit"):
        twin.exact_value(fixed_order(twin.model.tests), reader, lambda h, b, forced: None, budget=1)


def test_run_episode_matches_a_hand_rolled_loop_and_hides_the_patient():
    model = tm.build("duplicated_panel", reduced=True)
    twin = Twin(model)
    reader, selector, commit = SimulatedReader(twin, drift="fixed"), fixed_order(tuple(reversed(model.tests))), threshold(0.85)
    for patient in twin.sample_patients(60, seed=9, split="test"):
        outcome_of = {test: twin.observe(patient, test) for test in model.tests}
        diagnosis, cost, requests = play(model, outcome_of, selector, reader, commit, budget=4)
        episode = twin.run_episode(patient, selector, reader, commit, budget=4)
        assert episode.diagnosis == model.classes[diagnosis]
        assert episode.correct == (episode.diagnosis == patient.diagnosis)
        assert episode.cost == pytest.approx(cost) and episode.requests == requests == len(episode.history)
        assert all(outcome == outcome_of[test] for test, outcome in episode.history)
        assert episode.utility == pytest.approx(float(episode.correct) - episode.cost)


# --- Bayes-optimal reference --------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.0])
def test_bayes_optimal_value_on_the_toy_model(lam):
    model = toy_model(availability=[[1.0, 1.0], [0.3, 0.6], [0.5, 0.5]])
    twin, joint = Twin(model), Joint(model)
    for budget in (0, 1, 2, 3):
        assert twin.bayes_optimal_value(budget=budget, lam=lam) == pytest.approx(expectimax(joint, (), budget, lam), abs=1e-12)


@pytest.mark.parametrize("variant,tests", [
    ("base", ("CT", "Ultrasound", "Liver Function Panel")),
    ("base", ("Physical Examination", "Complete Blood Count", "Urinalysis")),
    ("strong_correlation", ("CT", "Ultrasound", "Liver Function Panel")),
    ("informative_missingness", ("CT", "Ultrasound", "Liver Function Panel")),
    ("duplicated_panel", ("Ultrasound", "Liver Function Panel", "Liver Function Panel (repeat)")),
])
def test_bayes_optimal_value_on_three_test_twins_at_full_depth(variant, tests):
    model = tm.restrict(tm.build(variant, reduced=True, allow_held_out=True), tests)
    twin, joint = Twin(model), Joint(model)
    for lam in (0.25, 1.0):
        assert twin.bayes_optimal_value(lam=lam) == pytest.approx(expectimax(joint, (), 3, lam), abs=1e-12)


@pytest.mark.parametrize("variant", ["base", "duplicated_panel"])
def test_bayes_optimal_value_on_the_reduced_twin_at_shallow_depth(variant):
    model = tm.build(variant, reduced=True)
    twin, joint = Twin(model), Joint(model)
    for budget in (0, 1, 2):
        assert twin.bayes_optimal_value(budget=budget, lam=1.0) == pytest.approx(expectimax(joint, (), budget, 1.0), abs=1e-12)


def test_bayes_optimal_value_with_free_tests_is_the_value_of_seeing_everything():
    model = tm.build("base", reduced=True)
    twin, joint = Twin(model), Joint(model)
    see_everything = sum(p_y.max() for _, p_y in joint.outcome_tuples())
    assert twin.bayes_optimal_value(lam=0.0) == pytest.approx(see_everything, abs=1e-12)
    assert twin.bayes_optimal_value(budget=0, lam=1.0) == pytest.approx(957 / 2400)
    assert twin.bayes_optimal_value(lam=1e6) == pytest.approx(957 / 2400)       # no test is worth that price


def test_bayes_optimal_policy_reproduces_its_value_and_dominates_other_agents():
    twin = Twin(tm.build("base", reduced=True))
    reader = BayesReader(twin)
    values = [twin.bayes_optimal_value(budget=b, lam=1.0) for b in range(7)]
    assert all(later >= earlier - 1e-15 for earlier, later in zip(values, values[1:]))      # more budget never hurts
    optimal = twin.bayes_optimal(lam=1.0)
    assert optimal.value == values[6]

    def commit(history, belief, forced):
        return int(np.argmax(belief)) if forced or optimal.should_stop(history) else None

    walked = twin.exact_value(lambda h, b: optimal.best_test(h), reader, commit, lam=1.0)
    assert walked.utility == pytest.approx(optimal.value, abs=1e-12)
    for selector, rule in [
        (most_informative(twin), threshold(0.9)),
        (most_informative(twin), threshold(0.6)),
        (fixed_order(twin.model.tests), threshold(0.8)),
        (fixed_order(("CT", "Ultrasound", "Liver Function Panel", "Physical Examination", "Complete Blood Count", "Urinalysis")), threshold(0.95)),
    ]:
        assert twin.exact_value(selector, reader, rule, budget=3, lam=1.0).utility <= values[3] + 1e-12
    assert optimal.value < 1.0                       # exact search is not clairvoyance


def test_bayes_optimal_policy_is_a_function_of_the_revealed_history_only():
    twin = Twin(tm.build("base", reduced=True))
    optimal = twin.bayes_optimal(lam=1.0)
    first = optimal.best_test(())
    assert first in twin.model.tests and not optimal.should_stop(())
    other = "Ultrasound" if first != "Ultrasound" else "CT"
    a = ((first, "appendicitis"), (other, UNAVAILABLE))
    b = ((other, UNAVAILABLE), (first, "appendicitis"))
    assert optimal.best_test(a) == optimal.best_test(b) and optimal.should_stop(a) == optimal.should_stop(b)
    full = tuple((test, "nonspecific") for test in twin.model.tests)
    assert optimal.should_stop(full) and optimal.best_test(full) is None


def test_bayes_optimal_policy_stops_on_a_tie():
    """A test that is never in the chart is free and says nothing; requesting it is worth exactly as much as stopping."""
    twin = Twin(tm.restrict(toy_model(availability=(1.0, 0.3, 0.0)), ("t2",)))
    optimal = twin.bayes_optimal(lam=1.0)
    assert optimal.value == 0.6 and optimal.should_stop(()) and optimal.best_test(()) == "t2"


def test_exact_dynamic_programming_refuses_the_twelve_test_twin():
    with pytest.raises(ValueError, match="reduced"):
        Twin(tm.build("base", reduced=False)).bayes_optimal_value()
