"""The vectorized readers and world models of E1 against the twin's objects, history by history."""
from __future__ import annotations

import numpy as np
import pytest

from tiicee.experiments.e1 import filters
from tiicee.interfaces import UNAVAILABLE
from tiicee.twin import model as tm
from tiicee.twin import world_models as wm
from tiicee.twin.env import Twin
from tiicee.twin.readers import BayesReader, SimulatedReader

READERS = {"independence": {}, "entrench": {"drift": "entrench"}, "fixed": {"drift": "fixed"}, "over": {"power": 2.0},
           "under": {"power": 0.5}, "entrench+over": {"drift": "entrench", "power": 2.0},
           "fixed+under": {"drift": "fixed", "power": 0.5}}


def dense(twin, predictive):
    out = np.zeros(len(twin.model.outcomes))
    for outcome, weight in zip(predictive.outcomes, predictive.weights):
        out[twin.model.outcomes.index(outcome)] += weight
    return out


class Walk:
    """Random real histories of a twin, as index arrays and as the tuples the objects take."""

    def __init__(self, twin, n=40, steps=5, seed=3):
        m = twin.model
        rng = np.random.default_rng(seed)
        patients = twin.sample_patients(n, seed=seed, split="validation")
        self.twin, self.n, self.steps = twin, n, steps
        self.tests = np.stack([rng.permutation(len(m.tests))[:steps] for _ in range(n)])
        self.outcomes = np.array([[m.outcomes.index(twin.observe(p, m.tests[t])) for t in row] for p, row in zip(patients, self.tests)])
        self.item_of = np.array(m.item_of)

    def history(self, i, k):
        m = self.twin.model
        return tuple((m.tests[t], m.outcomes[o]) for t, o in zip(self.tests[i, :k], self.outcomes[i, :k]))

    def seen(self, k):
        seen = filters.new_seen(self.n, self.twin.model)
        for step in range(k):
            seen = filters.record(seen, self.item_of, self.tests[:, step], self.outcomes[:, step])
        return seen


@pytest.fixture(scope="module", params=["reduced/base", "reduced/duplicated_panel", "full/base"])
def walk(request):
    size, variant = request.param.split("/")
    return Walk(Twin(tm.build(variant, reduced=size == "reduced")))


def test_latent_filter_is_the_twins_exact_inference(walk):
    twin = walk.twin
    truth = filters.LatentFilter(twin.model)
    w = truth.start(walk.n)
    for k in range(walk.steps):
        seen, tests = walk.seen(k), walk.tests[:, k]
        joint, joint_all = truth.joint(w, seen, tests), truth.joint_all(w, seen)
        for i in range(walk.n):
            history = walk.history(i, k)
            np.testing.assert_allclose(w[i].sum(1) / w[i].sum(), twin.true_posterior(history), rtol=0, atol=1e-12)
            for t in ({int(tests[i])} | set(range(0, len(twin.model.tests), 3))):
                name = twin.model.tests[t]
                got = joint_all[i, t]
                np.testing.assert_allclose(filters.predictive(got), dense(twin, twin.true_predictive(history, name)), rtol=0, atol=1e-12)
            np.testing.assert_allclose(joint[i], joint_all[i, tests[i]], rtol=0, atol=1e-15)
        w = truth.advance(w, seen, tests, walk.outcomes[:, k], keep_impossible=False)


def test_information_of_the_joint_is_the_twins_true_information(walk):
    from tiicee.experiments.e1 import batch

    twin = walk.twin
    truth = filters.LatentFilter(twin.model)
    w = truth.start(walk.n)
    for k in range(walk.steps):
        seen = walk.seen(k)
        info = batch.information(truth.joint_all(w, seen).reshape(-1, len(twin.model.classes), len(twin.model.outcomes)))
        info = info.reshape(walk.n, -1)
        for i in range(0, walk.n, 4):
            for t, name in enumerate(twin.model.tests):
                assert info[i, t] == pytest.approx(twin.true_information(walk.history(i, k), name), abs=1e-12)
        w = truth.advance(w, seen, walk.tests[:, k], walk.outcomes[:, k], keep_impossible=False)


def test_expanding_a_level_gives_the_mass_and_the_belief_of_every_child():
    twin = Twin(tm.build("duplicated_panel", reduced=True))
    m = twin.model
    truth = filters.LatentFilter(m)
    lfp, repeat, pe = m.tests.index("Liver Function Panel"), m.tests.index("Liver Function Panel (repeat)"), 0
    w, seen = truth.start(1), filters.new_seen(1, m)
    first = np.array([lfp])
    w = truth.advance(w, seen, first, np.array([1]), keep_impossible=False)            # panel says cholecystitis
    seen = filters.record(seen, truth.item_of, first, np.array([1]))
    history = (("Liver Function Panel", "cholecystitis"),)
    for test in (repeat, pe):
        mass = truth.expand(w, seen, np.array([test]), keep_impossible=False)[0]
        belief = truth.expand(w, seen, np.array([test]), keep_impossible=True)[0]
        for o, outcome in enumerate(m.outcomes):
            extended = history + ((m.tests[test], outcome),)
            try:
                expected = twin.true_posterior(extended)
            except ValueError:                                      # the repeat disagrees, or the examination is missing
                assert mass[o].sum() == 0.0
                np.testing.assert_array_equal(belief[o], w[0])      # a belief ignores a result its model rules out
                continue
            np.testing.assert_allclose(mass[o].sum(1) / mass[o].sum(), expected, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(belief[o], mass[o])
        assert mass.sum() == pytest.approx(w.sum(), rel=1e-12)      # the children carry all of the parent's mass


@pytest.mark.parametrize("name", list(READERS))
def test_r4_reader_matches_the_simulated_reader(walk, name):
    twin = walk.twin
    reference = SimulatedReader(twin, **READERS[name])
    reader = filters.reader(twin.model, name)
    assert isinstance(reader, filters.R4Reader)
    state = reader.start(walk.n)
    for k in range(walk.steps):
        seen, tests = walk.seen(k), walk.tests[:, k]
        reactions, everything = reader.reactions(state, seen, tests), reader.reactions_all(state, seen)
        for i in range(0, walk.n, 3):
            history = walk.history(i, k)
            np.testing.assert_allclose(reader.belief(state)[i], reference.belief(history), rtol=0, atol=1e-12)
            for o, outcome in enumerate(twin.model.outcomes):
                expected = reference.read(history, twin.model.tests[tests[i]], outcome)
                np.testing.assert_allclose(reactions[i, o], expected, rtol=0, atol=1e-12)
                np.testing.assert_allclose(everything[i, tests[i], o], expected, rtol=0, atol=1e-12)
            other = (tests[i] + 1) % len(twin.model.tests)
            np.testing.assert_allclose(everything[i, other, 0], reference.read(history, twin.model.tests[other], twin.model.outcomes[0]), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(reader.expand(state, seen, tests), reactions)
        state = reader.advance(state, seen, tests, walk.outcomes[:, k])


def test_r0_reader_matches_the_bayes_reader(walk):
    twin = walk.twin
    reference = BayesReader(twin)
    reader = filters.reader(twin.model, "r0")
    state = reader.start(walk.n)
    for k in range(walk.steps):
        seen, tests = walk.seen(k), walk.tests[:, k]
        reactions, everything = reader.reactions(state, seen, tests), reader.reactions_all(state, seen)
        for i in range(0, walk.n, 3):
            history = walk.history(i, k)
            np.testing.assert_allclose(reader.belief(state)[i], reference.belief(history), rtol=0, atol=1e-12)
            for t in ({int(tests[i])} | set(range(len(twin.model.tests) - 2, len(twin.model.tests)))):
                for o, outcome in enumerate(twin.model.outcomes):    # includes results the true model rules out
                    expected = reference.read(history, twin.model.tests[t], outcome)
                    np.testing.assert_allclose(everything[i, t, o], expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(reactions[i], everything[i, tests[i]], rtol=0, atol=1e-15)
        state = reader.advance(state, seen, tests, walk.outcomes[:, k])


def test_count_filter_matches_w1(walk):
    twin = walk.twin
    w1 = wm.fit_w1(twin, seed=2, n_train=300)
    count = filters.CountFilter(w1)
    b = count.start(walk.n)
    from tiicee.experiments.e1 import batch

    for k in range(walk.steps):
        seen, tests = walk.seen(k), walk.tests[:, k]
        joint_all = count.joint_all(b, seen)
        budget = batch.information(count.joint(b, seen, tests))
        for i in range(0, walk.n, 3):
            history = walk.history(i, k)
            for t in (int(tests[i]), 0):
                np.testing.assert_allclose(filters.predictive(joint_all[i, t]), dense(twin, w1.outcomes(history, twin.model.tests[t], None, None)), rtol=0, atol=1e-12)
            assert budget[i] == pytest.approx(w1.budget(history, twin.model.tests[tests[i]]), abs=1e-12)
        b = count.advance(b, seen, tests, walk.outcomes[:, k])


def test_latent_filter_of_a_fitted_or_corrupted_model_matches_its_world_model(walk):
    from tiicee.experiments.e1 import batch

    twin = walk.twin
    for model in (wm.fit_w2(twin, seed=2, n_train=300), wm.mirror_corrupt(twin, 0.4)):
        latent = filters.LatentFilter(model.twin.model)
        w = latent.start(walk.n)
        for k in range(walk.steps):
            seen, tests = walk.seen(k), walk.tests[:, k]
            joint = latent.joint(w, seen, tests)
            budget = batch.information(joint)
            for i in range(0, walk.n, 3):
                history, name = walk.history(i, k), twin.model.tests[tests[i]]
                np.testing.assert_allclose(filters.predictive(joint[i]), dense(twin, model.outcomes(history, name, None, None)), rtol=0, atol=1e-12)
                assert budget[i] == pytest.approx(model.budget(history, name), abs=1e-12)
            w = latent.advance(w, seen, tests, walk.outcomes[:, k], keep_impossible=True)


def test_reshaped_predictives_match_the_controlled_errors(walk):
    twin = walk.twin
    truth = filters.LatentFilter(twin.model)
    w = truth.start(walk.n)
    readers = {name: (filters.reader(twin.model, name), ref) for name, ref in
               (("r0", BayesReader(twin)), ("independence", SimulatedReader(twin)), ("entrench+over", SimulatedReader(twin, drift="entrench", power=2.0)))}
    states = {name: r.start(walk.n) for name, (r, _) in readers.items()}
    slow_total = 0
    for k in range(walk.steps):
        seen, tests = walk.seen(k), walk.tests[:, k]
        p_star = filters.predictive(truth.joint(w, seen, tests))
        for temp in (0.5, 2.0):
            got = filters.temperature(p_star, temp)
            reference = wm.temperature(twin, temp)
            for i in range(0, walk.n, 3):
                np.testing.assert_allclose(got[i], dense(twin, reference.outcomes(walk.history(i, k), twin.model.tests[tests[i]], None, None)), rtol=0, atol=1e-13)
        for name, (reader, ref_reader) in readers.items():
            ignored, responsive, slow = filters.matched_pair(p_star, reader.reactions(states[name], seen, tests), 0.02)
            slow_total += slow
            ref_ign, ref_res = wm.matched_pair(twin, ref_reader, kl=0.02)
            for i in range(0, walk.n, 5):
                history, test = walk.history(i, k), twin.model.tests[tests[i]]
                np.testing.assert_allclose(ignored[i], dense(twin, ref_ign.outcomes(history, test, None, None)), rtol=0, atol=1e-9)
                np.testing.assert_allclose(responsive[i], dense(twin, ref_res.outcomes(history, test, None, None)), rtol=0, atol=1e-9)
            states[name] = reader.advance(states[name], seen, tests, walk.outcomes[:, k])
        w = truth.advance(w, seen, tests, walk.outcomes[:, k], keep_impossible=False)
    assert slow_total == 0                                      # every point is handled on arrays


def test_matched_pair_where_the_reader_cannot_tell_two_findings_apart():
    """Urinalysis: two findings are equally likely under every phenotype, so the reader ignores a plane of directions."""
    twin = Twin(tm.build("base", reduced=False))
    m = twin.model
    urinalysis = m.tests.index("Urinalysis")
    flat = m.finding_lik[urinalysis, 0]                         # [phenotype, finding]
    assert np.ptp(flat[:, 1]) == 0 and np.ptp(flat[:, 3]) == 0  # cholecystitis and pancreatitis findings say nothing
    reference = SimulatedReader(twin, drift="entrench")
    reader = filters.reader(m, "entrench")
    ref_ign, ref_res = wm.matched_pair(twin, reference, kl=0.02)
    truth = filters.LatentFilter(m)
    histories = [(), (("CT", "appendicitis"),), (("CT", "nonspecific"), ("Liver Function Panel", "cholecystitis"))]
    for history in histories:
        w, seen, state = truth.start(1), filters.new_seen(1, m), reader.start(1)
        for name, outcome in history:
            tests, outcomes = np.array([m.tests.index(name)]), np.array([m.outcomes.index(outcome)])
            w = truth.advance(w, seen, tests, outcomes, keep_impossible=False)
            state = reader.advance(state, seen, tests, outcomes)
            seen = filters.record(seen, truth.item_of, tests, outcomes)
        tests = np.array([urinalysis])
        p_star = filters.predictive(truth.joint(w, seen, tests))
        ignored, responsive, slow = filters.matched_pair(p_star, reader.reactions(state, seen, tests), 0.02)
        assert slow == 0
        np.testing.assert_allclose(ignored[0], dense(twin, ref_ign.outcomes(history, "Urinalysis", None, None)), rtol=0, atol=1e-9)
        np.testing.assert_allclose(responsive[0], dense(twin, ref_res.outcomes(history, "Urinalysis", None, None)), rtol=0, atol=1e-9)


def test_matched_pair_general_route_agrees_with_the_fast_one():
    rng = np.random.default_rng(0)
    p_star = np.concatenate([rng.dirichlet(np.ones(5), size=12) * 0.7, np.full((12, 1), 0.3)], axis=1)
    posteriors = rng.dirichlet(np.ones(4), size=(12, 6))
    fast = filters.matched_pair(p_star, posteriors, 0.02)
    assert fast[2] == 0
    for i in range(12):                                         # the same points through tiicee.twin's own routine
        p = p_star[i, :5] / 0.7
        for member, direction in zip(fast[:2], wm._reader_axes(posteriors[i, :5], p)):
            expected = 0.7 * filters._at_divergence(p[None], direction[None], np.array([0.02 / 0.7]))[0]
            np.testing.assert_allclose(member[i, :5], expected, rtol=0, atol=1e-10)
            assert member[i, 5] == 0.3
    flat = np.repeat(rng.dirichlet(np.ones(4), size=(12, 1)), 6, axis=1)             # readers that do not respond
    unmoved = filters.matched_pair(p_star, flat, 0.02)
    assert unmoved[2] == 0 and np.isfinite(unmoved[0]).all()
    np.testing.assert_array_equal(unmoved[0], unmoved[1])       # nothing to respond to: both members take an ignored direction
    for i in range(12):
        p = p_star[i, :5] / 0.7
        direction = wm._reader_axes(flat[i, :5], p)[0]
        expected = 0.7 * filters._at_divergence(p[None], direction[None], np.array([0.02 / 0.7]))[0]
        np.testing.assert_allclose(unmoved[0][i, :5], expected, rtol=0, atol=1e-10)
    partial = p_star.copy()                                     # a finding of probability zero goes point by point
    partial[:, 0] = 0.0
    six = rng.dirichlet(np.ones(3), size=(12, 6))               # three classes: four findings still leave a direction to ignore
    assert filters.matched_pair(partial, six, 0.02)[2] == 12
    holes = p_star.copy()
    holes[:, 0] = 0.0                                           # four possible findings, four classes
    with pytest.raises(ValueError, match="more findings than classes"):
        filters.matched_pair(holes, posteriors, 0.02)           # the twin's routine says so; nothing is invented


def test_reader_names_are_the_cards():
    model = tm.build("base", reduced=True)
    for name in ("r0", "independence", "entrench", "fixed", "over", "under", "entrench+over", "fixed+under"):
        assert filters.reader(model, name).name == name
    with pytest.raises(KeyError):
        filters.reader(model, "fixed+over")
    assert UNAVAILABLE == model.outcomes[-1]
