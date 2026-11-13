"""The exact layer of E1 on the reduced twin against a path-by-path recursion through the library's own functions."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from tiicee import cee
from tiicee.experiments.e1 import reduced, tree
from tiicee.experiments.e1.config import tiny
from tiicee.experiments.e1.context import Context
from tiicee.twin import world_models as wm
from tiicee.twin.env import ImpossibleHistory
from tiicee.twin.readers import BayesReader, SimulatedReader

OPTIONS = {"independence": {}, "entrench": {"drift": "entrench"}, "fixed": {"drift": "fixed"}, "over": {"power": 2.0},
           "under": {"power": 0.5}, "entrench+over": {"drift": "entrench", "power": 2.0}}


def reader_object(twin, name):
    return BayesReader(twin) if name == "r0" else SimulatedReader(twin, **OPTIONS[name])


def world_object(ctx, name):
    return wm.w0(ctx.twin("reduced")) if name == "W0" else ctx.fitted("reduced", name)


def dense(model, predictive):
    out = np.zeros(len(model.outcomes))
    for outcome, weight in zip(predictive.outcomes, predictive.weights):
        out[model.outcomes.index(outcome)] += weight
    return out


def selector_object(ctx, name):
    twin = ctx.twin("reduced")
    tests = twin.model.tests
    if name == "rand":
        order = [tests[t] for t in ctx.order("reduced")]
        return lambda history: next(t for t in order if t not in {a for a, _ in history})
    return lambda history: max(twin.eligible(history), key=lambda t: (twin.true_information(history, t), -tests.index(t)))


def reference(ctx, selector_name, reader_name, world_name, scaling=None):
    """Expected log loss and accuracy by horizon, for raw, stateless and stateful beliefs, one real path at a time."""
    twin, horizon = ctx.twin("reduced"), ctx.horizon("reduced")
    model = twin.model
    reader, world, select = reader_object(twin, reader_name), world_object(ctx, world_name), selector_object(ctx, selector_name)
    out = {arm: np.zeros((horizon + 1, 2)) for arm in ("raw", "stateless", "stateful")}

    def walk(history, probability, stateless, stateful):
        posterior = twin.true_posterior(history)
        beliefs = {"raw": cee.floor_norm(reader.belief(history)), "stateless": stateless, "stateful": stateful}
        for arm, belief in beliefs.items():
            out[arm][len(history)] += probability * np.array([-(posterior @ np.log(belief)), posterior[np.argmax(belief)]])
        if len(history) == horizon:
            return
        test = select(history)
        truth = dense(model, twin.true_predictive(history, test))
        weights = dense(model, world.outcomes(history, test, None, None))
        posteriors = [reader.read(history, test, o) for o in model.outcomes]
        arg = None if scaling is None else (twin.true_information(history, test), scaling)
        for o, outcome in enumerate(model.outcomes):
            if truth[o] > 0:
                a = cee.stateless_step(reader.belief(history), weights, posteriors, o, arg)
                b = cee.stateful_step(stateful, reader.belief(history), weights, posteriors, o, arg)
                walk(history + ((test, outcome),), probability * truth[o], a.belief, b.belief)

    start = cee.floor_norm(reader.belief(()))
    walk((), 1.0, start, start)
    return out


@pytest.fixture(scope="module", params=["base", "duplicated_panel"])
def ctx(request):
    return Context(tiny(n_train=300), 0, request.param)


@pytest.fixture(scope="module")
def scenes(ctx):
    return {name: reduced.Scene(ctx, name) for name in reduced.SELECTORS}


def test_tree_enumerates_every_path_with_its_exact_probability(ctx, scenes):
    twin = ctx.twin("reduced")
    for name, scene in scenes.items():
        select = selector_object(ctx, name)
        assert [lv.n for lv in scene.levels] == [6 ** t for t in range(ctx.horizon("reduced") + 1)]
        for lv in scene.levels:
            assert lv.class_mass.sum() == pytest.approx(1.0, abs=1e-12)
            histories = lv.histories(twin.model)
            assert len(set(histories)) == lv.n
            for i in range(0, lv.n, max(1, lv.n // 40)):
                history = histories[i]
                try:
                    posterior = twin.true_posterior(history)
                except ImpossibleHistory:
                    assert lv.class_mass[i].sum() == 0.0
                    continue
                np.testing.assert_allclose(lv.class_mass[i] / lv.class_mass[i].sum(), posterior, rtol=0, atol=1e-12)
                assert lv.cost[i] == pytest.approx(sum(twin.charged_cost(t, o) for t, o in history), abs=1e-15)
                if lv.tests is not None:
                    assert twin.model.tests[lv.tests[i]] == select(history)
                    assert scene.exact_budget[lv.t][i] == pytest.approx(twin.true_information(history, select(history)), abs=1e-12)


@pytest.mark.parametrize("reader_name,world_name", [
    ("independence", "W0"), ("entrench", "W0"), ("fixed", "W2"), ("over", "W1"), ("r0", "W1"), ("r0", "W0"), ("entrench+over", "W2")])
def test_part_b_horizons_match_the_path_by_path_reference(ctx, scenes, reader_name, world_name):
    for name, scene in scenes.items():
        got = reduced.part_b_cell(scene, reader_name, world_name)
        expected = reference(ctx, name, reader_name, world_name)
        for row in got["horizon"]:
            log_loss, accuracy = expected[row["arm"]][row["horizon"]]
            assert row["log_loss"] == pytest.approx(log_loss, abs=1e-10), (name, row)
            assert row["accuracy"] == pytest.approx(accuracy, abs=1e-12), (name, row)
        assert len(got["horizon"]) == 3 * (ctx.horizon("reduced") + 1)


def test_r0_under_the_truth_is_left_alone(ctx, scenes):
    for scene in scenes.values():
        got = reduced.part_b_cell(scene, "r0", "W0")
        raw = {row["horizon"]: row["log_loss"] for row in got["horizon"] if row["arm"] == "raw"}
        for row in got["horizon"]:
            assert abs(row["log_loss"] - raw[row["horizon"]]) < 1e-9


@pytest.mark.parametrize("variant", ["shrink-only", "two-sided"])
def test_part_d_scaled_arms_match_the_path_by_path_reference(ctx, scenes, variant):
    scene = scenes["rand"]
    for reader_name, world_name in (("over", "W0"), ("under", "W2"), ("independence", "W2")):
        got = reduced.part_d_cell(scene, reader_name, world_name, budget_names=("exact",))
        expected = reference(ctx, "rand", reader_name, world_name, scaling=variant)
        plain = reference(ctx, "rand", reader_name, world_name)
        for row in got["horizon"]:
            if row["scale"] == variant:
                want = expected[row["track"]][row["horizon"]]
            elif row["scale"] == "none":
                want = plain[row["track"]][row["horizon"]]
            else:
                continue
            assert row["log_loss"] == pytest.approx(want[0], abs=1e-9), row
            assert row["accuracy"] == pytest.approx(want[1], abs=1e-12), row


def test_threshold_commit_matches_exact_value_of_the_twin(ctx, scenes):
    twin = ctx.twin("reduced")
    for name, scene in scenes.items():
        select = selector_object(ctx, name)
        for reader_name in ("independence", "over"):
            reader = reader_object(twin, reader_name)
            got = {(r["commit"], r["arm"]): r for r in reduced.part_b_cell(scene, reader_name, "W0")["commit"]}
            for thr in (0.7, 0.8, 0.9):
                commit = lambda history, belief, forced, thr=thr: int(np.argmax(belief)) if forced or belief.max() >= thr else None
                value = twin.exact_value(lambda history, belief: select(history), reader, commit, budget=ctx.horizon("reduced"))
                row = got[(f"threshold_{thr:g}", "raw")]
                assert row["accuracy"] == pytest.approx(value.accuracy, abs=1e-12)
                assert row["cost"] == pytest.approx(value.cost, abs=1e-12)
                assert row["requests"] == pytest.approx(value.requests, abs=1e-12)
            # the step limit: every test of the horizon is requested, and the cost is p_available x cost summed along the way
            never = lambda history, belief, forced: int(np.argmax(belief)) if forced else None
            value = twin.exact_value(lambda history, belief: select(history), reader, never, budget=ctx.horizon("reduced"))
            row = got[("step_limit", "raw")]
            assert row["accuracy"] == pytest.approx(value.accuracy, abs=1e-12) and row["cost"] == pytest.approx(value.cost, abs=1e-12)
            assert row["requests"] == pytest.approx(ctx.horizon("reduced"), abs=1e-12)
            scaled = {(r["commit"], r["track"], r["scale"]): r for r in reduced.part_d_cell(scene, reader_name, "W0", budget_names=())["commit"]}
            assert scaled[("step_limit", "raw", "none")]["cost"] == pytest.approx(value.cost, abs=1e-12)
            assert scaled[("threshold_0.8", "raw", "none")]["accuracy"] == pytest.approx(got[("threshold_0.8", "raw")]["accuracy"], abs=1e-12)


def test_cee_commit_matches_the_library_rule_through_exact_value(ctx, scenes):
    """The stateless track with CEE-Commit, rebuilt from cee.project, cee.coherent_voi and cee.commit."""
    twin = ctx.twin("reduced")
    model = twin.model
    scene = scenes["rand"]
    select = selector_object(ctx, "rand")
    reader, world = reader_object(twin, "entrench"), world_object(ctx, "W2")
    utility = cee.identity_utility(len(model.classes))

    class Tracked:
        classes = model.classes

        def belief(self, history):
            if not history:
                return cee.floor_norm(reader.belief(()))
            parent, (test, outcome) = history[:-1], history[-1]
            weights = dense(model, world.outcomes(parent, test, None, None))
            posteriors = [reader.read(parent, test, o) for o in model.outcomes]
            return cee.stateless_step(reader.belief(parent), weights, posteriors, model.outcomes.index(outcome)).belief

    got = {(r["commit"], r["lam"], r["arm"]): r for r in reduced.part_b_cell(scene, "entrench", "W2")["commit"]}
    for lam in (0.0, 1.0, 2.0):
        def commit(history, belief, forced, lam=lam):
            net = []
            for test in twin.eligible(history):
                predictive = world.outcomes(history, test, None, None)
                weights = dense(model, predictive)
                projection = cee.project(reader.belief(history), weights, [reader.read(history, test, o) for o in model.outcomes])
                voi = cee.coherent_voi(projection.prior, projection.weights, projection.corrected, utility)
                cost = cee.expected_charged_cost(1.0 - weights[-1], float(model.cost_norm[model.tests.index(test)]))
                net.append(cee.net_value(voi, cost, lam))
            stop, diagnosis = cee.commit(belief, utility, net, budget_reached=forced)
            return diagnosis if stop else None

        value = twin.exact_value(lambda history, belief: select(history), Tracked(), commit, budget=ctx.horizon("reduced"))
        row = got[("cee_commit", lam, "stateless")]
        assert row["accuracy"] == pytest.approx(value.accuracy, abs=1e-10), lam
        assert row["cost"] == pytest.approx(value.cost, abs=1e-12), lam
        assert row["requests"] == pytest.approx(value.requests, abs=1e-12), lam


def test_accounting_matches_the_library_on_every_path(ctx, scenes):
    """The three terms of 5.12 (iii) for a world model with wider support than the truth (W1)."""
    twin, horizon = ctx.twin("reduced"), ctx.horizon("reduced")
    model = twin.model
    scene = scenes["rand"]
    select = selector_object(ctx, "rand")
    reader, world = reader_object(twin, "fixed"), world_object(ctx, "W1")
    got = {(r["track"], r["horizon"]): r for r in reduced.part_d_cell(scene, "fixed", "W1", budget_names=())["horizon"] if r["scale"] == "none"}
    n_out = len(model.outcomes)

    def history_of(path):
        history = ()
        for o in path:
            history = history + ((select(history), model.outcomes[o]),)
        return history

    def raw(history):
        return cee.floor_norm(reader.belief(history))

    for depth in range(1, horizon + 1):
        paths = list(itertools.product(range(n_out), repeat=depth))
        truth, probs, leaves = [], [], []
        for path in paths:
            history = history_of(path)
            try:
                mass = float(np.prod([dense(model, twin.true_predictive(history[:k], history[k][0]))[path[k]] for k in range(depth)]))
                truth.append(mass * twin.true_posterior(history) if mass > 0 else np.zeros(len(model.classes)))
            except ImpossibleHistory:
                truth.append(np.zeros(len(model.classes)))
            probs.append(float(np.prod([dense(model, world.outcomes(history[:k], history[k][0], None, None))[path[k]] for k in range(depth)])))
            leaves.append(raw(history))
        prefixes = {p: raw(history_of(p)) for k in range(depth) for p in itertools.product(range(n_out), repeat=k)}
        ref = cee.multistep_accounting(paths, np.array(truth), np.array(probs), np.array(leaves), prefixes)
        row = got[("raw", depth)]
        assert row["excess"] == pytest.approx(ref.excess_log_loss, abs=1e-10)
        assert row["prior_kl"] == pytest.approx(ref.prior_kl, abs=1e-10)
        assert row["likelihood_kl"] == pytest.approx(ref.likelihood_kl, abs=1e-10)
        assert row["world_model_kl"] == pytest.approx(ref.world_model_kl, abs=1e-10)
        assert abs(row["residual"]) < 1e-10 and abs(ref.residual) < 1e-10


def test_the_corrected_agent_has_no_accumulated_drift(ctx, scenes):
    """Statement (ii) of 5.12: on the stateful track the implied class marginal is c_0, so prior_kl is the same at every horizon."""
    scene = scenes["ig"]
    rows = reduced.part_d_cell(scene, "entrench", "W2", budget_names=())["horizon"]
    stateful = [r["prior_kl"] for r in rows if r["track"] == "stateful" and r["scale"] == "none"]
    raw = [r["prior_kl"] for r in rows if r["track"] == "raw"]
    assert max(stateful) - min(stateful) < 1e-7
    assert max(raw) - min(raw) > 1e-4
    assert all(abs(r["residual"]) < 1e-9 for r in rows)


def test_tree_stays_well_formed_on_paths_the_truth_rules_out():
    """The examination is always in the chart, so its UNAVAILABLE branch has no mass; the tree still requests a new test there."""
    cfg = tiny(n_train=300, reduced_tests=("Physical Examination", "Complete Blood Count", "CT"))
    context = Context(cfg, 0, "base")
    twin = context.twin("reduced")
    for name in reduced.SELECTORS:
        scene = reduced.Scene(context, name)
        dead = 0
        for lv in scene.levels:
            assert (lv.asked.sum(1) == lv.t).all()                           # one new test per step on every path
            assert len(set(lv.histories(twin.model))) == lv.n
            assert lv.class_mass.sum() == pytest.approx(1.0, abs=1e-12)
            empty = lv.class_mass.sum(1) == 0
            dead += int(empty.sum())
            if lv.tests is not None:
                assert not lv.asked[np.arange(lv.n), lv.tests].any()         # also where nothing is left to learn
        assert dead > 0
        for reader_name, world_name in (("r0", "W1"), ("entrench", "W0")):
            cell = reduced.part_d_cell(scene, reader_name, world_name, budget_names=("exact",))
            assert all(np.isfinite(r["log_loss"]) and abs(r["residual"]) < 1e-9 for r in cell["horizon"])


def test_budgets_of_a_level(ctx, scenes):
    scene = scenes["rand"]
    twin = ctx.twin("reduced")
    available = reduced.budgets(scene, 1)
    exact, everywhere = available["exact"]
    assert everywhere.all() and set(available) == {"exact", "exact_x0.5", "exact_x2", "exact_noise", "w1", "w2cc"}
    np.testing.assert_allclose(available["exact_x0.5"][0], 0.5 * exact)
    np.testing.assert_allclose(available["exact_x2"][0], 2 * exact)
    noise = available["exact_noise"][0][exact > 0] / exact[exact > 0]
    assert len(np.unique(np.round(noise, 9))) == len(noise) and (noise > 0).all()
    np.testing.assert_array_equal(reduced.budgets(scene, 1)["exact_noise"][0], available["exact_noise"][0])      # same stream, same draw
    histories = scene.levels[1].histories(twin.model)
    test = twin.model.tests[scene.levels[1].tests[0]]
    for i in range(len(histories)):
        assert available["w1"][0][i] == pytest.approx(ctx.fitted("reduced", "W1").budget(histories[i], test), abs=1e-12)
        try:
            assert available["w2cc"][0][i] == pytest.approx(ctx.fitted("reduced", "W2").budget(histories[i], test), abs=1e-12)
        except ImpossibleHistory:
            pass
