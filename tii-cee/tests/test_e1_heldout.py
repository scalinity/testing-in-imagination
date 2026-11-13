"""The held-out budget of part D: the reference reader's fit, the strata, the bound and the eligibility test."""
from __future__ import annotations

import numpy as np
import pytest

from tiicee.cee import bootstrap_upper_bound, floor_norm
from tiicee.experiments.e1 import heldout, reduced
from tiicee.experiments.e1.config import tiny
from tiicee.experiments.e1.context import Context
from tiicee.twin.env import stream


def test_reference_reader_reaches_the_minimum_of_its_objective():
    rng = np.random.default_rng(0)
    reader = heldout.ReferenceReader(n_tests=3, n_outcomes=4, n_classes=3, l2=1.0)
    answers = rng.integers(-1, 4, size=(400, 3))
    x = reader.features(answers)
    y = (answers[:, 0] + rng.integers(0, 2, size=400)) % 3
    reader.fit(x, y)
    assert np.abs(reader.gradient(x, y)).max() <= 1e-9 * 400
    best = reader.objective(x, y)
    for _ in range(50):                                   # no nearby point is lower
        assert reader.objective(x, y, reader.weights + 1e-3 * rng.normal(size=reader.weights.shape)) > best
    slow = np.zeros_like(reader.weights)                  # plain gradient descent arrives at the same value
    for _ in range(20000):
        slow -= 0.002 * reader.gradient(x, y, slow)
    assert reader.objective(x, y, slow) == pytest.approx(best, abs=1e-6)
    np.testing.assert_allclose(reader.predict(x), reader._probabilities(x, slow), atol=1e-4)
    assert reader.predict(x).sum(1) == pytest.approx(1.0)


def test_features_mark_each_result_read_and_nothing_else():
    reader = heldout.ReferenceReader(n_tests=3, n_outcomes=4, n_classes=2, l2=1.0)
    x = reader.features(np.array([[-1, -1, -1], [2, -1, 0], [3, 3, 3]]))
    assert x.shape == (3, 13) and x[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert np.flatnonzero(x[0]).tolist() == [0]
    assert np.flatnonzero(x[1]).tolist() == [0, 1 + 0 * 4 + 2, 1 + 2 * 4 + 0]
    assert np.flatnonzero(x[2]).tolist() == [0, 4, 8, 12]


def test_strata_follow_specification_5_11():
    n_results = np.array([0, 0, 1, 1, 2, 5])
    largest = np.array([0.5, 0.9, 0.8, 0.81, 0.3, 0.99])
    assert heldout.stratum(n_results, largest).tolist() == [0, 1, 2, 3, 4, 5]


@pytest.fixture(scope="module")
def estimate():
    ctx = Context(tiny(n_train=600, n_boot=400), 1, "base")
    return ctx, heldout.HeldOut(ctx)


def test_bound_and_eligibility_are_recomputed_from_the_patients(estimate):
    ctx, got = estimate
    cfg, twin = ctx.cfg, ctx.twin("reduced")
    m = twin.model
    n, n_tests = cfg.n_train, len(m.tests)
    patients = twin.sample_patients(n, ctx.seed, "train", purpose="fitting")
    order = stream(ctx.seed, "behavior", "train", n_tests).permuted(np.tile(np.arange(n_tests), (n, 1)), axis=1)
    assert sorted(order[0].tolist()) == list(range(n_tests)) and len({tuple(r) for r in order}) > 1

    values = {}                                           # (test, stratum) -> [(patient, fold, reduction, exact information)]
    for i, p in enumerate(patients):
        fold = i % cfg.heldout_folds
        others = [j for j in range(n) if j % cfg.heldout_folds != fold]
        if i < cfg.heldout_folds:                         # one fit per fold, reused for the fold's patients
            rows_x, rows_y = [], []
            for j in others:
                answers = np.full(n_tests, -1)
                for k in range(n_tests + 1):
                    rows_x.append(answers.copy())
                    rows_y.append(m.classes.index(patients[j].diagnosis))
                    if k < n_tests:
                        answers[order[j, k]] = m.outcomes.index(twin.observe(patients[j], m.tests[order[j, k]]))
            reader = heldout.ReferenceReader(n_tests, len(m.outcomes), len(m.classes), cfg.heldout_l2)
            values[("reader", fold)] = reader.fit(reader.features(np.array(rows_x)), np.array(rows_y))
        reader = values[("reader", fold)]
        y = m.classes.index(p.diagnosis)
        answers, history, n_results = np.full(n_tests, -1), (), 0
        for k in range(n_tests):
            test = order[i, k]
            before = floor_norm(reader.predict(reader.features(answers[None])))[0]
            outcome = twin.observe(p, m.tests[test])
            exact = twin.true_information(history, m.tests[test])
            cell = min(n_results, 2) * 2 + int(reader.predict(reader.features(answers[None]))[0].max() > 0.8)
            answers[test] = m.outcomes.index(outcome)
            after = floor_norm(reader.predict(reader.features(answers[None])))[0]
            values.setdefault((int(test), cell), []).append((i, fold, np.log(after[y]) - np.log(before[y]), exact))
            history = history + ((m.tests[test], outcome),)
            n_results += outcome != m.outcomes[-1]

    checked = 0
    for record in got.records:
        t, s = m.tests.index(record["test"]), record["stratum"]
        rows = values.get((t, s), [])
        est = [r for r in rows if r[1] < cfg.heldout_folds - 1]
        check = [r for r in rows if r[1] == cfg.heldout_folds - 1]
        assert record["n_estimation"] == len(est) and record["n_check"] == len(check)
        if not (est and check):
            assert not record["eligible"] and record["budget"] == 0.0 and not got.eligible[t, s]
            continue
        reductions = np.array([r[2] for r in est])
        rng = stream(ctx.seed, "montecarlo", "train", t, s)
        bound = -bootstrap_upper_bound(-reductions, np.array([r[0] for r in est]), rng, level=0.90, n_boot=cfg.n_boot)
        assert record["mean_estimation"] == pytest.approx(reductions.mean(), abs=1e-9)
        assert record["bound"] == pytest.approx(bound, abs=1e-9) and record["bound"] <= record["mean_estimation"]
        assert record["mean_check"] == pytest.approx(np.mean([r[2] for r in check]), abs=1e-9)
        assert record["exact_mean"] == pytest.approx(np.mean([r[3] for r in est]), abs=1e-10)
        assert record["budget"] == max(record["bound"], 0.0) == got.budget[t, s]
        assert record["eligible"] == (record["bound"] <= record["mean_check"]) == bool(got.eligible[t, s])
        for name, factor in (("x1", 1.0), ("x0.5", 0.5), ("x2", 2.0)):
            assert record[f"stress_{name}"] == factor * record["budget"]
            assert record[f"overestimated_{name}"] == (factor * record["budget"] > record["exact_mean"])
            assert record[f"admitted_{name}"] == (factor * record["budget"] <= record["mean_check"])
        assert record["stress_noise"] > 0 or record["budget"] == 0
        checked += 1
    assert checked >= 6 and len(got.records) == n_tests * heldout.N_STRATA


def test_budget_lookup_on_a_tree_level(estimate):
    ctx, got = estimate
    scene = reduced.Scene(ctx, "rand")
    level = scene.levels[2]
    budget, eligible = got.at(level)
    cell = got.strata(level.answers)
    absent = len(scene.model.outcomes) - 1
    for i in range(level.n):
        n_results = int(((level.answers[i] >= 0) & (level.answers[i] != absent)).sum())
        largest = got.reader.predict(got.reader.features(level.answers[i][None]))[0].max()
        assert cell[i] == min(n_results, 2) * 2 + int(largest > 0.8)
        assert budget[i] == got.budget[level.tests[i], cell[i]] and eligible[i] == got.eligible[level.tests[i], cell[i]]
    available = reduced.budgets(scene, 2, got)
    np.testing.assert_array_equal(available["heldout"][0], budget)


def test_scale_is_skipped_where_the_held_out_budget_is_not_eligible(estimate):
    ctx, got = estimate
    scene = reduced.Scene(ctx, "rand")
    blocked = heldout.HeldOut.__new__(heldout.HeldOut)
    blocked.__dict__.update(got.__dict__)
    blocked.eligible = np.zeros_like(got.eligible)
    rows = reduced.part_d_cell(scene, "over", "W2", heldout=blocked, budget_names=("heldout",))
    by_arm = {(r["track"], r["scale"], r["horizon"]): r["log_loss"] for r in rows["horizon"]}
    for track in ("stateless", "stateful"):
        for h in range(ctx.horizon("reduced") + 1):
            assert by_arm[(track, "two-sided", h)] == by_arm[(track, "none", h)]      # projection only everywhere
    assert all(r["scaled"] == 0.0 and r["mean_log_tau"] == 0.0 for r in rows["tau"])
    open_rows = reduced.part_d_cell(scene, "over", "W2", heldout=got, budget_names=("heldout",))
    assert any(r["scaled"] > 0 for r in open_rows["tau"]) == bool(got.eligible.any())
