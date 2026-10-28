"""Tests for the test-selection agents and the shared episode loop."""

from __future__ import annotations

import pytest

from extension.envs.synthetic_diagnosis.actions import A_DIAG, A_TEST, COST_USD
from extension.envs.synthetic_diagnosis.collect import collect_rows
from extension.envs.synthetic_diagnosis.env import SyntheticCDMEnv
from extension.planners import (
    ImaginationSelector,
    ModelFreeSelector,
    RandomSelector,
    run_episode,
)
from extension.world_model import corrupt_model, fit_count_model, oracle_model

V1_MISS = {"version": "v1", "missingness": True}


@pytest.fixture(scope="module")
def rows():
    return collect_rows(1500, seed=0, env_kwargs=V1_MISS)


@pytest.fixture(scope="module")
def learned(rows):
    return fit_count_model(rows)


def _run(selector, belief_model, n=300, tau=0.9, seed=100):
    env = SyntheticCDMEnv(seed=seed, split="test", **V1_MISS)
    return [run_episode(env, selector, belief_model, tau) for _ in range(n)]


def _utility(results, lam):
    return sum(r.correct - lam * r.cost_norm for r in results) / len(results)


def test_episode_result_bookkeeping(learned):
    for r in _run(ImaginationSelector(learned, 1.0), learned, n=50):
        assert r.y_true in A_DIAG
        assert r.y_pred is None or r.y_pred in A_DIAG
        assert r.n_tests == len(r.tests) <= r.n_requests <= 13
        assert r.cost_usd == pytest.approx(sum(COST_USD[a] for a in r.tests))
        assert len(set(r.tests)) == len(r.tests)


def test_agent_always_diagnoses_within_budget(learned):
    # Random never stops on its own: it requests all 12 tests, then must
    # diagnose. With a tighter budget the loop diagnoses on the last step.
    results = _run(RandomSelector(0), learned, n=100, tau=1.01)
    assert all(r.y_pred is not None for r in results)
    assert all(r.n_requests == len(A_TEST) for r in results)
    env = SyntheticCDMEnv(seed=100, split="test", max_steps=5, **V1_MISS)
    for _ in range(50):
        r = run_episode(env, RandomSelector(1), learned, tau=1.01)
        assert r.y_pred is not None and r.n_requests == 4


def test_same_patients_across_arms(learned):
    a = _run(ImaginationSelector(learned, 1.0), learned, n=50)
    b = _run(RandomSelector(0), learned, n=50)
    assert [r.y_true for r in a] == [r.y_true for r in b]


def test_imagination_prefers_informative_tests_first(learned):
    sel = ImaginationSelector(oracle_model(missingness=True), lam=0.0)
    sel.reset()
    first = sel.select(list(A_TEST), {})
    assert first in ("CT", "Physical Examination", "Ultrasound", "MRI")
    # With a huge cost weight nothing is worth ordering.
    pricey = ImaginationSelector(learned, lam=100.0)
    assert pricey.select(list(A_TEST), {}) is None


def test_model_free_learns_from_experience(rows, learned):
    mf = ModelFreeSelector.fit(rows, learned, lam=1.0)
    assert set(mf.q_global) == set(A_TEST)
    # Realized value of MRI is low: it is rarely in the chart and expensive.
    assert mf.q_global["MRI"] < mf.q_global["Physical Examination"]


def test_arm_ordering_on_utility(rows, learned):
    lam = 1.0
    oracle = oracle_model(missingness=True)
    u = {
        "oracle": _utility(_run(ImaginationSelector(oracle, lam), oracle), lam),
        "wm": _utility(_run(ImaginationSelector(learned, lam), learned), lam),
        "mf": _utility(_run(ModelFreeSelector.fit(rows, learned, lam), learned), lam),
        "bad": _utility(
            _run(ImaginationSelector(corrupt_model(learned, 1.0), lam), learned), lam
        ),
        "random": _utility(_run(RandomSelector(0), learned), lam),
    }
    assert u["oracle"] >= max(u["wm"], u["mf"]) - 0.02
    assert min(u["wm"], u["mf"]) > u["bad"]
    assert min(u["wm"], u["mf"]) > u["random"]


def test_run_episode_rejects_v0(learned):
    with pytest.raises(ValueError):
        run_episode(SyntheticCDMEnv(seed=0), RandomSelector(0), learned, 0.9)
