"""Tests for the discrete world models (learned, oracle, corrupted)."""

from __future__ import annotations

import math

import pytest

from extension.envs.synthetic_diagnosis.actions import A_DIAG, A_TEST
from extension.envs.synthetic_diagnosis.collect import collect_rows
from extension.envs.synthetic_diagnosis.profiles import (
    AVAILABILITY,
    FINDINGS,
    LIKELIHOOD,
    PRIOR_MIMIC_CDM,
)
from extension.world_model import (
    corrupt_model,
    fit_count_model,
    informativeness_ranking,
    oracle_model,
)

V1 = {"version": "v1"}
V1_MISS = {"version": "v1", "missingness": True}


@pytest.fixture(scope="module")
def train_rows():
    return collect_rows(3000, seed=0, env_kwargs=V1_MISS)


@pytest.fixture(scope="module")
def heldout_rows():
    return collect_rows(500, seed=1, split="val", env_kwargs=V1_MISS)


def test_predictive_and_posterior_are_distributions(train_rows):
    for wm in (oracle_model(), fit_count_model(train_rows)):
        b = wm.initial_belief()
        for a in A_TEST:
            pred = wm.predictive(b, a)
            assert set(pred) == set(FINDINGS)
            assert math.isclose(sum(pred.values()), 1.0)
        b = wm.update(b, "CT", "appendicitis")
        post = wm.posterior_y(b)
        assert math.isclose(sum(post.values()), 1.0)
        assert max(post, key=post.get) == "appendicitis"


def test_unavailable_finding_leaves_belief_unchanged():
    wm = oracle_model()
    b = wm.initial_belief()
    assert wm.update(b, "MRI", None) == b


def test_oracle_prior_marginal_is_class_prior():
    post = oracle_model().posterior_y(oracle_model().initial_belief())
    for y in A_DIAG:
        assert math.isclose(post[y], PRIOR_MIMIC_CDM[y])


def test_learned_model_recovers_env(train_rows):
    wm = fit_count_model(train_rows)
    for y in A_DIAG:
        assert abs(wm.prior_z[y] - PRIOR_MIMIC_CDM[y]) < 0.04
    # Learned P(s | CT, y) ≈ marginal over phenotype, dominated by e = y.
    assert abs(
        wm.lik["CT"]["appendicitis"]["appendicitis"]
        - LIKELIHOOD["CT"]["appendicitis"]["appendicitis"]
    ) < 0.1
    for a in ("Physical Examination", "CT", "Complete Blood Count"):
        assert abs(wm.availability[a] - AVAILABILITY[a]) < 0.06


def test_heldout_fit_ordering(train_rows, heldout_rows):
    oracle = oracle_model(missingness=True)
    learned = fit_count_model(train_rows)
    small = fit_count_model(train_rows[:40])
    bad = corrupt_model(learned, 1.0)
    ll = {m.name: m.log_likelihood(heldout_rows) for m in (oracle, learned, bad)}
    ll["small"] = small.log_likelihood(heldout_rows)
    assert ll["oracle"] >= ll["learned"] - 0.01
    assert ll["learned"] > ll["small"]
    assert ll["learned"] > ll[bad.name]


def test_information_gain_ranks_imaging_over_urinalysis():
    wm = oracle_model()
    b = wm.initial_belief()
    assert wm.information_gain(b, "MRI") > wm.information_gain(b, "Urinalysis")
    assert wm.information_gain(b, "CT") > wm.information_gain(b, "Radiograph")
    # Seeing a test resolves uncertainty: IG of a repeated look is smaller.
    b2 = wm.update(b, "CT", "appendicitis")
    assert wm.information_gain(b2, "CT") < wm.information_gain(b, "CT")


def test_corrupt_model_alpha_zero_is_identity_and_one_mirrors(train_rows):
    base = fit_count_model(train_rows)
    same = corrupt_model(base, 0.0)
    assert same.lik == base.lik
    bad = corrupt_model(base, 1.0)
    ranked = informativeness_ranking(base)
    top, bottom = ranked[0], ranked[-1]
    assert bad.lik[top] == base.lik[bottom]
    assert bad.lik[bottom] == base.lik[top]
    b = bad.initial_belief()
    assert bad.information_gain(b, bottom) > bad.information_gain(b, top)
    with pytest.raises(ValueError):
        corrupt_model(base, 1.5)


def test_fit_rejects_v0_rows():
    rows = collect_rows(5, seed=0)
    with pytest.raises(ValueError):
        fit_count_model(rows)
