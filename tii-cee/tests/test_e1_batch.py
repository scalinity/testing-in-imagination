"""The vectorized routines of E1 against the CEE library, point by point. The library is the reference."""
from __future__ import annotations

import numpy as np
import pytest

from tiicee import cee
from tiicee.cee import scaling
from tiicee.experiments.e1 import batch


def random_points(rng, n, n_out=6, n_class=4, sharp=1.0, zero_weights=False):
    """Raw reader vectors, as a reader returns them. The library floors them itself; the batch routines get them floored once."""
    prior = rng.dirichlet(np.ones(n_class) * sharp, size=n)
    weights = rng.dirichlet(np.ones(n_out), size=n)
    if zero_weights:
        weights[:, 0] = 0.0
        weights = weights / weights.sum(1, keepdims=True)
    posteriors = rng.dirichlet(np.ones(n_class) * sharp, size=(n, n_out))
    return prior, weights, posteriors


def floored(points):
    prior, weights, posteriors = points
    return cee.floor_norm(prior), weights, cee.floor_norm(posteriors)


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(20261113)
    easy = random_points(rng, 60)
    sharp = random_points(rng, 60, sharp=0.15)                # near-degenerate beliefs, the floor active
    holes = random_points(rng, 30, zero_weights=True)
    return tuple(np.concatenate(parts) for parts in zip(easy, sharp, holes))


def test_project_matches_the_library(points):
    prior, weights, posteriors = points
    got = batch.project(*floored(points))
    assert (np.minimum(prior.min(1), posteriors.min((1, 2))) < 1e-6).sum() > 10        # the floor is exercised
    for i in range(len(prior)):
        ref = cee.project(prior[i], weights[i], posteriors[i])
        assert got.converged[i] == ref.converged and got.iterations[i] == ref.iterations
        np.testing.assert_allclose(got.u[i], ref.u, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got.corrected[i], ref.corrected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got.log_z[i], ref.log_z, rtol=0, atol=1e-12)
        assert got.kl_inc[i] == pytest.approx(ref.kl_inc, abs=1e-12)
        assert got.violation[i] == pytest.approx(ref.max_violation, abs=1e-14)


def test_project_reports_non_convergence_like_the_library(points):
    prior, weights, posteriors = points
    for limit in (3, 30):
        got = batch.project(*floored(points), max_iter=limit)
        flags = [cee.project(prior[i], weights[i], posteriors[i], max_iter=limit).converged for i in range(len(prior))]
        assert got.converged.tolist() == flags
        assert got.iterations.max() == limit and not all(flags)
    assert any(flags)


def test_tempered_responsiveness_and_grid_match_the_library(points):
    prior, weights, posteriors = points
    tau = np.exp(np.random.default_rng(1).uniform(np.log(0.125), np.log(16), size=len(prior)))
    fp, _, fq = floored(points)
    got = batch.tempered(fp, fq, tau)
    r = batch.responsiveness(weights, fq)
    for i in range(len(prior)):
        np.testing.assert_allclose(got[i], cee.temper(prior[i], posteriors[i], tau[i]), rtol=0, atol=1e-13)
        assert r[i] == pytest.approx(cee.responsiveness(weights[i], posteriors[i]), abs=1e-13)
    for bounds in cee.VARIANT_BOUNDS.values():
        np.testing.assert_array_equal(batch.tau_grid(*bounds), scaling._tau_grid(*bounds))
    assert batch.PATHS == ("solved", "clipped_low", "clipped_high", "fallback")


@pytest.mark.parametrize("variant", ["shrink-only", "two-sided"])
def test_scale_matches_the_library(points, variant):
    prior, weights, posteriors = points
    rng = np.random.default_rng(2)
    budget = np.concatenate([rng.uniform(0, 0.4, size=len(prior) - 20), np.zeros(10), np.full(10, 5.0)])
    search, projection = batch.scale(*floored(points), budget, variant)
    seen = set()
    for i in range(len(prior)):
        ref = cee.scale(prior[i], weights[i], posteriors[i], budget[i], variant)
        assert batch.PATHS[search.path[i]] == ref.path, i
        assert search.tau[i] == pytest.approx(ref.tau_b, rel=1e-12)
        assert search.monotone[i] == ref.monotone and search.converged[i] == ref.search_converged
        assert projection.converged[i] == ref.converged
        np.testing.assert_allclose(projection.corrected[i], ref.repaired, rtol=0, atol=1e-11)
        np.testing.assert_allclose(projection.u[i], ref.u, rtol=0, atol=1e-11)
        assert projection.kl_inc[i] == pytest.approx(ref.kl_inc, abs=1e-11)
        seen.add(ref.path)
    assert {"solved", "clipped_low", "clipped_high"} <= seen


def test_tau_ratio_matches_the_library(points):
    prior, weights, posteriors = points
    flat = np.repeat(prior[:5, None, :], posteriors.shape[1], axis=1)       # readers that do not move
    prior = np.concatenate([prior, prior[:5]])
    weights = np.concatenate([weights, weights[:5]])
    posteriors = np.concatenate([posteriors, flat])
    got = batch.tau_ratio(*floored((prior, weights, posteriors)))
    for i in range(len(prior)):
        ref = cee.tau_ratio(prior[i], weights[i], posteriors[i]).tau
        assert (np.isnan(got[i]) and np.isnan(ref)) or got[i] == ref, i
    assert np.isnan(got[-5:]).all() and not np.isnan(got[:-5]).any()


@pytest.mark.parametrize("scaling_arg", [None, "shrink-only", "two-sided"])
def test_track_steps_match_the_library(points, scaling_arg):
    prior, weights, posteriors = points
    rng = np.random.default_rng(3)
    c = rng.dirichlet(np.ones(4) * 2, size=len(prior))
    budget = rng.uniform(0, 0.3, size=len(prior))
    args = None if scaling_arg is None else (budget, scaling_arg)
    fp, _, fq = floored(points)
    stateless = batch.stateless_step(fp, weights, fq, args)
    stateful = batch.stateful_step(c, fp, weights, fq, args)
    for i in range(0, len(prior), 3):
        ref_args = None if scaling_arg is None else (budget[i], scaling_arg)
        for o in range(posteriors.shape[1]):
            ref = cee.stateless_step(prior[i], weights[i], posteriors[i], o, ref_args)
            np.testing.assert_allclose(stateless.beliefs[i, o], ref.belief, rtol=0, atol=1e-11)
            assert stateless.fallback[i] == ref.fallback
            ref = cee.stateful_step(c[i], prior[i], weights[i], posteriors[i], o, ref_args)
            np.testing.assert_allclose(stateful.beliefs[i, o], ref.belief, rtol=0, atol=1e-11)
            assert stateful.fallback[i] == ref.fallback


def test_a_step_that_does_not_converge_carries_the_raw_update():
    rng = np.random.default_rng(4)
    raw = random_points(rng, 400, sharp=0.1)
    prior, weights, posteriors = raw
    fp, _, fq = floored(raw)
    c = rng.dirichlet(np.ones(4), size=400)
    stateless, stateful = batch.stateless_step(fp, weights, fq), batch.stateful_step(c, fp, weights, fq)
    assert stateless.fallback.any() and not stateless.fallback.all()      # some of these do not converge in 500 updates
    for step, reference in ((stateless, lambda i, o: cee.stateless_step(prior[i], weights[i], posteriors[i], o)),
                            (stateful, lambda i, o: cee.stateful_step(c[i], prior[i], weights[i], posteriors[i], o))):
        for i in np.flatnonzero(step.fallback)[:4]:
            for o in (0, 3):
                ref = reference(i, o)
                assert ref.fallback
                np.testing.assert_allclose(step.beliefs[i, o], ref.belief, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(stateless.beliefs[stateless.fallback], fq[stateless.fallback])


def test_information_is_the_mutual_information_of_each_joint():
    rng = np.random.default_rng(5)
    joint = rng.dirichlet(np.ones(24), size=20).reshape(20, 4, 6) * rng.uniform(0.1, 3.0, size=(20, 1, 1))
    joint[3, :, 2] = 0.0
    joint[7] = 0.0
    got = batch.information(joint)
    for i in range(20):
        if joint[i].sum() == 0:
            assert got[i] == 0.0
            continue
        j = joint[i] / joint[i].sum()
        expected = sum(j[y, o] * np.log(j[y, o] / (j[y].sum() * j[:, o].sum())) for y in range(4) for o in range(6) if j[y, o] > 0)
        assert got[i] == pytest.approx(expected, abs=1e-13)
