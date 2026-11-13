"""The CEE library's per-point routines, vectorized over many decision points.

E1 evaluates tens of millions of decision points; tiicee.cee handles one at a time. Every function
here is the same computation as the library function named in its docstring, on arrays with a
leading axis of points, and tests/test_e1_batch.py holds each to the library's result. Nothing here
is a second definition of the method: where the two could differ, the library is the reference.

Shapes: prior [n, class]; weights [n, outcome]; posteriors [n, outcome, class]. As in the library's
internal routines, inputs are strictly positive reader vectors that the caller has floored once
with tiicee.cee.floor_norm; weights are never floored and may hold zeros.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from tiicee.cee import MAX_ITER, TAU_RATIO_GRID, TOL, VARIANT_BOUNDS, kl
from tiicee.cee.scaling import GRID_RATIO, MONOTONE_SLACK, R_ZERO, RTOL

PATHS = ("solved", "clipped_low", "clipped_high", "fallback")


class Projection(NamedTuple):
    u: np.ndarray               # [n, class]
    log_z: np.ndarray           # [n, outcome]
    corrected: np.ndarray       # [n, outcome, class]
    kl_inc: np.ndarray          # [n]
    converged: np.ndarray       # [n] bool
    iterations: np.ndarray      # [n]
    violation: np.ndarray       # [n]


def project(prior, weights, posteriors, tol=TOL, max_iter=MAX_ITER) -> Projection:
    """tiicee.cee.core.ipf for every point: iterative proportional fitting, each point stopping on its own."""
    n, _, n_class = posteriors.shape
    u = np.full((n, n_class), 1.0 / n_class)
    iterations = np.zeros(n, dtype=int)
    live = np.arange(n)
    p, w, q, v = prior, weights, posteriors, u
    while live.size:
        z = np.einsum("noc,nc->no", q, v)
        mean = np.einsum("no,noc->nc", w, q * v[:, None, :] / z[:, :, None])
        violation = np.abs(mean - p).max(1)
        done = (violation <= tol) | (iterations[live] >= max_iter)
        u[live] = v
        if done.all():
            break
        keep = ~done
        v = v[keep] * p[keep] / mean[keep]
        v = v / v.sum(1, keepdims=True)
        live, p, w, q = live[keep], p[keep], w[keep], q[keep]
        iterations[live] += 1
    z = np.einsum("noc,nc->no", posteriors, u)
    corrected = posteriors * u[:, None, :] / z[:, :, None]
    violation = np.abs(np.einsum("no,noc->nc", weights, corrected) - prior).max(1)
    log_z = np.log(z)
    kl_inc = (prior * np.log(u)).sum(1) - (weights * log_z).sum(1)
    return Projection(u, log_z, corrected, kl_inc, violation <= tol, iterations, violation)


def tempered(prior, posteriors, tau) -> np.ndarray:
    """tiicee.cee.scaling.tempered: b (q / b)^(1/tau), normalized. tau is one value per point or a scalar."""
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (prior.shape[0],))
    log_b = np.log(prior)[:, None, :]
    log_t = log_b + (np.log(posteriors) - log_b) / tau[:, None, None]
    t = np.exp(log_t - log_t.max(-1, keepdims=True))
    return t / t.sum(-1, keepdims=True)


def responsiveness(weights, posteriors) -> np.ndarray:
    """tiicee.cee.measures.responsiveness_of: sum_o P_hat(o) KL(q_o || b_bar)."""
    mean = np.einsum("no,noc->nc", weights, posteriors)
    return (weights * kl(posteriors, mean[:, None, :])).sum(1)


def information(joint) -> np.ndarray:
    """Mutual information, in nats, of joints given as [n, class, outcome] (any positive scale). Zero mass gives 0."""
    total = joint.sum((1, 2), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        j = np.where(total > 0, joint / total, 0.0)
        independent = j.sum(2, keepdims=True) * j.sum(1, keepdims=True)
        terms = np.where(j > 0, j * (np.log(j) - np.log(independent)), 0.0)
    return np.maximum(terms.sum((1, 2)), 0.0)


def _projected(prior, weights, posteriors, tau, tol, max_iter):
    projection = project(prior, weights, tempered(prior, posteriors, tau), tol, max_iter)
    return projection, (weights * kl(projection.corrected, prior[:, None, :])).sum(1)


def tau_grid(lo, hi) -> np.ndarray:
    """tiicee.cee.scaling._tau_grid."""
    grid = [float(lo)]
    while grid[-1] * GRID_RATIO < hi * (1.0 - 1e-12):
        grid.append(grid[-1] * GRID_RATIO)
    return np.array(grid + [float(hi)])


class TauSearch(NamedTuple):
    tau: np.ndarray             # [n]
    path: np.ndarray            # [n] index into PATHS
    monotone: np.ndarray        # [n] bool
    converged: np.ndarray       # [n] bool: every projection of the search converged


def tau_search(prior, weights, posteriors, budget, bounds, rtol=RTOL, tol=TOL, max_iter=MAX_ITER) -> TauSearch:
    """tiicee.cee.scaling.search: the largest tau in bounds with r'(tau) >= budget, by grid and bisection on log tau."""
    lo, hi = bounds
    n = prior.shape[0]
    budget = np.broadcast_to(np.asarray(budget, dtype=float), (n,))
    grid = tau_grid(lo, hi)
    r_grid = np.empty((n, grid.size))
    ok_grid = np.empty((n, grid.size), dtype=bool)
    for g, tau in enumerate(grid):
        projection, r_grid[:, g] = _projected(prior, weights, posteriors, tau, tol, max_iter)
        ok_grid[:, g] = projection.converged
    monotone = np.ones(n, dtype=bool)                   # r' non-increasing over the converged grid points
    previous = np.full(n, np.nan)
    for g in range(grid.size):
        ok = ok_grid[:, g]
        monotone &= ~(ok & ~np.isnan(previous) & (r_grid[:, g] - previous > MONOTONE_SLACK * tol))
        previous = np.where(ok, r_grid[:, g], previous)
    meets = r_grid >= budget[:, None]
    any_meet = meets.any(1)
    last = grid.size - 1 - np.argmax(meets[:, ::-1], axis=1)
    tau = np.where(any_meet, hi, lo).astype(float)
    path = np.where(any_meet, 2, 1)
    converged = ok_grid.all(1)
    rows = np.flatnonzero(any_meet & (last < grid.size - 1))
    if rows.size:
        low, high = np.log(grid[last[rows]]), np.log(grid[last[rows] + 1])
        pr, wr, qr, br = prior[rows], weights[rows], posteriors[rows], budget[rows]
        ok_rows = np.ones(rows.size, dtype=bool)
        active = high - low > np.log1p(rtol)
        while active.any():
            mid = 0.5 * (low + high)
            idx = np.flatnonzero(active)
            projection, r = _projected(pr[idx], wr[idx], qr[idx], np.exp(mid[idx]), tol, max_iter)
            ok_rows[idx] &= projection.converged
            up = r >= br[idx]
            low[idx] = np.where(up, mid[idx], low[idx])
            high[idx] = np.where(up, high[idx], mid[idx])
            active = high - low > np.log1p(rtol)
        tau[rows] = np.exp(0.5 * (low + high))
        path[rows] = np.where(monotone[rows], 0, 3)
        converged[rows] &= ok_rows
    return TauSearch(tau, path, monotone, converged)


def scale(prior, weights, posteriors, budget, variant, tol=TOL, max_iter=MAX_ITER):
    """tiicee.cee.scaling.scale_positive: (the search, the projection of the pair tempered at tau_B)."""
    found = tau_search(prior, weights, posteriors, budget, VARIANT_BOUNDS[variant], RTOL, tol, max_iter)
    return found, _projected(prior, weights, posteriors, found.tau, tol, max_iter)[0]


def tau_ratio(prior, weights, posteriors, grid=TAU_RATIO_GRID, tol=TOL, max_iter=MAX_ITER) -> np.ndarray:
    """tiicee.cee.scaling.tau_ratio: the grid value minimizing KL_inc(tau) / r(tau); nan where no grid value is usable."""
    grid = np.asarray(grid, dtype=float)
    ratio = np.full((prior.shape[0], grid.size), np.nan)
    for g, tau in enumerate(grid):
        t = tempered(prior, posteriors, tau)
        projection = project(prior, weights, t, tol, max_iter)
        r = responsiveness(weights, t)
        usable = projection.converged & (r > R_ZERO)
        ratio[usable, g] = projection.kl_inc[usable] / r[usable]
    usable = ~np.isnan(ratio).all(1)
    out = np.full(prior.shape[0], np.nan)
    out[usable] = grid[np.nanargmin(ratio[usable], axis=1)]
    return out


class Step(NamedTuple):
    beliefs: np.ndarray                 # [n, outcome, class]: the belief carried forward after each outcome
    projection: Projection
    search: Optional[TauSearch]
    fallback: np.ndarray                # [n] bool: the projection did not converge and the raw update is carried


def _step(target, weights, proposals, scaling):
    if scaling is None:
        search, projection = None, project(target, weights, proposals)
    else:
        search, projection = scale(target, weights, proposals, *scaling)
    beliefs = np.where(projection.converged[:, None, None], projection.corrected, proposals)
    return Step(beliefs, projection, search, ~projection.converged)


def stateless_step(b, weights, posteriors, scaling=None) -> Step:
    """tiicee.cee.tracks.stateless_step for every outcome at once. scaling is (budget [n], variant) or None."""
    return _step(b, weights, posteriors, scaling)


def stateful_step(c, b, weights, posteriors, scaling=None) -> Step:
    """tiicee.cee.tracks.stateful_step for every outcome at once: proposals c x q / b projected onto mean c."""
    proposals = c[:, None, :] * posteriors / b[:, None, :]
    return _step(c, weights, proposals / proposals.sum(-1, keepdims=True), scaling)
