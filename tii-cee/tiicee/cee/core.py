"""CEE-Read: the floor and the minimum-KL projection (SPEC 5 and 5.3; PROOF sections 2 and 3).

All logarithms are natural. Reader vectors (a prior b, imagined posteriors q_o, a real posterior
q_real) are floored once, on entry to the public functions that take them. Quantities derived from
them (class weights, corrected or tempered posteriors, stateful proposals, a tracked belief) are
strictly positive by construction and are never floored again.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOOR = 1e-6        # SPEC 5: probabilities entering a logarithm are floored, then renormalized
TOL = 1e-8          # SPEC 5.3: largest class-marginal violation at convergence
MAX_ITER = 500      # SPEC 5.3: iteration limit
SUM_TOL = 1e-6      # how far a supplied distribution may be from summing to one


def floor_norm(p, floor=FLOOR):
    """Floor at `floor`, then renormalize along the last axis (SPEC 5; the floor is part of the method).

    Not idempotent: a floored entry becomes floor / (1 + excess) < floor, so a second pass moves it
    again by a relative amount of about the floor. Floor a reader vector once.
    """
    p = np.maximum(np.asarray(p, dtype=float), floor)
    return p / p.sum(-1, keepdims=True)


def kl(p, q):
    """KL(p || q) in nats along the last axis, with 0 log 0 = 0 (SPEC 5).

    For two distributions the divergence is non-negative; rounding can make the computed sum a
    few units in the last place below zero, so it is clipped at zero.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)
    return np.maximum(terms.sum(-1), 0.0)


def entropy(p):
    """Shannon entropy in nats along the last axis, with 0 log 0 = 0 (SPEC 5)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log(p), 0.0)
    return terms.sum(-1)


def distribution(p, name):
    """Check that `p` is finite, non-negative and sums to one along the last axis; return it as an array."""
    a = np.asarray(p, dtype=float)
    if a.ndim == 0 or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(np.abs(a.sum(-1) - 1.0) > SUM_TOL):
        raise ValueError(f"{name} must be finite, non-negative and sum to one")
    return a


def reader_vector(p, name):
    """A reader's probability vector (or rows of them), checked and floored (SPEC 5)."""
    return floor_norm(distribution(p, name))


def world_weights(weights):
    """World-model weights P_hat(o): checked, never floored. A zero weight is a legitimate point mass."""
    w = distribution(weights, "weights")
    if w.ndim != 1:
        raise ValueError("weights must be one number per outcome")
    return w / w.sum()


def reader_rows(weights, posteriors):
    """Weights and floored posteriors, one row of posteriors per outcome."""
    w = world_weights(weights)
    q = reader_vector(posteriors, "posteriors")
    if q.ndim != 2 or q.shape[0] != w.size:
        raise ValueError("posteriors must have one row per outcome")
    return w, q


def reader_pair(prior, weights, posteriors):
    """Floored prior, weights, and floored posteriors of one decision point, with shapes checked."""
    w, q = reader_rows(weights, posteriors)
    b = reader_vector(prior, "prior")
    if b.ndim != 1 or b.size != q.shape[1]:
        raise ValueError("prior and posteriors must range over the same classes")
    return b, w, q


@dataclass(frozen=True)
class Projection:
    """Result of the projection of SPEC 5.3.

    prior, posteriors   the vectors the projection used: floored reader vectors, or derived ones
    weights             P_hat(o)
    u                   class weights, rescaled to sum to one (only ratios matter)
    log_z               log Z_o per outcome, Z_o = sum_y u(y) q_o(y)
    corrected           q'_o(y) = u(y) q_o(y) / Z_o
    kl_inc              location incoherence, sum_y b log u - sum_o P_hat log Z_o
    converged           largest class-marginal violation <= tol; when False the raw reader is used
    iterations          number of updates of u
    max_violation       max_y |sum_o P_hat(o) q'_o(y) - b(y)| for the returned u
    """

    prior: np.ndarray
    weights: np.ndarray
    posteriors: np.ndarray
    u: np.ndarray
    log_z: np.ndarray
    corrected: np.ndarray
    kl_inc: float
    converged: bool
    iterations: int
    max_violation: float


def ipf(prior, weights, posteriors, tol, max_iter):
    """Iterative proportional fitting on strictly positive inputs, without flooring (SPEC 5.3; PROOF section 2).

    u <- u x b / (imagined mean of the corrected posteriors), rescaled, until the largest
    class-marginal violation is at most tol or max_iter updates have been made. The outcome
    marginal is fitted exactly at every step through Z_o. Non-convergence is reported, not raised.
    """
    u = np.full(prior.size, 1.0 / prior.size)
    iterations = 0
    while True:
        z = posteriors @ u
        corrected = posteriors * u / z[:, None]
        mean = weights @ corrected
        violation = float(np.abs(mean - prior).max())
        if violation <= tol or iterations >= max_iter:
            break
        u = u * prior / mean
        u = u / u.sum()
        iterations += 1
    log_z = np.log(z)
    kl_inc = float(prior @ np.log(u) - weights @ log_z)
    return Projection(prior, weights, posteriors, u, log_z, corrected, kl_inc, violation <= tol, iterations, violation)


def project(prior, weights, posteriors, tol=TOL, max_iter=MAX_ITER):
    """CEE-Read (SPEC 5.3): project the raw joint P_hat(o) q_o(y) onto joints with marginals b and P_hat.

    prior and posteriors are reader vectors and are floored; weights are not. Returns a Projection.
    KL_inc is a divergence, and non-negative, only once the marginal constraint holds (PROOF
    section 3); with converged False the caller uses the raw reader.
    """
    return ipf(*reader_pair(prior, weights, posteriors), tol, max_iter)


def apply_to_real(u, q_real):
    """Corrected belief after the real result, with the same class weights (SPEC 5.3).

    Returns (q'_real, log Z_real) with q'_real(y) = u(y) q_real(y) / Z_real. q_real is floored.
    """
    q = reader_vector(q_real, "q_real")
    u = np.asarray(u, dtype=float)
    if q.ndim != 1 or u.shape != q.shape:
        raise ValueError("u and q_real must range over the same classes")
    z = float(q @ u)
    return u * q / z, float(np.log(z))
