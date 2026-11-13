"""The tempered family, the budget-implied scale tau_B, CEE-Scale and tau_ratio (SPEC 5.5, 5.11; PROOF section 6).

There is deliberately no function that minimizes KL_inc over tau: an agent that never updates is
coherent, so that rule selects it (SPEC 5.5, "must not be used for anything").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import MAX_ITER, TOL, Projection, ipf, kl, reader_pair, reader_vector
from .measures import responsiveness_of

VARIANT_BOUNDS = {"shrink-only": (1.0, 16.0), "two-sided": (0.125, 16.0)}      # SPEC 5.11
TAU_RATIO_GRID = (0.5, 0.67, 1, 1.5, 2, 3, 5, 8)                               # SPEC 5.5
RTOL = 0.01             # SPEC 5.11: bisection on log tau to 1% relative tolerance
GRID_RATIO = 2.0        # spacing of the tau grid on which r' is checked for monotonicity
MONOTONE_SLACK = 100    # r' may rise by this many projection tolerances before it counts as a rise
R_ZERO = 1e-12          # a responsiveness at or below this many nats is a reader that does not move


def tempered(prior, posteriors, tau):
    """b (q / b)^(1/tau), normalized, for strictly positive inputs; no flooring."""
    if not (np.isfinite(tau) and tau > 0):
        raise ValueError("tau must be positive and finite")
    log_t = np.log(prior) + (np.log(posteriors) - np.log(prior)) / tau
    t = np.exp(log_t - log_t.max(-1, keepdims=True))
    return t / t.sum(-1, keepdims=True)


def temper(prior, posteriors, tau):
    """Tempered family q_o(tau) proportional to b (q_o / b)^(1/tau) (SPEC 5.5).

    tau above 1 flattens the reader's reactions; tau below 1 amplifies them; tau = 1 is the reader.
    posteriors may be one vector or one row per outcome. prior and posteriors are floored.
    """
    b, q = reader_vector(prior, "prior"), reader_vector(posteriors, "posteriors")
    if b.ndim != 1 or q.shape[-1] != b.size:
        raise ValueError("prior and posteriors must range over the same classes")
    return tempered(b, q, tau)


def _projected(prior, weights, posteriors, tau, tol, max_iter):
    """Project the pair tempered at tau. Returns the projection and r'(tau) = sum_o P_hat KL(q'_o(tau) || b)."""
    projection = ipf(prior, weights, tempered(prior, posteriors, tau), tol, max_iter)
    return projection, float(weights @ kl(projection.corrected, prior))


def projected_responsiveness(prior, weights, posteriors, tau, tol=TOL, max_iter=MAX_ITER):
    """r'(tau): responsiveness of the projected tempered pair (SPEC 5.11).

    For a converged projection this is the mutual information of the coherent joint, in nats.
    """
    return _projected(*reader_pair(prior, weights, posteriors), tau, tol, max_iter)[1]


@dataclass(frozen=True)
class TauBudget:
    """The budget-implied scale and how it was found (SPEC 5.5, 5.11).

    tau              the largest tau in the interval with r'(tau) >= budget; the lower end if none
    path             "solved": r' was non-increasing on the grid and the crossing was bisected
                     "clipped_low": r' is below the budget everywhere; tau is the lower end
                     "clipped_high": r' meets the budget at the upper end; tau is the upper end
                     "fallback": r' rose somewhere on the grid; the last crossing was bisected
    monotone         r' non-increasing in tau over the converged grid points
    converged        every projection used in the search converged
    grid, r_grid     the tau grid (factor 2 apart, plus the upper end) and r' on it
    grid_converged   which grid projections converged
    """

    tau: float
    path: str
    monotone: bool
    converged: bool
    grid: np.ndarray
    r_grid: np.ndarray
    grid_converged: np.ndarray


def _tau_grid(lo, hi):
    grid = [float(lo)]
    while grid[-1] * GRID_RATIO < hi * (1.0 - 1e-12):
        grid.append(grid[-1] * GRID_RATIO)
    return np.array(grid + [float(hi)])


def search(prior, weights, posteriors, budget, bounds, rtol, tol, max_iter):
    """tau_budget for strictly positive inputs; no flooring."""
    lo, hi = bounds
    if not (0 < lo < hi and np.isfinite(hi)):
        raise ValueError("bounds must satisfy 0 < low < high")
    if not (np.isfinite(budget) and budget >= 0):
        raise ValueError("the information budget must be finite and non-negative")
    flags = []

    def r_prime(tau):
        projection, r = _projected(prior, weights, posteriors, tau, tol, max_iter)
        flags.append(projection.converged)
        return r

    grid = _tau_grid(lo, hi)
    r_grid = np.array([r_prime(tau) for tau in grid])
    grid_converged = np.array(flags, dtype=bool)
    monotone = bool(np.all(np.diff(r_grid[grid_converged]) <= MONOTONE_SLACK * tol))
    meets = np.flatnonzero(r_grid >= budget)
    if meets.size == 0:
        tau, path = float(lo), "clipped_low"
    elif meets[-1] == grid.size - 1:
        tau, path = float(hi), "clipped_high"
    else:
        low, high = np.log(grid[meets[-1]]), np.log(grid[meets[-1] + 1])      # r'(low) >= budget > r'(high)
        while high - low > np.log1p(rtol):
            mid = 0.5 * (low + high)
            if r_prime(float(np.exp(mid))) >= budget:
                low = mid
            else:
                high = mid
        tau, path = float(np.exp(0.5 * (low + high))), ("solved" if monotone else "fallback")
    return TauBudget(tau, path, monotone, all(flags), grid, r_grid, grid_converged)


def tau_budget(prior, weights, posteriors, budget, bounds, rtol=RTOL, tol=TOL, max_iter=MAX_ITER):
    """Budget-implied scale tau_B: the tau in `bounds` at which the projected pair carries `budget` nats (SPEC 5.5, 5.11).

    r'(tau) is evaluated on a grid across the interval and the crossing r' = budget is bisected on
    log tau until the bracket is narrower than rtol, then clipped to the interval. Monotonicity of
    r' is assumed by the specification, not proven: if r' rises anywhere on the grid, the largest
    tau with r' >= budget is taken and the path is reported as "fallback". A rise narrower than one
    grid step is not detected. tau_B above 1 means the reader reacts more than the test can
    justify; below 1, less. bounds is (low, high); (1/8, 16) is the regime-map diagnostic.
    """
    return search(*reader_pair(prior, weights, posteriors), budget, bounds, rtol, tol, max_iter)


@dataclass(frozen=True)
class ScaleResult:
    """Result of CEE-Scale (SPEC 5.11): tau_B, how it was found, and the projection at tau_B."""

    tau_b: float
    path: str
    monotone: bool
    search_converged: bool
    projection: Projection      # of the pair tempered at tau_b; its posteriors are the tempered ones

    @property
    def u(self):
        """Class weights at tau_b; with tau_b they are what is applied to the real result."""
        return self.projection.u

    @property
    def log_z(self):
        return self.projection.log_z

    @property
    def repaired(self):
        """The repaired posteriors: the projected posteriors at tau_b."""
        return self.projection.corrected

    @property
    def kl_inc(self):
        """Location incoherence of the tempered pair at tau_b."""
        return self.projection.kl_inc

    @property
    def converged(self):
        """Whether the projection at tau_b converged; when False the raw reader is used."""
        return self.projection.converged


def scale_positive(prior, weights, posteriors, budget, variant, rtol, tol, max_iter):
    """CEE-Scale for strictly positive inputs; no flooring."""
    if variant not in VARIANT_BOUNDS:
        raise ValueError(f"variant must be one of {sorted(VARIANT_BOUNDS)}")
    found = search(prior, weights, posteriors, budget, VARIANT_BOUNDS[variant], rtol, tol, max_iter)
    projection, _ = _projected(prior, weights, posteriors, found.tau, tol, max_iter)
    return ScaleResult(found.tau, found.path, found.monotone, found.converged, projection)


def scale(prior, weights, posteriors, budget, variant, rtol=RTOL, tol=TOL, max_iter=MAX_ITER):
    """CEE-Scale (SPEC 5.11): temper to the information budget, then project.

    variant "shrink-only" allows tau in [1, 16] (reactions flattened, never amplified);
    "two-sided" allows [1/8, 16]. Returns tau_B, the path, the class weights, the repaired
    posteriors and the location incoherence at tau_B. The rule is not chosen by, and does not
    minimize, incoherence, and nothing is guaranteed when the budget is wrong: an overestimated
    budget amplifies a reader past the evidence.
    """
    return scale_positive(*reader_pair(prior, weights, posteriors), budget, variant, rtol, tol, max_iter)


def apply_scale_to_real(prior, q_real, tau, u):
    """Apply the same tau_B and class weights to the real result (SPEC 5.11).

    Returns (belief, log Z_real) with belief proportional to u x b (q_real / b)^(1/tau).
    prior and q_real are floored.
    """
    b, q = reader_vector(prior, "prior"), reader_vector(q_real, "q_real")
    u = np.asarray(u, dtype=float)
    if b.ndim != 1 or not (q.shape == b.shape == u.shape):
        raise ValueError("prior, q_real and u must range over the same classes")
    t = tempered(b, q, tau)
    z = float(t @ u)
    return u * t / z, float(np.log(z))


@dataclass(frozen=True)
class TauRatio:
    """The budget-free scale and the table it was read from (SPEC 5.5)."""

    tau: float                      # grid value minimizing ratio; nan if no grid value is usable
    grid: np.ndarray
    kl_inc: np.ndarray              # location incoherence of the pair tempered at each grid value
    responsiveness: np.ndarray      # r of the tempered raw pair at each grid value
    ratio: np.ndarray               # kl_inc / responsiveness; nan where the reader does not move or the projection failed
    converged: np.ndarray


def tau_ratio(prior, weights, posteriors, grid=TAU_RATIO_GRID, tol=TOL, max_iter=MAX_ITER):
    """Budget-free scale: the grid value minimizing KL_inc(tau) / r(tau) (SPEC 5.5).

    A secondary diagnostic, reported next to tau_B to measure how well it tracks it. It is never a
    repair: in the sandbox it recovered pure scale errors and misbehaved when drift or correlation
    was present. r(tau) is the responsiveness of the tempered raw pair. Ties go to the smaller tau.
    A pair that is coherent at every scale (a symmetric one, for instance) has ratio zero across
    the grid, and the value returned for it carries no information.
    """
    b, w, q = reader_pair(prior, weights, posteriors)
    grid = np.asarray(grid, dtype=float)
    kl_inc, resp, converged = np.empty(grid.size), np.empty(grid.size), np.empty(grid.size, dtype=bool)
    for i, tau in enumerate(grid):
        t = tempered(b, q, tau)
        projection = ipf(b, w, t, tol, max_iter)
        kl_inc[i], converged[i], resp[i] = projection.kl_inc, projection.converged, responsiveness_of(w, t)
    usable = converged & (resp > R_ZERO)
    ratio = np.full(grid.size, np.nan)
    ratio[usable] = kl_inc[usable] / resp[usable]
    value = float(grid[np.nanargmin(ratio)]) if usable.any() else float("nan")
    return TauRatio(value, grid, kl_inc, resp, ratio, converged)
