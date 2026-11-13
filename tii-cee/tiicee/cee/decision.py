"""Coherent value of information, CEE-Select and CEE-Commit (SPEC 5.6 and 5.7).

Select and commit are separate named interventions with their own ablations; neither calls the
other. Ties are broken deterministically by the lowest index.
"""
from __future__ import annotations

import numpy as np

from .core import reader_pair


def _utility_table(utility, n_classes):
    table = np.asarray(utility, dtype=float)
    if table.ndim != 2 or table.shape[1] != n_classes or table.shape[0] == 0:
        raise ValueError("utility must be a table with one row per diagnosis and one column per class")
    return table


def phi(belief, utility):
    """phi(c) = max_d sum_y c(y) U(d, y): the expected utility of committing now (SPEC 5.6).

    utility has one row per diagnosis d and one column per class y. belief may be one vector or
    one row per outcome, in which case one value per row is returned. phi is convex.
    """
    c = np.asarray(belief, dtype=float)
    return (c @ _utility_table(utility, c.shape[-1]).T).max(-1)


def _value_change(prior, weights, posteriors, utility):
    return float(np.asarray(weights, dtype=float) @ phi(posteriors, utility) - phi(prior, utility))


def coherent_voi(prior, weights, corrected, utility):
    """VOI(a | h) = sum_o P_hat(o) phi(q'_o) - phi(b) (SPEC 5.6).

    Non-negative because the projected posteriors average to b (PROOF section 5); with a residual
    m - b the bound degrades by at most max|U| x ||m - b||_1. Pass the outputs of a converged
    projection: Projection.prior, Projection.weights, Projection.corrected. Nothing is floored.
    """
    return _value_change(prior, weights, corrected, utility)


def raw_voi_score(prior, weights, posteriors, utility):
    """The same expression with the uncorrected posteriors, for the baseline (SPEC 5.6).

    Not a value of information: it is bounded below only by phi(b_bar) - phi(b) and can be
    negative. prior and posteriors are reader vectors and are floored.
    """
    return _value_change(*reader_pair(prior, weights, posteriors), utility)


def identity_utility(n_classes):
    """U(d, y) = 1[d = y], the twin's utility (SPEC 5.6)."""
    return np.eye(n_classes)


def recall_aligned_utility(class_freqs):
    """U(d, y) = 1[d = y] / (C pi_y), so that expected utility is mean class recall (SPEC 5.6).

    class_freqs are the training class frequencies pi, or counts; they are normalized to sum to one.
    """
    pi = np.asarray(class_freqs, dtype=float)
    if pi.ndim != 1 or pi.size == 0 or not np.all(np.isfinite(pi)) or np.any(pi <= 0):
        raise ValueError("class frequencies must be positive, one per class")
    pi = pi / pi.sum()
    return np.diag(1.0 / (pi.size * pi))


def expected_charged_cost(p_available, cost):
    """p_available x cost: unavailable results are free and consume a step (SPEC 5.6; REVIEW-006)."""
    if not 0.0 <= p_available <= 1.0:
        raise ValueError("p_available must be a probability")
    return p_available * cost


def net_value(voi, expected_cost, lam=1.0):
    """VOI minus lambda times the expected charged cost (SPEC 5.6). Primary lambda is 1 on normalized cost."""
    return voi - lam * expected_cost


def _worth_requesting(net_values, eligible):
    """Indices of eligible tests with positive net value, and the values."""
    values = np.asarray(net_values, dtype=float)
    allowed = np.ones(values.shape, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if values.ndim != 1 or allowed.shape != values.shape:
        raise ValueError("net_values and eligible must be one entry per test")
    return np.flatnonzero(allowed & (values > 0)), values


def select(net_values, eligible=None):
    """CEE-Select (SPEC 5.7): the eligible test with the highest positive net value, or None.

    Selection is one-step. net_values holds one net value per test; eligible is an optional mask.
    Returns the test's index; ties go to the lowest index.
    """
    candidates, values = _worth_requesting(net_values, eligible)
    if candidates.size == 0:
        return None
    return int(candidates[np.argmax(values[candidates])])


def commit(belief, utility, net_values, eligible=None, budget_reached=False):
    """CEE-Commit (SPEC 5.7): commit when no eligible test has positive net value, or the step or token budget is reached.

    Returns (True, diagnosis) with diagnosis = argmax_d sum_y c(y) U(d, y) under the coherent belief
    c, ties to the lowest index; (False, None) when a test is still worth requesting.
    """
    candidates, _ = _worth_requesting(net_values, eligible)
    if not budget_reached and candidates.size > 0:
        return False, None
    c = np.asarray(belief, dtype=float)
    return True, int(np.argmax(_utility_table(utility, c.size) @ c))
