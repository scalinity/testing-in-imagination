"""The gain identity and the realized gain (SPEC 5.4; PROOF section 4).

Pure algebra on the outputs of a projection: nothing here is floored. Pass the prior the
projection used (Projection.prior), so that the incoherence term is the reported KL_inc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GainTerms:
    """The three terms of the gain identity and their sum (SPEC 5.4).

    total is the expected improvement in log score, E_star[log q'_O(Y) - log q_O(Y)], for the
    class weights supplied. The identity is exact for any positive weights and any truth; only
    with projected weights is the incoherence term a divergence (KL_inc >= 0).
    """

    incoherence: float          # F(u) = sum_y b log u - sum_o P_hat log Z
    prior_calibration: float    # sum_y (b_star - b) log u
    world_model: float          # -sum_o (P_star - P_hat) log Z
    total: float


def gain_terms(prior, weights, u, log_z, b_star, p_star):
    """Incoherence, prior-calibration and world-model terms of the expected gain (SPEC 5.4; PROOF section 4).

    u are positive class weights and log_z the matching log Z_o, one per outcome; b_star and p_star
    are the true label and outcome distributions given the information the projection used. All
    four terms are unchanged if u is rescaled and log_z recomputed.
    """
    b, w = np.asarray(prior, dtype=float), np.asarray(weights, dtype=float)
    log_u, log_z = np.log(np.asarray(u, dtype=float)), np.asarray(log_z, dtype=float)
    b_star, p_star = np.asarray(b_star, dtype=float), np.asarray(p_star, dtype=float)
    if not (b.shape == log_u.shape == b_star.shape and w.shape == log_z.shape == p_star.shape and b.ndim == w.ndim == 1):
        raise ValueError("class vectors and outcome vectors must have matching lengths")
    incoherence = float(b @ log_u - w @ log_z)
    prior_calibration = float((b_star - b) @ log_u)
    world_model = float(-((p_star - w) @ log_z))
    return GainTerms(incoherence, prior_calibration, world_model, incoherence + prior_calibration + world_model)


def realized_gain(u, log_z_real, y_true):
    """g = log u(y_true) - log Z_real, the per-case change in log score (SPEC 5.4, consequence 3).

    Observable on labeled data from factual outcomes alone; its mean estimates the identity's
    total. y_true is the index of the true class. Unchanged if u is rescaled and Z recomputed.
    """
    return float(np.log(np.asarray(u, dtype=float)[y_true]) - log_z_real)
