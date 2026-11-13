"""Audit quantities of SPEC 5.2 and the legacy entropy-change surrogate (PROOF section 5).

Every function here takes reader vectors and floors them (SPEC 5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import entropy, kl, reader_pair, reader_rows, reader_vector


def imagined_mean(weights, posteriors):
    """b_bar = sum_o P_hat(o) q_o (SPEC 5.2)."""
    w, q = reader_rows(weights, posteriors)
    return w @ q


def drift(prior, weights, posteriors):
    """d = b_bar - b, as a vector over classes (SPEC 5.2).

    A reader with d != 0 moves in a direction that does not depend on what the result says.
    """
    b, w, q = reader_pair(prior, weights, posteriors)
    return w @ q - b


def responsiveness_of(weights, posteriors):
    """r for vectors that are already positive; no flooring."""
    return float(weights @ kl(posteriors, weights @ posteriors))


def responsiveness(weights, posteriors):
    """r = sum_o P_hat(o) KL(q_o || b_bar) (SPEC 5.2). Zero for a reader that does not move."""
    return responsiveness_of(*reader_rows(weights, posteriors))


def reaction_kl(prior, posterior):
    """KL(q || b): how far one result moved the reader (SPEC 5.2).

    The measurement behind both model-free probes: the reaction to UNAVAILABLE where missingness
    is known to be uninformative, and the reaction to an exact re-exposure of evidence already in
    the history. Both should be zero.
    """
    b, q = reader_vector(prior, "prior"), reader_vector(posterior, "posterior")
    if b.ndim != 1 or q.shape != b.shape:
        raise ValueError("prior and posterior must range over the same classes")
    return float(kl(q, b))


@dataclass(frozen=True)
class EntropyChange:
    """The legacy entropy-change surrogate and its decomposition (PROOF section 5).

    score = reaction_term + drift_term. The reaction term is non-negative. The drift term vanishes
    for a coherent pair and has no sign otherwise, so the score is not an expected information
    gain unless the pair is coherent.
    """

    score: float            # H(b) - sum_o P_hat(o) H(q_o)
    reaction_term: float    # sum_o P_hat(o) KL(q_o || b)
    drift_term: float       # sum_y (b_bar - b)(y) log b(y)


def entropy_change_surrogate(prior, weights, posteriors):
    """H(b) - sum_o P_hat(o) H(q_o), with the two terms it decomposes into (PROOF section 5).

    The score the existing study used to rank tests. It is reported for comparison with the raw
    agent and is not a value of information: it can be negative.
    """
    b, w, q = reader_pair(prior, weights, posteriors)
    return EntropyChange(
        score=float(entropy(b) - w @ entropy(q)),
        reaction_term=float(w @ kl(q, b)),
        drift_term=float((w @ q - b) @ np.log(b)),
    )
