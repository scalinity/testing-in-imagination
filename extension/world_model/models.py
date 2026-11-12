"""Constructors for the experiment arms' world models.

- ``fit_count_model``: learned p_phi from logged v1 tuples (counts + Laplace
  smoothing). It conditions findings on the diagnosis y and assumes tests are
  independent given y, which ignores the phenotype correlation in the true
  env: a realistic misspecification that persists even with infinite data.
- ``oracle_model``: the true generative model (latent = phenotype).
- ``corrupt_model``: a deliberately flawed p_phi for the Bad-WM arm.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping

from extension.envs.synthetic_diagnosis.actions import A_DIAG, A_TEST
from extension.envs.synthetic_diagnosis.profiles import (
    AVAILABILITY,
    FINDINGS,
    LIKELIHOOD,
    P_ATYPICAL,
    PRIOR_MIMIC_CDM,
    phenotype_given_y,
)

from .discrete import DiscreteWorldModel, Likelihood


def fit_count_model(
    rows: Iterable[Mapping], smoothing: float = 1.0, name: str = "learned"
) -> DiscreteWorldModel:
    """Fit P(y), P(s | a, y) and P(available | a) from v1 tuple dicts."""
    rows = list(rows)
    if any(r.get("meta", {}).get("env_version") != "v1" for r in rows):
        raise ValueError("fit_count_model needs v1 tuples (meta.finding)")

    y_by_episode = {r["episode_id"]: r["y_true"] for r in rows}
    y_counts = Counter(y_by_episode.values())
    n_ep = len(y_by_episode)
    prior = {
        y: (y_counts[y] + smoothing) / (n_ep + smoothing * len(A_DIAG))
        for y in A_DIAG
    }

    finding_counts: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    requested: Counter = Counter()
    provided: Counter = Counter()
    for r in rows:
        a = r["a_t"]
        requested[a] += 1
        s = r["meta"].get("finding")
        if s is None:
            continue
        provided[a] += 1
        finding_counts[a][r["y_true"]][s] += 1

    lik: Likelihood = {}
    for a in A_TEST:
        lik[a] = {}
        for y in A_DIAG:
            c = finding_counts[a][y]
            total = sum(c.values()) + smoothing * len(FINDINGS)
            lik[a][y] = {s: (c[s] + smoothing) / total for s in FINDINGS}

    availability = {a: (provided[a] + 1) / (requested[a] + 2) for a in A_TEST}
    return DiscreteWorldModel(prior, lik, availability=availability, name=name)


def oracle_model(
    p_atypical: float = P_ATYPICAL,
    prior: Mapping[str, float] = PRIOR_MIMIC_CDM,
    missingness: bool = False,
) -> DiscreteWorldModel:
    """Exact env model with the phenotype e as latent."""
    pe_y = phenotype_given_y(p_atypical)
    prior_e = {e: sum(prior[y] * pe_y[y][e] for y in A_DIAG) for e in A_DIAG}
    y_given_e = {
        e: {y: prior[y] * pe_y[y][e] / prior_e[e] for y in A_DIAG} for e in A_DIAG
    }
    availability = AVAILABILITY if missingness else {a: 1.0 for a in A_TEST}
    return DiscreteWorldModel(
        prior_e, LIKELIHOOD, y_given_e, availability=availability, name="oracle"
    )


def informativeness_ranking(model: DiscreteWorldModel) -> List[str]:
    """Tests from most to least informative about y under the model's prior."""
    b0 = model.initial_belief()
    return sorted(A_TEST, key=lambda a: -model.information_gain(b0, a))


def corrupt_model(base: DiscreteWorldModel, alpha: float) -> DiscreteWorldModel:
    """Mix each test's likelihood with that of its informativeness mirror.

    Tests are ranked by information gain under ``base``; test k is paired
    with test n-1-k, so at alpha=1 the model believes the most informative
    test (e.g. MRI) behaves like the least informative (e.g. Urinalysis) and
    vice versa. The model is wrong about *which* tests are worth paying for,
    which is the failure mode that can make planning worse than model-free.
    Availability and the prior are left intact.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    ranked = informativeness_ranking(base)
    mirror = {a: ranked[len(ranked) - 1 - i] for i, a in enumerate(ranked)}
    lik: Likelihood = {}
    for a in A_TEST:
        b = mirror[a]
        lik[a] = {
            z: {
                s: (1 - alpha) * base.lik[a][z][s] + alpha * base.lik[b][z][s]
                for s in FINDINGS
            }
            for z in base.lik[a]
        }
    return DiscreteWorldModel(
        base.prior_z,
        lik,
        base.y_given_z,
        availability=base.availability,
        name=f"corrupt(alpha={alpha:g})",
    )
