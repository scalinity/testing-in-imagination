"""Env v1 generative parameters: priors, test likelihoods, availability.

v1 replaces v0's "every test always shows the true disease" with a discrete
finding per (patient, test):

    y ~ PRIOR                         true diagnosis
    e | y ~ phenotype_given_y         presenting phenotype (atypical: e != y)
    s | e, a ~ LIKELIHOOD[a][e]       finding in FINDINGS (4 diseases + nonspecific)
    available_a ~ Bernoulli(AVAILABILITY[a])   only when missingness is on

Findings depend on the phenotype e, not on y, so an atypical patient looks
like another disease on *every* test. That caps Bayes accuracy at roughly
1 - P_ATYPICAL even with all twelve tests, the analogue of the paper's
SFT-all information upper bound (92.8 mean acc).

Every number here is a synthetic design choice, not a MIMIC statistic, except
where a comment cites a source. Sanity checks against the paper live in
``extension.experiments.sanity``.
"""

from __future__ import annotations

import random
from typing import Dict, Mapping, Tuple

from .actions import A_DIAG, A_TEST, COST_USD

NONSPECIFIC = "nonspecific"
FINDINGS: Tuple[str, ...] = A_DIAG + (NONSPECIFIC,)

# Class counts of MIMIC-IV-Ext-CDM (Hager et al., 2024): 2,400 patients.
# Verify against the downloaded dataset before citing in the write-up.
_MIMIC_CDM_COUNTS: Dict[str, int] = {
    "appendicitis": 957,
    "cholecystitis": 648,
    "diverticulitis": 257,
    "pancreatitis": 538,
}
PRIOR_MIMIC_CDM: Dict[str, float] = {
    y: n / sum(_MIMIC_CDM_COUNTS.values()) for y, n in _MIMIC_CDM_COUNTS.items()
}

# Probability a patient presents like a different disease on every test.
P_ATYPICAL = 0.07

# Per-test profile: (P(s = e) for e in A_DIAG order, P(s = d) for each d != e).
# The remainder is P(s = nonspecific). Imaging is strong (CT/MRI broadly,
# ultrasound for cholecystitis), the radiograph and most panels are weak,
# LFTs point at biliary/pancreatic disease, metabolic panels at pancreatitis.
_PROFILE: Dict[str, Tuple[Tuple[float, float, float, float], float]] = {
    #                              app   chole  div   panc   confusion
    "Physical Examination":       ((0.55, 0.50, 0.45, 0.40), 0.08),
    "CT":                         ((0.88, 0.60, 0.88, 0.80), 0.02),
    "MRI":                        ((0.90, 0.80, 0.85, 0.80), 0.02),
    "Radiograph":                 ((0.10, 0.10, 0.10, 0.12), 0.05),
    "Ultrasound":                 ((0.70, 0.88, 0.25, 0.30), 0.04),
    "Complete Blood Count":       ((0.30, 0.25, 0.30, 0.20), 0.12),
    "Basic Metabolic Panel":      ((0.10, 0.10, 0.10, 0.30), 0.06),
    "Comprehensive Metabolic Panel": ((0.12, 0.20, 0.12, 0.40), 0.06),
    "Renal Function Panel":       ((0.08, 0.08, 0.10, 0.25), 0.06),
    "Liver Function Panel":       ((0.10, 0.65, 0.10, 0.40), 0.05),
    "Urinalysis":                 ((0.15, 0.05, 0.10, 0.05), 0.05),
    "Electrolyte Panel":          ((0.05, 0.05, 0.10, 0.20), 0.05),
}


def _build_likelihood() -> Dict[str, Dict[str, Dict[str, float]]]:
    lik: Dict[str, Dict[str, Dict[str, float]]] = {}
    for test in A_TEST:
        hits, confusion = _PROFILE[test]
        lik[test] = {}
        for e, hit in zip(A_DIAG, hits):
            row = {d: (hit if d == e else confusion) for d in A_DIAG}
            row[NONSPECIFIC] = 1.0 - sum(row.values())
            if row[NONSPECIFIC] < 0:
                raise ValueError(f"Profile for {test!r}/{e!r} exceeds 1")
            lik[test][e] = row
    return lik


# LIKELIHOOD[test][phenotype][finding]
LIKELIHOOD: Dict[str, Dict[str, Dict[str, float]]] = _build_likelihood()

# P(test is in the chart). Synthetic, calibrated so the expected cost of
# ordering every available test is ~$3.8k, close to the paper's SFT-all
# average test cost ($3792.79, Table 1). PE is always available (it is
# part of p_0-adjacent history in practice); MRI is rare.
AVAILABILITY: Dict[str, float] = {
    "Physical Examination": 1.00,
    "CT": 0.75,
    "MRI": 0.10,
    "Radiograph": 0.35,
    "Ultrasound": 0.50,
    "Complete Blood Count": 0.95,
    "Basic Metabolic Panel": 0.75,
    "Comprehensive Metabolic Panel": 0.55,
    "Renal Function Panel": 0.45,
    "Liver Function Panel": 0.70,
    "Urinalysis": 0.60,
    "Electrolyte Panel": 0.60,
}

# LA-CDM's effective budget: max_steps 13 with step counted from 0 gives
# 14 actions (REPRODUCTION.md F-3).
V1_MAX_STEPS = 14


def phenotype_given_y(p_atypical: float = P_ATYPICAL) -> Dict[str, Dict[str, float]]:
    """P(e | y): typical with 1 - p_atypical, else uniform over the others."""
    other = p_atypical / (len(A_DIAG) - 1)
    return {
        y: {e: (1.0 - p_atypical if e == y else other) for e in A_DIAG}
        for y in A_DIAG
    }


def expected_all_available_cost_usd(
    availability: Mapping[str, float] = AVAILABILITY,
) -> float:
    return sum(COST_USD[a] * availability[a] for a in A_TEST)


def sample_categorical(dist: Mapping[str, float], rng: random.Random) -> str:
    """Draw a key of ``dist`` in insertion order (deterministic given rng)."""
    u = rng.random()
    acc = 0.0
    last = ""
    for k, p in dist.items():
        acc += p
        last = k
        if u < acc:
            return k
    return last
