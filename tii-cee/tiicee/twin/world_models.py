"""World models of the registry (specification section 6.2), for the twin.

Every model answers outcomes(history, test, k, rng): the exact predictive when k is None, k draws
from it otherwise. Models built on a generative model also give its class-conditional form,
likelihood(history, test), and the information budget B-model, budget(history, test).

    W0       the true model
    W1       class-conditional independent count model, fitted on training patients. It has no notion
             of a repeated measurement: a duplicate of a result it has seen is predicted afresh
    W2       latent phenotype model fitted on training patients by EM; exact predictive of the fitted
             model, with evidence identity. W2-cc is the same fitted model read through likelihood()
             and budget()
    mirror   the exact predictive of the true model with each item's finding likelihood mixed with
             that of its mirror in the information ranking (the corruption of the snapshot study)
    temperature, matched pair   the true predictive with its findings reshaped at each (history, test)

Evidence identity: every model except W1 gives a point mass to a result already in the history, for
the same test or for another test of the same item. The controlled errors leave the probability of
UNAVAILABLE at its true value, so they are errors about content only.

Training patients come from the fitting stream of the training split (tiicee.twin.env.stream).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from tiicee.interfaces import UNAVAILABLE, Predictive
from tiicee.twin.env import Twin
from tiicee.twin.model import TwinModel


def _information(joint: np.ndarray) -> float:
    """Mutual information of a joint distribution given as [class, outcome]."""
    joint = joint / joint.sum()
    independent = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
    positive = joint > 0
    return float((joint[positive] * np.log(joint[positive] / independent[positive])).sum())


class _WorldModel:
    name = "world model"

    def _exact(self, history, test) -> Predictive:
        raise NotImplementedError

    def outcomes(self, history, test, k: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> Predictive:
        exact = self._exact(history, test)
        if k is None:
            return exact
        if rng is None:
            raise ValueError("k draws need a random generator")
        drawn = rng.choice(len(exact.outcomes), size=k, p=exact.weights)
        return Predictive(tuple(exact.outcomes[j] for j in drawn), np.full(k, 1.0 / k), merged=False)


# ---------------------------------------------------------------------------------------------
# W0, and every model that is the exact predictive of some generative model
# ---------------------------------------------------------------------------------------------
class ExactWorldModel(_WorldModel):
    """The exact predictive of one generative model: the truth (W0), a fitted model (W2), a corrupted one."""

    def __init__(self, twin: Twin, name: str, trace: tuple = ()):
        self.twin = twin
        self.name = name
        self.trace = trace              # objective per EM iteration, for a fitted model

    def _exact(self, history, test) -> Predictive:
        return self.twin.true_predictive(history, test)

    def likelihood(self, history, test):
        """Class-conditional form: (outcomes, P(o | class, h, a) as [class, outcome])."""
        return self.twin.class_predictive(history, test)

    def budget(self, history, test) -> float:
        """B-model: the model's own I(Y; O | h, a). Exact when the model is the truth."""
        return self.twin.true_information(history, test)


def w0(twin: Twin) -> ExactWorldModel:
    return ExactWorldModel(twin, "W0")


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------
def _training(twin: Twin, seed: int, n_train: int):
    """Class, finding and availability of every test for the training patients of the fitting stream."""
    m = twin.model
    patients = twin.sample_patients(n_train, seed, "train", purpose="fitting")
    y = np.array([m.classes.index(p.diagnosis) for p in patients])
    finding = np.array([[m.findings.index(f) for f in p.findings] for p in patients])
    available = np.array([p.available for p in patients], dtype=bool)
    return y, finding, available


class CountWorldModel(_WorldModel):
    """W1: P(y), P(finding | test, y) and P(available | test), with tests independent given the class."""

    name = "W1"

    def __init__(self, tests, findings, prior, lik, availability):
        self.tests, self.findings = tuple(tests), tuple(findings)
        self.prior, self.lik, self.availability = prior, lik, availability      # [class], [test, class, finding], [test]
        self._outcomes = self.findings + (UNAVAILABLE,)
        self._test = {name: t for t, name in enumerate(self.tests)}
        self._finding = {name: s for s, name in enumerate(self.findings)}

    def _belief(self, history) -> np.ndarray:
        belief = self.prior
        for test, outcome in history:
            if outcome != UNAVAILABLE:                       # availability does not depend on the class here
                belief = belief * self.lik[self._test[test], :, self._finding[outcome]]
        return belief / belief.sum()

    def likelihood(self, history, test):
        t = self._test[test]
        a = self.availability[t]
        absent = np.full((len(self.prior), 1), 1.0 - a)
        return self._outcomes, np.concatenate([a * self.lik[t], absent], axis=1)

    def _exact(self, history, test) -> Predictive:
        outcomes, lik = self.likelihood(history, test)
        p = self._belief(history) @ lik
        return Predictive(outcomes, p / p.sum())

    def budget(self, history, test) -> float:
        _, lik = self.likelihood(history, test)
        return _information(self._belief(history)[:, None] * lik)


def fit_w1(twin: Twin, seed: int, n_train: int = 2000, smoothing: float = 1.0) -> CountWorldModel:
    """Counts with Laplace smoothing, as fit_count_model of the snapshot study does from its logged tuples."""
    m = twin.model
    y, finding, available = _training(twin, seed, n_train)
    n_class, n_find = len(m.classes), len(m.findings)
    by_class = np.eye(n_class)[y]                                           # [patient, class]
    seen = (finding[:, :, None] == np.arange(n_find)) & available[:, :, None]   # [patient, test, finding]
    counts = np.einsum("ny,nts->tys", by_class, seen)
    prior = (by_class.sum(0) + smoothing) / (n_train + smoothing * n_class)
    lik = (counts + smoothing) / (counts.sum(-1, keepdims=True) + smoothing * n_find)
    availability = (available.sum(0) + 1.0) / (n_train + 2.0)
    return CountWorldModel(m.tests, m.findings, prior, lik, availability)


def fit_w2(twin: Twin, seed: int, n_train: int = 2000, smoothing: float = 1.0, max_iter: int = 500,
           tol: float = 1e-9) -> ExactWorldModel:
    """W2 and W2-cc: the latent phenotype model, fitted by EM on labeled training patients.

    The fitted family is the twin's own structure with every number unknown: a class prior, a
    phenotype given the class, each item's findings given the phenotype, and one availability per
    item. The class and the results are observed, the phenotype is not. The prior and availability
    are smoothed counts. EM runs on the rest from a fixed start (each phenotype begins as the class
    of the same index, with the class-conditional counts as its findings) and maximizes

        sum over patients of log sum_e P(e | y) prod_i P(s_i | e)  +  smoothing x sum of log parameters,

    whose values per iteration are returned as .trace. The fit is deterministic given the patients.
    Each item is counted once however many tests read it, and the returned model is the exact
    predictive of the fitted model, so it keeps evidence identity.
    """
    m = twin.model
    n_class, n_pheno, n_find = len(m.classes), len(m.phenotypes), len(m.findings)
    if n_pheno != n_class:
        raise ValueError("the fit starts each phenotype at the class of the same index")
    if not smoothing > 0:
        raise ValueError("smoothing must be positive")
    y, finding, available = _training(twin, seed, n_train)
    first = [m.item_of.index(i) for i in range(len(m.items))]                # one test per item
    finding, available = finding[:, first], available[:, first]
    by_class = np.eye(n_class)[y]
    seen = ((finding[:, :, None] == np.arange(n_find)) & available[:, :, None]).astype(float)   # [patient, item, finding]

    prior = (by_class.sum(0) + smoothing) / (n_train + smoothing * n_class)
    availability = (available.sum(0) + 1.0) / (n_train + 2.0)
    counts = np.einsum("ne,nis->ies", by_class, seen)
    lik = (counts + smoothing) / (counts.sum(-1, keepdims=True) + smoothing * n_find)       # [item, phenotype, finding]
    pheno = np.full((n_class, n_pheno), 0.1 / (n_pheno - 1))
    np.fill_diagonal(pheno, 0.9)

    trace = []
    while True:
        log_joint = np.log(pheno)[y] + np.einsum("nis,ies->ne", seen, np.log(lik))           # [patient, phenotype]
        top = log_joint.max(1, keepdims=True)
        log_evidence = top[:, 0] + np.log(np.exp(log_joint - top).sum(1))
        trace.append(float(log_evidence.sum() + smoothing * (np.log(pheno).sum() + np.log(lik).sum())))
        if (len(trace) > 1 and trace[-1] - trace[-2] < tol * n_train) or len(trace) > max_iter:
            break
        responsibility = np.exp(log_joint - log_evidence[:, None])
        pheno_counts = by_class.T @ responsibility
        pheno = (pheno_counts + smoothing) / (pheno_counts.sum(1, keepdims=True) + smoothing * n_pheno)
        lik_counts = np.einsum("ne,nis->ies", responsibility, seen)
        lik = (lik_counts + smoothing) / (lik_counts.sum(-1, keepdims=True) + smoothing * n_find)

    fitted = replace(
        m, name=f"{m.name}/W2(seed={seed}, n={n_train})", prior=prior, phenotype_given_class=pheno,
        finding_lik=np.repeat(lik[:, None], n_class, axis=1),
        availability=np.repeat(availability[:, None], n_class, axis=1),
    )
    return ExactWorldModel(Twin(fitted), "W2", trace=tuple(trace))


# ---------------------------------------------------------------------------------------------
# Controlled errors
# ---------------------------------------------------------------------------------------------
def mirror_map(model: TwinModel) -> dict:
    """Each item's mirror: items ranked by the information of their finding at the prior, rank k paired with n-1-k.

    Availability is left out of the ranking, as in corrupt_model of the snapshot study.
    """
    always = Twin(replace(model, availability=np.ones_like(model.availability)))
    information = [always.true_information((), model.tests[model.item_of.index(i)]) for i in range(len(model.items))]
    ranked = sorted(range(len(model.items)), key=lambda i: -information[i])
    return {model.items[a]: model.items[b] for a, b in zip(ranked, reversed(ranked))}


def mirror_corrupt(twin: Twin, alpha: float) -> ExactWorldModel:
    """Mirror corruption of strength alpha in [0, 1]: a wrong generative model, and its exact predictive.

    At alpha = 1 the model believes the most informative item behaves like the least informative one
    and the reverse. The prior, the phenotype model and availability are the truth's.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    m = twin.model
    mirror = mirror_map(m)
    partner = [m.items.index(mirror[name]) for name in m.items]
    lik = (1.0 - alpha) * m.finding_lik + alpha * m.finding_lik[partner]
    corrupted = replace(m, name=f"{m.name}/mirror({alpha:g})", finding_lik=lik)
    return ExactWorldModel(Twin(corrupted), f"mirror({alpha:g})")


class _Reshaped(_WorldModel):
    """The true predictive with the distribution of its findings, given that there is a result, replaced.

    UNAVAILABLE keeps its true probability and a result already in the history stays certain.
    """

    def __init__(self, twin: Twin, name: str):
        self.twin = twin
        self.name = name

    def _reshape(self, history, test, findings, p, mass) -> np.ndarray:
        raise NotImplementedError

    def _exact(self, history, test) -> Predictive:
        exact = self.twin.true_predictive(history, test)
        found = np.array([o != UNAVAILABLE for o in exact.outcomes])
        if found.sum() < 2:
            return exact
        weights = exact.weights.copy()
        mass = weights[found].sum()
        findings = tuple(o for o in exact.outcomes if o != UNAVAILABLE)
        weights[found] = mass * self._reshape(history, test, findings, weights[found] / mass, mass)
        return Predictive(exact.outcomes, weights)


class _Temperature(_Reshaped):
    def __init__(self, twin: Twin, temperature: float):
        super().__init__(twin, f"temperature({temperature:g})")
        self.temperature = float(temperature)

    def _reshape(self, history, test, findings, p, mass) -> np.ndarray:
        powered = p ** (1.0 / self.temperature)
        return powered / powered.sum()


def temperature(twin: Twin, temperature: float) -> _Temperature:
    """Temperature miscalibration: findings proportional to the truth's raised to 1 / temperature.

    Above 1 the model is too flat, below 1 too sharp; 1 is the truth.
    """
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    return _Temperature(twin, temperature)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _reader_axes(posteriors: np.ndarray, p: np.ndarray):
    """Directions in the space of finding probabilities that a reader ignores and responds to most.

    posteriors[s] is the reader's belief after finding s and p the true distribution of the
    findings. Moving the predictive by d (summing to zero) moves the reader's imagined mean belief
    by d @ posteriors. With d = sqrt(p) * w, a small move has KL from the truth |w|^2 / 2 whatever
    its direction, so the singular vectors of w -> (sqrt(p) * w) @ posteriors compare directions at
    equal KL: the top one moves the imagined mean most, a null one leaves it exactly where it was.
    """
    root = np.sqrt(p)
    unit_root = _unit(root)                               # this direction changes the total mass; it is excluded
    response = ((posteriors - p @ posteriors) * root[:, None]).T
    _, singular, vt = np.linalg.svd(response, full_matrices=True)
    scale = singular[0] if len(singular) else 0.0
    rank = int((singular > 1e-11 * scale).sum()) if scale > 1e-12 else 0
    null = vt[rank:] - np.outer(vt[rank:] @ unit_root, unit_root)
    _, kept, basis = np.linalg.svd(null, full_matrices=False)
    basis = basis[kept > 1e-8]
    if not len(basis):
        raise ValueError("the reader responds to every direction of this test's findings; "
                         "an ignored direction needs more findings than classes")
    ignored = None
    for k in range(len(p)):                               # a fixed choice within the ignored subspace
        candidate = basis.T @ basis[:, k]
        if np.linalg.norm(candidate) > 1e-6:
            ignored = _unit(candidate)
            break
    if rank:
        top = _unit(vt[0] - (vt[0] @ unit_root) * unit_root)
    else:
        top = ignored                                     # a reader that does not respond ignores everything
    directions = []
    for w in (ignored, top):
        d = root * w
        directions.append(d if d[np.argmax(np.abs(d))] > 0 else -d)
    return directions


class _MatchedMember(_Reshaped):
    def __init__(self, twin: Twin, reader, kl: float, responsive: bool):
        super().__init__(twin, f"matched({'responsive' if responsive else 'ignored'}, kl={kl:g})")
        self.reader, self.kl, self.responsive = reader, float(kl), responsive

    def _reshape(self, history, test, findings, p, mass) -> np.ndarray:
        posteriors = np.stack([np.asarray(self.reader.read(history, test, s), dtype=float) for s in findings])
        direction = _reader_axes(posteriors, p)[1 if self.responsive else 0]
        target = self.kl / mass                           # UNAVAILABLE contributes nothing to the divergence

        def divergence(step):
            q = p + step * direction
            return np.inf if (q <= 0).any() else float((p * np.log(p / q)).sum())

        lo, hi = 0.0, float(min(p[s] / -direction[s] for s in range(len(p)) if direction[s] < 0))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if divergence(mid) < target else (lo, mid)
        q = p + lo * direction
        return q / q.sum()


def matched_pair(twin: Twin, reader, kl: float = 0.02):
    """Two world models at the same exact KL from the truth, differing in what the reader makes of the error.

    At every (history, test) both members move the true distribution of the findings along one
    direction until KL(truth || model) equals kl nats, found by bisection to far below the 0.005
    nats of specification section 9.4. The first member moves along a direction the reader ignores:
    the reader's imagined mean belief is exactly the one it has under the truth. The second moves
    along the direction the reader responds to most at equal KL (see _reader_axes). Where the result
    is already in the history, or only one finding is possible, both members are the truth.

    Returns (ignored, responsive).
    """
    if not kl > 0:
        raise ValueError("kl must be positive")
    return _MatchedMember(twin, reader, kl, responsive=False), _MatchedMember(twin, reader, kl, responsive=True)
