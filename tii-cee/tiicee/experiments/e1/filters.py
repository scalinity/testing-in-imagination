"""The twin's readers and world models, vectorized over many histories.

The same computations as tiicee.twin.readers and tiicee.twin.world_models, carried as arrays so
that a whole level of the outcome tree, or two thousand patients, advances in one step.
tests/test_e1_filters.py holds every class here to the object it mirrors.

A batch of n histories is described by `seen` [n, item], the outcome index recorded for each item
(-1 if the item has not been read), plus one state array per component:

    LatentFilter   weights [n, class, phenotype] of a generative model: the truth (W0, and R0's
                   belief), a fitted model (W2, W2-cc) or a mirror-corrupted one
    CountFilter    W1's own naive-Bayes belief [n, class]
    R4Reader       the simulated reader's belief [n, class]

Outcomes are indexed as in TwinModel.outcomes: the findings, then UNAVAILABLE.
"""
from __future__ import annotations

import numpy as np

from tiicee.twin import world_models as twin_wm
from tiicee.twin.model import TwinModel


def new_seen(n: int, model: TwinModel) -> np.ndarray:
    return np.full((n, len(model.items)), -1, dtype=np.int8)


def record(seen, item_of, tests, outcomes) -> np.ndarray:
    """seen after each history reads its test: the first outcome recorded for an item stays."""
    out = seen.copy()
    rows = np.arange(len(seen))
    items = item_of[tests]
    fresh = out[rows, items] < 0
    out[rows[fresh], items[fresh]] = outcomes[fresh]
    return out


def predictive(joint) -> np.ndarray:
    """P(outcome) from a joint [..., class, outcome] of any positive scale."""
    p = joint.sum(-2)
    return p / p.sum(-1, keepdims=True)


class LatentFilter:
    """Exact filtering in one generative model, as tiicee.twin.env.Twin does it."""

    def __init__(self, model: TwinModel):
        self.model = model
        self.item_of = np.array(model.item_of)
        self.lik = model.outcome_lik()                                  # [item, class, phenotype, outcome]
        self.lik_by_test = self.lik[self.item_of]                        # [test, class, phenotype, outcome]
        self.prior = model.prior[:, None] * model.phenotype_given_class

    def start(self, n: int) -> np.ndarray:
        return np.broadcast_to(self.prior, (n,) + self.prior.shape).copy()

    def joint(self, w, seen, tests) -> np.ndarray:
        """P(class, outcome of each history's test, history) as [n, class, outcome]. A result already read is certain."""
        rows = np.arange(len(w))
        items = self.item_of[tests]
        joint = np.einsum("nye,nyeo->nyo", w, self.lik[items])
        known = seen[rows, items]
        again = np.flatnonzero(known >= 0)
        joint[again] = 0.0
        joint[again, :, known[again]] = w[again].sum(2)
        return joint

    def joint_all(self, w, seen) -> np.ndarray:
        """joint for every test at once: [n, test, class, outcome]."""
        joint = np.einsum("nye,tyeo->ntyo", w, self.lik_by_test)
        known = seen[:, self.item_of]                                   # [n, test]
        rows, tests = np.nonzero(known >= 0)
        joint[rows, tests] = 0.0
        joint[rows, tests, :, known[rows, tests]] = w[rows].sum(2)
        return joint

    def expand(self, w, seen, tests, keep_impossible: bool) -> np.ndarray:
        """Weights after each outcome of each history's test: [n, outcome, class, phenotype].

        keep_impossible=False gives the probability mass (zero for an outcome the model rules out);
        True gives a belief, which an outcome the model rules out leaves where it was.
        """
        rows = np.arange(len(w))
        items = self.item_of[tests]
        out = w[:, None] * np.moveaxis(self.lik[items], -1, 1)
        known = seen[rows, items]
        again = np.flatnonzero(known >= 0)
        out[again] = 0.0
        out[again, known[again]] = w[again]
        if keep_impossible:
            dead = out.sum((2, 3)) <= 0
            out = np.where(dead[:, :, None, None], w[:, None], out)
        return out

    def advance(self, w, seen, tests, outcomes, keep_impossible: bool) -> np.ndarray:
        return self.expand(w, seen, tests, keep_impossible)[np.arange(len(w)), outcomes]


class CountFilter:
    """W1 (tiicee.twin.world_models.CountWorldModel): tests independent given the class, no evidence identity."""

    def __init__(self, w1: twin_wm.CountWorldModel):
        absent = np.broadcast_to((1.0 - w1.availability)[:, None, None], w1.lik.shape[:2] + (1,))
        self.prior = w1.prior
        self.lik = np.concatenate([w1.availability[:, None, None] * w1.lik, absent], axis=2)    # [test, class, outcome]
        update = np.concatenate([w1.lik, np.ones(w1.lik.shape[:2] + (1,))], axis=2)              # UNAVAILABLE says nothing
        self.update = np.moveaxis(update, 2, 1)                                                  # [test, outcome, class]

    def start(self, n: int) -> np.ndarray:
        return np.broadcast_to(self.prior, (n, len(self.prior))).copy()

    def joint(self, b, seen, tests) -> np.ndarray:
        return b[:, :, None] * self.lik[tests]

    def joint_all(self, b, seen) -> np.ndarray:
        return b[:, None, :, None] * self.lik[None]

    def expand(self, b, seen, tests, keep_impossible: bool = True) -> np.ndarray:
        out = b[:, None, :] * self.update[tests]
        return out / out.sum(-1, keepdims=True)

    def advance(self, b, seen, tests, outcomes, keep_impossible: bool = True) -> np.ndarray:
        return self.expand(b, seen, tests)[np.arange(len(b)), outcomes]


class R4Reader:
    """tiicee.twin.readers.SimulatedReader."""

    def __init__(self, model: TwinModel, name: str, power: float = 1.0, drift=None, delta: float = 0.4, fixed_class: int = 1):
        self.name, self.power, self.drift, self.delta, self.fixed_class = name, power, drift, delta, fixed_class
        self.prior = np.array(model.prior)
        self.lik = np.moveaxis(model.class_lik(), 2, 1) ** power        # [test, outcome, class]

    def start(self, n: int) -> np.ndarray:
        return np.broadcast_to(self.prior, (n, len(self.prior))).copy()

    def belief(self, b) -> np.ndarray:
        return b

    def _react(self, b, lik) -> np.ndarray:
        """b [n, class] against lik [n, ..., class]: the reader's belief after each of the results lik describes."""
        shape = (len(b),) + (1,) * (lik.ndim - 2) + (b.shape[1],)
        prior = b.reshape(shape)
        q = prior * lik
        q = np.where(q.sum(-1, keepdims=True) > 0, q, prior)            # no class allows the result: no evidence
        if self.drift is not None:
            k = np.argmax(b, axis=1) if self.drift == "entrench" else np.full(len(b), self.fixed_class)
            boost = np.ones_like(b)
            boost[np.arange(len(b)), k] = np.exp(self.delta)
            q = q * boost.reshape(shape)
        return q / q.sum(-1, keepdims=True)

    def reactions(self, b, seen, tests) -> np.ndarray:
        """The belief after each outcome of each history's test: [n, outcome, class]."""
        return self._react(b, self.lik[tests])

    def reactions_all(self, b, seen) -> np.ndarray:
        """[n, test, outcome, class]."""
        return self._react(b, np.broadcast_to(self.lik, (len(b),) + self.lik.shape))

    def expand(self, b, seen, tests) -> np.ndarray:
        return self.reactions(b, seen, tests)

    def advance(self, b, seen, tests, outcomes) -> np.ndarray:
        return self.reactions(b, seen, tests)[np.arange(len(b)), outcomes]


class R0Reader:
    """tiicee.twin.readers.BayesReader: the posterior of the true model; an impossible result leaves it unchanged."""

    name = "r0"

    def __init__(self, model: TwinModel):
        self.filter = LatentFilter(model)

    def start(self, n: int) -> np.ndarray:
        return self.filter.start(n)

    def belief(self, w) -> np.ndarray:
        by_class = w.sum(2)
        return by_class / by_class.sum(1, keepdims=True)

    @staticmethod
    def _posterior(joint, belief) -> np.ndarray:
        """joint [..., class, outcome] -> the posterior after each outcome [..., outcome, class]."""
        mass = joint.sum(-2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(mass > 0, joint / mass, belief[..., None])
        return np.swapaxes(q, -1, -2)

    def reactions(self, w, seen, tests) -> np.ndarray:
        return self._posterior(self.filter.joint(w, seen, tests), self.belief(w))

    def reactions_all(self, w, seen) -> np.ndarray:
        return self._posterior(self.filter.joint_all(w, seen), self.belief(w)[:, None, :])

    def expand(self, w, seen, tests) -> np.ndarray:
        return self.filter.expand(w, seen, tests, keep_impossible=True)

    def advance(self, w, seen, tests, outcomes) -> np.ndarray:
        return self.filter.advance(w, seen, tests, outcomes, keep_impossible=True)


def grid_name(power: float, delta: float, drift: str) -> str:
    """The name of a reader of the harmful-region grid: a likelihood power with a drift of the given kind and size."""
    return f"grid:{power:g}:{delta:g}:{drift}"


def reader(model: TwinModel, name: str):
    """A reader by name: r0, a member of the card's R4 family, or a grid reader named by grid_name."""
    if name == "r0":
        return R0Reader(model)
    if name.startswith("grid:"):
        _, power, delta, drift = name.split(":")
        return R4Reader(model, name, power=float(power), drift=drift if float(delta) > 0 else None, delta=float(delta))
    options = {"independence": {}, "entrench": {"drift": "entrench"}, "fixed": {"drift": "fixed"},
               "over": {"power": 2.0}, "under": {"power": 0.5},
               "entrench+over": {"drift": "entrench", "power": 2.0}, "fixed+under": {"drift": "fixed", "power": 0.5}}
    return R4Reader(model, name, **options[name])


# ---------------------------------------------------------------------------------------------
# Controlled errors that reshape the true predictive (tiicee.twin.world_models._Reshaped)
# ---------------------------------------------------------------------------------------------
def _findings(p_star):
    """The findings part of predictives [m, outcome]: its mass, and which rows have at least two possible findings."""
    found = p_star[:, :-1]
    return found, found.sum(1), (found > 0).sum(1) >= 2


def temperature(p_star, temp: float) -> np.ndarray:
    """tiicee.twin.world_models.temperature on predictives [m, outcome]: findings reshaped, UNAVAILABLE untouched."""
    found, mass, movable = _findings(p_star)
    out = p_star.copy()
    powered = (found[movable] / mass[movable, None]) ** (1.0 / temp)
    out[movable, :-1] = mass[movable, None] * powered / powered.sum(1, keepdims=True)
    return out


def matched_pair(p_star, posteriors, kl: float):
    """tiicee.twin.world_models.matched_pair on predictives [m, outcome] with the reader's raw posteriors [m, outcome, class].

    Returns (ignored, responsive, slow). slow counts the points sent one at a time through the
    twin's own routine: those where some finding has probability zero.
    """
    found, mass, movable = _findings(p_star)
    ignored, responsive = p_star.copy(), p_star.copy()
    n_find = found.shape[1]
    regular = movable & (found > 0).all(1)
    rows = np.flatnonzero(regular)
    slow = np.flatnonzero(movable & ~regular)
    if rows.size:
        p = found[rows] / mass[rows, None]
        q = posteriors[rows, :n_find]
        root = np.sqrt(p)
        unit_root = root / np.linalg.norm(root, axis=1, keepdims=True)
        response = np.swapaxes((q - np.einsum("ms,msc->mc", p, q)[:, None, :]) * root[:, :, None], 1, 2)
        _, singular, vt = np.linalg.svd(response, full_matrices=True)
        scale = singular[:, 0]
        rank = np.where(scale > 1e-12, (singular > 1e-11 * scale[:, None]).sum(1), 0)
        w_ignored = np.empty_like(p)
        for r in np.unique(rank):                               # points grouped by the rank of the reader's response
            here = np.flatnonzero(rank == r)
            null = vt[here, r:, :]
            null = null - np.einsum("mks,ms->mk", null, unit_root[here])[:, :, None] * unit_root[here][:, None, :]
            _, kept, basis = np.linalg.svd(null, full_matrices=False)
            basis = basis * (kept > 1e-8)[:, :, None]           # the directions the reader ignores
            if not (kept > 1e-8).any(1).all():
                raise ValueError("the reader responds to every direction of this test's findings; "
                                 "an ignored direction needs more findings than classes")
            projector = np.einsum("mjs,mjk->msk", basis, basis)  # column k: the k-th coordinate direction projected onto them
            first = np.argmax(np.linalg.norm(projector, axis=1) > 1e-6, axis=1)     # the fixed choice within the subspace
            chosen = projector[np.arange(len(here)), :, first]
            w_ignored[here] = chosen / np.linalg.norm(chosen, axis=1, keepdims=True)
        top = vt[:, 0, :] - (vt[:, 0, :] * unit_root).sum(1, keepdims=True) * unit_root
        with np.errstate(divide="ignore", invalid="ignore"):
            top = top / np.linalg.norm(top, axis=1, keepdims=True)
        w_top = np.where((rank > 0)[:, None], top, w_ignored)   # a reader that does not respond ignores everything
        for target, w in ((ignored, w_ignored), (responsive, w_top)):
            direction = root * w
            lead = direction[np.arange(len(direction)), np.argmax(np.abs(direction), axis=1)]
            direction = direction * np.where(lead > 0, 1.0, -1.0)[:, None]
            target[rows, :-1] = mass[rows, None] * _at_divergence(p, direction, kl / mass[rows])
    for i in slow:                                              # a finding of probability zero: the twin's routine, one point at a time
        support = np.flatnonzero(found[i] > 0)
        p = found[i, support] / mass[i]
        axes = twin_wm._reader_axes(posteriors[i, support], p)
        for target, direction in zip((ignored, responsive), axes):
            target[i, support] = mass[i] * _at_divergence(p[None], direction[None], np.array([kl / mass[i]]))[0]
    return ignored, responsive, int(len(slow))


def _at_divergence(p, direction, target) -> np.ndarray:
    """p + step x direction with KL(p || .) equal to target, by the bisection of world_models._MatchedMember."""
    with np.errstate(divide="ignore", invalid="ignore"):
        limit = np.where(direction < 0, p / -direction, np.inf).min(1)
    lo, hi = np.zeros(len(p)), limit
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            q = p + mid[:, None] * direction
            divergence = np.where((q <= 0).any(1), np.inf, (p * np.log(p / q)).sum(1))
            below = divergence < target
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    q = p + lo[:, None] * direction
    return q / q.sum(1, keepdims=True)
