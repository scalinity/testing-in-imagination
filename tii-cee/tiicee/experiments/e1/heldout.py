"""B-held-out on the twin, estimated as it will be on charts (specification 5.11, card part D).

A reference reader, a regularized partial-evidence multinomial logistic classifier, is fitted on
prefixes of the simulated training patients' results in a random order per patient, cross-fitted by
patient. Its mean reduction in log loss when one more result is added, by test and by stratum
(results already read: 0, 1, 2 or more; reference reader's largest probability above 0.8 or not),
is the held-out information of that test there. The budget is the one-sided lower 90% bootstrap
bound of that mean over patients of the estimation folds, floored at zero. A cell is eligible for
CEE-Scale only if the bound does not exceed the mean reduction on the one fold kept apart.

Streams: patients from (seed, fitting, train); their random orders from (seed, behavior, train);
bootstrap draws from (seed, montecarlo, train); the stress factor from (seed, corruption, train).
"""
from __future__ import annotations

import numpy as np

from tiicee.cee import bootstrap_upper_bound, floor_norm
from tiicee.experiments.e1 import batch, filters
from tiicee.experiments.e1.context import Context
from tiicee.twin.env import stream

N_STRATA = 6
CONFIDENT = 0.8


class ReferenceReader:
    """Multinomial logistic regression on which results have been read, with an L2 penalty on the weights.

    Features: an intercept, then one indicator per (test, outcome) for each result in the history;
    the order of the results does not enter. Fitted by majorization with Boehning's bound on the
    Hessian, which needs one factorization and converges monotonically.
    """

    def __init__(self, n_tests: int, n_outcomes: int, n_classes: int, l2: float):
        self.n_tests, self.n_outcomes, self.n_classes, self.l2 = n_tests, n_outcomes, n_classes, float(l2)
        self.weights = np.zeros((1 + n_tests * n_outcomes, n_classes))
        self.iterations = 0

    def features(self, answers: np.ndarray) -> np.ndarray:
        """answers [n, test] holds each requested test's outcome index, -1 if not requested."""
        n = len(answers)
        x = np.zeros((n, 1 + self.n_tests * self.n_outcomes))
        x[:, 0] = 1.0
        rows, tests = np.nonzero(answers >= 0)
        x[rows, 1 + tests * self.n_outcomes + answers[rows, tests]] = 1.0
        return x

    def objective(self, x, y, weights=None) -> float:
        w = self.weights if weights is None else weights
        logits = x @ w
        top = logits.max(1, keepdims=True)
        log_prob = logits - top - np.log(np.exp(logits - top).sum(1, keepdims=True))
        return float(-log_prob[np.arange(len(y)), y].sum() + 0.5 * self.l2 * (w[1:] ** 2).sum())

    def gradient(self, x, y, weights=None) -> np.ndarray:
        w = self.weights if weights is None else weights
        residual = self._probabilities(x, w)
        residual[np.arange(len(y)), y] -= 1.0
        penalty = self.l2 * w
        penalty[0] = 0.0
        return x.T @ residual + penalty

    @staticmethod
    def _probabilities(x, w) -> np.ndarray:
        logits = x @ w
        p = np.exp(logits - logits.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def fit(self, x, y, tol: float = 1e-9, max_iter: int = 5000) -> "ReferenceReader":
        """Stops when the largest gradient entry is below tol per row. The intercepts are centred across classes."""
        ridge = np.full(x.shape[1], self.l2)
        ridge[0] = 0.0
        bound = np.linalg.inv(0.5 * (x.T @ x) + np.diag(ridge))      # for the directions that differ across classes
        w = np.zeros_like(self.weights)
        for self.iterations in range(1, max_iter + 1):
            g = self.gradient(x, y, w)
            if np.abs(g).max() <= tol * len(y):
                break
            common = g.mean(1, keepdims=True)                        # the same shift for every class changes nothing
            w = w - bound @ (g - common)
            w[1:] -= common[1:] / self.l2
        w[0] -= w[0].mean()
        self.weights = w
        return self

    def predict(self, x) -> np.ndarray:
        return self._probabilities(x, self.weights)


def stratum(n_results, largest) -> np.ndarray:
    """Specification 5.11: results already read (0, 1, 2 or more) crossed with the reference reader's confidence."""
    return np.minimum(n_results, 2) * 2 + (largest > CONFIDENT)


class HeldOut:
    """The held-out budget of one (seed, variant) on the reduced twin, and the record of how it was estimated."""

    def __init__(self, ctx: Context):
        cfg, twin = ctx.cfg, ctx.twin("reduced")
        m = twin.model
        n, n_tests, n_out, absent = cfg.n_train, len(m.tests), len(m.outcomes), len(m.outcomes) - 1
        patients = twin.sample_patients(n, ctx.seed, "train", purpose="fitting")
        y = np.array([m.classes.index(p.diagnosis) for p in patients])
        observed = np.array([[m.outcomes.index(twin.observe(p, t)) for t in m.tests] for p in patients])
        order = stream(ctx.seed, "behavior", "train", n_tests).permuted(np.tile(np.arange(n_tests), (n, 1)), axis=1)
        rows = np.arange(n)

        self.reader = ReferenceReader(n_tests, n_out, len(m.classes), cfg.heldout_l2)
        answers = np.full((n_tests + 1, n, n_tests), -1, dtype=np.int64)     # prefix k: the first k tests of each patient's order
        for k in range(n_tests):
            answers[k + 1] = answers[k]
            answers[k + 1, rows, order[:, k]] = observed[rows, order[:, k]]
        x = np.stack([self.reader.features(a) for a in answers])             # [prefix, patient, feature]

        fold = rows % cfg.heldout_folds
        probability = np.empty((n_tests + 1, n, len(m.classes)))
        self.fold_iterations = []
        for f in range(cfg.heldout_folds):                                   # cross-fitted by patient
            inside = fold == f
            fitted = ReferenceReader(n_tests, n_out, len(m.classes), cfg.heldout_l2).fit(
                x[:, ~inside].reshape(-1, x.shape[2]), np.tile(y[~inside], n_tests + 1))
            self.fold_iterations.append(fitted.iterations)
            probability[:, inside] = fitted.predict(x[:, inside].reshape(-1, x.shape[2])).reshape(n_tests + 1, -1, len(m.classes))
        self.reader.fit(x.reshape(-1, x.shape[2]), np.tile(y, n_tests + 1))  # all training patients: strata of new histories

        loss = -np.log(floor_norm(probability)[:, rows, y])                  # [prefix, patient]
        reduction = (loss[:-1] - loss[1:]).T                                 # [patient, step]: adding the step's test
        largest = probability.max(2)[:-1].T
        n_results = np.stack([(answers[k] >= 0) & (answers[k] != absent) for k in range(n_tests)]).sum(2).T
        cell = stratum(n_results, largest)

        truth = filters.LatentFilter(m)                                      # exact information of the same additions
        w, seen, exact = truth.start(n), filters.new_seen(n, m), np.empty((n, n_tests))
        for k in range(n_tests):
            tests, outcomes = order[:, k], observed[rows, order[:, k]]
            exact[:, k] = batch.information(truth.joint(w, seen, tests))
            w = truth.advance(w, seen, tests, outcomes, keep_impossible=False)
            seen = filters.record(seen, truth.item_of, tests, outcomes)

        estimation = fold[:, None] < cfg.heldout_folds - 1                   # the last fold is kept apart for eligibility
        self.budget = np.zeros((n_tests, N_STRATA))
        self.eligible = np.zeros((n_tests, N_STRATA), dtype=bool)
        self.records = []
        factors = np.exp(cfg.noise_sigma * stream(ctx.seed, "corruption", "train").normal(size=(n_tests, N_STRATA)))
        patient = np.broadcast_to(rows[:, None], order.shape)
        for t in range(n_tests):
            for s in range(N_STRATA):
                here = (order == t) & (cell == s)
                est, check = here & estimation, here & ~estimation
                record = {"test": m.tests[t], "stratum": s, "n_estimation": int(est.sum()), "n_check": int(check.sum()),
                          "mean_estimation": None, "bound": None, "mean_check": None, "exact_mean": None,
                          "budget": 0.0, "eligible": False}
                if est.any() and check.any():
                    rng = stream(ctx.seed, "montecarlo", "train", t, s)
                    bound = -bootstrap_upper_bound(-reduction[est], patient[est], rng, level=cfg.heldout_level, n_boot=cfg.n_boot)
                    record.update(mean_estimation=float(reduction[est].mean()), bound=float(bound),
                                  mean_check=float(reduction[check].mean()), exact_mean=float(exact[est].mean()),
                                  budget=float(max(bound, 0.0)), eligible=bool(bound <= reduction[check].mean()))
                    self.budget[t, s], self.eligible[t, s] = record["budget"], record["eligible"]
                    for name, factor in (("x1", 1.0), ("x0.5", 0.5), ("x2", 2.0), ("noise", float(factors[t, s]))):
                        stressed = factor * record["budget"]
                        record[f"stress_{name}"] = stressed
                        record[f"overestimated_{name}"] = bool(stressed > record["exact_mean"])
                        record[f"admitted_{name}"] = bool(stressed <= record["mean_check"])
                self.records.append(record)
        self._absent = absent

    def strata(self, answers: np.ndarray) -> np.ndarray:
        """The stratum of each history, given as answers [n, test]."""
        largest = self.reader.predict(self.reader.features(answers)).max(1)
        return stratum(((answers >= 0) & (answers != self._absent)).sum(1), largest)

    def at(self, level):
        """(budget [n], eligible [n]) for the test each node of a tree level requests."""
        cell = self.strata(level.answers)
        return self.budget[level.tests, cell], self.eligible[level.tests, cell]

    def lookup(self, answers: np.ndarray, tests: np.ndarray):
        """(budget, eligible) for arbitrary (history, test) pairs."""
        cell = self.strata(answers)
        return self.budget[tests, cell], self.eligible[tests, cell]
