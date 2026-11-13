"""T5, its paired contrast, the sign-flip randomization test, and that test's sensitivity.

T5 is the mean of the per-class recalls, in points. With K classes, w_i = 100 / (K n_class(i)) and
d_i = correct_a(i) - correct_b(i), Delta T5 = sum_i w_i d_i. These are the conventions of
cdm-extension/daniel-escalante/tim/analysis.py (SPEC section 6 there), for any number of classes.
"""
from __future__ import annotations

import functools

import numpy as np

from tiicee.stats import _resample

ALPHA = 0.05
TOLERANCE = 1e-9  # a sign assignment that ties the observed |S| counts as extreme
TEST = "two-sided paired sign-flip randomization test of S = sum_i w_i d_i, exact null distribution"
NULL = ("within each unit the two arms' outcomes are exchangeable, so each nonzero paired difference "
        "is equally likely to carry either sign; the test is exact for this null and not for the "
        "weaker null Delta T5 = 0 when classes move in opposite directions")


def _classes(truth, classes):
    classes = list(classes)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("classes must list at least two distinct labels")
    lookup = {label: k for k, label in enumerate(classes)}
    truth = np.asarray(truth, dtype=object)
    if truth.ndim != 1 or truth.size == 0:
        raise ValueError("truth must be a nonempty one-dimensional sequence")
    try:
        index = np.array([lookup[label] for label in truth])
    except (KeyError, TypeError):
        raise ValueError("truth holds a label outside classes") from None
    counts = np.bincount(index, minlength=len(classes))
    if (counts == 0).any():
        raise ValueError("every class needs at least one unit: a class without support has no recall, "
                         "and T5 is never computed with an implied zero")
    return lookup, index, counts


def _correct(pred, index, lookup):
    """Correctness of predictions shaped (units,) or (policies, units); None is a nonanswer."""
    pred = np.asarray(pred, dtype=object)
    if pred.ndim not in (1, 2) or pred.shape[-1] != len(index):
        raise ValueError("predictions must be shaped (units,) or (policies, units), aligned with truth")
    try:
        codes = np.array([-1 if p is None else lookup[p] for p in pred.ravel()])
    except (KeyError, TypeError):
        raise ValueError("a prediction must be one of the classes, or None for no answer") from None
    return codes.reshape(pred.shape) == index


def t5(pred, truth, classes) -> float:
    """The mean of the per-class recalls, times 100. A nonanswer (None) counts as wrong."""
    lookup, index, counts = _classes(truth, classes)
    correct = _correct(pred, index, lookup)
    if correct.ndim != 1:
        raise ValueError("t5 scores one set of predictions")
    recalls = np.bincount(index, weights=correct, minlength=len(counts)) / counts
    return float(np.mean(100 * recalls))


# ---- the exact sign-flip null ---------------------------------------------------------------

@functools.lru_cache(maxsize=4096)
def _binomial_half(k: int) -> np.ndarray:
    """P(B = b) for b = 0..k and B ~ Binomial(k, 1/2): exact integers, each quotient correctly rounded."""
    pmf, ways = np.empty(k + 1), 1
    for b in range(k + 1):
        pmf[b] = ways / (1 << k)
        ways = ways * (k - b) // (b + 1)
    pmf.flags.writeable = False
    return pmf


def _lattice(magnitudes):
    """Distribution of sum_i s_i a_i over independent fair signs s_i, for positive integers a_i.

    Units that share a magnitude a contribute a (2B - k) with B binomial, so the sum is a
    convolution of scaled binomials on the integers. Returns (support, probabilities).
    """
    dense, reach = np.ones(1), 0
    for a, k in zip(*np.unique(np.asarray(magnitudes, dtype=np.int64), return_counts=True)):
        a, k = int(a), int(k)
        step = np.zeros(2 * a * k + 1)
        step[:: 2 * a] = _binomial_half(k)
        dense, reach = np.convolve(dense, step), reach + a * k
    support = np.flatnonzero(dense)
    return support - reach, dense[support]


def _half(parts):
    """Distribution of sum_c scale_c X_c over the given classes: sorted values, their
    probabilities, and the cumulative probability before each value."""
    values, probs = np.zeros(1), np.ones(1)
    for scale, support, p in parts:
        values = (values[:, None] + scale * support[None, :]).ravel()
        probs = (probs[:, None] * p[None, :]).ravel()
    order = np.argsort(values)
    return values[order], probs[order], np.concatenate([[0.0], np.cumsum(probs[order])])


@functools.lru_cache(maxsize=1024)
def _binary_half(scales: tuple, discordant: tuple):
    """_half for classes whose discordant units all weigh the same: one scaled binomial each."""
    return _half([(scale, 2 * np.arange(k + 1) - k, _binomial_half(k)) for scale, k in zip(scales, discordant)])


def _two_sided(a, b, threshold) -> float:
    """P(|A + B| >= threshold) for independent halves A and B, by a sorted search over B."""
    a_values, a_probs, _ = a
    b_values, _, cumulative = b
    upper = cumulative[-1] - cumulative[np.searchsorted(b_values, threshold - a_values, side="left")]
    lower = cumulative[np.searchsorted(b_values, -threshold - a_values, side="right")]
    return min(1.0, float(np.sum(a_probs * (upper + lower))))


def sign_flip_p(groups, statistic) -> float:
    """Exact two-sided sign-flip p-value of S = sum_c scale_c sum_i s_i a_ci.

    groups holds one (scale, magnitudes) pair per class: the class's common weight and the
    positive integer magnitudes a_ci of its discordant units (all 1 for one policy). Under the
    null every sign s_i is +1 or -1 with probability 1/2, independently. The classes' sums are
    split in two halves that are combined by a sorted search, so the value equals enumeration of
    all 2^k sign assignments without its cost. Ties with |statistic| count as extreme.
    """
    threshold = abs(statistic) - TOLERANCE
    if threshold <= 0:
        return 1.0
    parts = [(float(scale), *_lattice(magnitudes)) for scale, magnitudes in groups]
    half = len(parts) // 2
    return _two_sided(_half(parts[:half]), _half(parts[half:]), threshold)


# ---- the contrast ---------------------------------------------------------------------------

def t5_contrast(pred_a, pred_b, truth, classes, n_boot=10_000, *, rng, within=None) -> dict:
    """Delta T5 (a minus b) paired by unit, its 95% interval and its sign-flip p-value.

    pred_a and pred_b are aligned with truth, one prediction per unit, or shaped (policies, units)
    when the contrast is averaged over several policies that each scored every unit. `within`
    gives one stratum per unit (cohort X: the fold) when units must be resampled inside strata.

    Interval: class-stratified paired percentile bootstrap. Each resample draws units with
    replacement inside every class (inside every class-by-stratum cell with `within`), so class
    sizes never change and pairs stay together. With several policies it also draws as many
    policies as there are, so units and policies are resampled together; the between-policy
    spread is reported under "policies" and takes no part in the interval.

    Test: two-sided paired sign-flip randomization test of S = Delta T5, paired by unit on the
    unit's mean difference over policies. The p-value is exact: in a class every discordant unit
    carries the same weight, so the null distribution of S is a sum of scaled binomials, one per
    class and magnitude. "null" names the hypothesis tested.

    "changes" counts, per class, the units whose difference is positive, negative or zero; a
    class's discordance for `mde` is (improved + regressed) / its size.
    """
    _resample.check(rng, n_boot)
    classes = list(classes)
    lookup, index, counts = _classes(truth, classes)
    correct_a, correct_b = _correct(pred_a, index, lookup), _correct(pred_b, index, lookup)
    if correct_a.shape != correct_b.shape:
        raise ValueError("pred_a and pred_b must have the same shape")
    d = np.atleast_2d(correct_a.astype(np.int8) - correct_b.astype(np.int8))
    n_policies, n_units = d.shape
    class_weight = 100 / (len(classes) * counts)
    contributions = d * class_weight[index]
    per_policy = contributions.sum(axis=1)
    estimate = float(per_policy.mean())

    total = d.sum(axis=0, dtype=np.int64)  # the unit's mean difference, times the number of policies
    groups = [(class_weight[k] / n_policies, np.abs(total[(index == k) & (total != 0)]))
              for k in range(len(classes))]

    stratum = index
    if within is not None:
        _, within_index = _resample.index(within, "within", n_units)
        stratum = index * (within_index.max() + 1) + within_index
    cells = [np.flatnonzero(stratum == s) for s in np.unique(stratum)]
    samples = _resample.unit_sum_samples(contributions, cells, n_boot, rng)

    out = {
        "delta_t5": estimate, "ci": _resample.percentile(samples, 0.95), "level": 0.95,
        "p_value": sign_flip_p(groups, estimate), "method": "exact", "test": TEST, "null": NULL,
        "discordant": int(np.count_nonzero(total)),
        "changes": {label: {"improved": int(np.sum(total[index == k] > 0)),
                            "regressed": int(np.sum(total[index == k] < 0)),
                            "unchanged": int(np.sum(total[index == k] == 0))}
                    for k, label in enumerate(classes)},
        "n_units": n_units, "n_boot": n_boot,
        "interval": "class-stratified paired percentile bootstrap over units"
                    + (" within strata" if within is not None else "")
                    + (", units and policies resampled together" if n_policies > 1 else ""),
    }
    if n_policies > 1:
        out["policies"] = {"n": n_policies, "delta_t5": [float(v) for v in per_policy],
                           "sd": float(np.std(per_policy, ddof=1))}
    return out


# ---- sensitivity ----------------------------------------------------------------------------

def t5_power(probabilities, sizes, classes, n_sims=10_000, *, rng, alpha=ALPHA) -> float:
    """Share of simulated sign-flip tests with p < alpha.

    probabilities[c] = (improve, regress): the chance that a unit of class c is correct under a
    only, and under b only. Each simulated trial draws every class's (improved, regressed,
    unchanged) counts from a multinomial at sizes[c] and is tested exactly. This is the
    simulation of tim.analysis.power: one policy, one binary outcome per unit and arm.
    """
    _resample.check(rng, n_sims)
    classes = list(classes)
    draws = []
    for c in classes:
        p, q = probabilities[c]
        draws.append(rng.multinomial(sizes[c], [p, q, max(0.0, 1 - p - q)], size=n_sims))
    scales = tuple(100 / (len(classes) * sizes[c]) for c in classes)
    half = len(classes) // 2
    discordant = np.stack([d[:, 0] + d[:, 1] for d in draws], axis=1).tolist()
    net = np.stack([d[:, 0] - d[:, 1] for d in draws], axis=1).tolist()
    rejected = 0
    for k, signed in zip(discordant, net):
        threshold = abs(sum(scale * n for scale, n in zip(scales, signed))) - TOLERANCE
        p_value = 1.0 if threshold <= 0 else _two_sided(
            _binary_half(scales[:half], tuple(k[:half])), _binary_half(scales[half:], tuple(k[half:])), threshold)
        rejected += p_value < alpha
    return rejected / n_sims


def mde(discordance, sizes, classes, n_sims=10_000, *, rng, target=0.8, alpha=ALPHA):
    """Minimum detectable effect of the sign-flip test on Delta T5, in T5 points, or None.

    discordance[c] is the share of class c's units on which the two arms disagree and sizes[c]
    the class size the test will run at. Each class's discordance is held fixed and the balance
    theta between improvement and regression is bisected: a class improves with probability
    discordance (1 + theta) / 2 and regresses with discordance (1 - theta) / 2. The result is the
    expected Delta T5, (100 / K) theta sum_c discordance[c], at the smallest theta found whose
    simulated power reaches `target` at two-sided `alpha`; None when even theta = 1 falls short.
    Every bisection step replays the same simulated trials: the generator is rewound to its
    state on entry, so mde(..., rng=np.random.default_rng(seed)) reproduces
    tim.analysis.minimum_detectable_effect(..., seed). On return the generator has advanced by
    one step's draws.

    It is a statement about the test's sensitivity, recomputed for each clinical contrast from
    that contrast's own discordance. It is never an equivalence margin: a nonsignificant result
    is reported as no detectable effect at this sensitivity, not as equivalence.
    """
    _resample.check(rng, n_sims)
    classes = list(classes)
    state = rng.bit_generator.state

    def power_at(theta):
        rng.bit_generator.state = state
        balance = {c: (discordance[c] * (1 + theta) / 2, discordance[c] * (1 - theta) / 2) for c in classes}
        return t5_power(balance, sizes, classes, n_sims, rng=rng, alpha=alpha)

    if power_at(1.0) < target:
        return None
    low, high = 0.0, 1.0
    for _ in range(14):
        middle = (low + high) / 2
        low, high = (low, middle) if power_at(middle) >= target else (middle, high)
    return (100 / len(classes)) * high * sum(discordance[c] for c in classes)
