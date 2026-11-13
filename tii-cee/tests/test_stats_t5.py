"""T5, its paired contrast, the exact sign-flip test, the class-stratified bootstrap, power and MDE.

References here are independent of the module: hand calculations, brute-force enumeration of
every sign assignment, and exact binomial tails in integer arithmetic.
"""
import itertools
import math

import numpy as np
import pytest

from tiicee import stats
from tiicee.stats import _resample

CLASSES = ("a", "b", "c", "d")


def random_pair(seed, sizes, flips, right=0.6, classes=CLASSES):
    """Paired predictions: b is right with probability `right`; a differs from b in correctness
    on exactly `flips` units."""
    rng = np.random.default_rng(seed)
    truth = [label for label, n in zip(classes, sizes) for _ in range(n)]
    b = [label if rng.random() < right else None for label in truth]
    a = list(b)
    for i in rng.choice(len(truth), size=flips, replace=False):
        a[i] = truth[i] if b[i] != truth[i] else None
    order = rng.permutation(len(truth))
    return [a[i] for i in order], [b[i] for i in order], [truth[i] for i in order]


def brute_force_p(values):
    """Share of the 2^k sign assignments of `values` at least as extreme as the observed sum."""
    values = np.asarray(values, dtype=float)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    return float(np.mean(np.abs(signs @ values) >= abs(values.sum()) - 1e-9))


def weighted_differences(a, b, truth, classes=CLASSES):
    """w_i d_i of the discordant units, for one policy (rows) or the mean over several."""
    a, b = np.atleast_2d(np.array(a, dtype=object)), np.atleast_2d(np.array(b, dtype=object))
    truth = np.array(truth, dtype=object)
    d = ((a == truth).astype(float) - (b == truth).astype(float)).mean(axis=0)
    w = np.array([100 / (len(classes) * np.sum(truth == t)) for t in truth])
    return (w * d)[d != 0]


# ---- T5 -------------------------------------------------------------------------------------

def test_t5_by_hand():
    truth = ["a"] * 4 + ["b"] * 2 + ["c"] * 5 + ["d"]
    pred = ["a", "a", "a", "b"] + ["b", None] + ["c", "c", "d", None, "a"] + ["d"]
    # recalls 3/4, 1/2, 2/5, 1/1; their mean is 2.65 / 4
    assert stats.t5(pred, truth, CLASSES) == pytest.approx(66.25, abs=1e-12)


def test_t5_is_the_mean_recall_not_the_accuracy():
    truth = ["a"] * 97 + ["b", "c", "d"]
    assert stats.t5(["a"] * 100, truth, CLASSES) == pytest.approx(25.0, abs=1e-12)  # accuracy would be 97


def test_t5_accepts_integer_labels_and_numpy_arrays():
    truth = np.array([0, 0, 1, 1, 2, 3])
    assert stats.t5(np.array([0, 1, 1, 1, 2, 0]), truth, range(4)) == pytest.approx(100 * (0.5 + 1 + 1 + 0) / 4, abs=1e-12)


def test_t5_refuses_what_it_cannot_score():
    with pytest.raises(ValueError, match="implied zero"):
        stats.t5(["a", "b", "c"], ["a", "b", "c"], CLASSES)  # class d has no support
    with pytest.raises(ValueError, match="outside classes"):
        stats.t5(["a", "b", "c", "d", "a"], ["a", "b", "c", "d", "e"], CLASSES)
    with pytest.raises(ValueError, match="one of the classes"):
        stats.t5(["a", "b", "c", "D"], ["a", "b", "c", "d"], CLASSES)  # a mislabelled prediction is not a wrong answer
    with pytest.raises(ValueError, match="aligned"):
        stats.t5(["a", "b", "c"], ["a", "b", "c", "d"], CLASSES)


# ---- the contrast's estimate and p-value ------------------------------------------------------

def test_contrast_by_hand():
    classes = ("x", "y")
    truth = ["x", "x", "y", "y", "y", "y"]
    b = [None, "x", "y", None, "y", "y"]
    a = ["x", "x", None, "y", "y", "y"]
    # Weights 100 / (2 * 2) = 25 for x and 100 / (2 * 4) = 12.5 for y; d = +1 (x), -1 and +1 (y).
    # S = 25. The eight sign assignments give 50, 25, 25, 0, 0, -25, -25, -50: six reach |25|.
    result = stats.t5_contrast(a, b, truth, classes, n_boot=50, rng=np.random.default_rng(0))
    assert result["delta_t5"] == pytest.approx(25.0, abs=1e-12)
    assert result["delta_t5"] == pytest.approx(stats.t5(a, truth, classes) - stats.t5(b, truth, classes), abs=1e-12)
    assert result["p_value"] == pytest.approx(0.75, abs=1e-12)
    assert result["discordant"] == 3 and result["n_units"] == 6 and result["method"] == "exact"
    assert result["changes"] == {"x": {"improved": 1, "regressed": 0, "unchanged": 1},
                                 "y": {"improved": 1, "regressed": 1, "unchanged": 2}}


def test_contrast_names_the_null_it_tests():
    a, b, truth = random_pair(1, (5, 4, 3, 6), 4)
    result = stats.t5_contrast(a, b, truth, CLASSES, n_boot=10, rng=np.random.default_rng(0))
    assert "exchangeable" in result["null"] and "equally likely" in result["null"]
    assert "sign-flip" in result["test"] and "two-sided" in result["test"]


@pytest.mark.parametrize("seed, sizes, flips", [
    (1, (5, 4, 3, 6), 1), (2, (5, 4, 3, 6), 2), (3, (5, 4, 3, 6), 6), (4, (9, 7, 4, 8), 9), (5, (9, 7, 4, 8), 12),
    (6, (6, 6, 6, 6), 7), (7, (6, 6, 6, 6), 12),  # equal class sizes: equal weights, so many exact ties
    (8, (30, 2, 11, 5), 10), (9, (3, 3, 2, 2), 10),
])
def test_p_value_equals_enumeration_of_every_sign_assignment(seed, sizes, flips):
    a, b, truth = random_pair(seed, sizes, flips)
    result = stats.t5_contrast(a, b, truth, CLASSES, n_boot=2, rng=np.random.default_rng(0))
    values = weighted_differences(a, b, truth)
    assert result["discordant"] == len(values) == flips
    assert result["delta_t5"] == pytest.approx(values.sum(), abs=1e-12)
    assert result["delta_t5"] == pytest.approx(stats.t5(a, truth, CLASSES) - stats.t5(b, truth, CLASSES), abs=1e-9)
    assert result["p_value"] == pytest.approx(brute_force_p(values), abs=1e-12)


def test_p_value_is_one_without_evidence():
    truth = ["a", "a", "b", "b", "c", "c", "d", "d"]
    same = stats.t5_contrast(truth, truth, truth, CLASSES, n_boot=20, rng=np.random.default_rng(0))
    assert same["p_value"] == 1.0 and same["delta_t5"] == 0.0 and same["discordant"] == 0 and same["ci"] == [0.0, 0.0]
    a = ["a", "a", None, "b", "c", "c", "d", "d"]  # one class b unit regresses ...
    b = [None, "a", "b", "b", "c", "c", "d", "d"]  # ... and one class a unit improves: Delta T5 = 0
    cancelled = stats.t5_contrast(a, b, truth, CLASSES, n_boot=20, rng=np.random.default_rng(0))
    assert cancelled["delta_t5"] == pytest.approx(0.0, abs=1e-12) and cancelled["p_value"] == 1.0
    assert cancelled["discordant"] == 2


@pytest.mark.parametrize("seed, policies", [(1, 2), (2, 3), (3, 5)])
def test_several_policies_p_value_equals_enumeration_on_the_unit_mean_difference(seed, policies):
    rng = np.random.default_rng(seed)
    truth = [label for label, n in zip(CLASSES, (4, 3, 2, 3)) for _ in range(n)]
    draw = lambda: [[label if rng.random() < 0.6 else None for label in truth] for _ in range(policies)]
    a, b = draw(), draw()
    result = stats.t5_contrast(a, b, truth, CLASSES, n_boot=2, rng=np.random.default_rng(0))
    values = weighted_differences(a, b, truth)
    assert 2 <= len(values) <= 12 and result["discordant"] == len(values)
    assert result["delta_t5"] == pytest.approx(values.sum(), abs=1e-12)
    assert result["p_value"] == pytest.approx(brute_force_p(values), abs=1e-12)
    per_policy = [stats.t5(x, truth, CLASSES) - stats.t5(y, truth, CLASSES) for x, y in zip(a, b)]
    assert result["policies"]["delta_t5"] == pytest.approx(per_policy, abs=1e-9)
    assert result["delta_t5"] == pytest.approx(np.mean(per_policy), abs=1e-9)
    assert result["policies"]["sd"] == pytest.approx(np.std(per_policy, ddof=1), abs=1e-9)


# ---- sign_flip_p against exact references beyond enumeration ----------------------------------

def binomial_two_sided(k, t):
    """P(|2B - k| >= t) for B ~ Binomial(k, 1/2), in integer arithmetic."""
    return sum(math.comb(k, b) for b in range(k + 1) if abs(2 * b - k) >= t) / 2 ** k


@pytest.mark.parametrize("k, t", [(1, 1), (10, 4), (51, 9), (400, 38), (1500, 80)])
def test_one_class_is_a_binomial_tail(k, t):
    weight = 100 / (4 * 1600)
    p = stats.sign_flip_p([(weight, np.ones(k, dtype=int))], weight * t)
    assert p == pytest.approx(binomial_two_sided(k, t), rel=1e-12)


def test_classes_of_equal_weight_pool_into_one_binomial():
    weight = 0.37
    groups = [(weight, np.ones(k, dtype=int)) for k in (40, 0, 75, 13)]
    assert stats.sign_flip_p(groups, weight * 24) == pytest.approx(binomial_two_sided(128, 24), rel=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_integer_magnitudes_and_unequal_weights_equal_enumeration(seed):
    rng = np.random.default_rng(seed)
    groups = [(float(rng.uniform(0.1, 2.0)), rng.integers(1, 6, size=rng.integers(0, 4))) for _ in range(4)]
    values = np.concatenate([scale * magnitudes for scale, magnitudes in groups])
    signs = rng.choice([-1.0, 1.0], size=len(values))
    assert stats.sign_flip_p(groups, float(signs @ values)) == pytest.approx(brute_force_p(signs * values), abs=1e-12)


def dense_lattice_p(steps, observed):
    """Exact two-sided p-value by a unit-by-unit recursion on one integer lattice: no grouping
    by class or magnitude and no split in halves. steps are positive integers, one per unit."""
    reach = sum(steps)
    mass = np.zeros(2 * reach + 1)
    mass[reach] = 1.0
    for a in steps:
        moved = np.zeros_like(mass)
        moved[a:] += 0.5 * mass[:-a]
        moved[:-a] += 0.5 * mass[a:]
        mass = moved
    return float(mass[np.abs(np.arange(-reach, reach + 1)) >= abs(observed)].sum())


@pytest.mark.parametrize("seed, sizes, policies", [(1, (96, 65, 25, 54), 1), (2, (40, 33, 21, 28), 3), (3, (96, 65, 25, 54), 5)])
def test_p_value_beyond_enumeration_equals_a_unit_by_unit_recursion(seed, sizes, policies):
    """With w_i = 100 / (4 n_class) and P policies, every w_i d_i is a whole multiple of
    100 / (4 P lcm(sizes)), so the null distribution lives on an integer lattice."""
    rng = np.random.default_rng(seed)
    truth = np.array([label for label, n in zip(CLASSES, sizes) for _ in range(n)], dtype=object)
    a, b = (np.where(rng.random((policies, len(truth))) < right, truth, None) for right in (0.76, 0.72))
    result = stats.t5_contrast(a, b, truth, CLASSES, n_boot=2, rng=np.random.default_rng(0))
    total = ((a == truth).astype(int) - (b == truth).astype(int)).sum(axis=0)
    per_unit = np.array([math.lcm(*sizes) // sizes[CLASSES.index(label)] for label in truth])
    steps = (np.abs(total) * per_unit)[total != 0].tolist()
    assert result["discordant"] == len(steps) > 60
    assert result["p_value"] == pytest.approx(dense_lattice_p(steps, int((total * per_unit).sum())), abs=1e-12)


# ---- the class-stratified paired bootstrap -----------------------------------------------------

def test_interval_is_reproducible_and_depends_on_the_generator():
    a, b, truth = random_pair(11, (20, 14, 9, 12), 16)
    first = stats.t5_contrast(a, b, truth, CLASSES, n_boot=400, rng=np.random.default_rng(5))
    assert first == stats.t5_contrast(a, b, truth, CLASSES, n_boot=400, rng=np.random.default_rng(5))
    other = stats.t5_contrast(a, b, truth, CLASSES, n_boot=400, rng=np.random.default_rng(6))
    assert other["ci"] != first["ci"] and other["p_value"] == first["p_value"] and other["delta_t5"] == first["delta_t5"]
    assert first["ci"][0] < first["delta_t5"] < first["ci"][1] and first["level"] == 0.95 and first["n_boot"] == 400


def test_interval_restated_independently():
    """The documented procedure, written out: per resample, class by class in class order, draw
    the class's size with replacement; the statistic is the T5 difference of the resampled rows."""
    a, b, truth = random_pair(12, (11, 8, 5, 9), 13)
    rng, samples = np.random.default_rng(77), []
    members = [[i for i, t in enumerate(truth) if t == label] for label in CLASSES]
    for _ in range(300):
        rows = [group[j] for group in members for j in rng.integers(0, len(group), size=len(group))]
        pick = lambda values: [values[i] for i in rows]
        samples.append(stats.t5(pick(a), pick(truth), CLASSES) - stats.t5(pick(b), pick(truth), CLASSES))
    result = stats.t5_contrast(a, b, truth, CLASSES, n_boot=300, rng=np.random.default_rng(77))
    assert result["ci"] == pytest.approx(list(np.quantile(samples, [0.025, 0.975])), abs=1e-9)


def test_pairs_are_resampled_together():
    """Two arms with different predictions but identical correctness differ by nothing on every resample."""
    _, b, truth = random_pair(13, (20, 14, 9, 12), 0)
    a = [x if x is not None else next(c for c in CLASSES if c != t) for x, t in zip(b, truth)]  # wrong answers for nonanswers
    result = stats.t5_contrast(a, b, truth, CLASSES, n_boot=200, rng=np.random.default_rng(1))
    assert a != b and result["ci"] == [0.0, 0.0]


def spy_on_strata(monkeypatch):
    seen = {}
    original = _resample.unit_sum_samples

    def spy(x, groups, n_boot, rng):
        seen["groups"] = groups
        return original(x, groups, n_boot, rng)

    monkeypatch.setattr(_resample, "unit_sum_samples", spy)
    return seen


def test_resampling_is_stratified_by_class(monkeypatch):
    seen = spy_on_strata(monkeypatch)
    a, b, truth = random_pair(14, (20, 14, 9, 12), 16)
    stats.t5_contrast(a, b, truth, CLASSES, n_boot=5, rng=np.random.default_rng(0))
    assert [sorted(group.tolist()) for group in seen["groups"]] == [
        [i for i, t in enumerate(truth) if t == label] for label in CLASSES]
    # ... and a resample keeps every stratum's size.
    rng, labels = np.random.default_rng(3), np.array(truth)
    for _ in range(200):
        drawn = labels[_resample.draw_units(seen["groups"], rng)]
        assert [int(np.sum(drawn == label)) for label in CLASSES] == [20, 14, 9, 12]


def test_class_sizes_never_vary_across_resamples():
    """Every class a unit improves and nothing else changes: with class a's size fixed, each
    resample's Delta T5 is exactly 25."""
    truth = ["a"] * 7 + ["b"] * 5 + ["c"] * 4 + ["d"] * 6
    b = [None] * 7 + truth[7:]
    result = stats.t5_contrast(truth, b, truth, CLASSES, n_boot=300, rng=np.random.default_rng(2))
    assert result["delta_t5"] == pytest.approx(25.0, abs=1e-12)
    assert result["ci"] == pytest.approx([25.0, 25.0], abs=1e-12)
    assert result["p_value"] == pytest.approx(2 / 2 ** 7, abs=1e-15)


def test_within_resamples_inside_class_by_fold_cells(monkeypatch):
    seen = spy_on_strata(monkeypatch)
    truth = ["a"] * 8 + ["b"] * 6 + ["c"] * 4 + ["d"] * 6
    fold = [0, 0, 0, 1, 1, 1, 1, 1] + [0, 1] * 3 + [0, 0, 1, 1] + [1, 0] * 3
    b = [None, None, None] + truth[3:]  # the three class a units of fold 0 improve; nothing else changes
    folded = stats.t5_contrast(truth, b, truth, CLASSES, n_boot=300, rng=np.random.default_rng(4), within=fold)
    cells = {(truth[group[0]], fold[group[0]]): len(group) for group in seen["groups"]}
    assert cells == {("a", 0): 3, ("a", 1): 5, ("b", 0): 3, ("b", 1): 3, ("c", 0): 2, ("c", 1): 2, ("d", 0): 3, ("d", 1): 3}
    assert all(len({(truth[i], fold[i]) for i in group}) == 1 for group in seen["groups"])
    # Inside folds the three improved units are always drawn three times in all: no spread.
    assert folded["ci"] == pytest.approx([folded["delta_t5"]] * 2, abs=1e-12)
    pooled = stats.t5_contrast(truth, b, truth, CLASSES, n_boot=300, rng=np.random.default_rng(4))
    assert pooled["ci"][1] - pooled["ci"][0] > 5 and pooled["p_value"] == folded["p_value"]
    assert "within strata" in folded["interval"] and "within strata" not in pooled["interval"]


def test_units_and_policies_are_resampled_together():
    """Policy 0 improves every class a unit, policy 1 changes nothing. Units add no spread, so a
    resample's value is the mean of two policy draws from (25, 0): 0, 12.5 or 25 with chances
    1/4, 1/2, 1/4, and the 95% percentile interval is [0, 25]."""
    truth = ["a"] * 4 + ["b"] * 3 + ["c"] * 3 + ["d"] * 2
    b = [[None] * 4 + truth[4:]] * 2
    a = [truth, b[1]]
    result = stats.t5_contrast(a, b, truth, CLASSES, n_boot=2000, rng=np.random.default_rng(8))
    assert result["delta_t5"] == pytest.approx(12.5, abs=1e-12)
    assert result["ci"] == pytest.approx([0.0, 25.0], abs=1e-12)
    assert result["policies"] == {"n": 2, "delta_t5": pytest.approx([25.0, 0.0], abs=1e-12),
                                  "sd": pytest.approx(25 / math.sqrt(2), abs=1e-12)}
    # Each class a unit's mean difference is +1/2: only the two one-signed assignments are as extreme.
    assert result["p_value"] == pytest.approx(2 / 2 ** 4, abs=1e-15)
    assert "policies resampled together" in result["interval"]


def test_identical_policies_change_neither_estimate_nor_test():
    a, b, truth = random_pair(15, (20, 14, 9, 12), 16)
    one = stats.t5_contrast(a, b, truth, CLASSES, n_boot=4000, rng=np.random.default_rng(9))
    three = stats.t5_contrast([a] * 3, [b] * 3, truth, CLASSES, n_boot=4000, rng=np.random.default_rng(9))
    assert three["delta_t5"] == pytest.approx(one["delta_t5"], abs=1e-12)
    assert three["p_value"] == pytest.approx(one["p_value"], abs=1e-12)
    assert three["policies"]["sd"] == pytest.approx(0.0, abs=1e-12)
    assert three["ci"] == pytest.approx(one["ci"], abs=1.5)  # the same distribution from different draws
    assert "policies" not in one


def test_generator_is_required_and_global_state_is_untouched():
    a, b, truth = random_pair(16, (5, 4, 3, 6), 4)
    for bad in (None, 7, np.random.RandomState(0)):
        with pytest.raises(TypeError, match="Generator"):
            stats.t5_contrast(a, b, truth, CLASSES, rng=bad)
    with pytest.raises(TypeError):
        stats.t5_contrast(a, b, truth, CLASSES)  # no default generator
    with pytest.raises(ValueError, match="n_boot"):
        stats.t5_contrast(a, b, truth, CLASSES, n_boot=1, rng=np.random.default_rng(0))
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    stats.t5_contrast(a, b, truth, CLASSES, n_boot=50, rng=np.random.default_rng(0))
    stats.mde({c: 0.3 for c in CLASSES}, {c: 20 for c in CLASSES}, CLASSES, 50, rng=np.random.default_rng(0))
    assert (np.random.get_state()[1] == before).all()


# ---- power and the minimum detectable effect ---------------------------------------------------

SIZES = dict(zip(CLASSES, (96, 65, 25, 54)))


def test_power_is_the_share_of_simulated_tests_rejected():
    """Restated with brute-force p-values at sizes small enough to enumerate."""
    sizes = dict(zip(CLASSES, (5, 4, 3, 4)))
    probabilities = {"a": (0.5, 0.1), "b": (0.4, 0.0), "c": (0.3, 0.2), "d": (0.6, 0.05)}
    rng, rejected = np.random.default_rng(21), 0
    draws = [rng.multinomial(sizes[c], [*probabilities[c], 1 - sum(probabilities[c])], size=300) for c in CLASSES]
    for s in range(300):
        values = [sign * 100 / (4 * sizes[c]) for c, d in zip(CLASSES, draws) for sign, column in ((1, 0), (-1, 1))
                  for _ in range(d[s, column])]
        rejected += brute_force_p(values) < 0.05 if values else 0
    assert stats.t5_power(probabilities, sizes, CLASSES, 300, rng=np.random.default_rng(21)) == rejected / 300


def test_the_test_holds_its_level_and_gains_power_with_the_effect():
    null = stats.t5_power({c: (0.1, 0.1) for c in CLASSES}, SIZES, CLASSES, 3000, rng=np.random.default_rng(1))
    assert null <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 3000)
    weak = stats.t5_power({c: (0.12, 0.08) for c in CLASSES}, SIZES, CLASSES, 1000, rng=np.random.default_rng(1))
    strong = stats.t5_power({c: (0.19, 0.01) for c in CLASSES}, SIZES, CLASSES, 1000, rng=np.random.default_rng(1))
    assert null < weak < strong and strong > 0.95


def test_mde_is_reproducible_and_bisects_at_fixed_discordance():
    discordance = {"a": 0.1, "b": 0.15, "c": 0.2, "d": 0.05}
    first = stats.mde(discordance, SIZES, CLASSES, 500, rng=np.random.default_rng(3))
    assert first == stats.mde(discordance, SIZES, CLASSES, 500, rng=np.random.default_rng(3))
    # The effect is (100 / K) theta sum(discordance) at a theta on the 2^-14 bisection grid.
    steps = first / (25 * sum(discordance.values())) * 2 ** 14
    assert 0 < steps <= 2 ** 14 and steps == pytest.approx(round(steps), abs=1e-6)
    theta = round(steps) / 2 ** 14
    balance = lambda t: {c: (discordance[c] * (1 + t) / 2, discordance[c] * (1 - t) / 2) for c in CLASSES}
    power = lambda t: stats.t5_power(balance(t), SIZES, CLASSES, 500, rng=np.random.default_rng(3))
    assert power(theta) >= 0.8 > power(theta - 2 ** -14)  # every step replays the same simulated trials


def test_mde_is_none_when_no_effect_is_detectable_and_is_never_a_margin():
    assert stats.mde({c: 0.0 for c in CLASSES}, SIZES, CLASSES, 100, rng=np.random.default_rng(0)) is None
    assert stats.mde({c: 0.05 for c in CLASSES}, {c: 6 for c in CLASSES}, CLASSES, 200, rng=np.random.default_rng(0)) is None
    assert "never an equivalence margin" in stats.mde.__doc__
