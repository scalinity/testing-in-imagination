import itertools

import numpy as np
import pytest

from tim import analysis
from tim.enums import DISEASES

SIZES = {"appendicitis": 96, "cholecystitis": 65, "diverticulitis": 25, "pancreatitis": 54}


def rows_from(labels, predictions):
    return [{"patient_id": 1000 + i, "label": label, "prediction": prediction}
            for i, (label, prediction) in enumerate(zip(labels, predictions))]


def random_pair(seed, n_discordant):
    rng = np.random.default_rng(seed)
    labels = [DISEASES[i % 4] for i in range(60)]
    base = [label if rng.random() < 0.7 else None for label in labels]
    method = list(base)
    for i in rng.choice(60, size=n_discordant, replace=False):
        method[i] = labels[i] if base[i] != labels[i] else None
    return rows_from(labels, method), rows_from(labels, base)


def brute_force_p(w, d):
    v = [wi * di for wi, di in zip(w, d) if di]
    observed = abs(sum(v))
    flips = [abs(sum(s * x for s, x in zip(signs, v))) for signs in itertools.product((-1, 1), repeat=len(v))]
    return sum(f >= observed - 1e-9 for f in flips) / len(flips)


def test_weights_make_delta_t5_the_mean_recall_difference():
    a, b = random_pair(1, 9)
    w, d = analysis.paired(a, b)
    recall = lambda rows: np.mean([np.mean([r["prediction"] == r["label"] for r in rows if r["label"] == c]) for c in DISEASES])
    assert analysis.delta_t5(w, d) == pytest.approx(100 * (recall(a) - recall(b)))


@pytest.mark.parametrize("seed, k", [(2, 1), (3, 7), (4, 12)])
def test_exact_randomization_test_equals_brute_force(seed, k):
    a, b = random_pair(seed, k)
    w, d = analysis.paired(a, b)
    result = analysis.randomization_test(w, d, seed=0)
    assert result["method"] == "exact" and result["discordant"] == k
    assert result["p_value"] == pytest.approx(brute_force_p(w, d))
    assert result["p_value"] == pytest.approx(analysis.exact_p_by_class(w, d))


def test_drawn_randomization_test_is_reproducible_and_near_exact():
    a, b = random_pair(5, 30)
    w, d = analysis.paired(a, b)
    first = analysis.randomization_test(w, d, seed=11)
    assert first == analysis.randomization_test(w, d, seed=11)
    assert first["method"] == "draws" and first["draws"] == 100_000
    assert first["p_value"] == pytest.approx(analysis.exact_p_by_class(w, d), abs=0.006)


def test_bootstrap_point_estimate_is_the_weighted_sum():
    from tim.access import baseline_cfg
    a, b = random_pair(6, 15)
    for rows in (a, b):
        for r in rows:
            r.update(provided_tests=[], requested_tests=[], ece_triplets=[])
    w, d = analysis.paired(a, b)
    result = analysis.bootstrap(a, b, _cfg(), seed=3, n_resamples=200)
    assert result["estimate"] == pytest.approx(analysis.delta_t5(w, d), abs=1e-9)
    assert result["ci95"][0] <= result["estimate"] <= result["ci95"][1]


def _cfg():
    from replication.config import compose
    return compose("primary")


def test_v1_never_changes_a_non_null_prediction():
    rows = [{"patient_id": 1, "label": "appendicitis", "raw_prediction": "pancreatitis", "prediction": "pancreatitis",
             "hypotheses": ["appendicitis"]},
            {"patient_id": 2, "label": "appendicitis", "raw_prediction": None, "prediction": None,
             "hypotheses": ["unknown", "appendicitis", "unknown"]}]
    v1 = analysis.v1_rows(rows)
    assert [r["prediction"] for r in v1] == ["pancreatitis", "appendicitis"]
    assert [r["prediction_source"] for r in v1] == ["diagnosis", "hypothesis_fallback"]


def test_analytic_controls_reproduce_the_specs_references():
    nonanswers = {"appendicitis": 5, "cholecystitis": 6, "diverticulitis": 4, "pancreatitis": 14}
    labels, predictions = [], []
    for disease, size in SIZES.items():
        labels += [disease] * size
        predictions += [None] * nonanswers[disease] + [disease] * (size - nonanswers[disease])
    controls = analysis.analytic_controls(rows_from(labels, predictions), majority="appendicitis")
    assert round(controls["uniform_random"], 2) == 3.52
    assert round(controls["training_majority"], 2) == 1.30
    assert round(controls["perfect_recovery"], 2) == 14.09


@pytest.mark.parametrize("discordance, expected", [(0.10, 6.4), (0.20, 9.1)])
def test_minimum_detectable_effect_matches_the_specs_orientation(discordance, expected):
    mde = analysis.minimum_detectable_effect({c: discordance for c in DISEASES}, SIZES, sims=4000, seed=1)
    assert mde == pytest.approx(expected, abs=0.35)


def test_recovery_and_regression_counts():
    comparator = rows_from(["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"],
                           [None, None, "diverticulitis", "pancreatitis"])
    method = rows_from(["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"],
                       ["appendicitis", "pancreatitis", "diverticulitis", "appendicitis"])
    counts = analysis.recovery(method, comparator)
    assert counts == {"baseline_nonanswers": 2, "newly_answered": 2, "correct_recoveries": 1,
                      "recovery_yield": 0.5, "newly_answered_accuracy": 0.5, "regressions": 1}


@pytest.mark.parametrize("p, ci, significant", [(0.01, [0.5, 3.0], True), (0.2, [0.5, 3.0], False),
                                                (0.01, [-0.5, 3.0], False)])
def test_significance_wording(p, ci, significant):
    text = analysis.significance(p, ci)
    assert (text == "statistically significant") is significant
    assert significant or text == "not statistically significant"


def test_repeated_requests_include_repeats_of_unavailable_tests():
    from tim.access import baseline_cfg
    rows = [{"patient_id": i, "label": d, "prediction": d, "provided_tests": [], "requested_tests": ["ct", "ct"],
             "unavailable_requests": ["ct", "ct"], "ece_triplets": []} for i, d in enumerate(DISEASES)]
    requests = analysis.summarize(rows, _cfg())["requests"]
    assert requests == {"requested": 8, "provided": 0, "unavailable": 8, "repeated": 4}
