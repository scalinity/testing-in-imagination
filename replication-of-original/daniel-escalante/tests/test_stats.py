"""Hand-calculated metric and paired-inference checks against the real upstream."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from omegaconf import OmegaConf

from replication.stats import (
    CLASSES, exact_mcnemar, expected_cost, holm_adjust, paired_bootstrap,
    seed_difference_ci, single_arm_intervals, summary, upper_bound_invariant,
)


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "evaluation": {"test_costs_usd": {"ct": 100, "physical examination": 20}},
        "reward_function": {"reward_functions": [
            {"_target_": "src.reward_function.diagnosis_reward.DiagnosisReward", "correct_reward": 1,
             "incorrect_reward": 0, "undiagnosed_reward": 0},
            {"_target_": "src.reward_function.format_reward.FormatReward", "invalid_reward": 0, "valid_reward": 1},
            {"_target_": "src.reward_function.test_cost_reward.TestCostReward", "test_cost": {"ct": -1}, "single_cost": False},
        ]},
    })


@pytest.fixture
def rows():
    predictions = ["appendicitis", "appendicitis", "diverticulitis", None, "pancreatitis"]
    labels = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis", "pancreatitis"]
    triplets = [[("appendicitis", .9, labels[0])], [("appendicitis", .9, labels[1])],
                [("unknown", .5, labels[2])], [], [("pancreatitis", .3, labels[4])]]
    return [{"patient_id": i, "label": label, "prediction": pred, "provided_tests": ["physical examination"],
             "requested_tests": ["physical examination", "ct"], "ece_triplets": trips}
            for i, (label, pred, trips) in enumerate(zip(labels, predictions, triplets))]


def test_known_confusion_counts_and_none_behavior(rows, cfg):
    result = summary(rows, cfg)
    assert result["claims"] == pytest.approx({"T1": 100, "T2": 0, "T3": 100, "T4": 50,
                                              "T5": 62.5, "T6": 100 * 6 / 9,
                                              "T7": 100 * (2 / 3 + 0 + 1 + 2 / 3) / 4, "T8": 20})
    assert result["pooled_accuracy"] == 60
    assert result["upstream"]["ignore/accuracy"] == .75
    assert result["upstream"]["ignore/micro_f1"] == .75
    assert result["upstream"]["none_fraction"] == .2
    assert result["hypothesis"]["accuracy"] == .5
    assert result["hypothesis"]["ece"] == pytest.approx(.5)
    assert result["hypothesis"]["unknown_fraction"] == .25
    assert result["hypothesis"]["ece_triplet_count"] == 3
    assert result["cost"] == {"provided_mean_usd": 20, "provided_total_usd": 100,
        "requested_mean_usd": 120, "requested_total_usd": 600, "unavailable_requested_mean_usd": 100,
        "repeated_available_request_mean_usd": 0,
        "missing_count": 5, "missing_mean": 1, "provided_count": 5, "requested_count": 10}
    json.dumps(result, allow_nan=False)


def test_missing_observations_are_unassessable(rows, cfg):
    for row in rows:
        row["ece_triplets"] = []
    result = summary(rows, cfg)
    assert result["hypothesis"]["accuracy"] is None
    assert result["hypothesis"]["ece"] is None
    assert result["upstream"]["ece"] == 0  # raw upstream behavior retained
    result = summary(rows[:1], cfg)
    assert result["claims"]["T2"] is None
    assert result["claims"]["T5"] is None


def test_unknowns_count_for_hypothesis_accuracy_but_not_ece(rows, cfg):
    for row in rows:
        row["ece_triplets"] = [("unknown", .7, row["label"])]
    result = summary(rows, cfg)
    assert result["hypothesis"]["accuracy"] == 0
    assert result["hypothesis"]["ece"] is None


@pytest.mark.parametrize("confidence", [-.1, 1.1, float("nan"), float("inf"), "0.5", True])
def test_invalid_probabilities_rejected(rows, cfg, confidence):
    rows[0]["ece_triplets"][0] = ("appendicitis", confidence, "appendicitis")
    with pytest.raises(ValueError):
        summary(rows, cfg)


def test_empty_duplicate_invalid_and_missing_cost_rows_rejected(rows, cfg):
    with pytest.raises(ValueError, match="empty"):
        summary([], cfg)
    with pytest.raises(ValueError, match="duplicate"):
        summary(rows + [rows[0]], cfg)
    rows[0]["provided_tests"] = ["ct", "ct"]
    with pytest.raises(ValueError, match="unique"):
        summary(rows, cfg)
    rows[0]["provided_tests"] = ["unpriced test"]
    with pytest.raises(ValueError, match="configured"):
        summary(rows, cfg)


def test_repeated_requests_preserve_multiplicity(rows, cfg):
    rows[0]["requested_tests"].append("ct")
    result = summary(rows, cfg)
    assert result["cost"]["missing_count"] == 6
    assert result["cost"]["requested_total_usd"] == 700


def test_repeated_available_requests_are_free_but_are_not_missing(rows, cfg):
    rows[0]["requested_tests"].append("physical examination")
    result = summary(rows, cfg)
    assert result["claims"]["T8"] == 20
    assert result["cost"]["missing_count"] == 5
    assert result["cost"]["unavailable_requested_mean_usd"] == 100
    assert result["cost"]["repeated_available_request_mean_usd"] == 4
    assert result["cost"]["requested_total_usd"] == 620


def test_paired_bootstrap_reproducible_and_zero_difference(rows, cfg):
    result = paired_bootstrap(rows, rows, cfg, n_resamples=80)
    assert result == paired_bootstrap(rows, rows, cfg, n_resamples=80)
    assert result["direction"] == "a_minus_b"
    for metric in result["metrics"].values():
        assert metric["estimate"] == 0
        assert metric["ci95"] == [0, 0]
    assert result["mcnemar"]["p_value"] == 1


def test_single_arm_and_paired_known_interval(rows, cfg):
    perfect = copy.deepcopy(rows)
    for row in perfect:
        row["prediction"] = row["label"]
    intervals = single_arm_intervals(perfect, cfg, n_resamples=100)
    assert intervals["metrics"]["T5"]["ci95"] == [100, 100]
    result = paired_bootstrap(perfect, rows, cfg, n_resamples=500)
    assert result["metrics"]["T5"]["estimate"] == 37.5
    lo, hi = result["metrics"]["T5"]["ci95"]
    assert lo == 25 and hi == 50
    assert result["metrics"]["T8"]["ci95"] == [0, 0]
    reverse = paired_bootstrap(rows, perfect, cfg, n_resamples=500)
    assert reverse["metrics"]["T5"]["ci95"] == [-hi, -lo]


def test_mismatched_patients_order_and_labels_rejected(rows, cfg):
    wrong = copy.deepcopy(rows)
    wrong[0]["patient_id"] = 99
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap(rows, wrong, cfg, n_resamples=10)
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap(rows, rows[::-1], cfg, n_resamples=10)
    wrong = copy.deepcopy(rows)
    wrong[0]["label"] = CLASSES[1]
    wrong[0]["ece_triplets"] = []
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap(rows, wrong, cfg, n_resamples=10)


def test_clusters_preserve_all_admissions_and_class_coverage(rows, cfg):
    # Two repeated subjects cover all four classes; resampling entire subjects
    # gives a variable denominator (three versus two admissions).
    subjects = [1, 1, 2, 2, 1]
    result = single_arm_intervals(rows, cfg, n_resamples=40, cluster_ids=subjects)
    assert result["n_clusters"] == 2
    assert "subject-cluster" in result["method"]
    assert result["metrics"]["T5"]["valid_resamples"] == 40
    # Each class-presence profile has only one subject here, hence degeneracy.
    assert result["metrics"]["T5"]["ci95"] == [62.5, 62.5]
    with pytest.raises(ValueError, match="align"):
        single_arm_intervals(rows, cfg, n_resamples=10, cluster_ids=[1])
    with pytest.raises(ValueError, match="four"):
        single_arm_intervals(rows[:1], cfg, n_resamples=10)


def test_cluster_draws_move_whole_subjects(cfg):
    rows = [{"patient_id": i, "label": label,
             "prediction": label if i < 4 else None,
             "provided_tests": ["ct"] if i < 4 else ["physical examination"],
             "requested_tests": ["ct", "physical examination"], "ece_triplets": []}
            for i, label in enumerate(CLASSES * 2)]
    result = single_arm_intervals(rows, cfg, n_resamples=200, cluster_ids=[1] * 4 + [2] * 4)
    assert result["metrics"]["T5"]["estimate"] == 50
    assert result["metrics"]["T5"]["ci95"] == [0, 100]
    assert result["metrics"]["T8"]["ci95"] == [20, 100]
    paired = paired_bootstrap(rows, rows, cfg, n_resamples=30, cluster_ids=[1] * 4 + [2] * 4)
    assert paired["mcnemar"]["p_value"] is None
    assert paired["mcnemar"]["status"] == "unassessable"


def test_sparse_calibration_does_not_discard_undefined_draws(rows, cfg):
    for row in rows:
        row["ece_triplets"] = []
    rows[-1]["ece_triplets"] = [("pancreatitis", .3, "pancreatitis")]
    result = single_arm_intervals(rows, cfg, n_resamples=100)
    assert result["metrics"]["ece"]["estimate"] == pytest.approx(.7)
    assert result["metrics"]["ece"]["valid_resamples"] < 100
    assert result["metrics"]["ece"]["ci95"] is None
    assert result["metrics"]["ece"]["status"] == "unassessable"


def test_exact_binomial_known_cases():
    result = exact_mcnemar([True] * 10, [False] * 10)
    assert result["a_only_correct"] == 10 and result["b_only_correct"] == 0
    assert result["p_value"] == pytest.approx(2 / 1024)
    assert exact_mcnemar([True, False], [False, True])["p_value"] == 1
    assert exact_mcnemar([False], [False])["p_value"] == 1
    for a, b in (([], []), ([True], [True, False]), ([1], [0])):
        with pytest.raises(ValueError):
            exact_mcnemar(a, b)


def test_student_t_known_and_unfinished_seed_handling():
    result = seed_difference_ci([1, 2, 3])
    assert result["estimate"] == 2 and result["df"] == 2
    assert result["ci95"] == pytest.approx([-.4841377117, 4.4841377117])
    assert seed_difference_ci([0, 0])["ci95"] == [0, 0]
    assert seed_difference_ci([1])["status"] == "unassessable"
    assert seed_difference_ci([])["estimate"] is None
    incomplete = seed_difference_ci([1, 2, None])
    assert incomplete["declared_n"] == 3 and incomplete["n"] == 2
    assert incomplete["ci95"] is None


def test_holm_uses_whole_planned_family():
    adjusted = holm_adjust({"zs": .01, "no_cost": .04, "decision_only": .03, "minus1": None})
    assert adjusted["zs"]["holm_p_value"] == .04
    assert adjusted["no_cost"]["holm_p_value"] == .09
    assert adjusted["decision_only"]["holm_p_value"] == .09
    assert adjusted["zs"]["reject"]
    assert not adjusted["minus1"]["reject"]


def test_expected_cost_and_justified_reward_ceiling(cfg):
    assert expected_cost(100, .3) == 30
    assert expected_cost(100, 0) == 0
    assert expected_cost(100, 1) == 100
    for arm in ("primary", "decision_only", "no_cost", "incorrect_minus1"):
        assert upper_bound_invariant([0, 1, 2], cfg, arm)["holds"]
    assert not upper_bound_invariant([2.01], cfg)["holds"]
    with pytest.raises(ValueError, match="justified"):
        upper_bound_invariant([2], cfg, "oracle")
    cfg.reward_function.reward_functions[2].test_cost.ct = 1
    with pytest.raises(ValueError, match="positive"):
        upper_bound_invariant([2], cfg)
    for cost, availability in ((-1, .5), (2, 1.1), (float("nan"), .5)):
        with pytest.raises(ValueError):
            expected_cost(cost, availability)


def test_cpu_import_has_no_gpu_dependencies():
    subprocess.run([sys.executable, "-c",
                    "import sys; import replication.stats; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"],
                   check=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PYTHONDONTWRITEBYTECODE": "1"})
