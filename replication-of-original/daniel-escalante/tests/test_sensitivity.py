"""Post-S6 supplementary aggregates: confusion counts and the decoding check."""

import copy

import pytest
from omegaconf import OmegaConf

from replication.manifest import assert_aggregate
from replication.sensitivity import ALL_ARMS, admit, build, render


@pytest.fixture
def cfg():
    return OmegaConf.create({"evaluation": {"test_costs_usd": {"ct": 100, "physical examination": 20}}})


@pytest.fixture
def rows():
    predictions = ["appendicitis", "appendicitis", "diverticulitis", None, "pancreatitis"]
    labels = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis", "pancreatitis"]
    return [{"patient_id": 900 + i, "label": label, "prediction": pred, "provided_tests": ["physical examination"],
             "requested_tests": ["physical examination", "ct", "ct"], "ece_triplets": [("appendicitis", .5, label)]}
            for i, (label, pred) in enumerate(zip(labels, predictions))]


def test_greedy_only_supplement_reports_counts_without_patient_fields(rows, cfg):
    greedy = {arm: copy.deepcopy(rows) for arm in ALL_ARMS}
    result = build(greedy, cfg, seed=269, n_resamples=50)
    assert_aggregate(result)
    primary = result["greedy"]["primary"]
    assert primary["confusion"]["pancreatitis"]["undiagnosed"] == 1
    assert primary["undiagnosed"] == 1 and primary["requests"] == 15 and primary["repeated_requests"] == 5
    assert "upstream_decoding" not in result
    text = render(result)
    assert "Confusion" in text and "undiagnosed" in text
    assert "900" not in text  # patient identifiers never reach the report


def test_decoding_check_reports_intervals_and_paired_differences(rows, cfg):
    greedy = {arm: copy.deepcopy(rows) for arm in ALL_ARMS}
    sampled = copy.deepcopy(greedy)
    for row in sampled["zero_shot"]:
        row["prediction"] = row["label"]
    result = build(greedy, cfg, seed=269, n_resamples=50, sampled=sampled)
    assert_aggregate(result)
    zero_shot = result["upstream_decoding"]["zero_shot"]
    assert zero_shot["temperature"] == 1.0
    assert zero_shot["claims"]["T5"] == 100
    assert zero_shot["minus_greedy"]["T5"]["estimate"] == 100 - result["upstream_decoding"]["primary"]["claims"]["T5"]
    assert zero_shot["intervals"]["T5"]["ci95"] == [100, 100]
    comparisons = result["upstream_decoding_comparisons"]
    assert set(comparisons) == {"primary_vs_zero_shot", "primary_vs_decision_only", "primary_vs_no_cost",
                                "primary_vs_incorrect_minus1"}
    assert comparisons["primary_vs_zero_shot"]["holm"]["holm_p_value"] is not None
    assert "temperature 1.0" in render(result)


def test_decoding_check_admits_each_arm_once_even_after_an_interruption(tmp_path):
    path = tmp_path / "zero_shot-test.json"
    admit(path)
    # Admission is recorded before generation: an interrupted run cannot be retried on test.
    with pytest.raises(RuntimeError, match="runs once"):
        admit(path)
    other = tmp_path / "primary-test.json"
    other.write_text("{}")
    with pytest.raises(RuntimeError, match="runs once"):
        admit(other)


def test_partial_decoding_check_keeps_the_planned_holm_family(rows, cfg):
    greedy = {arm: copy.deepcopy(rows) for arm in ALL_ARMS}
    sampled = {"primary": copy.deepcopy(rows), "zero_shot": copy.deepcopy(rows)}
    for row in sampled["zero_shot"]:
        row["prediction"] = row["label"]
    result = build(greedy, cfg, seed=269, n_resamples=50, sampled=sampled)
    comparison = result["upstream_decoding_comparisons"]["primary_vs_zero_shot"]
    assert comparison["holm"]["holm_p_value"] == pytest.approx(min(1, 4 * comparison["mcnemar"]["p_value"]))
    assert result["upstream_decoding_planned_family"] == ["primary_vs_decision_only", "primary_vs_no_cost",
                                                         "primary_vs_incorrect_minus1", "primary_vs_zero_shot"]


def test_repeated_subjects_make_the_admission_level_mcnemar_unassessable(rows, cfg):
    greedy = {arm: copy.deepcopy(rows) for arm in ALL_ARMS}
    sampled = copy.deepcopy(greedy)
    clusters = [1, 1, 2, 3, 4]
    result = build(greedy, cfg, seed=269, n_resamples=50, sampled=sampled, cluster_ids=clusters)
    assert result["upstream_decoding_comparisons"]["primary_vs_zero_shot"]["mcnemar"]["p_value"] is None
    assert result["upstream_decoding"]["primary"]["intervals"]["T5"]["status"] in ("assessed", "unassessable")


def test_rendered_blocks_name_their_protocol_and_seed(rows, cfg):
    result = build({arm: copy.deepcopy(rows) for arm in ALL_ARMS}, cfg, seed=269, n_resamples=50)
    assert "Protocol: grouped; training seed: 7" in render(result, "grouped", 7)
