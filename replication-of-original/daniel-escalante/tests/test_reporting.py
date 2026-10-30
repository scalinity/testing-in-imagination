"""Reports identify every claim and never promote synthetic demonstrations."""

import json

import pytest

from replication.reporting import RESULTS, render_report


def test_report_all_sections_synthetic_banner_and_no_placeholders(tmp_path):
    path = render_report({}, tmp_path / "report.md", "synthetic")
    report = path.read_text()
    assert report.index("SYNTHETIC") < report.index("T1–T8")
    assert "INVENTED DATA" in report and "MOCK UPDATES ARE NOT LEARNING" in report
    assert "${" not in report
    for claim in [*(f"T{i}" for i in range(1, 9)), "B1", "B2", "C1", "C2", "C3", "C4", "I1", "I5", "F-1", "F-2", "F-3", "F-4"]:
        assert claim in report
    for section in ("Ablations", "Paired uncertainty and significance", "Leakage and split sensitivity", "Provided versus requested cost", "Honest gaps"):
        assert section in report
    assert "re-implementation on a different backbone" in report
    assert "incorrect reward 0" in report and "incorrect reward −1" in report
    assert "ReAct implementation absent" in report
    assert "seed 269" in report and "D-4" in report
    assert "unassessable subject leakage, never zero" in report
    assert "Table 3 values are absent" in report


def test_renderer_populates_aggregate_values_intervals_and_cost(tmp_path):
    arm = {"n": 5, "claims": {"T5": 62.5, "T8": 20}, "pooled_accuracy": 60,
           "hypothesis": {"accuracy": .5, "ece": .2},
           "cost": {"provided_mean_usd": 20, "requested_mean_usd": 120,
                    "unavailable_requested_mean_usd": 100, "missing_count": 5}}
    comparison = {"metrics": {"T5": {"estimate": 5, "ci95": [1, 10]}}, "seed": 269,
                  "n_resamples": 10000, "method": "class-stratified patient percentile bootstrap",
                  "mcnemar": {"p_value": .001}}
    inputs = {"arms": {"primary": arm, "zero_shot": arm},
              "paired": {"primary_vs_zero_shot": comparison},
              "intervals": {"primary": {"metrics": {"T5": {"ci95": [50, 75]}}}},
              "seed_differences": {"primary_vs_zero_shot": {"n": 1, "declared_n": 1, "estimate": 5, "ci95": None, "status": "unassessable"}},
              "leakage": {"admission_overlap": 0, "subject_overlap": None},
              "run": {"seed": 269, "seeds": [269], "training_completed": True},
              "gaps": ["CUDA smoke check unrun."]}
    report = render_report(inputs, tmp_path / "report.md", "synthetic").read_text()
    assert "62.50" in report and "[50.00, 75.00]" in report
    assert "[1.00, 10.00]" in report
    assert "0.004000" in report  # planned four-member Holm family
    assert "120.00" in report and "100.00" in report
    assert "not established by synthetic/mock execution" in report
    assert "CUDA smoke check unrun." in report
    assert "1/1" in report


def test_paper_values_ledger_only_and_missing_table3_is_null():
    paper = json.loads((RESULTS / "paper_values.json").read_text())
    assert paper["source"] == "REPRODUCTION.md"
    assert paper["table1"]["T8"]["value"] == 1295.61
    assert paper["baselines"]["B1"]["value"] == 64.5
    assert paper["dynamics"]["C4"]["decision_only_mean_recall"] is None
    assert paper["table3"]["primary_mean_recall"] is None


def test_mimic_source_and_invalid_sources(tmp_path):
    report = render_report({}, tmp_path / "report.md", "mimic").read_text()
    assert "MIMIC source" in report
    assert "different backbone" in report
    with pytest.raises(ValueError):
        render_report({}, tmp_path / "bad.md", "training")
    with pytest.raises(ValueError):
        render_report({"arms": {"oracle": {}}}, tmp_path / "bad.md", "mimic")
