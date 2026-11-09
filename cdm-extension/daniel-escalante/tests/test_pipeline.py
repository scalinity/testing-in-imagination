import json
import os
from pathlib import Path
import subprocess

import pytest

from tim.events import load_schema, validate
from tim.paths import OWN

PUBLISHED = ("validate", "smoke", "bench-a", "val-a", "analyze-val-a", "test-a", "phase-a",
             "hpi-audit", "wm-fit", "inventory", "leakage", "bench-b", "val-b", "analyze-val-b", "test-b", "phase-b")


def run_all(environ, results):
    env = {**os.environ, **environ, "TIM_RESULTS_DIR": str(results)}
    return subprocess.run(["bash", str(OWN / "scripts/tim.sh"), "all"], env=env, capture_output=True, text=True,
                          timeout=900)


def tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    outputs = []
    for name in ("first", "second"):
        base = tmp_path_factory.mktemp(name)
        dirs = {k: base / k for k in ("data", "baseline", "work", "admissions")}
        for path in dirs.values():
            path.mkdir()
        environ = {"CDM_DATA_SOURCE": "synthetic", "CDM_DATA_DIR": str(dirs["data"]), "CDM_WORK_DIR": str(dirs["baseline"]),
                   "TIM_WORK_DIR": str(dirs["work"]), "TIM_ADMISSION_DIR": str(dirs["admissions"])}
        result = run_all(environ, base / "results")
        assert result.returncode == 0, (dirs["work"] / "private-logs").glob("*").__next__().read_text()[-3000:]
        outputs.append(base / "results" / "synthetic")
    return outputs


def test_synthetic_end_to_end_publishes_every_phase_a_bundle(two_runs):
    first, _ = two_runs
    assert sorted(p.name for p in first.iterdir()) == sorted(PUBLISHED)
    metrics = json.loads((first / "phase-a" / "metrics.json").read_text())
    validate(metrics, load_schema("metrics"))
    assert metrics["source"] == "synthetic"
    assert metrics["v0_rerun"] == {"identical": 24, "total": 24, "qualified": False}
    validation = json.loads((first / "analyze-val-a" / "validation.json").read_text())
    assert validation["shadow"]["passed"] and validation["premise"]["passed"]


def test_second_run_is_byte_identical(two_runs):
    first, second = two_runs
    assert tree(first) == tree(second)


def test_no_canary_reaches_any_published_file(two_runs):
    for content in tree(two_runs[0]).values():
        assert b"TIM-CANARY" not in content


def test_report_carries_the_synthetic_banner_and_rendered_numbers(two_runs):
    report = (two_runs[0] / "phase-a" / "REPORT.md").read_text()
    assert report.startswith("> **SYNTHETIC**")
    metrics = json.loads((two_runs[0] / "phase-a" / "metrics.json").read_text())
    assert f"{metrics['primary']['delta_t5']:.2f}" in report
    assert "not a reproduction" in report.lower() or "not a reproduction of the paper" in report.lower()


def test_phase_b_publishes_the_specs_deliverables(two_runs):
    from tim.enums import FAMILY_B, PHASE_B_ARMS
    phase_b = two_runs[0] / "phase-b"
    assert {p.name for p in phase_b.iterdir() if p.is_file()} == {
        "metrics.json", "REPORT.md", "wm_quality.json", "leakage.json", "inventory_card.json", "hpi_audit.json"}
    metrics = json.loads((phase_b / "metrics.json").read_text())
    validate(metrics, load_schema("metrics"))
    assert set(metrics["family_b"]) == set(FAMILY_B)
    assert all(metrics["family_b"][k]["p_holm"] is not None for k in FAMILY_B)
    assert set(metrics["descriptive"]) == {f"{arm}-P0" for arm in PHASE_B_ARMS}
    assert set(metrics["variants"]) == {"P0", *PHASE_B_ARMS}
    assert set(metrics["quality"]) == {"val", "test"} and set(metrics["leakage"]) == {"val", "test"}
    report = (phase_b / "REPORT.md").read_text()
    assert report.startswith("> **SYNTHETIC**") and "not full world-model planning" in report
    assert f"{metrics['family_b']['B-oracle-B-uniform']['delta_t5']:+.2f}" in report
