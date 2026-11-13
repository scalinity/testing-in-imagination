"""Gate G_resume (gates/resume_gate.py): verdicts on hand-made bench reports, under both interpreters."""
import json
import pathlib
import subprocess
import sys

import pytest

RULE = pathlib.Path(__file__).resolve().parents[1] / "gates" / "resume_gate.py"


def good():
    run = {"outcome": "ok", "follows_plan": True, "optimizer_steps": 120}
    return {"stage": "bench-nemotron", "reference": dict(run), "training": dict(run),
            "sequences_identical": True, "visits_identical": True}


def verdict(tmp_path, report, python=sys.executable):
    files = []
    if report is not None:
        path = tmp_path / "bench-nemotron.json"
        path.write_text(json.dumps(report))
        files.append(str(path))
    out = subprocess.run([python, str(RULE), *files], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_met_when_every_check_holds(tmp_path):
    v = verdict(tmp_path, good())
    assert v["result"] == "met" and v["path"] == "training_track_proceeds"
    assert set(v["evidence"].values()) == {1}


@pytest.mark.parametrize("change", [
    lambda r: r.update(sequences_identical=False),
    lambda r: r.update(visits_identical=None),
    lambda r: r.update(sequences_identical="true"),
    lambda r: r["reference"].update(outcome="departed_from_plan"),
    lambda r: r["training"].update(outcome="cuda_out_of_memory"),
    lambda r: r["training"].update(follows_plan=False),
    lambda r: r.update(reference=None),
    lambda r: r.pop("visits_identical"),
])
def test_not_met_on_any_failed_or_missing_check(tmp_path, change):
    report = good()
    change(report)
    v = verdict(tmp_path, report)
    assert v["result"] == "not_met" and v["path"] == "training_track_held"


def test_not_met_without_a_report(tmp_path):
    assert verdict(tmp_path, None)["result"] == "not_met"
    bad = tmp_path / "bench-nemotron.json"
    bad.write_text("{not json")
    out = subprocess.run([sys.executable, str(RULE), str(bad)], capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["evidence"]["report_present"] == 0


def test_same_verdict_under_the_watcher_python(tmp_path):
    python39 = pathlib.Path("/usr/bin/python3")
    if not python39.exists():
        pytest.skip("no /usr/bin/python3")
    for report in (good(), {**good(), "sequences_identical": False}):
        assert verdict(tmp_path, report, str(python39)) == verdict(tmp_path, report)
