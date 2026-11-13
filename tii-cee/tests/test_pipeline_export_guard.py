"""The export guard, and the Python 3.9 requirement on everything the watcher imports."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tiicee.pipeline import export_guard
from tiicee.pipeline.export_guard import Refused, approve, check_aggregate
from tiicee.pipeline.paths import Paths

PACKAGE = Path(__file__).resolve().parents[1]
# Everything that runs under the compute node's Python 3.9: the watcher, what it imports, and gate rules.
WATCHER_FILES = [PACKAGE / "tiicee" / "__init__.py"] + [
    PACKAGE / "tiicee" / "pipeline" / name for name in (
        "__init__.py", "paths.py", "util.py", "authorization.py", "graph.py", "manifest.py",
        "export_guard.py", "sbatch.py", "watcher.py")] + sorted((PACKAGE / "gates").glob("*.py"))

AGGREGATE = {"experiment": "E2", "status": "COMPLETED", "n": 240, "gain_nats": 0.0312, "ci": [-0.01, 0.07],
             "ok": True, "missing": None, "digest": "72ccc89c8a825c82578c8efdf942025e3ce60d72f179b8b13489737ee37221ae",
             "written": "2026-11-13T03:00:00-05:00", "by_reader": {"phi-4": {"auc": 0.71}, "olmo.13b": {"auc": 0.68}}}


def code_of(value):
    with pytest.raises(Refused) as caught:
        check_aggregate(value)
    return caught.value.code


def test_an_aggregate_of_names_counts_and_numbers_passes():
    check_aggregate(AGGREGATE)


def test_the_rules_of_the_replication_guard_still_hold():
    """Every key replication/manifest.py's assert_aggregate forbids is refused here too."""
    source = (PACKAGE.parent / "replication-of-original" / "daniel-escalante" / "replication" / "manifest.py").read_text()
    block = source[source.index("FORBIDDEN_KEYS = {"):source.index("def assert_aggregate")]
    original = ast.literal_eval(block.split("=", 1)[1].strip())
    assert original == export_guard.FORBIDDEN_KEYS
    for key in original:
        assert code_of({"summary": {"nested": [{key: 1}]}}) == "forbidden_key", key
        assert code_of({key.upper(): 1}) == "forbidden_key", key


@pytest.mark.parametrize("key", [
    "row", "rows", "row_id", "per_row", "patient", "patients", "n_patients", "patientId", "subject", "subject_ids",
    "hadm", "hadmid", "stay", "stay_id", "icu_stays", "text", "result_text", "note", "notes", "discharge_note",
    "prompt", "prompts", "full_prompt", "generation", "generations", "hpi", "HPI", "by_hpi"])
def test_keys_that_resemble_patient_level_fields_are_refused(key):
    assert code_of({key: 1}) == "forbidden_key"


@pytest.mark.parametrize("key", ["context", "growth", "throw", "arrow", "narrow_ci", "patience", "n_admissions",
                                 "generated", "notable", "restart", "shipping"])
def test_ordinary_keys_are_not_mistaken_for_them(key):
    check_aggregate({key: 1})


@pytest.mark.parametrize("value, code", [
    ({"summary": "the patient improved after treatment"}, "string_not_identifier"),
    ({"label": "two words"}, "string_not_identifier"),
    ({"label": ""}, "string_not_identifier"),
    ({"label": "x" * 200}, "string_not_identifier"),
    ({"a key with spaces": 1}, "key_not_identifier"),
    ({"1st": 1}, "key_not_identifier"),
    ({"k" * 65: 1}, "key_not_identifier"),
    ({"value": float("nan")}, "number_not_finite"),
    ({"value": float("inf")}, "number_not_finite"),
    ({"value": b"bytes"}, "value_type"),
    ({"value": {1: 2}}, "forbidden_key"),
    ([{"n": 1}], "not_an_object"),
    ("a string", "not_an_object"),
])
def test_anything_else_is_refused_with_a_code_that_does_not_echo_it(value, code):
    assert code_of(value) == code


def test_approve_writes_only_what_passes(tmp_path):
    paths = Paths(root=tmp_path, code=tmp_path)
    target = approve(paths, "e2_summary.json", AGGREGATE)
    assert target == paths.approved / "e2_summary.json" and json.loads(target.read_text()) == AGGREGATE
    assert approve(paths, "E2/readers.json", {"n": 9}) == paths.approved / "E2" / "readers.json"
    for name in ("../escape.json", "notes.txt", "a b.json", ".hidden.json", "E2/../x.json", "/abs.json"):
        with pytest.raises(Refused) as caught:
            approve(paths, name, AGGREGATE)
        assert caught.value.code == "file_name"
    with pytest.raises(Refused):
        approve(paths, "bad.json", {"rows": [1, 2]})
    assert sorted(path.name for path in paths.approved.rglob("*.json")) == ["e2_summary.json", "readers.json"]
    assert export_guard.surface_check(paths) == []


def test_command_line_approves_or_refuses_without_echoing_content(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TII_ROOT", str(tmp_path / "tii"))
    good, bad, broken = tmp_path / "good.json", tmp_path / "bad.json", tmp_path / "broken.json"
    good.write_text(json.dumps(AGGREGATE))
    bad.write_text(json.dumps({"n": 3, "hpi": "65 year old with chest pain"}))
    broken.write_text("{not json")
    assert export_guard.main(["approve", str(good)]) == 0
    assert capsys.readouterr().out.strip() == "approved good.json"
    assert export_guard.main(["approve", str(bad), "renamed.json"]) == 1
    assert capsys.readouterr().out.strip() == "refused forbidden_key"
    assert export_guard.main(["approve", str(broken)]) == 1 and capsys.readouterr().out.strip() == "refused not_json"
    (tmp_path / "nan.json").write_text('{"value": NaN}')
    assert export_guard.main(["approve", str(tmp_path / "nan.json")]) == 1
    assert capsys.readouterr().out.strip() == "refused number_not_finite"
    assert [path.name for path in (tmp_path / "tii" / "export" / "approved").iterdir()] == ["good.json"]


def test_event_and_status_shapes():
    ok = "2026-11-13T03:10:00-0500\tblocked\titem=W-001 key=allowance:E2 spent=12.5 estimate=90 allowance=200"
    assert export_guard.event_ok(ok)
    for line in ("2026-11-13T03:10:00-0500\tblocked\tsomething went wrong",
                 "2026-11-13T03:10:00-0500\tsurprise\tstage=a", "blocked\tstage=a",
                 "2026-11-13T03:10:00-0500\tblocked\tstage=a note=free text here", ok + "\n", ok + " "):
        assert not export_guard.event_ok(line), line
    good = "\n".join([export_guard.STATUS_TITLE, "", "updated: 2026-11-13T03:10:00-0500", "## Stages by status", "",
                      "| status | count |", "|---|---|", "| SKIPPED BY GATE | 2 |",
                      "| 2026-11-13T03:10:00-0500 | stage completed | stage=a experiment=E2 attempts=1 gpu_hours=0.458 |", ""])
    assert export_guard.status_violations(good) == []
    assert export_guard.status_violations(good + "## Notes\nnext: call the patient\n| a | two words |\n") == [10, 11, 12]


# ---- Python 3.9 ---------------------------------------------------------------------------------

def python39():
    for candidate in (os.environ.get("TII_TEST_WATCHER_PYTHON"), "/usr/bin/python3", "python3.9"):
        if not candidate:
            continue
        try:
            version = subprocess.run([candidate, "-c", "import sys; print(sys.version_info[:2])"],
                                     capture_output=True, text=True).stdout.strip()
        except OSError:
            continue
        if version == "(3, 9)":
            return candidate
    return None


def test_everything_the_watcher_imports_compiles_and_imports_under_python_39():
    interpreter = python39()
    if interpreter is None:
        pytest.skip("no Python 3.9 interpreter on this machine")
    compiled = subprocess.run([interpreter, "-c", "import sys\nfor name in sys.argv[1:]:\n"
                               "    compile(open(name).read(), name, 'exec')"] + [str(f) for f in WATCHER_FILES],
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    imported = subprocess.run([interpreter, "-c", "import sys, tiicee.pipeline.watcher\n"
                               "extra = sorted(m for m in sys.modules if m.startswith('tiicee') or m == 'numpy')\n"
                               "print(extra)"], cwd=PACKAGE, capture_output=True, text=True,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == str(sorted(
        ["tiicee", "tiicee.pipeline"] + ["tiicee.pipeline." + name for name in (
            "paths", "util", "authorization", "graph", "manifest", "export_guard", "sbatch", "watcher")]))


def test_watcher_and_stage_contract_code_use_the_standard_library_only():
    allowed = set(sys.stdlib_module_names)
    stage_files = [PACKAGE / "tiicee" / "pipeline" / name for name in ("stage.py", "stages.py")]
    for path in WATCHER_FILES + stage_files:
        for node in ast.walk(ast.parse(path.read_text())):
            assert not isinstance(node, ast.Match), path  # Python 3.10 syntax
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module.split(".")[0]]
            else:
                continue
            assert set(names) <= allowed, (path.name, names)
