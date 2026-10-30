import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from replication.config import OWN, Settings
from replication.manifest import file_digest
from replication.pipeline import STAGES, run_stage, s4, _saved_evaluation


def test_every_stage_dry_runs_with_source_manifest(tmp_path):
    settings = Settings("synthetic", tmp_path / "raw", tmp_path / "work")
    for stage in STAGES:
        result = run_stage(settings, stage, dry_run=True)
        assert result["source"] == "synthetic" and result["dry_run"]
        assert (settings.work_dir / "manifests" / f"dry-run-{stage}.json").exists()
    assert not settings.data_dir.exists()


def test_fake_summarizer_cannot_enter_mimic_mode(tmp_path):
    env = dict(os.environ, CDM_DATA_SOURCE="mimic", CDM_DATA_DIR=str(tmp_path / "real"), CDM_WORK_DIR=str(tmp_path / "work"))
    result = subprocess.run([sys.executable, str(OWN / "scripts/run_stage.py"), "s1", "--fake-summarizer"], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "only permitted with synthetic" in result.stderr


def test_training_and_selection_never_read_test(tmp_path, monkeypatch):
    import replication.pipeline as pipeline
    settings = Settings("synthetic", tmp_path / "raw", tmp_path / "work")
    accesses = []
    def evaluate(settings, arm, protocol, seed, split, adapter_dir=None):
        accesses.append(split)
        return [{"invented": "smoke"}]
    monkeypatch.setattr(pipeline, "_evaluate", evaluate)
    s4(settings)
    assert accesses == ["train", "val"] * 4
    assert "test" not in accesses
    ledger = json.loads((settings.work_dir / "training-ledger.json").read_text())
    assert len(ledger["runs"]) == 4 and all(r["status"] == "PASS" for r in ledger["runs"])


def test_failed_seed_is_retained_and_evaluation_is_not_replayed(tmp_path, monkeypatch):
    import replication.pipeline as pipeline
    settings = Settings("synthetic", tmp_path / "raw", tmp_path / "work")
    def fail(*args, **kwargs):
        raise RuntimeError("controlled failure")
    monkeypatch.setattr(pipeline, "_evaluate", fail)
    with pytest.raises(RuntimeError, match="controlled failure"):
        s4(settings)
    ledger = json.loads((settings.work_dir / "training-ledger.json").read_text())
    assert ledger["runs"][0]["status"] == "FAILED"
    assert ledger["runs"][0]["seed"] == 269
    with pytest.raises(RuntimeError, match="controlled failure"):
        _saved_evaluation(settings, "primary", "parity", 269)
    with pytest.raises(RuntimeError, match="interrupted after admission"):
        _saved_evaluation(settings, "primary", "parity", 269)


def test_full_synthetic_shell_run_and_deterministic_resume(tmp_path):
    env = dict(os.environ, CDM_DATA_SOURCE="synthetic", CDM_DATA_DIR=str(tmp_path / "raw"),
               CDM_WORK_DIR=str(tmp_path / "work"), CDM_SYNTHETIC_N="120", CDM_TRAINING_SEEDS="269", CDM_PYTHON=sys.executable)
    command = ["bash", str(OWN / "scripts/run_all.sh")]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    work = tmp_path / "work"
    report = (work / "REPORT.md").read_text()
    assert "SYNTHETIC" in report and "re-implementation on a different backbone" in report
    for claim in ("T1", "T8", "B1", "B2", "C1", "C4"):
        assert claim in report
    assert all((work / "receipts" / f"{stage}.json").exists() for stage in STAGES)
    before = {str(p.relative_to(work)): file_digest(p) for p in work.rglob("*") if p.is_file()}
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert again.returncode == 0, again.stdout + again.stderr
    assert again.stdout.count("RESUMED") == 7
    after = {str(p.relative_to(work)): file_digest(p) for p in work.rglob("*") if p.is_file()}
    assert before == after
    (work / "prepared/test.csv").write_text("corrupted")
    failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert failed.returncode != 0 and "changed artifact" in failed.stderr
