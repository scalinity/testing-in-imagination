import json
from pathlib import Path
import subprocess

import pytest

from replication.config import Settings, compose
from replication.manifest import (assert_aggregate, digest, make_manifest, stamp_metrics,
                                  tracked_data_violations)


def test_manifest_records_identity_runtime_composed_config_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V1", "0")
    settings = Settings("synthetic", tmp_path / "raw", tmp_path / "work")
    first = make_manifest(settings, "s2", compose(), {"example": "input"})
    second = make_manifest(settings, "s2", compose(), {"example": "input"})
    assert first == second
    required = {"git_sha", "upstream_sha", "model_revision", "source", "config", "overrides", "runtime", "training_seeds", "split_seed", "wandb_run_ids", "manifest_id", "source_fingerprint", "bootstrap_resamples"}
    assert required <= first.keys()
    assert first["runtime"]["VLLM_USE_V1"] == "0"
    assert first["runtime"]["python"].startswith(("3.11", "3.12"))
    assert "packages" in first["runtime"] and "gpu_driver" in first["runtime"]
    assert any(item["key"] == "model.model_name" for item in first["overrides"])


def test_metrics_reuse_shared_writer_without_mutating_it(tmp_path):
    from lacdm import metric_writer
    before = metric_writer.SOURCES
    path = stamp_metrics("smoke", {"accuracy": 0.25}, "synthetic", tmp_path, written_at="fixed")
    payload = json.loads(path.read_text())
    assert payload["source"] == "synthetic"
    assert payload["label"] == "re-implementation on a different backbone"
    assert payload["written_at"] == "fixed"
    assert metric_writer.SOURCES == before
    assert "upstream_sha" in payload
    with pytest.raises(ValueError):
        stamp_metrics("invalid", {}, "fixture", tmp_path)


@pytest.mark.parametrize("value", [{"patient_id": 1}, {"nested": [{"subject_id": 2}]}, {"patient_history": "invented"}, {"Physical Examination": "invented clinical narrative"}, {"rows": []}])
def test_aggregate_export_rejects_patient_fields(value):
    with pytest.raises(ValueError):
        assert_aggregate(value)


def test_no_tracked_data():
    assert tracked_data_violations() == []


def test_staged_patient_rows_and_renamed_binary_data_are_detected(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    own = tmp_path / "replication"
    own.mkdir()
    path = own / "innocent.txt"
    path.write_text("hadm_id,subject_id,patient_history\n1,2,invented\n")
    binary = own / "renamed.dat"
    binary.write_bytes(b"PAR1" + b"\0" * 20)
    subprocess.run(["git", "-C", str(tmp_path), "add", "replication"], check=True)
    path.write_text("empty working copy")
    violations = tracked_data_violations(root=tmp_path, own=own)
    assert len(violations) == 2
    assert any("patient columns" in item for item in violations)
    assert any("data format signature" in item for item in violations)


def test_runtime_observations_do_not_change_locked_run_identity(tmp_path, monkeypatch):
    import replication.manifest as manifests
    settings = Settings("synthetic", tmp_path / "raw", tmp_path / "work")
    monkeypatch.setattr(manifests, "runtime_info", lambda: {"packages": {"torch": "NOT_INSTALLED"}, "gpu": "initial observation"})
    first = make_manifest(settings, "s0", compose(), {})
    monkeypatch.setattr(manifests, "runtime_info", lambda: {"packages": {"torch": "2.6.0"}, "gpu": "different utilization"})
    second = make_manifest(settings, "s0", compose(), {})
    assert first["manifest_id"] == second["manifest_id"]
    assert first["runtime"] != second["runtime"]


def test_kernel_lock_recovers_after_process_termination(tmp_path):
    import os
    import sys
    from replication.manifest import stage_lock
    code = "from pathlib import Path; from replication.manifest import stage_lock; import sys,time\nwith stage_lock(Path(sys.argv[1]), 's4'):\n print('locked', flush=True)\n time.sleep(30)"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="already running"):
            with stage_lock(tmp_path, "s4"):
                pass
        process.kill()
        process.wait(timeout=10)
        with stage_lock(tmp_path, "s4"):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
