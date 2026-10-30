import json
from pathlib import Path

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


@pytest.mark.parametrize("value", [{"patient_id": 1}, {"nested": [{"subject_id": 2}]}, {"patient_history": "invented"}, {"rows": []}])
def test_aggregate_export_rejects_patient_fields(value):
    with pytest.raises(ValueError):
        assert_aggregate(value)


def test_no_tracked_data():
    assert tracked_data_violations() == []
