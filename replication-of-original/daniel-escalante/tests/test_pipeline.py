import json
import os
from pathlib import Path
import subprocess
import sys
import shlex

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


def test_selected_weights_cannot_change_after_validation_selection(tmp_path, monkeypatch):
    import replication.pipeline as pipeline
    import replication.gpu as gpu
    settings = Settings("mimic", tmp_path / "raw", tmp_path / "work")
    monkeypatch.setattr(pipeline, "_require_cuda", lambda: None)
    def train(cfg, output_dir, data_dir, seed):
        adapter = output_dir / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"invented weight fixture")
        return {"adapter_dir": str(adapter), "wandb_run_ids": []}
    monkeypatch.setattr(gpu, "train", train)
    s4(settings)
    weights = settings.work_dir / "models/parity/269/primary/adapter/adapter_model.safetensors"
    weights.write_bytes(b"changed fixture")
    with pytest.raises(RuntimeError, match="selected model"):
        pipeline.s5(settings)
    with pytest.raises(RuntimeError, match="selected model"):
        s4(settings)


def test_shell_honors_env_file_before_selecting_runtime(tmp_path):
    work = tmp_path / "mimic work"
    runtime = work / "runtime-env/bin/python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\necho SELECTED_PRIVATE_RUNTIME\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    runtime.chmod(0o755)
    env_file = tmp_path / "mimic.env"
    env_file.write_text(f'CDM_DATA_SOURCE=mimic\nCDM_DATA_DIR="{tmp_path / "raw"}"\nCDM_WORK_DIR="{work}"\n')
    env = {key: value for key, value in os.environ.items() if not key.startswith("CDM_")}
    env["CDM_PYTHON"] = sys.executable
    result = subprocess.run(["bash", str(OWN / "scripts/stage.sh"), "s2", "--dry-run", "--env-file", str(env_file)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SELECTED_PRIVATE_RUNTIME" in result.stdout
    assert json.loads((work / "manifests/dry-run-s2.json").read_text())["source"] == "mimic"
def test_full_synthetic_shell_run_and_deterministic_resume(tmp_path, monkeypatch):
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
    aggregate = json.loads((work / "aggregates.json").read_text())["payload"]
    assert set(aggregate["protocols"]) == {"parity", "grouped"}
    primary = aggregate["protocols"]["parity"]["seeds"]["269"]["arms"]["primary"]
    assert set(primary["summary"]["claims"]) == {f"T{i}" for i in range(1, 9)}
    assert all(value is not None for value in primary["summary"]["claims"].values())
    assert primary["intervals"]["n_resamples"] == 10000
    import replication.pipeline as pipeline
    monkeypatch.setattr(pipeline, "OWN", tmp_path / "public")
    pipeline.export_aggregates(Settings("synthetic", tmp_path / "raw", work, synthetic_n=120))
    exported = json.loads((tmp_path / "public/results/synthetic/metrics.json").read_text())
    assert exported["source"] == "synthetic" and "provenance" in exported["metrics"]
    before = {str(p.relative_to(work)): file_digest(p) for p in work.rglob("*") if p.is_file() and "locks" not in p.relative_to(work).parts}
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert again.returncode == 0, again.stdout + again.stderr
    assert again.stdout.count("RESUMED") == 7
    after = {str(p.relative_to(work)): file_digest(p) for p in work.rglob("*") if p.is_file() and "locks" not in p.relative_to(work).parts}
    assert before == after
    manifest = work / "manifests/s0.json"
    original_manifest = manifest.read_bytes()
    changed = json.loads(original_manifest)
    changed["git_sha"] = "f" * 40
    manifest.write_text(json.dumps(changed))
    provenance = subprocess.run(["bash", str(OWN / "scripts/stage.sh"), "s6"], env=env, capture_output=True, text=True, timeout=60)
    assert provenance.returncode != 0 and "changed artifact" in provenance.stderr
    manifest.write_bytes(original_manifest)
    # Direct S6 admission must validate the earlier baseline's owning receipt.
    baseline = work / "episodes/parity/269/zero_shot-test.json"
    original = baseline.read_bytes()
    baseline.write_text("corrupted")
    direct = subprocess.run(["bash", str(OWN / "scripts/stage.sh"), "s6"], env=env, capture_output=True, text=True, timeout=60)
    assert direct.returncode != 0 and "changed artifact" in direct.stderr
    baseline.write_bytes(original)
    (work / "prepared/test.csv").write_text("corrupted")
    failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert failed.returncode != 0 and "changed artifact" in failed.stderr


def test_optional_seeds_have_actual_between_seed_intervals(tmp_path):
    # Small resample count only for this structure test; Settings.load fixes real runs at 10,000.
    settings = Settings("synthetic", tmp_path / "raw", tmp_path / "work", seeds=(269, 270), synthetic_n=120, bootstrap_resamples=20)
    for stage in STAGES:
        run_stage(settings, stage)
    result = json.loads((settings.work_dir / "aggregates.json").read_text())["payload"]
    for protocol in result["protocols"].values():
        assert set(protocol["seeds"]) == {"269", "270"}
        intervals = protocol["seed_differences"]
        assert intervals and all(item["declared_n"] == 2 and item["ci95"] is not None for item in intervals.values())
