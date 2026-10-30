"""Private run manifests, resumable receipts, and aggregate-only export checks."""

from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

from omegaconf import OmegaConf

from .config import LABEL, MODEL, OWN, REVISION, ROOT, UPSTREAM, UPSTREAM_SHA, overrides_against_defaults


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)
    return path


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed: {args[0]}")
    return result.stdout.strip()


def source_fingerprint() -> str:
    files = [p for p in OWN.rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".slurm", ".toml", ".lock", ".json", ".md")
             and not any(part.startswith(".") or part in ("__pycache__", "results") for part in p.relative_to(OWN).parts)]
    files.extend([OWN / "results/paper_values.json", OWN / "results/REPORT_TEMPLATE.md"])
    return digest({str(p.relative_to(OWN)): file_digest(p) for p in sorted(set(files)) if p.exists()})


def tree_fingerprint(directory: Path, patterns=("*.csv", "*.pkl")) -> dict:
    if not directory.exists():
        return {}
    return {str(p.relative_to(directory)): file_digest(p)
            for p in sorted({p for pattern in patterns for p in directory.rglob(pattern)}) if p.is_file()}


def runtime_info() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "hydra-core", "torch", "transformers", "trl", "vllm", "accelerate", "peft", "wandb"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    try:
        gpu = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        cuda = command(["nvidia-smi"])
    except (OSError, RuntimeError):
        gpu = "NOT_AVAILABLE"
        cuda = "NOT_AVAILABLE"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "gpu_driver": gpu, "cuda_driver_report": cuda,
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "UNSET")}


def make_manifest(settings, stage: str, cfg, inputs: dict, dry_run=False) -> dict:
    upstream = command(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"])
    if upstream != UPSTREAM_SHA:
        raise RuntimeError("upstream checkout does not match the required pin")
    if command(["git", "-C", str(UPSTREAM), "status", "--porcelain"]):
        raise RuntimeError("upstream has local changes; restore its exact pin before running")
    value = {"schema_version": 1, "stage": stage, "source": settings.source,
             "label": LABEL, "execution": "mock" if settings.mock else "cuda",
             "git_sha": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
             "source_fingerprint": source_fingerprint(), "upstream_sha": upstream,
             "model": MODEL, "model_revision": REVISION, "split_seed": 269,
             "training_seeds": list(settings.seeds), "rng_scheme": "sha256(seed,split,stage)",
             "config": OmegaConf.to_container(cfg, resolve=True),
             "overrides": overrides_against_defaults(cfg), "runtime": runtime_info(),
             "inputs": inputs, "dry_run": dry_run, "wandb_run_ids": [],
             "bootstrap_resamples": settings.bootstrap_resamples}
    value["manifest_id"] = digest(value)
    return value


@contextmanager
def stage_lock(work_dir: Path, stage: str):
    path = work_dir / "locks" / f"{stage}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"stage is already running or interrupted; inspect the private lock: {path}") from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def stamp_metrics(name: str, metrics: dict, source: str, out_dir: Path, written_at=None):
    if source not in ("synthetic", "mimic"):
        raise ValueError("metrics require an explicit synthetic or mimic source")
    # A private module instance extends the source vocabulary without changing shared code.
    spec = importlib.util.spec_from_file_location("_replication_metric_writer", ROOT / "lacdm/metric_writer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCES = ("synthetic", "mimic")
    path = module.write_metrics(name, metrics, source, out_dir=out_dir)
    payload = json.loads(path.read_text())
    payload["label"] = LABEL
    if written_at is not None:
        payload["written_at"] = written_at
    write_json(path, payload)
    return path


def eastern_now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


FORBIDDEN_KEYS = {"patient_id", "patient id", "hadm_id", "subject_id", "patient history",
                  "patient_history", "patient history summary", "laboratory tests", "radiology",
                  "test_results", "generations", "completions", "prompts", "rows"}


def assert_aggregate(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"patient-level field is forbidden in exports: {key}")
            assert_aggregate(item)
    elif isinstance(value, list):
        for item in value:
            assert_aggregate(item)


def tracked_data_violations() -> list[str]:
    output = command(["git", "-C", str(ROOT), "ls-files", "-z", "--", str(OWN.relative_to(ROOT))])
    violations = []
    for relative in output.split("\0"):
        if not relative:
            continue
        path = ROOT / relative
        suffix = path.suffix.lower()
        reason = None
        if suffix in (".pkl", ".pickle", ".parquet", ".jsonl", ".safetensors", ".pt", ".pth", ".bin"):
            reason = "data or weights suffix"
        elif path.name.startswith(".env") and path.name != ".env.example":
            reason = "private configuration"
        elif suffix == ".csv":
            if path.stat().st_size > 4096:
                reason = "CSV exceeds 4 KiB"
            elif any(key in path.read_text().splitlines()[0].lower() for key in FORBIDDEN_KEYS):
                reason = "patient columns"
        elif suffix == ".json":
            try:
                assert_aggregate(json.loads(path.read_text()))
            except ValueError:
                reason = "patient-level JSON"
        if reason:
            violations.append(f"{relative}: {reason}")
    return violations
