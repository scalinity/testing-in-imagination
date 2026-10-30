"""Private run manifests, resumable receipts, and aggregate-only export checks."""

from contextlib import contextmanager
from datetime import datetime
import hashlib
import fcntl
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import socket
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
    files = [p for p in OWN.rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".slurm", ".toml", ".lock", ".json")
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
        report = command(["nvidia-smi"])
        match = re.search(r"CUDA Version:\s*([\d.]+)", report)
        cuda = match.group(1) if match else "UNREPORTED"
    except (OSError, RuntimeError):
        gpu = "NOT_AVAILABLE"
        cuda = "NOT_AVAILABLE"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "gpu_driver": gpu, "cuda_driver_report": cuda,
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "UNSET")}


def make_manifest(settings, stage: str, cfg, inputs: dict, dry_run=False) -> dict:
    from .gpu import COMPATIBILITY_DEVIATIONS
    upstream = command(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"])
    if upstream != UPSTREAM_SHA:
        raise RuntimeError("upstream checkout does not match the required pin")
    if command(["git", "-C", str(UPSTREAM), "status", "--porcelain"]):
        raise RuntimeError("upstream has local changes; restore its exact pin before running")
    value = {"schema_version": 1, "stage": stage, "source": settings.source,
             "label": LABEL, "execution": "mock" if settings.mock else "cuda",
             "git_sha": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
             "git_dirty": bool(command(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(OWN.relative_to(ROOT))])),
             "source_fingerprint": source_fingerprint(), "upstream_sha": upstream,
             "model": MODEL, "model_revision": REVISION, "split_seed": 269,
             "training_seeds": list(settings.seeds), "rng_scheme": "sha256(seed,split,stage)",
             "config": OmegaConf.to_container(cfg, resolve=True),
             "overrides": overrides_against_defaults(cfg), "runtime": runtime_info(),
             "inputs": inputs, "dry_run": dry_run, "wandb_run_ids": [],
             "bootstrap_resamples": settings.bootstrap_resamples}
    value["local_compatibility_deviations"] = COMPATIBILITY_DEVIATIONS + [
        "Empty physical-examination strings become the upstream not-available sentinel before CSV parsing.",
        "The absent xgrammar 0.1.16 package release is sourced from its publisher's immutable same-version commit.",
    ]
    # Observations document the launch, while immutable code/config/lock/input
    # identities determine resume. S0 deliberately transitions from bootstrap
    # Python to the locked CUDA environment; its observed packages will differ.
    value["manifest_id"] = digest({key: item for key, item in value.items() if key not in ("runtime", "git_sha", "git_dirty", "wandb_run_ids")})
    return value


@contextmanager
def stage_lock(work_dir: Path, stage: str):
    path = work_dir / "locks" / f"{stage}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"stage is already running; inspect the private lock: {path}") from None
        os.ftruncate(fd, 0)
        os.write(fd, canonical({"pid": os.getpid(), "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"), "started_at": eastern_now()}).encode())
        yield
    finally:
        # Keep the inode stable; kernel ownership is released even after SIGKILL.
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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
    payload["written_at"] = written_at if written_at is not None else eastern_now()
    write_json(path, payload)
    return path


def eastern_now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


FORBIDDEN_KEYS = {"patient_id", "patient id", "hadm_id", "subject_id", "patient history",
                  "patient_history", "patient history summary", "laboratory tests", "radiology",
                  "physical examination", "physical_examination", "test_results", "generations", "completions", "prompts", "rows"}


def assert_aggregate(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"patient-level field is forbidden in exports: {key}")
            assert_aggregate(item)
    elif isinstance(value, list):
        for item in value:
            assert_aggregate(item)


def tracked_data_violations(root=ROOT, own=OWN) -> list[str]:
    root, own = Path(root), Path(own)
    output = command(["git", "-C", str(root), "ls-files", "-z", "--", str(own.relative_to(root))])
    violations = []
    for relative in output.split("\0"):
        if not relative:
            continue
        path = root / relative
        suffix = path.suffix.lower()
        reason = None
        if suffix in (".pkl", ".pickle", ".parquet", ".jsonl", ".safetensors", ".pt", ".pth", ".bin"):
            reason = "data or weights suffix"
        elif path.name.startswith(".env") and path.name != ".env.example":
            reason = "private configuration"
        # Inspect the index as well as the working file: staged data must not escape
        # detection merely because the working copy was replaced with an empty file.
        indexed = subprocess.run(["git", "-C", str(root), "show", f":{relative}"], capture_output=True, check=False)
        contents = [indexed.stdout] if indexed.returncode == 0 else []
        if path.is_file():
            contents.append(path.read_bytes())
        for content in contents:
            if content[:4] == b"PAR1" or content[:2] in (b"\x80\x04", b"\x80\x05"):
                reason = "data format signature"
            if suffix in (".csv", ".tsv") and len(content) > 4096:
                reason = "CSV exceeds 4 KiB"
            text = content.decode("utf-8", errors="replace")
            first = text.splitlines()[0].lower() if text.splitlines() else ""
            if suffix not in (".py", ".md", ".toml", ".lock", ".sh", ".slurm") and any(key in first for key in FORBIDDEN_KEYS) and ("," in first or "\t" in first):
                reason = "patient columns"
            if suffix == ".json":
                try:
                    assert_aggregate(json.loads(text))
                except ValueError:
                    reason = "patient-level JSON"
        if reason:
            violations.append(f"{relative}: {reason}")
    return violations
