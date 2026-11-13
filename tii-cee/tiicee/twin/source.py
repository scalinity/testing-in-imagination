"""Read-only access to the vendored snapshot of Edwin Salcedo's synthetic study (PR #1, commit df467ea).

The snapshot under vendor/synthetic_df467ea is byte-exact and is never modified; anything the study
changes lives in the adapters of this package. Importing this module verifies every vendored file
against SHA256SUMS and refuses a snapshot that does not match. snapshot_digest() is what result
manifests record as the source of the twin's parameters (REVIEW-011).

The snapshot's modules import each other relatively, so they are loaded as a private package whose
path is the vendored directory. The top-level name `extension` is never claimed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

SNAPSHOT_AUTHOR = "Edwin Salcedo"
SNAPSHOT_COMMIT = "df467eadd07fe2bc761b618dafce8ed9f198641a"
SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "vendor" / "synthetic_df467ea"

_PACKAGE = "_tiicee_snapshot_df467ea"
_MODULE_DIR = "extension/envs/synthetic_diagnosis"


class SnapshotError(RuntimeError):
    """The vendored snapshot is not the recorded one."""


def _listed(directory: Path) -> dict:
    listed = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        listed[path] = digest
    return listed


def verify(directory: Path = SNAPSHOT_DIR) -> None:
    """Every listed file has its recorded SHA-256, and no other Python file is present."""
    directory = Path(directory)
    listed = _listed(directory)
    for path, digest in listed.items():
        file = directory / path
        if not file.is_file():
            raise SnapshotError(f"vendored file missing: {path}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise SnapshotError(f"vendored file differs from SHA256SUMS: {path}")
    for file in sorted(directory.rglob("*.py")):
        path = file.relative_to(directory).as_posix()
        if path not in listed:
            raise SnapshotError(f"file not listed in SHA256SUMS: {path}")


def snapshot_digest(directory: Path = SNAPSHOT_DIR) -> str:
    """SHA-256 of SHA256SUMS, which names every vendored file and its digest."""
    return "sha256:" + hashlib.sha256((Path(directory) / "SHA256SUMS").read_bytes()).hexdigest()


def _load(name: str):
    qualified = f"{_PACKAGE}.{name}"
    if qualified not in sys.modules:
        spec = importlib.util.spec_from_file_location(qualified, SNAPSHOT_DIR / _MODULE_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # leave the snapshot directory as it is
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = write_bytecode
    return sys.modules[qualified]


verify()
if _PACKAGE not in sys.modules:
    _package = importlib.util.module_from_spec(importlib.util.spec_from_loader(_PACKAGE, loader=None, is_package=True))
    _package.__path__ = [str(SNAPSHOT_DIR / _MODULE_DIR)]
    sys.modules[_PACKAGE] = _package

actions = _load("actions")
profiles = _load("profiles")
generators = _load("generators")
