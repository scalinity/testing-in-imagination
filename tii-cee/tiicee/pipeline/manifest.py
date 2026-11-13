"""Immutable stage manifests (SPEC_RESEARCH_ADVANCEMENT.md sections 8.3 and 19).

A manifest is written once, before a stage's first submission, and verified by the stage on every
start and every resume. Verification recomputes every digest; a mismatch is a refusal (exit 20), a
failed authorization check is a refusal (exit 30), and neither is ever retried.

    python -m tiicee.pipeline.manifest build <stage_id>
    python -m tiicee.pipeline.manifest verify <stage_id>

Standard library only and Python 3.9 compatible: the watcher builds manifests under 3.9 and the
stages verify them under 3.12.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import authorization
from . import graph as graphs
from .paths import Paths
from .util import digest, now, read_json, sha256_file, tree_digest, write_new_json

IDENTITY = 20       # code, card, artifact or input changed
AUTHORIZATION = 30  # authorization or data-use refusal

CODE_SUFFIXES = (".py", ".json", ".sh", ".toml", ".lock")
NOT_STAGE_CODE = ("tests", "local")  # top-level folders of tii-cee/ that no stage runs on the cluster


class Refusal(Exception):
    """A stage must not run. code and role are names, never content."""

    def __init__(self, exit_code, code, role=None):
        super().__init__(code if role is None else "%s:%s" % (code, role))
        self.exit_code, self.code, self.role = exit_code, code, role


def _is_code(relative):
    parts = relative.split("/")
    return (relative.endswith(CODE_SUFFIXES) and parts[0] not in NOT_STAGE_CODE
            and not any(part.startswith(".") or part == "__pycache__" for part in parts))


def _git(code_dir, *args):
    try:
        result = subprocess.run(["git", "-C", str(code_dir)] + list(args),
                                capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def snapshot(paths):
    return tree_digest(paths.package, keep=_is_code)[0]


def code_identity(paths):
    """The commit when tii-cee/ is clean; otherwise (or without git) a digest of its code files."""
    commit = _git(paths.code, "rev-parse", "HEAD")
    # --no-optional-locks: many jobs start at once in one checkout and must not contend for its index.
    dirty = _git(paths.code, "--no-optional-locks", "status", "--porcelain", "--", "tii-cee") if commit else None
    if commit and dirty == "":
        return {"commit": commit, "clean": True, "snapshot": None}
    return {"commit": commit, "clean": False, "snapshot": snapshot(paths)}


def resolve(paths, spec):
    if spec.startswith("$TII/"):
        return paths.root / spec[len("$TII/"):]
    path = Path(spec)
    return path if path.is_absolute() else paths.code / path


def _bound(paths, role, spec):
    """A file's SHA-256, or for a directory the tree digest of sorted relative paths and file hashes."""
    path = resolve(paths, spec)
    if path.is_dir():
        value, files = tree_digest(path)
        return {"path": spec, "type": "tree", "sha256": value, "files": files}
    try:
        return {"path": spec, "type": "file", "sha256": sha256_file(path)}
    except OSError:
        raise Refusal(IDENTITY, "bound_missing", role)


def _described(graph, stage):
    """The part of a manifest that restates the stage as graph.json defines it."""
    allowance = graph.allowances[stage["allowance_key"]]
    return {
        "study": graph.study, "experiment": stage["experiment"], "stage": stage["id"],
        "kind": stage["kind"], "intervention": stage["intervention"],
        "seeds": stage["seeds"], "pairing_unit": stage["pairing_unit"],
        "allowance": {"key": stage["allowance_key"], "gpu_hours": allowance["allowance_gpu_hours"],
                      "estimate_gpu_hours": stage["estimate_gpu_hours"]},
        "command": stage["command"],
    }


def _id(manifest):
    return digest({k: v for k, v in manifest.items() if k not in ("manifest_id", "created")})


def build(paths, graph, stage):
    code = authorization.check(paths.study_dir)
    if code != "ok":
        raise Refusal(AUTHORIZATION, code)
    manifest = dict(_described(graph, stage), schema=1, code=code_identity(paths))
    manifest["bound"] = {role: _bound(paths, role, spec) for role, spec in sorted(stage["bind"].items())}
    manifest["authorization"] = {"sha256": authorization.AUTHORIZATION_SHA256,
                                 "expires": authorization.EXPIRES.isoformat(),
                                 "addenda": authorization.addenda(paths.study_dir)}
    manifest["manifest_id"] = _id(manifest)
    manifest["created"] = now()
    return manifest


def ensure(paths, graph, stage):
    """The stage's manifest: the one already bound, or a new one written exactly once."""
    existing = read_json(paths.manifest(stage["id"]))
    if existing is not None:
        return existing
    manifest = build(paths, graph, stage)
    try:
        write_new_json(paths.manifest(stage["id"]), manifest)
    except FileExistsError:
        return read_json(paths.manifest(stage["id"]))
    return manifest


def verify(paths, graph, stage):
    """Return the manifest if everything it binds is unchanged; otherwise raise Refusal."""
    code = authorization.check(paths.study_dir)
    if code != "ok":
        raise Refusal(AUTHORIZATION, code)
    manifest = read_json(paths.manifest(stage["id"]))
    if manifest is None:
        raise Refusal(IDENTITY, "manifest_missing")
    if _id(manifest) != manifest.get("manifest_id"):
        raise Refusal(IDENTITY, "manifest_altered")
    if manifest["authorization"]["sha256"] != authorization.AUTHORIZATION_SHA256:
        raise Refusal(IDENTITY, "authorization_digest")
    for field, value in _described(graph, stage).items():
        if manifest.get(field) != value:
            raise Refusal(IDENTITY, "stage_changed", field)
    if {role: entry["path"] for role, entry in manifest["bound"].items()} != stage["bind"]:
        raise Refusal(IDENTITY, "stage_changed", "bind")

    bound_code = manifest["code"]
    if bound_code["clean"]:
        current = code_identity(paths)
        if not current["clean"] or current["commit"] != bound_code["commit"]:
            raise Refusal(IDENTITY, "code_changed")
    elif snapshot(paths) != bound_code["snapshot"]:
        raise Refusal(IDENTITY, "code_changed")

    for role, entry in sorted(manifest["bound"].items()):
        current = _bound(paths, role, entry["path"])
        if current["sha256"] != entry["sha256"] or current["type"] != entry["type"]:
            raise Refusal(IDENTITY, "bound_changed", role)
    return manifest


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("build", "verify"):
        print("usage: python -m tiicee.pipeline.manifest build|verify <stage_id>", file=sys.stderr)
        return 2
    paths = Paths()
    graph = graphs.load(paths)
    stage = graph.by_id.get(argv[1])
    if stage is None:
        print("unknown stage", file=sys.stderr)
        return 2
    try:
        manifest = ensure(paths, graph, stage) if argv[0] == "build" else verify(paths, graph, stage)
    except Refusal as refusal:
        print("refused %d %s" % (refusal.exit_code, refusal))
        return refusal.exit_code
    print(manifest["manifest_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
