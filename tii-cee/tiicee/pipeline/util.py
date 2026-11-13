"""Digests and atomic file writes shared by the watcher and the stages.

Standard library only and Python 3.9 compatible. A digest computed by the watcher (3.9) and
recomputed by a stage (3.12) must be the same bytes, so everything here is deterministic.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def iso(epoch=None):
    """Local time as ISO 8601 with offset, 2026-11-13T03:15:02-05:00 (monitor/DATA_CONTRACT.md)."""
    text = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch))
    return text[:-2] + ":" + text[-2:]


def now():
    return iso()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tree_digest(directory, keep=None):
    """Digest of a directory: sorted relative paths with each file's SHA-256. Returns (digest, files).

    keep(relative_posix_path) -> bool filters files; directory links are not followed.
    """
    directory = Path(directory)
    entries = []
    for base, dirs, files in os.walk(directory, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            path = Path(base) / name
            relative = path.relative_to(directory).as_posix()
            if keep is None or keep(relative):
                entries.append([relative, sha256_file(path)])
    entries.sort()
    return digest(entries), len(entries)


def write_text(path, text):
    """Write to a temporary name in the same directory, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    with open(temp, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return path


def write_json(path, value):
    return write_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_new_json(path, value):
    """Create a record that must never be rewritten. Raises FileExistsError if it already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(".%s.new.%d" % (path.name, os.getpid()))
    with open(temp, "w") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temp, path)  # fails if the record exists; a rename would silently replace it
    finally:
        os.unlink(temp)
    os.chmod(path, 0o444)
    return path


def read_json(path, default=None):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def append_line(path, line):
    """One write call on an append-only descriptor, so concurrent readers never see half a line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)
