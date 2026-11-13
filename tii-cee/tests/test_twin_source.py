"""The vendored snapshot of Edwin Salcedo's synthetic study is byte-exact, verified, and isolated."""
from __future__ import annotations

import hashlib
import shutil
import sys

import pytest

from tiicee.twin import source

# `git ls-tree -r df467eadd07fe2bc761b618dafce8ed9f198641a -- extension/envs/synthetic_diagnosis`
BLOBS_AT_COMMIT = {
    "extension/envs/synthetic_diagnosis/actions.py": "f59f0ddca6b86d7a3ed499bd8d34f9de449139d2",
    "extension/envs/synthetic_diagnosis/profiles.py": "d89fc8d7359b7999700fa9e5689225bb4a57770a",
    "extension/envs/synthetic_diagnosis/generators.py": "3d81a2ff893f7d72d268e3c1af75b1a22b6ddeb8",
}


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def test_vendored_files_are_the_bytes_of_the_commit():
    assert source.SNAPSHOT_COMMIT == "df467eadd07fe2bc761b618dafce8ed9f198641a"
    for path, blob in BLOBS_AT_COMMIT.items():
        assert git_blob_id((source.SNAPSHOT_DIR / path).read_bytes()) == blob, path


def test_sha256sums_lists_exactly_the_vendored_code():
    listed = {}
    for line in (source.SNAPSHOT_DIR / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        listed[path] = digest
    assert set(listed) == set(BLOBS_AT_COMMIT)
    for path, digest in listed.items():
        assert hashlib.sha256((source.SNAPSHOT_DIR / path).read_bytes()).hexdigest() == digest
    on_disk = {p.relative_to(source.SNAPSHOT_DIR).as_posix() for p in source.SNAPSHOT_DIR.rglob("*.py")}
    assert on_disk == set(listed)


def test_snapshot_digest_binds_the_checksum_file():
    expected = "sha256:" + hashlib.sha256((source.SNAPSHOT_DIR / "SHA256SUMS").read_bytes()).hexdigest()
    assert source.snapshot_digest() == expected
    assert source.snapshot_digest() == source.snapshot_digest()


def test_attribution_names_the_author_the_pull_request_and_the_commit():
    text = (source.SNAPSHOT_DIR / "ATTRIBUTION.md").read_text()
    assert "Edwin Salcedo" in text and "pull request #1" in text and source.SNAPSHOT_COMMIT in text
    assert "read-only snapshot" in text and "tiicee/twin" in text
    assert source.SNAPSHOT_AUTHOR == "Edwin Salcedo"


def test_a_modified_file_is_refused(tmp_path):
    copy = tmp_path / "snapshot"
    shutil.copytree(source.SNAPSHOT_DIR, copy)
    source.verify(copy)
    target = copy / "extension/envs/synthetic_diagnosis/profiles.py"
    target.write_bytes(target.read_bytes().replace(b"P_ATYPICAL = 0.07", b"P_ATYPICAL = 0.08"))
    with pytest.raises(source.SnapshotError, match="profiles.py"):
        source.verify(copy)


def test_a_missing_file_and_an_unlisted_file_are_refused(tmp_path):
    missing = tmp_path / "missing"
    shutil.copytree(source.SNAPSHOT_DIR, missing)
    (missing / "extension/envs/synthetic_diagnosis/actions.py").unlink()
    with pytest.raises(source.SnapshotError, match="actions.py"):
        source.verify(missing)

    extra = tmp_path / "extra"
    shutil.copytree(source.SNAPSHOT_DIR, extra)
    (extra / "extension/envs/synthetic_diagnosis/tuples.py").write_text("X = 1\n")
    with pytest.raises(source.SnapshotError, match="tuples.py"):
        source.verify(extra)


def test_snapshot_modules_load_from_the_vendored_path_under_a_private_name():
    for module in (source.actions, source.profiles, source.generators):
        assert pathlib_parent(module.__file__) == source.SNAPSHOT_DIR / "extension/envs/synthetic_diagnosis"
    # the adapter must not claim the top-level name of Edwin's real package
    assert "extension" not in sys.modules
    assert source.profiles.A_TEST is source.actions.A_TEST
    assert source.profiles.P_ATYPICAL == 0.07
    assert len(source.actions.A_TEST) == 12 and len(source.actions.A_DIAG) == 4


def pathlib_parent(path):
    import pathlib

    return pathlib.Path(path).resolve().parent
