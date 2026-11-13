"""The standing authorization: is it in force, and which Slurm accounts does it allow.

Mirrors reports/research-advancement/AUTHORIZATION.md (sections 1, 3 and 12) and
AUTHORIZATION_ADDENDUM_1.md (section 2). Those files are read and hashed here, never parsed and
never written. Standard library only and Python 3.9 compatible.
"""
from __future__ import annotations

import datetime
from collections import namedtuple
from pathlib import Path

from .util import sha256_file

AUTHORIZATION = "AUTHORIZATION.md"
SIGNATURE = "SIGNATURE.md"
AUTHORIZATION_SHA256 = "72ccc89c8a825c82578c8efdf942025e3ce60d72f179b8b13489737ee37221ae"
EXPIRES = datetime.date(2027, 3, 31)  # AUTHORIZATION.md section 1: valid through March 31, 2027

GPU_PARTITION = "hpg-rtx6000"
GPU_GRES = "gpu:rtx_pro_6000:1"

# kind is "gpu" or "cpu"; limit is GPUs or CPUs in use at once on that QOS.
Pool = namedtuple("Pool", "kind account qos limit")

# AUTHORIZATION.md section 3.
BASE_POOLS = (
    Pool("gpu", "cai6734", "cai6734", 8),
    Pool("cpu", "cai6734", "cai6734-b", 288),
)

# AUTHORIZATION_ADDENDUM_1.md section 2. These count only while that exact text is signed: the
# file's SHA-256 must equal the value below (the draft these pools were copied from, quoted in
# BLOCKED-001) and SIGNATURE.md must contain it. A reworded addendum needs this block updated.
ADDENDUM_1 = "AUTHORIZATION_ADDENDUM_1.md"
ADDENDUM_1_SHA256 = "f59cd6fa638d00b2d1bc9bdc4848ba9ae412f9abecb7cdfcfa83cb4475e05658"
ADDENDUM_1_POOLS = (
    Pool("gpu", "cai5724", "cai5724", 8),
    Pool("cpu", "cai5724", "cai5724-b", 288),
    Pool("cpu", "cai5720", "cai5720-b", 288),
    Pool("cpu", "cai5731", "cai5731-b", 576),
    Pool("cpu", "cai5720", "cai5720", 32),
    Pool("cpu", "cai5731", "cai5731", 64),
)


def _read(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def inspect(study_dir, today=None):
    """Each check by itself, and "code": "ok" or the first that failed (hash, signature, expiry)."""
    study_dir = Path(study_dir)
    try:
        found = sha256_file(study_dir / AUTHORIZATION)
    except OSError:
        found = None
    signature = _read(study_dir / SIGNATURE)
    sha256_ok = found == AUTHORIZATION_SHA256
    signature_ok = signature is not None and AUTHORIZATION_SHA256 in signature
    if found is None:
        code = "authorization_missing"
    elif not sha256_ok:
        code = "authorization_changed"
    elif signature is None:
        code = "signature_missing"
    elif not signature_ok:
        code = "signature_mismatch"
    elif (today or datetime.date.today()) > EXPIRES:
        code = "authorization_expired"
    else:
        code = "ok"
    return {"code": code, "sha256_ok": sha256_ok, "signature_ok": signature_ok}


def check(study_dir, today=None):
    """Return "ok" or the code of the first failed check."""
    return inspect(study_dir, today)["code"]


def addenda(study_dir):
    """Each addendum on disk: its digest, whether that is the text these pools mirror, and whether
    it counts now (SIGNATURE.md contains that digest)."""
    study_dir = Path(study_dir)
    try:
        found = sha256_file(study_dir / ADDENDUM_1)
    except OSError:
        return {}
    signature = _read(study_dir / SIGNATURE) or ""
    sha256_ok = found == ADDENDUM_1_SHA256
    return {ADDENDUM_1: {"sha256": found, "sha256_ok": sha256_ok, "signed": sha256_ok and found in signature}}


def pools(study_dir):
    """Usable (account, QOS) pairs in fill order: the class allocation first, then signed addenda."""
    if check(study_dir) != "ok":
        return ()
    signed = addenda(study_dir).get(ADDENDUM_1, {}).get("signed", False)
    return BASE_POOLS + (ADDENDUM_1_POOLS if signed else ())


def pool_key(pool):
    """What a pool's use is counted under: GPUs by account (the class limit is per account,
    whatever the partition or QOS a GPU job runs on); CPUs by account and QOS."""
    return pool.account if pool.kind == "gpu" else (pool.account, pool.qos)


def pick_pool(usable, kind, need, in_use):
    """First pool of this kind with room. in_use maps pool_key(pool) to GPUs or CPUs held now."""
    for pool in usable:
        if pool.kind == kind and in_use.get(pool_key(pool), 0) + need <= pool.limit:
            return pool
    return None
