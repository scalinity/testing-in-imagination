"""Cluster side of a code deployment: a git bundle becomes an immutable, per-commit code export.

Nothing is ever pushed to GitHub. tii-cee/local/deploy_code.sh on the Mac bundles the study branch
and copies the bundle to $TII/deploy/ through the existing SSH master, then runs this file there:

    python3 deploy.py --bundle $TII/deploy/<commit>.bundle [--sha256 HEX]
                      [--root $TII] [--checkout DIR] [--branch NAME]

1. Fetch the bundle into the cluster checkout as the branch of the same name. Only that one ref
   moves, fast-forward only. The checkout's working tree, its other branches and its worktrees are
   left alone; if the branch is checked out anywhere, nothing is done.
2. `git archive <commit>` into $TII/code/<commit>/, then make it read-only.
3. A submodule is not in a bundle. For each one the commit pins (lacdm/upstream), the checkout's
   own copy must sit at exactly that commit with no local changes; the export links to it.
4. Point $TII/code/current at the new export, atomically.

A stage runs from the export recorded in its manifest for its whole life, so a later deployment
never changes the code of work in flight; new stages bind `current`. This file never deletes an
export. If exports are ever pruned by hand, one that a stage which is not final still names in
state/manifests/ must stay.

It prints one JSON line (names and counts only), or `refused <code>` and exits 1. Standard library
only, Python 3.9, and no imports from this package, so it also runs as a copied single file
before the package exists on the cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DEFAULT_ROOT = "/blue/cai6734/daniel.escalante/private/tii-cee-1"       # as paths.DEFAULT_ROOT
DEFAULT_CHECKOUT = "/blue/cai6734/daniel.escalante/testing-in-imagination"
BRANCH = "tii-cee-1/2026-11-13"                                         # AUTHORIZATION.md section 1
META = ".tii-export.json"


class Refused(Exception):
    """code is a name; it never carries a path's contents."""


def git(directory, *args, check=True):
    """Run git in a directory and return its output. A failure raises Refused("git_failed"), or
    returns None when the caller asked to decide for itself (check=False)."""
    try:
        result = subprocess.run(["git", "-C", str(directory)] + list(args), capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        raise Refused("git_failed")
    if result.returncode != 0:
        if check:
            raise Refused("git_failed")
        return None
    return result.stdout.strip()


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch(checkout, bundle, branch):
    """Bring the bundle's branch into the checkout as that branch. Returns its commit."""
    ref = "refs/heads/" + branch
    if not (checkout / ".git").exists():
        raise Refused("checkout_missing")
    if git(checkout, "bundle", "verify", str(bundle), check=False) is None:
        raise Refused("bundle_invalid")
    offered = {}  # what the bundle carries: ref name -> commit
    for line in git(checkout, "bundle", "list-heads", str(bundle)).splitlines():
        commit, name = line.split(" ", 1)
        offered[name] = commit
    if ref not in offered:
        raise Refused("branch_not_in_bundle")
    # git would refuse to move a branch that some working tree has checked out; say it plainly first.
    if "branch " + ref in git(checkout, "worktree", "list", "--porcelain").splitlines():
        raise Refused("branch_checked_out")
    before = git(checkout, "rev-parse", "--verify", "--quiet", ref, check=False)  # None: not there yet
    # One explicit refspec and no "+": only this ref can move, and only forward. No garbage
    # collection in the shared checkout, no tags, and no visit to the submodule's GitHub remote.
    fetched = git(checkout, "-c", "gc.auto=0", "fetch", "--no-tags", "--no-recurse-submodules",
                  str(bundle), "%s:%s" % (ref, ref), check=False)
    if fetched is None:
        # `bundle verify` reads only the header, so a bundle cut short fails here. If the commit
        # did arrive and the branch on the cluster is not its ancestor, git refused to rewrite it.
        arrived = git(checkout, "cat-file", "-e", offered[ref] + "^{commit}", check=False) is not None
        diverged = arrived and before is not None and git(
            checkout, "merge-base", "--is-ancestor", before, offered[ref], check=False) is None
        raise Refused("not_fast_forward" if diverged else "bundle_invalid")
    if git(checkout, "rev-parse", "--verify", ref) != offered[ref]:
        raise Refused("git_failed")
    return offered[ref]


def pinned_submodules(checkout, commit):
    """Each submodule the commit pins, checked against the checkout's own copy of it."""
    found = []
    for line in git(checkout, "ls-tree", "-r", commit).splitlines():
        mode, kind, rest = line.split(" ", 2)
        if mode != "160000":
            continue
        pin, path = rest.split("\t", 1)
        source = checkout / path
        if git(source, "rev-parse", "--show-toplevel", check=False) != str(source.resolve()):
            raise Refused("submodule_missing")
        if git(source, "rev-parse", "HEAD") != pin:
            raise Refused("submodule_not_at_pin")
        if git(source, "--no-optional-locks", "status", "--porcelain"):
            raise Refused("submodule_not_clean")
        found.append({"path": path, "commit": pin, "source": str(source)})
    return found


def make_read_only(export):
    """Clear every write bit under the export. Links (to a submodule's checkout) are not followed."""
    for base, dirs, files in os.walk(export, followlinks=False):
        for name in dirs + files:
            path = os.path.join(base, name)
            if not os.path.islink(path):
                os.chmod(path, os.stat(path).st_mode & ~0o222)
    os.chmod(export, os.stat(export).st_mode & ~0o222)


def export_commit(root, checkout, commit, branch, submodules):
    """Write $TII/code/<commit>/ once. Returns (path, number of files)."""
    code = root / "code"
    export = code / commit
    if not export.exists():
        partial = code / (".%s.partial.%d" % (commit, os.getpid()))
        partial.mkdir(parents=True)
        try:
            archive = subprocess.Popen(["git", "-C", str(checkout), "archive", "--format=tar", commit],
                                       stdout=subprocess.PIPE)
            with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
                # Python 3.9.17 and later can refuse members that would land outside the folder.
                tar.extractall(partial, **({"filter": "tar"} if hasattr(tarfile, "tar_filter") else {}))
            if archive.wait() != 0:
                raise Refused("archive_failed")
            for submodule in submodules:  # the archive leaves an empty folder where a submodule goes
                place = partial / submodule["path"]
                if place.is_dir():
                    place.rmdir()
                place.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(submodule["source"], place)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            (partial / META).write_text(json.dumps({
                "commit": commit, "branch": branch, "exported_at": stamp[:-2] + ":" + stamp[-2:],
                "submodules": submodules}, sort_keys=True, indent=2) + "\n")
            os.rename(partial, export)
        except (OSError, tarfile.TarError, Refused) as error:
            shutil.rmtree(partial, ignore_errors=True)
            if not export.exists():  # otherwise another deployment of the same commit won the rename
                raise error if isinstance(error, Refused) else Refused("export_failed")
    make_read_only(export)  # every time, so a deployment cut short here is finished by the next
    return export, sum(len(files) for _, _, files in os.walk(export, followlinks=False))


def point_current(root, commit):
    """Swap $TII/code/current to the export in one rename."""
    link = root / "code" / "current"
    fresh = link.with_name(".current.%d" % os.getpid())
    if os.path.lexists(fresh):
        os.unlink(fresh)
    os.symlink(commit, fresh)
    os.replace(fresh, link)
    return link


def deploy(root, checkout, bundle, branch=BRANCH, sha256=None):
    root, checkout, bundle = Path(root), Path(checkout), Path(bundle)
    if not bundle.is_file():
        raise Refused("bundle_missing")
    if sha256 is not None and sha256_file(bundle) != sha256:
        raise Refused("bundle_digest")
    commit = fetch(checkout, bundle, branch)
    submodules = pinned_submodules(checkout, commit)
    export, files = export_commit(root, checkout, commit, branch, submodules)
    link = point_current(root, commit)
    return {"commit": commit, "branch": branch, "export": str(export), "current": str(link),
            "files": files, "submodules": len(submodules)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Turn a git bundle into a read-only code export.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--root", default=os.environ.get("TII_ROOT") or DEFAULT_ROOT)
    parser.add_argument("--checkout", default=os.environ.get("TII_CHECKOUT") or DEFAULT_CHECKOUT)
    parser.add_argument("--branch", default=BRANCH)
    options = parser.parse_args(argv)
    try:
        result = deploy(options.root, options.checkout, options.bundle, options.branch, options.sha256)
    except Refused as refused:
        print("refused %s" % refused)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
