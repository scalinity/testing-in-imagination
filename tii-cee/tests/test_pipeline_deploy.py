"""Code deployment without GitHub: a git bundle becomes a read-only export per commit.

Everything here is under the pytest temp directory: a "Mac" repository holding the study branch,
and a "cluster" with its own checkout, its pinned upstream checkout and the study's storage root.
The remote is fake: deploy.py is run on local folders, and deploy_code.sh talks to a stub ssh that
runs the "remote" command locally. Set TII_TEST_WATCHER_PYTHON to a Python 3.9 to run deploy.py
and the watcher on the interpreter they meet on the cluster.
"""
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fake_slurm"))
from pipeline_harness import PACKAGE, SLOW, WATCHER_PYTHON, Study, pin, stage  # noqa: E402

from tiicee.pipeline import deploy  # noqa: E402
from tiicee.pipeline import graph as graphs  # noqa: E402
from tiicee.pipeline import manifest as manifests  # noqa: E402
from tiicee.pipeline.manifest import Refusal  # noqa: E402
from tiicee.pipeline.paths import DEFAULT_ROOT, Paths  # noqa: E402

BRANCH = "tii-cee-1/2026-11-13"
# git with nobody's settings: no global configuration, no signing, a fixed identity.
GIT_ENV = {"PATH": os.environ["PATH"], "HOME": "/nonexistent", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}


def git(directory, *arguments):
    return subprocess.run(["git", "-C", str(directory), "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false"]
                          + list(arguments), check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


class Sites:
    def __init__(self, tmp_path, study):
        self.tmp, self.study, self.root = tmp_path, study, study.root
        self.mac = study.code  # the study's code copy becomes the repository on the Mac
        origin = tmp_path / "upstream-origin"
        origin.mkdir()
        (origin / "upstream.py").write_text("VALUE = 1\n")
        git(origin, "init", "-q")
        git(origin, "add", "-A")
        git(origin, "commit", "-q", "-m", "the upstream the study pins")
        self.pin = git(origin, "rev-parse", "HEAD")

        (self.mac / "lacdm" / "upstream").mkdir(parents=True)  # a submodule that is not checked out here
        git(self.mac, "init", "-q")
        git(self.mac, "checkout", "-q", "-b", BRANCH)
        git(self.mac, "add", "-A")
        git(self.mac, "update-index", "--add", "--cacheinfo", "160000,%s,lacdm/upstream" % self.pin)
        git(self.mac, "commit", "-q", "-m", "study code")

        self.checkout = tmp_path / "cluster" / "testing-in-imagination"
        self.checkout.mkdir(parents=True)
        (self.checkout / "README.md").write_text("the cluster's checkout, on its own branch\n")
        git(self.checkout, "init", "-q")
        git(self.checkout, "add", "-A")
        git(self.checkout, "commit", "-q", "-m", "cluster")
        self.upstream = self.checkout / "lacdm" / "upstream"
        subprocess.run(["git", "clone", "-q", str(origin), str(self.upstream)], check=True, capture_output=True, env=GIT_ENV)

    def commit(self, message="a later commit", path="tii-cee/tiicee/pipeline/stages.py", text="\n# a later change\n"):
        with open(self.mac / path, "a") as handle:
            handle.write(text)
        git(self.mac, "add", "-A")
        git(self.mac, "commit", "-q", "-m", message)
        return self.tip()

    def tip(self):
        return git(self.mac, "rev-parse", "refs/heads/" + BRANCH)

    def bundle(self, name="code.bundle", ref="refs/heads/" + BRANCH, source=None):
        path = self.tmp / name
        git(source or self.mac, "bundle", "create", str(path), ref)
        return path

    def deploy(self, bundle=None, *extra):
        """Run deploy.py the way deploy_code.sh does: as a copied single file, outside any package."""
        step = self.root / "deploy" / "deploy.py"
        step.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PACKAGE / "tiicee" / "pipeline" / "deploy.py", step)
        return subprocess.run([WATCHER_PYTHON, str(step), "--bundle", str(bundle or self.bundle()), "--root", str(self.root),
                               "--checkout", str(self.checkout)] + list(extra), capture_output=True, text=True, env=GIT_ENV)

    def deployed(self, *extra):
        result = self.deploy(None, *extra)
        assert result.returncode == 0, result.stdout + result.stderr
        self.study.env["TII_CODE"] = str(self.root / "code" / "current")  # as watcher.env has it
        return json.loads(result.stdout)

    def checkout_state(self):
        """Everything a deployment must leave alone in the cluster checkout."""
        return {"head": git(self.checkout, "symbolic-ref", "HEAD"), "main": git(self.checkout, "rev-parse", "main"),
                "status": git(self.checkout, "status", "--porcelain"), "worktrees": git(self.checkout, "worktree", "list"),
                "files": sorted(p.name for p in self.checkout.iterdir()),
                "readme": (self.checkout / "README.md").read_text(), "upstream": git(self.upstream, "rev-parse", "HEAD")}


@pytest.fixture
def sites(tmp_path, monkeypatch):
    pin(monkeypatch)
    study = Study(tmp_path, [stage("a")])
    yield Sites(tmp_path, study)
    study.close()


def writable(path):
    return bool(stat.S_IMODE(os.lstat(path).st_mode) & 0o222)


# ---- deploy.py ------------------------------------------------------------------------------------

def test_bundle_becomes_a_read_only_export_and_current_points_at_it(sites):
    before = sites.checkout_state()
    commit = sites.tip()
    result = sites.deployed()
    export = sites.root / "code" / commit
    assert result == {"commit": commit, "branch": BRANCH, "export": str(export), "current": str(sites.root / "code" / "current"),
                      "files": result["files"], "submodules": 1}

    # The cluster checkout gained the branch and nothing else moved: not its working tree, its
    # own branch, its worktrees, or the upstream checkout.
    assert git(sites.checkout, "rev-parse", "refs/heads/" + BRANCH) == commit
    assert sites.checkout_state() == before

    # The export is the commit's tree, byte for byte, plus the record of what it is.
    tracked = [line for line in git(sites.mac, "ls-tree", "-r", "--name-only", commit).splitlines() if line != "lacdm/upstream"]
    exported = sorted(str(path.relative_to(export)) for path in export.rglob("*") if path.is_file() and not path.is_symlink()
                      and "upstream" not in path.parts)
    assert exported == sorted(tracked + [deploy.META]) and result["files"] == len(exported)
    for name in ("tii-cee/tiicee/pipeline/watcher.py", "reports/research-advancement/AUTHORIZATION.md"):
        assert (export / name).read_text() == git(sites.mac, "show", "%s:%s" % (commit, name)) + "\n"
    record = json.loads((export / deploy.META).read_text())
    assert (record["commit"], record["branch"]) == (commit, BRANCH)
    assert record["submodules"] == [{"path": "lacdm/upstream", "commit": sites.pin, "source": str(sites.upstream)}]

    # The submodule is not in a bundle: the export links to the cluster's own pinned checkout.
    link = export / "lacdm" / "upstream"
    assert link.is_symlink() and os.readlink(link) == str(sites.upstream)
    assert (link / "upstream.py").read_text() == "VALUE = 1\n"

    # Read-only throughout, and the upstream checkout it links to is not touched.
    assert not writable(export) and not any(writable(path) for path in export.rglob("*") if not path.is_symlink())
    assert writable(sites.upstream) and writable(sites.upstream / "upstream.py")
    if os.geteuid() != 0:
        with pytest.raises(PermissionError):
            (export / "tii-cee" / "new.py").write_text("x")

    current = sites.root / "code" / "current"
    assert current.is_symlink() and os.readlink(current) == commit and current.resolve() == export

    again = sites.deployed()  # the same commit again: nothing to do, nothing changes
    assert again == result and sites.checkout_state() == before


def test_later_commit_gets_its_own_export_and_the_earlier_one_stays(sites):
    first = sites.deployed()
    earlier = (Path(first["export"]) / "tii-cee" / "tiicee" / "pipeline" / "stages.py").read_text()
    second_commit = sites.commit()
    second = sites.deployed()
    assert second["commit"] == second_commit != first["commit"]
    assert os.readlink(sites.root / "code" / "current") == second_commit
    assert (Path(first["export"]) / "tii-cee" / "tiicee" / "pipeline" / "stages.py").read_text() == earlier
    assert (Path(second["export"]) / "tii-cee" / "tiicee" / "pipeline" / "stages.py").read_text() == earlier + "\n# a later change\n"
    assert sorted(p.name for p in (sites.root / "code").iterdir()) == sorted([first["commit"], second_commit, "current"])


def refused(result):
    assert result.returncode == 1, result.stdout + result.stderr
    return result.stdout.strip()


def test_upstream_must_sit_at_the_pinned_commit_with_no_change_a_stage_could_run(sites):
    # An untracked Python file, at the top or in a new folder, is code that was not pinned.
    (sites.upstream / "extra.py").write_text("VALUE = 3\n")
    assert refused(sites.deploy()) == "refused submodule_not_clean"
    (sites.upstream / "extra.py").unlink()
    (sites.upstream / "new" / "deep").mkdir(parents=True)
    (sites.upstream / "new" / "deep" / "module.py").write_text("VALUE = 4\n")
    assert refused(sites.deploy()) == "refused submodule_not_clean"
    shutil.rmtree(sites.upstream / "new")
    # So is any change to a tracked file, committed or not.
    (sites.upstream / "upstream.py").write_text("VALUE = 2\n")
    assert refused(sites.deploy()) == "refused submodule_not_clean"
    git(sites.upstream, "commit", "-q", "-a", "-m", "moved off the pin")
    assert refused(sites.deploy()) == "refused submodule_not_at_pin"
    shutil.rmtree(sites.upstream)
    assert refused(sites.deploy()) == "refused submodule_missing"
    assert not (sites.root / "code").exists()  # no export, no current


def test_stray_file_in_upstream_stops_neither_a_deployment_nor_a_stage(sites):
    for name in (".upstream.py.swp", ".nfs000000000a1b2c3d", "notes.txt", "__pycache__/upstream.cpython-312.pyc"):
        (sites.upstream / name).parent.mkdir(exist_ok=True)
        (sites.upstream / name).write_text("left behind\n")
    sites.deployed()
    paths = Paths(sites.root, sites.root / "code" / "current")
    graph = graphs.load(paths)
    manifests.ensure(paths, graph, graph.by_id["a"])
    manifests.verify(paths, graph, graph.by_id["a"])
    # An untracked Python file appearing later is caught at the next stage start.
    (sites.upstream / "extra.py").write_text("VALUE = 3\n")
    with pytest.raises(Refusal) as caught:
        manifests.verify(paths, graph, graph.by_id["a"])
    assert (caught.value.exit_code, caught.value.code) == (20, "submodule_changed")


def test_branch_checked_out_on_the_cluster_is_never_moved(sites):
    git(sites.checkout, "branch", BRANCH)  # the cluster has the branch, at its own unrelated commit
    there = git(sites.checkout, "rev-parse", BRANCH)
    assert refused(sites.deploy()) == "refused not_fast_forward"  # never rewritten
    assert git(sites.checkout, "rev-parse", BRANCH) == there

    git(sites.checkout, "checkout", "-q", BRANCH)
    before = sites.checkout_state()
    assert refused(sites.deploy()) == "refused branch_checked_out"
    assert sites.checkout_state() == before

    git(sites.checkout, "checkout", "-q", "main")
    git(sites.checkout, "worktree", "add", "-q", str(sites.tmp / "cluster" / "worktree"), BRANCH)
    assert refused(sites.deploy()) == "refused branch_checked_out"  # a linked worktree counts too
    assert not (sites.root / "code").exists()


def test_bundle_that_is_wrong_is_refused_before_anything_changes(sites):
    before = sites.checkout_state()
    good = sites.bundle()
    assert refused(sites.deploy(good, "--sha256", "0" * 64)) == "refused bundle_digest"
    cut = sites.tmp / "cut.bundle"
    cut.write_bytes(good.read_bytes()[:-200])  # a copy that was cut short
    assert refused(sites.deploy(cut)) == "refused bundle_invalid"
    assert refused(sites.deploy(sites.tmp / "absent.bundle")) == "refused bundle_missing"
    other = sites.bundle("main.bundle", "refs/heads/main", source=sites.checkout)
    assert refused(sites.deploy(other)) == "refused branch_not_in_bundle"
    assert sites.checkout_state() == before and not (sites.root / "code").exists()
    assert sites.deploy(good, "--sha256", deploy.sha256_file(good)).returncode == 0


def test_defaults_are_the_cluster_paths_on_record():
    assert deploy.DEFAULT_ROOT == DEFAULT_ROOT
    assert deploy.DEFAULT_CHECKOUT == "/blue/cai6734/daniel.escalante/testing-in-imagination"
    assert deploy.BRANCH == BRANCH


# ---- stages and their exports ---------------------------------------------------------------------

def test_manifest_binds_the_export_and_any_change_to_it_is_a_refusal(sites):
    first = sites.deployed()
    paths = Paths(sites.root, sites.root / "code" / "current")
    assert str(paths.code) == first["export"]  # `current` is resolved once, to the export it names
    graph = graphs.load(paths)
    manifest = manifests.ensure(paths, graph, graph.by_id["a"])
    assert manifest["code"] == {
        "export": first["export"], "commit": first["commit"], "clean": True,
        "snapshot": manifests.tree_digest(paths.code)[0],
        "submodules": [{"path": "lacdm/upstream", "commit": sites.pin, "source": str(sites.upstream)}]}
    assert manifests.verify(paths, graph, graph.by_id["a"]) == manifest

    def code_of(call):
        with pytest.raises(Refusal) as caught:
            call()
        return caught.value.exit_code, caught.value.code

    # A newer deployment does not disturb a stage bound to the earlier export ...
    sites.commit()
    second = sites.deployed()
    assert manifests.verify(paths, graph, graph.by_id["a"]) == manifest
    # ... but the stage must run from that export, not from the new one.
    elsewhere = Paths(sites.root, second["export"])
    assert code_of(lambda: manifests.verify(elsewhere, graphs.load(elsewhere), graph.by_id["a"])) == (20, "code_changed")

    (sites.upstream / "upstream.py").write_text("VALUE = 2\n")
    assert code_of(lambda: manifests.verify(paths, graph, graph.by_id["a"])) == (20, "submodule_changed")
    git(sites.upstream, "checkout", "-q", "--", "upstream.py")
    manifests.verify(paths, graph, graph.by_id["a"])

    target = paths.code / "tii-cee" / "tiicee" / "pipeline" / "stages.py"
    os.chmod(target, 0o644)
    target.write_text(target.read_text() + "# edited in place\n")
    assert code_of(lambda: manifests.verify(paths, graph, graph.by_id["a"])) == (20, "code_changed")


def test_stage_in_flight_stays_on_its_export_and_new_stages_use_current(sites):
    s = sites.study
    s.write_graph([stage("slow", SLOW), stage("later", depends_on=["slow"])])
    first = sites.deployed()
    s.cluster.plan("tii-slow", [{"outcome": "usr1", "steps": 2}])
    s.cycle()
    s.advance()  # the first part ends on the warning signal, two items in

    sites.commit()
    second = sites.deployed()  # new code arrives while the stage is between parts
    s.cycle()
    assert "cd %s/tii-cee || exit 1" % first["export"] in s.script("slow", 2).splitlines()
    assert "export TII_ROOT=%s TII_CODE=%s" % (s.root, first["export"]) in s.script("slow", 2).splitlines()
    s.advance()
    s.cycle()
    assert s.status("slow") == "COMPLETED" and s.items("slow") == ["step-%04d" % step for step in range(6)]
    assert "cd %s/tii-cee || exit 1" % second["export"] in s.script("later", 1).splitlines()
    s.advance()
    s.cycle()
    assert s.status("later") == "COMPLETED" and s.final()["kind"] == "study completed"
    bound = {name: json.loads(s.paths.manifest(name).read_text())["code"] for name in ("slow", "later")}
    assert (bound["slow"]["export"], bound["slow"]["commit"]) == (first["export"], first["commit"])
    assert (bound["later"]["export"], bound["later"]["commit"]) == (second["export"], second["commit"])
    # The allowances and the status were read from the newest export throughout.
    assert s.status_json()["watcher"]["completed"] is True


def test_authorization_is_read_from_the_newest_deployment(sites):
    s = sites.study
    s.write_graph([stage("a"), stage("b", depends_on=["a"])])
    first = sites.deployed()
    s.cycle()  # a is queued, bound to the first export

    git(sites.mac, "rm", "-q", "reports/research-advancement/SIGNATURE.md")
    git(sites.mac, "commit", "-q", "-m", "the signature is withdrawn")
    second = sites.deployed()
    assert Paths(sites.root, first["export"]).study_dir == Path(second["export"]) / "reports" / "research-advancement"
    assert (Path(first["export"]) / "reports" / "research-advancement" / "SIGNATURE.md").exists()

    s.advance()  # a starts now, from its own export, and still finds the authorization withdrawn
    s.cycle()
    assert s.state("a")["refusal"] == {"exit": 30, "reason": "signature_missing"}
    assert (s.status("a"), s.status("b")) == ("REFUSED", "BLOCKED")
    assert s.blocked()["authorization"]["detail"] == {"code": "signature_missing"}


def test_stage_whose_export_is_gone_is_not_resubmitted(sites):
    s = sites.study
    s.write_graph([stage("slow", SLOW)])
    first = sites.deployed()
    s.cluster.plan("tii-slow", ["preempt"])
    s.cycle()
    s.advance()
    sites.commit()
    sites.deployed()
    os.rename(first["export"], first["export"] + ".removed")  # what must never be done by hand
    s.cycle()
    assert s.status("slow") == "BLOCKED" and len(s.calls("sbatch")) == 1
    assert s.blocked()["manifest:slow"]["detail"] == {"stage": "slow", "code": "export_missing", "role": None}


# ---- deploy_code.sh, on the Mac -------------------------------------------------------------------

SSH_STUB = """#!/bin/bash
printf "ssh" >> "$STUB_LOG"; printf "\\t%s" "$@" >> "$STUB_LOG"; printf "\\n" >> "$STUB_LOG"
if [ "$1" = "-O" ]; then [ -f "$STUB_DIR/master_up" ]; exit $?; fi
[ -f "$STUB_DIR/master_up" ] || exit 255
[ "$1 $2 $3 $4 $5 $6" = "-o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false" ] || exit 99
shift 7
exec /bin/bash -c "$*"
"""


class Mac:
    def __init__(self, sites):
        self.sites, tmp = sites, sites.tmp
        (tmp / "bin").mkdir()
        (tmp / "bin" / "ssh").write_text(SSH_STUB)
        (tmp / "bin" / "ssh").chmod(0o755)
        (tmp / "work").mkdir()
        (tmp / "master_up").touch()
        self.log = tmp / "ssh.log"
        conf = tmp / "local_loop.conf"
        conf.write_text('TII_HPG_HOST="hpg"\nTII_REMOTE_ROOT="%s"\nTII_REPO="%s"\nTII_REMOTE_CHECKOUT="%s"\n'
                        'TII_REMOTE_PYTHON="%s"\n' % (sites.root, sites.mac, sites.checkout, WATCHER_PYTHON))
        self.env = dict(GIT_ENV, HOME=str(tmp / "home"), PATH="%s:/usr/bin:/bin:/usr/sbin:/sbin" % (tmp / "bin"),
                        TII_LOCAL_LOOP_CONF=str(conf), STUB_LOG=str(self.log), STUB_DIR=str(tmp), TMPDIR=str(tmp / "work"))

    def run(self):
        return subprocess.run(["/bin/bash", str(PACKAGE / "local" / "deploy_code.sh")], env=self.env,
                              capture_output=True, text=True, timeout=120)

    def calls(self):
        return [line.split("\t") for line in self.log.read_text().splitlines()] if self.log.exists() else []


mac_only = pytest.mark.skipif(sys.platform != "darwin", reason="deploy_code.sh is for macOS")


@mac_only
def test_deploy_code_sends_a_bundle_through_the_master_and_runs_the_deploy_step(sites):
    mac = Mac(sites)
    commit = sites.tip()
    result = mac.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "deploy_code: commit %s is deployed" % commit in result.stdout
    export = sites.root / "code" / commit
    assert json.loads(result.stdout.splitlines()[0])["export"] == str(export)
    assert os.readlink(sites.root / "code" / "current") == commit and (export / deploy.META).exists()

    # What reached the cluster: the bundle, named by its commit, and the deploy step as committed.
    assert sorted(path.name for path in (sites.root / "deploy").iterdir()) == sorted([commit + ".bundle", "deploy.py"])
    assert (sites.root / "deploy" / "deploy.py").read_text() == (sites.mac / "tii-cee" / "tiicee" / "pipeline" / "deploy.py").read_text()
    assert list((sites.tmp / "work").iterdir()) == []  # nothing left behind on the Mac

    # One check that contacts no server, then only commands that cannot open a connection.
    calls = mac.calls()
    assert calls[0] == ["ssh", "-O", "check", "hpg"] and len(calls) == 4  # check, bundle, deploy step, run
    assert all(call[1:8] == ["-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ProxyCommand=false", "hpg"]
               for call in calls[1:])

    # Only committed work is sent, and the script says so.
    (sites.mac / "tii-cee" / "notes.py").write_text("x = 1\n")
    later = mac.run()
    assert later.returncode == 0 and "1 uncommitted change(s) are not part of this deployment" in later.stdout
    assert not (export / "tii-cee" / "notes.py").exists()


@mac_only
def test_deploy_code_never_opens_a_connection_and_reports_a_refusal(sites):
    mac = Mac(sites)
    (sites.tmp / "master_up").unlink()
    result = mac.run()
    assert result.returncode == 3 and "In Terminal run: ssh -fN hpg" in result.stderr
    assert mac.calls() == [["ssh", "-O", "check", "hpg"]] and not (sites.root / "deploy").exists()

    (sites.tmp / "master_up").touch()
    (sites.upstream / "extra.py").write_text("VALUE = 3\n")
    result = mac.run()
    assert result.returncode == 5 and "refused submodule_not_clean" in result.stdout
    assert not (sites.root / "code").exists()

    # The storage root is made by the launch session, owner-only; this script never creates it.
    os.rename(sites.root, str(sites.root) + ".elsewhere")
    result = mac.run()
    assert result.returncode == 4 and "create it first, owner-only" in result.stderr
    assert not sites.root.exists()
