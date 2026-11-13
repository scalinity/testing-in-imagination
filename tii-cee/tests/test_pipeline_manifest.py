"""Manifests (specification 8.3 and 19), the graph file and the authorization checks."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fake_slurm"))
from pipeline_harness import (ADDENDUM_SHA, AUTHORIZATION_SHA, BOUND_ADDENDUM_SHA,  # noqa: E402
                              BOUND_AUTHORIZATION_SHA, PACKAGE, Study, pin, stage)

from tiicee.pipeline import authorization  # noqa: E402
from tiicee.pipeline import graph as graphs  # noqa: E402
from tiicee.pipeline import manifest as manifests  # noqa: E402
from tiicee.pipeline.authorization import BASE_POOLS, ADDENDUM_1_POOLS, Pool  # noqa: E402
from tiicee.pipeline.manifest import Refusal  # noqa: E402
from tiicee.pipeline.paths import DEFAULT_ROOT, Paths  # noqa: E402
from tiicee.pipeline.util import sha256_file, tree_digest  # noqa: E402

BIND = {"card": "cards/E4.md", "calibration": "$TII/private/artifacts/calibration.json",
        "policy_checkpoints": "$TII/private/checkpoints"}


def _append(path, text):
    """Append and close, so no file handle is left for the garbage collector (a ResourceWarning under -W error)."""
    with path.open("a") as handle:
        handle.write(text)


@pytest.fixture(autouse=True)
def synthetic_authorization(monkeypatch):
    """Every test here works on the study's own synthetic files, never the repository's."""
    pin(monkeypatch)


@pytest.fixture
def study(tmp_path):
    spec = stage("a", ["noop", "--count", "2"], experiment="E4", gpus=1, estimate=7, bind=BIND)
    spec["manifest"].update(seeds={"sampling": 11, "bootstrap": 12}, pairing_unit="admission",
                            intervention="legacy repair")
    spec["env"] = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    s = Study(tmp_path, [spec])
    (s.code / "cards").mkdir()
    (s.code / "cards" / "E4.md").write_text("card\n")
    (s.paths.private / "artifacts").mkdir(parents=True)
    (s.paths.private / "artifacts" / "calibration.json").write_text('{"scale": 1.0}')
    for seed in ("seed-1", "seed-2"):
        (s.paths.private / "checkpoints" / seed).mkdir(parents=True)
        (s.paths.private / "checkpoints" / seed / "adapter.bin").write_bytes(seed.encode())
    return s


def load(s):
    graph = graphs.load(s.paths)
    return graph, graph.by_id["a"]


def code_of(call):
    with pytest.raises(Refusal) as caught:
        call()
    return caught.value.exit_code, caught.value.code, caught.value.role


# ---- what a manifest binds ----------------------------------------------------------------------

def test_manifest_binds_every_field_of_the_specification(study):
    graph, a = load(study)
    manifest = manifests.build(study.paths, graph, a)
    assert {key: manifest[key] for key in ("study", "experiment", "stage", "kind", "intervention",
                                           "seeds", "pairing_unit", "command", "env")} == {
        "study": "TII-CEE-1", "experiment": "E4", "stage": "a", "kind": "normal",
        "intervention": "legacy repair", "seeds": {"sampling": 11, "bootstrap": 12},
        "pairing_unit": "admission", "command": ["noop", "--count", "2"],
        "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}
    assert manifest["allowance"] == {"key": "E4", "gpu_hours": 400, "estimate_gpu_hours": 7}
    assert manifest["authorization"] == {
        "sha256": AUTHORIZATION_SHA, "expires": "2027-03-31",
        "addenda": {"AUTHORIZATION_ADDENDUM_1.md": {"sha256": ADDENDUM_SHA, "sha256_ok": True, "signed": False}}}
    assert sha256_file(study.docs / "AUTHORIZATION.md") == AUTHORIZATION_SHA
    # Not a git checkout here, so the code is bound by a snapshot digest.
    assert manifest["code"] == {"export": None, "commit": None, "clean": False,
                                "snapshot": manifests.snapshot(study.paths), "submodules": []}
    bound = manifest["bound"]
    assert bound["card"] == {"path": "cards/E4.md", "type": "file", "sha256": sha256_file(study.code / "cards" / "E4.md")}
    assert bound["calibration"]["sha256"] == sha256_file(study.paths.private / "artifacts" / "calibration.json")
    assert bound["policy_checkpoints"] == {"path": "$TII/private/checkpoints", "type": "tree", "files": 2,
                                           "sha256": tree_digest(study.paths.private / "checkpoints")[0]}
    assert manifest["manifest_id"] == manifests.build(study.paths, graph, a)["manifest_id"]


def test_tree_digest_is_sorted_paths_with_file_hashes(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    for root, order in ((one, ["b/x.bin", "a.bin"]), (two, ["a.bin", "b/x.bin"])):
        for name in order:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(name.encode())
    assert tree_digest(one) == tree_digest(two) and tree_digest(one)[1] == 2
    (two / "b" / "x.bin").rename(two / "b" / "y.bin")
    assert tree_digest(one)[0] != tree_digest(two)[0]  # a renamed file is a different tree
    assert tree_digest(one, keep=lambda path: path.endswith("a.bin"))[1] == 1


def test_manifest_is_written_once_and_never_rewritten(study):
    graph, a = load(study)
    first = manifests.ensure(study.paths, graph, a)
    assert not os.access(study.paths.manifest("a"), os.W_OK)
    (study.paths.private / "artifacts" / "calibration.json").write_text('{"scale": 9.0}')
    assert manifests.ensure(study.paths, graph, a) == first  # the bound manifest, not a new one


# ---- verification -------------------------------------------------------------------------------

def test_unchanged_stage_verifies(study):
    graph, a = load(study)
    bound = manifests.ensure(study.paths, graph, a)
    assert manifests.verify(study.paths, graph, a) == bound


@pytest.mark.parametrize("change, expected", [
    (lambda s: (s.paths.private / "artifacts" / "calibration.json").write_text("{}"), (20, "bound_changed", "calibration")),
    (lambda s: (s.code / "cards" / "E4.md").unlink(), (20, "bound_missing", "card")),
    (lambda s: (s.paths.private / "checkpoints" / "seed-2" / "adapter.bin").write_bytes(b"other"),
     (20, "bound_changed", "policy_checkpoints")),
    (lambda s: (s.paths.private / "checkpoints" / "seed-3").mkdir() or
     (s.paths.private / "checkpoints" / "seed-3" / "adapter.bin").write_bytes(b"new"), (20, "bound_changed", "policy_checkpoints")),
    (lambda s: _append(s.code / "tii-cee" / "tiicee" / "pipeline" / "stages.py", "\n# edited\n"), (20, "code_changed", None)),
    (lambda s: _append(s.docs / "AUTHORIZATION.md", "x"), (30, "authorization_changed", None)),
    (lambda s: (s.docs / "SIGNATURE.md").write_text("unsigned"), (30, "signature_mismatch", None)),
    (lambda s: s.paths.manifest("a").unlink(), (20, "manifest_missing", None)),
])
def test_any_change_to_what_is_bound_is_a_refusal(study, change, expected):
    graph, a = load(study)
    manifests.ensure(study.paths, graph, a)
    change(study)
    assert code_of(lambda: manifests.verify(study.paths, graph, a)) == expected


def test_changed_stage_definition_or_edited_manifest_is_a_refusal(study):
    graph, a = load(study)
    manifests.ensure(study.paths, graph, a)
    changed = dict(a, command=["noop", "--count", "3"])
    assert code_of(lambda: manifests.verify(study.paths, graph, changed)) == (20, "stage_changed", "command")
    changed = dict(a, seeds={"sampling": 99, "bootstrap": 12})
    assert code_of(lambda: manifests.verify(study.paths, graph, changed)) == (20, "stage_changed", "seeds")
    # The environment a stage was bound with is part of what it is: training cannot quietly gain
    # the evaluation setting, or lose its own.
    for env in ({}, {"CUBLAS_WORKSPACE_CONFIG": None}, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "EXTRA": "1"}):
        changed = dict(a, env=env)
        assert code_of(lambda: manifests.verify(study.paths, graph, changed)) == (20, "stage_changed", "env")

    path = study.paths.manifest("a")
    edited = json.loads(path.read_text())
    edited["seeds"]["sampling"] = 99
    os.chmod(path, 0o644)
    path.write_text(json.dumps(edited))
    assert code_of(lambda: manifests.verify(study.paths, graph, a)) == (20, "manifest_altered", None)


def test_test_documentation_and_mac_side_files_are_not_stage_code(study):
    before = manifests.snapshot(study.paths)
    for relative in ("tests/test_new.py", "local/local_loop.sh", "README.md", ".venv/lib/x.py"):
        path = study.paths.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    assert manifests.snapshot(study.paths) == before
    (study.paths.gate_rules / "g9.py").write_text("x")
    changed = manifests.snapshot(study.paths)
    assert changed != before
    nested = study.paths.package / "tiicee" / "twin" / "local" / "model.py"  # only the top-level folders are skipped
    nested.parent.mkdir(parents=True)
    nested.write_text("x")
    assert manifests.snapshot(study.paths) != changed


def test_clean_git_checkout_is_bound_by_its_commit(study):
    environment = {"PATH": os.environ["PATH"], "HOME": str(study.root), "GIT_CONFIG_GLOBAL": os.devnull,
                   "GIT_CONFIG_SYSTEM": os.devnull}

    def git(*arguments):
        subprocess.run(["git", "-C", str(study.code), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false"] + list(arguments), check=True, capture_output=True, env=environment)

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "throwaway checkout for a test")
    graph, a = load(study)
    manifest = manifests.ensure(study.paths, graph, a)
    assert manifest["code"]["clean"] is True and len(manifest["code"]["commit"]) == 40
    assert manifest["code"]["snapshot"] is None
    manifests.verify(study.paths, graph, a)
    _append(study.paths.package / "tiicee" / "pipeline" / "stages.py", "\n# uncommitted\n")
    assert code_of(lambda: manifests.verify(study.paths, graph, a)) == (20, "code_changed", None)


def test_command_line_builds_and_verifies(study, monkeypatch, capsys):
    monkeypatch.setenv("TII_ROOT", str(study.root))
    monkeypatch.setenv("TII_CODE", str(study.code))
    assert manifests.main(["build", "a"]) == 0
    identity = capsys.readouterr().out.strip()
    assert manifests.main(["verify", "a"]) == 0 and capsys.readouterr().out.strip() == identity
    (study.paths.private / "artifacts" / "calibration.json").write_text("{}")
    assert manifests.main(["verify", "a"]) == 20
    assert capsys.readouterr().out.strip() == "refused 20 bound_changed:calibration"
    assert manifests.main(["verify", "nothing"]) == 2


# ---- the authorization --------------------------------------------------------------------------

def test_the_committed_code_binds_the_recorded_hashes():
    """The values written in SIGNATURE.md at the time of signing; no file is read to check them."""
    assert BOUND_AUTHORIZATION_SHA == "72ccc89c8a825c82578c8efdf942025e3ce60d72f179b8b13489737ee37221ae"
    assert BOUND_ADDENDUM_SHA == "f59cd6fa638d00b2d1bc9bdc4848ba9ae412f9abecb7cdfcfa83cb4475e05658"
    assert authorization.EXPIRES == datetime.date(2027, 3, 31)


def test_authorization_check_codes(study):
    docs = study.docs
    assert authorization.inspect(docs) == {"code": "ok", "sha256_ok": True, "signature_ok": True}
    assert authorization.check(docs, today=datetime.date(2027, 3, 31)) == "ok"
    assert authorization.check(docs, today=datetime.date(2027, 4, 1)) == "authorization_expired"
    (docs / "SIGNATURE.md").write_text("a different hash\n")
    assert authorization.inspect(docs) == {"code": "signature_mismatch", "sha256_ok": True, "signature_ok": False}
    (docs / "SIGNATURE.md").unlink()
    assert authorization.check(docs) == "signature_missing"
    (docs / "AUTHORIZATION.md").write_text("changed\n")
    assert authorization.inspect(docs) == {"code": "authorization_changed", "sha256_ok": False, "signature_ok": False}
    (docs / "AUTHORIZATION.md").unlink()
    assert authorization.check(docs) == "authorization_missing"
    assert authorization.pools(docs) == ()


def test_addendum_counts_only_while_its_exact_text_is_signed(study):
    docs = study.docs
    assert authorization.pools(docs) == BASE_POOLS  # on disk, hash as bound, but not in SIGNATURE.md
    assert authorization.addenda(docs)["AUTHORIZATION_ADDENDUM_1.md"] == {
        "sha256": ADDENDUM_SHA, "sha256_ok": True, "signed": False}
    assert BASE_POOLS == (Pool("gpu", "cai6734", "cai6734", 8), Pool("cpu", "cai6734", "cai6734-b", 288))
    study.sign_addendum()
    assert authorization.pools(docs) == BASE_POOLS + ADDENDUM_1_POOLS
    assert ADDENDUM_1_POOLS[0] == Pool("gpu", "cai5724", "cai5724", 8)
    assert {pool.qos for pool in ADDENDUM_1_POOLS if pool.kind == "cpu"} == {
        "cai5724-b", "cai5720-b", "cai5731-b", "cai5720", "cai5731"}

    with open(docs / "AUTHORIZATION_ADDENDUM_1.md", "a") as handle:
        handle.write("and also hpg-b200\n")
    assert authorization.pools(docs) == BASE_POOLS  # the signed text is no longer the text on disk
    assert authorization.addenda(docs)["AUTHORIZATION_ADDENDUM_1.md"]["sha256_ok"] is False
    study.sign_addendum()                           # even a signature of the new text does not
    assert authorization.pools(docs) == BASE_POOLS  # widen compute this code has not been given
    (docs / "AUTHORIZATION_ADDENDUM_1.md").unlink()
    assert authorization.addenda(docs) == {} and authorization.pools(docs) == BASE_POOLS


def test_pools_fill_in_order(study):
    study.sign_addendum()
    usable = authorization.pools(study.docs)
    # GPUs in use are counted per account; CPUs per account and QOS.
    assert authorization.pick_pool(usable, "gpu", 1, {"cai6734": 7}).account == "cai6734"
    assert authorization.pick_pool(usable, "gpu", 1, {"cai6734": 8}).account == "cai5724"
    assert authorization.pick_pool(usable, "gpu", 1, {"cai6734": 8, "cai5724": 8}) is None
    assert authorization.pick_pool(usable, "cpu", 4, {}).qos == "cai6734-b"
    assert authorization.pick_pool(usable, "cpu", 4, {("cai6734", "cai6734-b"): 286}).qos == "cai5724-b"


# ---- the graph and the allowances ---------------------------------------------------------------

def test_allowances_file_mirrors_the_authorization():
    paths = Paths(root="/nowhere", code=PACKAGE.parent)
    allowances = graphs.load_allowances(paths)
    whole = json.loads(paths.allowances.read_text())
    assert whole["authorization_sha256"] == BOUND_AUTHORIZATION_SHA
    # The totals AUTHORIZATION.md section 4 states: 1,333 estimated and 2,810 allowed.
    assert sum(entry["estimate_gpu_hours"] for entry in allowances.values()) == 1333
    assert sum(entry["allowance_gpu_hours"] for entry in allowances.values()) == 2810
    assert [key for key, entry in allowances.items() if not entry["authorized"]] == ["E11"]
    assert (allowances["E2"]["allowance_gpu_hours"], allowances["E5"]["allowance_gpu_hours"]) == (200, 800)


def test_only_the_id_that_collides_with_the_watchers_job_is_refused(tmp_path):
    study = Study(tmp_path)
    # The name the guard protects is the one the scrontab entry gives the watcher's job.
    entry = (PACKAGE / "tiicee" / "pipeline" / "scrontab_entry.txt").read_text()
    assert "#SCRON --job-name=%s\n" % graphs.WATCHER_JOB in entry and graphs.WATCHER_JOB == "tii-watcher"
    study.write_graph([stage("watcher")])
    with pytest.raises(graphs.GraphError, match="stage watcher: this id would give the stage the watcher's own job name"):
        graphs.load(study.paths)
    # Ids that merely contain the word give other job names, and stay allowed.
    study.write_graph([stage("watcher2"), stage("the-watcher"), stage("Watcher"), stage("watch")])
    assert [s["id"] for s in graphs.load(study.paths).stages] == ["watcher2", "the-watcher", "Watcher", "watch"]


def test_graph_keeps_env_and_stage_modules(tmp_path):
    study = Study(tmp_path)
    modules = ["tiicee.clinical.folds", "tiicee.clinical.train", "tiicee.clinical.rollout",
               "tiicee.clinical.bench", "tiicee.clinical.acceptance"]
    training = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUBLAS_WORKSPACE_CONFIG": None}
    study.write_graph([stage("train", gpus=1, env=training), stage("plain")], stage_modules=modules)
    graph = graphs.load(study.paths)
    assert graph.stage_modules == modules
    assert graph.by_id["train"]["env"] == training and graph.by_id["plain"]["env"] == {}


def test_the_clinical_stage_modules_register_through_stage_modules():
    """The modules the study's graph will list. Skipped where their own imports are not installed."""
    from tiicee.pipeline import stages
    wanted = {"tiicee.clinical.folds": ["clinical_data", "clinical_folds"], "tiicee.clinical.train": ["clinical_train"],
              "tiicee.clinical.rollout": ["clinical_rollout"], "tiicee.clinical.bench": ["clinical_bench"],
              "tiicee.clinical.acceptance": ["clinical_acceptance"]}
    for module, names in wanted.items():
        for name in names:
            try:
                assert callable(stages.resolve(name, [module]))
            except ImportError as error:
                pytest.skip("%s cannot be imported here: %s" % (module, type(error).__name__))


def test_committed_noop_graph_loads_and_paths_follow_the_design(tmp_path):
    study = Study(tmp_path)
    study.paths.graph.write_text((PACKAGE / "tiicee" / "pipeline" / "noop_graph.json").read_text())
    graph = graphs.load(study.paths)
    assert [s["id"] for s in graph.stages] == ["noop_a", "noop_b", "noop_c"]
    assert graph.waits_on(graph.by_id["noop_b"]) == ["noop_a", "noop_a"]
    default = Paths(code="/code")
    assert str(default.root) == DEFAULT_ROOT == "/blue/cai6734/daniel.escalante/private/tii-cee-1"
    assert [str(path.relative_to(default.root)) for path in (
        default.graph, default.stage_state("x"), default.lock, default.runlog, default.events,
        default.status, default.blocked_md, default.gate_record("G1"), default.approved, default.private)] == [
        "pipeline/graph.json", "state/stages/x.json", "state/lock", "state/runlog.jsonl", "state/events.tsv",
        "state/STATUS.md", "state/BLOCKED.md", "gates/G1.json", "export/approved", "private"]


@pytest.mark.parametrize("change, message", [
    (lambda g: g["stages"][0]["resources"].update(time_hours=24.5), "time_hours"),
    (lambda g: g["stages"][0]["resources"].update(time_hours=0.1), "time_hours"),
    (lambda g: g["stages"][0]["resources"].update(gpus=2), "gpus must be 0 or 1"),
    (lambda g: g["stages"][0].update(estimate_gpu_hours=0), "positive estimate"),
    (lambda g: g["stages"][0].update(allowance_key="E99"), "allowance_key"),
    (lambda g: g["stages"][0].update(depends_on=["nothing"]), "unknown dependency"),
    (lambda g: g["stages"][0].update(depends_on=["b"]), "cycle"),
    (lambda g: g["stages"][0].update(gate="G"), "cycle"),
    (lambda g: g["stages"][0].update(kind="final"), "kind"),
    (lambda g: g["stages"][0].update(id="has space"), "bad stage id"),
    (lambda g: g["stages"][0].update(id="watcher"), "the watcher's own job name"),
    (lambda g: g["stages"][0].update(command="noop"), "command"),
    (lambda g: g["stages"][0].update(manifest={"bind": {"two words": "x"}}), "role names"),
    (lambda g: g["stages"][0].update(env={"NOT A NAME": "1"}), "env must map"),
    (lambda g: g["stages"][0].update(env={"THREADS": 4}), "env must map"),
    (lambda g: g["stages"][0].update(env={"TII_CODE": "/elsewhere"}), "env must map"),
    (lambda g: g["stages"][0].update(env={"LINES": "one\ntwo"}), "env must map"),
    (lambda g: g.update(stage_modules="tiicee.clinical.train"), "stage_modules"),
    (lambda g: g.update(stage_modules=["tiicee/clinical/train.py"]), "stage_modules"),
    (lambda g: g["stages"].append(dict(g["stages"][1])), "duplicate stage id"),
    (lambda g: g["gates"][0].update(rule="../elsewhere.py"), "rule"),
    (lambda g: g["gates"][0].update(reads=["../private/rows.csv"]), "reads"),
    (lambda g: g["gates"][0].update(inputs=[]), "at least one input"),
])
def test_graph_that_breaks_a_rule_is_rejected(tmp_path, change, message):
    study = Study(tmp_path)
    raw = {"stages": [stage("a", gpus=1), stage("b", depends_on=["a"])],
           "gates": [{"id": "G", "rule": "noop_gate.py", "inputs": ["b"], "reads": []}]}
    change(raw)
    study.paths.graph.write_text(json.dumps(raw))
    with pytest.raises(graphs.GraphError, match=message):
        graphs.load(study.paths)
