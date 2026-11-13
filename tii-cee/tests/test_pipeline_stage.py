"""run_stage in-process: the stage contract of WATCHER_DESIGN.md section 3."""
import fcntl
import json
import os
import signal
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fake_slurm"))
from pipeline_harness import Study, pin, stage  # noqa: E402

from tiicee.pipeline import export_guard  # noqa: E402
from tiicee.pipeline import graph as graphs  # noqa: E402
from tiicee.pipeline import manifest as manifests  # noqa: E402
from tiicee.pipeline import stage as runner  # noqa: E402
from tiicee.pipeline import stages  # noqa: E402
from tiicee.pipeline.util import tree_digest  # noqa: E402

ARTIFACT = "$TII/private/artifacts/index.bin"


@pytest.fixture
def study(tmp_path, monkeypatch):
    pin(monkeypatch)  # the study's own synthetic authorization files, never the repository's
    for name in ("SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS"):
        monkeypatch.delenv(name, raising=False)

    def make(*specs):
        s = Study(tmp_path, specs or [stage("a")])
        (s.paths.private / "artifacts").mkdir(parents=True)
        (s.paths.private / "artifacts" / "index.bin").write_bytes(b"first")
        return s

    return make


def bind(s, name="a"):
    graph = graphs.load(s.paths)
    return manifests.ensure(s.paths, graph, graph.by_id[name])


def refusal(s, name="a"):
    written = json.loads(s.paths.refusal(name).read_text())
    return written["exit"], written["code"], written["role"]


def test_completed_stage_publishes_outputs_then_the_marker_and_is_not_run_again(study, monkeypatch):
    s = study()
    manifest = bind(s)
    assert runner.run_stage(s.paths, "a") == 0
    marker = json.loads(s.paths.done_marker("a").read_text())
    assert marker["output_sha256"] == tree_digest(s.paths.stage_out("a"))[0]
    assert marker["manifest_id"] == manifest["manifest_id"] and marker["files"] == 1
    assert not (s.paths.stage_dir("a") / "out.partial").exists()  # the temporary name is gone
    assert json.loads((s.paths.approved / "a.json").read_text()) == {"count": 1, "stage": "a"}

    def must_not_run(ctx):
        raise AssertionError("a completed stage ran again")

    monkeypatch.setitem(stages.REGISTRY, "noop", must_not_run)
    assert runner.run_stage(s.paths, "a") == 0


def test_outputs_changed_after_completion_are_an_identity_refusal(study):
    s = study()
    bind(s)
    runner.run_stage(s.paths, "a")
    (s.paths.stage_out("a") / "result.json").write_text("{}")
    assert runner.run_stage(s.paths, "a") == 20 and refusal(s) == (20, "output_changed", None)


def test_stage_without_a_manifest_or_with_a_changed_artifact_exits_20(study):
    s = study(stage("a", bind={"index": ARTIFACT}))
    assert runner.run_stage(s.paths, "a") == 20 and refusal(s) == (20, "manifest_missing", None)
    bind(s)
    (s.paths.private / "artifacts" / "index.bin").write_bytes(b"second")
    assert runner.run_stage(s.paths, "a") == 20 and refusal(s) == (20, "bound_changed", "index")
    assert not s.paths.done_marker("a").exists() and not s.paths.stage_out("a").exists()


def test_missing_authorization_exits_30(study):
    s = study()
    bind(s)
    (s.docs / "AUTHORIZATION.md").rename(s.docs / "gone.md")
    assert runner.run_stage(s.paths, "a") == 30 and refusal(s) == (30, "authorization_missing", None)


def test_job_on_an_account_the_authorization_does_not_allow_exits_30(study, monkeypatch):
    s = study()
    bind(s)
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "cai5724")
    monkeypatch.setenv("SLURM_JOB_QOS", "cai5724")
    assert runner.run_stage(s.paths, "a") == 30 and refusal(s) == (30, "account_not_authorized", None)
    s.sign_addendum()
    assert runner.run_stage(s.paths, "a") == 0


def test_warning_signal_sets_the_stop_flag_and_the_stage_exits_10_then_resumes(study, monkeypatch):
    s = study()
    bind(s)
    seen = []

    def two_then_stop(ctx):
        """Completes two items per start; the first start is interrupted by the warning signal."""
        for key in ("case-1", "case-2", "case-3", "case-4"):
            if ctx.is_done(key):
                continue
            seen.append(key)
            ctx.mark_done(key, {"score": len(key)})
            if key == "case-2":
                os.kill(os.getpid(), signal.SIGUSR1)
            if ctx.stop:
                ctx.save_checkpoint({"after": key})
                raise stages.StageInterrupted()
        assert ctx.load_checkpoint() == {"after": "case-2"}
        (ctx.out_dir / "scores.json").write_text(json.dumps(ctx.results(), sort_keys=True))

    monkeypatch.setitem(stages.REGISTRY, "noop", two_then_stop)
    assert runner.run_stage(s.paths, "a") == 10
    assert not s.paths.done_marker("a").exists()
    assert runner.run_stage(s.paths, "a") == 0
    assert seen == ["case-1", "case-2", "case-3", "case-4"]  # nothing scored twice
    assert sorted(json.loads((s.paths.stage_out("a") / "scores.json").read_text())) == seen
    assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL


def test_memory_error_exits_40(study, monkeypatch):
    s = study()
    bind(s)

    def too_big(ctx):
        raise MemoryError()

    monkeypatch.setitem(stages.REGISTRY, "noop", too_big)
    monkeypatch.setenv("TII_ROOT", str(s.root))
    monkeypatch.setenv("TII_CODE", str(s.code))
    assert runner.main(["a"]) == 40


def test_an_item_cannot_be_completed_twice_and_a_torn_line_is_dropped(study):
    s = study()
    manifest = bind(s)
    graph = graphs.load(s.paths)
    s.paths.stage_dir("a").mkdir(parents=True)
    context = runner.Context(s.paths, graph.by_id["a"], manifest)
    context.mark_done("case-1", 0.5)
    with pytest.raises(ValueError):
        context.mark_done("case-1", 0.7)
    log = s.paths.stage_dir("a") / "items.jsonl"
    whole = log.read_bytes()
    log.write_bytes(whole + b'{"key": "case-2", "resu')  # a kill in the middle of a write
    again = runner.Context(s.paths, graph.by_id["a"], manifest)
    assert again.results() == {"case-1": 0.5} and not again.is_done("case-2")
    assert log.read_bytes() == whole


def test_second_instance_of_a_running_stage_exits_without_working(study, monkeypatch):
    s = study()
    bind(s)
    s.paths.stage_dir("a").mkdir(parents=True)
    held = os.open(s.paths.stage_dir("a") / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    monkeypatch.setitem(stages.REGISTRY, "noop", lambda ctx: pytest.fail("ran under another instance's lock"))
    assert runner.run_stage(s.paths, "a") == 10
    os.close(held)


def test_kill_between_the_rename_and_the_marker_is_finished_on_the_next_start(study, monkeypatch):
    s = study()
    bind(s)
    out = s.paths.stage_out("a")
    out.mkdir(parents=True)
    (out / "result.json").write_text('{"count": 1}\n')
    monkeypatch.setitem(stages.REGISTRY, "noop", lambda ctx: pytest.fail("the work was already complete"))
    assert runner.run_stage(s.paths, "a") == 0
    assert json.loads(s.paths.done_marker("a").read_text())["output_sha256"] == tree_digest(out)[0]


def test_test_stage_runs_once_resume_is_the_same_run_and_a_new_manifest_is_refused(study, monkeypatch):
    s = study(stage("e6", kind="test", bind={"index": ARTIFACT}))
    first = bind(s, "e6")
    starts = []

    def interrupted_once(ctx):
        starts.append(ctx.manifest["manifest_id"])
        if len(starts) == 1:
            ctx.mark_done("case-1")
            raise stages.StageInterrupted()
        ctx.mark_done("case-2")

    monkeypatch.setitem(stages.REGISTRY, "noop", interrupted_once)
    assert runner.run_stage(s.paths, "e6") == 10
    entry = s.paths.ledger_entry("e6").read_text()
    assert json.loads(entry)["manifest_id"] == first["manifest_id"]

    # A second start under a different manifest: the artifact is replaced and the stage rebound.
    (s.paths.private / "artifacts" / "index.bin").write_bytes(b"second")
    os.rename(s.paths.manifest("e6"), s.paths.manifest("e6").with_suffix(".first"))
    assert bind(s, "e6")["manifest_id"] != first["manifest_id"]
    assert runner.run_stage(s.paths, "e6") == 20 and refusal(s, "e6") == (20, "test_ledger_mismatch", None)
    assert starts == [first["manifest_id"]] and s.paths.ledger_entry("e6").read_text() == entry

    # The original manifest again: a resume, the same run, and the ledger is untouched.
    (s.paths.private / "artifacts" / "index.bin").write_bytes(b"first")
    os.replace(s.paths.manifest("e6").with_suffix(".first"), s.paths.manifest("e6"))
    assert runner.run_stage(s.paths, "e6") == 0
    assert starts == [first["manifest_id"]] * 2 and s.paths.ledger_entry("e6").read_text() == entry
    assert s.items("e6") == ["case-1", "case-2"]
    assert not s.paths.ledger_entry("a").exists()


def test_progress_is_counts_only_written_at_most_once_a_minute_and_once_at_the_end(study, monkeypatch):
    s = study()
    manifest = bind(s)
    s.paths.stage_dir("a").mkdir(parents=True)
    context = runner.Context(s.paths, graphs.load(s.paths).by_id["a"], manifest)
    clock = [1000.0]
    monkeypatch.setattr(runner, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))

    # A progress file is prepared outside state/progress/ and renamed in, so a watcher cycle
    # running at that instant finds nothing unexpected there to move away.
    moments, rename = [], os.replace
    monkeypatch.setattr(os, "replace", lambda source, target: (
        moments.append((Path(source).parent, export_guard.file_violations(s.paths))), rename(source, target))[1])
    context.progress(1, 2160, "admission")
    assert moments == [(s.paths.staging, [])]
    first = s.progress("a")
    assert {key: first[key] for key in ("stage", "unit", "done", "total")} == {
        "stage": "a", "unit": "admission", "done": 1, "total": 2160}
    assert set(first) == {"stage", "unit", "done", "total", "updated_at"}
    clock[0] += 59
    context.progress(40, 2160, "admission")
    assert s.progress("a")["done"] == 1      # inside the minute: nothing written
    clock[0] += 1
    context.progress(41, 2160, "admission")
    assert s.progress("a")["done"] == 41     # a minute on: written
    context.progress(42, 2160, "admission")
    context.flush_progress()                  # what the runner does when the attempt ends
    assert s.progress("a")["done"] == 42
    written = s.paths.progress_file("a").stat().st_mtime_ns
    context.flush_progress()
    assert s.paths.progress_file("a").stat().st_mtime_ns == written  # nothing new, nothing rewritten

    for bad in ((1.5, 10, "admission"), (-1, 10, "admission"), (1, "ten", "admission"), (1, 10, "two words")):
        with pytest.raises(ValueError):
            context.progress(*bad)


def test_stage_that_stops_itself_to_continue_in_a_new_job_keeps_its_progress(study, monkeypatch):
    s = study()
    bind(s)

    def two_phases(ctx):
        if (ctx.load_checkpoint() or {}).get("phase") != "second":
            ctx.progress(1, 2, "phase")
            ctx.save_checkpoint({"phase": "second"})
            raise stages.StageInterrupted()  # no signal: a fresh process is wanted for the next phase
        ctx.progress(2, 2, "phase")

    monkeypatch.setitem(stages.REGISTRY, "noop", two_phases)
    assert runner.run_stage(s.paths, "a") == 10 and s.progress("a")["done"] == 1
    assert runner.run_stage(s.paths, "a") == 0 and s.progress("a")["done"] == 2


def test_normal_stage_writes_no_ledger_entry(study):
    s = study()
    bind(s)
    runner.run_stage(s.paths, "a")
    assert not s.paths.ledger.exists()
