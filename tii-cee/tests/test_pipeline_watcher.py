"""The watcher against a fake Slurm: WATCHER_DESIGN.md section 7, tests 1 to 6 and 8, and the retry table.

Each test builds a throwaway study (tests/fake_slurm/pipeline_harness.py) and runs real watcher
cycles as subprocesses. Set TII_TEST_WATCHER_PYTHON to a Python 3.9 to run the watcher on the
interpreter it meets on the cluster.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fake_slurm"))
from pipeline_harness import PACKAGE, Study, stage  # noqa: E402

from tiicee.pipeline import export_guard, watcher  # noqa: E402
from tiicee.pipeline.util import sha256_file  # noqa: E402

SLOW = ["noop_slow", "--steps", "6", "--sleep", "0.05"]
OTHER_ENTRIES_ABOVE = """#SCRON --job-name=nightly
#SCRON --time=00:05:00
0 3 * * * /home/someone/nightly.sh
"""
OTHER_ENTRIES_BELOW = """# a comment that stays
30 4 * * 1 /home/someone/weekly.sh    # weekly
"""


@pytest.fixture
def study(tmp_path):
    made = []

    def make(stages=(), gates=()):
        made.append(Study(tmp_path, stages, gates))
        return made[-1]

    yield make
    for one in made:
        one.close()


def install_scrontab(s):
    entry = (PACKAGE / "tiicee" / "pipeline" / "scrontab_entry.txt").read_text().replace("<TII>", str(s.root))
    s.cluster.crontab.write_text(OTHER_ENTRIES_ABOVE + entry + OTHER_ENTRIES_BELOW)


# ---- test 1: a three-stage no-op graph with a gate ----------------------------------------------

def test_noop_graph_runs_to_completion_and_disables_only_its_own_entry(study):
    graph = json.loads((PACKAGE / "tiicee" / "pipeline" / "noop_graph.json").read_text())
    s = study(graph["stages"], graph["gates"])
    install_scrontab(s)
    cycles = 0
    while not s.final():
        s.cycle()
        cycles += 1
        assert cycles < 12
        if not s.final():
            s.advance()

    assert [s.status(name) for name in ("noop_a", "noop_b", "noop_c")] == ["COMPLETED"] * 3
    record = json.loads(s.paths.gate_record("G_noop").read_text())
    assert (record["result"], record["path"], record["skipped"]) == ("met", "met", [])
    assert record["inputs"] == {"noop_a": "COMPLETED"}
    assert record["input_digests"] == [{"file": "export/approved/noop_a.json",
                                        "sha256": sha256_file(s.paths.approved / "noop_a.json")}]
    assert record["rule_sha256"] == sha256_file(s.paths.gate_rules / "noop_gate.py")
    assert not os.access(s.paths.gate_record("G_noop"), os.W_OK)  # immutable
    assert [kind for kind, _ in s.events()] == ["stage completed", "gate decided", "stage completed",
                                                "stage completed", "study completed"]

    # UF's rule: one accounting query and one queue query per cycle, nothing else but submissions.
    assert len(s.calls("sacct")) == len(s.calls("squeue")) == cycles
    assert len(s.calls("sbatch")) == 3 and not s.calls("scontrol")
    # Only at completion: list, reinstall without the tagged entry, verify.
    assert [call[1:] for call in s.calls("scrontab")] == [["-l"], [str(s.paths.state / "scrontab.new")], ["-l"]]
    assert s.cluster.crontab.read_text() == OTHER_ENTRIES_ABOVE + OTHER_ENTRIES_BELOW
    assert s.final() == dict(s.final(), kind="study completed", scrontab_disabled=True)

    s.cycle()  # a later cycle changes nothing and leaves scrontab alone
    assert len(s.events()) == 5 and len(s.calls("scrontab")) == 3
    assert export_guard.surface_check(s.paths) == []
    assert not s.paths.lock.exists()


def test_entry_that_could_not_be_disabled_is_reported_and_tried_again(study):
    s = study([stage("a")])
    install_scrontab(s)
    s.cycle()
    s.advance()
    s.cycle(FAKE_SLURM_FAIL="scrontab")  # the study completes but the new crontab is not accepted
    assert s.final()["kind"] == "study completed" and s.final()["scrontab_disabled"] is False
    assert watcher.TAG in s.cluster.crontab.read_text()
    s.cycle(FAKE_SLURM_FAIL="scrontab")
    assert s.blocked()["scrontab"]["detail"] == {"code": "entry_not_disabled"}
    assert len(s.events("study completed")) == 1
    s.cycle()
    assert s.final()["scrontab_disabled"] is True
    assert s.cluster.crontab.read_text() == OTHER_ENTRIES_ABOVE + OTHER_ENTRIES_BELOW
    s.cycle()
    assert not s.blocked()


def test_watcher_cycle_script_runs_one_cycle_from_its_own_settings(study):
    s = study([stage("a")])
    script = s.root / "pipeline" / "watcher_cycle.sh"
    shutil.copy(PACKAGE / "tiicee" / "pipeline" / "watcher_cycle.sh", script)
    settings = {"TII_ROOT": s.root, "TII_CODE": s.code, "TII_PYTHON": sys.executable,
                "TII_WATCHER_PYTHON": s.env.get("TII_TEST_WATCHER_PYTHON") or sys.executable, "HF_HOME": "/cache/hf"}
    (s.root / "pipeline" / "watcher.env").write_text(
        "".join("%s=%s\n" % (key, shlex.quote(str(value))) for key, value in settings.items()))
    bare = {key: value for key, value in s.env.items() if not key.startswith("TII_")}
    result = subprocess.run(["bash", str(script)], env=bare, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "watcher cycle start" in lines[0] and lines[-1].endswith("watcher cycle end rc=0")
    assert s.status("a") == "QUEUED"
    assert s.cluster.jobs()[0]["env"]["HF_HOME"] == "/cache/hf"  # the jobs inherit watcher.env


# ---- test 2: a stage killed mid-run -------------------------------------------------------------

def test_killed_stage_resumes_from_its_checkpoint_and_repeats_no_key(study):
    s = study([stage("slow", SLOW)])
    items = s.paths.stage_dir("slow") / "items.jsonl"
    s.cluster.plan("tii-slow", [{"outcome": "kill", "state": "NODE_FAIL", "wait_file": str(items), "wait_lines": 2}])
    s.cycle()
    s.advance()
    before = s.items("slow")
    assert 2 <= len(before) < 6

    s.cycle()  # sees the node failure and resubmits from the checkpoint
    state = s.state("slow")
    assert (state["status"], state["retries"]["interrupt"]) == ("QUEUED", 1)
    s.advance()
    s.cycle()
    assert s.status("slow") == "COMPLETED"
    keys = s.items("slow")
    assert keys == ["step-%04d" % step for step in range(6)]  # every key once
    assert keys[:len(before)] == before                         # and none of the earlier ones again
    checkpoint = json.loads((s.paths.stage_dir("slow") / "checkpoint.json").read_text())
    assert checkpoint["next"] == 6 and len(before) - 1 <= checkpoint["resumed_at"] <= len(before)
    assert [a["state"] for a in s.state("slow")["attempts"]] == ["NODE_FAIL", "COMPLETED"]
    assert json.loads((s.paths.stage_out("slow") / "result.json").read_text())["count"] == 6


def test_warning_signal_checkpoints_and_the_continuation_is_not_a_retry(study):
    s = study([stage("slow", SLOW)])
    items = s.paths.stage_dir("slow") / "items.jsonl"
    s.cluster.plan("tii-slow", [{"outcome": "usr1", "wait_file": str(items), "wait_lines": 2}])
    s.cycle()
    s.advance()
    s.cycle()
    state = s.state("slow")
    assert (state["attempts"][0]["exit_code"], state["attempts"][0]["outcome"]) == (10, "continue")
    assert state["retries"] == {"interrupt": 0, "oom": 0, "other": 0} and state["status"] == "QUEUED"
    assert 2 <= len(s.items("slow")) < 6
    s.advance()
    s.cycle()
    assert s.status("slow") == "COMPLETED" and s.items("slow") == ["step-%04d" % step for step in range(6)]


def test_timeout_is_a_retry_only_when_the_checkpoint_did_not_move(study):
    s = study([stage("slow", SLOW)])
    items = s.paths.stage_dir("slow") / "items.jsonl"
    s.cluster.plan("tii-slow", [{"outcome": "kill", "state": "TIMEOUT", "wait_file": str(items), "wait_lines": 2},
                                "timeout"])
    s.cycle()
    s.advance()
    s.cycle()
    assert s.state("slow")["retries"]["interrupt"] == 0
    s.advance()
    s.cycle()
    state = s.state("slow")
    assert state["retries"]["interrupt"] == 1
    assert [a["outcome"] for a in state["attempts"]] == ["continue", "interrupt"]


# ---- the retry table ----------------------------------------------------------------------------

def test_classification_table():
    table = [(("COMPLETED", 0, False), "done"), (("FAILED", 10, False), "continue"),
             (("TIMEOUT", 0, True), "continue"), (("TIMEOUT", 0, False), "interrupt"),
             (("PREEMPTED", 0, True), "interrupt"), (("NODE_FAIL", 0, False), "interrupt"),
             (("OUT_OF_MEMORY", 0, False), "oom"), (("FAILED", 40, False), "oom"),
             (("FAILED", 20, False), "refused"), (("FAILED", 30, False), "refused"),
             (("FAILED", 1, False), "other"), (("CANCELLED", 0, False), "other"), (("MISSING", 0, False), "other")]
    for arguments, outcome in table:
        assert watcher.classify(*arguments) == outcome, arguments
    assert watcher.RETRIES == {"interrupt": 3, "oom": 1, "other": 1}


def test_three_retries_for_interruptions_then_failed_and_only_dependents_wait(study):
    s = study([stage("a"), stage("b", depends_on=["a"]), stage("c")])
    s.cluster.plan("tii-a", ["preempt", "node_fail", "timeout", "preempt"])
    for _ in range(4):
        s.cycle()
        s.advance()
    s.cycle()
    a = s.state("a")
    assert (a["status"], a["retries"]["interrupt"], len(a["attempts"])) == ("FAILED", 3, 4)
    assert s.status("b") == "PLANNED" and s.status("c") == "COMPLETED"
    assert len(s.calls("sbatch")) == 5
    failures = s.events("failure")
    assert len(failures) == 1 and failures[0][1]["stage"] == "a" and failures[0][1]["state"] == "PREEMPTED"
    assert s.final()["kind"] == "all remaining items blocked"
    assert [kind for kind, _ in s.events()][-1] == "all remaining items blocked"


def test_out_of_memory_is_retried_once_with_half_again_the_memory(study):
    s = study([stage("a", mem_gb=16)])
    s.cluster.plan("tii-a", ["oom", "oom"])
    s.cycle()
    s.advance()
    s.cycle()
    assert "#SBATCH --mem=16G" in s.script("a", 1) and "#SBATCH --mem=24G" in s.script("a", 2)
    s.advance()
    s.cycle()
    assert s.status("a") == "FAILED" and len(s.calls("sbatch")) == 2


def test_any_other_failure_is_retried_once(study):
    s = study([stage("a"), stage("b")])
    s.cluster.plan("tii-a", ["fail", "fail"])
    s.cluster.plan("tii-b", ["fail"])
    for _ in range(2):
        s.cycle()
        s.advance()
    s.cycle()
    assert s.status("a") == "FAILED" and len(s.state("a")["attempts"]) == 2
    assert s.status("b") == "COMPLETED" and s.state("b")["retries"]["other"] == 1


def test_every_attempt_is_logged_with_its_gpu_hours_and_charged(study):
    s = study([stage("g", gpus=1, estimate=5)])
    s.cluster.plan("tii-g", [{"outcome": "preempt", "elapsed": 1800}, {"outcome": "complete", "elapsed": 3600}])
    for _ in range(2):
        s.cycle()
        s.advance()
    s.cycle()
    attempts = [(a["stage"], a["state"], a["exit_code"], a["elapsed_s"], a["gpu_hours"]) for a in s.runlog("attempt")]
    assert attempts == [("g", "PREEMPTED", 0, 1800, 0.5), ("g", "COMPLETED", 0, 3600, 1.0)]
    assert s.state("g")["gpu_hours"] == 1.5
    assert "| benches | 1.5 | 10 | 30 |" in s.paths.status.read_text()
    assert s.events("stage completed")[0][1]["gpu_hours"] == "1.5"


# ---- test 3: a bound artifact changes -----------------------------------------------------------

def test_changed_bound_artifact_is_refused_never_retried_and_blocked(study):
    s = study([stage("a", bind={"calibration": "$TII/private/artifacts/calibration.json"}),
               stage("b", depends_on=["a"])])
    artifact = s.paths.private / "artifacts" / "calibration.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"scale": 1.0}')
    s.cycle()  # binds the manifest and submits
    artifact.write_text('{"scale": 2.0}')
    s.advance()
    s.cycle()
    state = s.state("a")
    assert state["status"] == "REFUSED" and state["attempts"][0]["exit_code"] == 20
    assert state["refusal"] == {"exit": 20, "reason": "bound_changed", "role": "calibration"}
    s.cycle()
    s.cycle()
    assert len(s.calls("sbatch")) == 1 and s.status("b") == "PLANNED"
    item = s.blocked()["refused:a"]
    assert [detail for _, detail in s.events("blocked")] == [
        {"item": item["id"], "key": "refused:a", "stage": "a", "exit": "20",
         "reason": "bound_changed", "role": "calibration"}]
    assert "| %s | refused:a |" % item["id"] in s.paths.status.read_text()
    assert item["id"] in s.paths.blocked_md.read_text()
    assert s.final()["kind"] == "all remaining items blocked"


# ---- test 4: an allowance below the next estimate -----------------------------------------------

def test_allowance_below_the_next_estimate_blocks_with_the_numbers(study):
    s = study([stage("a", gpus=1, estimate=3), stage("b", gpus=1, estimate=4, depends_on=["a"])])
    s.set_allowance("benches", 5)
    s.cluster.plan("tii-a", [{"outcome": "complete", "elapsed": 7200}])  # 2 GPU-hours
    s.cycle()
    s.advance()
    s.cycle()
    assert s.state("a")["gpu_hours"] == 2.0 and s.status("b") == "BLOCKED"
    assert len(s.calls("sbatch")) == 1
    assert s.blocked()["allowance:benches"]["detail"] == {
        "experiment": "benches", "stage": "b", "spent": 2.0, "outstanding": 0.0, "estimate": 4.0, "allowance": 5.0}
    event = s.events("blocked")[0][1]
    assert (event["spent"], event["estimate"], event["allowance"]) == ("2", "4", "5")
    assert "| benches | 2 | 10 | 5 |" in s.paths.status.read_text()
    assert s.final()["kind"] == "all remaining items blocked"

    s.cycle()  # asked once: no second event, still no submission
    assert len(s.events("blocked")) == 1 and len(s.calls("sbatch")) == 1

    s.set_allowance("benches", 10)  # as a new signed allowance would
    s.cycle()
    assert s.status("b") == "QUEUED" and not s.blocked() and s.final() is None


def test_estimates_of_stages_under_way_count_against_the_allowance(study):
    s = study([stage("a", gpus=1, estimate=3), stage("b", gpus=1, estimate=3)])
    s.set_allowance("benches", 5)
    s.cycle()
    assert (s.status("a"), s.status("b")) == ("QUEUED", "BLOCKED")
    assert s.blocked()["allowance:benches"]["detail"]["outstanding"] == 3


def test_experiment_the_authorization_does_not_name_is_not_submitted(study):
    s = study([stage("a", experiment="E11")])
    s.cycle()
    assert s.status("a") == "BLOCKED" and not s.calls("sbatch")


# ---- test 5: AUTHORIZATION.md renamed -----------------------------------------------------------

def test_without_authorization_nothing_new_is_submitted_and_restoring_it_resumes(study):
    s = study([stage("a", SLOW), stage("b", depends_on=["a"])])
    items = s.paths.stage_dir("a") / "items.jsonl"
    s.cluster.plan("tii-a", [{"outcome": "start", "wait_file": str(items), "wait_lines": 1}, "complete"])
    s.cycle()
    s.advance()  # a is verified, has made progress, and is left running
    authorization = s.docs / "AUTHORIZATION.md"
    authorization.rename(s.docs / "AUTHORIZATION.md.revoked")

    s.cycle()
    assert (s.status("a"), s.status("b")) == ("RUNNING", "BLOCKED")
    assert [detail["code"] for _, detail in s.events("blocked")] == ["authorization_missing"]
    assert "authorization: authorization_missing" in s.paths.status.read_text()
    s.cycle()
    assert len(s.events("blocked")) == 1  # one event, not one per cycle

    s.advance()  # the running job finishes its part
    s.cycle()
    assert (s.status("a"), s.status("b")) == ("COMPLETED", "BLOCKED")
    assert len(s.calls("sbatch")) == 1
    assert s.final()["kind"] == "all remaining items blocked"

    (s.docs / "AUTHORIZATION.md.revoked").rename(authorization)
    s.cycle()
    assert s.status("b") == "QUEUED" and not s.blocked() and s.final() is None
    s.advance()
    s.cycle()
    assert s.status("b") == "COMPLETED" and s.final()["kind"] == "study completed"


def test_changed_authorization_or_missing_signature_blocks_too(study):
    s = study([stage("a")])
    (s.docs / "SIGNATURE.md").unlink()
    s.cycle()
    assert s.blocked()["authorization"]["detail"] == {"code": "signature_missing"}
    (s.docs / "SIGNATURE.md").write_text("no hash here\n")
    with open(s.docs / "AUTHORIZATION.md", "a") as handle:
        handle.write("one more line\n")
    s.cycle()
    assert s.blocked()["authorization"]["detail"] == {"code": "authorization_changed"}
    assert s.status("a") == "BLOCKED" and not s.calls("sbatch")


# ---- test 6: two cycles at once -----------------------------------------------------------------

def test_two_cycles_at_once_one_exits_on_the_lock(study):
    s = study([stage("a")])
    first = s.start_cycle(FAKE_SLURM_DELAY="1.0")
    second = s.start_cycle(FAKE_SLURM_DELAY="1.0")
    outputs = [first.communicate(timeout=60)[0], second.communicate(timeout=60)[0]]
    assert (first.returncode, second.returncode) == (0, 0)
    assert sum("another cycle holds the lock" in output for output in outputs) == 1
    assert len(s.calls("squeue")) == len(s.calls("sacct")) == len(s.calls("sbatch")) == 1
    assert not s.paths.lock.exists()


def test_fresh_lock_is_respected_and_a_stale_one_is_taken_and_recorded(study):
    s = study([stage("a")])
    s.paths.lock.mkdir(parents=True)
    assert "another cycle holds the lock" in s.cycle()
    assert not s.cluster.calls() and s.paths.lock.is_dir()  # the loser leaves the lock alone

    old = time.time() - 21 * 60
    os.utime(s.paths.lock, (old, old))
    s.cycle()
    assert s.status("a") == "QUEUED" and not s.paths.lock.exists()
    assert s.runlog("stale_lock")[0]["age_s"] >= 20 * 60


# ---- the addendum -------------------------------------------------------------------------------

def test_unsigned_addendum_keeps_every_job_on_cai6734(study):
    s = study([stage("g%d" % index, gpus=1) for index in range(10)] + [stage("cpu")])
    s.cycle()
    jobs = s.cluster.jobs()
    assert [(job["account"], job["qos"]) for job in jobs if job["gpus"]] == [("cai6734", "cai6734")] * 8
    assert [(job["account"], job["qos"]) for job in jobs if not job["gpus"]] == [("cai6734", "cai6734-b")]
    assert (s.status("g8"), s.status("g9")) == ("PLANNED", "PLANNED")  # waiting for a GPU, not blocked
    assert "addendum_1: unsigned" in s.paths.status.read_text()
    s.cycle()
    assert len(s.cluster.jobs()) == 9  # eight GPUs in use: nothing more until one frees


def test_signed_addendum_adds_cai5724_once_cai6734_is_full(study):
    s = study([stage("g%d" % index, gpus=1) for index in range(10)])
    s.sign_addendum()
    s.cycle()
    accounts = [(job["account"], job["qos"]) for job in s.cluster.jobs()]
    assert accounts == [("cai6734", "cai6734")] * 8 + [("cai5724", "cai5724")] * 2
    assert "addendum_1: signed" in s.paths.status.read_text()


def test_generated_scripts_follow_the_project_conventions(study):
    s = study([stage("g", gpus=1, time_hours=24, mem_gb=16), stage("c", time_hours=0.5)])
    s.cycle()
    gpu, cpu = s.script("g").splitlines(), s.script("c").splitlines()
    for line in ("#SBATCH --job-name=tii-g", "#SBATCH --account=cai6734", "#SBATCH --qos=cai6734",
                 "#SBATCH --partition=hpg-rtx6000", "#SBATCH --gres=gpu:rtx_pro_6000:1",
                 "#SBATCH --time=24:00:00", "#SBATCH --mem=16G", "#SBATCH --signal=B:USR1@600",
                 "#SBATCH --no-requeue", "#SBATCH --output=%s/tii-g-%%j.log" % s.paths.logs,
                 "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
                 "exec %s -m tiicee.pipeline.stage g" % shlex.quote(sys.executable)):
        assert line in gpu, line
    assert "#SBATCH --qos=cai6734-b" in cpu and "#SBATCH --time=00:30:00" in cpu
    assert not any("gres" in line or "partition" in line or "CUBLAS" in line for line in cpu)


# ---- gates --------------------------------------------------------------------------------------

def test_gate_not_met_skips_its_stages_and_their_dependents(study):
    s = study([stage("a", ["noop", "--count", "0"]), stage("b", depends_on=["a"], gate="G"),
               stage("c", depends_on=["b"]), stage("d")],
              [{"id": "G", "rule": "noop_gate.py", "inputs": ["a"], "reads": ["export/approved/a.json"]}])
    s.run_to_end()
    assert [s.status(name) for name in "abcd"] == ["COMPLETED", "SKIPPED BY GATE", "SKIPPED BY GATE", "COMPLETED"]
    record = json.loads(s.paths.gate_record("G").read_text())
    assert (record["result"], record["skipped"]) == ("not_met", ["b", "c"])
    assert s.state("c")["skipped_by"] == "G"
    assert s.events("gate decided") == [("gate decided", {"gate": "G", "result": "not_met", "path": "not_met", "skipped": "2"})]
    assert len(s.calls("sbatch")) == 2 and s.final()["kind"] == "study completed"
    assert "| SKIPPED BY GATE | 2 |" in s.paths.status.read_text()


def test_gate_whose_rule_does_not_decide_is_a_blocked_item(study):
    s = study([stage("a"), stage("b", depends_on=["a"], gate="G")],
              [{"id": "G", "rule": "broken.py", "inputs": ["a"], "reads": []}])
    (s.paths.gate_rules / "broken.py").write_text("import sys\nsys.exit(3)\n")
    s.run_to_end()
    assert s.status("b") == "PLANNED" and not s.paths.gate_record("G").exists()
    assert s.blocked()["gate:G"]["detail"] == {"gate": "G", "code": "rule_failed"}
    assert s.final()["kind"] == "all remaining items blocked"


# ---- when the scheduler or the bookkeeping misbehaves -------------------------------------------

def test_scheduler_query_failure_changes_nothing(study):
    s = study([stage("a")])
    s.cycle(expect=1, FAKE_SLURM_FAIL="squeue")
    assert not s.calls("sbatch") and not s.paths.stage_state("a").exists()
    assert "scheduler: error" in s.paths.status.read_text()
    assert s.runlog("scheduler_error") == [dict(s.runlog("scheduler_error")[0], sacct=True, squeue=False)]
    s.cycle()
    assert s.status("a") == "QUEUED"


def test_job_already_in_the_queue_is_adopted_not_submitted_twice(study):
    s = study([stage("a")])
    s.cycle()
    s.paths.stage_state("a").unlink()  # as if the watcher died between sbatch and writing its state
    s.cycle()
    assert len(s.calls("sbatch")) == 1 and s.state("a")["job_id"] == s.cluster.jobs()[0]["id"]
    s.advance()
    s.cycle()
    assert s.status("a") == "COMPLETED"


def test_job_unknown_to_the_scheduler_becomes_a_failure_after_an_hour(study):
    s = study([stage("a")])
    s.cycle()
    s.cluster._locked(lambda state: state["jobs"].clear())
    for _ in range(watcher.MISSING_CYCLES):
        s.cycle()
    state = s.state("a")
    assert state["attempts"][0]["state"] == "MISSING" and state["retries"]["other"] == 1
    assert state["status"] == "QUEUED" and len(s.calls("sbatch")) == 2


def test_refused_submission_is_reported_and_tried_again(study):
    s = study([stage("a")])
    s.cluster.plan("tii-a", ["reject"])
    s.cycle()
    assert s.status("a") == "PLANNED" and s.blocked()["submit:a"]["detail"] == {"stage": "a", "rc": 1}
    s.cycle()
    assert s.status("a") == "QUEUED" and not s.blocked()


def test_unreadable_graph_is_reported_once(study):
    s = study([stage("a", time_hours=25)])
    s.cycle(expect=1)
    s.cycle(expect=1)
    assert [detail["code"] for _, detail in s.events("blocked")] == ["graph_invalid"]
    assert not s.calls("sbatch")


# ---- test 8: the surface the Mac reads ----------------------------------------------------------

def test_surface_holds_names_and_numbers_only_and_rejects_free_text(study):
    s = study([stage("a", ["noop", "--count", "0"]), stage("b", depends_on=["a"], gate="G"),
               stage("bad", bind={"card": "$TII/private/card.json"}), stage("g", gpus=1, estimate=40)],
              [{"id": "G", "rule": "noop_gate.py", "inputs": ["a"], "reads": ["export/approved/a.json"]}])
    (s.paths.private).mkdir(parents=True)
    (s.paths.private / "card.json").write_text("{}")
    s.cluster.plan("tii-bad", ["fail", "fail"])
    s.run_to_end()
    assert {kind for kind, _ in s.events()} == {"stage completed", "gate decided", "blocked", "failure",
                                                "all remaining items blocked"}
    assert export_guard.surface_check(s.paths) == []

    events, status = s.paths.events.read_text(), s.paths.status.read_text()
    count = len(events.splitlines())
    s.paths.events.write_text(events + "2026-11-13T03:00:00-0500\tfailure\tthe patient in bed 4 had chest pain\n")
    assert export_guard.surface_check(s.paths) == ["events.tsv:%d:not_an_event" % (count + 1)]
    s.paths.events.write_text(events + "a line with no fields\n")
    assert export_guard.surface_check(s.paths) == ["events.tsv:%d:not_an_event" % (count + 1)]
    s.paths.events.write_text(events)

    lines = len(status.split("\n"))
    s.paths.status.write_text(status + "Patient 10023 presented with chest pain.\n")
    assert export_guard.surface_check(s.paths) == ["STATUS.md:%d:not_a_status_line" % lines]
    s.paths.status.write_text(status + "| a | COMPLETED | presented with chest pain |\n")
    assert export_guard.surface_check(s.paths) == ["STATUS.md:%d:not_a_status_line" % lines]
    s.paths.status.write_text(status)

    (s.paths.approved / "rows.json").write_text(json.dumps({"hadm_id": 1}))
    (s.paths.approved / "notes.txt").write_text("free text\n")
    assert export_guard.surface_check(s.paths) == ["export/approved:1:file_name", "export/approved:2:forbidden_key"]

    s.cycle()  # the watcher moves both out of the tree the Mac copies and raises an item
    assert sorted(path.name for path in s.paths.approved.iterdir()) == ["a.json"]
    assert len(list(s.paths.quarantine.iterdir())) == 2
    assert s.blocked()["surface"]["detail"] == {"quarantined": 2, "violations": 0}
    assert export_guard.surface_check(s.paths) == []


def test_pure_helpers():
    assert [watcher.gpus_in(text) for text in (
        "billing=4,cpu=4,gres/gpu=1,gres/gpu:rtx_pro_6000=1,mem=16G,node=1", "gres/gpu:rtx_pro_6000:1",
        "gpu:2", "gres:gpu:rtx_pro_6000:3", "gpu", "N/A", "billing=1,cpu=1,mem=1G,node=1", "")] == [1, 1, 2, 3, 1, 0, 0, 0]
    entry = (PACKAGE / "tiicee" / "pipeline" / "scrontab_entry.txt").read_text()
    assert entry.splitlines()[-1].endswith(watcher.TAG) and entry.count("#SCRON") == 5
    assert watcher.remove_tagged(OTHER_ENTRIES_ABOVE + entry + OTHER_ENTRIES_BELOW) == OTHER_ENTRIES_ABOVE + OTHER_ENTRIES_BELOW
    assert watcher.remove_tagged(entry) == ""
    assert watcher.remove_tagged(OTHER_ENTRIES_BELOW) == OTHER_ENTRIES_BELOW
    assert [watcher.show(value) for value in (None, 2.0, 0.4583333, 0.0, 7, "E2")] == ["none", "2", "0.458", "0", "7", "E2"]
