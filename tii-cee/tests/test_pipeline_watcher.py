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
from pipeline_harness import PACKAGE, SLOW, Study, stage, wait_until  # noqa: E402

from tiicee.pipeline import export_guard, watcher  # noqa: E402
from tiicee.pipeline.util import iso, now, sha256_file  # noqa: E402

STEPS = ["step-%04d" % step for step in range(6)]  # what SLOW, the paced no-op stage, works through
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
    assert (record["result"], record["path_taken"], record["skipped"]) == ("met", "met", [])
    assert record["inputs"] == [{"stage": "noop_a", "status": "COMPLETED"}]
    # The record is also published through the export guard, for the Mac and the ledger.
    assert json.loads((s.paths.approved / "gates" / "G_noop.json").read_text()) == record
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
    s.cluster.plan("tii-slow", [{"outcome": "kill", "state": "NODE_FAIL", "steps": 2}])
    s.cycle()
    s.advance()  # the node dies when two of the six items are done
    assert s.items("slow") == STEPS[:2]

    s.cycle()  # sees the node failure and resubmits from the checkpoint
    state = s.state("slow")
    assert (state["status"], state["retries"]["interrupt"]) == ("QUEUED", 1)
    s.advance()
    s.cycle()
    assert s.status("slow") == "COMPLETED"
    assert s.items("slow") == STEPS  # every key once: the first two were not done again
    checkpoint = json.loads((s.paths.stage_dir("slow") / "checkpoint.json").read_text())
    assert checkpoint == {"next": 6, "resumed_at": 2}  # the second attempt began at the checkpoint
    assert [a["state"] for a in s.state("slow")["attempts"]] == ["NODE_FAIL", "COMPLETED"]
    assert json.loads((s.paths.stage_out("slow") / "result.json").read_text())["count"] == 6


def test_warning_signal_checkpoints_and_the_continuation_is_not_a_retry(study):
    s = study([stage("slow", SLOW)])
    s.cluster.plan("tii-slow", [{"outcome": "usr1", "steps": 2}])
    s.cycle()
    s.advance()
    s.cycle()
    state = s.state("slow")
    assert (state["attempts"][0]["exit_code"], state["attempts"][0]["outcome"]) == (10, "continue")
    assert state["retries"] == {"interrupt": 0, "oom": 0, "other": 0} and state["status"] == "QUEUED"
    assert s.items("slow") == STEPS[:2]
    assert json.loads((s.paths.stage_dir("slow") / "checkpoint.json").read_text()) == {"next": 2, "resumed_at": 0}
    s.advance()
    s.cycle()
    assert s.status("slow") == "COMPLETED" and s.items("slow") == STEPS


def test_timeout_is_a_retry_only_when_the_checkpoint_did_not_move(study):
    s = study([stage("slow", SLOW)])
    s.cluster.plan("tii-slow", [{"outcome": "kill", "state": "TIMEOUT", "steps": 2}, "timeout"])
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
    attempts = [(a["stage"], a["attempt"], a["state"], a["exit_code"], a["elapsed_s"], a["gpu_hours"])
                for a in s.runlog("ended")]
    assert attempts == [("g", 1, "PREEMPTED", 0, 1800, 0.5), ("g", 2, "COMPLETED", 0, 3600, 1.0)]
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
    s.cluster.plan("tii-a", [{"outcome": "start", "steps": 1}, "complete"])
    s.cycle()
    s.advance()  # a is verified, has done one item, and is left running
    authorization = s.docs / "AUTHORIZATION.md"
    authorization.rename(s.docs / "AUTHORIZATION.md.revoked")

    s.cycle()
    assert (s.status("a"), s.status("b")) == ("RUNNING", "BLOCKED")
    assert [detail["code"] for _, detail in s.events("blocked")] == ["authorization_missing"]
    assert "authorization: authorization_missing" in s.paths.status.read_text()
    status = s.status_json()
    assert status["authorization"] == dict(status["authorization"], ok=False, sha256_ok=False, signature_ok=True)
    assert status["gpu"] == {"limit_by_account": {}, "in_use_by_account": {}}  # nothing is usable now
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
    release = s.root / "let-the-first-cycle-go"
    first = s.start_cycle(FAKE_SLURM_HOLD=str(release))
    wait_until(lambda: s.calls("squeue"), "the first cycle reaching its queue query")
    # The first cycle is now in the middle of its work, holding the lock. A second one starts.
    assert s.paths.lock.is_dir()
    assert "another cycle holds the lock" in s.cycle()
    assert len(s.calls("squeue")) == len(s.calls("sacct")) == 1 and not s.calls("sbatch")
    assert s.paths.lock.is_dir()  # the one that lost leaves the lock where it is

    release.touch()
    output = first.communicate(timeout=120)[0]
    assert first.returncode == 0 and "another cycle" not in output
    assert len(s.calls("squeue")) == len(s.calls("sacct")) == len(s.calls("sbatch")) == 1
    assert s.status("a") == "QUEUED" and not s.paths.lock.exists()


def test_fresh_lock_is_respected_and_a_stale_one_is_taken_and_recorded(study):
    s = study([stage("a")])
    s.paths.lock.mkdir(parents=True)
    assert "another cycle holds the lock" in s.cycle()
    assert not s.cluster.calls() and s.paths.lock.is_dir()  # the loser leaves the lock alone

    old = time.time() - 21 * 60
    os.utime(s.paths.lock, (old, old))
    taken = [line for line in s.cycle().splitlines() if "stale_lock taken" in line]
    assert len(taken) == 1 and int(taken[0].rsplit("age_s=", 1)[1]) >= 21 * 60
    assert s.status("a") == "QUEUED" and not s.paths.lock.exists()
    assert s.status_json()["watcher"]["lock"] == "stale_taken"

    s.paths.lock.mkdir()
    written = {path.name: path.read_bytes() for path in s.paths.state.iterdir() if path.is_file()}
    assert "another cycle holds the lock" in s.cycle()  # said in the watcher's own log, and only there
    assert {path.name: path.read_bytes() for path in s.paths.state.iterdir() if path.is_file()} == written
    assert len(s.calls("squeue")) == 1  # a cycle without the lock asks the scheduler nothing and writes no file
    s.paths.lock.rmdir()
    s.cycle()
    assert s.status_json()["watcher"] == dict(s.status_json()["watcher"], lock="ok", cycle=2)


# ---- the addendum -------------------------------------------------------------------------------

def test_unsigned_addendum_keeps_every_job_on_cai6734(study):
    s = study([stage("g%d" % index, gpus=1) for index in range(10)] + [stage("cpu")])
    s.cycle()
    jobs = s.cluster.jobs()
    assert [(job["account"], job["qos"]) for job in jobs if job["gpus"]] == [("cai6734", "cai6734")] * 8
    assert [(job["account"], job["qos"]) for job in jobs if not job["gpus"]] == [("cai6734", "cai6734-b")]
    assert (s.status("g8"), s.status("g9")) == ("PLANNED", "PLANNED")  # waiting for a GPU, not blocked
    assert "addendum_1: unsigned" in s.paths.status.read_text()
    status = s.status_json()
    assert status["gpu"] == {"limit_by_account": {"cai6734": 8}, "in_use_by_account": {"cai6734": 8}}
    assert status["authorization"]["addenda"] == [
        {"name": "AUTHORIZATION_ADDENDUM_1.md", "sha256_ok": True, "signed": False}]
    s.cycle()
    assert len(s.cluster.jobs()) == 9  # eight GPUs in use: nothing more until one frees


def test_signed_addendum_adds_cai5724_once_cai6734_is_full(study):
    s = study([stage("g%d" % index, gpus=1) for index in range(10)])
    s.sign_addendum()
    s.cycle()
    accounts = [(job["account"], job["qos"]) for job in s.cluster.jobs()]
    assert accounts == [("cai6734", "cai6734")] * 8 + [("cai5724", "cai5724")] * 2
    assert "addendum_1: signed" in s.paths.status.read_text()
    status = s.status_json()
    assert status["gpu"] == {"limit_by_account": {"cai6734": 8, "cai5724": 8},
                             "in_use_by_account": {"cai6734": 8, "cai5724": 2}}
    assert status["authorization"]["addenda"][0]["signed"] is True


def test_gpu_jobs_take_the_other_account_while_the_first_waits_on_its_group_limit(study):
    s = study([stage("g0", gpus=1), stage("g1", gpus=1), stage("prep"),
               stage("g2", gpus=1, depends_on=["prep"]), stage("g3", gpus=1, depends_on=["prep"])])
    s.sign_addendum()
    # The class holds every GPU of the cai6734 QOS: our job there waits on the group limit.
    s.cluster.plan("tii-g0", [{"outcome": "hold", "reason": "QOSGrpGRES"}] * 2)
    s.cluster.plan("tii-g1", [{"outcome": "hold"}] * 2)
    s.cycle()
    s.advance()
    s.cycle()  # g2 and g3 are ready now
    assert {job["name"][4:]: job["account"] for job in s.cluster.jobs()} == {
        "g0": "cai6734", "g1": "cai6734", "prep": "cai6734", "g2": "cai5724", "g3": "cai5724"}
    # Still one queue query per cycle; the reason is one more column of it.
    assert s.calls("squeue") == [["squeue"] + watcher.SQUEUE[1:]] * 2 and "Reason:" in watcher.SQUEUE[-1]


# What the live no-op run on HiPerGator showed on November 13, 2026: other work under the same
# account (jobs named v2-*, on another partition) asks for its GPU per job, so squeue's per-node
# column says N/A for it. Three such jobs ran on cai6734 and two on cai5724.
PER_JOB_GPU = {"per_node": "N/A", "per_job": "gres/gpu:1",
               "held": "cpu=6,mem=60G,node=1,billing=6,gres/gpu=1,gres/gpu:b200=1"}


def other_work(s):
    for name, account in (("v2-C2-s4", "cai6734"), ("v2-C2-s5", "cai6734"), ("v2-C3-s1", "cai6734"),
                          ("v2-C1-s2", "cai5724"), ("v2-C1-s3", "cai5724")):
        s.cluster.add_job(name, "RUNNING", account, account, cpus=6, **PER_JOB_GPU)
    s.cluster.add_job("v2-eval-C1", "PENDING", "cai6734", "cai6734-b", cpus=32, reason="Dependency",
                      held="cpu=32,mem=120G,node=1,billing=32")
    s.cluster.add_job("tii-watcher", "PENDING", "cai6734", "cai6734-b", reason="BeginTime",
                      held="cpu=1,mem=1G,node=1,billing=1")


def test_gpus_asked_for_per_job_by_other_work_count_against_the_account(study):
    s = study([stage("g%d" % index, gpus=1) for index in range(9)])
    s.sign_addendum()
    other_work(s)
    output = s.cycle()
    ours = [job["account"] for job in s.cluster.jobs() if job["name"].startswith("tii-g")]
    # 3 of cai6734's 8 GPUs and 2 of cai5724's are already held: 5 more fit on the first, then
    # 4 of the 6 left on the second. Reading only the per-node column would have put 8 on cai6734.
    assert ours == ["cai6734"] * 5 + ["cai5724"] * 4
    assert s.status_json()["gpu"] == {"limit_by_account": {"cai6734": 8, "cai5724": 8},
                                      "in_use_by_account": {"cai6734": 8, "cai5724": 6}}
    # One queue query, and the watcher's own queued job (tii-watcher) is not taken for a stage.
    assert len(s.calls("squeue")) == 1 and "adopted" not in output
    assert {line["stage"] for line in s.runlog()} == {"g%d" % index for index in range(9)}


def test_without_the_addendum_other_work_leaves_five_gpus(study):
    s = study([stage("g%d" % index, gpus=1) for index in range(7)])
    other_work(s)
    s.cycle()
    assert len(s.calls("sbatch")) == 5 and (s.status("g5"), s.status("g6")) == ("PLANNED", "PLANNED")
    assert s.status_json()["gpu"]["in_use_by_account"] == {"cai6734": 8}


def test_pending_gpu_requests_count_even_if_squeue_shows_no_tres_for_them(study):
    """The manual says tres-alloc shows a pending job's request. Should the installed squeue show
    nothing there, the per-node and per-job columns still carry it."""
    s = study([stage("g%d" % index, gpus=1) for index in range(10)])
    s.cluster.add_job("v2-wait", "PENDING", "cai6734", "cai6734", cpus=6, reason="Priority",
                      per_node="N/A", per_job="gres/gpu:2", held="N/A")
    s.cycle(FAKE_SLURM_PENDING_TRES="absent")
    assert len(s.calls("sbatch")) == 6  # 8 less the 2 another job is waiting for
    s.cycle(FAKE_SLURM_PENDING_TRES="absent")
    assert len(s.calls("sbatch")) == 6  # our own six, still pending, are counted from their per-node request


def test_cpu_jobs_take_another_qos_when_other_work_fills_the_burst_qos(study):
    s = study([stage("prep")])
    s.sign_addendum()
    for index in range(10):  # ten pending 32-CPU jobs: 320 of cai6734-b's 288 CPUs asked for
        s.cluster.add_job("v2-eval-%d" % index, "PENDING", "cai6734", "cai6734-b", cpus=32, reason="Dependency")
    s.cycle()
    assert [(job["account"], job["qos"]) for job in s.cluster.jobs() if job["name"] == "tii-prep"] == [("cai5724", "cai5724-b")]


def test_queue_that_cannot_be_read_stops_the_cycle_from_deciding(study):
    s = study([stage("g", gpus=1)])
    output = s.cycle(expect=1, FAKE_SLURM_FAIL="squeue_layout")  # lines in some other layout
    assert "squeue_unreadable lines=1" in output and "scheduler_error sacct=True squeue=False" in output
    assert not s.calls("sbatch") and "scheduler: error" in s.paths.status.read_text()
    s.cycle()
    assert s.status("g") == "QUEUED"


# squeue's answer on HiPerGator (Slurm 25.11.8) on November 13, 2026 to the watcher's own command,
# as the lead read it there: every line 956 characters wide; shown here with the padding squeezed.
LIVE_QUEUE = """\
51145811 PENDING cai6734 cai6734 6 N/A gres/gpu:1 cpu=6,mem=60G,node=1,billing=6,gres/gpu=1 QOSGrpMemLimit v2-cal-C2
51137362 RUNNING cai5724 cai5724 6 N/A gres/gpu:1 cpu=6,mem=60G,node=1,billing=6,gres/gpu=1,gres/gpu:b200=1 None v2-C1-s2
51150702 PENDING cai6734 cai6734-b 32 N/A N/A cpu=32,mem=400G,node=1,billing=32 Dependency v2-eval-C1
51144941 PENDING cai6734 cai6734-b 1 N/A N/A cpu=1,mem=1G,node=1,billing=1 BeginTime tii-watcher
"""


def test_lines_from_the_live_cluster_are_read_correctly():
    padded = ["".join("%-*s" % (width, cell) for (_, width), cell in zip(watcher.SQUEUE_FIELDS, line.split(" ")))
              for line in LIVE_QUEUE.splitlines()]
    assert {len(line) for line in padded} == {956}  # the width the cluster printed
    rows = watcher.parse_queue("\n".join(padded) + "\n")
    assert [(row["id"], row["state"], row["account"], row["qos"], row["cpus"], row["gpus"], row["reason"], row["name"])
            for row in rows] == [
        # A GPU asked for per job: N/A in the per-node column, found under tres-per-job and
        # tres-alloc. Pending, its tres-alloc is its request, so it counts before it runs.
        ("51145811", "PENDING", "cai6734", "cai6734", 6, 1, "QOSGrpMemLimit", "v2-cal-C2"),
        ("51137362", "RUNNING", "cai5724", "cai5724", 6, 1, "None", "v2-C1-s2"),
        ("51150702", "PENDING", "cai6734", "cai6734-b", 32, 0, "Dependency", "v2-eval-C1"),
        ("51144941", "PENDING", "cai6734", "cai6734-b", 1, 0, "BeginTime", "tii-watcher")]


def test_memory_group_limit_reroutes_gpu_jobs_like_the_gpu_limit(study):
    """The class GPU QOS also caps memory at 250 GB. Other work waiting on that cap (the live
    line above: QOSGrpMemLimit) means cai6734 cannot start another GPU job now."""
    s = study([stage("g%d" % index, gpus=1) for index in range(3)])
    s.sign_addendum()
    s.cluster.add_job("v2-cal-C2", "PENDING", "cai6734", "cai6734", cpus=6, reason="QOSGrpMemLimit",
                      per_job="gres/gpu:1", held="cpu=6,mem=60G,node=1,billing=6,gres/gpu=1")
    s.cycle()
    assert [job["account"] for job in s.cluster.jobs() if job["name"].startswith("tii-")] == ["cai5724"] * 3
    assert s.status_json()["gpu"]["in_use_by_account"] == {"cai6734": 1, "cai5724": 3}


def test_queue_lines_are_read_by_position():
    def line(job, state, account, qos, cpus, per_node, per_job, held, reason, name):
        cells = (job, state, account, qos, cpus, per_node, per_job, held, reason, name)
        return "".join("%-*.*s" % (width, width, cell) for (_, width), cell in zip(watcher.SQUEUE_FIELDS, cells))

    text = "\n".join([
        line("51138665", "RUNNING", "cai6734", "cai6734", "6", "N/A", "gres/gpu:1",
             "cpu=6,mem=60G,node=1,billing=6,gres/gpu=1,gres/gpu:b200=1", "None", "v2-C2-s4"),
        line("51150702", "PENDING", "cai6734", "cai6734-b", "32", "N/A", "N/A", "cpu=32,mem=120G,node=1,billing=32",
             "Dependency", "v2-eval-C1"),
        line("51152783", "PENDING", "cai5724", "cai5724-b", "1", "N/A", "N/A", "", "None", "tii-noop_c"),
        line("51160001_7", "PENDING", "cai6734", "cai6734", "4", "gres/gpu:rtx_pro_6000:1", "N/A", "N/A",
             "ReqNodeNotAvail, Reserved for maintenance", "a name with spaces"),
        ""]) + "\n"
    rows = watcher.parse_queue(text)
    assert [(row["id"], row["name"], row["state"], row["account"], row["qos"], row["cpus"], row["gpus"], row["reason"])
            for row in rows] == [
        ("51138665", "v2-C2-s4", "RUNNING", "cai6734", "cai6734", 6, 1, "None"),
        ("51150702", "v2-eval-C1", "PENDING", "cai6734", "cai6734-b", 32, 0, "Dependency"),
        ("51152783", "tii-noop_c", "PENDING", "cai5724", "cai5724-b", 1, 0, "None"),
        ("51160001_7", "a name with spaces", "PENDING", "cai6734", "cai6734", 4, 1,
         "ReqNodeNotAvail, Reserved for maintenance")]
    assert watcher.parse_queue("") == [] and watcher.parse_queue("\n\n") == []
    # Anything not laid out as asked makes the whole answer unusable, never a guess.
    for bad in ("51138665|v2-C2-s4|RUNNING|cai6734|cai6734|N/A|6|None",                  # the -o layout
                line("JOBID", "STATE", "ACCOUNT", "QOS", "CPUS", "", "", "", "REASON", "NAME"),  # a header
                line("51138665", "RUNNING", "cai6734", "cai6734", "six", "N/A", "N/A", "", "None", "x"),
                line("51138665", "RUNNING", "cai6734", "cai6734", "6", "N/A", "N/A", "", "None", "x")[40:]):
        assert watcher.parse_queue(text + bad + "\n") is None, bad[:40]
    assert watcher.SQUEUE[:5] == ["squeue", "--me", "-h", "-r", "-O"]
    assert watcher.SQUEUE[5].startswith("JobID:24,State:20,Account:40,QOS:40,NumCPUs:12,tres-per-node:100,")
    assert watcher.SQUEUE[5].endswith("tres-per-job:100,tres-alloc:300,Reason:120,Name:200")


def test_group_limit_changes_nothing_without_another_account_or_for_an_ordinary_wait(study):
    s = study([stage("g0", gpus=1), stage("prep"), stage("g2", gpus=1, depends_on=["prep"])])
    s.cluster.plan("tii-g0", [{"outcome": "hold", "reason": "QOSGrpGRES"}] * 2)
    s.cycle()
    s.advance()
    s.cycle()  # unsigned addendum: cai6734 is the only account, so the job queues there
    assert {job["name"]: job["account"] for job in s.cluster.jobs()}["tii-g2"] == "cai6734"


def test_ordinary_queue_wait_does_not_reroute(study):
    s = study([stage("g0", gpus=1), stage("prep"), stage("g2", gpus=1, depends_on=["prep"])])
    s.sign_addendum()
    s.cluster.plan("tii-g0", [{"outcome": "hold", "reason": "Priority"}] * 2)
    s.cycle()
    s.advance()
    s.cycle()  # waiting for priority is not a full QOS: cai6734 is still filled first
    assert {job["name"]: job["account"] for job in s.cluster.jobs()}["tii-g2"] == "cai6734"


TRAINING = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUBLAS_WORKSPACE_CONFIG": None}
EVALUATION = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def test_generated_scripts_follow_the_project_conventions(study):
    s = study([stage("g", gpus=1, time_hours=24, mem_gb=16, env=EVALUATION), stage("c", time_hours=0.5),
               stage("train", gpus=1, env=TRAINING), stage("plain", gpus=1)])
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

    # A stage runs with its own environment and no other stage's. Training, as the replication
    # trained: the allocator setting, and CUBLAS_WORKSPACE_CONFIG explicitly not set.
    train = s.script("train").splitlines()
    assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in train
    assert "unset CUBLAS_WORKSPACE_CONFIG" in train and not any("export CUBLAS" in line for line in train)
    assert not any("PYTORCH_CUDA_ALLOC_CONF" in line for line in gpu)
    # Nothing is exported for a GPU stage just because it is a GPU stage.
    assert not any("CUBLAS" in line or "PYTORCH" in line for line in s.script("plain").splitlines())


PROBE = """import json, os
from tiicee.pipeline.stages import register


@register("env_probe")
def env_probe(ctx):
    (ctx.out_dir / "env.json").write_text(json.dumps({name: os.environ.get(name) for name in ctx.args}))
"""


def test_stage_registered_by_a_listed_module_runs_with_its_own_environment(study):
    names = ["PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG", "CDM_WORK_DIR"]
    s = study()
    (s.paths.package / "probe_stages.py").write_text(PROBE)  # a module that registers a stage
    s.write_graph([stage("train", ["env_probe"] + names, env=TRAINING),
                   stage("evaluate", ["env_probe"] + names, env=EVALUATION),
                   stage("other", ["env_probe"] + names)], stage_modules=["probe_stages"])
    # What watcher.env would hold: the baseline work directory for every job, and (by mistake) a
    # CUBLAS setting that training must still not see.
    s.env.update(CDM_WORK_DIR="/blue/baseline", CUBLAS_WORKSPACE_CONFIG=":16:8")
    s.run_to_end()
    seen = {name: json.loads((s.paths.stage_out(name) / "env.json").read_text()) for name in ("train", "evaluate", "other")}
    assert seen["train"] == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                             "CUBLAS_WORKSPACE_CONFIG": None, "CDM_WORK_DIR": "/blue/baseline"}
    assert seen["evaluate"] == {"PYTORCH_CUDA_ALLOC_CONF": None, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                "CDM_WORK_DIR": "/blue/baseline"}
    assert seen["other"]["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"  # no map: only what the watcher's settings give
    # The environment is bound: the manifest records it.
    assert json.loads(s.paths.manifest("train").read_text())["env"] == TRAINING
    assert json.loads(s.paths.manifest("other").read_text())["env"] == {}


def test_stage_whose_module_is_not_listed_fails_and_is_not_run_blind(study):
    s = study()
    (s.paths.package / "probe_stages.py").write_text(PROBE)
    s.write_graph([stage("a", ["env_probe"])])  # the module is there but graph.json does not list it
    s.run_to_end()
    assert s.status("a") == "FAILED" and [a["exit_code"] for a in s.state("a")["attempts"]] == [1, 1]


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
    assert record["path_taken"] == "not_met" and export_guard.TIME.match(record["decided_at"])
    assert len(s.calls("sbatch")) == 2 and s.final()["kind"] == "study completed"
    assert "| SKIPPED BY GATE | 2 |" in s.paths.status.read_text()


def test_gate_records_are_published_through_the_export_guard(study):
    s = study([stage("a", ["noop", "--count", "0"]), stage("b", depends_on=["a"], gate="G"), stage("c")],
              [{"id": "G", "rule": "noop_gate.py", "inputs": ["a"], "reads": ["export/approved/a.json"]},
               {"id": "G-after", "rule": "noop_gate.py", "inputs": ["b", "c"], "reads": ["export/approved/c.json"]}])
    s.run_to_end()
    published = {path.name: json.loads(path.read_text()) for path in (s.paths.approved / "gates").iterdir()}
    assert sorted(published) == ["G-after.json", "G.json"]
    for name, record in published.items():
        assert record == json.loads(s.paths.gate_record(name[:-5]).read_text())
        export_guard.check_aggregate(record)
    # A gate whose input was itself skipped by a gate: the status is one of the fixed names.
    assert published["G-after.json"]["inputs"] == [{"stage": "b", "status": "SKIPPED BY GATE"},
                                                   {"stage": "c", "status": "COMPLETED"}]
    assert published["G-after.json"]["result"] == "met" and export_guard.surface_check(s.paths) == []
    # Published once: a record removed from the export tree by hand comes back, unchanged.
    (s.paths.approved / "gates" / "G.json").unlink()
    s.cycle()
    assert json.loads((s.paths.approved / "gates" / "G.json").read_text()) == published["G.json"]


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
    assert "scheduler_error sacct=True squeue=False" in s.cycle(expect=1, FAKE_SLURM_FAIL="squeue")
    assert not s.calls("sbatch") and not s.paths.stage_state("a").exists() and not s.runlog()
    assert "scheduler: error" in s.paths.status.read_text()
    assert s.status_json()["counts"]["PLANNED"] == 1  # still written, from what is on disk
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


def test_stage_that_would_share_the_watchers_job_name_is_refused_before_anything_runs(study):
    s = study([stage("watcher")])
    # The watcher's own scrontab job, as the queue shows it. Named tii-watcher, it is exactly what
    # the watcher would look for as the job of a stage called "watcher", and adopt.
    s.cluster.add_job("tii-watcher", "RUNNING", "cai6734", "cai6734-b")
    output = s.cycle(expect=1)
    assert "graph_invalid" in output and "adopted" not in output
    assert not s.calls("squeue") and not s.calls("sbatch") and not s.paths.stage_state("watcher").exists()
    assert [detail["code"] for _, detail in s.events("blocked")] == ["graph_invalid"]


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
    s.paths.events.write_text(events + "2026-11-13T03:00:00-05:00\tfailure\tthe patient in bed 4 had chest pain\n")
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
    assert sorted(path.name for path in s.paths.approved.iterdir()) == ["a.json", "gates"]
    assert [path.name for path in (s.paths.approved / "gates").iterdir()] == ["G.json"]  # the gate's own record
    assert len(list(s.paths.quarantine.iterdir())) == 2
    assert s.blocked()["surface"]["detail"] == {"quarantined": 2, "violations": 0}
    assert export_guard.surface_check(s.paths) == []


def test_pure_helpers():
    assert [watcher.gpus_in(text) for text in (
        "billing=4,cpu=4,gres/gpu=1,gres/gpu:rtx_pro_6000=1,mem=16G,node=1", "gres/gpu:rtx_pro_6000:1",
        "gpu:2", "gres:gpu:rtx_pro_6000:3", "gpu", "N/A", "billing=1,cpu=1,mem=1G,node=1", "",
        "cpu=6,mem=60G,node=1,billing=6,gres/gpu=1,gres/gpu:b200=1", "gres/gpu:1", "gres/gpu:b200=2",
        "cpu=32,mem=250G,node=1,billing=32,gres/gpu=4,gres/gpu:a100=2,gres/gpu:b200=2")] == [
        1, 1, 2, 3, 1, 0, 0, 0, 1, 1, 2, 4]
    entry = (PACKAGE / "tiicee" / "pipeline" / "scrontab_entry.txt").read_text()
    assert entry.splitlines()[-1].endswith(watcher.TAG) and entry.count("#SCRON") == 5
    assert watcher.remove_tagged(OTHER_ENTRIES_ABOVE + entry + OTHER_ENTRIES_BELOW) == OTHER_ENTRIES_ABOVE + OTHER_ENTRIES_BELOW
    assert watcher.remove_tagged(entry) == ""
    assert watcher.remove_tagged(OTHER_ENTRIES_BELOW) == OTHER_ENTRIES_BELOW
    assert [watcher.show(value) for value in (None, 2.0, 0.4583333, 0.0, 7, "E2")] == ["none", "2", "0.458", "0", "7", "E2"]


# ---- the monitor's files (monitor/DATA_CONTRACT.md) ---------------------------------------------

STAGE_FIELDS = {"id", "experiment", "kind", "status", "depends_on", "gate", "attempts", "job_id", "account",
                "gpus", "submitted_at", "started_at", "ended_at", "elapsed_s", "gpu_hours", "exit_code",
                "last_state", "estimate_gpu_hours"}


def is_time(text):
    return isinstance(text, str) and bool(export_guard.TIME.match(text))


def test_status_json_has_the_shape_of_the_data_contract(study):
    s = study([stage("prep"),
               stage("e8-seed1-part1", SLOW, experiment="E8", gpus=1, estimate=25.6, depends_on=["prep"]),
               stage("e8-eval", experiment="E8", gpus=1, estimate=2, depends_on=["e8-seed1-part1"], gate="G2a"),
               stage("big", experiment="E2", gpus=1, estimate=500)],  # more than E2's whole allowance
              [{"id": "G2a", "rule": "noop_gate.py", "inputs": ["e8-seed1-part1"], "reads": []}])
    s.cluster.plan("tii-e8-seed1-part1", [{"outcome": "start", "steps": 1, "elapsed": 3600},
                                          {"outcome": "complete", "elapsed": 7200}])
    s.cycle()
    s.advance()
    s.cycle()
    s.advance()  # the training part is now running, an hour in
    s.cycle()
    status = s.status_json()

    assert list(status) == sorted(["schema", "generated_at", "milestone", "watcher", "authorization", "gpu",
                                   "counts", "experiments", "stages", "gates", "blocked"])
    assert (status["schema"], status["milestone"]) == ("tii-status/1", "M0") and is_time(status["generated_at"])
    cycle = status["watcher"]
    assert set(cycle) == {"cycle", "started_at", "finished_at", "interval_s", "lock", "completed"}
    assert (cycle["cycle"], cycle["interval_s"], cycle["lock"], cycle["completed"]) == (3, 900, "ok", False)
    assert is_time(cycle["started_at"]) and is_time(cycle["finished_at"])
    assert status["authorization"] == {
        "ok": True, "sha256_ok": True, "signature_ok": True, "expires": "2027-03-31",
        "addenda": [{"name": "AUTHORIZATION_ADDENDUM_1.md", "sha256_ok": True, "signed": False}]}
    assert status["gpu"] == {"limit_by_account": {"cai6734": 8}, "in_use_by_account": {"cai6734": 1}}
    assert status["counts"] == {"PLANNED": 1, "BLOCKED": 1, "QUEUED": 0, "RUNNING": 1, "INTERRUPTED": 0,
                                "FAILED": 0, "REFUSED": 0, "COMPLETED": 1, "SKIPPED BY GATE": 0}
    assert status["experiments"] == [
        {"id": "benches", "status": "COMPLETED", "estimate_gpu_hours": 10, "allowance_gpu_hours": 30, "spent_gpu_hours": 0},
        {"id": "E8", "status": "RUNNING", "estimate_gpu_hours": 105, "allowance_gpu_hours": 210, "spent_gpu_hours": 1.0},
        {"id": "E2", "status": "BLOCKED", "estimate_gpu_hours": 90, "allowance_gpu_hours": 200, "spent_gpu_hours": 0}]

    rows = {row["id"]: row for row in status["stages"]}
    assert list(rows) == ["prep", "e8-seed1-part1", "e8-eval", "big"]
    assert all(set(row) == STAGE_FIELDS for row in rows.values())
    job = [j for j in s.cluster.jobs() if j["name"] == "tii-e8-seed1-part1"][0]
    running = rows["e8-seed1-part1"]
    assert {key: running[key] for key in STAGE_FIELDS - {"id", "submitted_at", "started_at"}} == {
        "experiment": "E8", "kind": "normal", "status": "RUNNING", "depends_on": ["prep"], "gate": None,
        "attempts": 1, "job_id": job["id"], "account": "cai6734", "gpus": 1, "ended_at": None,
        "elapsed_s": 3600, "gpu_hours": 1.0, "exit_code": None, "last_state": "RUNNING",
        "estimate_gpu_hours": 25.6}
    assert is_time(running["submitted_at"]) and running["started_at"] == iso(job["start"])
    done = rows["prep"]
    assert (done["status"], done["exit_code"], done["last_state"], done["attempts"], done["gpus"]) == (
        "COMPLETED", 0, "COMPLETED", 1, 0)
    assert all(is_time(done[key]) for key in ("submitted_at", "started_at", "ended_at"))
    never = rows["e8-eval"]
    assert [never[key] for key in ("status", "gate", "attempts", "job_id", "account", "submitted_at",
                                   "elapsed_s", "exit_code", "last_state", "gpu_hours")] == [
        "PLANNED", "G2a", 0, None, None, None, None, None, None, 0]
    assert rows["big"]["status"] == "BLOCKED"
    assert status["gates"] == [{"id": "G2a", "decided": False, "result": None, "path_taken": None, "decided_at": None}]
    assert [(b["id"], b["title"], b["status"]) for b in status["blocked"]] == [("BLOCKED-W001", "allowance:E2", "OPEN")]
    assert set(status["blocked"][0]) == {"id", "title", "opened_at", "status"} and is_time(status["blocked"][0]["opened_at"])
    assert "| E8 | 1 | 105 | 210 |" in s.paths.status.read_text()  # STATUS.md shows the same numbers
    assert export_guard.surface_check(s.paths) == []
    assert [path.name for path in s.paths.state.iterdir() if ".tmp" in path.name] == []  # renamed into place

    s.advance()  # the part completes; its gate is decided, not met with nothing to read
    s.cycle()
    status = s.status_json()
    part = {row["id"]: row for row in status["stages"]}["e8-seed1-part1"]
    assert (part["status"], part["exit_code"], part["last_state"], part["elapsed_s"], part["gpu_hours"]) == (
        "COMPLETED", 0, "COMPLETED", 7200, 2.0)
    assert is_time(part["ended_at"])
    gate = status["gates"][0]
    assert (gate["decided"], gate["result"], gate["path_taken"]) == (True, "not_met", "not_met") and is_time(gate["decided_at"])
    assert status["counts"]["SKIPPED BY GATE"] == 1
    # Nothing is left that can run: the watcher has finished, with E2 still blocked.
    assert status["watcher"]["completed"] is True and status["watcher"]["cycle"] == 4
    assert [e["status"] for e in status["experiments"]] == ["COMPLETED", "COMPLETED", "BLOCKED"]
    assert status["gpu"]["in_use_by_account"] == {"cai6734": 0}


def test_runlog_lines_follow_the_data_contract(study):
    s = study([stage("a", bind={"card": "$TII/private/card.json"}), stage("g", gpus=1, estimate=3),
               stage("x", depends_on=["g"], gate="G")],
              [{"id": "G", "rule": "noop_gate.py", "inputs": ["g"], "reads": []}])
    s.paths.private.mkdir(parents=True)
    (s.paths.private / "card.json").write_text("{}")
    s.cluster.plan("tii-g", [{"outcome": "preempt", "elapsed": 1800}, {"outcome": "complete", "elapsed": 3600}])
    s.cycle()
    (s.paths.private / "card.json").write_text('{"changed": 1}')
    for _ in range(2):
        s.advance()
        s.cycle()
    lines = s.runlog()
    assert all(is_time(line["time"]) for line in lines)
    assert {line["event"] for line in lines} == {"submitted", "started", "ended", "resubmitted", "refused",
                                                 "gate_decided", "blocked"}

    first, second = [job for job in s.cluster.jobs() if job["name"] == "tii-g"]
    described = {"stage": "g", "experiment": "benches", "account": "cai6734"}
    submitted = [line for line in lines if line["event"] == "submitted" and line["stage"] == "g"][0]
    resubmitted = [line for line in lines if line["event"] == "resubmitted"][0]
    assert [line for line in lines if line.get("stage") == "g"] == [
        dict(described, time=submitted["time"], event="submitted", attempt=1, job_id=first["id"]),
        # started and ended carry the scheduler's own times, not the time the watcher looked
        dict(described, time=iso(first["start"]), event="started", attempt=1, job_id=first["id"], state="RUNNING"),
        dict(described, time=iso(first["end"]), event="ended", attempt=1, job_id=first["id"], state="PREEMPTED",
             exit_code=0, elapsed_s=1800, gpu_hours=0.5),
        dict(described, time=resubmitted["time"], event="resubmitted", attempt=2, job_id=second["id"]),
        dict(described, time=iso(second["start"]), event="started", attempt=2, job_id=second["id"], state="RUNNING"),
        dict(described, time=iso(second["end"]), event="ended", attempt=2, job_id=second["id"], state="COMPLETED",
             exit_code=0, elapsed_s=3600, gpu_hours=1.0)]

    refused = [line for line in lines if line["event"] == "refused"]
    job = [j for j in s.cluster.jobs() if j["name"] == "tii-a"][0]
    assert refused == [{"time": iso(job["end"]), "event": "refused", "stage": "a", "experiment": "benches",
                        "attempt": 1, "job_id": job["id"], "account": "cai6734", "exit_code": 20}]
    record = json.loads(s.paths.gate_record("G").read_text())
    assert [line for line in lines if line["event"] == "gate_decided"] == [
        {"time": record["decided_at"], "event": "gate_decided", "gate": "G", "result": "not_met", "path_taken": "not_met"}]
    blocked = [line for line in lines if line["event"] == "blocked"]
    assert [{k: v for k, v in line.items() if k != "time"} for line in blocked] == [
        {"event": "blocked", "item": "BLOCKED-W001", "title": "refused:a", "stage": "a"}]


def test_running_stage_reports_progress_for_the_monitor(study):
    s = study([stage("slow", SLOW), stage("quick")])
    s.cluster.plan("tii-slow", [{"outcome": "usr1", "steps": 2}])
    s.cycle()
    s.advance()
    partial = s.progress("slow")
    assert set(partial) == {"stage", "unit", "done", "total", "updated_at"} and is_time(partial["updated_at"])
    # Two of six, written when the attempt ended although a minute had not passed.
    assert {key: partial[key] for key in ("stage", "unit", "done", "total")} == {
        "stage": "slow", "unit": "step", "done": 2, "total": 6}
    assert {key: s.progress("quick")[key] for key in ("stage", "unit", "done", "total")} == {
        "stage": "quick", "unit": "item", "done": 1, "total": 1}
    s.cycle()
    s.advance()
    s.cycle()
    assert s.progress("slow")["done"] == 6 and s.status("slow") == "COMPLETED"
    assert export_guard.surface_check(s.paths) == []


def test_surface_check_covers_the_monitors_files(study):
    s = study([stage("a")])
    s.run_to_end()
    assert export_guard.surface_check(s.paths) == []
    good = s.paths.status_json.read_text()

    def with_change(change):
        status = json.loads(good)
        change(status)
        s.paths.status_json.write_text(json.dumps(status))
        return export_guard.surface_check(s.paths)

    assert with_change(lambda st: st["stages"][0].update(note="seen at 3 am")) == ["status.json:1:forbidden_key"]
    assert with_change(lambda st: st["blocked"].append({"id": "x", "title": "the patient in bed 4"})) == [
        "status.json:1:string_not_identifier"]
    assert with_change(lambda st: st.pop("counts")) == ["status.json:1:status_shape"]
    assert with_change(lambda st: st.update(extra=1)) == ["status.json:1:status_shape"]
    assert with_change(lambda st: st.update(schema="tii-status/2")) == ["status.json:1:status_shape"]
    s.paths.status_json.write_text("{not json")
    assert export_guard.surface_check(s.paths) == ["status.json:1:not_json"]
    s.paths.status_json.write_text(good)

    report = {"stage": "b", "unit": "admission", "done": 3, "total": 9, "updated_at": now()}
    for name, content, code in (
            ("b.json", dict(report, done="three of nine so far"), "string_not_identifier"),
            ("b.json", dict(report, stage="a"), "progress_shape"),
            ("b.json", dict(report, total=None), "progress_shape"),
            ("b.json", dict(report, detail=1), "progress_shape"),
            ("two words.json", report, "file_name"), ("b.txt", report, "file_name")):
        (s.paths.progress / name).write_text(json.dumps(content))
        assert export_guard.surface_check(s.paths) == ["state/progress:1:%s" % code], name
        (s.paths.progress / name).unlink()
    (s.paths.progress / "b.json").write_text(json.dumps(report))
    assert export_guard.surface_check(s.paths) == []  # a well-formed report from another stage is fine

    # A write that was cut off for good leaves its unfinished file where writes are prepared,
    # outside the folders the watcher polices: nothing to move, nothing to raise.
    (s.paths.staging / ".a.json.tmp.4242").write_text('{"stage": "a", "un')
    s.cycle()
    assert not s.blocked() and not s.paths.quarantine.exists()

    (s.paths.progress / "rogue.json").write_text(json.dumps({"hpi": "65 year old with chest pain"}))
    s.cycle()  # the watcher moves it out of what the monitor reads and raises an item
    assert sorted(path.name for path in s.paths.progress.iterdir()) == ["a.json", "b.json"]
    assert s.blocked()["surface"]["detail"] == {"quarantined": 1, "violations": 0}
    assert s.status_json()["blocked"][-1]["title"] == "surface"
