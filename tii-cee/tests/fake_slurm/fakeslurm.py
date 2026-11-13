"""A small fake Slurm over a state directory, for the pipeline tests. Nothing here contacts a cluster.

The executables beside this file (sbatch, squeue, sacct, scrontab, scontrol) call into it with the
directory named by FAKE_SLURM_DIR. sbatch only records a job; nothing runs until a test advances
the cluster with Cluster.run() (or `scontrol run`), which applies the next planned outcome to each
pending or running job:

    complete   run the job script to its end; COMPLETED on exit 0, else FAILED with its exit code
               (the default)
    usr1       let it reach step "steps", send SIGUSR1 (the ten-minute warning), let it exit
    kill       let it reach step "steps", SIGKILL it; "state" names what Slurm reports
               (NODE_FAIL by default, or TIMEOUT, PREEMPTED, ...)
    start      let it reach step "steps" and leave it there, alive, reported RUNNING; the next
               advance applies the next outcome to the same process. Only through Cluster.run()
               on one Cluster object, which holds the process
    timeout, preempt, node_fail, oom, fail, cancel
               end the job in that state without running it
    hold       leave it PENDING for this advance; "reason" sets what squeue shows for it
               (QOSGrpGRES when the QOS's shared GPUs are all in use)

usr1, kill and start act on a job in the middle of its work, so they need a job that cannot run
ahead of the test: a stage paced by permission (noop_slow --pace <Cluster.pace(job name)>). The
cluster grants it exactly "steps" steps and acts once the job says it is waiting for the next.
Nothing here depends on how fast the machine is. A job that does not get there in two minutes,
or ends first, is an error in the test and raises.

squeue answers `-O field:width,...` as the real one does: fixed-width columns, no separator.
add_job() puts another job of the same user in the queue (work that is not the study's), with
whatever TRES strings a test needs: a GPU asked for per job shows N/A under tres-per-node.

An outcome is a name or {"outcome": name, "steps": n, "state": ..., "elapsed": seconds}; "elapsed"
is what sacct reports (for "start": the time run so far). Planning "reject" makes the next sbatch
for that job name fail. With FAKE_SLURM_HOLD=<file> in its environment, squeue does not answer
until that file exists, so a test can keep a cycle inside its queue query, holding its lock.
Every executable call is appended to calls.log.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

LIMIT_S = 120  # how long a wait may last before it is reported as an error; never a pace
ENDED_WITHOUT_RUNNING = {"timeout": ("TIMEOUT", "0:0"), "preempt": ("PREEMPTED", "0:0"),
                         "node_fail": ("NODE_FAIL", "0:0"), "oom": ("OUT_OF_MEMORY", "0:125"),
                         "fail": ("FAILED", "1:0"), "cancel": ("CANCELLED", "0:0")}


class Cluster:
    def __init__(self, directory=None):
        self.dir = Path(directory or os.environ["FAKE_SLURM_DIR"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "state.json"
        self.crontab = self.dir / "scrontab"
        self.held = {}  # job id -> (process, start time) of jobs left RUNNING by "start"

    # ---- state -----------------------------------------------------------------------------

    def _locked(self, change):
        """Read the state, apply change(state), write it back; one caller at a time."""
        with open(self.dir / "lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.file.read_text()) if self.file.exists() else {"next": 1001, "jobs": {}, "plan": {}}
            result = change(state)
            temp = self.file.with_name("state.json.tmp")
            temp.write_text(json.dumps(state))
            os.replace(temp, self.file)
            return result

    def jobs(self):
        return sorted(self._locked(lambda state: list(state["jobs"].values())), key=lambda job: int(job["id"]))

    def calls(self):
        """Every executable call so far, as [command, arguments...]."""
        path = self.dir / "calls.log"
        return [line.split("\t") for line in path.read_text().splitlines()] if path.exists() else []

    def record(self, command, argv):
        with open(self.dir / "calls.log", "a") as handle:
            handle.write("\t".join([command] + list(argv)) + "\n")

    def add_job(self, name, state, account, qos, cpus=1, reason="None", per_node="N/A", per_job="N/A", held=""):
        """Another job of this user, not submitted by the watcher. It is never run or advanced."""
        def add(store):
            job_id = str(store["next"])
            store["next"] += 1
            store["jobs"][job_id] = {"id": job_id, "name": name, "state": state, "account": account, "qos": qos,
                                     "cpus": cpus, "reason": reason, "script": None, "tres": [per_node, per_job, held],
                                     "gres": "", "gpus": 0, "mem": "0", "exit": "0:0", "elapsed": 0,
                                     "start": None, "end": None, "output": None, "env": {}}
            return job_id
        return self._locked(add)

    def plan(self, name, outcomes):
        self._locked(lambda state: state["plan"].setdefault(name, []).extend(outcomes))

    def _next_outcome(self, state, name):
        queue = state["plan"].get(name) or ["complete"]
        step = queue.pop(0) if state["plan"].get(name) else queue[0]
        return {"outcome": step} if isinstance(step, str) else dict(step)

    # ---- the commands ----------------------------------------------------------------------

    def sbatch(self, argv):
        script = argv[-1]
        options = {}
        for line in Path(script).read_text().splitlines():
            if line.startswith("#SBATCH --"):
                key, _, value = line[len("#SBATCH --"):].partition("=")
                options[key] = value

        def add(state):
            if (state["plan"].get(options["job-name"]) or [None])[0] == "reject":
                state["plan"][options["job-name"]].pop(0)
                return None
            job_id = str(state["next"])
            state["next"] += 1
            gres = options.get("gres", "")
            state["jobs"][job_id] = {
                "id": job_id, "name": options["job-name"], "account": options["account"],
                "qos": options["qos"], "partition": options.get("partition"), "gres": gres,
                "gpus": int(gres.rsplit(":", 1)[1]) if gres else 0,
                "cpus": int(options.get("cpus-per-task", 1)), "mem": options.get("mem"),
                "time": options.get("time"), "signal": options.get("signal"),
                "requeue": "no-requeue" not in options, "output": options.get("output"),
                "script": script, "state": "PENDING", "reason": "Priority", "exit": "0:0",
                "elapsed": 0, "start": None, "end": None, "env": dict(os.environ)}
            return job_id

        job_id = self._locked(add)
        if job_id is None:
            print("sbatch: error: Batch job submission failed: Invalid qos specification")
            return 1
        print(job_id if "--parsable" in argv else "Submitted batch job " + job_id)
        return 0

    def squeue(self, argv):
        hold = os.environ.get("FAKE_SLURM_HOLD")
        if hold and not _wait_for(lambda: os.path.exists(hold)):
            return 1
        if os.environ.get("FAKE_SLURM_FAIL") == "squeue":
            return 1
        if os.environ.get("FAKE_SLURM_FAIL") == "squeue_layout":
            print("51138665|v2-C2-s4|RUNNING|cai6734|cai6734|N/A|6|None")  # not the layout asked for
            return 0
        columns = [(name.lower(), int(width)) for name, width in
                   (item.split(":") for item in argv[argv.index("-O") + 1].split(","))]
        for job in self.jobs():
            if job["state"] not in ("PENDING", "RUNNING"):
                continue
            if "tres" in job:
                per_node, per_job, held = job["tres"]
            else:  # submitted through sbatch: its --gres is a per-node request
                per_node, per_job = ("gres/" + job["gres"] if job["gres"] else "N/A"), "N/A"
                held = "cpu=%d,mem=%s,node=1,billing=%d" % (job["cpus"], job["mem"], job["cpus"])
                if job["gpus"]:
                    held += ",gres/gpu=%d,gres/%s=%d" % (job["gpus"], job["gres"].rsplit(":", 1)[0], job["gpus"])
                if job["state"] == "PENDING" and os.environ.get("FAKE_SLURM_PENDING_TRES") == "absent":
                    held = "N/A"  # a squeue that shows nothing under tres-alloc until a job runs
            values = {"jobid": job["id"], "state": job["state"], "account": job["account"], "qos": job["qos"],
                      "numcpus": str(job["cpus"]), "tres-per-node": per_node, "tres-per-job": per_job,
                      "tres-alloc": held, "name": job["name"],
                      "reason": job["reason"] if job["state"] == "PENDING" else "None"}
            print("".join("%-*.*s" % (width, width, values[name]) for name, width in columns))
        return 0

    def sacct(self, argv):
        if os.environ.get("FAKE_SLURM_FAIL") == "sacct":
            return 1
        names = [a.split("=", 1)[1].split(",") for a in argv if a.startswith("--name=")]
        columns = [a.split("=", 1)[1].split(",") for a in argv if a.startswith("--format=")][0]
        since = time.mktime(time.strptime(argv[argv.index("-S") + 1], "%Y-%m-%dT%H:%M:%S")) if "-S" in argv else 0
        for job in self.jobs():
            if names and job["name"] not in names[0]:
                continue
            if job["end"] is not None and job["end"] < since:
                continue
            tres = ""
            if job["state"] != "PENDING":
                tres = "billing=%d,cpu=%d" % (job["cpus"], job["cpus"])
                if job["gpus"]:
                    tres += ",gres/gpu=%d,gres/%s=%d" % (job["gpus"], job["gres"].rsplit(":", 1)[0], job["gpus"])
                tres += ",mem=%s,node=1" % job["mem"]
            values = {"JobID": job["id"], "JobName": job["name"], "State": job["state"],
                      "ExitCode": job["exit"], "ElapsedRaw": str(job["elapsed"]), "AllocTRES": tres,
                      "Account": job["account"], "QOS": job["qos"],
                      "Start": _clock(job["start"]), "End": _clock(job["end"])}
            print("|".join(values[column] for column in columns))
        return 0

    def scrontab(self, argv):
        if argv == ["-l"]:
            if not self.crontab.exists():
                print("no crontab for user")
                return 1
            print(self.crontab.read_text(), end="")
        elif argv == ["-r"]:
            self.crontab.unlink()
        elif os.environ.get("FAKE_SLURM_FAIL") == "scrontab":
            return 1  # the new crontab is not accepted
        else:
            self.crontab.write_text(Path(argv[0]).read_text())
        return 0

    def scontrol(self, argv):
        if argv[:1] == ["plan"]:
            self.plan(argv[1], argv[2:])
        elif argv[:1] == ["run"]:
            self.run()
        elif argv[:2] == ["show", "job"]:
            print(json.dumps([job for job in self.jobs() if job["id"] == argv[2]], indent=2))
        else:
            print("fake scontrol: plan NAME OUTCOME... | run | show job ID")
            return 1
        return 0

    # ---- the simulation --------------------------------------------------------------------

    def run(self):
        """Advance once: apply the next planned outcome to every pending or running job."""
        for job in self.jobs():
            if job["state"] not in ("PENDING", "RUNNING") or job["script"] is None:
                continue
            step = self._locked(lambda state: self._next_outcome(state, job["name"]))
            outcome = step["outcome"]
            if outcome == "hold":
                if "reason" in step:
                    self._locked(lambda state: state["jobs"][job["id"]].update(reason=step["reason"]))
                continue
            if outcome in ENDED_WITHOUT_RUNNING:
                state, code = ENDED_WITHOUT_RUNNING[outcome]
                elapsed = step.get("elapsed", 60)
                update = {"state": state, "exit": code, "elapsed": elapsed,
                          "start": time.time() - elapsed, "end": time.time()}
            else:
                update = self._execute(job, step)
            self._locked(lambda state: state["jobs"][job["id"]].update(update))

    def pace(self, job_name):
        """The folder through which a paced job (noop_slow --pace) is granted its steps."""
        return self.dir / "pace" / job_name

    def _grant(self, job, steps):
        """Permit a paced job its first `steps` steps and no more (None: all of them).

        Returns the marker the job leaves when it is waiting for the step after those.
        """
        pace = self.pace(job["name"])
        pace.mkdir(parents=True, exist_ok=True)
        for permit in pace.glob("go-*"):
            permit.unlink()
        if steps is None:
            (pace / "go-all").touch()
            return None
        for step in range(steps):
            (pace / ("go-%d" % step)).touch()
        waiting = pace / ("waiting-%d" % steps)
        if job["id"] not in self.held and waiting.exists():
            waiting.unlink()  # left by an earlier attempt of this job name
        return waiting

    def _execute(self, job, step):
        outcome = step["outcome"]
        steered = outcome in ("usr1", "kill", "start")
        waiting = self._grant(job, step["steps"] if steered else None)
        if job["id"] in self.held:
            process, started = self.held.pop(job["id"])
        else:
            environment = dict(job["env"], SLURM_JOB_ID=job["id"], SLURM_JOB_NAME=job["name"],
                               SLURM_JOB_ACCOUNT=job["account"], SLURM_JOB_QOS=job["qos"])
            output = job["output"].replace("%j", job["id"]) if job["output"] else os.devnull
            started = time.time()
            with open(output, "ab") as log:
                process = subprocess.Popen(["bash", job["script"]], env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        if steered:
            # The job cannot pass step "steps", so there is no race to win: wait until it says it
            # is there. It is then idle between two steps, with its checkpoint written.
            arrived = _wait_for(lambda: waiting.exists() or process.poll() is not None)
            if not arrived or process.poll() is not None:
                process.kill()
                process.wait()
                raise RuntimeError("fake Slurm: %s did not reach step %d alive; '%s' needs a paced stage"
                                   % (job["name"], step["steps"], outcome))
            if outcome == "start":
                self.held[job["id"]] = (process, started)
                return {"state": "RUNNING", "start": started, "elapsed": step.get("elapsed", 0)}
            if outcome == "usr1":
                os.kill(process.pid, signal.SIGUSR1)  # the script execs the stage: same pid
            else:
                os.killpg(process.pid, signal.SIGKILL)
        code = process.wait(timeout=LIMIT_S)
        ended = time.time()
        elapsed = step.get("elapsed", max(1, int(math.ceil(ended - started))))
        timing = {"elapsed": elapsed, "start": ended - elapsed if "elapsed" in step else started, "end": ended}
        if outcome == "kill":
            return dict(timing, state=step.get("state", "NODE_FAIL"), exit="0:0")
        return dict(timing, state="COMPLETED" if code == 0 else "FAILED",
                    exit="%d:0" % code if code >= 0 else "0:%d" % -code)


def _wait_for(condition):
    """Wait until condition() holds. False if it has not within LIMIT_S, which callers treat as an error."""
    deadline = time.monotonic() + LIMIT_S
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def _clock(epoch):
    """A start or end time as sacct prints it: local time without an offset, or Unknown."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch)) if epoch else "Unknown"


def main(command, argv):
    """Entry point of the executables beside this file."""
    cluster = Cluster()
    cluster.record(command, argv)
    return getattr(cluster, command)(argv)
