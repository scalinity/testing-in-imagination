"""A small fake Slurm over a state directory, for the pipeline tests. Nothing here contacts a cluster.

The executables beside this file (sbatch, squeue, sacct, scrontab, scontrol) call into it with the
directory named by FAKE_SLURM_DIR. sbatch only records a job; nothing runs until a test advances
the cluster with Cluster.run() (or `scontrol run`), which applies the next planned outcome to each
pending or running job:

    complete   run the job script; COMPLETED on exit 0, else FAILED with its exit code (default)
    usr1       run it, send SIGUSR1 once the trigger is met (the ten-minute warning), let it exit
    kill       run it, SIGKILL it once the trigger is met; "state" names what Slurm reports
               (NODE_FAIL by default, or TIMEOUT, PREEMPTED, ...)
    timeout, preempt, node_fail, oom, fail, cancel
               end the job in that state without running it
    start      start the job script, freeze it (SIGSTOP) once the trigger is met, and report it
               RUNNING; the next advance resumes it and applies the next outcome. Only through
               Cluster.run() on one Cluster object, which holds the frozen process
    hold       leave it PENDING for this advance

An outcome is a name or {"outcome": name, "state": ..., "elapsed": seconds, "wait_file": path,
"wait_lines": n}; the trigger is "wait_file has at least wait_lines lines". Planning "reject" makes
the next sbatch for that job name fail. Every executable call is appended to calls.log.
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

ENDED_WITHOUT_RUNNING = {"timeout": ("TIMEOUT", "0:0"), "preempt": ("PREEMPTED", "0:0"),
                         "node_fail": ("NODE_FAIL", "0:0"), "oom": ("OUT_OF_MEMORY", "0:125"),
                         "fail": ("FAILED", "1:0"), "cancel": ("CANCELLED", "0:0")}


class Cluster:
    def __init__(self, directory=None):
        self.dir = Path(directory or os.environ["FAKE_SLURM_DIR"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "state.json"
        self.crontab = self.dir / "scrontab"
        self.frozen = {}  # job id -> (process, start time) of jobs left RUNNING by "start"

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
                "script": script, "state": "PENDING", "exit": "0:0", "elapsed": 0, "end": None,
                "env": dict(os.environ)}
            return job_id

        job_id = self._locked(add)
        if job_id is None:
            print("sbatch: error: Batch job submission failed: Invalid qos specification")
            return 1
        print(job_id if "--parsable" in argv else "Submitted batch job " + job_id)
        return 0

    def squeue(self, argv):
        time.sleep(float(os.environ.get("FAKE_SLURM_DELAY", "0")))
        if os.environ.get("FAKE_SLURM_FAIL") == "squeue":
            return 1
        layout = argv[argv.index("-o") + 1]
        for job in self.jobs():
            if job["state"] in ("PENDING", "RUNNING"):
                values = {"%i": job["id"], "%j": job["name"], "%T": job["state"], "%a": job["account"],
                          "%q": job["qos"], "%b": "gres/" + job["gres"] if job["gres"] else "N/A",
                          "%C": str(job["cpus"])}
                line = layout
                for token, value in values.items():
                    line = line.replace(token, value)
                print(line)
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
                      "Account": job["account"], "QOS": job["qos"]}
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
            if job["state"] not in ("PENDING", "RUNNING"):
                continue
            step = self._locked(lambda state: self._next_outcome(state, job["name"]))
            outcome = step["outcome"]
            if outcome == "hold":
                continue
            if outcome in ENDED_WITHOUT_RUNNING:
                state, code = ENDED_WITHOUT_RUNNING[outcome]
                update = {"state": state, "exit": code, "elapsed": step.get("elapsed", 60), "end": time.time()}
            else:
                update = self._execute(job, step)
            self._locked(lambda state: state["jobs"][job["id"]].update(update))

    def _execute(self, job, step):
        outcome = step["outcome"]
        if job["id"] in self.frozen:
            process, started = self.frozen.pop(job["id"])
            os.killpg(process.pid, signal.SIGCONT)
        else:
            environment = dict(job["env"], SLURM_JOB_ID=job["id"], SLURM_JOB_NAME=job["name"],
                               SLURM_JOB_ACCOUNT=job["account"], SLURM_JOB_QOS=job["qos"])
            output = job["output"].replace("%j", job["id"]) if job["output"] else os.devnull
            started = time.time()
            with open(output, "ab") as log:
                process = subprocess.Popen(["bash", job["script"]], env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        if outcome in ("usr1", "kill", "start"):
            deadline = time.time() + 30
            while process.poll() is None and time.time() < deadline and not self._triggered(step):
                time.sleep(0.01)
            if process.poll() is None:
                if outcome == "start":
                    os.killpg(process.pid, signal.SIGSTOP)
                    self.frozen[job["id"]] = (process, started)
                    return {"state": "RUNNING"}
                if outcome == "usr1":
                    os.kill(process.pid, signal.SIGUSR1)  # the script execs the stage: same pid
                else:
                    os.killpg(process.pid, signal.SIGKILL)
        code = process.wait(timeout=120)
        elapsed = step.get("elapsed", max(1, int(math.ceil(time.time() - started))))
        if outcome == "kill":
            return {"state": step.get("state", "NODE_FAIL"), "exit": "0:0", "elapsed": elapsed, "end": time.time()}
        return {"state": "COMPLETED" if code == 0 else "FAILED",
                "exit": "%d:0" % code if code >= 0 else "0:%d" % -code, "elapsed": elapsed, "end": time.time()}

    def _triggered(self, step):
        if "wait_file" not in step:
            return True
        try:
            with open(step["wait_file"], "rb") as handle:
                return handle.read().count(b"\n") >= step.get("wait_lines", 1)
        except OSError:
            return False


def main(command, argv):
    """Entry point of the executables beside this file."""
    cluster = Cluster()
    cluster.record(command, argv)
    return getattr(cluster, command)(argv)
