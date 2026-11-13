"""The watcher: one cycle per run, started every 15 minutes by scrontab through watcher_cycle.sh.

Implements WATCHER_DESIGN.md section 4, steps 1 to 9. Scheduler calls per cycle: one sacct, one
squeue, one sbatch per submission, and scrontab only when the study has nothing left to run.
It runs no analysis; the only judgments are the retry table and the committed gate rule scripts.

Everything it writes to events.tsv and STATUS.md is names, counts, numbers and codes, checked by
export_guard before it is written. Standard library only and Python 3.9 compatible, because it
runs under the compute node's /usr/bin/python3.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time

from . import authorization, export_guard, sbatch
from . import graph as graphs
from . import manifest as manifests
from .paths import Paths
from .util import append_line, canonical, now, read_json, sha256_file, write_json, write_new_json, write_text

# Status words of SPEC_RESEARCH_ADVANCEMENT.md section 4, plus REFUSED and BLOCKED.
PLANNED, QUEUED, RUNNING, INTERRUPTED = "PLANNED", "QUEUED", "RUNNING", "INTERRUPTED"
FAILED, COMPLETED, SKIPPED, REFUSED, BLOCKED = "FAILED", "COMPLETED", "SKIPPED BY GATE", "REFUSED", "BLOCKED"
STATUSES = (PLANNED, BLOCKED, QUEUED, RUNNING, INTERRUPTED, FAILED, REFUSED, COMPLETED, SKIPPED)
ACTIVE = (QUEUED, RUNNING)

LOCK_STALE_SECONDS = 20 * 60
RETRIES = {"interrupt": 3, "oom": 1, "other": 1}  # per stage; "continue" and "done" are not counted
MISSING_CYCLES = 4  # a submitted job unknown to both squeue and sacct for this long is a failure
ENDED = ("COMPLETED", "FAILED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY", "CANCELLED",
         "BOOT_FAIL", "DEADLINE", "REVOKED", "MISSING")
SACCT_FORMAT = "JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES,Account,QOS"
SQUEUE_FORMAT = "%i|%j|%T|%a|%q|%b|%C"
TAG = "# tii-watcher"


def classify(state, exit_code, progressed):
    """The outcome table of WATCHER_DESIGN.md section 4, step 4."""
    if state == "OUT_OF_MEMORY" or (state == "FAILED" and exit_code == 40):
        return "oom"
    if state == "TIMEOUT":
        return "continue" if progressed else "interrupt"
    if state in ("PREEMPTED", "NODE_FAIL"):
        return "interrupt"
    if state == "COMPLETED" and exit_code == 0:
        return "done"
    if state == "FAILED" and exit_code == 10:
        return "continue"
    if state == "FAILED" and exit_code in (20, 30):
        return "refused"
    return "other"


def gpus_in(tres):
    """GPU count in a Slurm TRES or gres string: gres/gpu=1, gpu:rtx_pro_6000:1, gres:gpu:2."""
    match = re.search(r"gpu(?::[A-Za-z0-9_.-]+)?[:=](\d+)", tres)
    return int(match.group(1)) if match else (1 if "gpu" in tres else 0)


def remove_tagged(crontab):
    """Drop the tagged entry and the #SCRON lines directly above it, which apply only to it."""
    lines = crontab.splitlines()
    drop = set()
    for index, line in enumerate(lines):
        if TAG in line:
            drop.add(index)
            above = index - 1
            while above >= 0 and lines[above].startswith("#SCRON"):
                drop.add(above)
                above -= 1
    kept = [line for index, line in enumerate(lines) if index not in drop]
    return "\n".join(kept) + "\n" if kept else ""


def call(command):
    """Run one scheduler command. Returns (return code, stdout)."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return result.returncode, result.stdout


def show(value):
    """A value as one token of an event or a status cell."""
    if value is None:
        return "none"
    if isinstance(value, float):
        return ("%.3f" % value).rstrip("0").rstrip(".") or "0"
    return str(value)


def pairs(detail):
    return " ".join("%s=%s" % (key, show(value)) for key, value in detail.items())


def progress_mark(paths, stage_id):
    """Newest modification time of a stage's checkpoint files. The files are never read here."""
    newest = 0
    for name in ("checkpoint.json", "items.jsonl"):
        try:
            newest = max(newest, os.stat(paths.stage_dir(stage_id) / name).st_mtime_ns)
        except OSError:
            pass
    return newest


def take_lock(paths):
    """Step 1. Returns None if another cycle holds the lock, else the age of a stale lock or 0."""
    paths.state.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(paths.lock)
        return 0
    except FileExistsError:
        pass
    try:
        age = time.time() - os.stat(paths.lock).st_mtime
    except OSError:
        return None
    if age < LOCK_STALE_SECONDS:
        return None
    aside = paths.lock.with_name("lock.stale.%d" % os.getpid())
    try:
        os.rename(paths.lock, aside)  # only one of two cycles that both see a stale lock wins this
        os.mkdir(paths.lock)
        os.rmdir(aside)
    except OSError:
        return None
    return int(age)


class Cycle:
    def __init__(self, paths):
        self.paths = paths
        self.started = time.time()
        self.meta = read_json(paths.watcher, {})
        self.python = os.environ.get("TII_PYTHON")
        self.block_keys = {}  # every condition blocking something this cycle: key -> detail
        self.counts = {"ended": 0, "submitted": 0, "events": 0}

    # ---- records ---------------------------------------------------------------------------

    def log(self, event, **fields):
        append_line(self.paths.runlog, json.dumps(dict(fields, time=now(), event=event), sort_keys=True))

    def emit(self, kind, **detail):
        line = "%s\t%s\t%s" % (now(), kind, pairs(detail))
        if not export_guard.event_ok(line):
            raise ValueError("an event must hold names, counts, numbers and codes only")
        append_line(self.paths.events, line)
        self.log("event", kind=kind, text=pairs(detail))
        self.counts["events"] += 1

    def load_state(self, stage):
        fresh = {"id": stage["id"], "status": PLANNED, "job_id": None, "submitted": None,
                 "attempts": [], "retries": {"interrupt": 0, "oom": 0, "other": 0},
                 "mem_gb": int(math.ceil(stage["resources"]["mem_gb"])), "checkpoint": None,
                 "progress_mark": 0, "manifest_id": None, "gpu_hours": 0.0, "missing": 0}
        return dict(fresh, **read_json(self.paths.stage_state(stage["id"]), {}))

    def save(self, state):
        write_json(self.paths.stage_state(state["id"]), state)
        self.saved[state["id"]] = canonical(state)

    def block(self, state, key):
        state.update(blocked_from=state["status"], blocked_by=key, status=BLOCKED)

    # ---- the cycle -------------------------------------------------------------------------

    def run(self):
        paths = self.paths
        if not self.python:
            self.log("config_error", code="TII_PYTHON_unset")
            return 2
        try:
            self.graph = graphs.load(paths)
        except graphs.GraphError as error:
            self.log("graph_invalid", detail=str(error))
            self.block_keys["graph"] = {"code": "graph_invalid"}
            self.sync_blocks(keep_open=True)
            return 1
        self.states = {stage["id"]: self.load_state(stage) for stage in self.graph.stages}
        # What is on disk now, so the end of the cycle writes only what changed (or is new).
        self.saved = {name: canonical(state) if self.paths.stage_state(name).exists() else None
                      for name, state in self.states.items()}
        self.gate_records = {}
        for gate in self.graph.gates:
            record = read_json(paths.gate_record(gate["id"]))
            if record is not None:
                self.gate_records[gate["id"]] = record

        # Step 2: authorization. The marking of stages happens with the submissions below.
        self.auth = authorization.check(paths.study_dir)
        self.pools = authorization.pools(paths.study_dir)

        # Step 3: the cycle's only accounting query and only queue query.
        accounting, queue = self.sacct(), self.squeue()
        if accounting is None or queue is None:
            self.log("scheduler_error", sacct=accounting is not None, squeue=queue is not None)
            self.sync_blocks(keep_open=True)
            self.write_status(scheduler="error")
            return 1

        for state in self.states.values():  # blocks are derived afresh every cycle
            if state["status"] == BLOCKED:
                state["status"] = state.pop("blocked_from")
                state.pop("blocked_by", None)
        if self.auth != "ok":
            self.block_keys["authorization"] = {"code": self.auth}
        final = self.meta.get("final")
        if final and not final["scrontab_disabled"]:
            self.block_keys["scrontab"] = {"code": "entry_not_disabled"}

        self.account(accounting, queue)   # step 4
        self.submit(queue)                # step 5
        self.decide_gates()               # step 6
        self.check_surface()
        for name, state in self.states.items():
            if state["status"] == REFUSED:
                self.block_keys["refused:" + name] = dict(stage=name, **state.get("refusal", {}))
        self.sync_blocks()
        final = self.finish()             # step 8, decided before the status is written
        for state in self.states.values():
            if canonical(state) != self.saved[state["id"]]:
                self.save(state)
        self.meta.update(last_cycle=self.started, cycle=self.meta.get("cycle", 0) + 1)
        write_json(paths.watcher, self.meta)
        self.write_status()               # step 7
        if final and not final["scrontab_disabled"]:
            final["scrontab_disabled"] = self.disable_scrontab()
            write_json(paths.watcher, self.meta)
        return 0

    # ---- step 3 ----------------------------------------------------------------------------

    def sacct(self):
        if not self.graph.stages:
            return []
        tracked = [state["submitted"] for state in self.states.values() if state["job_id"]]
        since = min([self.meta.get("last_cycle", self.started)] + [t for t in tracked if t]) - 300
        code, out = call(["sacct", "-X", "-n", "-P",
                          "--name=" + ",".join("tii-" + stage["id"] for stage in self.graph.stages),
                          "-S", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since)),
                          "--format=" + SACCT_FORMAT])
        if code != 0:
            return None
        records = []
        for line in out.splitlines():
            fields = line.split("|")
            if len(fields) != 8 or not fields[0].isdecimal():
                continue
            records.append({"id": fields[0], "name": fields[1], "state": (fields[2].split() or [""])[0],
                            "exit_code": int(fields[3].split(":")[0] or 0), "elapsed": int(fields[4] or 0),
                            "gpus": gpus_in(fields[5]), "account": fields[6], "qos": fields[7]})
        return records

    def squeue(self):
        code, out = call(["squeue", "--me", "-h", "-o", SQUEUE_FORMAT])
        if code != 0:
            return None
        rows = []
        for line in out.splitlines():
            fields = line.split("|")
            if len(fields) == 7 and fields[0].isdecimal():
                rows.append({"id": fields[0], "name": fields[1], "state": fields[2], "account": fields[3],
                             "qos": fields[4], "gpus": gpus_in(fields[5]),
                             "cpus": int(fields[6]) if fields[6].isdecimal() else 0})
        return rows

    # ---- step 4 ----------------------------------------------------------------------------

    def account(self, accounting, queue):
        in_queue = {row["id"]: row for row in queue}
        known = {record["id"] for record in accounting}
        for stage in self.graph.stages:
            state = self.states[stage["id"]]
            # Every ended job that carries this stage's name is logged and charged exactly once,
            # including one this watcher did not submit.
            for record in accounting:
                if (record["name"] == "tii-" + stage["id"] and record["state"] in ENDED
                        and not any(a["job_id"] == record["id"] for a in state["attempts"])):
                    self.ended(stage, state, record, record["id"] == state["job_id"])
            job = state["job_id"]
            if job is None:
                continue
            if job in in_queue:
                state["status"] = QUEUED if in_queue[job]["state"] == "PENDING" else RUNNING
                state["missing"] = 0
            elif job in known:
                state["missing"] = 0  # ended, but accounting has not caught up; decide next cycle
            else:
                state["missing"] += 1
                if state["missing"] >= MISSING_CYCLES:
                    self.ended(stage, state, {"id": job, "state": "MISSING", "exit_code": 0, "elapsed": 0,
                                              "gpus": 0, "account": state.get("account"),
                                              "qos": state.get("qos")}, True)

    def ended(self, stage, state, record, tracked):
        name = stage["id"]
        progressed = progress_mark(self.paths, name) > state["progress_mark"]
        outcome = classify(record["state"], record["exit_code"], progressed) if tracked else "untracked"
        attempt = {"job_id": record["id"], "state": record["state"], "exit_code": record["exit_code"],
                   "elapsed_s": record["elapsed"],
                   "gpu_hours": round(record["gpus"] * record["elapsed"] / 3600.0, 6),
                   "outcome": outcome, "account": record["account"], "qos": record["qos"]}
        self.log("attempt", stage=name, experiment=stage["experiment"], **attempt)
        self.counts["ended"] += 1
        state["attempts"].append(attempt)
        state["gpu_hours"] = round(state["gpu_hours"] + attempt["gpu_hours"], 6)
        if tracked:
            state.update(job_id=None, submitted=None, missing=0)
            if outcome == "done":
                state["status"] = COMPLETED
                self.emit("stage completed", stage=name, experiment=stage["experiment"],
                          attempts=len(state["attempts"]), gpu_hours=state["gpu_hours"])
            elif outcome == "continue":
                state["status"] = INTERRUPTED
            elif outcome == "refused":
                state["status"] = REFUSED
                state["refusal"] = self.refusal(name, record["exit_code"])
            elif state["retries"][outcome] < RETRIES[outcome]:
                state["retries"][outcome] += 1
                state["status"] = INTERRUPTED
                if outcome == "oom":
                    state["mem_gb"] = int(math.ceil(state["mem_gb"] * 1.5))
            else:
                state["status"] = FAILED
                self.emit("failure", stage=name, experiment=stage["experiment"], outcome=outcome,
                          state=record["state"], exit=record["exit_code"], attempts=len(state["attempts"]))
        self.save(state)

    def refusal(self, name, exit_code):
        """The reason the stage wrote, reduced to codes."""
        written = read_json(self.paths.refusal(name), {})
        detail = {"exit": exit_code, "reason": "unknown"}
        for key, field in (("reason", "code"), ("role", "role")):
            value = written.get(field)
            if isinstance(value, str) and export_guard.KEY.match(value):
                detail[key] = value
        return detail

    # ---- step 5 ----------------------------------------------------------------------------

    def ready(self, stage):
        if any(self.states[name]["status"] != COMPLETED for name in stage["depends_on"]):
            return False
        return stage["gate"] is None or self.gate_records.get(stage["gate"], {}).get("result") == "met"

    def remaining(self, stage):
        return max(stage["estimate_gpu_hours"] - self.states[stage["id"]]["gpu_hours"], 0.0)

    def spent(self, key):
        return sum(self.states[s["id"]]["gpu_hours"] for s in self.graph.stages if s["allowance_key"] == key)

    def submit(self, queue):
        tracked = {state["job_id"] for state in self.states.values() if state["job_id"]}
        strays = {}  # live jobs with a stage's name that no stage tracks
        in_use = {"gpu": {}, "cpu": {}}
        for row in queue:
            if row["id"] not in tracked:
                strays.setdefault(row["name"], row)
            kind, held = ("gpu", row["gpus"]) if row["gpus"] else ("cpu", row["cpus"])
            pool = (row["account"], row["qos"])
            in_use[kind][pool] = in_use[kind].get(pool, 0) + held
        reserved = {}  # allowance key -> GPU-hours still expected from stages already under way
        for stage in self.graph.stages:
            if self.states[stage["id"]]["status"] in ACTIVE + (INTERRUPTED,):
                key = stage["allowance_key"]
                reserved[key] = reserved.get(key, 0.0) + self.remaining(stage)

        waiting = [s for s in self.graph.stages if self.states[s["id"]]["status"] == INTERRUPTED]
        waiting += [s for s in self.graph.stages if self.states[s["id"]]["status"] == PLANNED]
        for stage in waiting:
            name, state = stage["id"], self.states[stage["id"]]
            stray = strays.pop("tii-" + name, None)
            if stray:  # adopt a job that is already there instead of starting a second
                state.update(job_id=stray["id"], submitted=self.started, account=stray["account"],
                             qos=stray["qos"], status=QUEUED if stray["state"] == "PENDING" else RUNNING,
                             progress_mark=progress_mark(self.paths, name))
                self.save(state)
                self.log("adopted", stage=name, job_id=stray["id"])
                continue
            if self.auth != "ok":
                self.block(state, "authorization")
                continue
            resuming = state["status"] == INTERRUPTED
            if not resuming and not self.ready(stage):
                continue

            key = stage["allowance_key"]
            allowance = self.graph.allowances[key]
            mine = self.remaining(stage)
            others = max(reserved.get(key, 0.0) - (mine if resuming else 0.0), 0.0)
            if not allowance["authorized"] or self.spent(key) + others + mine > allowance["allowance_gpu_hours"] + 1e-9:
                self.block(state, "allowance:" + key)
                self.block_keys.setdefault("allowance:" + key, {
                    "experiment": key, "stage": name, "spent": self.spent(key), "outstanding": others,
                    "estimate": mine, "allowance": float(allowance["allowance_gpu_hours"])})
                continue

            resources = stage["resources"]
            kind, need = ("gpu", resources["gpus"]) if resources["gpus"] else ("cpu", resources["cpus"])
            pool = authorization.pick_pool(self.pools, kind, need, in_use[kind])
            if pool is None:
                continue  # every usable account is at its limit; wait for a slot
            try:
                manifest = manifests.ensure(self.paths, self.graph, stage)
            except manifests.Refusal as refusal:
                self.block(state, "manifest:" + name)
                self.block_keys["manifest:" + name] = {"stage": name, "code": refusal.code, "role": refusal.role}
                continue
            attempt = len(state["attempts"]) + 1
            job_id, code = sbatch.submit(self.paths, stage, pool, state["mem_gb"], self.python, attempt)
            if job_id is None:
                self.block_keys["submit:" + name] = {"stage": name, "rc": code}
                self.log("submit_failed", stage=name, rc=code)
                continue
            state.update(status=QUEUED, job_id=job_id, submitted=self.started, account=pool.account,
                         qos=pool.qos, manifest_id=manifest["manifest_id"],
                         checkpoint=str(self.paths.stage_dir(name)),
                         progress_mark=progress_mark(self.paths, name))
            self.save(state)
            self.log("submitted", stage=name, job_id=job_id, account=pool.account, qos=pool.qos,
                     mem_gb=state["mem_gb"], attempt=attempt)
            self.counts["submitted"] += 1
            in_use[kind][(pool.account, pool.qos)] = in_use[kind].get((pool.account, pool.qos), 0) + need
            if not resuming:
                reserved[key] = reserved.get(key, 0.0) + mine

    # ---- step 6 ----------------------------------------------------------------------------

    def decide_gates(self):
        self.skip_by_gate()  # anything earlier records already imply
        for gate in self.graph.gates:
            name = gate["id"]
            if name in self.gate_records:
                continue
            if any(self.states[s]["status"] not in (COMPLETED, SKIPPED) for s in gate["inputs"]):
                continue
            verdict = self.run_rule(gate)
            if verdict is None:
                self.block_keys["gate:" + name] = {"gate": name, "code": "rule_failed"}
                self.log("gate_rule_failed", gate=name)
                continue
            digests = []
            for item in gate["reads"]:
                path = self.paths.root / item
                digests.append({"file": item, "sha256": sha256_file(path) if path.is_file() else None})
            record = {"gate": name, "decided": now(), "rule": gate["rule"],
                      "rule_sha256": sha256_file(self.paths.gate_rules / gate["rule"]),
                      "inputs": {s: self.states[s]["status"] for s in gate["inputs"]},
                      "input_digests": digests, "result": verdict["result"],
                      "path": verdict.get("path", verdict["result"]),
                      "evidence": verdict.get("evidence", {})}
            self.gate_records[name] = record
            record["skipped"] = self.skip_by_gate()
            self.emit("gate decided", gate=name, result=record["result"], path=record["path"],
                      skipped=len(record["skipped"]))
            try:
                write_new_json(self.paths.gate_record(name), record)
            except FileExistsError:
                self.gate_records[name] = read_json(self.paths.gate_record(name))
        self.skip_by_gate()

    def run_rule(self, gate):
        """Run the committed rule script on the files the gate may read. None if it did not decide."""
        command = [sys.executable, str(self.paths.gate_rules / gate["rule"])]
        command += [str(self.paths.root / item) for item in gate["reads"]]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300, cwd=str(self.paths.root))
            verdict = json.loads(result.stdout)
            export_guard.check_aggregate(verdict)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        if (result.returncode != 0 or verdict.get("result") not in ("met", "not_met")
                or not isinstance(verdict.get("path", ""), str) or not isinstance(verdict.get("evidence", {}), dict)):
            return None
        return verdict

    def skip_by_gate(self):
        """Mark stages whose gate was not met, and everything that depends on them. Returns their names."""
        skipped, changed = [], True
        while changed:
            changed = False
            for stage in self.graph.stages:
                state = self.states[stage["id"]]
                if state["status"] not in (PLANNED, BLOCKED):
                    continue
                record = self.gate_records.get(stage["gate"])
                by = stage["gate"] if record is not None and record["result"] != "met" else None
                for name in stage["depends_on"]:
                    if by is None and self.states[name]["status"] == SKIPPED:
                        by = self.states[name].get("skipped_by") or "dependency"
                if by is not None:
                    state.pop("blocked_from", None)
                    state.pop("blocked_by", None)
                    state.update(status=SKIPPED, skipped_by=by)
                    skipped.append(stage["id"])
                    changed = True
        return skipped

    # ---- surface, BLOCKED items --------------------------------------------------------------

    def check_surface(self):
        """Move anything unguarded out of export/approved/, then check the whole surface."""
        for path, code in export_guard.approved_violations(self.paths):
            self.paths.quarantine.mkdir(parents=True, exist_ok=True)
            os.replace(path, self.paths.quarantine / ("%d-%s" % (time.time_ns(), path.name)))
            self.log("quarantined", code=code)
        try:
            quarantined = len(os.listdir(self.paths.quarantine))
        except OSError:
            quarantined = 0
        violations = export_guard.surface_check(self.paths)
        if quarantined or violations:
            self.block_keys["surface"] = {"quarantined": quarantined, "violations": len(violations)}
            self.log("surface", quarantined=quarantined, violations=violations[:20])

    def sync_blocks(self, keep_open=False):
        """Open an item, with one event, for each new condition; close items whose condition is gone.

        keep_open leaves the other items alone, for a cycle that could not evaluate them.
        """
        book = read_json(self.paths.blocked, {"next": 1, "items": []})
        open_items = {item["key"]: item for item in book["items"] if item["closed"] is None}
        for key, detail in self.block_keys.items():
            if key in open_items:
                open_items[key]["detail"] = detail
                continue
            item = {"id": "W-%03d" % book["next"], "key": key, "detail": detail, "opened": now(), "closed": None}
            book["next"] += 1
            book["items"].append(item)
            self.emit("blocked", item=item["id"], key=key, **detail)
        for key, item in open_items.items():
            if key not in self.block_keys and not keep_open:
                item["closed"] = now()
                self.log("blocked_closed", item=item["id"], key=key)
        write_json(self.paths.blocked, book)
        self.open_items = [item for item in book["items"] if item["closed"] is None]
        lines = ["# BLOCKED items raised by the watcher, study TII-CEE-1", "",
                 "Conditions the watcher cannot resolve by itself. Open items are copied into the",
                 "repository's BLOCKED.md with the exact request. Items are closed, never deleted.", ""]
        for item in book["items"]:
            lines += ["## %s: %s (%s)" % (item["id"], item["key"], "CLOSED" if item["closed"] else "OPEN"),
                      "- opened: %s" % item["opened"], "- closed: %s" % show(item["closed"]),
                      "- detail: %s" % pairs(item["detail"]), ""]
        write_text(self.paths.blocked_md, "\n".join(lines))

    # ---- steps 7 and 8 -----------------------------------------------------------------------

    def can_finish(self, name, memo):
        """Could this stage still complete or be skipped without anyone stepping in?"""
        if name not in memo:
            stage, status = self.graph.by_id[name], self.states[name]["status"]
            if status != PLANNED:
                memo[name] = status in ACTIVE + (INTERRUPTED, COMPLETED, SKIPPED)
            else:
                gate = stage["gate"]
                undecided = gate is not None and gate not in self.gate_records
                memo[name] = ("gate:%s" % gate) not in self.block_keys and all(
                    self.can_finish(before, memo)
                    for before in stage["depends_on"] + (self.graph.gate_by_id[gate]["inputs"] if undecided else []))
        return memo[name]

    def finish(self):
        """Step 8. Returns the final record when nothing is pending, running or retryable."""
        memo = {}
        statuses = [state["status"] for state in self.states.values()]
        if any(status in ACTIVE + (INTERRUPTED,) for status in statuses) or any(
                state["status"] == PLANNED and self.can_finish(name, memo) for name, state in self.states.items()):
            self.meta.pop("final", None)
            return None
        complete = all(status in (COMPLETED, SKIPPED) for status in statuses)
        kind = "study completed" if complete else "all remaining items blocked"
        final = self.meta.get("final")
        if not final or final["kind"] != kind:
            self.emit(kind, stages=len(statuses), completed=statuses.count(COMPLETED),
                      skipped=statuses.count(SKIPPED), failed=statuses.count(FAILED),
                      refused=statuses.count(REFUSED), blocked=statuses.count(BLOCKED),
                      waiting=statuses.count(PLANNED))
            final = self.meta["final"] = {"kind": kind, "time": now(), "scrontab_disabled": False}
            self.log("final", kind=kind)
        return final

    def disable_scrontab(self):
        """Remove only the tagged entry; every other line is reinstalled as it was. True if verified."""
        code, before = call(["scrontab", "-l"])
        if code == 127:  # the command itself could not be run: nothing is known
            self.log("scrontab", action="scrontab_unavailable")
            return False
        if code != 0 or TAG not in before:
            self.log("scrontab", action="nothing_to_disable")
            return True
        kept = remove_tagged(before)
        path = write_text(self.paths.state / "scrontab.new", kept)
        code, _ = call(["scrontab", str(path)])
        check, after = call(["scrontab", "-l"])
        verified = code == 0 and TAG not in after and after.strip() == kept.strip()
        self.log("scrontab", action="disabled" if verified else "disable_not_verified",
                 kept_lines=len(kept.splitlines()))
        return verified

    def write_status(self, scheduler="ok"):
        final = self.meta.get("final")
        addendum = authorization.addenda(self.paths.study_dir).get(authorization.ADDENDUM_1)
        lines = [export_guard.STATUS_TITLE, "",
                 "updated: " + now(),
                 "milestone: " + self.graph.milestone,
                 "cycle: %d" % self.meta.get("cycle", 0),
                 "authorization: " + self.auth,
                 "addendum_1: " + ("absent" if addendum is None else "signed" if addendum["signed"] else "unsigned"),
                 "scheduler: " + scheduler,
                 "final: " + (final["kind"].replace(" ", "_") if final else "none")]

        def table(heading, header, rows):
            lines.extend(["", "## " + heading, "", "| " + " | ".join(header) + " |",
                          "|" + "---|" * len(header)])
            lines.extend("| " + " | ".join(show(cell) for cell in row) + " |" for row in rows)

        statuses = [state["status"] for state in self.states.values()]
        table("Stages by status", ("status", "count"), [(s, statuses.count(s)) for s in STATUSES])
        table("Running and queued jobs", ("stage", "status", "job", "account", "qos", "gpus", "attempt"),
              [(name, state["status"], state["job_id"], state.get("account"), state.get("qos"),
                self.graph.by_id[name]["resources"]["gpus"], len(state["attempts"]) + 1)
               for name, state in self.states.items() if state["status"] in ACTIVE])
        keys = sorted({stage["allowance_key"] for stage in self.graph.stages})
        table("GPU-hours by experiment", ("experiment", "spent", "estimate", "allowance"),
              [(key, self.spent(key), float(self.graph.allowances[key]["estimate_gpu_hours"]),
                float(self.graph.allowances[key]["allowance_gpu_hours"])) for key in keys])
        table("Open BLOCKED items", ("item", "key", "opened", "detail"),
              [(item["id"], item["key"], item["opened"], pairs(item["detail"])) for item in self.open_items])
        table("Latest gate records", ("gate", "result", "path", "decided", "skipped"),
              [(name, record["result"], record["path"], record["decided"], len(record["skipped"]))
               for name, record in sorted(self.gate_records.items())])
        try:
            events = self.paths.events.read_text().splitlines()[-30:]
        except OSError:
            events = []
        table("Latest events", ("time", "kind", "text"),
              [line.split("\t") for line in events if export_guard.event_ok(line)])

        text = "\n".join(lines) + "\n"
        if export_guard.status_violations(text):
            raise ValueError("STATUS.md must hold names, counts, numbers and codes only")
        write_text(self.paths.status, text)


def main():
    paths = Paths()
    stale = take_lock(paths)
    if stale is None:
        print("%s another cycle holds the lock" % now())
        return 0
    try:
        cycle = Cycle(paths)
        if stale:
            cycle.log("stale_lock", age_s=stale)
        code = cycle.run()
        print("%s cycle %s rc=%d ended=%d submitted=%d events=%d" % (
            now(), cycle.meta.get("cycle", 0), code, cycle.counts["ended"],
            cycle.counts["submitted"], cycle.counts["events"]))
        return code
    finally:
        os.rmdir(paths.lock)  # step 9


if __name__ == "__main__":
    sys.exit(main())
