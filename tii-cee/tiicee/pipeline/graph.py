"""The stage graph, $TII/pipeline/graph.json, and the committed GPU allowances.

graph.json:
  {"study": "TII-CEE-1", "milestone": "M0", "stage_modules": ["package.module", ...],
   "stages": [{"id", "experiment", "kind": "normal" | "test", "command": [stage name, args...],
               "resources": {"gpus": 0 | 1, "cpus", "mem_gb", "time_hours"},
               "depends_on": [stage ids], "gate": gate id (optional),
               "allowance_key", "estimate_gpu_hours",
               "manifest": {"intervention": "new intervention" | "legacy repair",
                            "bind": {role: path}, "seeds": {purpose: int}, "pairing_unit"}}],
   "gates": [{"id", "rule": file in tii-cee/gates/, "inputs": [stage ids], "reads": [paths under $TII]}]}

A stage with a gate runs only when the gate's record says "met"; otherwise it is SKIPPED BY GATE.
A bound path is relative to the code checkout, or under the storage root when it starts with
"$TII/", or absolute. Standard library only and Python 3.9 compatible.
"""
from __future__ import annotations

import json

from .paths import ID, STUDY

KINDS = ("normal", "test")
INTERVENTIONS = ("new intervention", "legacy repair")
# One GPU per job and 24 hours at most (AUTHORIZATION.md section 3). The lower bound keeps the
# ten-minute warning signal from arriving the moment a job starts.
MIN_HOURS, MAX_HOURS = 0.25, 24


class GraphError(ValueError):
    pass


def load_allowances(paths):
    with open(paths.allowances) as handle:
        return json.load(handle)["experiments"]


def _need(condition, message):
    if not condition:
        raise GraphError(message)


def _ids(value, known, what):
    _need(isinstance(value, list) and all(item in known for item in value), what)
    return list(value)


def _stage(raw, stage_ids, gate_ids, allowances):
    _need(isinstance(raw, dict), "a stage is not an object")
    name = raw.get("id")
    _need(isinstance(name, str) and ID.match(name), "bad stage id")
    where = "stage %s: " % name
    _need(isinstance(raw.get("experiment"), str) and ID.match(raw["experiment"]), where + "bad experiment")
    _need(raw.get("kind", "normal") in KINDS, where + "kind must be normal or test")
    command = raw.get("command")
    _need(isinstance(command, list) and command and all(isinstance(part, str) for part in command),
          where + "command must be a list of strings")
    resources = raw.get("resources")
    _need(isinstance(resources, dict), where + "resources missing")
    gpus, cpus = resources.get("gpus", 0), resources.get("cpus", 1)
    mem, hours = resources.get("mem_gb"), resources.get("time_hours")
    _need(gpus in (0, 1), where + "gpus must be 0 or 1")
    _need(isinstance(cpus, int) and cpus >= 1, where + "cpus must be a positive integer")
    _need(isinstance(mem, (int, float)) and mem > 0, where + "mem_gb must be positive")
    _need(isinstance(hours, (int, float)) and MIN_HOURS <= hours <= MAX_HOURS,
          where + "time_hours must be between %s and %s" % (MIN_HOURS, MAX_HOURS))
    key, estimate = raw.get("allowance_key"), raw.get("estimate_gpu_hours")
    _need(key in allowances, where + "allowance_key is not in the allowances file")
    _need(isinstance(estimate, (int, float)) and estimate >= 0, where + "estimate_gpu_hours missing")
    _need(estimate > 0 or gpus == 0, where + "a GPU stage needs a positive estimate")
    gate = raw.get("gate")
    _need(gate is None or gate in gate_ids, where + "unknown gate")
    bound = raw.get("manifest", {})
    _need(isinstance(bound, dict), where + "manifest must be an object")
    intervention = bound.get("intervention", "new intervention")
    _need(intervention in INTERVENTIONS, where + "intervention must be one of %s" % (INTERVENTIONS,))
    bind, seeds = bound.get("bind", {}), bound.get("seeds", {})
    _need(isinstance(bind, dict) and all(ID.match(role) and isinstance(path, str) for role, path in bind.items()),
          where + "bind must map role names to paths")
    _need(isinstance(seeds, dict) and all(isinstance(v, int) for v in seeds.values()), where + "bad seeds")
    return {
        "id": name, "experiment": raw["experiment"], "kind": raw.get("kind", "normal"),
        "command": list(command),
        "resources": {"gpus": gpus, "cpus": cpus, "mem_gb": mem, "time_hours": hours},
        "depends_on": _ids(raw.get("depends_on", []), stage_ids, where + "unknown dependency"),
        "gate": gate, "allowance_key": key, "estimate_gpu_hours": estimate,
        "intervention": intervention, "bind": dict(bind), "seeds": dict(seeds),
        "pairing_unit": bound.get("pairing_unit"),
    }


def _gate(raw, stage_ids):
    _need(isinstance(raw, dict), "a gate is not an object")
    name = raw.get("id")
    _need(isinstance(name, str) and ID.match(name), "bad gate id")
    where = "gate %s: " % name
    rule = raw.get("rule")
    _need(isinstance(rule, str) and rule.endswith(".py") and "/" not in rule, where + "rule must be a file name")
    reads = raw.get("reads", [])
    _need(isinstance(reads, list) and all(
        isinstance(item, str) and item and not item.startswith("/") and ".." not in item.split("/")
        for item in reads), where + "reads must be paths under the storage root")
    inputs = _ids(raw.get("inputs", []), stage_ids, where + "unknown input stage")
    _need(inputs, where + "a gate needs at least one input stage")
    return {"id": name, "rule": rule, "inputs": inputs, "reads": list(reads)}


class Graph:
    def __init__(self, raw, allowances):
        _need(isinstance(raw, dict), "graph.json is not an object")
        self.study = raw.get("study", STUDY)
        self.milestone = raw.get("milestone", "M0")
        _need(isinstance(self.milestone, str) and ID.match(self.milestone), "bad milestone")
        self.stage_modules = list(raw.get("stage_modules", []))
        self.allowances = allowances
        raw_stages, raw_gates = raw.get("stages", []), raw.get("gates", [])
        _need(isinstance(raw_stages, list) and isinstance(raw_gates, list), "stages and gates must be lists")
        stage_ids = [s.get("id") if isinstance(s, dict) else None for s in raw_stages]
        gate_ids = [g.get("id") if isinstance(g, dict) else None for g in raw_gates]
        _need(len(set(stage_ids)) == len(stage_ids), "duplicate stage id")
        _need(len(set(gate_ids)) == len(gate_ids), "duplicate gate id")
        self.gates = [_gate(g, stage_ids) for g in raw_gates]
        self.stages = [_stage(s, stage_ids, gate_ids, allowances) for s in raw_stages]
        self.by_id = {s["id"]: s for s in self.stages}
        self.gate_by_id = {g["id"]: g for g in self.gates}
        self._check_acyclic()

    def waits_on(self, stage):
        """Stages that must finish first: dependencies, and the inputs of the stage's gate."""
        gate = self.gate_by_id[stage["gate"]]["inputs"] if stage["gate"] else []
        return stage["depends_on"] + gate

    def _check_acyclic(self):
        done, visiting = set(), set()

        def visit(name):
            if name in done:
                return
            _need(name not in visiting, "the graph has a cycle through %s" % name)
            visiting.add(name)
            for before in self.waits_on(self.by_id[name]):
                visit(before)
            visiting.discard(name)
            done.add(name)

        for stage in self.stages:
            visit(stage["id"])


def load(paths):
    try:
        with open(paths.graph) as handle:
            raw = json.load(handle)
    except (OSError, ValueError) as error:
        raise GraphError("graph.json cannot be read: %s" % type(error).__name__)
    return Graph(raw, load_allowances(paths))
