"""Where the study's files live on HiPerGator (WATCHER_DESIGN.md section 2).

Standard library only and Python 3.9 compatible: the watcher imports this under /usr/bin/python3.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

STUDY = "TII-CEE-1"
DEFAULT_ROOT = "/blue/cai6734/daniel.escalante/private/tii-cee-1"  # AUTHORIZATION.md section 3
ID = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,47}\Z")  # stage, gate and experiment names


class Paths:
    """root is $TII (env TII_ROOT); code is the repository checkout (env TII_CODE)."""

    def __init__(self, root=None, code=None):
        self.root = Path(root or os.environ.get("TII_ROOT") or DEFAULT_ROOT)
        self.code = Path(code or os.environ.get("TII_CODE") or Path(__file__).resolve().parents[3])

        self.graph = self.root / "pipeline" / "graph.json"
        self.state = self.root / "state"
        self.stages = self.state / "stages"
        self.lock = self.state / "lock"
        self.runlog = self.state / "runlog.jsonl"
        self.events = self.state / "events.tsv"
        self.status = self.state / "STATUS.md"
        self.blocked_md = self.state / "BLOCKED.md"
        self.blocked = self.state / "blocked.json"
        self.watcher = self.state / "watcher.json"
        self.manifests = self.state / "manifests"
        self.ledger = self.state / "ledger"
        self.logs = self.state / "logs"
        self.scripts = self.state / "sbatch"
        self.gates = self.root / "gates"
        self.approved = self.root / "export" / "approved"
        self.private = self.root / "private"
        self.quarantine = self.private / "quarantine"

        self.package = self.code / "tii-cee"
        self.gate_rules = self.package / "gates"
        self.allowances = self.package / "tiicee" / "pipeline" / "allowances.json"
        self.study_dir = self.code / "reports" / "research-advancement"

    def stage_state(self, stage_id):
        return self.stages / (stage_id + ".json")

    def done_marker(self, stage_id):
        return self.stages / (stage_id + ".done")

    def refusal(self, stage_id):
        return self.stages / (stage_id + ".refusal.json")

    def manifest(self, stage_id):
        return self.manifests / (stage_id + ".json")

    def ledger_entry(self, stage_id):
        return self.ledger / (stage_id + ".json")

    def gate_record(self, gate_id):
        return self.gates / (gate_id + ".json")

    def stage_dir(self, stage_id):
        """Private work directory of one stage: lock, checkpoint, item log, outputs."""
        return self.private / "stages" / stage_id

    def stage_out(self, stage_id):
        """Published outputs of a completed stage."""
        return self.stage_dir(stage_id) / "out"
