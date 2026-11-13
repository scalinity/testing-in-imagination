"""Where the study's files live on HiPerGator (WATCHER_DESIGN.md section 2).

Standard library only and Python 3.9 compatible: the watcher imports this under /usr/bin/python3.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

STUDY = "TII-CEE-1"
DEFAULT_ROOT = "/blue/cai6734/daniel.escalante/private/tii-cee-1"  # AUTHORIZATION.md section 3
ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")  # stage, gate and experiment names


class Paths:
    """root is $TII (env TII_ROOT); code is the code this process runs from (env TII_CODE).

    On the cluster TII_CODE is a code export, $TII/code/<commit>/ (deploy.py): the watcher is given
    $TII/code/current and a stage is given the export its manifest names. Both are resolved once,
    here, so one cycle or one stage sees one export even if `current` moves meanwhile.
    """

    def __init__(self, root=None, code=None):
        self.root = Path(root or os.environ.get("TII_ROOT") or DEFAULT_ROOT)
        self.code = Path(os.path.realpath(code or os.environ.get("TII_CODE") or Path(__file__).resolve().parents[3]))
        self.current = self.root / "code" / "current"

        self.graph = self.root / "pipeline" / "graph.json"
        self.state = self.root / "state"
        self.stages = self.state / "stages"
        self.lock = self.state / "lock"
        self.runlog = self.state / "runlog.jsonl"
        self.events = self.state / "events.tsv"
        self.status = self.state / "STATUS.md"
        self.status_json = self.state / "status.json"   # for the monitor (monitor/DATA_CONTRACT.md)
        self.progress = self.state / "progress"         # one small file per running stage
        # Where a write into progress/ or export/approved/ is prepared before it is renamed in.
        # The watcher polices those two folders and moves anything unexpected out of them, so a
        # file that is still being written must never be seen there.
        self.staging = self.state / "tmp"
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
        # The authorization is a live fact, not part of a stage's frozen code: everything reads it
        # where the newest deployment put it, whichever export the reader itself runs from.
        newest = Path(os.path.realpath(self.current)) if self.current.exists() else self.code
        self.study_dir = newest / "reports" / "research-advancement"

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

    def progress_file(self, stage_id):
        return self.progress / (stage_id + ".json")

    def gate_record(self, gate_id):
        return self.gates / (gate_id + ".json")

    def stage_dir(self, stage_id):
        """Private work directory of one stage: lock, checkpoint, item log, outputs."""
        return self.private / "stages" / stage_id

    def stage_out(self, stage_id):
        """Published outputs of a completed stage."""
        return self.stage_dir(stage_id) / "out"
