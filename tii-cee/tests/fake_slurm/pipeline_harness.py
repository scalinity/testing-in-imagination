"""A throwaway study for the pipeline tests.

Builds, under a temporary directory: a storage root ($TII), a copy of the code the watcher and the
stages run (so a test can change an allowance or rename AUTHORIZATION.md without touching the
repository), copies of the authorization files, and a fake Slurm. The watcher runs as a
subprocess under TII_TEST_WATCHER_PYTHON when that is set (point it at a Python 3.9 to exercise
the watcher on the interpreter it will meet on the cluster), else under the test interpreter.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import fakeslurm
from tiicee.pipeline.authorization import AUTHORIZATION_SHA256
from tiicee.pipeline.paths import Paths
from tiicee.pipeline.util import sha256_file

FAKE = Path(__file__).resolve().parent
PACKAGE = FAKE.parents[1]
DOCS = PACKAGE.parent / "reports" / "research-advancement"
WATCHER_PYTHON = os.environ.get("TII_TEST_WATCHER_PYTHON") or sys.executable



def stage(name, command=("noop",), depends_on=(), gate=None, gpus=0, experiment="benches",
          estimate=None, kind="normal", bind=None, mem_gb=1, time_hours=1):
    spec = {"id": name, "experiment": experiment, "kind": kind, "command": list(command),
            "resources": {"gpus": gpus, "cpus": 1, "mem_gb": mem_gb, "time_hours": time_hours},
            "depends_on": list(depends_on), "allowance_key": experiment,
            "estimate_gpu_hours": (1 if gpus else 0) if estimate is None else estimate}
    if gate:
        spec["gate"] = gate
    if bind:
        spec["manifest"] = {"bind": bind}
    return spec


class Study:
    def __init__(self, tmp_path, stages=(), gates=()):
        self.code, self.root = tmp_path / "code", tmp_path / "tii"
        package = self.code / "tii-cee"
        skip = shutil.ignore_patterns("__pycache__")
        shutil.copytree(PACKAGE / "tiicee" / "pipeline", package / "tiicee" / "pipeline", ignore=skip)
        shutil.copy(PACKAGE / "tiicee" / "__init__.py", package / "tiicee" / "__init__.py")
        shutil.copytree(PACKAGE / "gates", package / "gates", ignore=skip)
        self.docs = self.code / "reports" / "research-advancement"
        self.docs.mkdir(parents=True)
        for name in ("AUTHORIZATION.md", "AUTHORIZATION_ADDENDUM_1.md"):
            shutil.copy(DOCS / name, self.docs / name)
        # The study's own signature file, so a test decides whether the addendum is signed
        # whatever the repository's SIGNATURE.md says today.
        (self.docs / "SIGNATURE.md").write_text("AUTHORIZATION.md, SHA-256 %s\n" % AUTHORIZATION_SHA256)
        self.paths = Paths(self.root, self.code)
        self.cluster = fakeslurm.Cluster(tmp_path / "slurm")
        self.env = dict(os.environ, PATH="%s%s%s" % (FAKE, os.pathsep, os.environ["PATH"]),
                        FAKE_SLURM_DIR=str(self.cluster.dir), TII_ROOT=str(self.root),
                        TII_CODE=str(self.code), TII_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
        self.write_graph(stages, gates)

    def close(self):
        for process, _ in self.cluster.frozen.values():
            process.kill()
            process.wait()

    def write_graph(self, stages, gates=()):
        self.paths.graph.parent.mkdir(parents=True, exist_ok=True)
        self.paths.graph.write_text(json.dumps({"study": "TII-CEE-1", "milestone": "M0",
                                                "stages": list(stages), "gates": list(gates)}))

    def set_allowance(self, key, gpu_hours):
        allowances = json.loads(self.paths.allowances.read_text())
        allowances["experiments"][key]["allowance_gpu_hours"] = gpu_hours
        self.paths.allowances.write_text(json.dumps(allowances))

    def sign_addendum(self):
        """What Daniel's signature of addendum 1 would add to SIGNATURE.md: its SHA-256."""
        with open(self.docs / "SIGNATURE.md", "a") as handle:
            handle.write("\nAddendum 1, SHA-256 %s\n" % sha256_file(self.docs / "AUTHORIZATION_ADDENDUM_1.md"))

    # ---- driving -----------------------------------------------------------------------------

    def start_cycle(self, **extra):
        return subprocess.Popen([WATCHER_PYTHON, "-m", "tiicee.pipeline.watcher"], cwd=self.code / "tii-cee",
                                env=dict(self.env, **extra), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

    def cycle(self, expect=0, **extra):
        """One watcher cycle. Returns what it printed."""
        process = self.start_cycle(**extra)
        output = process.communicate(timeout=120)[0]
        assert process.returncode == expect, output
        return output

    def advance(self):
        """Let the fake cluster apply the next planned outcome to every live job."""
        self.cluster.run()

    def run_to_end(self, limit=20):
        for _ in range(limit):
            self.cycle()
            if self.final():
                return
            self.advance()
        raise AssertionError("the study did not reach a final status in %d cycles" % limit)

    # ---- reading what the watcher wrote ----------------------------------------------------------

    def state(self, name):
        return json.loads(self.paths.stage_state(name).read_text())

    def status(self, name):
        return self.state(name)["status"]

    def final(self):
        return json.loads(self.paths.watcher.read_text()).get("final") if self.paths.watcher.exists() else None

    def events(self, kind=None):
        """Queued events as (kind, {key: value})."""
        if not self.paths.events.exists():
            return []
        found = []
        for line in self.paths.events.read_text().splitlines():
            _, event_kind, text = line.split("\t")
            if kind in (None, event_kind):
                found.append((event_kind, dict(pair.split("=", 1) for pair in text.split(" "))))
        return found

    def runlog(self, event):
        if not self.paths.runlog.exists():
            return []
        lines = [json.loads(line) for line in self.paths.runlog.read_text().splitlines()]
        return [line for line in lines if line["event"] == event]

    def blocked(self):
        """Open BLOCKED items by key."""
        book = json.loads(self.paths.blocked.read_text()) if self.paths.blocked.exists() else {"items": []}
        return {item["key"]: item for item in book["items"] if item["closed"] is None}

    def calls(self, command):
        return [call for call in self.cluster.calls() if call[0] == command]

    def script(self, name, attempt=1):
        return (self.paths.scripts / ("tii-%s.%d.sbatch" % (name, attempt))).read_text()

    def items(self, name):
        path = self.paths.stage_dir(name) / "items.jsonl"
        return [json.loads(line)["key"] for line in path.read_text().splitlines()] if path.exists() else []
