"""run_stage: `python -m tiicee.pipeline.stage <stage_id>`, started inside a Slurm job.

The contract of WATCHER_DESIGN.md section 3:
  1. If <id>.done exists and its output digest matches, exit 0.
  2. Verify the manifest. A mismatch exits 20, a failed authorization check exits 30; the reason is
     written to state/stages/<id>.refusal.json. Neither is retried.
  3. Resume from the checkpoint if there is one.
  4. Outputs are written under a temporary name, renamed when complete, and <id>.done is written last.
  5. SIGUSR1 (Slurm sends it ten minutes before the time limit) sets the stop flag; the
     implementation checkpoints and the stage exits 10.
  6. Completed items are keyed by case fingerprint in an append-only log, so none is scored twice.

Exit codes: 0 done, 10 continue, 20 identity refusal, 30 authorization refusal, 40 out of memory;
anything else is a failure. A test-kind stage also writes a one-shot ledger entry before its first
row: a second start under a different manifest is refused, a resume under the same one is the same run.

For the monitor (monitor/DATA_CONTRACT.md) a stage may report counts with ctx.progress(done, total,
unit); they go to state/progress/<id>.json at most once a minute, and once more when the attempt ends.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import sys
import time

from . import authorization, export_guard
from . import graph as graphs
from . import manifest as manifests
from .manifest import Refusal
from .paths import ID, Paths
from .stages import StageInterrupted, resolve
from .util import now, read_json, tree_digest, write_json, write_new_json

DONE, CONTINUE, OUT_OF_MEMORY = 0, 10, 40
PROGRESS_INTERVAL_S = 60


class Context:
    """What a stage implementation is given."""

    def __init__(self, paths, stage, manifest):
        self.paths = paths
        self.stage_id = stage["id"]
        self.args = list(stage["command"][1:])
        self.manifest = manifest
        self.dir = paths.stage_dir(self.stage_id)
        self.out_dir = self.dir / "out.partial"  # renamed to out/ when the stage completes
        self.stop = False                        # set by SIGUSR1
        self._checkpoint = self.dir / "checkpoint.json"
        self._items_path = self.dir / "items.jsonl"
        self._items = self._load_items()
        self._progress = None        # the latest counts the stage reported
        self._progress_at = None     # when they were last written, on the monotonic clock

    def _load_items(self):
        """Read the completed-item log, dropping a last line torn by a kill."""
        items, kept = {}, 0
        if self._items_path.exists():
            data = self._items_path.read_bytes()
            for line in data.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    break
                try:
                    entry = json.loads(line)
                except ValueError:
                    break
                items[entry["key"]] = entry["result"]
                kept += len(line)
            if kept != len(data):
                with open(self._items_path, "r+b") as handle:
                    handle.truncate(kept)
        return items

    def load_checkpoint(self):
        return read_json(self._checkpoint)

    def save_checkpoint(self, state):
        write_json(self._checkpoint, state)

    def is_done(self, key):
        return key in self._items

    def mark_done(self, key, result=None):
        """Record one completed item together with its result, in a single appended line."""
        if key in self._items:
            raise ValueError("item already completed: it would be scored twice")
        with open(self._items_path, "ab") as handle:
            handle.write((json.dumps({"key": key, "result": result}, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        self._items[key] = result

    def results(self):
        return dict(self._items)

    def approve(self, name, value):
        """Publish an aggregate to export/approved/ through the export guard."""
        return export_guard.approve(self.paths, name, value)

    def progress(self, done, total, unit="item"):
        """Report how far the stage is, for the monitor: done of total, in a unit such as "admission".

        Counts only. Call it as often as is convenient; the file is written at most once a minute.
        """
        if type(done) is not int or type(total) is not int or done < 0 or total < 0 or not ID.match(unit):
            raise ValueError("progress is two counts and a unit name")
        self._progress = {"stage": self.stage_id, "unit": unit, "done": done, "total": total}
        if self._progress_at is None or time.monotonic() - self._progress_at >= PROGRESS_INTERVAL_S:
            self.flush_progress()

    def flush_progress(self):
        """Write the latest reported counts now. The runner calls this when the attempt ends."""
        if self._progress is not None:
            write_json(self.paths.progress_file(self.stage_id), dict(self._progress, updated_at=now()),
                       self.paths.staging)
            self._progress, self._progress_at = None, time.monotonic()


def _refuse(paths, stage_id, refusal):
    write_json(paths.refusal(stage_id), {"stage": stage_id, "exit": refusal.exit_code,
                                         "code": refusal.code, "role": refusal.role, "time": now()})
    print("stage %s refused: exit %d %s" % (stage_id, refusal.exit_code, refusal))
    return refusal.exit_code


def _check_account(paths):
    """Inside a Slurm job, the account and QOS it runs on must be ones the authorization allows now."""
    account, qos = os.environ.get("SLURM_JOB_ACCOUNT"), os.environ.get("SLURM_JOB_QOS")
    if account and qos and not any(
            (pool.account, pool.qos) == (account, qos) for pool in authorization.pools(paths.study_dir)):
        raise Refusal(manifests.AUTHORIZATION, "account_not_authorized")


def _check_ledger(paths, stage, manifest):
    """One-shot ledger for a test stage: written before the first test row, never rewritten."""
    entry = {"stage": stage["id"], "manifest_id": manifest["manifest_id"], "time": now(),
             "job": os.environ.get("SLURM_JOB_ID")}
    try:
        write_new_json(paths.ledger_entry(stage["id"]), entry)
    except FileExistsError:
        if read_json(paths.ledger_entry(stage["id"]))["manifest_id"] != manifest["manifest_id"]:
            raise Refusal(manifests.IDENTITY, "test_ledger_mismatch")


def _publish(paths, stage_id, manifest):
    digest, files = tree_digest(paths.stage_out(stage_id))
    write_json(paths.done_marker(stage_id), {"stage": stage_id, "output_sha256": digest, "files": files,
                                             "manifest_id": manifest["manifest_id"], "time": now()})


def run_stage(paths, stage_id):
    graph = graphs.load(paths)
    stage = graph.by_id[stage_id]
    paths.stage_dir(stage_id).mkdir(parents=True, exist_ok=True)
    # One instance per stage. The kernel drops the lock when the process dies, however it dies
    # (the convention of replication/manifest.py's stage_lock).
    lock = os.open(paths.stage_dir(stage_id) / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("stage %s is already running elsewhere" % stage_id)
            return CONTINUE
        return _run(paths, graph, stage)
    finally:
        os.close(lock)


def _run(paths, graph, stage):
    stage_id, out = stage["id"], paths.stage_out(stage["id"])
    try:
        marker = read_json(paths.done_marker(stage_id))
        if marker is not None:
            if out.is_dir() and tree_digest(out)[0] == marker["output_sha256"]:
                return DONE
            raise Refusal(manifests.IDENTITY, "output_changed")

        manifest = manifests.verify(paths, graph, stage)
        _check_account(paths)
        if stage["kind"] == "test":
            _check_ledger(paths, stage, manifest)
    except Refusal as refusal:
        return _refuse(paths, stage_id, refusal)

    if out.is_dir():  # an earlier attempt finished and was killed before it wrote the marker
        _publish(paths, stage_id, manifest)
        return DONE

    context = Context(paths, stage, manifest)
    context.out_dir.mkdir(exist_ok=True)
    previous = signal.signal(signal.SIGUSR1, lambda signum, frame: setattr(context, "stop", True))
    try:
        resolve(stage["command"][0], graph.stage_modules)(context)
    except StageInterrupted:
        print("stage %s checkpointed and will continue in a new job" % stage_id)
        return CONTINUE
    finally:
        signal.signal(signal.SIGUSR1, previous)
        context.flush_progress()
    os.rename(context.out_dir, out)
    _publish(paths, stage_id, manifest)
    return DONE


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m tiicee.pipeline.stage <stage_id>", file=sys.stderr)
        return 2
    try:
        return run_stage(Paths(), argv[0])
    except MemoryError:
        print("stage %s ran out of memory" % argv[0])
        return OUT_OF_MEMORY


if __name__ == "__main__":
    sys.exit(main())
