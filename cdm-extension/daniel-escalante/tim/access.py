"""The only gate to patient data: the stage x split table, admissions and episode logs (SPEC §2)."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from tim.enums import DISEASES
from tim.events import TimError

_TEST_ONLY = {"test": "admitted"}
ACCESS = {
    "validate": {"train": "digests", "val": "digests", "test": "baseline"},
    "smoke": {"val": "listed"},
    "bench-a": {"val": "listed"},
    "bench-b": {"val": "listed"},
    "val-a": {"val": "full"},
    "analyze-val-a": {"val": "full"},
    "hpi-audit": {"train": "full", "val": "full"},
    "wm-fit": {"train": "full", "val": "full"},
    "inventory": {"train": "full", "val": "full"},
    "leakage": {"train": "full", "val": "full"},
    "val-b": {"val": "full"},
    "analyze-val-b": {"val": "full"},
    **{stage: _TEST_ONLY for stage in ("test-a", "analyze-a", "test-a2", "analyze-a2", "test-b", "analyze-b")},
}
BENCH_PER_CLASS, SMOKE_PATIENTS = 6, 2


def mode(stage: str, split: str) -> str:
    found = ACCESS.get(stage, {}).get(split)
    if found is None:
        raise TimError("access_denied", f"{stage} may not read {split}")
    return found


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _order(tag: str, patient_id) -> str:
    return hashlib.sha256(f"{tag}|{patient_id}".encode()).hexdigest()


def listed(stage: str, ids, labels) -> list:
    """Smoke: two val patients; bench: six per class; both in sha256 order (SPEC §9)."""
    if stage == "smoke":
        return sorted(ids, key=lambda p: _order("smoke", p))[:SMOKE_PATIENTS]
    chosen = []
    for disease in DISEASES:
        members = [p for p, label in zip(ids, labels) if label == disease]
        chosen.extend(sorted(members, key=lambda p: _order("bench", p))[:BENCH_PER_CLASS])
    return sorted(chosen, key=lambda p: _order("bench", p))


def _replace(path: Path, write) -> None:
    """Write through a uniquely named temporary file, then rename it into place, so a job
    reading the same private file while another job rewrites it sees a complete file."""
    import socket
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    write(temporary)
    os.replace(temporary, path)


def _record_ids(settings, split, ids):
    folder = settings.work_dir / "private" / "ids"
    folder.mkdir(parents=True, exist_ok=True)
    _replace(folder / f"{split}.txt", lambda tmp: tmp.write_text("".join(f"{i}\n" for i in ids)))


def baseline_cfg(settings):
    from replication.pipeline import cfg_for
    return cfg_for(settings.baseline(), "primary")


def open_split(settings, stage: str, split: str):
    """The only reader of split data. Opens the one authorized split file and nothing else.

    The stage gets a private data directory holding that split (only its listed patients, for
    smoke and bench) and the lab mapping; GPU jobs hand that directory to the baseline evaluator.
    """
    found = mode(stage, split)
    if found in ("digests", "baseline"):
        raise TimError("access_denied", f"{stage} reads only digests or disclosed aggregates of {split}")
    from replication.data import IMAGING_TESTS, LAB_TESTS
    from replication.pipeline import prepared
    from src.dataset.mimic_cdm_dataset import MIMICCDMDataset
    dataset_cfg = baseline_cfg(settings).dataset
    if (tuple(dataset_cfg.lab_tests) != LAB_TESTS or tuple(dataset_cfg.imaging_tests) != IMAGING_TESTS
            or tuple(dataset_cfg.other_tests) != ("Physical Examination",)):
        raise TimError("baseline_mismatch", "configured test vocabulary differs from the pinned schema")
    source = prepared(settings.baseline())
    with (source / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
        header, *records = list(csv.reader(handle))
    ids = [int(r[header.index("Patient ID")]) for r in records]
    _record_ids(settings, split, ids)
    if found == "listed":
        position = {pid: i for i, pid in enumerate(ids)}
        chosen = listed(stage, ids, [r[header.index("Label")] for r in records])
        records = [records[position[p]] for p in chosen]
    data_dir = settings.work_dir / "private" / "prepared" / stage
    data_dir.mkdir(parents=True, exist_ok=True)
    _replace(data_dir / "lab_test_mapping.csv", lambda tmp: shutil.copyfile(source / "lab_test_mapping.csv", tmp))

    def write_split(tmp):
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([header, *records])
    _replace(data_dir / f"{split}.csv", write_split)
    dataset = MIMICCDMDataset(data_file=str(data_dir / f"{split}.csv"),
                              lab_test_mapping_file=str(data_dir / "lab_test_mapping.csv"),
                              lab_tests=list(dataset_cfg.lab_tests), imaging_tests=list(dataset_cfg.imaging_tests),
                              other_tests=list(dataset_cfg.other_tests))
    dataset.data_dir = data_dir
    return dataset


def official_rows(settings, stage: str) -> list[dict]:
    """The baseline's primary test rows: the comparator, for the stages the table allows."""
    if mode(stage, "test") not in ("baseline", "admitted"):
        raise TimError("access_denied", f"{stage} may not read the official rows")
    from replication.pipeline import _rows_path, private_read
    rows = private_read(_rows_path(settings.baseline(), "parity", 269, "primary", "test"), settings.source)
    _record_ids(settings, "test", [row["patient_id"] for row in rows])
    return rows


def check_admission_path(settings, protocol: dict) -> None:
    """A credentialed run must use the admission directory frozen in the protocol."""
    if not settings.synthetic and Path(protocol["admission_dir"]).resolve() != settings.admission_dir:
        raise TimError("admission_refused", "TIM_ADMISSION_DIR differs from the protocol's admission path")


def run_key(protocol_hash, stage, variant, split, commit) -> str:
    """Test admissions are keyed without the commit, so a code change cannot reopen them."""
    key = f"{protocol_hash[:16]}:{stage}:{variant}:primary"
    return key if split == "test" else f"{key}:{commit[:12]}"


class Ledger:
    """Append-only record of admissions, completed patients, interruptions and resumptions."""

    def __init__(self, admission_dir: Path):
        self.path = Path(admission_dir) / "ledger.jsonl"

    def record(self, event: str, **fields) -> None:
        from replication.manifest import eastern_now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps({"event": event, "time": eastern_now(), **fields}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def entries(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []


class Episodes:
    """One run's admission marker and immutable per-patient log; resumes at the first incomplete patient.

    A run holds an exclusive lock on its key while it writes, so a second job cannot resume an
    admission whose writer is still alive. Crash states are reconciled without re-evaluation:
    a row written before its ledger record is recovered, a truncated final line is dropped (its
    patient was never recorded as complete), and a run with every row but no finish just finishes.
    """

    def __init__(self, settings, protocol_hash, stage, variant, split, commit, total):
        self.key = run_key(protocol_hash, stage, variant, split, commit)
        name = self.key.replace(":", "_")
        self.split, self.total = split, total
        self.ledger = Ledger(settings.admission_dir)
        self.marker = settings.admission_dir / "markers" / f"{name}.json"
        self.lock_path = settings.admission_dir / "locks" / f"{name}.lock"
        self.log = settings.work_dir / "private" / "episodes" / f"{name}.jsonl"
        self.done, self._lock = set(), None

    def _acquire(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise TimError("admission_refused", "another job is writing this run") from None
        self._lock = handle

    def close(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def _lines(self):
        if not self.log.exists():
            return []
        text = self.log.read_text()
        if text and not text.endswith("\n"):
            kept = text[:text.rfind("\n") + 1]
            self.log.write_text(kept)
            self.ledger.record("dropped_partial_entry", key=self.key)
            text = kept
        lines = [json.loads(line) for line in text.splitlines()]
        if any(digest(line["row"]) != line["sha256"] for line in lines):
            raise TimError("resume_mismatch", "an episode log entry does not match its digest")
        return lines

    def open(self) -> list[dict]:
        self._acquire()
        try:
            return self._open()
        except BaseException:
            self.close()
            raise

    def _open(self) -> list[dict]:
        entries = [e for e in self.ledger.entries() if e.get("key") == self.key]
        recorded = [e["sha256"] for e in entries if e["event"] == "patient"]
        if self.marker.exists():
            if json.loads(self.marker.read_text())["log"] != str(self.log):
                raise TimError("resume_mismatch", "the admitted run's episode log lives in another work directory")
            if any(e["event"] == "completed" for e in entries):
                raise TimError("admission_refused", "this run already completed")
            lines = self._lines()
            digests = [line["sha256"] for line in lines]
            if digests[:len(recorded)] != recorded or len(digests) > len(recorded) + 1:
                raise TimError("resume_mismatch", "the episode log differs from the ledger")
            if len(digests) == len(recorded) + 1:
                self.ledger.record("patient", key=self.key, sha256=digests[-1], recovered=True)
            self.ledger.record("interrupted", key=self.key, completed=len(lines))
            self.ledger.record("resumed", key=self.key)
            self.done = {line["patient_id"] for line in lines}
            return [line["row"] for line in lines]
        if self.log.exists() or recorded:
            raise TimError("resume_mismatch", "episodes exist without an admission marker")
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.marker.open("x") as handle:
                json.dump({"key": self.key, "split": self.split, "total": self.total, "log": str(self.log)}, handle)
        except FileExistsError:
            raise TimError("admission_refused", "another job admitted this run") from None
        self.ledger.record("admitted", key=self.key, split=self.split, total=self.total)
        return []

    @property
    def complete(self) -> bool:
        return len(self.done) >= self.total

    def append(self, patient_id, row) -> None:
        line = {"patient_id": patient_id, "row": row, "sha256": digest(row)}
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.ledger.record("patient", key=self.key, sha256=line["sha256"])
        self.done.add(patient_id)

    def finish(self) -> None:
        self.ledger.record("completed", key=self.key, total=self.total)
        self.close()
