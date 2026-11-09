"""The only gate to patient data: the stage x split table, admissions and episode logs (SPEC §2)."""
import hashlib
import json
import os
from pathlib import Path

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


class Selection:
    """A patient subset of an upstream dataset, in the given order."""

    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def _record_ids(settings, split, ids):
    folder = settings.work_dir / "private" / "ids"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{split}.txt").write_text("".join(f"{i}\n" for i in ids))


def baseline_cfg(settings):
    from replication.pipeline import cfg_for
    return cfg_for(settings.baseline(), "primary")


def open_split(settings, stage: str, split: str):
    """The only caller of the replication dataset loader."""
    found = mode(stage, split)
    if found in ("digests", "baseline"):
        raise TimError("access_denied", f"{stage} reads only digests or disclosed aggregates of {split}")
    from replication.data import load_dataset
    from replication.pipeline import prepared
    dataset = load_dataset(prepared(settings.baseline()), split, baseline_cfg(settings))
    ids = [int(i) for i in dataset.data["Patient ID"]]
    _record_ids(settings, split, ids)
    if found != "listed":
        return dataset
    chosen = listed(stage, ids, list(dataset.data["Label"]))
    position = {pid: index for index, pid in enumerate(ids)}
    return Selection(dataset, [position[p] for p in chosen])


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
    """One run's admission marker and immutable per-patient log; resumes at the first incomplete patient."""

    def __init__(self, settings, protocol_hash, stage, variant, split, commit, total):
        self.key = run_key(protocol_hash, stage, variant, split, commit)
        name = self.key.replace(":", "_")
        self.split, self.total = split, total
        self.ledger = Ledger(settings.admission_dir)
        self.marker = settings.admission_dir / "markers" / f"{name}.json"
        self.log = settings.work_dir / "private" / "episodes" / f"{name}.jsonl"
        self.done = set()

    def _lines(self):
        lines = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        if any(digest(line["row"]) != line["sha256"] for line in lines):
            raise TimError("resume_mismatch", "an episode log entry does not match its digest")
        return lines

    def open(self) -> list[dict]:
        entries = [e for e in self.ledger.entries() if e.get("key") == self.key]
        recorded = [e["sha256"] for e in entries if e["event"] == "patient"]
        if self.marker.exists():
            if json.loads(self.marker.read_text())["log"] != str(self.log):
                raise TimError("resume_mismatch", "the admitted run's episode log lives in another work directory")
            lines = self._lines()
            if [line["sha256"] for line in lines] != recorded:
                raise TimError("resume_mismatch", "the episode log differs from the ledger")
            if any(e["event"] == "completed" for e in entries) or len(lines) >= self.total:
                raise TimError("admission_refused", "this run already completed")
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
