"""TupleRecord schema and JSONL writer for world-model training data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO


REQUIRED_KEYS = (
    "episode_id",
    "t",
    "h_t",
    "a_t",
    "o_t1",
    "y_true",
    "source",
    "meta",
)

META_REQUIRED_KEYS = ("split", "available_tests_at_t", "seed")


@dataclass
class TupleRecord:
    episode_id: str
    t: int
    h_t: str
    a_t: str
    o_t1: str
    y_true: str
    source: str = "synthetic"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def validate(self) -> None:
        d = self.to_dict()
        for k in REQUIRED_KEYS:
            if k not in d:
                raise ValueError(f"Missing required key: {k}")
        if self.source != "synthetic":
            raise ValueError(
                f"Stage-1 synthetic tuples must have source='synthetic', "
                f"got {self.source!r}"
            )
        if not isinstance(self.meta, dict):
            raise ValueError("meta must be a dict")
        for k in META_REQUIRED_KEYS:
            if k not in self.meta:
                raise ValueError(f"meta missing required key: {k}")
        if not isinstance(self.h_t, str) or not self.h_t:
            raise ValueError("h_t must be a non-empty string")
        if not isinstance(self.a_t, str) or not self.a_t:
            raise ValueError("a_t must be a non-empty string")
        if not isinstance(self.o_t1, str) or not self.o_t1:
            raise ValueError("o_t1 must be a non-empty string")
        if "UNAVAILABLE" in self.o_t1 and self.source == "synthetic":
            # Soft check: v0 synthetic should not emit UNAVAILABLE; warn via error
            # for training cleanliness. Allow only if explicitly intended later.
            raise ValueError(
                "v0 synthetic tuples must not contain UNAVAILABLE observations"
            )


def validate_record_dict(d: Dict[str, Any]) -> None:
    for k in REQUIRED_KEYS:
        if k not in d:
            raise ValueError(f"Missing required key: {k}")
    if d.get("source") != "synthetic":
        raise ValueError(f"source must be 'synthetic', got {d.get('source')!r}")
    meta = d.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict")
    for k in META_REQUIRED_KEYS:
        if k not in meta:
            raise ValueError(f"meta missing required key: {k}")


class JSONLWriter:
    """Append TupleRecord rows to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[TextIO] = open(self.path, "a", encoding="utf-8")
        self.n_written = 0

    def write(self, record: TupleRecord) -> None:
        record.validate()
        assert self._fh is not None
        self._fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self.n_written += 1

    def write_many(self, records: Iterable[TupleRecord]) -> int:
        n = 0
        for r in records:
            self.write(r)
            n += 1
        return n

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
