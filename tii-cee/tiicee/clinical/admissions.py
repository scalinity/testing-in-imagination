"""Admission records of a prepared split file: their exact bytes and a content fingerprint.

A split file is cut into records without re-encoding anything, so a file written from a subset of
these records holds every column of every kept admission byte for byte, Patient History Summary
included. Standard library only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass

ID_COLUMN, LABEL_COLUMN = "Patient ID", "Label"


@dataclass(frozen=True)
class Admission:
    patient_id: int
    label: str
    fingerprint: str  # SHA-256 over the column names and this record's field values
    raw: bytes        # the record exactly as its file holds it, ending in a line terminator


@dataclass(frozen=True)
class Table:
    header: bytes     # the header line exactly as the file holds it
    admissions: tuple

    def subset(self, members) -> bytes:
        """A split file holding only these admissions, in the order given."""
        return self.header + b"".join(member.raw for member in members)


def read(data: bytes) -> Table:
    """Parse a split file's bytes. The file is never re-encoded: Latin-1 maps each byte to one
    character, and the CSV structure (commas, quotes, line ends) is ASCII in any encoding."""
    csv.field_size_limit(2 ** 31 - 1)  # one field can hold several radiology reports
    lines = list(io.StringIO(data.decode("latin-1"), newline=""))
    reader = csv.reader(lines)
    names, header, admissions, start = None, None, [], 0
    for fields in reader:
        raw, start = "".join(lines[start:reader.line_num]), reader.line_num
        if not fields:
            continue  # a blank line, which the upstream dataset reader skips as well
        if not raw.endswith(("\n", "\r")):
            raw += "\n"
        if names is None:
            names, header = fields, raw.encode("latin-1")
            id_at, label_at = names.index(ID_COLUMN), names.index(LABEL_COLUMN)
            continue
        if len(fields) != len(names):
            raise ValueError("a record's field count differs from the header's")
        fingerprint = hashlib.sha256(json.dumps([names, fields]).encode()).hexdigest()
        admissions.append(Admission(int(fields[id_at]), fields[label_at], fingerprint, raw.encode("latin-1")))
    if names is None:
        raise ValueError("a split file has no header")
    return Table(header, tuple(admissions))


def require_unique(admissions) -> None:
    """Refuse a set in which one admission, or one patient_id, appears twice."""
    if len({a.patient_id for a in admissions}) != len(admissions) or \
            len({a.fingerprint for a in admissions}) != len(admissions):
        raise ValueError("an admission appears more than once")
