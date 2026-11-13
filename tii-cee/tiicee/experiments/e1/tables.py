"""Per-seed result tables as CSV: one file per (table, seed, variant), aggregates only."""
from __future__ import annotations

import csv
from pathlib import Path


def write(path: Path, rows: list) -> None:
    """Write rows (dicts) with the union of their keys as the header, in first-seen order. None is an empty cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else repr(v) if isinstance(v, float) else v) for k, v in row.items()})


def _parse(text: str):
    if text == "":
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def read(path: Path) -> list:
    with open(path, newline="") as handle:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def cell_file(out: Path, table: str, seed: int, variant: str) -> Path:
    return Path(out) / "tables" / table / f"seed{seed:02d}_{variant}.csv"


def read_table(out: Path, table: str) -> list:
    """Every row of a table across its per-seed files, in file-name order."""
    rows = []
    for path in sorted((Path(out) / "tables" / table).glob("*.csv")):
        rows.extend(read(path))
    return rows
