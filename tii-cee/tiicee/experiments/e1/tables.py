"""Per-seed result tables as gzipped CSV: one file per (table, seed, variant), aggregates only.

Tables that the rules or the figure step read live under <out>/tables and are committed. The large
detail tables live under <out>/large, which git ignores; the manifest records every file's SHA-256.
Files are written with a fixed gzip header, so the same rows give the same bytes.
"""
from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

LARGE = ("D_horizon",)      # every arm at every horizon with the terms of 5.12 (iii); D_final holds what the rules read
SUFFIX = ".csv.gz"


def write(path: Path, rows: list) -> None:
    """Write rows (dicts) with the union of their keys as the header, in first-seen order. None is an empty cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(dict.fromkeys(key for row in rows for key in row))
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=header)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    data = text.getvalue().encode("utf-8")
    if path.name.endswith(".gz"):
        with open(path, "wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as packed:
            packed.write(data)
    else:
        path.write_bytes(data)


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
    opener = gzip.open if Path(path).name.endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def folder(out: Path, table: str) -> Path:
    return Path(out) / ("large" if table in LARGE else "tables") / table


def cell_file(out: Path, table: str, seed: int, variant: str) -> Path:
    return folder(out, table) / f"seed{seed:02d}_{variant}{SUFFIX}"


def read_table(out: Path, table: str) -> list:
    """Every row of a table across its per-seed files, in file-name order."""
    rows = []
    for path in sorted(folder(out, table).glob("*" + SUFFIX)):
        rows.extend(read(path))
    return rows


def files(out: Path) -> list:
    """Every table file under <out>, committed ones first."""
    out = Path(out)
    return [p for name in ("tables", "large") if (out / name).exists() for p in sorted((out / name).rglob("*" + SUFFIX))]
