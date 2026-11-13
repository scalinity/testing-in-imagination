"""Export guard: the only way a file enters $TII/export/approved/, the tree the Mac copies.

    python -m tiicee.pipeline.export_guard approve <aggregate.json> [name.json]

The rules start from the project's existing guard, `assert_aggregate` in
replication-of-original/daniel-escalante/replication/manifest.py: its FORBIDDEN_KEYS set is copied
below and applied the same way (any dictionary key, at any depth, compared in lower case). That
module cannot be imported here because it needs third-party packages. This guard then adds:
  - every key, and every string value, is a short identifier (no spaces, so no sentences);
  - a value is a number, a boolean, null, an identifier string, or a list or object of those
    (the study's few fixed multi-word names, such as the status SKIPPED BY GATE, also pass);
  - no key resembles row, rows, patient, subject, hadm, stay, text, note, prompt, generation, hpi.

surface_check applies the same idea to everything the Mac reads: events.tsv, STATUS.md,
status.json, progress/ and export/approved/ may hold names, counts, numbers and codes only. The
shapes of status.json and of a progress file are those of monitor/DATA_CONTRACT.md. A refusal
never echoes the offending key or value. Standard library only and Python 3.9 compatible.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

from .paths import ID, Paths
from .util import write_text

# Copied from replication/manifest.py (FORBIDDEN_KEYS). Keep the two in step.
FORBIDDEN_KEYS = {"patient_id", "patient id", "hadm_id", "subject_id", "patient history",
                  "patient_history", "patient history summary", "laboratory tests", "radiology",
                  "physical examination", "physical_examination", "test_results", "generations",
                  "completions", "prompts", "rows"}
FORBIDDEN_WORDS = ("row", "patient", "subject", "hadm", "stay", "text", "note", "prompt", "generation", "hpi")

KEY = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
# A name, code, number, digest, timestamp or relative path: one token, no whitespace.
NAME_TEXT = r"-?[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,127}"
NAME = re.compile(NAME_TEXT + r"\Z")
PAIRS_TEXT = r"[a-z][a-z0-9_]{0,31}=%s(?: [a-z][a-z0-9_]{0,31}=%s)*" % (NAME_TEXT, NAME_TEXT)
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.json\Z")

EVENT_KINDS = ("stage completed", "gate decided", "blocked", "failure", "study completed",
               "all remaining items blocked")
TIME_TEXT = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}"
TIME = re.compile(TIME_TEXT + r"\Z")
EVENT = re.compile(r"%s\t(?:%s)\t%s\Z" % (TIME_TEXT, "|".join(EVENT_KINDS), PAIRS_TEXT))

# STATUS.md, as watcher.py writes it.
STATUS_TITLE = "# TII-CEE-1 watcher status"
STATUS_HEADINGS = ("Stages by status", "Running and queued jobs", "GPU-hours by experiment",
                   "Open BLOCKED items", "Latest gate records", "Latest events")
STATUS_FIELDS = ("updated", "milestone", "cycle", "authorization", "addendum_1", "scheduler", "final")
# The study's fixed multi-word names: the event kinds and one status. Nothing else has a space.
STATUS_PHRASES = EVENT_KINDS + ("SKIPPED BY GATE",)
_FIELD = re.compile(r"(?:%s): %s\Z" % ("|".join(STATUS_FIELDS), NAME_TEXT))
_CELL = re.compile(r"(?:|-+|%s|%s|%s)\Z" % ("|".join(STATUS_PHRASES), NAME_TEXT, PAIRS_TEXT))

# status.json and progress/<stage>.json, as monitor/DATA_CONTRACT.md lays them out.
STATUS_SCHEMA = "tii-status/1"
STATUS_JSON_SHAPE = {"schema": str, "generated_at": str, "milestone": str, "watcher": dict,
                     "authorization": dict, "gpu": dict, "counts": dict, "experiments": list,
                     "stages": list, "gates": list, "blocked": list}
PROGRESS_KEYS = {"stage", "unit", "done", "total", "updated_at"}


class Refused(ValueError):
    """code names the rule that failed; it never carries the content that failed it."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _resembles_forbidden(key):
    words = re.findall(r"[a-z]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower())
    return any(word.startswith(FORBIDDEN_WORDS) for word in words)


def check_aggregate(value, top=True):
    """Raise Refused unless value is an aggregate: names, counts, numbers and codes only."""
    if top and not isinstance(value, dict):
        raise Refused("not_an_object")
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in STATUS_PHRASES:
                if not isinstance(key, str) or key.lower() in FORBIDDEN_KEYS or _resembles_forbidden(key):
                    raise Refused("forbidden_key")
                if not KEY.match(key):
                    raise Refused("key_not_identifier")
            check_aggregate(item, top=False)
    elif isinstance(value, list):
        for item in value:
            check_aggregate(item, top=False)
    elif isinstance(value, str):
        if value not in STATUS_PHRASES and not NAME.match(value):
            raise Refused("string_not_identifier")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise Refused("number_not_finite")
    elif not (value is None or isinstance(value, (bool, int))):
        raise Refused("value_type")


def _load(path):
    def refuse_constant(_):
        raise Refused("number_not_finite")

    try:
        with open(path) as handle:
            return json.load(handle, parse_constant=refuse_constant)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        if isinstance(error, Refused):
            raise
        raise Refused("not_json")


def approve(paths, name, value):
    """Check value and write it to export/approved/<name>. name may have identifier subfolders."""
    parts = name.split("/")
    if not FILE_NAME.match(parts[-1]) or not all(KEY.match(part) for part in parts[:-1]):
        raise Refused("file_name")
    check_aggregate(value)
    return write_text(paths.approved.joinpath(*parts),
                      json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", paths.staging)


def check_status_json(value):
    """status.json: the contract's top-level fields and nothing else, holding names and numbers only."""
    check_aggregate(value)
    if (set(value) != set(STATUS_JSON_SHAPE) or value["schema"] != STATUS_SCHEMA
            or not all(isinstance(value[key], kind) for key, kind in STATUS_JSON_SHAPE.items())
            or not TIME.match(value["generated_at"])
            or not all(isinstance(entry, dict) for key in ("experiments", "stages", "gates", "blocked")
                       for entry in value[key])):
        raise Refused("status_shape")


def check_progress(value, stage_id):
    """progress/<stage>.json: {"stage", "unit", "done", "total", "updated_at"}, counts only."""
    check_aggregate(value)
    if (set(value) != PROGRESS_KEYS or value["stage"] != stage_id or not ID.match(str(value["unit"]))
            or not all(type(value[key]) is int and value[key] >= 0 for key in ("done", "total"))
            or not TIME.match(str(value["updated_at"]))):
        raise Refused("progress_shape")


def event_ok(line):
    return bool(EVENT.match(line))


def status_violations(text):
    """Line numbers of STATUS.md lines that are not one of the shapes the watcher writes."""
    bad = []
    for number, line in enumerate(text.split("\n"), 1):
        if line in ("", STATUS_TITLE) or _FIELD.match(line):
            continue
        if line.startswith("## ") and line[3:] in STATUS_HEADINGS:
            continue
        if line.startswith("|") and line.endswith("|") and len(line) > 1 and all(
                _CELL.match(cell.strip()) for cell in line[1:-1].split("|")):
            continue
        bad.append(number)
    return bad


def file_violations(paths):
    """Files under export/approved/ and state/progress/ that should not be there, as (path, code)."""
    bad = []
    for base, dirs, files in os.walk(paths.approved, followlinks=False):
        dirs.sort()
        for name in sorted(files + [d for d in dirs if os.path.islink(os.path.join(base, d))]):
            path = Path(base) / name
            try:
                if path.is_symlink():
                    raise Refused("not_a_regular_file")
                if not FILE_NAME.match(name):
                    raise Refused("file_name")
                check_aggregate(_load(path))
            except Refused as refused:
                bad.append((path, refused.code))
    for path in sorted(paths.progress.iterdir()) if paths.progress.is_dir() else []:
        try:
            if path.is_symlink() or not path.is_file():
                raise Refused("not_a_regular_file")
            if path.suffix != ".json" or not ID.match(path.stem):
                raise Refused("file_name")
            check_progress(_load(path), path.stem)
        except Refused as refused:
            bad.append((path, refused.code))
    return bad


def surface_check(paths):
    """Everything the Mac reads. Returns violations as "file:line:code" strings; empty means clean."""
    found = []
    if paths.events.exists():
        for number, line in enumerate(paths.events.read_text(errors="replace").split("\n")[:-1], 1):
            if not event_ok(line):
                found.append("events.tsv:%d:not_an_event" % number)
    if paths.status.exists():
        for number in status_violations(paths.status.read_text(errors="replace")):
            found.append("STATUS.md:%d:not_a_status_line" % number)
    if paths.status_json.exists():
        try:
            check_status_json(_load(paths.status_json))
        except Refused as refused:
            found.append("status.json:1:%s" % refused.code)
    for index, (path, code) in enumerate(file_violations(paths), 1):
        tree = "state/progress" if path.parent == paths.progress else "export/approved"
        found.append("%s:%d:%s" % (tree, index, code))
    return found


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (2, 3) or argv[0] != "approve":
        print("usage: python -m tiicee.pipeline.export_guard approve <aggregate.json> [name.json]", file=sys.stderr)
        return 2
    try:
        target = approve(Paths(), argv[2] if len(argv) == 3 else Path(argv[1]).name, _load(argv[1]))
    except Refused as refused:
        print("refused %s" % refused.code)
        return 1
    print("approved %s" % target.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
