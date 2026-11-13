"""E10 folds and the E8/E9 data copy: bytes kept, stratification, disjointness, coverage, and test.csv untouched."""
import csv
import io
import json
import os
import sys
from collections import Counter
from contextlib import contextmanager

import pytest

from test_clinical_support import CLASSES, approved, context, labels, prepared, split_bytes
from tiicee.clinical import admissions, folds, sha256

AUDITED, RECORDING = [], [False]


def _audit(event, args):
    if RECORDING[0] and event in ("open", "os.listdir", "os.scandir", "glob.glob", "pathlib.Path.glob",
                                  "shutil.copyfile", "shutil.copytree", "os.rename", "os.link", "os.symlink"):
        AUDITED.append((event, [str(arg) for arg in args if isinstance(arg, (str, bytes)) or hasattr(arg, "__fspath__")]))


sys.addaudithook(_audit)  # audit hooks cannot be removed; this one only records while RECORDING is set


@contextmanager
def recording():
    """Every file opened and every directory listed while the block runs."""
    del AUDITED[:]
    RECORDING[0] = True
    try:
        yield AUDITED
    finally:
        RECORDING[0] = False


def parts(out, fold):
    return {part: admissions.read((out / f"fold-{fold}" / f"{part}.csv").read_bytes()).admissions
            for part in folds.PARTS}


def test_a_split_file_is_cut_and_rejoined_byte_for_byte():
    data = split_bytes(1000, labels(40))
    table = admissions.read(data)
    assert table.subset(table.admissions) == data
    assert len(table.admissions) == 40 and table.admissions[0].patient_id == 1000
    assert admissions.read(data[:-1]).subset(admissions.read(data[:-1]).admissions) == data  # no final newline


def test_fingerprint_follows_content_not_position_or_line_ends():
    data = split_bytes(1000, labels(12))
    table = admissions.read(data)
    reordered = admissions.read(table.subset(reversed(table.admissions)))
    assert {a.fingerprint for a in reordered.admissions} == {a.fingerprint for a in table.admissions}
    crlf = b"Patient ID,Label\r\n7,appendicitis\r\n"
    assert admissions.read(crlf).admissions[0].fingerprint == admissions.read(crlf.replace(b"\r\n", b"\n")).admissions[0].fingerprint
    assert admissions.read(b"Patient ID,Label\n7,appendicitis\n").admissions[0].fingerprint != \
        admissions.read(b"Patient ID,Label\n7,pancreatitis\n").admissions[0].fingerprint


def test_an_admission_listed_twice_is_refused():
    table = admissions.read(split_bytes(1000, labels(4)))
    with pytest.raises(ValueError):
        admissions.require_unique([*table.admissions, table.admissions[0]])


def test_folds_are_stratified_disjoint_and_cover_every_admission_once(tmp_path):
    source, out = prepared(tmp_path, train=1920, val=240), tmp_path / "out"
    summary = folds.write_folds(source, out)
    assert summary["admissions"] == 2160 and summary["heldout"] == [432] * 5
    assert summary["train"] == [1555] * 5 and summary["monitored"] == [173] * 5  # EXPERIMENT_CARDS/E10.md

    union = [a for name in ("train", "val") for a in admissions.read((source / f"{name}.csv").read_bytes()).admissions]
    everyone, totals = {a.fingerprint for a in union}, Counter(a.label for a in union)
    held = Counter()
    for fold in range(5):
        members = parts(out, fold)
        prints = {part: {a.fingerprint for a in group} for part, group in members.items()}
        assert not prints["train"] & prints["heldout"] and not prints["val"] & prints["heldout"]
        assert not prints["train"] & prints["val"]
        assert prints["train"] | prints["val"] | prints["heldout"] == everyone
        held.update(prints["heldout"])
        for label in CLASSES:  # stratified: each diagnosis within one admission of its share
            assert abs(sum(a.label == label for a in members["heldout"]) - totals[label] / 5) < 1
            rest = totals[label] - sum(a.label == label for a in members["heldout"])
            assert abs(sum(a.label == label for a in members["val"]) - rest / 10) < 1
    assert set(held) == everyone and set(held.values()) == {1}

    table = [json.loads(line) for line in (out / "folds.jsonl").read_text().splitlines()]
    assert len(table) == 2160 and {entry["fingerprint"] for entry in table} == everyone
    assert Counter(entry["source"] for entry in table) == {"train": 1920, "val": 240}


def test_fold_of_an_admission_does_not_depend_on_file_order(tmp_path):
    source = prepared(tmp_path)
    union = [a for name in ("train", "val") for a in admissions.read((source / f"{name}.csv").read_bytes()).admissions]
    assert folds.assign(union) == folds.assign(list(reversed(union)))
    assert folds.monitored(union, 2) == folds.monitored(list(reversed(union)), 2)
    assert folds.monitored(union, 2) != folds.monitored(union, 3)  # each fold policy has its own tenth


def test_fold_files_keep_every_column_byte_for_byte(tmp_path):
    source, out = prepared(tmp_path), tmp_path / "out"
    folds.write_folds(source, out)
    original = {a.fingerprint: a.raw for name in ("train", "val")
                for a in admissions.read((source / f"{name}.csv").read_bytes()).admissions}
    for fold in range(5):
        for group in parts(out, fold).values():
            assert all(original[a.fingerprint] == a.raw for a in group)
        assert (out / f"fold-{fold}" / "lab_test_mapping.csv").read_bytes() == (source / "lab_test_mapping.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO((out / "fold-0" / "heldout.csv").read_bytes().decode("utf-8"), newline="")))
    assert all(row["Patient History Summary"] == f"Invented summary {row['Patient ID']}\nsecond line \"quoted\" café \x0c   end"
               for row in rows)


def test_the_integrity_check_reads_the_written_files_and_catches_a_leak(tmp_path):
    source, out = prepared(tmp_path), tmp_path / "out"
    folds.write_folds(source, out)
    union = [a for name in ("train", "val") for a in admissions.read((source / f"{name}.csv").read_bytes()).admissions]
    mapping = (source / "lab_test_mapping.csv").read_bytes()
    folds.check_folds(out, union, mapping)
    leaked = parts(out, 1)["heldout"][0]
    with (out / "fold-1" / "train.csv").open("ab") as handle:
        handle.write(leaked.raw)
    with pytest.raises(ValueError, match="held-out"):
        folds.check_folds(out, union, mapping)


def test_test_csv_is_never_opened_copied_or_listed(tmp_path, monkeypatch):
    source = prepared(tmp_path, with_test=True)  # plants a test.csv beside the files the stages read
    monkeypatch.setenv("CDM_WORK_DIR", str(source.parent))
    fold_ctx, data_ctx = context(tmp_path, "E10_folds"), context(tmp_path, "E8_data")
    with recording() as audited:
        folds.run_folds(fold_ctx)
        folds.run_data(data_ctx)

    touched = [(event, args) for event, args in audited if any(arg.endswith("test.csv") for arg in args)]
    listed = [(event, args) for event, args in audited
              if event != "open" and any(arg.rstrip("/") == str(source) for arg in args)]
    assert touched == [] and listed == []
    assert any(event == "open" and args[0] == str(source / "train.csv") for event, args in audited)  # the hook sees opens
    with recording() as control:  # and it would have seen a listing or an open of test.csv
        os.listdir(source)
        (source / "test.csv").read_bytes()
    assert {event for event, args in control} == {"os.listdir", "open"}
    for ctx in (fold_ctx, data_ctx):
        written = [path for path in ctx.out_dir.rglob("*") if path.is_file()]
        assert written and not any(path.name == "test.csv" for path in written)
        assert not any(b"test.csv" in path.read_bytes() or b"Invented history 9000" in path.read_bytes() for path in written)


def test_stages_publish_counts_only_and_keep_the_tables_private(tmp_path, monkeypatch):
    source = prepared(tmp_path)
    monkeypatch.setenv("CDM_WORK_DIR", str(source.parent))
    fold_ctx, data_ctx = context(tmp_path, "E10_folds"), context(tmp_path, "E8_data")
    folds.run_folds(fold_ctx)
    folds.run_data(data_ctx)

    public = approved(fold_ctx)
    assert public["admissions"] == 120 and public["leaks"] == 0 and public["every_admission_heldout_once"] is True
    assert public["seed"] == 20261113 and sum(public["heldout"]) == 120
    assert public["sources"]["train_csv"] == sha256((source / "train.csv").read_bytes())
    assert "1000" not in json.dumps(public)  # no admission identifier leaves
    assert fold_ctx.paths.private in (fold_ctx.out_dir / "folds.jsonl").parents

    for name in ("train.csv", "val.csv", "lab_test_mapping.csv"):
        assert (data_ctx.out_dir / name).read_bytes() == (source / name).read_bytes()
    assert approved(data_ctx)["train"] == 96 and approved(data_ctx)["monitored"] == 24
    assert len((data_ctx.out_dir / "admissions.jsonl").read_text().splitlines()) == 120


def test_the_source_directory_comes_from_the_baseline_work_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("CDM_WORK_DIR", raising=False)
    with pytest.raises(ValueError, match="CDM_WORK_DIR"):
        folds.source_dir()
    monkeypatch.setenv("CDM_WORK_DIR", str(tmp_path))
    assert folds.source_dir() == tmp_path / "prepared"


def test_overlapping_splits_and_unknown_diagnoses_are_refused(tmp_path):
    source = prepared(tmp_path)
    (source / "val.csv").write_bytes((source / "train.csv").read_bytes())
    with pytest.raises(ValueError, match="more than once"):
        folds.write_folds(source, tmp_path / "out")
    (source / "val.csv").write_bytes(split_bytes(50000, ["gastritis"]))
    with pytest.raises(ValueError, match="unknown diagnosis"):
        folds.write_folds(source, tmp_path / "out2")
