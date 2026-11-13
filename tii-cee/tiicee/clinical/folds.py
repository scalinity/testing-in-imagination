"""Training data for the new policies, cut from the replication's prepared train and validation files.

  clinical_data   E8, E9: train.csv, val.csv and lab_test_mapping.csv, copied unchanged.
  clinical_folds  E10: the training-plus-validation admissions in five folds stratified by
                  diagnosis. Fold directory k holds heldout.csv (fold k), val.csv (a fixed
                  stratified tenth of the other four folds, monitored only) and train.csv (the rest).

Both read $CDM_WORK_DIR/prepared/ (the baseline work directory) by exact file name and never list
it. test.csv is never opened, copied or listed. Every output stays in the stage's private
directory; only counts are approved for export. numpy only.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from tiicee.clinical import admissions, sha256
from tiicee.pipeline.stages import register

SEED, FOLDS, TENTH = 20261113, 5, 10  # EXPERIMENT_CARDS/E10.md
CLASSES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
SPLITS, MAPPING = ("train", "val"), "lab_test_mapping.csv"
PARTS = ("train", "val", "heldout")


def source_dir() -> Path:
    work = os.environ.get("CDM_WORK_DIR")
    if not work:
        raise ValueError("set CDM_WORK_DIR to the baseline work directory")
    return Path(work) / "prepared"


def _load(source):
    """The two split files and the lab mapping, each read once; their bytes are what is hashed and used."""
    data = {name: (Path(source) / name).read_bytes() for name in (*(f"{s}.csv" for s in SPLITS), MAPPING)}
    tables = {split: admissions.read(data[f"{split}.csv"]) for split in SPLITS}
    if tables["train"].header != tables["val"].header:
        raise ValueError("train.csv and val.csv have different columns")
    union = [a for split in SPLITS for a in tables[split].admissions]
    admissions.require_unique(union)
    if any(a.label not in CLASSES for a in union):
        raise ValueError("an admission has an unknown diagnosis")
    return data, tables, union


def _by_diagnosis(members, purpose):
    """Admissions grouped by diagnosis in CLASSES order; each group in fingerprint order, then
    permuted by the stream (SEED, "folds", purpose). The order depends on content, not file position."""
    seed = hashlib.sha256(f"{SEED}\0folds\0{purpose}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
    ordered = []
    for label in CLASSES:
        group = sorted((a for a in members if a.label == label), key=lambda a: a.fingerprint)
        ordered.extend(group[i] for i in rng.permutation(len(group)))
    return ordered


def assign(union) -> dict:
    """fingerprint -> fold. One counter deals across the diagnoses, so fold sizes differ by at most
    one and so does each diagnosis's count per fold."""
    return {a.fingerprint: i % FOLDS for i, a in enumerate(_by_diagnosis(union, "assignment"))}


def monitored(training, fold) -> set:
    """Fingerprints of the stratified tenth of one fold policy's training portion that is only monitored."""
    return {a.fingerprint for i, a in enumerate(_by_diagnosis(training, f"monitor:{fold}")) if i % TENTH == 0}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_folds(source, out) -> dict:
    data, tables, union = _load(source)
    table, fold_of = tables["train"], assign(union)
    for fold in range(FOLDS):
        rest = [a for a in union if fold_of[a.fingerprint] != fold]
        watch = monitored(rest, fold)
        members = {"train": [a for a in rest if a.fingerprint not in watch],
                   "val": [a for a in rest if a.fingerprint in watch],
                   "heldout": [a for a in union if fold_of[a.fingerprint] == fold]}
        for part in PARTS:
            _write(out / f"fold-{fold}" / f"{part}.csv", table.subset(members[part]))
        _write(out / f"fold-{fold}" / MAPPING, data[MAPPING])
    origin = {a.fingerprint: split for split in SPLITS for a in tables[split].admissions}
    _write(out / "folds.jsonl", "".join(
        json.dumps({"patient_id": a.patient_id, "fingerprint": a.fingerprint, "source": origin[a.fingerprint],
                    "fold": fold_of[a.fingerprint]}, sort_keys=True) + "\n" for a in union).encode())
    return {**check_folds(out, union, data[MAPPING]), "seed": SEED,
            "sources": {name.replace(".", "_"): sha256(content) for name, content in data.items()}}


def check_folds(out, union, mapping) -> dict:
    """Integrity of the fold directories as written, by fingerprint; any violation raises.

    Every admission is held out by exactly one fold, and no fold policy's training or monitoring
    set contains an admission of its held-out fold (EXPERIMENT_CARDS/E10.md, decision rule).
    """
    everyone, held = {a.fingerprint for a in union}, Counter()
    sizes, classes = {part: [] for part in PARTS}, {part: {label: [] for label in CLASSES} for part in PARTS}
    for fold in range(FOLDS):
        directory = Path(out) / f"fold-{fold}"
        parts = {part: admissions.read((directory / f"{part}.csv").read_bytes()).admissions for part in PARTS}
        prints = {part: {a.fingerprint for a in members} for part, members in parts.items()}
        if any(len(prints[part]) != len(parts[part]) for part in PARTS):
            raise ValueError("a fold file lists an admission twice")
        if prints["train"] & prints["heldout"] or prints["val"] & prints["heldout"]:
            raise ValueError("a fold policy's training or monitoring set contains its held-out admissions")
        if prints["train"] & prints["val"] or prints["train"] | prints["val"] | prints["heldout"] != everyone:
            raise ValueError("a fold's files do not partition the admissions")
        if (directory / MAPPING).read_bytes() != mapping:
            raise ValueError("a fold's lab mapping differs from the source")
        held.update(prints["heldout"])
        for part in PARTS:
            sizes[part].append(len(parts[part]))
            counts = Counter(a.label for a in parts[part])
            for label in CLASSES:
                classes[part][label].append(counts[label])
    if set(held) != everyone or any(count != 1 for count in held.values()):
        raise ValueError("an admission is not in exactly one held-out fold")
    return {"admissions": len(everyone), "folds": FOLDS, "heldout": sizes["heldout"], "train": sizes["train"],
            "monitored": sizes["val"], "by_diagnosis": {"heldout": classes["heldout"], "train": classes["train"],
                                                        "monitored": classes["val"]},
            "every_admission_heldout_once": True, "leaks": 0}


def copy_splits(source, out) -> dict:
    data, tables, union = _load(source)
    for name, content in data.items():
        _write(out / name, content)
    origin = {a.fingerprint: split for split in SPLITS for a in tables[split].admissions}
    _write(out / "admissions.jsonl", "".join(
        json.dumps({"patient_id": a.patient_id, "fingerprint": a.fingerprint, "source": origin[a.fingerprint]},
                   sort_keys=True) + "\n" for a in union).encode())
    return {"admissions": len(union), "train": len(tables["train"].admissions),
            "monitored": len(tables["val"].admissions),
            "sources": {name.replace(".", "_"): sha256(content) for name, content in data.items()}}


def _finish(ctx, summary):
    (ctx.out_dir / "integrity.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    ctx.approve(ctx.stage_id + ".json", {"stage": ctx.stage_id, **summary})


@register("clinical_folds")
def run_folds(ctx):
    _finish(ctx, write_folds(source_dir(), ctx.out_dir))


@register("clinical_data")
def run_data(ctx):
    _finish(ctx, copy_splits(source_dir(), ctx.out_dir))
