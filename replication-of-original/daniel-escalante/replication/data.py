"""Raw data checks, pinned splits, and CPU access to the upstream dataset."""

from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import importlib
import importlib.util
import json
import math
from numbers import Integral
from pathlib import Path
import pickle
import random
import sys
import types
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split


CLASSES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
EXPECTED_COUNTS = dict(zip(CLASSES, (957, 648, 257, 538)))
LAB_TESTS = (
    "Complete Blood Count", "Basic Metabolic Panel", "Comprehensive Metabolic Panel",
    "Renal Function Panel", "Liver Function Panel", "Urinalysis", "Electrolyte Panel",
)
IMAGING_TESTS = ("CT", "MRI", "Radiograph", "Ultrasound")
RAW_FIELDS = ("Patient History", "Physical Examination", "Laboratory Tests", "Radiology")
SPLITS = ("train", "val", "test")
SPLIT_SEED = 269


def _external(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    root = Path(__file__).resolve().parents[3]
    roots = [root]
    if root.parent.name == ".worktrees":
        roots.append(root.parent.parent)
    if any(path == repo or repo in path.parents for repo in roots):
        raise ValueError("Patient data and split artifacts must be outside the repository")
    return path


def _integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _raise_errors(errors: Counter, context: str) -> None:
    if errors:
        # Only fixed schema names and counts are included, never record values.
        raise ValueError(f"{context}: {json.dumps(dict(sorted(errors.items())))}")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        if len(header) != len(set(header)):
            raise ValueError
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, ValueError, StopIteration, pd.errors.ParserError):
        raise ValueError(f"Missing or invalid CSV schema: {path.name}") from None


def _mapping(data_dir: Path) -> tuple[pd.DataFrame, set[int], dict]:
    csv_path, pkl_path = data_dir / "lab_test_mapping.csv", data_dir / "lab_test_mapping.pkl"
    if csv_path.exists():
        frame = _read_csv(csv_path)
        mapping_format = "csv"
    elif pkl_path.exists():
        try:
            with pkl_path.open("rb") as handle:
                frame = pickle.load(handle)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            raise ValueError("Invalid lab_test_mapping.pkl") from None
        mapping_format = "pkl"
    else:
        raise ValueError("Missing lab_test_mapping.csv or lab_test_mapping.pkl")
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Lab mapping must be a DataFrame with unique columns")
    if not {"itemid", "label", "corresponding_ids"}.issubset(frame.columns):
        raise ValueError("Lab mapping requires itemid, label, corresponding_ids")

    errors = Counter()
    rows = {}
    labels = Counter()
    normalized = []
    for row in frame.to_dict("records"):
        itemid, label, refs = row["itemid"], row["label"], row["corresponding_ids"]
        if not _integer(itemid):
            errors["mapping.invalid_itemid"] += 1
            continue
        itemid = int(itemid)
        if itemid in rows:
            errors["mapping.duplicate_itemid"] += 1
        if not isinstance(label, str) or not label.strip():
            errors["mapping.invalid_label"] += 1
            continue
        labels[label] += 1
        if label.endswith(" Panel") and label not in LAB_TESTS:
            errors["mapping.unexpected_panel"] += 1
        try:
            refs = ast.literal_eval(refs) if isinstance(refs, str) else refs
        except (SyntaxError, ValueError, TypeError):
            refs = None
        if not isinstance(refs, (list, tuple)) or not refs or not all(_integer(x) for x in refs):
            errors["mapping.invalid_corresponding_ids"] += 1
            continue
        refs = [int(x) for x in refs]
        if len(refs) != len(set(refs)):
            errors["mapping.duplicate_corresponding_ids"] += 1
        rows[itemid] = (label, refs)
        normalized.append({"itemid": itemid, "label": label, "corresponding_ids": repr(refs)})
    for panel in LAB_TESTS:
        if labels[panel] != 1:
            errors["mapping.panel_label_not_unique"] += 1
    for _, refs in rows.values():
        for itemid in refs:
            if itemid not in rows:
                errors["mapping.unresolved_corresponding_id"] += 1
    _raise_errors(errors, "Invalid lab mapping")

    panel_ids = set()
    for label, refs in rows.values():
        if label in LAB_TESTS:
            panel_ids.update(refs)
    return pd.DataFrame(normalized), panel_ids, {
        "format": mapping_format, "rows": len(rows), "panels": len(LAB_TESTS),
        "resolved_itemids": len(panel_ids), "all_references_resolved": True,
    }


def _record_errors(record, panel_ids: set[int]) -> Counter:
    errors = Counter()
    if not isinstance(record, dict):
        errors["raw.invalid_record"] += 1
        return errors
    for field in RAW_FIELDS:
        if field not in record:
            errors[f"raw.missing_field.{field}"] += 1
    history = record.get("Patient History")
    if not isinstance(history, str) or not history.strip():
        errors["raw.invalid_history"] += 1
    if not isinstance(record.get("Physical Examination"), str):
        errors["raw.invalid_physical_examination"] += 1
    labs = record.get("Laboratory Tests")
    if not isinstance(labs, dict):
        errors["raw.invalid_laboratory_tests"] += 1
    else:
        for itemid, value in labs.items():
            if not _integer(itemid):
                errors["raw.invalid_lab_itemid"] += 1
            elif itemid not in panel_ids:
                errors["raw.unexpected_lab_itemid"] += 1
            if not isinstance(value, str) or not value.strip():
                errors["raw.invalid_lab_result"] += 1
    imaging = record.get("Radiology")
    if not isinstance(imaging, list):
        errors["raw.invalid_radiology"] += 1
    else:
        for report in imaging:
            if not isinstance(report, dict):
                errors["raw.invalid_radiology_record"] += 1
                continue
            modality = report.get("Modality")
            if not isinstance(modality, str) or modality not in IMAGING_TESTS:
                errors["raw.unexpected_modality"] += 1
            text = report.get("Report")
            if not isinstance(text, str) or not text.strip():
                errors["raw.invalid_radiology_report"] += 1
    if "subject_id" in record and not _integer(record["subject_id"]):
        errors["raw.invalid_subject_id"] += 1
    return errors


def generate_synthetic(output_dir: Path, seed: int = 269, n: int = 2400) -> Path:
    """Write invented raw records and a parser-compatible lab mapping, deterministically."""
    output_dir = _external(output_dir)
    if not isinstance(n, int) or isinstance(n, bool) or n < len(CLASSES):
        raise ValueError("n must be an integer of at least four")
    seed_bytes = hashlib.sha256(f"{seed}:raw:generate".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    proportions = [n * EXPECTED_COUNTS[label] / 2400 for label in CLASSES]
    counts = [max(1, math.floor(x)) for x in proportions]
    while sum(counts) < n:
        idx = max(range(4), key=lambda i: (proportions[i] - counts[i], -i))
        counts[idx] += 1
    while sum(counts) > n:
        idx = max((i for i in range(4) if counts[i] > 1), key=lambda i: counts[i] - proportions[i])
        counts[idx] -= 1

    mapping_rows, itemids = [], []
    for i, panel in enumerate(LAB_TESTS):
        panel_id, itemid = 50000 + 10 * i, 50001 + 10 * i
        mapping_rows.extend([
            {"itemid": panel_id, "label": panel, "corresponding_ids": repr([itemid])},
            {"itemid": itemid, "label": f"Invented measure {i + 1}", "corresponding_ids": repr([itemid])},
        ])
        itemids.append(itemid)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).to_csv(output_dir / "lab_test_mapping.csv", index=False)
    hadm_id, next_subject = 100000, 200000
    for condition, count in zip(CLASSES, counts):
        records, subjects = {}, []
        repeated = round(count * 0.20)
        for i in range(count):
            if i >= count - repeated:
                subject_id = subjects[i - (count - repeated)]
            else:
                subject_id = next_subject
                next_subject += 1
                subjects.append(subject_id)
            records[hadm_id] = {
                "Patient History": (
                    f"Invented encounter: {rng.randint(1, 5)} days of abdominal discomfort. "
                    f"Pain is {rng.choice(('intermittent', 'steady', 'mild'))}; "
                    f"appetite is {rng.choice(('reduced', 'unchanged'))}. No prior operations reported."
                ),
                "Physical Examination": "" if rng.random() < 0.20 else "Invented examination: soft abdomen with localized tenderness.",
                "Laboratory Tests": {} if rng.random() < 0.10 else {
                    itemid: f"{rng.uniform(1, 15):.2f} invented units"
                    for itemid in itemids if rng.random() >= 0.25
                },
                "Radiology": [
                    {"Modality": modality, "Report": (
                        "TECHNIQUE:\nInvented acquisition.\nFINDINGS:\n"
                        f"Invented {modality} observation: mild regional change.\n"
                        "IMPRESSION:\nClinical correlation advised."
                    )}
                    for modality in IMAGING_TESTS if rng.random() >= 0.40
                ],
                "subject_id": subject_id,
            }
            hadm_id += 1
        with (output_dir / f"{condition}_hadm_info_first_diag.pkl").open("wb") as handle:
            pickle.dump(records, handle, protocol=4)
    return output_dir


def validate_raw(data_dir: Path, source: str = "synthetic") -> dict:
    """Reject incompatible raw data; return aggregate schema and cohort counts only."""
    data_dir = _external(data_dir)
    if source not in ("synthetic", "mimic"):
        raise ValueError("source must be synthetic or mimic")
    _, panel_ids, mapping_audit = _mapping(data_dir)
    errors, counts, missing, fields = Counter(), {}, Counter(), Counter()
    admissions, subjects = set(), []
    unexpected_fields = 0
    for condition in CLASSES:
        try:
            with (data_dir / f"{condition}_hadm_info_first_diag.pkl").open("rb") as handle:
                records = pickle.load(handle)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            raise ValueError(f"Missing or invalid raw dictionary for {condition}") from None
        if not isinstance(records, dict) or not records:
            raise ValueError(f"Raw {condition} must be a nonempty dictionary")
        counts[condition] = len(records)
        for hadm_id, record in records.items():
            if not _integer(hadm_id):
                errors["raw.invalid_admission_id"] += 1
            elif hadm_id in admissions:
                errors["raw.duplicate_admission_id_across_classes"] += 1
            else:
                admissions.add(int(hadm_id))
            errors.update(_record_errors(record, panel_ids))
            if not isinstance(record, dict):
                continue
            for field in (*RAW_FIELDS, "subject_id"):
                fields[field] += field in record
            unexpected_fields += len(set(record) - set(RAW_FIELDS) - {"subject_id", "Label"})
            if "Patient ID" in record or "Patient History Summary" in record:
                errors["raw.derived_field_present"] += 1
            if "Label" in record and (not isinstance(record["Label"], str) or record["Label"] != condition):
                errors["raw.inconsistent_label"] += 1
            if _integer(record.get("subject_id")):
                subjects.append(int(record["subject_id"]))
            for field, expected_type in (("Physical Examination", str), ("Laboratory Tests", dict), ("Radiology", list)):
                value = record.get(field)
                missing[field] += isinstance(value, expected_type) and not value
    total = sum(counts.values())
    if fields["subject_id"] not in (0, total):
        errors["raw.inconsistent_subject_id_presence"] += 1
    count_guard = {
        "enforced": source == "mimic", "expected_total": 2400,
        "total_relative_tolerance": 0.20, "class_relative_tolerance": 0.35,
        "expected_class_counts": EXPECTED_COUNTS.copy(),
    }
    if source == "mimic":
        if not 1920 <= total <= 2880:
            errors["cohort.total_outside_tolerance"] += 1
        for condition, count in counts.items():
            expected = EXPECTED_COUNTS[condition]
            if abs(count - expected) > expected * 0.35:
                errors["cohort.class_outside_tolerance"] += 1
    _raise_errors(errors, "Invalid raw schema or cohort")
    return {
        "source": source, "total": total, "class_counts": counts,
        "schema": {"valid": True, "field_counts": dict(fields), "empty_tests": dict(missing),
                   "extra_field_count": unexpected_fields},
        "mapping": mapping_audit, "count_guard": count_guard,
        "subjects": {"status": "assessable" if subjects else "unassessable",
                     "unique": len(set(subjects)) if subjects else None,
                     "repeated_admissions": len(subjects) - len(set(subjects)) if subjects else None},
    }


def _upstream_dir() -> Path:
    try:
        from .config import UPSTREAM
    except ModuleNotFoundError as exc:
        if exc.name != f"{__package__}.config":
            raise
    else:
        return Path(UPSTREAM)
    local = Path(__file__).resolve().parents[3] / "lacdm" / "upstream"
    for candidate in (local, *(Path(x) for x in sys.path)):
        if (candidate / "scripts" / "split_dataset.py").is_file() and (candidate / "src" / "dataset").is_dir():
            return candidate.resolve()
    raise RuntimeError("Pinned upstream submodule is unavailable; initialize it or expose its path")


def _upstream_module(name: str):
    upstream = str(_upstream_dir())
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    return importlib.import_module(name)


def _subject_audit(frames: dict[str, pd.DataFrame]) -> dict:
    present = ["subject_id" in frame for frame in frames.values()]
    if not any(present):
        return {"status": "unassessable", "unique": None, "repeated_admissions": None,
                "overlap": None, "any_overlap": None}
    if not all(present):
        raise ValueError("Inconsistent subject_id presence across splits")
    if any(not _integer(x) for frame in frames.values() for x in frame["subject_id"]):
        raise ValueError("Invalid subject_id in split data")
    ids = {split: set(frame["subject_id"]) for split, frame in frames.items()}
    intersections = [ids["train"] & ids["val"], ids["train"] & ids["test"], ids["val"] & ids["test"]]
    unique = len(set.union(*ids.values()))
    return {"status": "assessable", "unique": unique,
            "repeated_admissions": sum(len(frame) for frame in frames.values()) - unique,
            "overlap": dict(zip(("train_val", "train_test", "val_test"), map(len, intersections))),
            "any_overlap": len(set.union(*intersections))}


def assert_disjoint(output_dir: Path) -> dict:
    """Read all three splits, reject duplicate admissions, and report subject overlap."""
    output_dir = _external(output_dir)
    _, panel_ids, _ = _mapping(output_dir)
    frames, ids, errors = {}, {}, Counter()
    for split in SPLITS:
        frame = _read_csv(output_dir / f"{split}.csv")
        required = {*RAW_FIELDS, "Patient ID", "Patient History Summary", "Label"}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(f"Missing fields or empty split: {split}")
        for row in frame.to_dict("records"):
            if not _integer(row["Patient ID"]):
                errors["split.invalid_admission_id"] += 1
            if not isinstance(row["Label"], str) or row["Label"] not in CLASSES:
                errors["split.invalid_label"] += 1
            summary = row["Patient History Summary"]
            if not isinstance(summary, str) or not summary.strip():
                errors["split.invalid_summary"] += 1
            for field in ("Laboratory Tests", "Radiology"):
                try:
                    row[field] = ast.literal_eval(row[field])
                except (SyntaxError, ValueError, TypeError):
                    row[field] = None
            errors.update(_record_errors(row, panel_ids))
        if frame["Patient ID"].duplicated().any():
            errors["split.duplicate_admission_id_within_split"] += 1
        frames[split], ids[split] = frame, set(frame["Patient ID"])
    overlap = {"train_val": len(ids["train"] & ids["val"]),
               "train_test": len(ids["train"] & ids["test"]),
               "val_test": len(ids["val"] & ids["test"])}
    if any(overlap.values()):
        errors["split.admission_overlap"] += sum(overlap.values())
    _raise_errors(errors, "Invalid split data")
    return {"status": "valid", "total": sum(len(frame) for frame in frames.values()),
            "admission_overlap": overlap, "subjects": _subject_audit(frames),
            "splits": {split: {"count": len(frame), "class_counts": {
                label: int((frame["Label"] == label).sum()) for label in CLASSES
            }} for split, frame in frames.items()}}


def audit_subject_overlap(output_dir: Path) -> dict:
    return assert_disjoint(output_dir)["subjects"]


def _grouped_frames(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    # One deterministic majority class per subject; ties follow the configured class order.
    table = pd.crosstab(frame["subject_id"], frame["Label"]).reindex(columns=CLASSES, fill_value=0)
    subject_labels = table.idxmax(axis=1)
    stratification = {}

    def divide(subjects, fraction, stage):
        labels = subject_labels.loc[subjects]
        counts = labels.value_counts()
        n_test = math.ceil(len(subjects) * fraction)
        can_stratify = bool(len(counts) and counts.min() >= 2 and
                             min(n_test, len(subjects) - n_test) >= len(counts))
        stratification[stage] = can_stratify
        return train_test_split(list(subjects), test_size=fraction, random_state=SPLIT_SEED,
                                stratify=labels if can_stratify else None)

    if len(table) < 3:
        raise ValueError("Grouped split needs at least three distinct subjects")
    train, remaining = divide(table.index, 0.20, "train_holdout")
    if len(remaining) < 2:
        raise ValueError("Grouped split holdout needs at least two subjects; no retry was attempted")
    val, test = divide(remaining, 0.50, "val_test")
    grouped = {split: frame[frame["subject_id"].isin(subjects)].reset_index(drop=True)
               for split, subjects in zip(SPLITS, (train, val, test))}
    return grouped, {"seed": SPLIT_SEED, "method": "subject_shuffle",
                     "subject_class": "majority_class_with_class_order_ties",
                     "stratified": stratification, "attempts": 1,
                     "multi_class_subjects": int(((table > 0).sum(axis=1) > 1).sum()),
                     "summary_reuse": "primary_by_admission"}


def prepare_splits(
    data_dir: Path, output_dir: Path, seed: int = 269, fake_summarizer: bool = True,
    summarizer: Callable[[str, str], str] | None = None, source: str = "synthetic",
) -> dict:
    """Invoke the pinned splitter, summarize histories, and create grouped sensitivity CSVs."""
    if seed != SPLIT_SEED:
        raise ValueError("The pinned upstream splitter requires seed 269")
    data_dir, output_dir = _external(data_dir), _external(output_dir)
    raw_audit = validate_raw(data_dir, source=source)
    if summarizer is None:
        if not fake_summarizer or source != "synthetic":
            raise ValueError("A real summarizer callable is required for this source")
        summarizer = lambda text, split: text.split(". ", 1)[0] + "."
        summary_kind = "synthetic_extract"
    else:
        summary_kind = "callable"
    splitter = _upstream_module("scripts.split_dataset")
    if splitter.RANDOM_STATE != SPLIT_SEED or tuple(splitter.CONDITIONS) != CLASSES:
        raise RuntimeError("Upstream splitter does not match the pinned data contract")
    splitter.split_dataset(data_dir, output_dir)
    mapping, _, _ = _mapping(data_dir)
    mapping.to_csv(output_dir / "lab_test_mapping.csv", index=False)
    frames = {}
    paths = {split: str(output_dir / f"{split}.csv") for split in SPLITS}
    paths["lab_test_mapping"] = str(output_dir / "lab_test_mapping.csv")
    for split in SPLITS:
        frame = _read_csv(output_dir / f"{split}.csv")
        summaries = []
        for history in frame["Patient History"]:
            try:
                summary = summarizer(history, split)
            except Exception:
                raise ValueError(f"Summarization failed for {split}; patient text omitted") from None
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(f"Summarizer returned an invalid summary for {split}")
            summaries.append(summary)
        frame["Patient History Summary"] = summaries
        # The upstream parser only fills unavailable lab/imaging tests itself.
        frame["Physical Examination"] = frame["Physical Examination"].map(
            lambda value: value if value.strip() else "not available.\n"
        )
        frame.to_csv(output_dir / f"{split}.csv", index=False)
        frames[split] = frame
    audit = {"raw": raw_audit, "split_seed": SPLIT_SEED, "summarizer": summary_kind,
             "primary": assert_disjoint(output_dir)}
    paths["grouped"] = {}
    if raw_audit["subjects"]["status"] == "assessable":
        grouped, method = _grouped_frames(pd.concat(frames.values(), ignore_index=True))
        grouped_dir = output_dir / "grouped"
        grouped_dir.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(grouped_dir / "lab_test_mapping.csv", index=False)
        for split, frame in grouped.items():
            path = grouped_dir / f"{split}.csv"
            frame.to_csv(path, index=False)
            paths["grouped"][split] = str(path)
        paths["grouped"]["lab_test_mapping"] = str(grouped_dir / "lab_test_mapping.csv")
        audit["grouped"] = {**method, **assert_disjoint(grouped_dir)}
        if audit["grouped"]["subjects"]["any_overlap"] != 0:
            raise ValueError("Grouped split contains subject overlap")
    else:
        audit["grouped"] = {"status": "unassessable", "reason": "subject_id absent"}
    return {"audit": audit, "paths": paths}


@contextmanager
def _dataset_import_scope():
    """Supply only the Dataset base while importing on installations without torch."""
    if "torch" in sys.modules or importlib.util.find_spec("torch") is not None:
        yield
        return
    names = ("torch", "torch.utils", "torch.utils.data")
    modules = {name: types.ModuleType(name) for name in names}
    modules["torch"].utils = modules["torch.utils"]
    modules["torch.utils"].data = modules["torch.utils.data"]
    modules["torch.utils.data"].Dataset = type("Dataset", (), {})
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)


def load_dataset(output_dir: Path, split: str, cfg):
    """Return the real upstream dataset after checking every split, with no fallback."""
    if split not in SPLITS:
        raise ValueError("split must be train, val, or test")
    output_dir = _external(output_dir)
    assert_disjoint(output_dir)
    dataset_cfg = cfg.dataset
    if tuple(dataset_cfg.lab_tests) != LAB_TESTS or tuple(dataset_cfg.imaging_tests) != IMAGING_TESTS:
        raise ValueError("Configured test vocabulary differs from the pinned raw schema")
    if tuple(dataset_cfg.other_tests) != ("Physical Examination",):
        raise ValueError("Configured other tests differ from the pinned raw schema")
    with _dataset_import_scope():
        dataset_class = _upstream_module("src.dataset.mimic_cdm_dataset").MIMICCDMDataset
    return dataset_class(
        data_file=str(output_dir / f"{split}.csv"),
        lab_test_mapping_file=str(output_dir / "lab_test_mapping.csv"),
        lab_tests=list(dataset_cfg.lab_tests), imaging_tests=list(dataset_cfg.imaging_tests),
        other_tests=list(dataset_cfg.other_tests),
        small_sample=bool(getattr(dataset_cfg, "small_sample", False)),
    )
