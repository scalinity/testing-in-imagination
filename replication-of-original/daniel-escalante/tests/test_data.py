"""Data contracts verified against the pinned upstream splitter and parser."""

import ast
import importlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from replication.data import (
    CLASSES, EXPECTED_COUNTS, IMAGING_TESTS, LAB_TESTS,
    assert_disjoint, audit_subject_overlap, generate_synthetic, load_dataset,
    prepare_splits, validate_raw,
)


@pytest.fixture
def raw(tmp_path):
    return generate_synthetic(tmp_path / "raw", n=240)


@pytest.fixture
def prepared(raw, tmp_path):
    output = tmp_path / "prepared"
    prepare_splits(raw, output)
    return output


@pytest.fixture
def cfg():
    return SimpleNamespace(dataset=SimpleNamespace(
        lab_tests=LAB_TESTS, imaging_tests=IMAGING_TESTS,
        other_tests=("Physical Examination",), small_sample=False,
    ))


def _raw_records(raw, condition=CLASSES[0]):
    path = raw / f"{condition}_hadm_info_first_diag.pkl"
    with path.open("rb") as handle:
        return path, pickle.load(handle)


def _rewrite(path, records):
    with path.open("wb") as handle:
        pickle.dump(records, handle, protocol=4)


def _mutate_first(raw, change):
    path, records = _raw_records(raw)
    first = next(iter(records))
    change(records[first])
    _rewrite(path, records)


def test_default_mix_missing_tests_and_repeated_subjects(tmp_path):
    raw = generate_synthetic(tmp_path / "raw")
    audit = validate_raw(raw)
    assert audit["total"] == 2400
    assert audit["class_counts"] == EXPECTED_COUNTS
    assert audit["subjects"]["repeated_admissions"] == 480
    assert all(count > 0 for count in audit["schema"]["empty_tests"].values())
    records = [_raw_records(raw, condition)[1] for condition in CLASSES]
    assert all(isinstance(key, int) for group in records for key in group)
    assert any(len(record["Laboratory Tests"]) < len(LAB_TESTS) for group in records for record in group.values())
    assert any(len(record["Radiology"]) < len(IMAGING_TESTS) for group in records for record in group.values())
    assert audit["mapping"]["all_references_resolved"]
    assert not audit["count_guard"]["enforced"]


def test_generation_and_all_split_bytes_are_deterministic(tmp_path):
    first = generate_synthetic(tmp_path / "raw1", seed=37, n=240)
    second = generate_synthetic(tmp_path / "raw2", seed=37, n=240)
    third = generate_synthetic(tmp_path / "raw3", seed=38, n=240)
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert _raw_records(first)[0].read_bytes() != _raw_records(third)[0].read_bytes()
    output1, output2 = tmp_path / "split1", tmp_path / "split2"
    result1, result2 = prepare_splits(first, output1), prepare_splits(second, output2)
    assert result1["audit"] == result2["audit"]
    for path in output1.rglob("*.csv"):
        assert path.read_bytes() == (output2 / path.relative_to(output1)).read_bytes()


@pytest.mark.parametrize(("change", "error"), [
    (lambda record: record.pop("Patient History"), "missing_field.Patient History"),
    (lambda record: record.update({"Patient History": None}), "invalid_history"),
    (lambda record: record.update({"Physical Examination": []}), "invalid_physical_examination"),
    (lambda record: record.update({"Laboratory Tests": []}), "invalid_laboratory_tests"),
    (lambda record: record.update({"Laboratory Tests": {"50001": "1"}}), "invalid_lab_itemid"),
    (lambda record: record.update({"Laboratory Tests": {999999: "1"}}), "unexpected_lab_itemid"),
    (lambda record: record.update({"Laboratory Tests": {50001: 1.0}}), "invalid_lab_result"),
    (lambda record: record.update({"Radiology": {"CT": "text"}}), "invalid_radiology"),
    (lambda record: record.update({"Radiology": ["text"]}), "invalid_radiology_record"),
    (lambda record: record.update({"Radiology": [{"Modality": "PET", "Report": "text"}]}), "unexpected_modality"),
    (lambda record: record.update({"Radiology": [{"Modality": "CT", "Findings": "text"}]}), "invalid_radiology_report"),
    (lambda record: record.update({"Radiology": [{"Modality": "CT", "Report": None}]}), "invalid_radiology_report"),
    (lambda record: record.update({"subject_id": "unknown"}), "invalid_subject_id"),
    (lambda record: record.pop("subject_id"), "inconsistent_subject_id_presence"),
    (lambda record: record.update({"Label": CLASSES[1]}), "inconsistent_label"),
    (lambda record: record.update({"Patient ID": 12}), "derived_field_present"),
])
def test_rejects_corrupted_raw_fields(raw, change, error):
    _mutate_first(raw, change)
    with pytest.raises(ValueError, match=error):
        validate_raw(raw)


def test_rejects_wrong_record_dictionary_and_noninteger_keys(raw):
    path, records = _raw_records(raw)
    first = next(iter(records))
    record = records.pop(first)
    records[str(first)] = record
    _rewrite(path, records)
    with pytest.raises(ValueError, match="invalid_admission_id"):
        validate_raw(raw)
    records.pop(str(first))
    records[first] = "invalid record"
    _rewrite(path, records)
    with pytest.raises(ValueError, match="invalid_record"):
        validate_raw(raw)


def test_rejects_non_dictionary_class_file(raw):
    path, records = _raw_records(raw)
    _rewrite(path, list(records.values()))
    with pytest.raises(ValueError, match="nonempty dictionary"):
        validate_raw(raw)


def test_rejects_duplicate_admissions_across_classes(raw):
    _, first_class = _raw_records(raw)
    path, second_class = _raw_records(raw, CLASSES[1])
    first = next(iter(second_class))
    second_class[next(iter(first_class))] = second_class.pop(first)
    _rewrite(path, second_class)
    with pytest.raises(ValueError, match="duplicate_admission_id_across_classes"):
        validate_raw(raw)


@pytest.mark.parametrize("corruption", [
    "missing_column", "bad_refs", "empty_refs", "noninteger_refs", "dangling_panel_ref",
    "dangling_analyte_ref", "dangling_unused_ref", "duplicate_id", "duplicate_panel",
    "missing_panel", "unknown_panel", "duplicate_refs",
])
def test_semantic_lab_mapping_validation(raw, corruption):
    path = raw / "lab_test_mapping.csv"
    frame = pd.read_csv(path)
    if corruption == "missing_column":
        frame = frame.drop(columns="corresponding_ids")
    elif corruption == "bad_refs":
        frame.loc[0, "corresponding_ids"] = "unparseable"
    elif corruption == "empty_refs":
        frame.loc[0, "corresponding_ids"] = "[]"
    elif corruption == "noninteger_refs":
        frame.loc[0, "corresponding_ids"] = "['50001']"
    elif corruption == "dangling_panel_ref":
        frame.loc[0, "corresponding_ids"] = "[999999]"
    elif corruption == "dangling_analyte_ref":
        frame.loc[1, "corresponding_ids"] = "[999999]"
    elif corruption == "dangling_unused_ref":
        frame.loc[len(frame)] = [999997, "Unused analyte", "[999998]"]
    elif corruption in ("duplicate_id", "duplicate_panel"):
        extra = frame.iloc[[0]].copy()
        if corruption == "duplicate_panel":
            extra["itemid"] = 999999
        frame = pd.concat([frame, extra], ignore_index=True)
    elif corruption == "missing_panel":
        frame.loc[0, "label"] = "Other measure"
    elif corruption == "unknown_panel":
        frame.loc[len(frame)] = [999999, "Lipid Panel", "[999999]"]
    elif corruption == "duplicate_refs":
        frame.loc[0, "corresponding_ids"] = "[50001, 50001]"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="lab mapping|Lab mapping"):
        validate_raw(raw)


def test_pickle_mapping_conversion_and_parser(raw, tmp_path, cfg):
    csv_path = raw / "lab_test_mapping.csv"
    mapping = pd.read_csv(csv_path)
    mapping["corresponding_ids"] = mapping["corresponding_ids"].map(ast.literal_eval)
    _rewrite(raw / "lab_test_mapping.pkl", mapping)
    csv_path.unlink()
    assert validate_raw(raw)["mapping"]["format"] == "pkl"
    output = tmp_path / "prepared"
    prepare_splits(raw, output)
    assert load_dataset(output, "test", cfg)[0]["patient_history"]


def test_parser_follows_canonical_lab_references_and_multiple_reports(raw, tmp_path, cfg):
    mapping_path = raw / "lab_test_mapping.csv"
    mapping = pd.read_csv(mapping_path)
    mapping.loc[0, "corresponding_ids"] = "[50001, 59999]"
    mapping.loc[len(mapping)] = [59999, "Alias measure", "[60000]"]
    mapping.loc[len(mapping)] = [60000, "Canonical measure", "[60000]"]
    mapping.to_csv(mapping_path, index=False)
    for condition in CLASSES:
        path, records = _raw_records(raw, condition)
        for record in records.values():
            record["Laboratory Tests"] = {50001: "11 invented units", 59999: "22 invented units"}
            record["Radiology"] = [
                {"Modality": "CT", "Report": f"TECHNIQUE:\nAcquisition.\nFINDINGS:\nInvented finding {i}.\nIMPRESSION:\nObserve."}
                for i in (1, 2)
            ]
        _rewrite(path, records)
    assert validate_raw(raw)["mapping"]["all_references_resolved"]
    output = tmp_path / "prepared"
    prepare_splits(raw, output)
    item = load_dataset(output, "test", cfg)[0]
    assert item["test_results"][LAB_TESTS[0]] == (
        "Invented measure 1: 11 invented units, Canonical measure: 22 invented units\n"
    )
    ct = item["test_results"]["CT"]
    assert "There are 2 reports available" in ct
    assert "Report 1:" in ct and "Report 2:" in ct
    assert "Invented finding 1" in ct and "Invented finding 2" in ct
    assert "TECHNIQUE:" not in ct


def test_aggregate_audit_and_failures_never_include_patient_text(raw):
    sensitive = "UNIQUE_PRIVATE_RECORD_STRING_719"
    _mutate_first(raw, lambda record: record.update({"Patient History": sensitive}))
    assert sensitive not in json.dumps(validate_raw(raw))
    _mutate_first(raw, lambda record: record.update({"Laboratory Tests": {sensitive: sensitive}}))
    with pytest.raises(ValueError) as error:
        validate_raw(raw)
    assert sensitive not in str(error.value)


def test_mimic_cohort_guard_total_and_class_tolerances(tmp_path):
    raw = generate_synthetic(tmp_path / "raw")
    audit = validate_raw(raw, source="mimic")
    assert audit["count_guard"]["enforced"]
    assert audit["count_guard"]["total_relative_tolerance"] == 0.20
    assert audit["count_guard"]["class_relative_tolerance"] == 0.35
    path, records = _raw_records(raw, CLASSES[2])
    _rewrite(path, dict(list(records.items())[:100]))
    with pytest.raises(ValueError, match="class_outside_tolerance"):
        validate_raw(raw, source="mimic")
    small = generate_synthetic(tmp_path / "small", n=240)
    with pytest.raises(ValueError, match="total_outside_tolerance"):
        validate_raw(small, source="mimic")


def test_primary_calls_real_upstream_splitter(raw, tmp_path, monkeypatch):
    # The module is imported through the same path used by the implementation.
    from replication.data import _upstream_module
    upstream = _upstream_module("scripts.split_dataset")
    calls = []
    original = upstream.split_dataset

    def recording_splitter(data_dir, output_dir):
        calls.append((data_dir, output_dir))
        return original(data_dir, output_dir)

    monkeypatch.setattr(upstream, "split_dataset", recording_splitter)
    output = tmp_path / "prepared"
    result = prepare_splits(raw, output)
    assert calls == [(raw, output)]
    expected = tmp_path / "upstream_reference"
    original(raw, expected)
    for split in ("train", "val", "test"):
        actual = pd.read_csv(output / f"{split}.csv")
        reference = pd.read_csv(expected / f"{split}.csv")
        assert actual["Patient ID"].tolist() == reference["Patient ID"].tolist()
        assert actual["Label"].tolist() == reference["Label"].tolist()
    assert result["audit"]["split_seed"] == 269
    assert result["audit"]["primary"]["admission_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert {split: audit["count"] for split, audit in result["audit"]["primary"]["splits"].items()} == {
        "train": 192, "val": 24, "test": 24,
    }


def test_grouped_split_has_zero_subject_overlap_and_reports_counts(prepared):
    primary = audit_subject_overlap(prepared)
    assert primary["status"] == "assessable"
    assert primary["any_overlap"] > 0
    grouped = assert_disjoint(prepared / "grouped")
    assert grouped["subjects"]["any_overlap"] == 0
    assert grouped["total"] == 240
    primary_ids = set()
    grouped_ids = set()
    for split in ("train", "val", "test"):
        primary_ids.update(pd.read_csv(prepared / f"{split}.csv")["Patient ID"])
        frame = pd.read_csv(prepared / "grouped" / f"{split}.csv")
        grouped_ids.update(frame["Patient ID"])
        assert grouped["splits"][split]["count"] == len(frame)
        assert set(frame["Label"]) == set(CLASSES)
        assert "subject_id" in frame
    assert primary_ids == grouped_ids


def test_grouped_stratification_handles_mixed_and_rare_subject_classes():
    from replication.data import _grouped_frames
    rows = []
    for subject in range(1, 61):
        rows.append({"subject_id": subject, "Label": CLASSES[subject % 4]})
    rows.extend([{"subject_id": 1, "Label": CLASSES[0]}, {"subject_id": 2, "Label": CLASSES[0]}])
    grouped, audit = _grouped_frames(pd.DataFrame(rows))
    assert audit["multi_class_subjects"] == 2
    assert audit["attempts"] == 1
    assert all(audit["stratified"].values())
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(grouped[left]["subject_id"]) & set(grouped[right]["subject_id"])
    rows = [{"subject_id": subject, "Label": CLASSES[0] if subject < 20 else CLASSES[1]}
            for subject in range(1, 21)]
    _, rare = _grouped_frames(pd.DataFrame(rows))
    assert rare["stratified"]["train_holdout"] is False
    assert rare["attempts"] == 1


def test_missing_subject_ids_report_unassessable(raw, tmp_path):
    for condition in CLASSES:
        path, records = _raw_records(raw, condition)
        for record in records.values():
            record.pop("subject_id")
        _rewrite(path, records)
    assert validate_raw(raw)["subjects"]["status"] == "unassessable"
    output = tmp_path / "prepared"
    result = prepare_splits(raw, output)
    assert result["audit"]["grouped"]["status"] == "unassessable"
    assert result["paths"]["grouped"] == {}
    assert audit_subject_overlap(output)["any_overlap"] is None
    assert not (output / "grouped").exists()


def test_callable_summarizer_receives_split_names(raw, tmp_path):
    calls = []

    def summarize(text, split):
        calls.append((text, split))
        return f"Invented summary for {split}"

    output = tmp_path / "prepared"
    result = prepare_splits(raw, output, fake_summarizer=False, summarizer=summarize)
    assert result["audit"]["summarizer"] == "callable"
    assert len(calls) == 240
    assert {split for _, split in calls} == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert set(pd.read_csv(output / f"{split}.csv")["Patient History Summary"]) == {f"Invented summary for {split}"}


def test_requires_real_summarizer_and_canonical_seed(raw, tmp_path):
    with pytest.raises(ValueError, match="seed 269"):
        prepare_splits(raw, tmp_path / "wrong_seed", seed=123)
    with pytest.raises(ValueError, match="real summarizer"):
        prepare_splits(raw, tmp_path / "no_summarizer", fake_summarizer=False)
    full_raw = generate_synthetic(tmp_path / "full_raw")
    with pytest.raises(ValueError, match="real summarizer"):
        prepare_splits(full_raw, tmp_path / "fake_mimic", source="mimic")


def test_rejects_summary_failure_without_patient_text(raw, tmp_path):
    def fail(text, split):
        raise RuntimeError(text)

    with pytest.raises(ValueError, match="patient text omitted") as error:
        prepare_splits(raw, tmp_path / "prepared", summarizer=fail)
    assert "abdominal" not in str(error.value)
    with pytest.raises(ValueError, match="invalid summary"):
        prepare_splits(raw, tmp_path / "prepared2", summarizer=lambda text, split: None)


def test_loader_returns_actual_parser_and_shapes(prepared, cfg):
    before = set(sys.modules)
    dataset = load_dataset(prepared, "test", cfg)
    actual = importlib.import_module("src.dataset.mimic_cdm_dataset").MIMICCDMDataset
    assert type(dataset) is actual
    assert len(dataset) == 24
    frame = pd.read_csv(prepared / "test.csv", keep_default_na=False)
    saw_missing_lab = saw_report = saw_missing_exam = False
    for index in range(len(dataset)):
        item = dataset[index]
        row = frame.iloc[index]
        assert set(item) == {"patient_id", "patient_history", "condition", "test_results"}
        assert item["patient_id"] == row["Patient ID"]
        assert item["patient_history"] == row["Patient History Summary"]
        assert set(item["test_results"]) == {*LAB_TESTS, *IMAGING_TESTS, "Physical Examination"}
        labs = ast.literal_eval(row["Laboratory Tests"])
        for panel, itemid in zip(LAB_TESTS, range(50001, 50062, 10)):
            value = item["test_results"][panel]
            if itemid not in labs:
                assert value == "not available.\n"
                saw_missing_lab = True
            else:
                assert labs[itemid] in value
        for modality in IMAGING_TESTS:
            value = item["test_results"][modality]
            if value != "not available.\n":
                assert "FINDINGS:" in value and "IMPRESSION:" in value
                assert "TECHNIQUE:" not in value
                saw_report = True
        if item["test_results"]["Physical Examination"] == "not available.\n":
            saw_missing_exam = True
    assert saw_missing_lab and saw_report and saw_missing_exam
    assert not {"vllm", "transformers", "trl"} & (set(sys.modules) - before)
    if importlib.util.find_spec("torch") is None:
        assert not {"torch", "torch.utils", "torch.utils.data"} & (set(sys.modules) - before)


@pytest.mark.parametrize("corruption", ["cross_split", "within_split", "missing_test", "missing_subject", "bad_labs", "bad_summary"])
def test_every_loader_checks_all_splits_without_train_fallback(prepared, cfg, corruption):
    train = pd.read_csv(prepared / "train.csv", keep_default_na=False)
    test_path = prepared / "test.csv"
    test = pd.read_csv(test_path, keep_default_na=False)
    if corruption == "cross_split":
        test.loc[0, "Patient ID"] = train.loc[0, "Patient ID"]
    elif corruption == "within_split":
        test.loc[0, "Patient ID"] = test.loc[1, "Patient ID"]
    elif corruption == "missing_test":
        test_path.unlink()
    elif corruption == "missing_subject":
        test = test.drop(columns="subject_id")
    elif corruption == "bad_labs":
        test.loc[0, "Laboratory Tests"] = "[]"
    elif corruption == "bad_summary":
        test.loc[0, "Patient History Summary"] = ""
    if corruption != "missing_test":
        test.to_csv(test_path, index=False)
    with pytest.raises(ValueError):
        load_dataset(prepared, "train", cfg)


def test_unknown_split_is_rejected(prepared, cfg):
    with pytest.raises(ValueError, match="split must be"):
        load_dataset(prepared, "unknown", cfg)


def test_patient_data_cannot_be_generated_inside_repository():
    repo = Path(__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="outside the repository"):
        generate_synthetic(repo / "should_not_exist")
    assert not (repo / "should_not_exist").exists()


def test_absent_or_invalid_mapping_is_rejected(raw):
    (raw / "lab_test_mapping.csv").unlink()
    with pytest.raises(ValueError, match="Missing lab_test_mapping"):
        validate_raw(raw)
    _rewrite(raw / "lab_test_mapping.pkl", {"itemid": 1})
    with pytest.raises(ValueError, match="DataFrame"):
        validate_raw(raw)


def test_duplicate_csv_columns_are_rejected(raw):
    mapping = raw / "lab_test_mapping.csv"
    original = mapping.read_text()
    mapping.write_text(original.replace("itemid,label,corresponding_ids", "itemid,label,label"))
    with pytest.raises(ValueError, match="invalid CSV schema"):
        validate_raw(raw)


def test_mapping_csv_preferred_over_pickle(raw):
    _rewrite(raw / "lab_test_mapping.pkl", "invalid unused fallback")
    assert validate_raw(raw)["mapping"]["format"] == "csv"
