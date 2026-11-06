import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pandas as pd

from replication.data import CLASSES, LAB_TESTS, validate_raw

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_condition_pickles.py"
UNKNOWN_ADMISSION = 987654321


def _release(directory: Path, stray_exam: bool = False) -> Path:
    """Write an invented directory shaped like the PhysioNet v1.1 CSV release."""
    ids = {condition: [100 + 10 * i, 101 + 10 * i] for i, condition in enumerate(CLASSES)}
    hadm = [hadm_id for group in ids.values() for hadm_id in group]
    mapping = pd.DataFrame({
        "itemid": pd.array([50001] + [pd.NA] * len(LAB_TESTS), dtype="Int64"),
        "label": ["Invented measure", *LAB_TESTS],
        "corresponding_ids": "[50001]",
    })
    tables = {
        "history_of_present_illness.csv": pd.DataFrame({"hadm_id": hadm, "hpi": "Invented history."}),
        "physical_examination.csv": pd.DataFrame(
            {"hadm_id": hadm + [UNKNOWN_ADMISSION] * stray_exam, "pe": "Invented exam."}),
        "laboratory_tests.csv": pd.DataFrame({"hadm_id": hadm, "itemid": 50001, "valuestr": "1.0 invented units",
                                              "ref_range_lower": 0.5, "ref_range_upper": 2.0}),
        "microbiology.csv": pd.DataFrame({"hadm_id": hadm[:1], "test_itemid": 90001,
                                          "valuestr": "Invented culture", "spec_itemid": 70001}),
        "radiology_reports.csv": pd.DataFrame({"hadm_id": hadm, "note_id": [f"N{x}" for x in hadm], "modality": "CT",
                                               "region": "Abdomen", "exam_name": "Invented CT",
                                               "text": "FINDINGS:\nInvented.\nIMPRESSION:\nNone."}),
        "discharge_diagnosis.csv": pd.DataFrame({"hadm_id": hadm, "discharge_diagnosis": "Invented"}),
        "icd_diagnosis.csv": pd.DataFrame({"hadm_id": hadm, "icd_diagnosis": "Invented code"}),
        "discharge_procedures.csv": pd.DataFrame({"hadm_id": hadm[:1], "discharge_procedure": "Invented"}),
        "icd_procedures.csv": pd.DataFrame({"hadm_id": hadm[:1] * 2, "icd_code": ["0001", "00A0"],
                                            "icd_version": [9, 10], "icd_title": "Invented"}),
        "lab_test_mapping.csv": mapping,
    }
    directory.mkdir()
    for name, frame in tables.items():
        frame.to_csv(directory / name, index=False)
    (directory / "pathology_ids.json").write_text(json.dumps(ids))
    (directory / "LICENSE.txt").write_text("Invented licence.\n")
    names = sorted([*tables, "pathology_ids.json", "LICENSE.txt"])
    (directory / "SHA256SUMS.txt").write_text("".join(
        f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()} {name}\n" for name in names))
    return directory


def _build(raw: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(SCRIPT), "--raw-dir", str(raw), "--out-dir", str(out)],
                          capture_output=True, text=True)


def test_builds_reference_dictionaries_that_validate(tmp_path):
    result = _build(_release(tmp_path / "raw"), tmp_path / "out")
    assert result.returncode == 0, result.stderr
    audit = validate_raw(tmp_path / "out")
    assert audit["class_counts"] == dict.fromkeys(CLASSES, 2)
    # Trusted input: the pickle was just written by the converter under test from invented CSVs.
    with (tmp_path / "out" / f"{CLASSES[0]}_hadm_info_first_diag.pkl").open("rb") as handle:
        record = pickle.load(handle)[100]
    assert list(record) == [
        "Patient History", "Physical Examination", "Laboratory Tests", "Reference Range Lower",
        "Reference Range Upper", "Microbiology", "Microbiology Spec", "Radiology", "Discharge Diagnosis",
        "ICD Diagnosis", "Procedures Discharge", "Procedures ICD9", "Procedures ICD9 Title",
        "Procedures ICD10", "Procedures ICD10 Title",
    ]
    # A code column mixing "0001" and "00A0" is read as text, as in the reference converter.
    assert (record["Procedures ICD9"], record["Procedures ICD10"]) == (["0001"], ["00A0"])
    assert type(next(iter(record["Laboratory Tests"]))) is int


def test_unknown_admission_fails_without_echoing_its_id(tmp_path):
    result = _build(_release(tmp_path / "raw", stray_exam=True), tmp_path / "out")
    assert result.returncode != 0
    assert "physical_examination.csv: 1 rows" in result.stderr
    assert str(UNKNOWN_ADMISSION) not in result.stdout + result.stderr
