#!/usr/bin/env python3
"""Convert the PhysioNet MIMIC-IV-Ext-CDM CSV release into the four condition pickles.

The upstream LA-CDM splitter reads `<condition>_hadm_info_first_diag.pkl` dictionaries,
which Hager et al. build from the CSV release with ConvertPhysionet.py in
github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset at 4d33f98 (MIT licence).
This follows that script's field names, insertion order and value types; it omits the
unused lab_test_mapping.pkl, since the mapping CSV is copied instead. The checksum-verified
raw directory is only read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import pandas as pd

OWN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OWN))
from replication.config import external_path

REFERENCE = ("github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset@"
             "4d33f98b30491152c9432986afb356b6991ce51f:ConvertPhysionet.py")
CONDITIONS = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name: str, *columns: str):
    # Series iteration yields Python scalars, matching DataFrame.iterrows on these mixed-type CSVs.
    frame = pd.read_csv(raw / name)
    return zip(*(frame[column] for column in columns))


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--raw-dir", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
args = parser.parse_args()
raw, out = external_path(args.raw_dir), external_path(args.out_dir)

# PhysioNet writes "<digest> <name>" with one space, which `shasum -c` rejects while exiting 0.
expected = dict(line.split()[::-1] for line in (raw / "SHA256SUMS.txt").read_text().splitlines() if line.strip())
mismatched = sorted(name for name, digest in expected.items() if sha256(raw / name) != digest)
if mismatched or len(expected) != 12:
    sys.exit(f"raw checksum verification failed: {mismatched or f'{len(expected)} listed files, expected 12'}")
if out == raw or raw in out.parents or out in raw.parents:
    sys.exit("output directory must be separate from the raw directory")
if out.exists() and any(out.iterdir()):
    sys.exit("output directory must be new or empty")

hadm = {hadm_id: {} for hadm_id in pd.read_csv(raw / "history_of_present_illness.csv")["hadm_id"].to_list()}
for hadm_id, text in rows("history_of_present_illness.csv", "hadm_id", "hpi"):
    hadm[hadm_id]["Patient History"] = text
for hadm_id, text in rows("physical_examination.csv", "hadm_id", "pe"):
    hadm[hadm_id]["Physical Examination"] = text
for hadm_id, itemid, value, lower, upper in rows(
        "laboratory_tests.csv", "hadm_id", "itemid", "valuestr", "ref_range_lower", "ref_range_upper"):
    record = hadm[hadm_id]
    if "Laboratory Tests" not in record:
        record.update({"Laboratory Tests": {}, "Reference Range Lower": {}, "Reference Range Upper": {}})
    record["Laboratory Tests"][itemid] = value
    record["Reference Range Lower"][itemid] = lower
    record["Reference Range Upper"][itemid] = upper
for hadm_id, itemid, value, specimen in rows("microbiology.csv", "hadm_id", "test_itemid", "valuestr", "spec_itemid"):
    record = hadm[hadm_id]
    if "Microbiology" not in record:
        record.update({"Microbiology": {}, "Microbiology Spec": {}})
    record["Microbiology"][itemid] = value
    record["Microbiology Spec"][itemid] = specimen
for hadm_id, note, modality, region, exam, text in rows(
        "radiology_reports.csv", "hadm_id", "note_id", "modality", "region", "exam_name", "text"):
    hadm[hadm_id].setdefault("Radiology", []).append(
        {"Note ID": note, "Modality": modality, "Region": region, "Exam Name": exam, "Report": text})
for hadm_id, text in rows("discharge_diagnosis.csv", "hadm_id", "discharge_diagnosis"):
    hadm[hadm_id]["Discharge Diagnosis"] = text
for hadm_id, code in rows("icd_diagnosis.csv", "hadm_id", "icd_diagnosis"):
    hadm[hadm_id].setdefault("ICD Diagnosis", []).append(code)
for hadm_id, text in rows("discharge_procedures.csv", "hadm_id", "discharge_procedure"):
    hadm[hadm_id].setdefault("Procedures Discharge", []).append(text)
for hadm_id, code, version, title in rows("icd_procedures.csv", "hadm_id", "icd_code", "icd_version", "icd_title"):
    record = hadm[hadm_id]
    if "Procedures ICD9" not in record:
        record.update({"Procedures ICD9": [], "Procedures ICD9 Title": [],
                       "Procedures ICD10": [], "Procedures ICD10 Title": []})
    system = "ICD9" if version == 9 else "ICD10"
    record[f"Procedures {system}"].append(code)
    record[f"Procedures {system} Title"].append(title)

pathology = json.loads((raw / "pathology_ids.json").read_text())
out.mkdir(mode=0o700, parents=True, exist_ok=True)
for condition in CONDITIONS:
    with (out / f"{condition}_hadm_info_first_diag.pkl").open("xb") as handle:
        pickle.dump({hadm_id: hadm[hadm_id] for hadm_id in pathology[condition]}, handle)
shutil.copyfile(raw / "lab_test_mapping.csv", out / "lab_test_mapping.csv")

record = {
    "reference_converter": REFERENCE,
    "raw_sha256sums_sha256": sha256(raw / "SHA256SUMS.txt"),
    "raw_files_verified": len(expected),
    "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": sys.modules["numpy"].__version__,
    "outputs": {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()},
}
(out / ".conversion.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
