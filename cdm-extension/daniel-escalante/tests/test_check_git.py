import json
import subprocess

from tim.events import dumps
from tim.publish import check_git

FOLDER = "cdm-extension/daniel-escalante"


def make_repo(tmp_path, files):
    root = tmp_path / "repo"
    for name, content in files.items():
        path = root / FOLDER / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-f", "."], check=True)
    return root


def valid_event():
    return json.dumps({"type": "progress", "completed": 1, "total": 2}) + "\n"


def test_a_clean_folder_passes(tmp_path):
    root = make_repo(tmp_path, {"tim/x.py": "x = 1\n", "results/credentialed/smoke/events/V3.jsonl": valid_event(),
                                "results/credentialed/phase-a/REPORT.md": "# Report\n"})
    assert check_git(root, FOLDER) == []


def test_violations_are_named_by_path_and_rule(tmp_path):
    root = make_repo(tmp_path, {
        "results/credentialed/smoke/rows.json": dumps({"patient_id": 1}),
        "results/credentialed/smoke/events/V3.jsonl": '{"type": "progress", "completed": "TIM-CANARY-1", "total": 1}\n',
        "results/synthetic/phase-a/REPORT.md": "TIM-CANARY-9 leaked\n",
        "notes/physionet-copy.txt": "x\n",
        "mimic-cache/a.txt": "x\n",
        "tim/y.py": "# " + "Clau" + "de wrote this\n",
    })
    found = dict(check_git(root, FOLDER))
    assert found == {
        f"{FOLDER}/results/credentialed/smoke/rows.json": "unexpected public file",
        f"{FOLDER}/results/credentialed/smoke/events/V3.jsonl": "schema violation",
        f"{FOLDER}/results/synthetic/phase-a/REPORT.md": "canary",
        f"{FOLDER}/notes/physionet-copy.txt": "hidden by the root .gitignore",
        f"{FOLDER}/mimic-cache/a.txt": "hidden by the root .gitignore",
        f"{FOLDER}/tim/y.py": "attribution text",
    }
