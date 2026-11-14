"""E10's cohort-X integrity check over the outputs the fold stages wrote (tiicee/clinical/cohort_x.py)."""
import json
import shutil

import pytest

from test_clinical_support import approved, context, prepared, split_bytes
from tiicee.clinical import admissions, cohort_x, folds, sha256
from tiicee.clinical.train import inspect_data
from tiicee.pipeline import export_guard


def members(directory, part):
    return admissions.read((directory / f"{part}.csv").read_bytes()).admissions


def write_outputs(root, folds_dir):
    """What e10-train-fk and e10-rollout-fk leave behind, for a run that went as planned."""
    bound = {"folds": folds_dir}
    for k in range(folds.FOLDS):
        fold = folds_dir / f"fold-{k}"
        policy, rollout = root / f"e10-train-f{k}" / "out", root / f"e10-rollout-f{k}" / "out"
        policy.mkdir(parents=True)
        rollout.mkdir(parents=True)
        adapter = sha256(f"adapter of fold {k}".encode())
        (policy / "policy.json").write_text(json.dumps({"data": inspect_data(fold)["files"], "adapter_sha256": adapter}))
        (rollout / "rollout.json").write_text(json.dumps({
            "split_sha256": sha256((fold / "heldout.csv").read_bytes()), "adapter_sha256": adapter}))
        held = members(fold, "heldout")
        (rollout / "steps.jsonl").write_text("".join(
            json.dumps({"fingerprint": a.fingerprint, "patient_id": a.patient_id, "steps": []}) + "\n" for a in held))
        (rollout / "rows.jsonl").write_text("".join(json.dumps({"prediction": a.label}) + "\n" for a in held))
        bound[f"policy_f{k}"], bound[f"rollout_f{k}"] = policy, rollout
    return bound


@pytest.fixture
def ran(tmp_path):
    folds_dir = tmp_path / "stages" / "e10-folds" / "out"
    folds.write_folds(prepared(tmp_path), folds_dir)
    return write_outputs(tmp_path / "stages", folds_dir)


def run(tmp_path, bound):
    ctx = context(tmp_path, "e10-cohort-x", bound=bound)
    cohort_x.run(ctx)
    return ctx, approved(ctx)


def test_a_run_as_planned_leaves_cohort_x_intact(tmp_path, ran):
    ctx, report = run(tmp_path, ran)
    assert report["outcome"] == "intact" and report["reasons"] == []
    assert report["admissions"] == report["scored_once"] == 120  # the fixture's 96 + 24
    for k, fold in enumerate(report["folds"]):
        held = len(members(ran["folds"] / f"fold-{k}", "heldout"))
        assert fold == {"fold": k, "heldout": held, "scored": held, "v0_results": held, "leaks": 0,
                        "policy_data": True, "rollout_split": True, "rollout_adapter": True}
    export_guard.check_aggregate(report)
    assert json.loads((ctx.out_dir / "integrity.json").read_text()) == report


def test_scoring_an_admission_the_policy_trained_on_kills_cohort_x(tmp_path, ran):
    fold = ran["folds"] / "fold-2"
    trained = members(fold, "train")[0]
    steps = ran["rollout_f2"] / "steps.jsonl"
    lines = steps.read_text().splitlines(keepends=True)
    lines[0] = json.dumps({"fingerprint": trained.fingerprint, "patient_id": trained.patient_id, "steps": []}) + "\n"
    steps.write_text("".join(lines))
    _, report = run(tmp_path, ran)
    assert report["outcome"] == "killed"
    assert {"f2:leak", "f2:outside_heldout", "f2:missing", "not_once"} <= set(report["reasons"])
    assert report["folds"][2]["leaks"] == 1


def test_a_monitored_admission_counts_as_a_leak_too(tmp_path, ran):
    watched = members(ran["folds"] / "fold-4", "val")[0]
    steps = ran["rollout_f4"] / "steps.jsonl"
    steps.write_text(steps.read_text() + json.dumps({"fingerprint": watched.fingerprint, "steps": []}) + "\n")
    _, report = run(tmp_path, ran)
    assert report["outcome"] == "killed" and "f4:leak" in report["reasons"]


@pytest.mark.parametrize("record, key, reason", [
    ("rollout_f1/rollout.json", "split_sha256", "f1:rollout_split"),
    ("rollout_f1/rollout.json", "adapter_sha256", "f1:rollout_adapter"),
])
def test_a_rollout_of_another_file_or_policy_kills_cohort_x(tmp_path, ran, record, key, reason):
    role, name = record.split("/")
    path = ran[role] / name
    value = json.loads(path.read_text())
    value[key] = sha256(b"something else")
    path.write_text(json.dumps(value))
    _, report = run(tmp_path, ran)
    assert report["outcome"] == "killed" and report["reasons"] == [reason]


def test_a_policy_trained_on_another_folds_files_kills_cohort_x(tmp_path, ran):
    path = ran["policy_f3"] / "policy.json"
    value = json.loads(path.read_text())
    value["data"] = inspect_data(ran["folds"] / "fold-0")["files"]
    path.write_text(json.dumps(value))
    _, report = run(tmp_path, ran)
    assert report["outcome"] == "killed" and report["reasons"] == ["f3:policy_data"]


@pytest.mark.parametrize("name", cohort_x.FILES)
def test_each_file_the_policy_recorded_is_checked(tmp_path, ran, name):
    path = ran["policy_f3"] / "policy.json"
    value = json.loads(path.read_text())
    value["data"][name] = sha256(b"another file")
    path.write_text(json.dumps(value))
    _, report = run(tmp_path, ran)
    assert report["outcome"] == "killed" and report["reasons"] == ["f3:policy_data"]


def test_an_admission_scored_twice_or_not_at_all_kills_cohort_x(tmp_path, ran):
    steps = ran["rollout_f0"] / "steps.jsonl"
    lines = steps.read_text().splitlines(keepends=True)
    steps.write_text("".join([lines[0]] + lines[:-1]))  # the first twice, the last never
    _, report = run(tmp_path, ran)
    assert report["outcome"] == "killed"
    assert set(report["reasons"]) == {"f0:duplicate", "f0:missing", "not_once"}
    assert report["scored_once"] == report["admissions"] - 2


def test_v0_results_that_do_not_match_the_scored_admissions_kill_cohort_x(tmp_path, ran):
    rows = ran["rollout_f3"] / "rows.jsonl"
    rows.write_text("".join(rows.read_text().splitlines(keepends=True)[:-1]))
    _, report = run(tmp_path, ran)
    assert report["outcome"] == "killed" and report["reasons"] == ["f3:v0_count"]


def test_fold_files_that_no_longer_partition_the_admissions_kill_cohort_x(tmp_path, ran):
    fold = ran["folds"] / "fold-1"
    held = (fold / "heldout.csv").read_bytes()
    (fold / "train.csv").write_bytes((fold / "train.csv").read_bytes() + held.split(b"\n", 1)[1])
    _, report = run(tmp_path, ran)
    assert report["outcome"] == "killed" and "fold_partition" in report["reasons"]


def refused_without_a_report(tmp_path, bound, error):
    """The stage itself raises, and neither the private report nor the approved one is written."""
    ctx = context(tmp_path, "e10-cohort-x", bound=bound)
    with pytest.raises(error):
        cohort_x.run(ctx)
    assert not (ctx.out_dir / "integrity.json").exists()
    assert not (ctx.paths.approved / "e10-cohort-x.json").exists()


@pytest.mark.parametrize("missing", ["rollout_f2", "rollout_f2/rollout.json", "rollout_f2/steps.jsonl",
                                     "rollout_f2/rows.jsonl", "policy_f0/policy.json", "folds/fold-3/heldout.csv",
                                     "folds/folds.jsonl"])
def test_a_missing_output_is_an_error_not_a_result(tmp_path, ran, missing):
    role, _, name = missing.partition("/")
    path = ran[role] / name if name else ran[role]
    shutil.rmtree(path) if path.is_dir() else path.unlink()
    refused_without_a_report(tmp_path, ran, FileNotFoundError)


@pytest.mark.parametrize("damage", ["truncated", "not_an_object"])
def test_a_malformed_v0_result_is_an_error_not_a_count(tmp_path, ran, damage):
    rows = ran["rollout_f1"] / "rows.jsonl"
    lines = rows.read_text().splitlines(keepends=True)
    lines[-1] = lines[-1][:len(lines[-1]) // 2] + "\n" if damage == "truncated" else "[]\n"
    rows.write_text("".join(lines))  # the line count is unchanged
    refused_without_a_report(tmp_path, ran, ValueError)


def test_an_admission_dropped_from_every_fold_file_kills_cohort_x(tmp_path):
    """Consistent omission: every file, digest and log describes the shortened folds truthfully, and
    only the inventory written when the folds were cut still knows the admission."""
    folds_dir = tmp_path / "stages" / "e10-folds" / "out"
    folds.write_folds(prepared(tmp_path), folds_dir)
    gone = members(folds_dir / "fold-0", "heldout")[0].fingerprint
    for k in range(folds.FOLDS):
        for part in folds.PARTS:
            path = folds_dir / f"fold-{k}" / f"{part}.csv"
            table = admissions.read(path.read_bytes())
            path.write_bytes(table.subset([a for a in table.admissions if a.fingerprint != gone]))
    _, report = run(tmp_path, write_outputs(tmp_path / "stages", folds_dir))
    assert report["outcome"] == "killed" and set(report["reasons"]) == {"fold_partition", "not_once"}
    assert (report["admissions"], report["scored_once"]) == (120, 119)


def test_an_admission_held_out_by_another_fold_than_recorded_kills_cohort_x(tmp_path, ran):
    inventory = ran["folds"] / "folds.jsonl"
    lines = inventory.read_text().splitlines(keepends=True)
    record = json.loads(lines[0])
    lines[0] = json.dumps({**record, "fold": (record["fold"] + 1) % folds.FOLDS}) + "\n"
    inventory.write_text("".join(lines))
    _, report = run(tmp_path, ran)
    assert report["outcome"] == "killed" and report["reasons"] == ["fold_partition"]


def test_an_admission_added_to_the_fold_files_kills_cohort_x(tmp_path):
    """The other direction: an admission the inventory never recorded, held out by fold 0 and trained
    on by the rest, every output describing it truthfully. The cohort stays the recorded one."""
    folds_dir = tmp_path / "stages" / "e10-folds" / "out"
    folds.write_folds(prepared(tmp_path), folds_dir)
    extra = admissions.read(split_bytes(99000, ["appendicitis"])).admissions[0].raw
    for k, part in [(0, "heldout")] + [(k, "train") for k in range(1, folds.FOLDS)]:
        path = folds_dir / f"fold-{k}" / f"{part}.csv"
        path.write_bytes(path.read_bytes() + extra)
    _, report = run(tmp_path, write_outputs(tmp_path / "stages", folds_dir))
    assert report["outcome"] == "killed" and set(report["reasons"]) == {"fold_partition", "not_once"}
    assert (report["admissions"], report["scored_once"]) == (120, 120)
