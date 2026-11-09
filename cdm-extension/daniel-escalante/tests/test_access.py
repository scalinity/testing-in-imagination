import json

import pytest

from tim import access, fixture
from tim.events import TimError
from tim.settings import Settings


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return fixture.build(tmp_path_factory.mktemp("fx") / "baseline")


@pytest.fixture
def settings(environ, baseline, tmp_path):
    return Settings.load(env_file=tmp_path / "none.env", environ={**environ, "CDM_WORK_DIR": str(baseline)})


@pytest.mark.parametrize("stage, split", [
    ("val-a", "test"), ("test-a", "val"), ("smoke", "test"), ("bench-a", "train"),
    ("analyze-val-a", "test"), ("validate", "val"), ("wm-fit", "test"), ("hpi-audit", "test"),
])
def test_the_access_table_denies(settings, stage, split):
    with pytest.raises(TimError) as caught:
        access.open_split(settings, stage, split)
    assert caught.value.code == "access_denied"


def test_listed_stages_see_only_their_patients(settings):
    smoke = access.open_split(settings, "smoke", "val")
    bench = access.open_split(settings, "bench-a", "val")
    assert len(smoke) == 2
    assert len(bench) == 24
    assert {bench[i]["condition"] for i in range(len(bench))} == set(fixture.CUES)
    assert len(access.open_split(settings, "val-a", "val")) == fixture.SIZES["val"]


def test_loaded_ids_are_recorded_for_the_canary_scan(settings):
    access.open_split(settings, "val-a", "val")
    ids = (settings.work_dir / "private/ids/val.txt").read_text().split()
    assert len(ids) == fixture.SIZES["val"] and ids[0] == str(fixture.FIRST_ID["val"])


def test_official_rows_only_for_stages_the_table_allows(settings):
    assert len(access.official_rows(settings, "analyze-a")) == fixture.SIZES["test"]
    with pytest.raises(TimError):
        access.official_rows(settings, "val-a")


def episodes(settings, total=3, split="test"):
    return access.Episodes(settings, "p" * 64, "test-a", "V3", split, "c0ffee", total)


def test_admission_then_resume_without_reevaluation(settings):
    first = episodes(settings)
    assert first.open() == []
    first.append(90002001, {"x": 1})
    first.close()
    resumed = episodes(settings)
    assert resumed.open() == [{"x": 1}]
    resumed.append(90002002, {"x": 2})
    resumed.append(90002003, {"x": 3})
    resumed.finish()
    events = [e["event"] for e in access.Ledger(settings.admission_dir).entries()]
    assert events == ["admitted", "patient", "interrupted", "resumed", "patient", "patient", "completed"]
    with pytest.raises(TimError) as caught:
        episodes(settings).open()
    assert caught.value.code == "admission_refused"


def test_admission_survives_a_new_work_directory(settings, tmp_path):
    first = episodes(settings)
    first.open()
    first.close()
    moved = Settings(source=settings.source, data_dir=settings.data_dir, baseline_dir=settings.baseline_dir,
                     work_dir=tmp_path / "elsewhere", admission_dir=settings.admission_dir)
    with pytest.raises(TimError) as caught:
        episodes(moved).open()
    assert caught.value.code == "resume_mismatch"


def test_a_tampered_episode_log_is_refused(settings):
    run = episodes(settings)
    run.open()
    run.append(90002001, {"x": 1})
    run.close()
    line = json.loads(run.log.read_text())
    line["row"] = {"x": 9}
    run.log.write_text(json.dumps(line) + "\n")
    with pytest.raises(TimError) as caught:
        episodes(settings).open()
    assert caught.value.code == "resume_mismatch"


def test_validation_keys_include_the_commit_but_test_keys_do_not(settings):
    assert access.run_key("a" * 64, "test-a", "V3", "test", "c0ffee") == access.run_key("a" * 64, "test-a", "V3", "test", "beef00")
    assert access.run_key("a" * 64, "val-a", "V3", "val", "c0ffee") != access.run_key("a" * 64, "val-a", "V3", "val", "beef00")


def test_credentialed_runs_require_the_protocols_admission_path(settings, tmp_path):
    mimic = Settings(source="mimic", data_dir=settings.data_dir, baseline_dir=settings.baseline_dir,
                     work_dir=settings.work_dir, admission_dir=settings.admission_dir)
    access.check_admission_path(mimic, {"admission_dir": str(settings.admission_dir)})
    with pytest.raises(TimError) as caught:
        access.check_admission_path(mimic, {"admission_dir": str(tmp_path / "other")})
    assert caught.value.code == "admission_refused"


def test_loading_a_split_never_opens_another_split(settings, tmp_path):
    import shutil
    prepared = tmp_path / "split-only"
    shutil.copytree(settings.baseline_dir, prepared)
    for other in ("train.csv", "test.csv"):
        (prepared / "prepared" / other).chmod(0)
    local = Settings(source=settings.source, data_dir=settings.data_dir, baseline_dir=prepared,
                     work_dir=settings.work_dir, admission_dir=settings.admission_dir)
    try:
        assert len(access.open_split(local, "val-a", "val")) == fixture.SIZES["val"]
        assert len(access.open_split(local, "smoke", "val")) == 2
    finally:
        for other in ("train.csv", "test.csv"):
            (prepared / "prepared" / other).chmod(0o644)


def test_listed_stages_get_a_private_data_dir_with_only_their_patients(settings):
    import pandas as pd
    bench = access.open_split(settings, "bench-a", "val")
    files = sorted(p.name for p in bench.data_dir.iterdir())
    assert files == ["lab_test_mapping.csv", "val.csv"]
    assert len(pd.read_csv(bench.data_dir / "val.csv")) == 24


def test_a_running_admission_cannot_be_resumed_by_a_second_job(settings):
    first = episodes(settings)
    first.open()
    with pytest.raises(TimError) as caught:
        episodes(settings).open()
    assert caught.value.code == "admission_refused"
    first.close()
    assert episodes(settings).open() == []


def test_a_row_written_before_its_ledger_entry_is_recovered(settings):
    run = episodes(settings)
    run.open()
    run.append(90002001, {"x": 1})
    with run.log.open("a") as handle:  # crash after the row, before the ledger record
        line = {"patient_id": 90002002, "row": {"x": 2}, "sha256": access.digest({"x": 2})}
        handle.write(json.dumps(line, sort_keys=True) + "\n")
    run.close()
    assert episodes(settings).open() == [{"x": 1}, {"x": 2}]


def test_a_truncated_final_line_is_dropped_and_its_patient_rerun(settings):
    run = episodes(settings)
    run.open()
    run.append(90002001, {"x": 1})
    with run.log.open("a") as handle:
        handle.write('{"patient_id": 90002002, "row": {"x"')
    run.close()
    resumed = episodes(settings)
    assert resumed.open() == [{"x": 1}]
    assert resumed.done == {90002001}


def test_a_run_with_every_row_but_no_finish_completes_without_evaluation(settings):
    run = episodes(settings)
    run.open()
    for i in range(3):
        run.append(90002001 + i, {"x": i})
    run.close()
    resumed = episodes(settings)
    assert len(resumed.open()) == 3 and resumed.done == {90002001, 90002002, 90002003}
    resumed.finish()
    assert access.Ledger(settings.admission_dir).entries()[-1]["event"] == "completed"
