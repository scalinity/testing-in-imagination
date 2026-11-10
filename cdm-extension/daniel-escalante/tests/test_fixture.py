from collections import Counter

from tim import fixture
from tim.enums import DISEASES


def test_fixture_baseline_loads_through_the_replication_readers(tmp_path):
    from replication.data import assert_disjoint
    from replication.pipeline import private_read
    fixture.build(tmp_path / "baseline")
    audit = assert_disjoint(tmp_path / "baseline" / "prepared")
    assert {split: audit["splits"][split]["count"] for split in ("train", "val", "test")} == fixture.SIZES
    rows = private_read(tmp_path / "baseline" / "episodes/parity/269/primary-test.json", "synthetic")
    assert {row["label"] for row in rows} == set(DISEASES)
    assert any(row["prediction"] is None for row in rows)
    assert any(row["prediction"] == row["label"] for row in rows)


def test_fixture_is_byte_identical_across_builds(tmp_path):
    for name in ("a", "b"):
        fixture.build(tmp_path / name)
    files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert files
    for relative in files:
        assert (tmp_path / "a" / relative).read_bytes() == (tmp_path / "b" / relative).read_bytes()


def test_every_history_and_generation_carries_a_canary(tmp_path):
    import pandas as pd
    fixture.build(tmp_path / "baseline")
    frame = pd.read_csv(tmp_path / "baseline/prepared/val.csv", keep_default_na=False)
    assert frame["Patient History Summary"].str.contains(fixture.CANARY).all()
    policy = fixture.ScriptedPolicy()
    out = policy.generate(["Here is some context on the patient:\nx [script:direct]\n"])[0]
    assert fixture.CANARY in out.outputs[0].text


def test_scripted_kinds_cover_the_decision_table(tmp_path):
    fixture.build(tmp_path / "baseline")
    kinds = Counter(fixture.kind_of(i) for i in range(fixture.SIZES["test"]))
    assert set(kinds) == set(fixture.KINDS)
