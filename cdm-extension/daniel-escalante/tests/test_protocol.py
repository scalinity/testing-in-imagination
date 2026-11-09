import hashlib
import inspect
import json
from pathlib import Path

from tim import access, analysis, approvals, guard, runner
from tim.paths import OWN, ROOT

PROTOCOL = json.loads((OWN / "prereg" / "protocol.json").read_text())


def test_protocol_freezes_the_code_s_variants_limits_and_thoughts():
    phase_a = PROTOCOL["phase_a"]
    assert phase_a["primary_method"] == "V3" and phase_a["comparator"] == "official"
    assert phase_a["variants"] == runner.CONFIG["variants"]
    assert phase_a["limits"] == runner.CONFIG["limits"]
    assert phase_a["stages"] == runner.CONFIG["stages"]
    assert phase_a["fixed_thoughts"] == {"g1_replace": guard.FIXED_THOUGHT["g1_replace"],
                                         "g2_commit": guard.FIXED_THOUGHT["g2_commit"]}
    assert phase_a["bench_patients_per_class"] == access.BENCH_PER_CLASS
    assert phase_a["smoke_patients"] == access.SMOKE_PATIENTS


def test_protocol_statistics_match_the_implementation():
    stats = PROTOCOL["statistics"]
    defaults = inspect.signature(analysis.randomization_test).parameters
    assert stats["primary"]["draws"] == defaults["draws"].default
    assert stats["primary"]["exact_up_to_discordant"] == 20
    assert stats["primary"]["alpha"] == analysis.ALPHA
    assert stats["family_a"]["members"] == ["V1-official", "V3-V1"]
    assert PROTOCOL["analysis"]["bootstrap_resamples"] == 10_000
    assert PROTOCOL["analysis"]["sensitivity_simulations"] == 10_000
    assert sum(PROTOCOL["analysis"]["test_class_sizes"].values()) == 240


def test_protocol_records_the_admission_path_approver_and_disclosure():
    path = Path(PROTOCOL["admission_dir"])
    assert path.is_absolute() and ROOT not in path.parents
    assert PROTOCOL["approver"] == approvals.APPROVER
    assert "G3" in PROTOCOL["disclosure"]["g3_on_test"]
    assert PROTOCOL["disclosure"]["g3_on_test"] in PROTOCOL["disclosure"]["test_set_status"]


def test_protocol_names_the_spec_it_freezes():
    assert PROTOCOL["spec"]["sha256"] == hashlib.sha256((OWN / "SPEC.md").read_bytes()).hexdigest()
