"""E0 fixtures, part 8: the audit card (SPEC 9.1) and the reader / world-model entry point (SPEC 16).

The card is checked on three decision points whose values are worked by hand elsewhere:

  anchor-A (test_cee_core.py, test_cee_measures.py, test_cee_identity.py, test_cee_decision.py)
      drift (0.2, -0.2): L1 0.4, largest component 0.2; responsiveness 0.024157; KL_inc 0.092010;
      coherent VOI under 1[correct] 1.1 - 0.4 sqrt6 = 0.120204; real outcome "negative" (the second
      outcome) with label "alpha" (the first class) gives realized gain
      log u(alpha) - log Z_negative = log 1 - log(0.6 + 0.4 sqrt6) = -0.457296. The outcome and the
      label sit at different positions on purpose, so that swapping them cannot go unnoticed.

  anchor-S (test_cee_scale.py), with its budget 0.192745
      coherent, so drift 0 and KL_inc 0; responsiveness log 2 - H(16/17, 1/17) = 0.469429;
      tau_B = 2 under both intervals; coherent VOI 16/17 - 1/2 = 0.441176; no label.

  anchor-D (test_cee_core.py), with a supplied utility table
      drift (-0.3, 0.3): L1 0.6, largest component 0.3; responsiveness 0, so tau_ratio is undefined;
      KL_inc = KL((0.9, 0.1) || (0.6, 0.4)) = 0.226289; coherent VOI 0 for any utility because both
      corrected posteriors equal the prior; u proportional to (1.5, 0.25) with Z = 1, so the real
      outcome "negative" with label "beta" gives realized gain log 0.25 = -1.386294.

Summary means over the three points: drift L1 (0.4 + 0 + 0.6)/3 = 1/3, largest component
(0.2 + 0 + 0.3)/3 = 1/6.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tiicee
from tiicee.cee import TAU_RATIO_GRID, audit_point, audit_record
from tiicee.cee.audit import main
from tiicee.interfaces import UNAVAILABLE, Predictive

SQRT6 = math.sqrt(6.0)
A_KL_INC = 0.25 * math.log(6.0) - 0.5 * math.log(0.96 + 0.44 * SQRT6)
A_RESPONSIVENESS = 0.4 * math.log(8 / 7) + 0.1 * math.log(2 / 3) + 0.3 * math.log(6 / 7) + 0.2 * math.log(4 / 3)
A_VOI = 1.1 - 0.4 * SQRT6
A_GAIN = -math.log(0.6 + 0.4 * SQRT6)
S_BUDGET = math.log(2.0) + 0.8 * math.log(0.8) + 0.2 * math.log(0.2)
S_RESPONSIVENESS = math.log(2.0) + (16 / 17) * math.log(16 / 17) + (1 / 17) * math.log(1 / 17)
D_KL_INC = 0.9 * math.log(1.5) + 0.1 * math.log(0.25)

POINTS = [
    {
        "id": "anchor-A",
        "classes": ["alpha", "beta"],
        "prior": [0.5, 0.5],
        "outcomes": ["positive", "negative"],
        "weights": [0.5, 0.5],
        "posteriors": [[0.8, 0.2], [0.6, 0.4]],
        "real_outcome": "negative",
        "y_true": "alpha",
    },
    {
        "id": "anchor-S",
        "classes": ["alpha", "beta"],
        "prior": [0.5, 0.5],
        "outcomes": ["positive", UNAVAILABLE],
        "weights": [0.5, 0.5],
        "posteriors": [[16 / 17, 1 / 17], [1 / 17, 16 / 17]],
        "budget": S_BUDGET,
    },
    {
        "id": "anchor-D",
        "classes": ["alpha", "beta"],
        "prior": [0.9, 0.1],
        "outcomes": ["positive", "negative"],
        "weights": [0.5, 0.5],
        "posteriors": [[0.6, 0.4], [0.6, 0.4]],
        "utility": [[2 / 3, 0.0], [0.0, 2.0]],
        "real_outcome": "negative",
        "y_true": "beta",
    },
]


@pytest.fixture()
def card_file(tmp_path):
    path = tmp_path / "points.json"
    path.write_text(json.dumps(POINTS))
    return path


def test_record_for_anchor_a():
    r = audit_record(POINTS[0])
    assert r["id"] == "anchor-A" and r["n_classes"] == 2 and r["n_outcomes"] == 2
    assert r["drift_l1"] == pytest.approx(0.4, abs=1e-12) and r["drift_max"] == pytest.approx(0.2, abs=1e-12)
    assert r["responsiveness"] == pytest.approx(A_RESPONSIVENESS, abs=1e-12)
    assert r["kl_inc"] == pytest.approx(A_KL_INC, abs=1e-6)
    assert r["converged"] is True and 0 < r["iterations"] <= 500 and r["max_violation"] <= 1e-8
    assert r["budget"] is None
    assert r["tau_b_shrink_only"] is None and r["tau_b_shrink_only_path"] is None
    assert r["tau_b_two_sided"] is None and r["tau_b_two_sided_path"] is None
    assert r["tau_ratio"] in TAU_RATIO_GRID
    assert r["utility"] == "identity"
    assert r["coherent_voi"] == pytest.approx(A_VOI, abs=1e-6)
    assert r["realized_gain"] == pytest.approx(A_GAIN, abs=1e-6)
    assert A_GAIN == pytest.approx(-0.457296, abs=5e-7)


def test_record_for_anchor_s_reports_tau_b_under_both_intervals():
    r = audit_record(POINTS[1])
    assert r["drift_l1"] == pytest.approx(0.0, abs=1e-12) and r["kl_inc"] == pytest.approx(0.0, abs=1e-12)
    assert r["responsiveness"] == pytest.approx(S_RESPONSIVENESS, abs=1e-12)
    assert S_RESPONSIVENESS == pytest.approx(0.469429, abs=5e-7)
    assert r["budget"] == pytest.approx(S_BUDGET)
    assert abs(r["tau_b_shrink_only"] / 2.0 - 1.0) <= 0.01 and r["tau_b_shrink_only_path"] == "solved"
    assert abs(r["tau_b_two_sided"] / 2.0 - 1.0) <= 0.01 and r["tau_b_two_sided_path"] == "solved"
    assert r["coherent_voi"] == pytest.approx(16 / 17 - 0.5, abs=1e-12)
    assert r["realized_gain"] is None                      # no label, no realized gain


def test_record_for_anchor_d_uses_the_supplied_utility_and_leaves_tau_ratio_undefined():
    r = audit_record(POINTS[2])
    assert r["drift_l1"] == pytest.approx(0.6, abs=1e-12) and r["drift_max"] == pytest.approx(0.3, abs=1e-12)
    assert r["responsiveness"] == pytest.approx(0.0, abs=1e-12)
    assert r["kl_inc"] == pytest.approx(D_KL_INC, abs=1e-9)
    assert r["tau_ratio"] is None                          # a reader that never moves has no scale to report
    assert r["utility"] == "supplied"
    assert r["coherent_voi"] == pytest.approx(0.0, abs=1e-9)
    assert r["realized_gain"] == pytest.approx(math.log(0.25), abs=1e-9)


def test_a_non_converged_point_reports_no_coherent_quantities():
    r = audit_record(POINTS[0], max_iter=1)                # the anchor needs more than one iteration
    assert r["converged"] is False and r["iterations"] == 1
    assert r["coherent_voi"] is None and r["realized_gain"] is None
    assert r["drift_l1"] == pytest.approx(0.4, abs=1e-12)  # the raw quantities are still reported


def test_records_are_plain_json():
    for point in POINTS:
        r = audit_record(point)
        assert json.loads(json.dumps(r)) == r
        assert all(v is None or isinstance(v, (bool, int, float, str)) for v in r.values())


@pytest.mark.parametrize(
    "change",
    [
        {"weights": [0.5, 0.6]},                           # the world model does not sum to one
        {"classes": ["alpha", "beta", "gamma"]},           # class list and prior disagree
        {"outcomes": ["positive"]},                        # outcome list and posteriors disagree
        {"real_outcome": "equivocal"},                     # the real outcome was never imagined
        {"y_true": "gamma"},                               # the label is not a class
        {"utility": [[1.0, 0.0, 0.0]]},                    # utility columns are not the classes
    ],
)
def test_malformed_points_are_rejected(change):
    with pytest.raises(ValueError):
        audit_record({**POINTS[0], **change})


def test_missing_required_field_is_rejected():
    point = {k: v for k, v in POINTS[0].items() if k != "prior"}
    with pytest.raises(ValueError):
        audit_record(point)


# ------------------------------------------------------------------------------------------------
# The command
# ------------------------------------------------------------------------------------------------
def test_command_prints_a_markdown_card(card_file, capsys):
    assert main([str(card_file)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# CEE audit card")
    assert "floor 1e-06" in out and "tolerance 1e-08" in out and "500 iterations" in out
    assert "## Decision points" in out and "## Summary" in out
    for point in POINTS:
        assert f"| {point['id']} |" in out
    assert "solved" in out and "n/a" in out
    header = next(line for line in out.splitlines() if line.startswith("| id |"))
    for column in ("drift L1", "drift max", "responsiveness", "KL_inc", "converged", "tau_B shrink-only", "tau_B two-sided", "tau_ratio", "coherent VOI", "realized gain"):
        assert column in header
    rows = [line for line in out.splitlines() if line.startswith("| anchor-")]
    assert len(rows) == 3 and all(row.count("|") == header.count("|") for row in rows)
    # anchor-D: KL_inc 0.2263, no tau_ratio, coherent VOI exactly 0 on the card, realized gain log 0.25.
    cells = [cell.strip() for cell in rows[2].strip("|").split("|")]
    assert cells == ["anchor-D", "2", "2", "0.6", "0.3", "0", "0.2263", "yes (1)", "n/a", "n/a", "n/a", "0", "-1.386"]
    assert "e-16" not in out and "e-17" not in out          # rounding noise is not printed as a value


def test_command_prints_json(card_file, capsys):
    assert main([str(card_file), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["floor"] == 1e-6 and payload["tol"] == 1e-8 and payload["max_iter"] == 500
    assert payload["source"] == str(card_file)
    assert payload["points"] == [audit_record(point) for point in POINTS]
    s = payload["summary"]
    assert s["n_points"] == 3 and s["n_converged"] == 3 and s["n_labeled"] == 2 and s["n_budget"] == 1
    assert s["mean_drift_l1"] == pytest.approx(1 / 3, abs=1e-12)
    assert s["mean_drift_max"] == pytest.approx(1 / 6, abs=1e-12)
    assert s["mean_responsiveness"] == pytest.approx((A_RESPONSIVENESS + S_RESPONSIVENESS) / 3, abs=1e-12)
    assert s["mean_kl_inc"] == pytest.approx((A_KL_INC + D_KL_INC) / 3, abs=1e-6)
    assert s["mean_coherent_voi"] == pytest.approx((A_VOI + 16 / 17 - 0.5) / 3, abs=1e-6)
    assert s["mean_realized_gain"] == pytest.approx((A_GAIN + math.log(0.25)) / 2, abs=1e-6)
    assert abs(s["geomean_tau_b_shrink_only"] / 2.0 - 1.0) <= 0.01 and abs(s["geomean_tau_b_two_sided"] / 2.0 - 1.0) <= 0.01
    assert s["n_tau_ratio"] == 2                           # anchor-D has none


def test_command_reports_a_bad_point_and_fails(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([POINTS[0], {**POINTS[1], "weights": [0.9, 0.9]}]))
    assert main([str(path)]) == 2
    err = capsys.readouterr().err
    assert "anchor-S" in err and "weights" in err


def test_command_runs_as_a_module(card_file):
    root = Path(tiicee.__file__).resolve().parent.parent
    done = subprocess.run(
        [sys.executable, "-m", "tiicee.cee.audit", str(card_file), "--json"], cwd=root, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    assert [p["id"] for p in json.loads(done.stdout)["points"]] == ["anchor-A", "anchor-S", "anchor-D"]


# ------------------------------------------------------------------------------------------------
# Any reader and any world model, behind the interfaces of SPEC 16
# ------------------------------------------------------------------------------------------------
class _Reader:
    classes = ("alpha", "beta")

    def __init__(self):
        self.read_calls = []

    def belief(self, history):
        return np.array([0.5, 0.5])

    def read(self, history, test, outcome):
        self.read_calls.append((tuple(history), test, outcome))
        return {"positive": np.array([0.8, 0.2]), "negative": np.array([0.6, 0.4])}[outcome]


class _World:
    def __init__(self):
        self.calls = []

    def outcomes(self, history, test, k, rng):
        self.calls.append((tuple(history), test, k, rng))
        return Predictive(outcomes=("positive", "negative"), weights=np.array([0.5, 0.5]))


def test_audit_point_calls_the_interfaces_and_returns_the_same_record():
    reader, world, history = _Reader(), _World(), ["complaint: chest pain"]
    rng = np.random.default_rng(0)
    r = audit_point(reader, world, history, "troponin", k=7, rng=rng, budget=0.01)
    assert world.calls == [(("complaint: chest pain",), "troponin", 7, rng)]
    assert reader.read_calls == [(("complaint: chest pain",), "troponin", "positive"), (("complaint: chest pain",), "troponin", "negative")]
    assert history == ["complaint: chest pain"]            # the imagined outcomes were not written into the history
    by_file = audit_record({**{k: v for k, v in POINTS[0].items() if k not in ("real_outcome", "y_true")}, "id": "troponin", "budget": 0.01})
    assert r == by_file
    assert r["id"] == "troponin" and r["realized_gain"] is None
    assert r["tau_b_two_sided"] is not None and r["tau_b_shrink_only_path"] in ("solved", "clipped_low", "clipped_high")


def test_audit_point_defaults_to_exact_enumeration_and_no_budget():
    world = _World()
    r = audit_point(_Reader(), world, [], "troponin")
    assert world.calls == [((), "troponin", None, None)]
    assert r["budget"] is None and r["tau_b_two_sided"] is None
    assert r["kl_inc"] == pytest.approx(A_KL_INC, abs=1e-6)
