"""Conformance of tiicee.stats with the project's existing statistics.

Three independent references, strongest first:

1. Live: the project's own code (tim/analysis.py and replication/stats.py) run by the project's
   own interpreter (cdm-extension/daniel-escalante/.venv) on fixed synthetic inputs, compared
   to 1e-12. replication/stats.py imports scipy and the pinned upstream package at module
   level, and tim/analysis.py needs scipy for its exact p-value, power and MDE; this
   environment has numpy alone, so they cannot run in this process. These tests are skipped
   where that interpreter is absent.
2. In process: the parts of tim/analysis.py that need numpy alone (weights, Delta T5 and the
   enumerated randomization test), loaded from its file.
3. Recorded: numbers in the committed result bundles whose inputs are reconstructible from
   committed aggregates: every Delta T5, the sensitivity block (power and MDE, digit for digit),
   the Holm-adjusted families, exact p-values, and intervals up to Monte Carlo error.
"""
import hashlib
import importlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tiicee import stats

REPO = Path(__file__).resolve().parents[2]
EXTENSION = REPO / "cdm-extension" / "daniel-escalante"
BUNDLES = EXTENSION / "results" / "credentialed"
PROJECT_PYTHON = EXTENSION / ".venv" / "bin" / "python"

CLASSES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
TEST_SIZES = dict(zip(CLASSES, (96, 65, 25, 54)))  # SPEC section 5 of the extension: 96 / 65 / 25 / 54


# ---- shared builders ------------------------------------------------------------------------

def synthetic_pair(seed, sizes, flips, right=0.7):
    """Paired predictions in mixed row order: b is right with probability `right`, and a differs
    from b in correctness on exactly `flips` units."""
    rng = np.random.default_rng(seed)
    truth = [label for label, n in zip(CLASSES, sizes) for _ in range(n)]

    def wrong(label):
        return [None, *(c for c in CLASSES if c != label)][rng.integers(0, 4)]

    b = [label if rng.random() < right else wrong(label) for label in truth]
    a = list(b)
    for i in rng.choice(len(truth), size=flips, replace=False):
        a[i] = truth[i] if b[i] != truth[i] else wrong(truth[i])
    order = rng.permutation(len(truth))
    return [a[i] for i in order], [b[i] for i in order], [truth[i] for i in order]


def pair_from_changes(changes):
    """Paired predictions with the given per-class improved / regressed / unchanged counts."""
    a, b, truth = [], [], []
    for label in CLASSES:
        improved, regressed, unchanged = (changes[label][k] for k in ("improved", "regressed", "unchanged"))
        truth += [label] * (improved + regressed + unchanged)
        a += [label] * improved + [None] * regressed + [label] * unchanged
        b += [None] * improved + [label] * regressed + [label] * unchanged
    return a, b, truth


def brute_force_p(a, b, truth):
    """Enumerate every sign assignment of the discordant units (ties count as extreme)."""
    size = {label: truth.count(label) for label in CLASSES}
    v = np.array([100 / (4 * size[t]) * ((x == t) - (y == t)) for x, y, t in zip(a, b, truth) if (x == t) != (y == t)])
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(v))))
    return float(np.mean(np.abs(signs @ v) >= abs(v.sum()) - 1e-9))


def bundle(*parts):
    path = BUNDLES.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"{path.relative_to(REPO)} is not in this checkout")
    return json.loads(path.read_text())


def correct_counts(variant):
    return {label: variant["confusion"][label][label] for label in CLASSES}


def class_sizes(variant):
    return {label: sum(variant["confusion"][label].values()) for label in CLASSES}


# ---- 1. live: the project's code under the project's interpreter ------------------------------

REFERENCE_SCRIPT = r"""
import json, sys
sys.path.insert(0, ".")
import tim.paths  # puts the replication and the pinned upstream on sys.path
from tim import analysis
from replication.config import compose
from replication.stats import holm_adjust, paired_bootstrap, summary

cases, cfg, out = json.load(sys.stdin), compose("primary"), {}

def rows(truth, predictions):
    return [{"patient_id": i, "label": label, "prediction": prediction, "provided_tests": [],
             "requested_tests": [], "ece_triplets": []} for i, (label, prediction) in enumerate(zip(truth, predictions))]

out["contrasts"] = []
for case in cases["contrasts"]:
    a, b = rows(case["truth"], case["a"]), rows(case["truth"], case["b"])
    w, d = analysis.paired(a, b)
    boot = paired_bootstrap(a, b, cfg, seed=case["seed"], n_resamples=case["n_boot"])["metrics"]["T5"]
    out["contrasts"].append({"t5_a": summary(a, cfg)["claims"]["T5"], "t5_b": summary(b, cfg)["claims"]["T5"],
                             "delta_t5": analysis.delta_t5(w, d), "exact_p": analysis.exact_p_by_class(w, d),
                             "bootstrap_estimate": boot["estimate"], "ci95": boot["ci95"]})
out["holm"] = [{name: v["holm_p_value"] for name, v in holm_adjust(family).items()} for family in cases["holm"]]
out["power"] = [analysis.power({c: tuple(p) for c, p in case["probabilities"].items()}, case["sizes"],
                               case["sims"], case["seed"]) for case in cases["power"]]
out["mde"] = [analysis.minimum_detectable_effect(case["discordance"], case["sizes"], case["sims"], case["seed"])
              for case in cases["mde"]]
print(json.dumps(out))
"""

LIVE_CONTRASTS = [  # (seed of the synthetic pair, class sizes, discordant units, bootstrap seed, resamples)
    (11, (9, 6, 4, 5), 7, 5, 200),
    (12, (24, 17, 7, 12), 15, 269, 200),
    (13, (96, 65, 25, 54), 45, 2026, 150),
]
LIVE_HOLM = [
    {"a": 0.01, "b": 0.04, "c": 0.03},
    {"a": 0.2, "b": None, "c": 0.004, "d": 0.03},
    {"x": 0.5, "y": 0.6},
    {"one": 0.03},
]
LIVE_POWER = [
    {"probabilities": {"appendicitis": [0.08, 0.02], "cholecystitis": [0.1, 0.05], "diverticulitis": [0.2, 0.0],
                       "pancreatitis": [0.05, 0.05]}, "sizes": TEST_SIZES, "sims": 1500, "seed": 7},
    {"probabilities": {label: [0.03, 0.0] for label in CLASSES}, "sizes": TEST_SIZES, "sims": 1500, "seed": 8},
]
LIVE_MDE = [
    {"discordance": {label: 0.10 for label in CLASSES}, "sizes": TEST_SIZES, "sims": 400, "seed": 1},
    {"discordance": {"appendicitis": 0.05, "cholecystitis": 0.11, "diverticulitis": 0.12, "pancreatitis": 0.04},
     "sizes": {"appendicitis": 40, "cholecystitis": 30, "diverticulitis": 20, "pancreatitis": 25}, "sims": 400, "seed": 3},
    {"discordance": {label: 0.02 for label in CLASSES},
     "sizes": {label: 12 for label in CLASSES}, "sims": 400, "seed": 4},  # too little discordance: no MDE
]


@pytest.fixture(scope="module")
def live():
    """(inputs, the project's outputs for them), computed once by the project's interpreter."""
    if not PROJECT_PYTHON.is_file():
        pytest.skip("the project's own environment (cdm-extension/daniel-escalante/.venv, with scipy) is not on this machine")
    contrasts = []
    for pair_seed, sizes, flips, seed, n_boot in LIVE_CONTRASTS:
        a, b, truth = synthetic_pair(pair_seed, sizes, flips)
        contrasts.append({"a": a, "b": b, "truth": truth, "seed": seed, "n_boot": n_boot})
    cases = {"contrasts": contrasts, "holm": LIVE_HOLM, "power": LIVE_POWER, "mde": LIVE_MDE}
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    run = subprocess.run([str(PROJECT_PYTHON), "-B", "-c", REFERENCE_SCRIPT], input=json.dumps(cases), cwd=EXTENSION,
                         capture_output=True, text=True, timeout=300, env={**env, "PYTHONDONTWRITEBYTECODE": "1"})
    if run.returncode:
        pytest.skip("the project's code did not run under its own interpreter: " + run.stderr.strip()[-300:])
    return cases, json.loads(run.stdout)


def test_live_t5_contrast_matches_the_projects_estimate_test_and_bootstrap(live):
    cases, reference = live
    for case, expected in zip(cases["contrasts"], reference["contrasts"]):
        a, b, truth = case["a"], case["b"], case["truth"]
        result = stats.t5_contrast(a, b, truth, CLASSES, n_boot=case["n_boot"], rng=np.random.default_rng(case["seed"]))
        assert stats.t5(a, truth, CLASSES) == pytest.approx(expected["t5_a"], abs=1e-12)
        assert stats.t5(b, truth, CLASSES) == pytest.approx(expected["t5_b"], abs=1e-12)
        assert result["delta_t5"] == pytest.approx(expected["delta_t5"], abs=1e-12)
        assert result["delta_t5"] == pytest.approx(expected["bootstrap_estimate"], abs=1e-12)
        assert result["p_value"] == pytest.approx(expected["exact_p"], abs=1e-12)
        # Same generator, same draws: the interval is the replication's, not merely close to it.
        assert result["ci"] == pytest.approx(expected["ci95"], abs=1e-12)


def test_live_holm_matches_replication_holm_adjust(live):
    cases, reference = live
    for family, expected in zip(cases["holm"], reference["holm"]):
        assert stats.holm(family) == pytest.approx(expected, abs=1e-15)


def test_live_power_matches_tim_analysis_power(live):
    cases, reference = live
    for case, expected in zip(cases["power"], reference["power"]):
        probabilities = {label: tuple(p) for label, p in case["probabilities"].items()}
        assert stats.t5_power(probabilities, case["sizes"], CLASSES, case["sims"],
                              rng=np.random.default_rng(case["seed"])) == expected


def test_live_mde_matches_tim_analysis_minimum_detectable_effect(live):
    cases, reference = live
    assert reference["mde"][-1] is None
    for case, expected in zip(cases["mde"], reference["mde"]):
        found = stats.mde(case["discordance"], case["sizes"], CLASSES, case["sims"], rng=np.random.default_rng(case["seed"]))
        assert found == (None if expected is None else pytest.approx(expected, abs=1e-12))


# ---- 2. in process: the numpy-only parts of tim/analysis.py -----------------------------------

@pytest.fixture(scope="module")
def tim_analysis():
    """tim/analysis.py loaded from its file; its scipy and replication imports are lazy."""
    if not (EXTENSION / "tim" / "analysis.py").is_file():
        pytest.skip("cdm-extension/daniel-escalante/tim/analysis.py is not in this checkout")
    bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    sys.path.insert(0, str(EXTENSION))
    try:
        module = importlib.import_module("tim.analysis")
    finally:
        sys.path.remove(str(EXTENSION))
        sys.dont_write_bytecode = bytecode
        for name in [m for m in sys.modules if m == "tim" or m.startswith("tim.")]:
            del sys.modules[name]
    return module


def tim_rows(truth, predictions):
    return [{"patient_id": i, "label": label, "prediction": p} for i, (label, p) in enumerate(zip(truth, predictions))]


@pytest.mark.parametrize("seed, sizes, flips", [(21, (9, 6, 4, 5), 0), (22, (9, 6, 4, 5), 1), (23, (9, 6, 4, 5), 5),
                                                (24, (24, 17, 7, 12), 12), (25, (96, 65, 25, 54), 20)])
def test_estimate_and_exact_p_match_tims_enumerated_randomization_test(tim_analysis, seed, sizes, flips):
    a, b, truth = synthetic_pair(seed, sizes, flips)
    w, d = tim_analysis.paired(tim_rows(truth, a), tim_rows(truth, b))
    theirs = tim_analysis.randomization_test(w, d, seed=0)
    assert theirs["method"] == "exact" and theirs["discordant"] == flips
    ours = stats.t5_contrast(a, b, truth, CLASSES, n_boot=2, rng=np.random.default_rng(0))
    assert ours["delta_t5"] == pytest.approx(tim_analysis.delta_t5(w, d), abs=1e-12)
    assert ours["discordant"] == flips
    assert ours["p_value"] == pytest.approx(theirs["p_value"], abs=1e-12)


def test_exact_p_is_what_tims_drawn_estimate_estimates(tim_analysis):
    """Above 20 discordant units the project estimates the same p-value from 100,000 draws."""
    a, b, truth = synthetic_pair(26, (96, 65, 25, 54), 40, right=0.5)  # improvements and regressions in balance
    w, d = tim_analysis.paired(tim_rows(truth, a), tim_rows(truth, b))
    theirs = tim_analysis.randomization_test(w, d, seed=11)
    assert theirs["method"] == "draws" and theirs["draws"] == 100_000
    ours = stats.t5_contrast(a, b, truth, CLASSES, n_boot=2, rng=np.random.default_rng(0))["p_value"]
    standard_error = np.sqrt(ours * (1 - ours) / 100_000)
    assert 0.01 < ours < 0.99  # a p-value in the middle of the range, where the comparison is informative
    assert abs(theirs["p_value"] - ours) < 4 * standard_error + 1e-5


# ---- 3. recorded: numbers in the committed bundles ---------------------------------------------

# The recorded intervals used the private row order and the project's seed, so only Monte Carlo
# agreement is possible: with 10,000 resamples an interval end has a standard deviation of about
# 0.06 points here, and two independent runs differ by about 0.09. 0.4 is beyond four of those.
INTERVAL_TOLERANCE = 0.4


def stream_seed(seed, split, stage):
    """replication.config.stream_seed, restated."""
    return int.from_bytes(hashlib.sha256(f"{seed}\0{split}\0{stage}".encode()).digest()[:4], "big")


@pytest.mark.parametrize("name", ["V1-V0", "V2-V0", "V3-V0"])
def test_recorded_validation_contrasts(name):
    validation = bundle("analyze-val-a", "validation.json")
    recorded, changes = validation["descriptive"][name], validation["changes"][name]
    a, b, truth = pair_from_changes(changes)
    result = stats.t5_contrast(a, b, truth, CLASSES, rng=np.random.default_rng(0))
    assert result["delta_t5"] == pytest.approx(recorded["delta_t5"], abs=1e-12)
    assert result["discordant"] == recorded["discordant"]
    assert result["changes"] == changes
    assert result["ci"] == pytest.approx(recorded["ci95"], abs=INTERVAL_TOLERANCE)
    # Every discordant admission improved, so only the two one-signed assignments are as extreme:
    # p = 2 / 2^k, which is also the recorded exact McNemar p-value.
    assert all(changes[label]["regressed"] == 0 for label in CLASSES)
    assert result["p_value"] == pytest.approx(recorded["mcnemar_p"], rel=1e-12)


def test_recorded_v1_minus_official_on_test():
    metrics = bundle("phase-a", "metrics.json")
    recorded = metrics["family_a"]["V1-official"]
    v1, official = metrics["variants"]["V1"], metrics["variants"]["official"]
    sizes = class_sizes(official)
    assert sizes == TEST_SIZES
    net = {label: correct_counts(v1)[label] - correct_counts(official)[label] for label in CLASSES}
    # The net gain equals the recorded number of discordant admissions, so none regressed.
    assert sum(net.values()) == recorded["discordant"] and min(net.values()) >= 0
    changes = {label: {"improved": net[label], "regressed": 0, "unchanged": sizes[label] - net[label]} for label in CLASSES}
    a, b, truth = pair_from_changes(changes)
    result = stats.t5_contrast(a, b, truth, CLASSES, rng=np.random.default_rng(0))
    assert result["delta_t5"] == pytest.approx(recorded["delta_t5"], abs=1e-12)
    assert result["ci"] == pytest.approx(recorded["ci95"], abs=INTERVAL_TOLERANCE)
    assert result["p_value"] == pytest.approx(recorded["mcnemar_p"], rel=1e-12)
    assert result["p_value"] == pytest.approx(2.0 ** -24, rel=1e-12)
    # The project estimated this p-value from 100,000 draws, none of them as extreme: 1 / 100,001.
    assert recorded["method"] == "draws" and recorded["p_value"] == pytest.approx(1 / 100_001)
    assert result["p_value"] < recorded["p_value"]


@pytest.mark.parametrize("name", ["V1-V0", "V3-V0"])
def test_recorded_sensitivity_block_is_reproduced_digit_for_digit(name):
    """Power at the validation effect and the 80% MDE, as addendum A's simulation recorded them:
    10,000 simulated tests at the test class sizes, seed stream_seed(269, "val", "tim-sensitivity")."""
    metrics, validation = bundle("phase-a", "metrics.json"), bundle("analyze-val-a", "validation.json")
    recorded, changes = metrics["sensitivity"][name], validation["changes"][name]
    sizes = class_sizes(metrics["variants"]["official"])
    probabilities = {label: (changes[label]["improved"] / sum(changes[label].values()),
                             changes[label]["regressed"] / sum(changes[label].values())) for label in CLASSES}
    discordance = {label: sum(probabilities[label]) for label in CLASSES}
    assert discordance == recorded["discordance"]
    seed = stream_seed(269, "val", "tim-sensitivity")
    assert stats.t5_power(probabilities, sizes, CLASSES, 10_000, rng=np.random.default_rng(seed)) == recorded["power_at_val_effect"]
    assert stats.mde(discordance, sizes, CLASSES, 10_000, rng=np.random.default_rng(seed)) == pytest.approx(recorded["mde_80"], abs=1e-12)


@pytest.mark.parametrize("discordance, expected", [(0.10, 6.4), (0.20, 9.1)])
def test_mde_reproduces_the_orientation_figures(discordance, expected):
    """DECISIONS.md and SPEC section 5.7 of the extension: about 6.4 and 9.1 points at an equal
    10% and 20% discordance, at the test class sizes; the settings and tolerance of the
    project's own test (tests/test_analysis.py)."""
    found = stats.mde({label: discordance for label in CLASSES}, TEST_SIZES, CLASSES, 4000, rng=np.random.default_rng(1))
    assert found == pytest.approx(expected, abs=0.35)


def configurations(net, discordant, correct_b, sizes):
    """Every per-class (improved, regressed) split consistent with the committed aggregates:
    the class's net change in correct answers and the total number of discordant admissions."""
    spare = discordant - sum(abs(n) for n in net.values())
    assert spare >= 0 and spare % 2 == 0
    for extra in itertools.product(range(spare // 2 + 1), repeat=len(CLASSES)):
        if sum(extra) != spare // 2:
            continue
        changes = {}
        for label, e in zip(CLASSES, extra):
            improved, regressed = max(net[label], 0) + e, max(-net[label], 0) + e
            if improved > sizes[label] - correct_b[label] or regressed > correct_b[label]:
                break
            changes[label] = {"improved": improved, "regressed": regressed, "unchanged": sizes[label] - improved - regressed}
        else:
            yield changes


@pytest.mark.parametrize("phase, family, name, arm_a, arm_b", [
    ("phase-a", "family_a", "V3-V1", "V3", "V1"),
    ("phase-b", "family_b", "B-learned-s0-B-marginal", "B-learned-s0", "B-marginal"),
    ("phase-b", "family_b", "B-learned-s0-B-uniform", "B-learned-s0", "B-uniform"),
    ("phase-b", "family_b", "B-oracle-B-uniform", "B-oracle", "B-uniform"),
])
def test_recorded_exact_p_values(phase, family, name, arm_a, arm_b):
    """The bundles record which class each arm got right, not which admissions moved, so the
    committed aggregates fix the per-class changes only up to a handful of configurations. On
    every one of them the p-value equals brute-force enumeration, and the recorded p-value is
    the p-value of one of them."""
    metrics = bundle(phase, "metrics.json")
    recorded, variant_a, variant_b = metrics[family][name], metrics["variants"][arm_a], metrics["variants"][arm_b]
    assert recorded["method"] == "exact"
    sizes, correct_b = class_sizes(variant_b), correct_counts(variant_b)
    net = {label: correct_counts(variant_a)[label] - correct_b[label] for label in CLASSES}
    found = []
    for changes in configurations(net, recorded["discordant"], correct_b, sizes):
        a, b, truth = pair_from_changes(changes)
        result = stats.t5_contrast(a, b, truth, CLASSES, n_boot=2, rng=np.random.default_rng(0))
        assert result["delta_t5"] == pytest.approx(recorded["delta_t5"], abs=1e-12)
        assert result["p_value"] == pytest.approx(brute_force_p(a, b, truth), abs=1e-12)
        found.append(result["p_value"])
    assert found and any(p == pytest.approx(recorded["p_value"], abs=1e-12) for p in found)


@pytest.mark.parametrize("phase, family", [("phase-a", "family_a"), ("phase-b", "family_b")])
def test_recorded_holm_families(phase, family):
    members = bundle(phase, "metrics.json")[family]
    adjusted = stats.holm({name: member["p_value"] for name, member in members.items()})
    assert adjusted == pytest.approx({name: member["p_holm"] for name, member in members.items()}, abs=1e-15)
