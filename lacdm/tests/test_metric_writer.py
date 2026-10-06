"""Phase E item 19: eval metrics on a tiny labelled stub, written as JSON.

Expected values are worked by hand below, independently of
src/utils/metrics.py, so the test checks upstream's arithmetic rather than
restating it.
"""

import json
import subprocess

import pytest

from lacdm.metric_writer import ARTIFACTS_DIR, LACDM_DIR, write_metrics
from src.utils.metrics import calculate_classification_metrics

PREDICTIONS = ["appendicitis", "appendicitis", "diverticulitis", None, "pancreatitis"]
LABELS = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis", "pancreatitis"]
ECE_TRIPLETS = [  # (hypothesis, confidence in [0, 1], label)
    ("appendicitis", 0.9, "appendicitis"),
    ("appendicitis", 0.9, "cholecystitis"),
    ("pancreatitis", 0.3, "pancreatitis"),
    ("unknown", 0.5, "diverticulitis"),  # counted in hypothesis accuracy, dropped from ECE
]

# all/ (None counts as wrong): TP=3, FP=1 (appendicitis for cholecystitis),
#   FN=2 (cholecystitis missed; pancreatitis -> None).
#   Per-class F1: app 2/3, chole 0, div 1, panc 2/3.
# ignore/ (None dropped, 4 pairs): TP=3, FP=1, FN=1. Per-class F1: app 2/3, chole 0, div 1, panc 1.
# ECE: bin [.85,.95) holds 2 at conf .9, acc .5 -> |.9-.5| * 2/3 = 4/15
#      bin [.25,.35) holds 1 at conf .3, acc 1 -> |.3-1| * 1/3 = 7/30  => 1/2
EXPECTED = {
    "none_fraction": 1 / 5,
    "all/accuracy": 3 / 5,
    "all/micro_f1": 6 / 9,
    "all/macro_f1": (2 / 3 + 0 + 1 + 2 / 3) / 4,
    "ignore/accuracy": 3 / 4,
    "ignore/micro_f1": 6 / 8,
    "ignore/macro_f1": (2 / 3 + 0 + 1 + 1) / 4,
    "ece": 1 / 2,
    "hypothesis_accuracy": 2 / 4,
    "unknown_hypothesis_fraction": 1 / 4,
}


@pytest.fixture
def stub_metrics():
    return calculate_classification_metrics(PREDICTIONS, LABELS, ECE_TRIPLETS)


@pytest.mark.parametrize("key", EXPECTED)
def test_upstream_metrics_match_hand_computation(stub_metrics, key):
    assert stub_metrics[key] == pytest.approx(EXPECTED[key])


def test_micro_f1_is_not_accuracy_when_predictions_are_missing(stub_metrics):
    # A None prediction is a false negative but not a false positive.
    assert stub_metrics["all/micro_f1"] != pytest.approx(stub_metrics["all/accuracy"])


def test_writer_round_trips_with_provenance(stub_metrics, tmp_path):
    path = write_metrics("stub", stub_metrics, source="fixture", out_dir=tmp_path)
    payload = json.loads(path.read_text())
    assert payload["source"] == "fixture"
    assert payload["upstream_sha"] == subprocess.run(
        ["git", "-C", str(LACDM_DIR / "upstream"), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert payload["metrics"] == pytest.approx(stub_metrics)


def test_writer_rejects_unlabelled_source(stub_metrics, tmp_path):
    with pytest.raises(ValueError):
        write_metrics("stub", stub_metrics, source="paper", out_dir=tmp_path)


def test_default_output_lands_in_gitignored_artifacts(stub_metrics):
    path = write_metrics("phase_e_stub_metrics", stub_metrics, source="fixture")
    assert path.parent == ARTIFACTS_DIR
    assert json.loads(path.read_text())["metrics"]["ece"] == pytest.approx(0.5)
    ignored = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=LACDM_DIR)
    assert ignored.returncode == 0, "lacdm/artifacts output must never be committed"
