"""Smoke tests for the experiment runners and metrics."""

from __future__ import annotations

import json

import pytest

from extension.experiments import run_matrix, sanity
from extension.experiments.metrics import aggregate, episode_metrics
from extension.planners.agents import EpisodeResult


def _r(y, pred, cost=0.0):
    return EpisodeResult(y, pred, cost, cost / 10190.0, 1, 1, ("CT",))


def test_metrics_hand_computed():
    res = [
        _r("appendicitis", "appendicitis", 100),
        _r("appendicitis", "cholecystitis", 100),
        _r("cholecystitis", "cholecystitis", 100),
        _r("pancreatitis", None, 100),
    ]
    m = episode_metrics(res, lam=0.0)
    assert m["acc_appendicitis"] == 0.5
    assert m["acc_cholecystitis"] == 1.0
    assert m["acc_pancreatitis"] == 0.0
    assert m["acc_diverticulitis"] == 0.0  # no support
    assert m["overall_acc"] == 0.5
    # micro: tp=2, fp=1 (wrong dx), fn=2 (wrong + undiagnosed)
    assert m["micro_f1"] == pytest.approx(4 / 7)
    assert m["undiagnosed_rate"] == 0.25
    assert m["avg_cost_usd"] == 100
    agg = aggregate([m, m])
    assert agg["overall_acc"] == {"mean": 0.5, "std": 0.0}


def test_run_matrix_smoke(tmp_path):
    rc = run_matrix.main([
        "--out", str(tmp_path), "--seeds", "0", "1", "--n-train", "100",
        "--n-eval", "40", "--alphas", "0", "1", "--n-train-sweep", "20", "100",
        "--lam-sweep", "0", "1",
    ])
    assert rc == 0
    report = json.loads((tmp_path / "matrix.json").read_text())
    for setting in ("always_available", "missingness"):
        s = report["settings"][setting]
        assert set(s["main"]) == {"Oracle", "WM", "MF", "Random", "Bad-WM(alpha=1)"}
        assert s["main"]["Oracle"]["regret_vs_oracle"]["mean"] == 0.0
        assert set(s["bad_wm_alpha"]) == {"0", "1"}
        assert "WM/20" in s["data_efficiency"] and "MF/100" in s["data_efficiency"]
        assert "Oracle/0" in s["lambda_sweep"]


def test_sanity_smoke(tmp_path):
    assert sanity.main(["--out", str(tmp_path), "--n", "200"]) == 0
    out = json.loads((tmp_path / "sanity.json").read_text())
    assert len(out["single_test"]) == 12
    assert out["all_tests_always_available"]["overall_acc"] > 0.8
