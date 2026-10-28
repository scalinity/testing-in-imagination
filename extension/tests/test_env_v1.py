"""Tests for env v1: findings, phenotypes, missingness, step budget."""

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter

import pytest

from extension.envs.synthetic_diagnosis.actions import A_DIAG, A_TEST
from extension.envs.synthetic_diagnosis.collect import collect
from extension.envs.synthetic_diagnosis.env import SyntheticCDMEnv
from extension.envs.synthetic_diagnosis.profiles import (
    AVAILABILITY,
    FINDINGS,
    LIKELIHOOD,
    PRIOR_MIMIC_CDM,
    expected_all_available_cost_usd,
    phenotype_given_y,
)
from extension.envs.synthetic_diagnosis.tuples import read_jsonl, validate_record_dict


def test_profiles_are_distributions():
    for test in A_TEST:
        for e in A_DIAG:
            row = LIKELIHOOD[test][e]
            assert set(row) == set(FINDINGS)
            assert all(p >= 0 for p in row.values())
            assert math.isclose(sum(row.values()), 1.0)
    assert math.isclose(sum(PRIOR_MIMIC_CDM.values()), 1.0)
    for y, row in phenotype_given_y(0.07).items():
        assert math.isclose(sum(row.values()), 1.0)
        assert math.isclose(row[y], 0.93)


def test_all_available_cost_calibrated_to_paper_sft_all():
    # Paper Table 1 SFT-all avg test cost: $3792.79.
    assert abs(expected_all_available_cost_usd() - 3792.79) < 100
    assert AVAILABILITY["Physical Examination"] == 1.0


def test_v0_default_unchanged():
    env = SyntheticCDMEnv(seed=0)
    assert env.version == "v0" and env.max_steps is None
    with pytest.raises(ValueError):
        SyntheticCDMEnv(seed=0, missingness=True)


def test_v1_step_reports_finding_and_oracle_matches():
    env = SyntheticCDMEnv(seed=5, version="v1")
    env.reset(y="cholecystitis")
    for test in ("Ultrasound", "CT", "Radiograph"):
        f = env.oracle_finding(test)
        text = env.oracle_observe(test)
        o, _, done, info = env.step(test)
        assert not done
        assert info["finding"] == f in FINDINGS
        assert o == text
        assert "UNAVAILABLE" not in o


def test_v1_finding_frequencies_follow_likelihood():
    env = SyntheticCDMEnv(seed=11, version="v1", p_atypical=0.0)
    counts = Counter()
    n = 3000
    for _ in range(n):
        env.reset(y="appendicitis")
        counts[env.oracle_finding("CT")] += 1
    expected = LIKELIHOOD["CT"]["appendicitis"]["appendicitis"]
    assert abs(counts["appendicitis"] / n - expected) < 0.03


def test_v1_prior_matches_mimic_cdm():
    env = SyntheticCDMEnv(seed=2, version="v1")
    counts = Counter()
    n = 4000
    for _ in range(n):
        env.reset()
        counts[env.y_true] += 1
    for y, p in PRIOR_MIMIC_CDM.items():
        assert abs(counts[y] / n - p) < 0.03


def test_v1_patients_independent_of_fixed_y_call():
    # Same seed → same patient stream; the agent's actions do not change it.
    a = SyntheticCDMEnv(seed=9, version="v1", missingness=True)
    b = SyntheticCDMEnv(seed=9, version="v1", missingness=True)
    for _ in range(20):
        a.reset()
        b.reset()
        a.step("CT")
        assert (a.y_true, a.phenotype, a._in_chart) == (b.y_true, b.phenotype, b._in_chart)


def test_missing_test_is_free_burns_step_and_has_no_finding():
    env = SyntheticCDMEnv(seed=0, version="v1", missingness=True)
    for _ in range(200):
        env.reset()
        missing = [a for a in A_TEST if a not in env._in_chart]
        if missing:
            break
    test = missing[0]
    o, r, done, info = env.step(test)
    assert "UNAVAILABLE" in o and r == 0.0 and not done
    assert info["unavailable"] is True and info["finding"] is None
    assert env.t == 1 and env.tests_taken == []
    assert test not in env.available_tests
    # The counterfactual still exists (testing in imagination).
    assert env.oracle_finding(test) in FINDINGS


def test_missingness_rate_matches_availability():
    env = SyntheticCDMEnv(seed=3, version="v1", missingness=True)
    counts = Counter()
    n = 3000
    for _ in range(n):
        env.reset()
        counts.update(env._in_chart)
    for a in A_TEST:
        assert abs(counts[a] / n - AVAILABILITY[a]) < 0.03


def test_v1_budget_ends_undiagnosed():
    env = SyntheticCDMEnv(seed=1, version="v1", max_steps=3)
    env.reset()
    env.step("CT")
    env.step("MRI")
    _, r, done, info = env.step("Urinalysis")
    assert done and info["undiagnosed"] and info["correct"] is False
    assert r < 0


def test_collect_v1_with_missingness_rows_validate():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.jsonl")
        n = collect(
            n_episodes=100,
            seed=0,
            out_path=path,
            env_kwargs={"version": "v1", "missingness": True},
        )
        rows = read_jsonl(path)
        assert len(rows) == n
        unavailable = [r for r in rows if r["meta"]["unavailable"]]
        assert unavailable, "expected some UNAVAILABLE rows"
        for r in rows:
            validate_record_dict(r)
            if r["meta"]["unavailable"]:
                assert r["meta"]["finding"] is None
                assert "UNAVAILABLE" in r["o_t1"]
            else:
                assert r["meta"]["finding"] in FINDINGS
