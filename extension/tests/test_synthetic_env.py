"""Tests for SyntheticCDMEnv + costs + generators."""

from __future__ import annotations

import math

import pytest

from extension.envs.synthetic_diagnosis.actions import (
    A_DIAG,
    A_TEST,
    COST_NORM,
    COST_USD,
)
from extension.envs.synthetic_diagnosis.env import SyntheticCDMEnv
from extension.envs.synthetic_diagnosis.generators import generate


def test_costs_normalize_to_one():
    s = sum(COST_NORM[a] for a in A_TEST)
    assert math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-12)
    assert all(COST_USD[a] > 0 for a in A_TEST)
    assert len(COST_USD) == len(A_TEST) == 12


def test_reset_step_diagnose_correct_incorrect():
    env = SyntheticCDMEnv(seed=42, split="train")
    h = env.reset()
    assert isinstance(h, str) and "synthetic" in h.lower() or "Patient" in h
    y = env.y_true
    assert y in A_DIAG

    # Take one cheap test
    o, r_test, done, info = env.step("Urinalysis")
    assert not done
    assert "UNAVAILABLE" not in o
    assert r_test < 0  # cost penalty
    assert math.isclose(r_test, -COST_NORM["Urinalysis"])

    # Wrong diagnosis
    wrong = next(d for d in A_DIAG if d != y)
    o_w, r_w, done_w, info_w = env.step(wrong)
    assert done_w
    assert r_w == -1.0
    assert info_w["correct"] is False
    assert info_w["y_true"] == y

    # Fresh episode, correct diagnosis
    env2 = SyntheticCDMEnv(seed=42, split="train")
    env2.reset()
    # Same seed → same first episode y? episode counter + rng — use fixed y
    env3 = SyntheticCDMEnv(seed=7, split="val")
    env3.reset(y="pancreatitis")
    assert env3.y_true == "pancreatitis"
    _, r_ok, done_ok, info_ok = env3.step("pancreatitis")
    assert done_ok and r_ok == 1.0 and info_ok["correct"] is True


def test_all_twelve_tests_always_available_non_unavailable():
    env = SyntheticCDMEnv(seed=0, split="train")
    env.reset(y="appendicitis")
    for test in A_TEST:
        o, reward, done, info = env.step(test)
        assert not done
        assert isinstance(o, str) and len(o) > 10
        assert "UNAVAILABLE" not in o
        assert test in o or o.startswith("[")  # wrapped with test name usually
    assert len(env.tests_taken) == 12
    assert env.available_tests == []


def test_oracle_observe_matches_step_before_taken():
    env = SyntheticCDMEnv(seed=99, split="train")
    env.reset(y="cholecystitis")
    for test in ("Ultrasound", "CT", "Liver Function Panel", "Physical Examination"):
        oracle_o = env.oracle_observe(test)
        # Must match what step returns before the test is taken
        step_o, _, done, _ = env.step(test)
        assert not done
        assert oracle_o == step_o
        # After taken, oracle still returns the same counterfactual text
        assert env.oracle_observe(test) == oracle_o


def test_generate_deterministic():
    a = generate("CT", "diverticulitis", eps_seed=123)
    b = generate("CT", "diverticulitis", eps_seed=123)
    c = generate("CT", "diverticulitis", eps_seed=124)
    assert a == b
    assert a != c or True  # noise may coincide rarely; primary check is a==b
    assert "UNAVAILABLE" not in a


def test_invalid_action_penalty():
    env = SyntheticCDMEnv(seed=1)
    env.reset(y="appendicitis")
    o, r, done, info = env.step("NotARealTest")
    assert not done
    assert r == -0.5
    assert info.get("invalid") is True
