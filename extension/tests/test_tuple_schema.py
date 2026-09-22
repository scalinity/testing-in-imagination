"""Tests for TupleRecord schema and JSONL collection."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from extension.envs.synthetic_diagnosis.actions import A_TEST
from extension.envs.synthetic_diagnosis.collect import collect, rollout_episode
from extension.envs.synthetic_diagnosis.env import SyntheticCDMEnv
from extension.envs.synthetic_diagnosis.tuples import (
    REQUIRED_KEYS,
    META_REQUIRED_KEYS,
    JSONLWriter,
    TupleRecord,
    read_jsonl,
    validate_record_dict,
)
import random


def test_tuple_record_required_keys_and_source():
    rec = TupleRecord(
        episode_id="syn-train-s0-e0001-p1",
        t=0,
        h_t="Patient encounter ...",
        a_t="CT",
        o_t1="[CT] dilated appendix ...",
        y_true="appendicitis",
        source="synthetic",
        meta={
            "split": "train",
            "available_tests_at_t": list(A_TEST),
            "seed": 0,
        },
    )
    rec.validate()
    d = rec.to_dict()
    for k in REQUIRED_KEYS:
        assert k in d
    assert d["source"] == "synthetic"
    for k in META_REQUIRED_KEYS:
        assert k in d["meta"]


def test_jsonl_writer_and_collect_produces_valid_rows():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tuples.jsonl")
        n = collect(
            n_episodes=5,
            seed=0,
            out_path=path,
            policy="random",
            split="train",
        )
        assert n >= 1
        rows = read_jsonl(path)
        assert len(rows) == n
        assert len(rows) >= 1
        for row in rows:
            validate_record_dict(row)
            assert row["source"] == "synthetic"
            assert row["a_t"] in A_TEST
            assert "UNAVAILABLE" not in row["o_t1"]
            assert row["y_true"] in (
                "appendicitis",
                "cholecystitis",
                "diverticulitis",
                "pancreatitis",
            )
            assert isinstance(row["h_t"], str) and len(row["h_t"]) > 0


def test_only_test_actions_emitted():
    env = SyntheticCDMEnv(seed=3, split="val")
    rng = random.Random(3)
    recs = rollout_episode(env, "expert", rng)
    assert len(recs) >= 1
    for r in recs:
        assert r.a_t in A_TEST
        r.validate()


def test_reject_bad_source():
    rec = TupleRecord(
        episode_id="x",
        t=0,
        h_t="h",
        a_t="CT",
        o_t1="o",
        y_true="appendicitis",
        source="mimic_real",
        meta={"split": "train", "available_tests_at_t": [], "seed": 0},
    )
    with pytest.raises(ValueError):
        rec.validate()
