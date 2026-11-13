"""Random streams separated by purpose and split (REVIEW-002), reproducible cases, matched arms."""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pytest

from tiicee.interfaces import UNAVAILABLE
from tiicee.twin import model as tm
from tiicee.twin.env import PURPOSES, SPLITS, Twin, stream


@pytest.fixture(scope="module")
def twin():
    return Twin(tm.build("base", reduced=True))


def fingerprints(patients):
    return [p.fingerprint for p in patients]


def test_declared_purposes_and_splits():
    assert PURPOSES == ("generation", "behavior", "fitting", "corruption", "montecarlo")
    assert SPLITS == ("train", "validation", "test")


def test_every_purpose_and_split_has_its_own_stream():
    draws = {(p, s): stream(7, p, s).random(4).tolist() for p in PURPOSES for s in SPLITS}
    assert len({tuple(v) for v in draws.values()}) == len(PURPOSES) * len(SPLITS)
    assert stream(7, "generation", "test").random(4).tolist() == draws[("generation", "test")]
    assert stream(8, "generation", "test").random(4).tolist() != draws[("generation", "test")]
    assert stream(7, "generation", "test", 3).random(4).tolist() != stream(7, "generation", "test", 4).random(4).tolist()


def test_unknown_purpose_or_split_is_refused():
    with pytest.raises(ValueError):
        stream(0, "evaluation", "test")
    with pytest.raises(ValueError):
        stream(0, "generation", "val")


def test_train_validation_and_test_cases_differ_under_one_user_seed(twin):
    by_split = {split: fingerprints(twin.sample_patients(200, seed=0, split=split)) for split in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                same_position = sum(x == y for x, y in zip(by_split[a], by_split[b]))
                # equal content can recur by chance in a small discrete space; a replayed stream would match everywhere
                assert same_position < 20, (a, b, same_position)
                assert by_split[a] != by_split[b]


def test_the_same_split_and_seed_reproduce_identical_cases(twin):
    first = twin.sample_patients(60, seed=3, split="validation")
    again = twin.sample_patients(60, seed=3, split="validation")
    assert first == again and fingerprints(first) == fingerprints(again)
    # case i does not depend on how many cases are drawn
    assert twin.sample_patients(10, seed=3, split="validation") == first[:10]
    assert fingerprints(twin.sample_patients(60, seed=4, split="validation")) != fingerprints(first)


def test_fitting_cases_are_not_the_generation_cases(twin):
    generation = fingerprints(twin.sample_patients(100, seed=0, split="train"))
    fitting = fingerprints(twin.sample_patients(100, seed=0, split="train", purpose="fitting"))
    assert sum(x == y for x, y in zip(generation, fitting)) < 15


def test_arms_receive_identical_latent_cases():
    arm_a = Twin(tm.build("base", reduced=True))
    arm_b = Twin(tm.build("base", reduced=True))
    cases_a = arm_a.sample_patients(50, seed=11, split="test")
    # arm B uses the behavior and Monte Carlo streams before it asks for its cases
    stream(11, "behavior", "test").random(1000)
    stream(11, "montecarlo", "test").random(1000)
    cases_b = arm_b.sample_patients(50, seed=11, split="test")
    assert fingerprints(cases_a) == fingerprints(cases_b)
    assert [p.display_id for p in cases_a] == [p.display_id for p in cases_b]
    assert len(set(p.display_id for p in cases_a)) == 50


def test_variants_that_share_items_share_their_cases(twin):
    """Case i is drawn item by item, so the duplicated panel sees the base case with one result copied."""
    duplicated = Twin(tm.build("duplicated_panel", reduced=True))
    for base_case, dup_case in zip(twin.sample_patients(40, seed=6, split="test"), duplicated.sample_patients(40, seed=6, split="test")):
        assert (dup_case.diagnosis, dup_case.phenotype) == (base_case.diagnosis, base_case.phenotype)
        assert dup_case.findings == base_case.findings + (base_case.findings[4],)
        assert dup_case.available == base_case.available + (base_case.available[4],)
        assert dup_case.fingerprint != base_case.fingerprint          # one more result is different content


def test_generated_cases_follow_the_model(twin):
    model = twin.model
    patients = twin.sample_patients(8000, seed=1, split="train")
    n = len(patients)
    y = np.array([model.classes.index(p.diagnosis) for p in patients])
    e = np.array([model.phenotypes.index(p.phenotype) for p in patients])
    for c in range(4):
        p = model.prior[c]
        assert abs((y == c).mean() - p) < 4 * np.sqrt(p * (1 - p) / n)
    assert abs((y != e).mean() - 0.07) < 4 * np.sqrt(0.07 * 0.93 / n)
    typical = e == 0
    for t, test in enumerate(model.tests):
        available = np.array([p.available[t] for p in patients])
        p = model.availability[t, 0]
        assert abs(available.mean() - p) <= 4 * np.sqrt(p * (1 - p) / n)
        on_target = np.array([p.findings[t] == "appendicitis" for p in patients])[typical]
        hit = model.finding_lik[t, 0, 0, 0]
        assert abs(on_target.mean() - hit) < 4 * np.sqrt(hit * (1 - hit) / typical.sum())
        if 0 < p < 1:      # findings are drawn whether or not the test is in the chart, and do not depend on it
            charted = available[typical]
            spread = np.sqrt(hit * (1 - hit) * (1 / charted.sum() + 1 / (~charted).sum()))
            assert abs(on_target[charted].mean() - on_target[~charted].mean()) < 4 * spread
    assert all(twin.observe(p, "Physical Examination") != UNAVAILABLE for p in patients[:200])


@pytest.mark.parametrize("variant", ["base", "strong_correlation", "no_correlation", "duplicated_panel", "informative_missingness"])
def test_generated_cases_have_the_joint_distribution_of_the_model(variant):
    """Diagnosis and the observed results of three tests, cell by cell, against the enumerated joint."""
    from collections import Counter

    from test_twin_support import Joint

    model = tm.build(variant, reduced=True, allow_held_out=True)
    sampler, joint = Twin(model), Joint(model)
    tests = ("CT", "Ultrasound", model.tests[-1])        # the last test is the repeat in the duplicated variant
    n = 8000
    patients = sampler.sample_patients(n, seed=17, split="validation")
    seen = Counter((p.diagnosis,) + tuple(sampler.observe(p, t) for t in tests) for p in patients)
    checked = 0
    for y, name in enumerate(model.classes):
        for outcomes in itertools.product(model.outcomes, repeat=3):
            expected = joint.mass(tuple(zip(tests, outcomes))).sum(1)[y]
            if expected * n >= 40:
                checked += 1
                spread = np.sqrt(expected * (1 - expected) / n)
                assert abs(seen[(name,) + outcomes] / n - expected) < 4.5 * spread, (name, outcomes)
            elif expected == 0:
                assert seen[(name,) + outcomes] == 0, (name, outcomes)
    assert checked >= 20


def test_first_test_case_is_pinned(twin):
    """A change to the stream layout, the sampling order, or the fingerprint format would change every experiment."""
    patient = twin.sample_patients(1, seed=0, split="test")[0]
    assert (patient.diagnosis, patient.phenotype) == ("cholecystitis", "cholecystitis")
    assert patient.findings == ("cholecystitis", "cholecystitis", "cholecystitis", "nonspecific", "cholecystitis", "nonspecific")
    assert patient.available == (True, True, False, True, True, False)
    # the fingerprint is the SHA-256 of this exact text: sorted keys, no spaces, no display id
    content = (
        '{"diagnosis":"cholecystitis","phenotype":"cholecystitis","results":['
        '["Physical Examination","cholecystitis",true],["CT","cholecystitis",true],["Ultrasound","cholecystitis",false],'
        '["Complete Blood Count","nonspecific",true],["Liver Function Panel","cholecystitis",true],["Urinalysis","nonspecific",false]]}'
    )
    assert patient.fingerprint == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert patient.fingerprint == "a11d452313044defe81b4e6e4586bbc0ae395fe8d074aa952278116582eb76c2"
