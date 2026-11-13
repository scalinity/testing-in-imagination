"""The counterfactual interface: exact quantities against enumeration, the cost rule, identity, rendering."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tiicee.interfaces import UNAVAILABLE, Predictive
from tiicee.twin import model as tm
from tiicee.twin.env import TEMPLATES, Twin

from test_twin_support import Joint, sample_histories, toy_model

ALL_VARIANTS = tm.variants(allow_held_out=True)

# results/synthetic/sanity.json at df467ea: information gain of each test alone, no missingness.
PUBLISHED_INFORMATION = {
    "MRI": 0.7232772024668871, "CT": 0.693586580350203, "Ultrasound": 0.4871027923460073,
    "Liver Function Panel": 0.223781565483554, "Physical Examination": 0.21978133112707932,
    "Comprehensive Metabolic Panel": 0.07711016180960506, "Basic Metabolic Panel": 0.03835402100097385,
    "Complete Blood Count": 0.036453666896486325, "Renal Function Panel": 0.02556305024026506,
    "Electrolyte Panel": 0.01943990786728733, "Urinalysis": 0.013529109914602655,
    "Radiograph": 0.012567702027495908,
}


@pytest.fixture(scope="module", params=ALL_VARIANTS)
def case(request):
    model = tm.build(request.param, reduced=True, allow_held_out=True)
    twin = Twin(model)
    return twin, Joint(model), sample_histories(twin)


def as_dict(predictive: Predictive) -> dict:
    return dict(zip(predictive.outcomes, predictive.weights))


def test_true_posterior_matches_enumeration(case):
    twin, joint, histories = case
    for history in histories:
        np.testing.assert_allclose(twin.true_posterior(history), joint.posterior(history), rtol=0, atol=1e-12)


def test_true_predictive_matches_enumeration(case):
    twin, joint, histories = case
    for history in histories:
        for test in twin.model.tests:
            predictive = twin.true_predictive(history, test)
            reference = {o: p for o, p in joint.predictive(history, test).items() if p > 0}
            assert isinstance(predictive, Predictive) and predictive.merged
            assert set(predictive.outcomes) == set(reference)
            assert (predictive.weights > 0).all() and predictive.weights.sum() == pytest.approx(1.0, abs=1e-12)
            for outcome, weight in as_dict(predictive).items():
                assert weight == pytest.approx(reference[outcome], abs=1e-12)


def test_true_information_matches_enumeration(case):
    twin, joint, histories = case
    for history in histories[:12]:
        for test in twin.model.tests:
            assert twin.true_information(history, test) == pytest.approx(joint.information(history, test), abs=1e-12)
            assert twin.true_information(history, test) >= 0


def test_expected_charged_cost_matches_enumeration(case):
    twin, joint, histories = case
    for history in histories:
        for t, test in enumerate(twin.model.tests):
            predictive = joint.predictive(history, test)
            # enumerate the outcomes: an unavailable result is free, any finding is charged in full
            expected = sum(p * (0.0 if o == UNAVAILABLE else twin.model.cost_norm[t]) for o, p in predictive.items())
            assert twin.expected_charged_cost(history, test) == pytest.approx(expected, abs=1e-12)
            assert twin.p_available(history, test) == pytest.approx(1 - predictive[UNAVAILABLE], abs=1e-12)
            assert twin.expected_charged_cost(history, test, usd=True) == pytest.approx(
                (1 - predictive[UNAVAILABLE]) * twin.model.cost_usd[t], abs=1e-9)


def test_expected_cost_rule_at_zero_one_and_intermediate_availability():
    model = toy_model(availability=(1.0, 0.3, 0.0))
    twin, joint = Twin(model), Joint(model)
    for history in ((), (("t0", "pos"),), (("t0", "neg"), ("t1", UNAVAILABLE))):
        for t, (test, p) in enumerate(zip(model.tests, (1.0, 0.3, 0.0))):
            if any(test == requested for requested, _ in history):
                continue
            charged = sum(                                # every latent case, weighted by its probability
                mass * (0.0 if outcome == UNAVAILABLE else model.cost_norm[t])
                for outcome, mass in joint.predictive(history, test).items()
            )
            assert charged == pytest.approx(p * model.cost_norm[t], abs=1e-15)
            assert twin.expected_charged_cost(history, test) == pytest.approx(p * model.cost_norm[t], abs=1e-15)
    assert twin.expected_charged_cost((), "t2") == 0.0                       # p = 0: never charged
    assert twin.expected_charged_cost((), "t0") == model.cost_norm[0]        # p = 1: always charged
    assert twin.expected_charged_cost((), "t1") == pytest.approx(0.3 * 0.625)
    assert twin.charged_cost("t1", UNAVAILABLE) == 0.0
    assert twin.charged_cost("t1", "pos") == 0.625 and twin.charged_cost("t1", "neg", usd=True) == 250.0


def test_expected_cost_follows_the_history_when_missingness_is_informative():
    model = toy_model(availability=[[1.0, 1.0], [0.3, 0.6], [0.0, 0.0]])
    twin, joint = Twin(model), Joint(model)
    history = (("t0", "pos"),)
    posterior = joint.posterior(history)
    p_available = posterior @ np.array([0.3, 0.6])
    assert twin.p_available(history, "t1") == pytest.approx(p_available, abs=1e-15)
    assert twin.expected_charged_cost(history, "t1") == pytest.approx(p_available * 0.625, abs=1e-15)
    assert twin.p_available((), "t1") == pytest.approx(0.6 * 0.3 + 0.4 * 0.6, abs=1e-15)
    # here an unavailable result is evidence about the class
    np.testing.assert_allclose(twin.true_posterior((("t1", UNAVAILABLE),)),
                               np.array([0.6 * 0.7, 0.4 * 0.4]) / (0.6 * 0.7 + 0.4 * 0.4), atol=1e-15)


def test_published_information_gains_and_expected_cost_of_the_existing_study():
    always = Twin(tm.build("base", reduced=False, missingness=False))
    for test, published in PUBLISHED_INFORMATION.items():
        assert always.true_information((), test) == pytest.approx(published, abs=1e-12)
    full = Twin(tm.build("base", reduced=False))
    total = sum(full.expected_charged_cost((), test, usd=True) for test in full.model.tests)
    assert total == pytest.approx(3779.55, abs=1e-9)        # expected_all_available_cost_usd in sanity.json
    for test, published in PUBLISHED_INFORMATION.items():   # uninformative missingness scales information by p
        p = full.p_available((), test)
        assert full.true_information((), test) == pytest.approx(p * published, abs=1e-12)


def test_uninformative_missingness_leaves_the_posterior_unchanged():
    twin = Twin(tm.build("base", reduced=True))
    history = (("CT", "appendicitis"),)
    np.testing.assert_allclose(twin.true_posterior(history + (("Ultrasound", UNAVAILABLE),)),
                               twin.true_posterior(history), rtol=0, atol=1e-15)
    np.testing.assert_allclose(twin.true_posterior(()), twin.model.prior, rtol=0, atol=1e-15)


def test_evidence_identity_in_the_true_model():
    twin = Twin(tm.build("duplicated_panel", reduced=True))
    repeat = "Liver Function Panel (repeat)"
    for outcome in ("cholecystitis", "nonspecific", UNAVAILABLE):
        history = (("CT", "pancreatitis"), ("Liver Function Panel", outcome))
        for test in ("Liver Function Panel", repeat):           # the same test again, and its duplicate
            predictive = twin.true_predictive(history, test)
            assert predictive.outcomes == (outcome,) and predictive.weights.tolist() == [1.0]
            assert twin.true_information(history, test) == 0.0
            assert twin.p_available(history, test) == (0.0 if outcome == UNAVAILABLE else 1.0)
            np.testing.assert_array_equal(twin.true_posterior(history + ((test, outcome),)), twin.true_posterior(history))
        # the duplicate is charged like the original when it returns a result, and is free when it cannot
        assert twin.expected_charged_cost(history, repeat) == (0.0 if outcome == UNAVAILABLE else twin.model.cost_norm[6])
    # before the original is seen the duplicate is an ordinary test
    np.testing.assert_allclose(twin.true_predictive((), repeat).weights,
                               twin.true_predictive((), "Liver Function Panel").weights)


def test_impossible_and_malformed_histories_are_rejected():
    twin = Twin(tm.build("duplicated_panel", reduced=True))
    with pytest.raises(ValueError, match="contradict"):
        twin.true_posterior((("Liver Function Panel", "cholecystitis"), ("Liver Function Panel (repeat)", "nonspecific")))
    with pytest.raises(ValueError, match="probability zero"):
        twin.true_posterior((("Physical Examination", UNAVAILABLE),))        # the examination is always available
    with pytest.raises(ValueError):
        twin.true_posterior((("MRI", "appendicitis"),))                      # not a test of the reduced twin
    with pytest.raises(ValueError):
        twin.true_predictive((), "CT scan")
    with pytest.raises(ValueError):
        twin.true_posterior((("CT", "positive"),))


def test_eligible_tests_are_the_ones_not_yet_requested():
    twin = Twin(tm.build("base", reduced=True))
    assert twin.eligible(()) == twin.model.tests
    assert twin.eligible((("CT", UNAVAILABLE), ("Urinalysis", "nonspecific"))) == (
        "Physical Examination", "Ultrasound", "Complete Blood Count", "Liver Function Panel")


def test_counterfactual_interface_returns_every_outcome_and_availability():
    twin = Twin(tm.build("duplicated_panel", reduced=True))
    for patient in twin.sample_patients(40, seed=2, split="validation"):
        outcomes = twin.all_outcomes(patient)
        assert tuple(outcomes) == twin.model.tests
        for test, (finding, available) in outcomes.items():
            assert finding in twin.model.findings and isinstance(available, bool)
            assert twin.observe(patient, test) == (finding if available else UNAVAILABLE)
        assert outcomes["Liver Function Panel (repeat)"] == outcomes["Liver Function Panel"]
        assert outcomes["Physical Examination"][1] is True
        assert patient.diagnosis in twin.model.classes and patient.phenotype in twin.model.phenotypes


def test_fingerprint_covers_the_latent_content_and_ignores_the_display_id():
    twin = Twin(tm.build("base", reduced=True))
    patient = twin.sample_patients(1, seed=0, split="test")[0]
    assert len(patient.fingerprint) == 64 and int(patient.fingerprint, 16) >= 0
    assert dataclasses.replace(patient, display_id="another label").fingerprint == patient.fingerprint
    other_class = next(c for c in twin.model.classes if c != patient.diagnosis)
    other_phenotype = next(c for c in twin.model.phenotypes if c != patient.phenotype)
    flipped_finding = tuple(
        ("nonspecific" if f != "nonspecific" else "appendicitis") if i == 3 else f for i, f in enumerate(patient.findings))
    flipped_available = tuple((not a) if i == 5 else a for i, a in enumerate(patient.available))
    changed = [
        dataclasses.replace(patient, diagnosis=other_class),
        dataclasses.replace(patient, phenotype=other_phenotype),
        dataclasses.replace(patient, findings=flipped_finding),
        dataclasses.replace(patient, available=flipped_available),
    ]
    assert len({patient.fingerprint} | {p.fingerprint for p in changed}) == 5


def test_render_has_three_deterministic_templates_that_keep_every_result():
    twin = Twin(tm.build("duplicated_panel", reduced=True))
    assert TEMPLATES == ("structured", "narrative", "clinical")
    history = (("CT", "appendicitis"), ("Ultrasound", UNAVAILABLE), ("Liver Function Panel", "nonspecific"),
               ("Liver Function Panel (repeat)", "nonspecific"))
    texts = {template: twin.render(history, template) for template in TEMPLATES}
    assert len(set(texts.values())) == 3
    for template, text in texts.items():
        assert twin.render(history, template) == text
        positions = [text.index(test) for test, _ in history[:3]]
        assert positions == sorted(positions), template                      # results appear in the order read
        for name in twin.model.classes:
            assert name in text                                              # the candidate diagnoses are stated
        for word in ("seed", "split", "synthetic id", "fingerprint"):
            assert word not in text
    with pytest.raises(ValueError):
        twin.render(history, "poetic")


def test_render_distinguishes_every_single_result_and_the_order_of_results():
    twin = Twin(tm.build("base", reduced=False))
    for template in TEMPLATES:
        singles = {twin.render(((test, outcome),), template) for test in twin.model.tests for outcome in twin.model.outcomes}
        assert len(singles) == 12 * 6, template
        empty = twin.render((), template)
        assert empty not in singles and empty
        a, b = ("CT", "appendicitis"), ("MRI", "nonspecific")
        assert twin.render((a, b), template) != twin.render((b, a), template)


def test_clinical_template_uses_the_snapshot_text_without_its_random_distractors():
    from tiicee.twin import source

    twin = Twin(tm.build("base", reduced=False))
    text = twin.render((("CT", "appendicitis"), ("Radiograph", UNAVAILABLE)), "clinical")
    assert "CT abdomen/pelvis: dilated appendix (~11 mm) with periappendiceal fat stranding." in text
    assert "[Radiograph] UNAVAILABLE" in text
    for distractor in source.generators._DISTRACTORS:
        assert distractor not in text
    assert "can overlap with" not in text
