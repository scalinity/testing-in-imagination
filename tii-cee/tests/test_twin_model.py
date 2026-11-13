"""The generative model: parameters against cee_check.py and hand-typed values, and the variants."""
from __future__ import annotations

import numpy as np
import pytest

from tiicee.interfaces import UNAVAILABLE
from tiicee.twin import model as tm
from tiicee.twin import source

from test_twin_support import load_cee_check

CLASSES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
FULL_TESTS = (
    "Physical Examination", "CT", "MRI", "Radiograph", "Ultrasound", "Complete Blood Count",
    "Basic Metabolic Panel", "Comprehensive Metabolic Panel", "Renal Function Panel",
    "Liver Function Panel", "Urinalysis", "Electrolyte Panel",
)
# Typed from profiles.py and actions.py at df467ea, in the order of FULL_TESTS.
AVAILABILITY = (1.00, 0.75, 0.10, 0.35, 0.50, 0.95, 0.75, 0.55, 0.45, 0.70, 0.60, 0.60)
COST_USD = (300.0, 1306.0, 4866.0, 434.0, 1288.0, 71.0, 298.0, 636.0, 394.0, 413.0, 50.0, 134.0)
REDUCED = ("Physical Examination", "CT", "Ultrasound", "Complete Blood Count", "Liver Function Panel", "Urinalysis")


@pytest.fixture(scope="module")
def full():
    return tm.build("base", reduced=False)


@pytest.fixture(scope="module")
def reduced():
    return tm.build("base", reduced=True)


def test_full_base_model_matches_cee_check(full):
    ck = load_cee_check()
    assert full.classes == CLASSES and full.phenotypes == CLASSES
    assert full.findings == CLASSES + ("nonspecific",)
    assert full.outcomes == full.findings + (UNAVAILABLE,)
    assert full.tests == FULL_TESTS and full.items == FULL_TESTS and full.item_of == tuple(range(12))
    np.testing.assert_allclose(full.prior, ck["PRIOR"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(full.phenotype_given_class, ck["pe_given_y"](0.07), rtol=0, atol=1e-15)
    for y in range(4):                                 # findings depend on the phenotype, not the class
        np.testing.assert_allclose(full.finding_lik[:, y], ck["LIK"], rtol=0, atol=1e-12)
    hits = np.stack([np.diagonal(full.finding_lik[t, 0, :, :4]) for t in range(12)])
    np.testing.assert_allclose(hits, ck["HITS"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(full.finding_lik[:, 0, 0, 1], ck["CONF"], rtol=0, atol=1e-15)


def test_full_base_model_matches_hand_typed_values(full):
    assert full.prior[0] == pytest.approx(957 / 2400) and full.prior[2] == pytest.approx(257 / 2400)
    ct = FULL_TESTS.index("CT")
    # CT in an appendicitis phenotype: 0.88 on target, 0.02 on each other disease, the rest nonspecific
    assert full.finding_lik[ct, 3, 0] == pytest.approx([0.88, 0.02, 0.02, 0.02, 0.06])
    lfp = FULL_TESTS.index("Liver Function Panel")
    assert full.finding_lik[lfp, 0, 1] == pytest.approx([0.05, 0.65, 0.05, 0.05, 0.20])
    assert full.phenotype_given_class[1] == pytest.approx([0.07 / 3, 0.93, 0.07 / 3, 0.07 / 3])
    np.testing.assert_allclose(full.availability, np.repeat(np.array(AVAILABILITY)[:, None], 4, axis=1))
    np.testing.assert_allclose(full.cost_usd, COST_USD)
    np.testing.assert_allclose(full.cost_norm, np.array(COST_USD) / 10190.0)
    assert full.cost_norm.sum() == pytest.approx(1.0)
    assert full.step_limit == 14
    assert not full.held_out and full.variant == "base"
    assert full.source == source.snapshot_digest()


def test_every_distribution_is_normalized(full):
    np.testing.assert_allclose(full.prior.sum(), 1.0)
    np.testing.assert_allclose(full.phenotype_given_class.sum(1), 1.0)
    np.testing.assert_allclose(full.finding_lik.sum(-1), 1.0)
    # one structural zero in the snapshot: ultrasound in a cholecystitis phenotype is never nonspecific
    # (1 - 0.88 - 3 * 0.04). Every finding has positive probability given every class.
    zeros = {(full.tests[t], full.phenotypes[e], full.findings[s]) for t, e, s in np.argwhere(full.finding_lik[:, 0] <= 0)}
    assert zeros == {("Ultrasound", "cholecystitis", "nonspecific")}
    assert full.class_lik()[:, :, :5].min() > 0.003
    np.testing.assert_allclose(full.outcome_lik().sum(-1), 1.0)
    np.testing.assert_allclose(full.class_lik().sum(-1), 1.0)


def test_class_conditional_marginals_match_cee_check(full):
    ck = load_cee_check()
    nb = np.einsum("ye,tes->tys", ck["pe_given_y"](0.07), ck["LIK"])
    always = tm.build("base", reduced=False, missingness=False)
    np.testing.assert_allclose(always.availability, 1.0)
    np.testing.assert_allclose(always.class_lik()[:, :, :5], nb, rtol=0, atol=1e-12)
    np.testing.assert_allclose(always.class_lik()[:, :, 5], 0.0)
    # with missingness the same marginals are scaled by availability and the rest is UNAVAILABLE
    np.testing.assert_allclose(full.class_lik()[:, :, :5], nb * np.array(AVAILABILITY)[:, None, None], atol=1e-12)
    np.testing.assert_allclose(full.class_lik()[:, :, 5], 1 - np.repeat(np.array(AVAILABILITY)[:, None], 4, axis=1))


def test_reduced_twin_is_the_six_named_tests_with_the_snapshot_parameters(full, reduced):
    assert reduced.tests == REDUCED == tm.REDUCED_TESTS
    rows = [FULL_TESTS.index(t) for t in REDUCED]
    np.testing.assert_array_equal(reduced.finding_lik, full.finding_lik[rows])
    np.testing.assert_array_equal(reduced.availability, full.availability[rows])
    np.testing.assert_array_equal(reduced.cost_usd, full.cost_usd[rows])
    # costs stay on the existing study's scale: normalized by the twelve-test total of $10,190
    np.testing.assert_allclose(reduced.cost_norm, np.array(COST_USD)[rows] / 10190.0)
    np.testing.assert_array_equal(reduced.prior, full.prior)
    np.testing.assert_array_equal(reduced.phenotype_given_class, full.phenotype_given_class)
    assert reduced.step_limit == 14 and reduced.item_of == tuple(range(6))


def test_restrict_keeps_parameters_and_rejects_unknown_tests(reduced):
    sub = tm.restrict(reduced, ("Liver Function Panel", "CT"))
    assert sub.tests == ("Liver Function Panel", "CT")
    np.testing.assert_array_equal(sub.finding_lik[0], reduced.finding_lik[4])
    np.testing.assert_array_equal(sub.cost_norm, reduced.cost_norm[[4, 1]])
    with pytest.raises(ValueError):
        tm.restrict(reduced, ("MRI",))


def test_strong_correlation_changes_only_the_atypical_rate(reduced):
    ck = load_cee_check()
    strong = tm.build("strong_correlation", reduced=True)
    np.testing.assert_allclose(strong.phenotype_given_class, ck["pe_given_y"](0.20), atol=1e-15)
    assert tm.P_ATYPICAL_STRONG == 0.20
    for name in ("prior", "finding_lik", "availability", "cost_usd", "cost_norm"):
        np.testing.assert_array_equal(getattr(strong, name), getattr(reduced, name))
    assert strong.tests == reduced.tests and not strong.held_out


def test_no_correlation_keeps_single_test_marginals_and_makes_tests_independent_given_the_class(reduced):
    flat = tm.build("no_correlation", reduced=True)
    np.testing.assert_allclose(flat.class_lik(), reduced.class_lik(), rtol=0, atol=1e-15)
    for name in ("prior", "availability", "cost_usd", "cost_norm"):
        np.testing.assert_array_equal(getattr(flat, name), getattr(reduced, name))

    def pair_given_class(model, i, j, y):               # P(finding i, finding j | class), summing the phenotype
        return sum(
            model.phenotype_given_class[y, e] * np.outer(model.finding_lik[i, y, e], model.finding_lik[j, y, e])
            for e in range(4)
        )

    def single_given_class(model, i, y):
        return sum(model.phenotype_given_class[y, e] * model.finding_lik[i, y, e] for e in range(4))

    for y in range(4):
        product = np.outer(single_given_class(flat, 1, y), single_given_class(flat, 2, y))
        np.testing.assert_allclose(pair_given_class(flat, 1, 2, y), product, rtol=0, atol=1e-15)
        correlated = pair_given_class(reduced, 1, 2, y) - np.outer(single_given_class(reduced, 1, y), single_given_class(reduced, 2, y))
        assert np.abs(correlated).max() > 1e-3          # the base model is not independent given the class


def test_duplicated_panel_adds_one_test_that_measures_an_existing_item(reduced):
    dup = tm.build("duplicated_panel", reduced=True)
    assert dup.tests == REDUCED + ("Liver Function Panel (repeat)",)
    assert tm.DUPLICATED_TEST == "Liver Function Panel"
    assert dup.items == reduced.items
    assert dup.item_of == (0, 1, 2, 3, 4, 5, 4)
    assert dup.cost_usd[6] == dup.cost_usd[4] == 413.0 and dup.cost_norm[6] == dup.cost_norm[4]
    for name in ("prior", "phenotype_given_class", "finding_lik", "availability"):
        np.testing.assert_array_equal(getattr(dup, name), getattr(reduced, name))
    np.testing.assert_array_equal(dup.class_lik()[6], dup.class_lik()[4])


def test_informative_missingness_is_held_out_and_keeps_the_marginal_availability(reduced):
    with pytest.raises(tm.HeldOutVariant):
        tm.build("informative_missingness", reduced=True)
    informative = tm.build("informative_missingness", reduced=True, allow_held_out=True)
    assert informative.held_out and informative.variant == "informative_missingness"
    assert ((informative.availability >= 0) & (informative.availability <= 1)).all()
    np.testing.assert_allclose(informative.availability @ informative.prior, reduced.availability[:, 0], atol=1e-15)
    np.testing.assert_array_equal(informative.availability[0], 1.0)             # the examination is always there
    assert np.ptp(informative.availability[1:], axis=1).min() > 1e-3            # every other test depends on the class
    assert np.ptp(informative.availability[2]) > 0.25                           # ultrasound strongly so
    # hand value for ultrasound in cholecystitis: 0.5 + 0.5 * (0.88 - sum_y prior_y hit_y)
    hits = np.array([0.70, 0.88, 0.25, 0.30])
    assert informative.availability[2, 1] == pytest.approx(0.5 + 0.5 * (0.88 - hits @ reduced.prior))
    for name in ("prior", "phenotype_given_class", "finding_lik", "cost_norm"):
        np.testing.assert_array_equal(getattr(informative, name), getattr(reduced, name))


def test_variant_listing_excludes_the_held_out_variant_unless_asked():
    assert tm.variants() == ("base", "no_correlation", "strong_correlation", "duplicated_panel")
    assert tm.variants(allow_held_out=True) == tm.variants() + ("informative_missingness",)
    assert all(not tm.build(v, reduced=True).held_out for v in tm.variants())
    with pytest.raises(ValueError):
        tm.build("nonsense", reduced=True)


def test_parameters_cannot_be_modified_in_place(reduced):
    with pytest.raises(ValueError):
        reduced.prior[0] = 0.5
    with pytest.raises(ValueError):
        reduced.finding_lik[0, 0, 0, 0] = 0.5


def test_malformed_models_are_rejected(reduced):
    import dataclasses

    with pytest.raises(ValueError):
        dataclasses.replace(reduced, prior=np.array([0.5, 0.5, 0.5, 0.5]))
    with pytest.raises(ValueError):
        dataclasses.replace(reduced, item_of=(0, 1, 2, 3, 4, 9))
    with pytest.raises(ValueError):
        dataclasses.replace(reduced, availability=reduced.availability + 0.5)
