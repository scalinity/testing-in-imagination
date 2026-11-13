"""Readers R0 and R4: the Bayes reader against enumeration, the simulated family against cee_check.py and by hand."""
from __future__ import annotations

import numpy as np
import pytest

from tiicee.interfaces import UNAVAILABLE
from tiicee.twin import model as tm
from tiicee.twin.env import Twin
from tiicee.twin.readers import BayesReader, SimulatedReader, r4_family

from test_twin_support import Joint, load_cee_check, sample_histories, toy_model

E = float(np.exp(0.4))


def normalized(*values):
    v = np.array(values, dtype=float)
    return v / v.sum()


# --- R0 -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", tm.variants(allow_held_out=True))
def test_r0_is_the_exact_posterior(variant):
    model = tm.build(variant, reduced=True, allow_held_out=True)
    twin, joint = Twin(model), Joint(model)
    reader = BayesReader(twin)
    assert reader.classes == model.classes
    for history in sample_histories(twin, n=12):
        belief = reader.belief(history)
        np.testing.assert_allclose(belief, joint.posterior(history), rtol=0, atol=1e-12)
        for test in twin.eligible(history)[:3]:
            for outcome, p in joint.predictive(history, test).items():
                if p > 0:
                    extended = history + ((test, outcome),)
                    np.testing.assert_allclose(reader.read(history, test, outcome), joint.posterior(extended), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(reader.belief(history), belief)       # reading did not move the belief


def test_r0_does_not_move_on_re_exposure_of_identical_evidence():
    twin = Twin(tm.build("duplicated_panel", reduced=True))
    reader = BayesReader(twin)
    history = (("CT", "cholecystitis"), ("Liver Function Panel", "cholecystitis"), ("Ultrasound", UNAVAILABLE))
    belief = reader.belief(history)
    for test, outcome in history:                                           # the same result shown again
        np.testing.assert_array_equal(reader.read(history, test, outcome), belief)
    np.testing.assert_array_equal(reader.read(history, "Liver Function Panel (repeat)", "cholecystitis"), belief)
    np.testing.assert_array_equal(reader.belief(history + (("Liver Function Panel (repeat)", "cholecystitis"),)), belief)


def test_r0_ignores_an_imagined_result_the_true_model_rules_out_but_rejects_an_impossible_real_history():
    twin = Twin(tm.build("duplicated_panel", reduced=True))
    reader = BayesReader(twin)
    history = (("Liver Function Panel", "cholecystitis"),)
    belief = reader.belief(history)
    # a world model without evidence identity can imagine a duplicate that disagrees with the original
    np.testing.assert_array_equal(reader.read(history, "Liver Function Panel (repeat)", "nonspecific"), belief)
    np.testing.assert_array_equal(reader.read(history, "Physical Examination", UNAVAILABLE), belief)
    with pytest.raises(ValueError):
        reader.belief(history + (("Liver Function Panel (repeat)", "nonspecific"),))
    with pytest.raises(ValueError):
        reader.read(history, "MRI", "appendicitis")                         # not a test of this twin


# --- R4 against cee_check.py ---------------------------------------------------------------------------
THEIR_NAME = {"independence": "nb", "entrench": "entrench", "fixed": "fixed", "over": "over", "under": "under"}


@pytest.mark.parametrize("missingness", [False, True])
def test_r4_reactions_match_cee_check_over_one_and_two_steps(missingness):
    ck = load_cee_check()
    nb = np.einsum("ye,tes->tys", ck["pe_given_y"](0.07), ck["LIK"])        # [test, class, finding]
    twin = Twin(tm.build("base", reduced=False, missingness=missingness))
    tests, findings = twin.model.tests, twin.model.findings
    readers = r4_family(twin)
    for name, theirs in THEIR_NAME.items():
        reader = readers[name]
        np.testing.assert_allclose(reader.belief(()), ck["PRIOR"], rtol=0, atol=1e-15)
        for t1 in (1, 4, 9):
            q1 = ck["react_batch"](theirs, ck["PRIOR"][None], nb[t1][None])[0]              # [finding, class]
            for s1 in range(5):
                history = ((tests[t1], findings[s1]),)
                np.testing.assert_allclose(reader.read((), tests[t1], findings[s1]), q1[s1], rtol=0, atol=1e-12)
                np.testing.assert_allclose(reader.belief(history), q1[s1], rtol=0, atol=1e-12)
                for t2 in (0, 5):
                    q2 = ck["react_batch"](theirs, q1[s1][None], nb[t2][None])[0]
                    for s2 in range(5):
                        np.testing.assert_allclose(reader.read(history, tests[t2], findings[s2]), q2[s2], rtol=0, atol=1e-12)


def test_r4_defaults_follow_cee_check():
    twin = Twin(toy_model())
    family = r4_family(twin)
    assert tuple(family) == ("independence", "entrench", "fixed", "over", "under",
                             "entrench+over", "entrench+under", "fixed+over", "fixed+under")
    assert family["over"].power == 2.0 and family["under"].power == 0.5 and family["independence"].power == 1.0
    assert family["entrench"].delta == 0.4 and family["fixed"].delta == 0.4 and family["fixed"].fixed_class == 1
    assert all(reader.classes == twin.model.classes for reader in family.values())


# --- R4 by hand on the toy model -----------------------------------------------------------------------
# P(t0 = pos | class a) = 0.8*0.9 + 0.2*0.2 = 0.76,  P(t0 = pos | class b) = 0.3*0.9 + 0.7*0.2 = 0.41, prior (0.6, 0.4)
def test_r4_single_reaction_by_hand():
    family = r4_family(Twin(toy_model()))
    read = {name: reader.read((), "t0", "pos") for name, reader in family.items()}
    np.testing.assert_allclose(read["independence"], normalized(0.6 * 0.76, 0.4 * 0.41), atol=1e-15)
    np.testing.assert_allclose(read["over"], normalized(0.6 * 0.76 ** 2, 0.4 * 0.41 ** 2), atol=1e-15)
    np.testing.assert_allclose(read["under"], normalized(0.6 * 0.76 ** 0.5, 0.4 * 0.41 ** 0.5), atol=1e-15)
    np.testing.assert_allclose(read["entrench"], normalized(0.6 * 0.76 * E, 0.4 * 0.41), atol=1e-15)       # prior argmax is a
    np.testing.assert_allclose(read["fixed"], normalized(0.6 * 0.76, 0.4 * 0.41 * E), atol=1e-15)          # always class b
    np.testing.assert_allclose(read["entrench+over"], normalized(0.6 * 0.76 ** 2 * E, 0.4 * 0.41 ** 2), atol=1e-15)
    np.testing.assert_allclose(read["entrench+under"], normalized(0.6 * 0.76 ** 0.5 * E, 0.4 * 0.41 ** 0.5), atol=1e-15)
    np.testing.assert_allclose(read["fixed+over"], normalized(0.6 * 0.76 ** 2, 0.4 * 0.41 ** 2 * E), atol=1e-15)
    np.testing.assert_allclose(read["fixed+under"], normalized(0.6 * 0.76 ** 0.5, 0.4 * 0.41 ** 0.5 * E), atol=1e-15)


def test_r4_entrenching_follows_the_argmax_of_the_belief_before_the_update():
    reader = SimulatedReader(Twin(toy_model(availability=(1.0, 1.0, 0.0))), drift="entrench")
    # t0 = neg: likelihoods 0.24 and 0.59; the prior argmax (a) is entrenched, yet b ends ahead
    first = normalized(0.6 * 0.24 * E, 0.4 * 0.59)
    assert first[1] > first[0]
    np.testing.assert_allclose(reader.belief((("t0", "neg"),)), first, atol=1e-15)
    # t1 = pos: P(pos | a) = 0.8*0.7 + 0.2*0.4 = 0.64, P(pos | b) = 0.3*0.7 + 0.7*0.4 = 0.49; now b is entrenched
    second = normalized(first[0] * 0.64, first[1] * 0.49 * E)
    np.testing.assert_allclose(reader.read((("t0", "neg"),), "t1", "pos"), second, atol=1e-15)
    np.testing.assert_allclose(reader.belief((("t0", "neg"), ("t1", "pos"))), second, atol=1e-15)


def test_r4_reaction_to_an_uninformative_unavailable_result():
    family = r4_family(Twin(toy_model(availability=(1.0, 0.3, 0.0))))
    prior = np.array([0.6, 0.4])
    for name in ("independence", "over", "under"):                 # scale errors have nothing to scale
        np.testing.assert_allclose(family[name].read((), "t1", UNAVAILABLE), prior, atol=1e-15)
    np.testing.assert_allclose(family["entrench"].read((), "t1", UNAVAILABLE), normalized(0.6 * E, 0.4), atol=1e-15)
    np.testing.assert_allclose(family["fixed"].read((), "t1", UNAVAILABLE), normalized(0.6, 0.4 * E), atol=1e-15)


def test_r4_reads_informative_missingness_through_its_class_conditional_model():
    reader = SimulatedReader(Twin(toy_model(availability=[[1.0, 1.0], [0.3, 0.6], [0.0, 0.0]])))
    np.testing.assert_allclose(reader.read((), "t1", UNAVAILABLE), normalized(0.6 * 0.7, 0.4 * 0.4), atol=1e-15)


def test_r4_takes_no_evidence_from_a_result_its_model_rules_out_for_every_class():
    family = r4_family(Twin(toy_model(availability=(1.0, 0.3, 0.0))))
    prior = np.array([0.6, 0.4])
    for test, outcome in (("t2", "pos"), ("t0", UNAVAILABLE)):
        np.testing.assert_allclose(family["independence"].read((), test, outcome), prior, atol=1e-15)
        np.testing.assert_allclose(family["over"].read((), test, outcome), prior, atol=1e-15)
        np.testing.assert_allclose(family["entrench"].read((), test, outcome), normalized(0.6 * E, 0.4), atol=1e-15)


# --- R4 against R0 --------------------------------------------------------------------------------------
def test_independence_reader_is_exact_without_phenotype_correlation_and_double_counts_with_it():
    flat = Twin(tm.build("no_correlation", reduced=True))
    for history in sample_histories(flat, n=20):
        np.testing.assert_allclose(SimulatedReader(flat).belief(history), BayesReader(flat).belief(history), rtol=0, atol=1e-12)
    base = Twin(tm.build("base", reduced=True))
    history = (("CT", "cholecystitis"), ("Ultrasound", "cholecystitis"), ("Liver Function Panel", "cholecystitis"))
    independent, exact = SimulatedReader(base).belief(history), BayesReader(base).belief(history)
    assert independent[1] > exact[1] and np.abs(independent - exact).max() > 1e-3     # overconfident in cholecystitis


def test_independence_reader_double_counts_repeated_evidence_and_r0_does_not():
    twin = Twin(tm.build("duplicated_panel", reduced=True))
    history = (("Liver Function Panel", "cholecystitis"),)
    independent, exact = SimulatedReader(twin), BayesReader(twin)
    for test in ("Liver Function Panel", "Liver Function Panel (repeat)"):
        moved = independent.read(history, test, "cholecystitis")
        assert moved[1] > independent.belief(history)[1] + 0.05
        np.testing.assert_array_equal(exact.read(history, test, "cholecystitis"), exact.belief(history))


def test_readers_are_deterministic_and_normalized():
    twin = Twin(tm.build("base", reduced=True))
    readers = dict(r4_family(twin), r0=BayesReader(twin))
    for history in sample_histories(twin, n=6):
        for reader in readers.values():
            belief = reader.belief(history)
            assert belief.shape == (4,) and belief.sum() == pytest.approx(1.0, abs=1e-12) and (belief >= 0).all()
            np.testing.assert_array_equal(reader.belief(history), belief)
            for test in twin.eligible(history)[:2]:
                read = reader.read(history, test, "nonspecific")
                assert read.sum() == pytest.approx(1.0, abs=1e-12)
                np.testing.assert_array_equal(reader.read(history, test, "nonspecific"), read)


def test_reader_options_are_validated():
    twin = Twin(toy_model())
    with pytest.raises(ValueError):
        SimulatedReader(twin, drift="sideways")
    with pytest.raises(ValueError):
        SimulatedReader(twin, power=0.0)
    with pytest.raises(ValueError):
        SimulatedReader(twin, drift="fixed", fixed_class=5)
