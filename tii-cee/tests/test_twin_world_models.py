"""World models W0, W1, W2, W2-cc and the controlled errors."""
from __future__ import annotations

import collections

import numpy as np
import pytest

from tiicee.interfaces import UNAVAILABLE, Predictive
from tiicee.twin import model as tm
from tiicee.twin import world_models as wm
from tiicee.twin.env import Twin
from tiicee.twin.readers import BayesReader, SimulatedReader

from test_twin_support import Joint, kl, load_cee_check, sample_histories, toy_model


def dense(twin, predictive: Predictive) -> np.ndarray:
    """A predictive as a vector over the model's outcomes, findings first and UNAVAILABLE last."""
    out = np.zeros(len(twin.model.outcomes))
    for outcome, weight in zip(predictive.outcomes, predictive.weights):
        out[twin.model.outcomes.index(outcome)] += weight
    return out


def check_predictive(predictive: Predictive):
    assert isinstance(predictive, Predictive) and predictive.merged
    assert len(set(predictive.outcomes)) == len(predictive.outcomes)
    assert (predictive.weights > 0).all() and predictive.weights.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.fixture(scope="module")
def base():
    return Twin(tm.build("base", reduced=True))


@pytest.fixture(scope="module")
def duplicated():
    return Twin(tm.build("duplicated_panel", reduced=True))


# --- W0 -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", tm.variants(allow_held_out=True))
def test_w0_is_the_true_predictive_and_its_budget_is_the_true_information(variant):
    model = tm.build(variant, reduced=True, allow_held_out=True)
    twin, joint = Twin(model), Joint(model)
    w0 = wm.w0(twin)
    for history in sample_histories(twin, n=8):
        for test in model.tests:
            predictive = w0.outcomes(history, test, None, None)
            check_predictive(predictive)
            reference = joint.predictive(history, test)
            np.testing.assert_allclose(dense(twin, predictive), [reference[o] for o in model.outcomes], rtol=0, atol=1e-12)
            assert w0.budget(history, test) == pytest.approx(joint.information(history, test), abs=1e-12)


def test_class_conditional_form_recovers_the_predictive_and_the_budget(base):
    joint = Joint(base.model)
    w0 = wm.w0(base)
    for history in sample_histories(base, n=6):
        prior = joint.posterior(history)
        for test in base.eligible(history):
            outcomes, likelihood = w0.likelihood(history, test)            # likelihood[class, outcome]
            assert likelihood.shape == (4, len(outcomes))
            np.testing.assert_allclose(likelihood.sum(1), 1.0, atol=1e-12)
            total = joint.mass(history).sum()
            for j, outcome in enumerate(outcomes):                          # P(o | class, h) by enumeration
                by_class = joint.mass(history + ((test, outcome),)).sum(1) / total
                np.testing.assert_allclose(likelihood[:, j], by_class / prior, rtol=0, atol=1e-12)
            marginal = prior @ likelihood
            np.testing.assert_allclose(marginal, w0.outcomes(history, test, None, None).weights, atol=1e-12)
            # B-model from its definition: sum_o P(o) KL(post(. | o) || prior)
            budget = sum(marginal[j] * kl(prior * likelihood[:, j] / marginal[j], prior) for j in range(len(outcomes)))
            assert w0.budget(history, test) == pytest.approx(budget, abs=1e-12)


def test_sampled_predictive_draws_k_outcomes_from_the_exact_one(base):
    w0 = wm.w0(base)
    history = (("CT", "cholecystitis"),)
    exact = w0.outcomes(history, "Ultrasound", None, None)
    for k in (4, 8, 16, 32):
        drawn = w0.outcomes(history, "Ultrasound", k, np.random.default_rng(0))
        assert len(drawn.outcomes) == k and not drawn.merged
        np.testing.assert_allclose(drawn.weights, 1.0 / k)
        assert set(drawn.outcomes) <= set(exact.outcomes)
        again = w0.outcomes(history, "Ultrasound", k, np.random.default_rng(0))
        assert again.outcomes == drawn.outcomes
    many = w0.outcomes(history, "Ultrasound", 40000, np.random.default_rng(1))
    counts = collections.Counter(many.outcomes)
    for outcome, p in zip(exact.outcomes, exact.weights):
        assert abs(counts[outcome] / 40000 - p) < 4 * np.sqrt(p * (1 - p) / 40000)
    with pytest.raises(ValueError):
        w0.outcomes(history, "Ultrasound", 8, None)


def test_models_with_evidence_identity_give_a_point_mass_on_re_exposure(duplicated):
    reader = SimulatedReader(duplicated)
    ignored, responsive = wm.matched_pair(duplicated, reader, kl=0.02)
    models = {
        "W0": wm.w0(duplicated),
        "W2": wm.fit_w2(duplicated, seed=0, n_train=400),
        "mirror": wm.mirror_corrupt(duplicated, 0.75),
        "temperature": wm.temperature(duplicated, 2.0),
        "ignored": ignored,
        "responsive": responsive,
    }
    repeat = "Liver Function Panel (repeat)"
    for outcome in ("cholecystitis", UNAVAILABLE):
        history = (("CT", "nonspecific"), ("Liver Function Panel", outcome))
        for name, model in models.items():
            for test in ("Liver Function Panel", repeat):
                predictive = model.outcomes(history, test, None, None)
                assert predictive.outcomes == (outcome,) and predictive.weights.tolist() == [1.0], name
                drawn = model.outcomes(history, test, 5, np.random.default_rng(0))
                assert set(drawn.outcomes) == {outcome}, name
            if hasattr(model, "budget"):
                assert model.budget(history, repeat) == 0.0, name


# --- W1 -----------------------------------------------------------------------------------------------
def test_w1_is_the_count_model_of_the_training_patients(base):
    n, seed = 500, 3
    w1 = wm.fit_w1(base, seed=seed, n_train=n)
    model = base.model
    patients = base.sample_patients(n, seed=seed, split="train", purpose="fitting")
    by_class = collections.Counter(p.diagnosis for p in patients)
    prior = np.array([(by_class[c] + 1) / (n + 4) for c in model.classes])
    lik, avail = {}, {}
    for t, test in enumerate(model.tests):
        provided = [p for p in patients if p.available[t]]
        avail[test] = (len(provided) + 1) / (n + 2)
        for c in model.classes:
            seen = collections.Counter(p.findings[t] for p in provided if p.diagnosis == c)
            total = sum(seen.values())
            lik[test, c] = np.array([(seen[f] + 1) / (total + 5) for f in model.findings])

    history = (("CT", "cholecystitis"), ("Ultrasound", UNAVAILABLE), ("Liver Function Panel", "cholecystitis"))
    belief = prior.copy()                                    # naive Bayes over the findings; a missing result says nothing
    for test, outcome in history:
        if outcome != UNAVAILABLE:
            belief = belief * np.array([lik[test, c][model.findings.index(outcome)] for c in model.classes])
    belief = belief / belief.sum()
    for at, b in (((), prior), (history, belief)):
        for test in base.eligible(at):
            by_finding = sum(b[y] * lik[test, c] for y, c in enumerate(model.classes))
            expected = np.append(avail[test] * by_finding, 1 - avail[test])
            predictive = w1.outcomes(at, test, None, None)
            check_predictive(predictive)
            np.testing.assert_allclose(dense(base, predictive), expected, rtol=0, atol=1e-12)
            post = [b * np.array([lik[test, c][s] for c in model.classes]) / by_finding[s] for s in range(5)]
            budget = avail[test] * sum(by_finding[s] * kl(post[s], b) for s in range(5))
            assert w1.budget(at, test) == pytest.approx(budget, abs=1e-12)
            outcomes, likelihood = w1.likelihood(at, test)
            np.testing.assert_allclose(b @ likelihood, dense(base, predictive)[[model.outcomes.index(o) for o in outcomes]], atol=1e-12)


def test_w1_uses_the_fitting_stream_of_the_training_split(base):
    a = wm.fit_w1(base, seed=3, n_train=300).outcomes((), "CT", None, None).weights
    b = wm.fit_w1(base, seed=3, n_train=300).outcomes((), "CT", None, None).weights
    c = wm.fit_w1(base, seed=4, n_train=300).outcomes((), "CT", None, None).weights
    np.testing.assert_array_equal(a, b)
    assert np.abs(a - c).max() > 1e-6


def test_w1_has_no_notion_of_a_repeated_measurement(duplicated):
    w1 = wm.fit_w1(duplicated, seed=0, n_train=1000)
    history = (("Liver Function Panel", "cholecystitis"),)
    predictive = w1.outcomes(history, "Liver Function Panel (repeat)", None, None)
    assert len(predictive.outcomes) == 6 and predictive.weights.max() < 0.9      # imagines disagreeing repeats
    assert w1.budget(history, "Liver Function Panel (repeat)") > 0.01
    # and it counts the repeat as fresh evidence: its belief after both is the prior times the likelihood twice
    both = history + (("Liver Function Panel (repeat)", "cholecystitis"),)
    _, likelihood = w1.likelihood((), "Liver Function Panel")
    column = likelihood[:, duplicated.model.outcomes.index("cholecystitis")]
    _, ct = w1.likelihood((), "CT")
    for at, power in ((history, 1), (both, 2)):
        belief = w1.prior * column ** power
        np.testing.assert_allclose(dense(duplicated, w1.outcomes(at, "CT", None, None)), (belief / belief.sum()) @ ct, atol=1e-12)
    np.testing.assert_array_equal(w1.lik[6], w1.lik[4])            # the two tests saw the same results in training


# --- W2 and W2-cc ---------------------------------------------------------------------------------------
def test_w2_fit_increases_its_objective_and_reports_it_exactly(base):
    n, seed, smoothing = 300, 5, 1.0
    w2 = wm.fit_w2(base, seed=seed, n_train=n, smoothing=smoothing)
    trace = np.array(w2.trace)
    assert len(trace) >= 3 and (np.diff(trace) >= -1e-9).all()
    fitted = w2.twin.model
    assert fitted.tests == base.model.tests and fitted.item_of == base.model.item_of
    assert not np.allclose(fitted.finding_lik, base.model.finding_lik)        # fitted, not copied
    # penalized log-likelihood of the latent phenotype model, recomputed patient by patient
    patients = base.sample_patients(n, seed=seed, split="train", purpose="fitting")
    pheno, lik = fitted.phenotype_given_class, fitted.finding_lik[:, 0]       # lik[item, phenotype, finding]
    objective = smoothing * (np.log(pheno).sum() + np.log(lik).sum())
    for p in patients:
        y = fitted.classes.index(p.diagnosis)
        per_phenotype = pheno[y].copy()
        for i in range(len(fitted.items)):
            if p.available[i]:
                per_phenotype = per_phenotype * lik[i, :, fitted.findings.index(p.findings[i])]
        objective += np.log(per_phenotype.sum())
    assert trace[-1] == pytest.approx(objective, abs=1e-8)
    # the parts with a closed form are plain smoothed counts
    by_class = collections.Counter(p.diagnosis for p in patients)
    np.testing.assert_allclose(fitted.prior, [(by_class[c] + 1) / (n + 4) for c in fitted.classes], atol=1e-15)
    provided = np.array([sum(p.available[i] for p in patients) for i in range(6)])
    np.testing.assert_allclose(fitted.availability, np.repeat(((provided + 1) / (n + 2))[:, None], 4, axis=1), atol=1e-15)
    np.testing.assert_allclose(fitted.finding_lik, np.repeat(fitted.finding_lik[:, :1], 4, axis=1))


def test_w2_fit_stops_at_a_fixed_point_of_its_update_equations(base):
    n, seed, a = 250, 8, 1.0
    w2 = wm.fit_w2(base, seed=seed, n_train=n, smoothing=a, max_iter=20000, tol=1e-15)
    fitted = w2.twin.model
    pheno, lik = fitted.phenotype_given_class, fitted.finding_lik[:, 0]
    patients = base.sample_patients(n, seed=seed, split="train", purpose="fitting")
    pheno_counts, lik_counts = np.zeros((4, 4)), np.zeros((6, 4, 5))
    for p in patients:                                  # each patient's phenotype posterior under the fitted model
        y = fitted.classes.index(p.diagnosis)
        weight = pheno[y].copy()
        for i in range(6):
            if p.available[i]:
                weight = weight * lik[i, :, fitted.findings.index(p.findings[i])]
        weight = weight / weight.sum()
        pheno_counts[y] += weight
        for i in range(6):
            if p.available[i]:
                lik_counts[i, :, fitted.findings.index(p.findings[i])] += weight
    # re-estimating from those posteriors returns the same parameters
    np.testing.assert_allclose((pheno_counts + a) / (pheno_counts.sum(1, keepdims=True) + 4 * a), pheno, rtol=0, atol=1e-6)
    np.testing.assert_allclose((lik_counts + a) / (lik_counts.sum(-1, keepdims=True) + 5 * a), lik, rtol=0, atol=1e-6)


def test_w2_is_the_exact_predictive_of_its_fitted_model_and_w2cc_supplies_its_budget(duplicated):
    w2 = wm.fit_w2(duplicated, seed=1, n_train=600)
    joint = Joint(w2.twin.model)
    for history in sample_histories(duplicated, n=6):
        prior = joint.posterior(history)
        for test in duplicated.model.tests:
            predictive = w2.outcomes(history, test, None, None)
            check_predictive(predictive)
            reference = joint.predictive(history, test)
            np.testing.assert_allclose(dense(duplicated, predictive), [reference[o] for o in duplicated.model.outcomes], atol=1e-12)
            assert w2.budget(history, test) == pytest.approx(joint.information(history, test), abs=1e-12)
            outcomes, likelihood = w2.likelihood(history, test)
            np.testing.assert_allclose(prior @ likelihood, predictive.weights, atol=1e-12)


def test_w2_learns_the_phenotype_correlation_that_w1_cannot_represent():
    strong = Twin(tm.build("strong_correlation", reduced=True))
    w0, w1, w2 = wm.w0(strong), wm.fit_w1(strong, seed=0, n_train=8000), wm.fit_w2(strong, seed=0, n_train=8000)
    findings = strong.model.findings
    gaps = {"W1": [], "W2": []}
    for first in findings:                      # after two concordant imaging results, what will the panel show?
        history = (("CT", first), ("Ultrasound", first))
        truth = dense(strong, w0.outcomes(history, "Liver Function Panel", None, None))
        gaps["W1"].append(kl(truth, dense(strong, w1.outcomes(history, "Liver Function Panel", None, None))))
        gaps["W2"].append(kl(truth, dense(strong, w2.outcomes(history, "Liver Function Panel", None, None))))
    assert max(gaps["W2"]) < 0.01
    assert np.mean(gaps["W2"]) < 0.25 * np.mean(gaps["W1"])
    # it recovers the atypical-presentation rate of this variant, 0.20, which W1 has no parameter for
    fitted = w2.twin.model.phenotype_given_class
    assert np.abs(np.diag(fitted) - 0.80).max() < 0.03
    off_diagonal = fitted[~np.eye(4, dtype=bool)]
    assert np.abs(off_diagonal - 0.20 / 3).max() < 0.03
    # at the empty history both are fitted marginals and both are close
    truth = dense(strong, w0.outcomes((), "CT", None, None))
    assert kl(truth, dense(strong, w1.outcomes((), "CT", None, None))) < 1e-3
    assert kl(truth, dense(strong, w2.outcomes((), "CT", None, None))) < 1e-3


# --- controlled errors ----------------------------------------------------------------------------------
def test_mirror_map_reproduces_the_published_information_ranking():
    # results/synthetic/sanity.json at df467ea ranks the twelve tests by information; the mirror pairs rank k with 12 - 1 - k
    ranking = ("MRI", "CT", "Ultrasound", "Liver Function Panel", "Physical Examination", "Comprehensive Metabolic Panel",
               "Basic Metabolic Panel", "Complete Blood Count", "Renal Function Panel", "Electrolyte Panel", "Urinalysis", "Radiograph")
    mirror = wm.mirror_map(tm.build("base", reduced=False))
    assert mirror == {a: b for a, b in zip(ranking, reversed(ranking))}


def test_mirror_corruption_mixes_each_test_with_its_mirror(base):
    model = base.model
    always = Joint(tm.build("base", reduced=True, missingness=False))
    ranked = sorted(model.tests, key=lambda test: -always.information((), test))
    mirror = dict(zip(ranked, reversed(ranked)))
    assert wm.mirror_map(model) == mirror
    truth = wm.w0(base)
    np.testing.assert_allclose(
        dense(base, wm.mirror_corrupt(base, 0.0).outcomes((("CT", "appendicitis"),), "Ultrasound", None, None)),
        dense(base, truth.outcomes((("CT", "appendicitis"),), "Ultrasound", None, None)), atol=1e-15)
    for alpha in (0.25, 0.5, 1.0):
        corrupted = wm.mirror_corrupt(base, alpha)
        for t, test in enumerate(model.tests):
            own = dense(base, truth.outcomes((), test, None, None))
            other = dense(base, truth.outcomes((), mirror[test], None, None))
            p = model.availability[t, 0]
            # findings given a result are mixed with the mirror's; availability stays the test's own
            mixed = (1 - alpha) * own[:5] / own[:5].sum() + alpha * other[:5] / other[:5].sum()
            np.testing.assert_allclose(dense(base, corrupted.outcomes((), test, None, None)), np.append(p * mixed, 1 - p), atol=1e-12)
        i, j = 1, model.tests.index(mirror["CT"])
        np.testing.assert_allclose(corrupted.twin.model.finding_lik[i],
                                   (1 - alpha) * model.finding_lik[i] + alpha * model.finding_lik[j], atol=1e-15)
        for name in ("prior", "phenotype_given_class", "availability"):
            np.testing.assert_array_equal(getattr(corrupted.twin.model, name), getattr(model, name))
        joint = Joint(corrupted.twin.model)                     # a wrong model, but one model: its own exact predictive
        history = (("CT", "pancreatitis"), ("Urinalysis", "nonspecific"))
        reference = joint.predictive(history, "Liver Function Panel")
        np.testing.assert_allclose(dense(base, corrupted.outcomes(history, "Liver Function Panel", None, None)),
                                   [reference[o] for o in model.outcomes], atol=1e-12)
    with pytest.raises(ValueError):
        wm.mirror_corrupt(base, 1.5)


def test_temperature_reshapes_the_findings_and_leaves_availability_alone(base):
    truth = wm.w0(base)
    history = (("Physical Examination", "pancreatitis"),)
    for test in ("CT", "Ultrasound"):
        exact = dense(base, truth.outcomes(history, test, None, None))
        np.testing.assert_allclose(dense(base, wm.temperature(base, 1.0).outcomes(history, test, None, None)), exact, atol=1e-15)
        for temp in (0.5, 2.0, 4.0):
            got = dense(base, wm.temperature(base, temp).outcomes(history, test, None, None))
            powered = exact[:5] ** (1.0 / temp)
            expected = np.append(exact[:5].sum() * powered / powered.sum(), exact[5])
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
        flat = dense(base, wm.temperature(base, 4.0).outcomes(history, test, None, None))[:5]
        sharp = dense(base, wm.temperature(base, 0.5).outcomes(history, test, None, None))[:5]
        assert flat.max() < exact[:5].max() < sharp.max()
    with pytest.raises(ValueError):
        wm.temperature(base, 0.0)


@pytest.mark.parametrize("reader_name", ["r0", "independence", "over", "entrench"])
def test_matched_pair_has_equal_kl_and_opposite_relevance_to_the_reader(base, reader_name):
    reader = BayesReader(base) if reader_name == "r0" else {
        "independence": SimulatedReader(base), "over": SimulatedReader(base, power=2.0),
        "entrench": SimulatedReader(base, drift="entrench")}[reader_name]
    target = 0.02
    truth = wm.w0(base)
    ignored, responsive = wm.matched_pair(base, reader, kl=target)
    for history in sample_histories(base, n=6):
        for test in base.eligible(history):
            exact = truth.outcomes(history, test, None, None)
            p_star = dense(base, exact)
            p_ign = dense(base, ignored.outcomes(history, test, None, None))
            p_res = dense(base, responsive.outcomes(history, test, None, None))
            check_predictive(ignored.outcomes(history, test, None, None))
            check_predictive(responsive.outcomes(history, test, None, None))
            kl_ign, kl_res = kl(p_star, p_ign), kl(p_star, p_res)
            assert kl_ign == pytest.approx(target, abs=1e-8) and kl_res == pytest.approx(target, abs=1e-8)
            assert abs(kl_ign - kl_res) < 0.005
            assert p_ign[5] == pytest.approx(p_star[5], abs=1e-15) and p_res[5] == pytest.approx(p_star[5], abs=1e-15)
            # the reader's imagined mean belief: unchanged by the first model, moved by the second
            posteriors = np.stack([reader.read(history, test, o) for o in base.model.outcomes])
            mean_star, mean_ign, mean_res = p_star @ posteriors, p_ign @ posteriors, p_res @ posteriors
            shift_ign, shift_res = np.abs(mean_ign - mean_star).sum(), np.abs(mean_res - mean_star).sum()
            assert shift_ign < 1e-10
            assert shift_res > 1e4 * max(shift_ign, 1e-14)        # however certain the reader already is
            if history == () and test == "CT":
                assert shift_res > 0.02                           # a visible move where the reader is open


def test_matched_pair_responsive_member_moves_the_reader_more_than_other_directions(base):
    reader = SimulatedReader(base)
    target = 1e-5                                     # small enough for the first-order statement to be tight
    _, responsive = wm.matched_pair(base, reader, kl=target)
    truth = wm.w0(base)
    rng = np.random.default_rng(0)
    for history, test in (((), "CT"), ((("CT", "appendicitis"),), "Ultrasound"), ((("Ultrasound", "nonspecific"),), "Liver Function Panel")):
        p_star = dense(base, truth.outcomes(history, test, None, None))
        posteriors = np.stack([reader.read(history, test, o) for o in base.model.outcomes])
        moved = np.linalg.norm((dense(base, responsive.outcomes(history, test, None, None)) - p_star) @ posteriors)
        for _ in range(300):                          # any other change to the findings with the same KL from the truth
            direction = rng.normal(size=5)
            direction -= direction.mean()
            lo, hi = 0.0, min(p_star[s] / -direction[s] for s in range(5) if direction[s] < 0)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                candidate = p_star.copy()
                candidate[:5] += mid * direction
                lo, hi = (mid, hi) if kl(p_star, candidate) < target else (lo, mid)
            candidate = p_star.copy()
            candidate[:5] += lo * direction
            assert np.linalg.norm((candidate - p_star) @ posteriors) <= moved * 1.01


@pytest.mark.parametrize("reader_name", ["r0", "independence", "entrench", "over"])
def test_matched_pair_separates_reader_weighted_error_at_equal_kl(base, reader_name):
    """The quantity of specification section 5.5, with the projection of cee_check.py as the independent reference."""
    project = load_cee_check()["project"]
    reader = BayesReader(base) if reader_name == "r0" else {
        "independence": SimulatedReader(base), "entrench": SimulatedReader(base, drift="entrench"),
        "over": SimulatedReader(base, power=2.0)}[reader_name]
    truth = wm.w0(base)
    errors = {"ignored": [], "responsive": []}
    for history in sample_histories(base, n=8):
        belief = reader.belief(history)
        for test in base.eligible(history):
            p_star = dense(base, truth.outcomes(history, test, None, None))
            keep = p_star > 0
            posteriors = np.stack([reader.read(history, test, o) for o in base.model.outcomes])[keep]
            for name, member in zip(errors, wm.matched_pair(base, reader, kl=0.02)):
                p_hat = dense(base, member.outcomes(history, test, None, None))[keep]
                _, z, _, kl_inc, converged = project(belief, p_hat, posteriors)
                assert converged
                errors[name].append(abs(float((p_star[keep] - p_hat) @ np.log(z))))     # sum_o (P* - P_hat)(o) log Z_o
                if reader_name == "r0" and name == "ignored":
                    assert kl_inc < 1e-12              # a coherent pair stays coherent under the ignored error
    assert np.median(errors["responsive"]) > 1e-3
    assert np.median(errors["ignored"]) < 0.05 * np.median(errors["responsive"])
    if reader_name == "r0":
        assert max(errors["ignored"]) < 1e-12


def test_matched_pair_requires_a_positive_target(base):
    with pytest.raises(ValueError):
        wm.matched_pair(base, BayesReader(base), kl=0.0)


def test_matched_pair_says_so_when_the_reader_ignores_no_direction():
    twin = Twin(toy_model())            # two findings, two classes: the one direction there is moves the reader
    ignored, responsive = wm.matched_pair(twin, BayesReader(twin), kl=0.01)
    with pytest.raises(ValueError, match="more findings than classes"):
        ignored.outcomes((), "t0", None, None)
