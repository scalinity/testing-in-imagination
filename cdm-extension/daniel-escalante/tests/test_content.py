"""The latent-disease content world model and the EIG-per-dollar planner (SPEC_CONTENT_WM.md §1.3, §2, §3),
on invented examples: panel A separates appendicitis from the rest, panel B carries no information."""
import numpy as np
import pytest

from tim import content
from tim.enums import DISEASES


def examples(n=10):
    out = []
    for d in DISEASES:
        for _ in range(n):
            out.append((d, {"A": ("low",) if d == "appendicitis" else ("high",), "B": ("mid",), "C": None}))
    return out


@pytest.fixture(scope="module")
def model():
    return content.fit_model(examples())


def test_class_conditionals_are_add_one_smoothed_counts(model):
    # Panel A, item 0, sees {low, high} plus "other" and "absent": 4 outcomes; appendicitis saw low 10 times.
    assert model.outcomes[("A", 0)] == ("absent", "high", "low", "other")
    assert model.probs[("A", 0)][DISEASES.index("appendicitis")].tolist() == pytest.approx([1 / 14, 1 / 14, 11 / 14,
                                                                                            1 / 14])
    assert model.prior.tolist() == pytest.approx([0.25] * 4)
    assert ("C", 0) not in model.probs  # never in the chart in training


def test_the_posterior_multiplies_the_prior_by_the_observed_panel_likelihoods(model):
    prior = np.array([0.4, 0.2, 0.2, 0.2])
    belief = content.posterior(prior, model, [("A", ("low",))])
    expected = prior * model.lik("A", ("low",))
    assert belief.tolist() == pytest.approx((expected / expected.sum()).tolist())
    assert content.posterior(prior, model, [("B", ("mid",))]).tolist() == pytest.approx(prior.tolist())
    assert model.lik("A", ("never seen",)).tolist() == model.lik("A", ("other",)).tolist()


def test_eig_is_zero_for_an_uninformative_panel_and_scales_with_availability(model):
    belief = np.full(4, 0.25)
    rng = lambda: np.random.default_rng(0)
    assert content.eig(model, belief, "B", rng(), draws=64) == pytest.approx(0, abs=1e-12)
    informative = content.eig(model, belief, "A", rng(), draws=64)
    assert informative > 0.1
    assert content.eig(model, belief, "A", rng(), draws=64, p_avail=0.5) == pytest.approx(informative / 2)


def test_realized_gain_is_the_entropy_drop_under_the_true_result(model):
    belief = np.full(4, 0.25)
    after = content.posterior(belief, model, [("A", ("low",))])
    assert content.realized_gain(model, belief, "A", ("low",)) == pytest.approx(
        content.entropy(belief) - content.entropy(after))


def planner(model, kind, history_prior=None, label="cholecystitis", alpha=0.0, truth=None, avail=None, costs=None):
    return content.PatientPlanner(
        kind=kind, model=model, hpi_prior=np.array(history_prior or [0.25] * 4), label=label, alpha=alpha,
        truth=truth or {"A": ("high",), "B": ("mid",), "C": None}, avail=avail or {"A": 0.9, "B": 0.9, "C": 0.9},
        costs=costs or {"A": 10.0, "B": 1.0, "C": 1.0}, split="val", patient=7, draws=64)


def test_the_planner_ranks_untried_labs_by_eig_per_dollar_with_configured_tie_order(model):
    learned = planner(model, "learned")
    assert learned.choose(["B", "A"], [], t=2) == "A"           # A informs; B never does
    assert planner(model, "learned", costs={"A": 1e6, "B": 1.0, "C": 1.0}).choose(["B", "A"], [], t=2) == "A"
    flat = planner(model, "learned", history_prior=[1.0, 0, 0, 0])  # certain already: every score is 0
    assert flat.choose(["B", "A"], [], t=2) == "B" and flat.choose(["A", "B"], [], t=2) == "A"


def test_observed_panels_update_the_learned_belief(model):
    before = planner(model, "learned")
    assert before.belief(["B"]).tolist() == pytest.approx([0.25] * 4)
    after = before.belief(["A"])  # the truth for A is "high": appendicitis becomes unlikely
    assert after[DISEASES.index("appendicitis")] < 0.05


def test_the_prior_arm_ignores_this_patient(model):
    a = planner(model, "prior", history_prior=[0.97, 0.01, 0.01, 0.01])
    b = planner(model, "prior", history_prior=[0.01, 0.01, 0.01, 0.97])
    assert a.belief(["A"]).tolist() == b.belief([]).tolist() == model.prior.tolist()
    assert a.score("A", [], t=1) == pytest.approx(b.score("A", [], t=1))


def test_mislead_at_alpha_zero_is_the_learned_model_and_at_one_samples_the_wrong_class(model):
    learned = planner(model, "learned")
    assert planner(model, "mislead", alpha=0.0).score("A", [], t=3) == pytest.approx(learned.score("A", [], t=3))
    wrong = planner(model, "mislead", alpha=1.0, label="cholecystitis")
    assert wrong.wrong in [d for d in DISEASES if d != "cholecystitis"]
    assert wrong.score("A", [], t=3) != pytest.approx(learned.score("A", [], t=3))


def test_the_oracle_uses_the_true_result_and_true_availability(model):
    oracle = planner(model, "oracle", truth={"A": ("high",), "B": ("mid",), "C": None})
    gain = content.realized_gain(model, np.full(4, 0.25), "A", ("high",))
    assert oracle.score("A", [], t=0) == pytest.approx(gain / 10.0)
    assert oracle.score("C", [], t=0) == 0.0  # not in the chart


def test_scores_are_reproducible_per_patient_turn_and_panel(model):
    assert planner(model, "learned").score("A", [], t=4) == planner(model, "learned").score("A", [], t=4)


class HistoryPrior:
    """Stands in for the HPI model: leans 0.7 towards the disease the invented history names."""
    def predict(self, history):
        return np.array([0.7 if d in history else 0.1 for d in DISEASES])


def cases():
    out = []
    for k, d in enumerate(DISEASES * 6):
        truth = {panel: None for panel in content.LABS}
        truth["Complete Blood Count"] = ("low",) if d == "appendicitis" else ("high",)
        truth["Urinalysis"] = ("mid",)
        out.append((f"history {k} {d}", d, truth, 100 + k))
    return out


def lab_examples():
    return [(d, {"Complete Blood Count": ("low",) if d == "appendicitis" else ("high",), "Urinalysis": ("mid",)})
            for d in DISEASES for _ in range(10)]


def test_validation_quality_scores_log_loss_against_the_train_marginal_and_calibrates_eig():
    model = content.fit_model(lab_examples())
    result = content.quality(HistoryPrior(), model, cases(), split="val", draws=32)
    assert result["tuples"] == 24 * 2 * content.ORDERS and result["items"] == result["tuples"]
    assert result["nll"] < result["nll_marginal"] and result["skill"] == pytest.approx(
        1 - result["nll"] / result["nll_marginal"])
    assert len(result["deciles"]) == 10 and result["spearman"] > 0
    assert content.quality(HistoryPrior(), model, cases(), split="val", draws=32) == result


def test_the_premise_counts_validation_admissions_with_a_lab_request():
    rows = [{"requested_tests": ["ct", "complete blood count"]}, {"requested_tests": ["urinalysis", "electrolyte panel"]},
            {"requested_tests": ["physical examination"]}]
    assert content.premise(rows, minimum=2) == {"admissions_with_lab": 2, "lab_requests": 3, "minimum": 2,
                                                "passed": True}
    assert content.premise(rows, minimum=3)["passed"] is False


def hpi_items(n, offset=0):
    words = {"appendicitis": "right lower quadrant", "cholecystitis": "right upper quadrant murphy",
             "diverticulitis": "left lower quadrant sigmoid", "pancreatitis": "epigastric back lipase"}
    return [{"patient_history": f"invented case {k + offset} with {words[d]} pain", "condition": d}
            for k, d in enumerate(DISEASES * n)]


def test_the_hpi_prior_is_fit_on_train_and_its_temperature_on_validation():
    from tim.enums import CONFIGURED_TESTS
    prior = content.fit_prior(hpi_items(8), hpi_items(4, offset=100), list(CONFIGURED_TESTS))
    p = prior.predict("invented case with epigastric back lipase pain")
    assert p.shape == (4,) and p.sum() == pytest.approx(1) and int(np.argmax(p)) == DISEASES.index("pancreatitis")
    assert prior.temperature in content.TEMPERATURES and prior.c in (0.01, 0.1, 1, 10)
    again = content.fit_prior(hpi_items(8), hpi_items(4, offset=100), list(CONFIGURED_TESTS))
    assert again.temperature == prior.temperature and again.predict("x").tolist() == prior.predict("x").tolist()
    assert "_stop_words_id" not in vars(content.fit_prior(hpi_items(8), hpi_items(4, offset=100),
                                                          list(CONFIGURED_TESTS)).vectorizer)  # stable pickles
