"""The content world model and its planner (SPEC_CONTENT_WM.md §1.3, §2, §3).

The disease y is latent. p(y | h) is the HPI prior times the likelihood of each observed lab panel, and
p(o | h, a) = sum_y p(y | h) p(o | y, a), with a panel's items independent given y. The planner scores an
untried panel by its expected information gain over p(y | h), times availability, per dollar. Every fitted
object here is patient-derived and stays private.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from tim.enums import CONFIGURED_TESTS, DISEASES
from tim.labs import ABSENT, OTHER

LABS = CONFIGURED_TESTS[:7]
DRAWS = 64  # Monte Carlo draws per expected information gain (§2.1)
ORDERS = 8  # random test orders per admission for chart-derived histories (§1.4)
TEMPERATURES = tuple(round(0.5 + 0.1 * k, 1) for k in range(26))  # 0.5 ... 3.0, chosen on validation
KINDS = ("learned", "oracle", "prior", "mislead")


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


@dataclass(frozen=True)
class DiagnosisPrior:
    """p_psi0(y | HPI): Phase B's HPI TF-IDF multinomial regression (C by 5-fold CV on train), temperature-
    scaled on validation (SPEC_CONTENT_WM.md §1.3)."""
    vectorizer: object
    model: object
    temperature: float
    c: float

    def predict(self, history) -> np.ndarray:
        log_p = self.model.predict_log_proba(self.vectorizer.transform([history]))[0]
        classes = list(self.model.classes_)
        z = np.array([log_p[classes.index(d)] for d in DISEASES]) / self.temperature
        p = np.exp(z - z.max())
        return p / p.sum()


def fit_prior(train, val, tests) -> DiagnosisPrior:
    """Fit on train items; choose the temperature that minimizes validation log-loss (ties: the smaller)."""
    from tim import leakage
    c = leakage._choose_c(train, tests, "hpi")
    vectorizer, model = leakage._fit(train, tests, "hpi", c)

    def loss(temperature):
        fitted = DiagnosisPrior(vectorizer, model, temperature, c)
        return -sum(np.log(max(fitted.predict(i["patient_history"])[DISEASES.index(i["condition"])], 1e-12))
                    for i in val)
    temperature = min(TEMPERATURES, key=lambda t: (loss(t), t))
    # transform() re-caches a memory address that availability._fitted clears; drop it so the pickled
    # store, and so its published digest, is the same for identical fits.
    vars(vectorizer).pop("_stop_words_id", None)
    return DiagnosisPrior(vectorizer, model, temperature, c)


@dataclass(frozen=True)
class ContentModel:
    prior: np.ndarray  # train class prior over DISEASES
    outcomes: dict     # (panel, item index) -> outcome names
    probs: dict        # (panel, item index) -> array [disease, outcome] of p(outcome | disease, panel)

    def lik(self, panel, outcome) -> np.ndarray:
        """p(outcome | y, panel) for every disease; an unseen value counts as "other"."""
        out = np.ones(len(DISEASES))
        for j, value in enumerate(outcome):
            names = self.outcomes.get((panel, j))
            if names is not None:
                out = out * self.probs[(panel, j)][:, names.index(value if value in names else OTHER)]
        return out

    def sample(self, panel, y, rng) -> tuple:
        out, j = [], 0
        while (panel, j) in self.probs:
            names = self.outcomes[(panel, j)]
            out.append(names[rng.choice(len(names), p=self.probs[(panel, j)][y])])
            j += 1
        return tuple(out)


def fit_model(examples) -> ContentModel:
    """Class prior and add-one-smoothed outcome frequencies given the disease, from train examples
    (label, {panel: outcome tuple, or None when the panel is not in the chart})."""
    labels = [label for label, _ in examples]
    prior = np.array([labels.count(d) for d in DISEASES], dtype=float)
    counts = {}
    for label, panels in examples:
        for panel, outcome in panels.items():
            for j, value in enumerate(outcome or ()):
                row = counts.setdefault((panel, j), {}).setdefault(value, np.zeros(len(DISEASES)))
                row[DISEASES.index(label)] += 1
    outcomes, probs = {}, {}
    for key, by_value in counts.items():
        names = tuple(sorted(set(by_value) | {OTHER, ABSENT}))
        table = np.ones((len(DISEASES), len(names)))
        for k, name in enumerate(names):
            table[:, k] += by_value.get(name, 0)
        outcomes[key], probs[key] = names, table / table.sum(axis=1, keepdims=True)
    return ContentModel(prior / prior.sum(), outcomes, probs)


def posterior(prior, model, observed) -> np.ndarray:
    belief = np.array(prior, dtype=float)
    for panel, outcome in observed:
        belief = belief * model.lik(panel, outcome)
    total = belief.sum()
    return belief / total if total > 0 else np.array(prior, dtype=float)


def entropy(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def eig(model, belief, panel, rng, draws=DRAWS, p_avail=1.0, sample_from=None) -> float:
    """p_avail · [H(belief) − mean H(belief after an imagined result)], results drawn by first drawing a
    disease from sample_from (default: the belief) and then the panel's items given it."""
    if (panel, 0) not in model.probs:
        return 0.0
    q = belief if sample_from is None else sample_from
    after = [entropy(posterior(belief, model, [(panel, model.sample(panel, rng.choice(len(DISEASES), p=q), rng))]))
             for _ in range(draws)]
    return p_avail * (entropy(belief) - float(np.mean(after)))


def realized_gain(model, belief, panel, outcome) -> float:
    return entropy(belief) - entropy(posterior(belief, model, [(panel, outcome)]))


@dataclass
class PatientPlanner:
    """One patient's planner for one arm: learned, oracle, prior or mislead (§3)."""
    kind: str
    model: ContentModel
    hpi_prior: np.ndarray
    label: str
    alpha: float
    truth: dict   # panel -> this patient's outcome, or None when not in the chart
    avail: dict   # panel -> p(in the chart | HPI), the Phase B learned head
    costs: dict   # panel -> USD
    split: str
    patient: int
    draws: int = DRAWS

    @property
    def wrong(self) -> str:
        """The misleading arm's wrong disease, fixed per patient by sha256."""
        others = [d for d in DISEASES if d != self.label]
        return others[_seed("tim-mislead", self.split, self.patient) % len(others)]

    def belief(self, provided) -> np.ndarray:
        if self.kind == "prior":
            return self.model.prior.copy()  # imagined results ignore this patient
        observed = [(panel, self.truth[panel]) for panel in provided if self.truth.get(panel) is not None]
        return posterior(self.hpi_prior, self.model, observed)

    def score(self, panel, provided, t) -> float:
        belief = self.belief(provided)
        if self.kind == "oracle":
            truth = self.truth.get(panel)
            return 0.0 if truth is None else realized_gain(self.model, belief, panel, truth) / self.costs[panel]
        q = belief
        if self.kind == "mislead":
            q = (1 - self.alpha) * belief + self.alpha * np.eye(len(DISEASES))[DISEASES.index(self.wrong)]
        rng = np.random.default_rng(_seed("tim-eig", self.split, self.patient, t, panel))
        return eig(self.model, belief, panel, rng, self.draws, self.avail[panel], q) / self.costs[panel]

    def choose(self, untried, provided, t) -> str:
        """The untried lab panel with the highest score; ties go to the order given (the configured order)."""
        scores = [self.score(panel, provided, t) for panel in untried]
        return untried[int(np.argmax(scores))]


# ---- validation gates (§5): world-model quality and the premise ---------------------------------------

def _orders(split, patient, panels) -> list[list[str]]:
    return [sorted(panels, key=lambda panel: _seed("tim-order", split, patient, r, panel)) for r in range(ORDERS)]


def quality(prior_model, model, cases, split="val", draws=DRAWS) -> dict:
    """Gate 2 (§1.5): per-item log-loss of each charted panel's true result given the chart-derived history,
    against the train marginal, over ORDERS random orders per admission; and EIG calibration (predicted
    against realized gain, by decile of predicted) on the first order. cases: (history, label, truth, patient)."""
    import math
    from scipy.stats import spearmanr
    marginal = {key: model.prior @ table for key, table in model.probs.items()}
    nll = nll_marginal = 0.0
    items = tuples = 0
    pairs = []
    for history, _, truth, patient in cases:
        start = prior_model.predict(history)
        charted = [panel for panel in LABS if truth.get(panel) is not None]
        for r, order in enumerate(_orders(split, patient, charted)):
            observed = []
            for panel in order:
                belief, outcome = posterior(start, model, observed), truth[panel]
                nll -= math.log(float(belief @ model.lik(panel, outcome)))
                for j, value in enumerate(outcome):
                    names = model.outcomes.get((panel, j))
                    if names is not None:
                        nll_marginal -= math.log(float(marginal[(panel, j)][names.index(value if value in names
                                                                                       else OTHER)]))
                        items += 1
                tuples += 1
                if r == 0:
                    rng = np.random.default_rng(_seed("tim-eig-calibration", split, patient, panel))
                    pairs.append((eig(model, belief, panel, rng, draws), realized_gain(model, belief, panel, outcome)))
                observed.append((panel, outcome))
    pairs.sort(key=lambda pair: pair[0])
    groups = [g for g in np.array_split(np.arange(len(pairs)), min(10, len(pairs))) if len(g)]
    deciles = [{"predicted": float(np.mean([pairs[i][0] for i in g])), "realized": float(np.mean([pairs[i][1] for i in g]))}
               for g in groups]
    rho = spearmanr(range(len(deciles)), [d["realized"] for d in deciles]).correlation if len(deciles) > 1 else None
    per_item = (nll / items, nll_marginal / items) if items else (None, None)
    return {"tuples": tuples, "items": items, "nll": per_item[0], "nll_marginal": per_item[1],
            "skill": None if not items else 1 - per_item[0] / per_item[1],
            "deciles": deciles, "spearman": None if rho is None or rho != rho else float(rho)}


def premise(rows, minimum=24) -> dict:
    """Gate 1 (§2.5): validation admissions in which V3 requested at least one lab panel."""
    lab_names = {name.lower() for name in LABS}
    counts = [sum(name in lab_names for name in r["requested_tests"]) for r in rows]
    with_lab = sum(count > 0 for count in counts)
    return {"admissions_with_lab": with_lab, "lab_requests": sum(counts), "minimum": minimum,
            "passed": with_lab >= minimum}
