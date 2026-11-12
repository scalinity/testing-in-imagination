"""The content world model and its planner (SPEC_CONTENT_WM.md §1.3, §2, §3).

The disease y is latent. p(y | h) is the HPI prior times the likelihood of each observed lab panel, and
p(o | h, a) = sum_y p(y | h) p(o | y, a), with a panel's items independent given y. The planner scores an
untried panel by its expected information gain over p(y | h), times availability, per dollar. Every fitted
object here is patient-derived and stays private.
"""
from dataclasses import dataclass, replace
import hashlib
import re

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


def store_bytes(store) -> bytes:
    """Pickle the private store. Every transform() re-caches a memory address in the HPI vectorizer, so it is
    dropped here, right before pickling, to keep identical fits byte-identical (as availability._fitted does)."""
    import pickle
    vars(store["prior"].vectorizer).pop("_stop_words_id", None)
    return pickle.dumps(store)


@dataclass(frozen=True)
class ContentModel:
    prior: np.ndarray  # train class prior over DISEASES
    outcomes: dict     # (panel, item index) -> outcome names
    probs: dict        # (panel, item index) -> array [disease, outcome] of p(outcome | disease, panel)
    drop_absent: bool = False  # amendment C1, fix A: absent items carry no evidence about the disease
    tau: float = 1.0           # amendment C1, fix B: observed evidence is raised to 1 / tau

    def log_lik(self, panel, outcome, evidence=False) -> np.ndarray:
        """log p(outcome | y, panel) for every disease (an unseen value counts as "other"). As evidence for
        a belief update it follows C1: absent items are skipped under fix A and the sum is divided by tau."""
        out = np.zeros(len(DISEASES))
        for j, value in enumerate(outcome):
            if evidence and self.drop_absent and value == ABSENT:
                continue
            names = self.outcomes.get((panel, j))
            if names is not None:
                out = out + np.log(self.probs[(panel, j)][:, names.index(value if value in names else OTHER)])
        return out / self.tau if evidence else out

    def lik(self, panel, outcome) -> np.ndarray:
        """The predictive likelihood of the full outcome, absent items included (unchanged by C1)."""
        return np.exp(self.log_lik(panel, outcome))

    def evidence(self, panel, outcome) -> np.ndarray:
        return np.exp(self.log_lik(panel, outcome, evidence=True))

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


def _log(p) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(p, dtype=float))


def posterior(prior, model, observed) -> np.ndarray:
    """p(y | h) from the prior and each observed panel's evidence, in log space so that many items cannot
    underflow."""
    log_belief = _log(prior)
    for panel, outcome in observed:
        log_belief = log_belief + model.log_lik(panel, outcome, evidence=True)
    belief = np.exp(log_belief - log_belief.max())
    return belief / belief.sum()


def with_fix(model, drop_absent=False, tau=1.0) -> ContentModel:
    return replace(model, drop_absent=drop_absent, tau=float(tau))


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


def _reliability(pairs, bins=10) -> tuple:
    """Top-label ECE over equal-width confidence bins, and the non-empty bins' table; pairs = (confidence, correct)."""
    table = []
    for b in range(bins):
        members = [(c, ok) for c, ok in pairs if min(int(c * bins), bins - 1) == b]
        if members:
            table.append({"low": b / bins, "high": (b + 1) / bins, "count": len(members),
                          "confidence": float(np.mean([c for c, _ in members])),
                          "accuracy": float(np.mean([ok for _, ok in members]))})
    total = sum(row["count"] for row in table)
    ece = sum(row["count"] / total * abs(row["accuracy"] - row["confidence"]) for row in table) if total else None
    return ece, table


def quality(prior_model, model, cases, split="val", draws=DRAWS, eig_calibration=True) -> dict:
    """Gate 2 and calibration (§1.5) over chart-derived histories, ORDERS random orders per admission:
    per-item log-loss and multiclass Brier of each charted panel's true result against the train marginal,
    top-label ECE with its reliability table, and EIG calibration (predicted against realized gain, by decile
    of predicted) on the first order. cases: (history, label, truth, patient)."""
    import math
    from scipy.special import logsumexp
    from scipy.stats import spearmanr
    marginal = {key: model.prior @ table for key, table in model.probs.items()}
    nll = nll_marginal = brier = brier_marginal = 0.0
    items = tuples = 0
    pairs, confidence = [], []
    for history, _, truth, patient in cases:
        start = prior_model.predict(history)
        charted = [panel for panel in LABS if truth.get(panel) is not None]
        for r, order in enumerate(_orders(split, patient, charted)):
            observed = []
            for panel in order:
                belief, outcome = posterior(start, model, observed), truth[panel]
                nll -= float(logsumexp(_log(belief) + model.log_lik(panel, outcome)))
                for j, value in enumerate(outcome):
                    names = model.outcomes.get((panel, j))
                    if names is None:
                        continue
                    k = names.index(value if value in names else OTHER)
                    predicted, reference, true = belief @ model.probs[(panel, j)], marginal[(panel, j)], np.eye(len(names))[k]
                    nll_marginal -= math.log(float(reference[k]))
                    brier += float(((predicted - true) ** 2).sum())
                    brier_marginal += float(((reference - true) ** 2).sum())
                    confidence.append((float(predicted.max()), int(np.argmax(predicted)) == k))
                    items += 1
                tuples += 1
                if r == 0 and eig_calibration:
                    rng = np.random.default_rng(_seed("tim-eig-calibration", split, patient, panel))
                    pairs.append((eig(model, belief, panel, rng, draws), realized_gain(model, belief, panel, outcome)))
                observed.append((panel, outcome))
    pairs.sort(key=lambda pair: pair[0])
    groups = [g for g in np.array_split(np.arange(len(pairs)), min(10, len(pairs))) if len(g)] if pairs else []
    deciles = [{"predicted": float(np.mean([pairs[i][0] for i in g])), "realized": float(np.mean([pairs[i][1] for i in g]))}
               for g in groups]
    rho = spearmanr(range(len(deciles)), [d["realized"] for d in deciles]).correlation if len(deciles) > 1 else None
    ece, reliability = _reliability(confidence)
    mean = (lambda total: total / items) if items else (lambda total: None)
    return {"tuples": tuples, "items": items, "nll": mean(nll), "nll_marginal": mean(nll_marginal),
            "skill": None if not items else 1 - nll / nll_marginal,
            "brier": mean(brier), "brier_marginal": mean(brier_marginal),
            "brier_skill": None if not items else 1 - brier / brier_marginal,
            "ece": ece, "reliability": reliability,
            "deciles": deciles, "spearman": None if rho is None or rho != rho else float(rho)}


def posterior_quality(prior_model, model, cases) -> dict:
    """The diagnosis posterior on validation (§1.5): log-loss and top-label ECE of the HPI prior alone and
    after every charted lab panel."""
    import math
    nll_prior = nll_labs = 0.0
    prior_pairs, lab_pairs = [], []
    for history, label, truth, _ in cases:
        start = prior_model.predict(history)
        after = posterior(start, model, [(panel, truth[panel]) for panel in LABS if truth.get(panel) is not None])
        k = DISEASES.index(label)
        nll_prior -= math.log(max(float(start[k]), 1e-12))
        nll_labs -= math.log(max(float(after[k]), 1e-12))
        prior_pairs.append((float(start.max()), int(np.argmax(start)) == k))
        lab_pairs.append((float(after.max()), int(np.argmax(after)) == k))
    n = len(cases)
    return {"admissions": n, "nll_prior": nll_prior / n if n else None, "nll_labs": nll_labs / n if n else None,
            "ece_prior": _reliability(prior_pairs)[0], "ece_labs": _reliability(lab_pairs)[0]}


BIN_CLASSES = ("low", "mid", "high", "token", "other", "absent")


def bin_counts(truths) -> dict:
    """Per lab panel (by public code): charted admissions and item outcomes by class; tokens are merged into
    one "token" count so no value vocabulary is published."""
    from tim.enums import TEST_CODES
    out = {}
    for panel in LABS:
        counts = dict.fromkeys(("charted", *BIN_CLASSES), 0)
        for truth in truths:
            outcome = truth.get(panel)
            if outcome is None:
                continue
            counts["charted"] += 1
            for value in outcome:
                counts["token" if value.startswith("t:") else value] += 1
        out[TEST_CODES[panel]] = counts
    return out


_WORDS = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def mentions(history, names) -> set:
    """Panels with an item named in the history within 5 words of a number (§1.6, HPI copying)."""
    words = _WORDS.findall(history.lower())
    found = set()
    for panel, labels in names.items():
        for label in labels:
            target = _WORDS.findall(label.lower())
            for i in range(len(words) - len(target) + 1):
                if target and words[i:i + len(target)] == target:
                    window = words[max(0, i - 5):i] + words[i + len(target):i + len(target) + 5]
                    if any(w[0].isdigit() for w in window):
                        found.add(panel)
    return found


def hpi_copying(prior_model, model, cases, names, split="val") -> dict:
    """Leakage audit (§1.6): validation admissions whose history names a panel item next to a number, by panel,
    and per-item log-loss for admissions with any such mention against those without."""
    from tim.enums import TEST_CODES
    flagged = [mentions(history, names) for history, *_ in cases]
    by_test = {TEST_CODES[panel]: sum(panel in found for found in flagged) for panel in LABS}
    groups = {}
    for key, keep in (("mention", True), ("no_mention", False)):
        subset = [case for case, found in zip(cases, flagged) if bool(found) == keep]
        q = quality(prior_model, model, subset, split, eig_calibration=False) if subset else {"nll": None, "items": 0}
        groups[f"nll_{key}"], groups[f"items_{key}"] = q["nll"], q["items"]
    return {"admissions_with_mention": sum(bool(found) for found in flagged), "by_test": by_test, **groups}


def premise(rows, minimum=24) -> dict:
    """Gate 1 (§2.5): validation admissions in which V3 requested at least one lab panel."""
    lab_names = {name.lower() for name in LABS}
    counts = [sum(name in lab_names for name in r["requested_tests"]) for r in rows]
    with_lab = sum(count > 0 for count in counts)
    return {"admissions_with_lab": with_lab, "lab_requests": sum(counts), "minimum": minimum,
            "passed": with_lab >= minimum}


# ---- amendment C1: the diagnosis posterior's calibration -----------------------------------------------

FIXES = ("none", "A", "A+B")  # simplest first
TAUS = (1, 1.5, 2, 3, 5, 8, 13, 20, 30, 50)
TAU_FOLDS = 5


def _terms(prior_model, model, cases) -> list:
    """Per admission: log prior, the summed log evidence of its charted panels at tau = 1, and the label."""
    base = with_fix(model, drop_absent=model.drop_absent, tau=1.0)
    return [(_log(prior_model.predict(history)),
             sum((base.log_lik(panel, truth[panel], evidence=True) for panel in LABS if truth.get(panel) is not None),
                 np.zeros(len(DISEASES))),
             DISEASES.index(label), patient) for history, label, truth, patient in cases]


def _scored(terms, taus) -> tuple:
    """Mean log-loss and top-label ECE of the posteriors softmax(log prior + evidence / tau)."""
    losses, pairs = [], []
    for (log_prior, evidence, k, _), tau in zip(terms, taus):
        z = log_prior + evidence / tau
        z = z - z.max()
        p = np.exp(z) / np.exp(z).sum()
        losses.append(-float(np.log(max(p[k], 1e-12))))
        pairs.append((float(p.max()), int(np.argmax(p)) == k))
    return float(np.mean(losses)), _reliability(pairs)[0]


def cv_tau(prior_model, model, cases, split="val", folds=TAU_FOLDS, grid=TAUS) -> dict:
    """C1 fix B: tau chosen on the other folds and scored out of fold; the stored tau is chosen on all cases."""
    terms = _terms(prior_model, model, cases)
    fold = [_seed("tim-tau-fold", split, term[3]) % folds for term in terms]

    def best(subset):
        return min(grid, key=lambda t: (_scored(subset, [t] * len(subset))[0], t))
    fold_taus, held_terms, held_taus = [], [], []
    for k in range(folds):
        chosen = best([t for t, f in zip(terms, fold) if f != k])
        fold_taus.append(chosen)
        held = [t for t, f in zip(terms, fold) if f == k]
        held_terms += held
        held_taus += [chosen] * len(held)
    nll, ece = _scored(held_terms, held_taus)
    return {"nll": nll, "ece": ece, "tau": best(terms), "fold_taus": fold_taus}


def choose_fix(rows) -> str:
    """C1's rule: lowest log-loss; candidates within 0.01 nats tie; among them the lowest ECE, and among ECEs
    within 0.005 of it the simplest."""
    best = min(rows[f]["nll"] for f in FIXES)
    tied = [f for f in FIXES if rows[f]["nll"] - best <= 0.01 + 1e-12]
    lowest = min(rows[f]["ece"] for f in tied)
    return next(f for f in tied if rows[f]["ece"] - lowest <= 0.005 + 1e-12)


def compare_fixes(prior_model, model, cases, split="val") -> dict:
    """none, A and A+B (out of fold) on validation, and the candidate C1's rule chooses."""
    def scored(candidate):
        q = posterior_quality(prior_model, candidate, cases)
        return {"nll": q["nll_labs"], "ece": q["ece_labs"]}
    rows = {"none": scored(with_fix(model)), "A": scored(with_fix(model, drop_absent=True)),
            "A+B": cv_tau(prior_model, with_fix(model, drop_absent=True), cases, split)}
    return {**rows, "chosen": choose_fix(rows)}


def fixed(model, table) -> ContentModel:
    """The model with the chosen fix, as stored for the planner."""
    return {"none": with_fix(model), "A": with_fix(model, drop_absent=True),
            "A+B": with_fix(model, drop_absent=True, tau=table["A+B"]["tau"])}[table["chosen"]]
