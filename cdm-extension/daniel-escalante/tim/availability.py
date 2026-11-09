"""Chart-availability prediction (SPEC §4.1–4.3, 4.6).

y(p, x) = 1 when the chart holds a result for test x (upstream's retrieval rule exactly). The
predictors estimate p(test present | HPI); none of them predicts result content. Fitted models
are patient-derived and stay private.
"""
import hashlib
import math

import numpy as np

EPS = 0.001
SENTINEL = "not available.\n"
C_GRID = (0.01, 0.1, 1, 10)
MIN_CLASS = 10
FOLD_SEED = 0


def clip(p: float) -> float:
    return min(max(float(p), EPS), 1 - EPS)


def bits(item, tests) -> dict:
    """The only view of test results Phase B takes: availability bits, never result text."""
    return {t: int(item["test_results"][t] != SENTINEL) for t in tests}


def base_rates(items, tests) -> dict:
    return {t: float(np.mean([bits(i, tests)[t] for i in items])) for t in tests}


# ---- controlled error (SPEC 4.3) --------------------------------------------------------

def mask_draw(seed, split, patient, test) -> float:
    digest = hashlib.sha256(f"tim-mask|{seed}|{split}|{patient}|{test}".encode()).hexdigest()
    return int(digest[:16], 16) / 16 ** 16


def flipped(seed, split, patient, test, alpha) -> bool:
    return mask_draw(seed, split, patient, test) < alpha


def corrupted(y, pi, alpha, temperature, seed, split, patient, test) -> float:
    """Nominal posterior after flipping y with probability alpha, sharpened by 1/temperature."""
    observed = 1 - y if flipped(seed, split, patient, test, alpha) else y
    if observed == 1:
        q = (1 - alpha) * pi / ((1 - alpha) * pi + alpha * (1 - pi))
    else:
        q = alpha * pi / (alpha * pi + (1 - alpha) * (1 - pi))
    q = clip(q)
    return clip(1 / (1 + math.exp(-math.log(q / (1 - q)) / temperature)))


# ---- learned model (SPEC 4.2) -----------------------------------------------------------

def _vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True)


def _matrix(items, tests):
    return [i["patient_history"] for i in items], np.array([[bits(i, tests)[t] for t in tests] for i in items])


def _rare(column) -> bool:
    positives = int(column.sum())
    return positives < MIN_CLASS or len(column) - positives < MIN_CLASS


class LearnedModel:
    def __init__(self, vectorizer, models: dict):
        self.vectorizer, self.models = vectorizer, models

    @property
    def kinds(self) -> dict:
        return {t: kind for t, (kind, _) in self.models.items()}

    def predict(self, history: str) -> dict:
        x = self.vectorizer.transform([history])
        return {t: clip(payload if kind == "base_rate" else payload.predict_proba(x)[0, 1])
                for t, (kind, payload) in self.models.items()}


def _fit(histories, y, tests, c):
    from sklearn.linear_model import LogisticRegression
    vectorizer = _vectorizer()
    x = vectorizer.fit_transform(histories)
    models = {}
    for j, test in enumerate(tests):
        if _rare(y[:, j]):
            models[test] = ("base_rate", float(y[:, j].mean()))
        else:
            models[test] = ("logistic", LogisticRegression(C=c, penalty="l2", max_iter=2000).fit(x, y[:, j]))
    return LearnedModel(vectorizer, models)


def choose_c(items, tests) -> float:
    """One C for every test: lowest summed 5-fold CV log-loss on train (fold seed 0)."""
    from sklearn.model_selection import KFold
    histories, y = _matrix(items, tests)
    losses = {}
    for c in C_GRID:
        total = 0.0
        for train, held in KFold(5, shuffle=True, random_state=FOLD_SEED).split(histories):
            model = _fit([histories[i] for i in train], y[train], tests, c)
            for i in held:
                p = model.predict(histories[i])
                total -= sum(math.log(p[t]) if y[i, j] else math.log(1 - p[t]) for j, t in enumerate(tests))
        losses[c] = total
    return min(C_GRID, key=lambda c: (losses[c], c))


def fit_learned(items, tests, seed, c) -> LearnedModel:
    """Fit on a bootstrap resample of the training admissions drawn with this seed."""
    histories, y = _matrix(items, tests)
    sample = np.random.default_rng(seed).integers(0, len(histories), size=len(histories))
    return _fit([histories[i] for i in sample], y[sample], tests, c)


# ---- quality (SPEC 4.6) -----------------------------------------------------------------

def quality(y, p, base_rate) -> dict:
    """ECE (10 equal-width bins), AUROC, Brier and Brier skill against the marginal base rate."""
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean()) for b in range(10) if (bins == b).any())
    brier = float(np.mean((p - y) ** 2))
    reference = float(np.mean((np.asarray(base_rate, dtype=float) - y) ** 2))
    return {"n": int(len(y)), "positives": int(y.sum()), "ece": float(ece),
            "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "brier": brier, "brier_skill": 1 - brier / reference if reference > 0 else None}
