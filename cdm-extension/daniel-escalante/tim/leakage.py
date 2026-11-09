"""Leakage probes (SPEC 4.7): how much the diagnosis is predictable from HPI text, with and
without the 12 availability bits, under the learned model's frozen fitting protocol.
A weak probe cannot certify that leakage is absent."""
import numpy as np

from tim.availability import C_GRID, FOLD_SEED, _vectorizer, bits
from tim.enums import DISEASES

PROBES = ("hpi", "hpi_bits", "bits")


def _features(items, tests, probe, vectorizer):
    from scipy.sparse import csr_matrix, hstack
    parts = []
    if probe != "bits":
        parts.append(vectorizer.transform([i["patient_history"] for i in items]))
    if probe != "hpi":
        parts.append(csr_matrix(np.array([[bits(i, tests)[t] for t in tests] for i in items], dtype=float)))
    return hstack(parts).tocsr()


def _fit(items, tests, probe, c):
    from sklearn.linear_model import LogisticRegression
    vectorizer = _vectorizer().fit([i["patient_history"] for i in items]) if probe != "bits" else None
    model = LogisticRegression(C=c, max_iter=2000).fit(_features(items, tests, probe, vectorizer),
                                                         [i["condition"] for i in items])
    return vectorizer, model


def _choose_c(items, tests, probe) -> float:
    from sklearn.model_selection import KFold
    losses = {}
    for c in C_GRID:
        total = 0.0
        for train, held in KFold(5, shuffle=True, random_state=FOLD_SEED).split(items):
            vectorizer, model = _fit([items[i] for i in train], tests, probe, c)
            held_items = [items[i] for i in held]
            proba = model.predict_proba(_features(held_items, tests, probe, vectorizer))
            column = {label: j for j, label in enumerate(model.classes_)}
            total -= sum(np.log(max(proba[k, column[i["condition"]]], 1e-12)) if i["condition"] in column
                         else np.log(1e-12) for k, i in enumerate(held_items))
        losses[c] = total
    return min(C_GRID, key=lambda c: (losses[c], c))


def probes(train, evaluation, tests) -> dict:
    """Fit each probe on train, report mean class recall (T5) and per-class recall on evaluation."""
    report = {}
    for probe in PROBES:
        c = _choose_c(train, tests, probe)
        vectorizer, model = _fit(train, tests, probe, c)
        predicted = model.predict(_features(evaluation, tests, probe, vectorizer))
        recall = {}
        for disease in DISEASES:
            members = [p for p, i in zip(predicted, evaluation) if i["condition"] == disease]
            recall[disease] = 100 * float(np.mean([p == disease for p in members])) if members else None
        present = [r for r in recall.values() if r is not None]
        report[probe] = {"C": c, "T5": float(np.mean(present)), "recall": recall}
    return report
