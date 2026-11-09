"""Statistics of SPEC §6 and the Phase-A descriptives, from private rows to aggregate values."""
from collections import Counter

import numpy as np

from tim.enums import DISEASES
from tim.fallback import finalize

ALPHA = 0.05
TOLERANCE = 1e-9  # ties count as extreme


def weights(labels) -> list[float]:
    counts = Counter(labels)
    return [100 / (4 * counts[label]) for label in labels]


def paired(rows_a, rows_b):
    """Per-patient weights w_i and differences d_i = correct_a - correct_b, in a shared order."""
    if [(r["patient_id"], r["label"]) for r in rows_a] != [(r["patient_id"], r["label"]) for r in rows_b]:
        raise ValueError("paired rows must list the same patients in the same order")
    w = weights([r["label"] for r in rows_a])
    d = [int(a["prediction"] == a["label"]) - int(b["prediction"] == b["label"]) for a, b in zip(rows_a, rows_b)]
    return w, d


def delta_t5(w, d) -> float:
    return float(sum(wi * di for wi, di in zip(w, d)))


def randomization_test(w, d, seed, draws=100_000) -> dict:
    """Two-sided paired sign-flip test of S = sum w_i d_i: exact up to 20 discordant, else draws."""
    v = np.array([wi * di for wi, di in zip(w, d) if di], dtype=float)
    k, observed = len(v), abs(delta_t5(w, d))
    if k <= 20:
        extreme = 0
        for start in range(0, 2 ** k, 1 << 16):
            codes = np.arange(start, min(start + (1 << 16), 2 ** k))
            signs = ((codes[:, None] >> np.arange(k)) & 1).astype(np.int8) * 2 - 1
            extreme += int(np.sum(np.abs(signs @ v) >= observed - TOLERANCE))
        return {"statistic": delta_t5(w, d), "p_value": extreme / 2 ** k, "discordant": k,
                "method": "exact", "draws": None}
    rng, extreme, done = np.random.default_rng(seed), 0, 0
    while done < draws:
        n = min(10_000, draws - done)
        signs = rng.integers(0, 2, size=(n, k), dtype=np.int8) * 2 - 1
        extreme += int(np.sum(np.abs(signs @ v) >= observed - TOLERANCE))
        done += n
    return {"statistic": delta_t5(w, d), "p_value": (1 + extreme) / (draws + 1), "discordant": k,
            "method": "draws", "draws": draws}


def _distribution(groups):
    values, probs = np.zeros(1), np.ones(1)
    for k, w in groups:
        from scipy.stats import binom
        b = np.arange(k + 1)
        values = (values[:, None] + w * (2 * b - k)[None, :]).ravel()
        probs = (probs[:, None] * binom.pmf(b, k, 0.5)[None, :]).ravel()
    return values, probs


def _exact_p(groups, statistic) -> float:
    """Exact sign-flip p-value when weights take one value per class (k discordant of weight w)."""
    t = abs(statistic) - TOLERANCE
    if t <= 0:
        return 1.0
    half = len(groups) // 2
    a_values, a_probs = _distribution(groups[:half])
    b_values, b_probs = _distribution(groups[half:])
    order = np.argsort(b_values)
    b_values, cumulative = b_values[order], np.concatenate([[0.0], np.cumsum(b_probs[order])])
    upper = cumulative[-1] - cumulative[np.searchsorted(b_values, t - a_values, side="left")]
    lower = cumulative[np.searchsorted(b_values, -t - a_values, side="right")]
    return float(np.sum(a_probs * (upper + lower)))


def exact_p_by_class(w, d) -> float:
    counts = Counter(round(wi, 12) for wi, di in zip(w, d) if di)
    return _exact_p([(k, weight) for weight, k in counts.items()], delta_t5(w, d))


def bootstrap(rows_a, rows_b, cfg, seed, n_resamples=10_000) -> dict:
    """The replication's class-stratified paired percentile bootstrap, T5 only."""
    from replication.stats import paired_bootstrap
    result = paired_bootstrap(rows_a, rows_b, cfg, seed=seed, n_resamples=n_resamples)["metrics"]["T5"]
    w, d = paired(rows_a, rows_b)
    if abs(result["estimate"] - delta_t5(w, d)) > TOLERANCE:
        raise AssertionError("bootstrap point estimate differs from sum w_i d_i")
    return {"estimate": result["estimate"], "ci95": result["ci95"]}


def significance(p_value, ci95) -> str:
    if p_value < ALPHA and not ci95[0] <= 0 <= ci95[1]:
        return "statistically significant"
    return "not statistically significant"


def v1_rows(v0_rows) -> list[dict]:
    """V1: the V0 rows with G3 applied on CPU; a non-null prediction never changes."""
    out = []
    for row in v0_rows:
        finalized, source = finalize(row["raw_prediction"], row["hypotheses"], True)
        if row["raw_prediction"] is not None and finalized != row["raw_prediction"]:
            raise AssertionError("V1 changed a non-null prediction")
        out.append({**row, "variant": "V1", "prediction": finalized, "finalized_prediction": finalized,
                    "prediction_source": source})
    return out


def analytic_controls(comparator, majority) -> dict:
    """Expected T5 gains from filling the comparator's nonanswers (references, not results)."""
    w = weights([r["label"] for r in comparator])
    missing = [(wi, r["label"]) for wi, r in zip(w, comparator) if r["prediction"] is None]
    return {"uniform_random": sum(wi for wi, _ in missing) / 4,
            "training_majority": sum(wi for wi, label in missing if label == majority),
            "perfect_recovery": sum(wi for wi, _ in missing)}


def recovery(method, comparator) -> dict:
    gaps = [(m, c) for m, c in zip(method, comparator) if c["prediction"] is None]
    answered = [m for m, _ in gaps if m["prediction"] is not None]
    correct = sum(m["prediction"] == m["label"] for m in answered)
    regressions = sum(c["prediction"] == c["label"] and m["prediction"] != m["label"] for m, c in zip(method, comparator))
    return {"baseline_nonanswers": len(gaps), "newly_answered": len(answered), "correct_recoveries": correct,
            "recovery_yield": correct / len(gaps) if gaps else None,
            "newly_answered_accuracy": correct / len(answered) if answered else None,
            "regressions": regressions}


def power(probabilities, sizes, sims, seed) -> float:
    """Share of simulated primary tests with p < 0.05; probabilities[c] = (improve, regress)."""
    rng = np.random.default_rng(seed)
    draws = {c: rng.multinomial(sizes[c], [p, q, 1 - p - q], size=sims) for c, (p, q) in probabilities.items()}
    w = {c: 100 / (4 * sizes[c]) for c in sizes}
    rejected = 0
    for s in range(sims):
        groups = [(int(draws[c][s, 0] + draws[c][s, 1]), w[c]) for c in DISEASES]
        statistic = sum(w[c] * int(draws[c][s, 0] - draws[c][s, 1]) for c in DISEASES)
        rejected += _exact_p(groups, statistic) < ALPHA
    return rejected / sims


def minimum_detectable_effect(discordance, sizes, sims, seed, target=0.8):
    """Smallest expected Delta-T5 with power >= target at fixed per-class discordance (bisection)."""
    def at(theta):
        return {c: (discordance[c] * (1 + theta) / 2, discordance[c] * (1 - theta) / 2) for c in DISEASES}
    if power(at(1.0), sizes, sims, seed) < target:
        return None
    low, high = 0.0, 1.0
    for _ in range(14):
        middle = (low + high) / 2
        low, high = (low, middle) if power(at(middle), sizes, sims, seed) >= target else (middle, high)
    return 25 * high * sum(discordance[c] for c in DISEASES)
