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
        raw_prediction = row.get("raw_prediction", row["prediction"])  # official rows hold only the raw one
        finalized, source = finalize(raw_prediction, row["hypotheses"], True)
        if raw_prediction is not None and finalized != raw_prediction:
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


# ---- stage-level aggregates -------------------------------------------------------------

def _counts(values, names) -> dict:
    found = Counter(values)
    return {name: found[name] for name in names}


def summarize(rows, cfg) -> dict:
    """The public summary block for one set of rows (official or tim)."""
    from replication.stats import confusion, summary
    from tim.enums import GUARD_EVENTS, PREDICTION_SOURCES, TERMINATIONS
    result = summary(rows, cfg)
    unavailable = sum(len(r.get("unavailable_requests", ())) for r in rows)
    requested, provided = result["cost"]["requested_count"], result["cost"]["provided_count"]
    block = {
        "n": result["n"],
        "metrics": {f"T{i}": result["claims"][f"T{i}"] for i in range(1, 9)},
        "nonanswers": sum(r["prediction"] is None for r in rows),
        "nonanswers_by_class": _counts([r["label"] for r in rows if r["prediction"] is None], DISEASES),
        "requests": {"requested": requested, "provided": provided, "unavailable": unavailable,
                     "repeated": requested - provided - unavailable},
        "confusion": confusion(rows),
    }
    if all("termination" in r for r in rows):
        block["termination"] = _counts([r["termination"] for r in rows], TERMINATIONS)
        block["events"] = _counts([e["type"] for r in rows for e in r["events"]], GUARD_EVENTS)
        block["prediction_source"] = _counts([r["prediction_source"] for r in rows], PREDICTION_SOURCES)
        block["scorer_errors"] = sum(r["scorer_error_count"] for r in rows)
        block["replacement_failed"] = sum(r["replacement_failed_count"] for r in rows)
    return block


def raw(rows) -> list[dict]:
    """The online guards without G3: the raw upstream prediction."""
    return [{**r, "prediction": r["raw_prediction"]} for r in rows]


def compare(rows_a, rows_b, cfg, name, split, tested=True, resamples=10_000) -> dict:
    """Delta-T5 with its bootstrap interval; tested comparisons add the randomization test."""
    from replication.config import stream_seed
    from replication.stats import exact_mcnemar
    w, d = paired(rows_a, rows_b)
    interval = bootstrap(rows_a, rows_b, cfg, seed=stream_seed(269, split, "tim-bootstrap"), n_resamples=resamples)
    test = randomization_test(w, d, seed=stream_seed(269, split, f"tim-permutation:{name}")) if tested else None
    p = test["p_value"] if test else None
    mcnemar = exact_mcnemar([a["prediction"] == a["label"] for a in rows_a], [b["prediction"] == b["label"] for b in rows_b])
    return {"delta_t5": interval["estimate"], "ci95": interval["ci95"], "p_value": p, "p_holm": None,
            "method": test["method"] if test else None, "discordant": sum(1 for x in d if x),
            "significance": significance(p if p is not None else 1.0, interval["ci95"]),
            "mcnemar_p": mcnemar["p_value"]}


def holm(family: dict) -> dict:
    """Holm-adjust a family of tested comparisons in place and restate significance."""
    from replication.stats import holm_adjust
    adjusted = holm_adjust({name: c["p_value"] for name, c in family.items()})
    for name, c in family.items():
        c["p_holm"] = adjusted[name]["holm_p_value"]
        c["significance"] = significance(c["p_holm"], c["ci95"])
    return family


def changes(method, comparator) -> dict:
    """Per-disease improved / regressed / unchanged counts of method versus comparator."""
    out = {c: {"improved": 0, "regressed": 0, "unchanged": 0} for c in DISEASES}
    for m, c in zip(method, comparator):
        delta = int(m["prediction"] == m["label"]) - int(c["prediction"] == c["label"])
        out[m["label"]]["improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"] += 1
    return out


def sensitivity(change_counts, sizes, sims, seed) -> dict:
    """Power at the validation effect and the 80%-power MDE at the validation discordance."""
    probabilities, discordance = {}, {}
    for c in DISEASES:
        n = sum(change_counts[c].values())
        probabilities[c] = (change_counts[c]["improved"] / n, change_counts[c]["regressed"] / n)
        discordance[c] = sum(probabilities[c])
    effect = sum(25 * (p - q) for p, q in probabilities.values())
    mde = minimum_detectable_effect(discordance, sizes, sims, seed) if any(discordance.values()) else None
    return {"val_effect": effect, "discordance": discordance,
            "power_at_val_effect": power(probabilities, sizes, sims, seed), "mde_80": mde}


def validation(rows, cfg, source, test_sizes, sims=10_000, minimum=10, resamples=10_000) -> dict:
    """analyze-val-a: shadow control, premise check, the validation ladder and sensitivity inputs."""
    from replication.config import stream_seed
    v0, shadow = rows["V0"], rows["V0-shadow"]
    identical = sum(a["patient_id"] == b["patient_id"] and a["generation_digest"] == b["generation_digest"]
                    for a, b in zip(v0, shadow))
    shadow_block = {"identical": identical, "total": len(v0), "scorer_errors": sum(r["scorer_error_count"] for r in shadow),
                    "passed": identical == len(v0) == len(shadow)}
    nonanswers = sum(r["prediction"] is None for r in v0)
    premise = {"v0_nonanswers": nonanswers, "minimum": minimum, "passed": nonanswers >= minimum}
    v1 = v1_rows(v0)
    ladder = {"V0": v0, "V0-shadow": shadow, "V1": v1, "V2-raw": raw(rows["V2"]), "V2": rows["V2"],
              "V3-raw": raw(rows["V3"]), "V3": rows["V3"]}
    pairs = {"V1-V0": (v1, v0), "V2-V0": (rows["V2"], v0), "V3-V0": (rows["V3"], v0)}
    change_counts = {name: changes(a, b) for name, (a, b) in pairs.items()}
    seed = stream_seed(269, "val", "tim-sensitivity")
    return {
        "source": source, "shadow": shadow_block, "premise": premise, "go": shadow_block["passed"] and premise["passed"],
        "variants": {name: summarize(r, cfg) for name, r in ladder.items()},
        "changes": change_counts,
        "descriptive": {name: compare(a, b, cfg, name, "val", tested=False, resamples=resamples)
                        for name, (a, b) in pairs.items()},
        "sensitivity": {name: sensitivity(change_counts[name], test_sizes, sims, seed) for name in ("V1-V0", "V3-V0")},
    }


def metrics(rows, official, cfg, source, majority, sensitivity_block, t5_reference, resamples=10_000) -> dict:
    """analyze-a: the Phase-A metrics.json (SPEC 3.5, 5.6, 6)."""
    from functools import partial
    compare_ = partial(compare, resamples=resamples)
    rerun, v1 = rows["V0-rerun"], v1_rows(official)
    ladder = {"official": official, "V0-rerun": rerun, "V1": v1, "V2-raw": raw(rows["V2"]), "V2": rows["V2"],
              "V3-raw": raw(rows["V3"]), "V3": rows["V3"]}
    identical = sum(a["patient_id"] == b["patient_id"] and a["prediction"] == b["prediction"] for a, b in zip(rerun, official))
    qualified = not (identical == len(official) == len(rerun))
    family = holm({"V1-official": compare_(v1, official, cfg, "V1-official", "test"),
                   "V3-V1": compare_(rows["V3"], v1, cfg, "V3-V1", "test")})
    descriptive = {
        "V2-V1": compare_(rows["V2"], v1, cfg, "V2-V1", "test", tested=False),
        "V3-V2": compare_(rows["V3"], rows["V2"], cfg, "V3-V2", "test", tested=False),
        "V2raw-official": compare_(ladder["V2-raw"], official, cfg, "V2raw-official", "test", tested=False),
        "V3raw-official": compare_(ladder["V3-raw"], official, cfg, "V3raw-official", "test", tested=False),
    }
    second = {"V3-V0rerun": compare_(rows["V3"], rerun, cfg, "V3-V0rerun", "test")} if qualified else {}
    return {
        "source": source, "t5_reference": t5_reference,
        "variants": {name: summarize(r, cfg) for name, r in ladder.items()},
        "v0_rerun": {"identical": identical, "total": len(official), "qualified": qualified},
        "controls": analytic_controls(official, majority),
        "recovery": {name: recovery(ladder[name], official) for name in ("V1", "V2-raw", "V2", "V3-raw", "V3")},
        "primary": compare_(rows["V3"], official, cfg, "V3-official", "test"),
        "family_a": family, "descriptive": descriptive, "second_comparator": second,
        "sensitivity": sensitivity_block,
    }
