"""CPU statistics using the pinned upstream metric implementation.

Rows are mappings with patient_id, label, prediction, provided_tests,
requested_tests and ece_triplets. Test lists preserve repeated requests; triplets
are actual (hypothesis, confidence, label) observations, never final diagnoses
repackaged as hypotheses. Optional cluster_ids are ordered subject identifiers.
No patient rows are written by this module. Percent metrics use percentage
points for differences; ECE and hypothesis accuracy retain upstream fractions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np
from scipy.stats import binomtest, t
from src.utils.metrics import calculate_classification_metrics


CLASSES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
ARMS = ("primary", "decision_only", "no_cost", "incorrect_minus1")


def _number(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} outside its allowed range")
    return value


def expected_cost(cost, availability):
    """Expected provided cost for one request, with availability in [0, 1]."""
    return _number(cost, "cost", 0) * _number(availability, "availability", 0, 1)


def _validate(rows, cfg):
    rows = list(rows)
    if not rows:
        raise ValueError("rows must not be empty")
    costs = {k: _number(v, "test cost", 0) for k, v in cfg.evaluation.test_costs_usd.items()}
    seen = set()
    for row in rows:
        required = {"patient_id", "label", "prediction", "provided_tests", "requested_tests", "ece_triplets"}
        if not isinstance(row, Mapping) or not required <= row.keys():
            raise ValueError("each row must contain " + ", ".join(sorted(required)))
        patient_id = row["patient_id"]
        if isinstance(patient_id, bool) or not isinstance(patient_id, (str, int, np.integer)) or patient_id == "":
            raise ValueError("patient_id must be a nonempty string or integer")
        if patient_id in seen:
            raise ValueError("duplicate patient_id")
        seen.add(patient_id)
        if row["label"] not in CLASSES or row["prediction"] not in (*CLASSES, None):
            raise ValueError("invalid label or prediction")
        for key in ("provided_tests", "requested_tests"):
            if not isinstance(row[key], (list, tuple)) or any(test not in costs for test in row[key]):
                raise ValueError(f"{key} must contain configured test names")
        if len(set(row["provided_tests"])) != len(row["provided_tests"]):
            raise ValueError("upstream provided_tests are unique; repeated requests are recorded separately")
        if Counter(row["provided_tests"]) - Counter(row["requested_tests"]):
            raise ValueError("provided tests must be a sub-multiset of requested tests")
        if not isinstance(row["ece_triplets"], (list, tuple)):
            raise ValueError("ece_triplets must be a sequence")
        for triple in row["ece_triplets"]:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                raise ValueError("invalid hypothesis triplet")
            hypothesis, confidence, label = triple
            if hypothesis not in (*CLASSES, "unknown", None) or label != row["label"]:
                raise ValueError("triplet must use a valid hypothesis and the patient label")
            if confidence is not None:
                # Model-generated finite out-of-range values are observations,
                # not corrupt rows. Upstream excludes them from ECE only.
                _number(confidence, "confidence")
    return rows, costs


def _calculate(rows, costs):
    triplets = [triple for row in rows for triple in row["ece_triplets"]]
    upstream = calculate_classification_metrics(
        [r["prediction"] for r in rows], [r["label"] for r in rows], triplets, list(CLASSES)
    )
    upstream = {k: v.item() if isinstance(v, np.generic) else v for k, v in upstream.items()}
    claims = {f"T{i + 1}": 100 * upstream[f"all/recall_{label}"]
              if upstream[f"all/support_{label}"] else None for i, label in enumerate(CLASSES)}
    claims["T5"] = float(np.mean(list(claims.values()))) if all(v is not None for v in claims.values()) else None
    claims["T6"] = 100 * upstream["all/micro_f1"]
    claims["T7"] = 100 * upstream["all/macro_f1"]
    provided = [sum(costs[test] for test in r["provided_tests"]) for r in rows]
    requested = [sum(costs[test] for test in r["requested_tests"]) for r in rows]
    missing = [sum(test not in set(r["provided_tests"]) for test in r["requested_tests"]) for r in rows]
    unavailable_cost = [sum(costs[test] for test in r["requested_tests"] if test not in set(r["provided_tests"])) for r in rows]
    repeated_cost = [requested_cost - provided_cost - missing_cost
                     for requested_cost, provided_cost, missing_cost in zip(requested, provided, unavailable_cost)]
    claims["T8"] = float(np.mean(provided))
    eligible = sum(h not in (None, "unknown") and p is not None and 0 <= p <= 1 for h, p, _ in triplets)
    return {
        "n": len(rows), "upstream": upstream, "claims": claims,
        "pooled_accuracy": 100 * upstream["all/accuracy"],
        "hypothesis": {
            "triplet_count": len(triplets), "ece_triplet_count": eligible,
            "accuracy": upstream["hypothesis_accuracy"] if triplets else None,
            "ece": upstream["ece"] if eligible else None,
            "unknown_fraction": upstream["unknown_hypothesis_fraction"] if triplets else None,
            "invalid_confidence_count": sum(p is not None and not 0 <= p <= 1 for _, p, _ in triplets),
        },
        "cost": {
            "provided_mean_usd": claims["T8"], "provided_total_usd": float(sum(provided)),
            "requested_mean_usd": float(np.mean(requested)), "requested_total_usd": float(sum(requested)),
            "unavailable_requested_mean_usd": float(np.mean(unavailable_cost)),
            "repeated_available_request_mean_usd": float(np.mean(repeated_cost)),
            "missing_count": sum(missing), "missing_mean": float(np.mean(missing)),
            "provided_count": sum(len(r["provided_tests"]) for r in rows),
            "requested_count": sum(len(r["requested_tests"]) for r in rows),
        },
    }


def summary(rows, cfg):
    """Return JSON-ready upstream fractions, T1–T8 claims, and cost decomposition.

    T1–T7 are percentages, T8 USD per admission, pooled_accuracy a percentage.
    Missing class recalls and their T5 mean are None, rather than implied zeros.
    Raw upstream empty-triplet zeros remain in upstream; hypothesis marks them
    None so an absent hypothesis agent cannot appear perfectly calibrated.
    """
    rows, costs = _validate(rows, cfg)
    return _calculate(rows, costs)


def _estimates(result):
    return {**result["claims"], "pooled_accuracy": result["pooled_accuracy"],
            "hypothesis_accuracy": result["hypothesis"]["accuracy"], "ece": result["hypothesis"]["ece"],
            "requested_cost_usd": result["cost"]["requested_mean_usd"],
            "missing_count_mean": result["cost"]["missing_mean"]}


def _resampler(rows, cluster_ids, rng):
    if set(r["label"] for r in rows) != set(CLASSES):
        raise ValueError("bootstrap requires all four diagnosis classes")
    if cluster_ids is None:
        groups = [np.array([i for i, r in enumerate(rows) if r["label"] == label]) for label in CLASSES]
        return lambda: np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups]), "class-stratified patient percentile bootstrap"
    if len(cluster_ids) != len(rows):
        raise ValueError("cluster_ids must align with patient rows")
    clusters = {}
    for i, subject in enumerate(cluster_ids):
        if subject is None or isinstance(subject, bool) or not isinstance(subject, (str, int, np.integer)) or subject == "":
            raise ValueError("cluster ids must be nonmissing strings or integers")
        clusters.setdefault(subject, []).append(i)
    strata = {}
    for indices in clusters.values():
        profile = tuple(label for label in CLASSES if any(rows[i]["label"] == label for i in indices))
        strata.setdefault(profile, []).append(indices)

    def draw():
        return np.array([i for groups in strata.values()
                         for j in rng.integers(len(groups), size=len(groups)) for i in groups[j]])

    return draw, "subject-cluster percentile bootstrap stratified by class-presence profile"


def _bootstrap(rows_a, rows_b, cfg, seed, n_resamples, cluster_ids):
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer >= 2")
    rows_a, costs = _validate(rows_a, cfg)
    if rows_b is not None:
        rows_b, _ = _validate(rows_b, cfg)
        if [(r["patient_id"], r["label"]) for r in rows_a] != [(r["patient_id"], r["label"]) for r in rows_b]:
            raise ValueError("paired rows must have exactly aligned patient ids and labels")
    rng = np.random.default_rng(seed)
    draw, method = _resampler(rows_a, cluster_ids, rng)

    def estimate(indices):
        a = _estimates(_calculate([rows_a[i] for i in indices], costs))
        if rows_b is None:
            return a
        b = _estimates(_calculate([rows_b[i] for i in indices], costs))
        return {k: a[k] - b[k] if a[k] is not None and b[k] is not None else None for k in a}

    point = estimate(range(len(rows_a)))
    samples = {k: [] for k in point}
    for _ in range(n_resamples):
        for key, value in estimate(draw()).items():
            if value is not None:
                samples[key].append(value)
    metrics = {}
    for key, value in point.items():
        complete = value is not None and len(samples[key]) == n_resamples
        metrics[key] = {
            "estimate": value, "ci95": [float(v) for v in np.quantile(samples[key], [0.025, 0.975])] if complete else None,
            "valid_resamples": len(samples[key]), "status": "assessed" if complete else "unassessable",
        }
    result = {"n": len(rows_a), "seed": seed, "n_resamples": n_resamples,
              "method": method, "direction": "a_minus_b" if rows_b is not None else "single_arm",
              "metrics": metrics, "uncertainty": "held-out sampling, conditional on fitted model; not training-seed uncertainty"}
    if cluster_ids is not None:
        result["n_clusters"] = len(set(cluster_ids))
    if rows_b is not None:
        result["mcnemar"] = exact_mcnemar([r["prediction"] == r["label"] for r in rows_a],
                                           [r["prediction"] == r["label"] for r in rows_b])
        if cluster_ids is not None and len(set(cluster_ids)) < len(rows_a):
            result["mcnemar"].update(p_value=None, status="unassessable",
                                     reason="repeated subjects violate independent admission-pair assumption; use cluster intervals")
    return result


def paired_bootstrap(rows_a, rows_b, cfg, seed=269, n_resamples=10000, cluster_ids=None):
    """Paired a-minus-b percentile intervals, preserving identical patient order.

    With subject IDs, subjects are resampled within their class-presence profile;
    every selected subject contributes all admissions. This conditions on profile
    composition and may be degenerate for small strata. Without IDs, resampling
    assumes independent admissions within classes. Undefined calibration samples
    make that interval unassessable instead of silently discarding those draws.
    """
    return _bootstrap(rows_a, rows_b, cfg, seed, n_resamples, cluster_ids)


def single_arm_intervals(rows, cfg, seed=269, n_resamples=10000, cluster_ids=None):
    """Single-arm intervals under the same sampling assumptions as paired_bootstrap."""
    return _bootstrap(rows, None, cfg, seed, n_resamples, cluster_ids)


def exact_mcnemar(correct_a, correct_b):
    """Two-sided exact binomial test on independent paired correctness outcomes."""
    a, b = list(correct_a), list(correct_b)
    if not a or len(a) != len(b) or any(not isinstance(x, (bool, np.bool_)) for x in a + b):
        raise ValueError("correctness inputs must be nonempty aligned boolean sequences")
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    discordant = a_only + b_only
    return {"a_only_correct": a_only, "b_only_correct": b_only, "discordant": discordant,
            "p_value": float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0,
            "method": "two-sided exact McNemar binomial", "metric": "pooled_accuracy"}


def seed_difference_ci(differences):
    """Student-t CI across all declared paired training-seed differences.

    None denotes an unfinished/failed declared seed and makes the interval
    unassessable. One seed never establishes between-training-run uncertainty.
    """
    differences = list(differences)
    valid = [_number(v, "seed difference") for v in differences if v is not None]
    result = {"n": len(valid), "declared_n": len(differences), "estimate": float(np.mean(valid)) if valid else None,
              "ci95": None, "status": "unassessable", "method": "paired training-seed Student t"}
    if len(valid) >= 2 and len(valid) == len(differences):
        sem = float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
        margin = float(t.ppf(0.975, len(valid) - 1)) * sem
        result.update(ci95=[result["estimate"] - margin, result["estimate"] + margin], status="assessed", df=len(valid) - 1)
    return result


def holm_adjust(pvalues, alpha=0.05):
    """Holm adjustment over an explicitly planned comparison/metric family.

    Map planned test names to exact/paired-test p-values, or None for unrun tests.
    Unrun members remain in the family size and cannot be declared significant.
    Do not derive p-values from independent per-arm standard deviations.
    """
    alpha = _number(alpha, "alpha", 0, 1)
    valid = sorted(((_number(p, "p_value", 0, 1), name) for name, p in pvalues.items() if p is not None))
    result = {name: {"p_value": p, "holm_p_value": None, "reject": False} for name, p in pvalues.items()}
    previous = 0.0
    for i, (p, name) in enumerate(valid):
        previous = min(1.0, max(previous, (len(pvalues) - i) * p))
        result[name].update(holm_p_value=previous, reject=previous <= alpha)
    return result


def upper_bound_invariant(rewards, cfg, arm="primary"):
    """Check only an episode reward ceiling for the four configured arms.

    Diagnosis and format maxima bound their sum when all cost contributions
    are nonpositive. This asserts no inter-arm accuracy ordering and creates
    no oracle baseline. Unsupported reward types/positive costs are rejected.
    """
    if arm not in ARMS:
        raise ValueError("no justified reward bound for this arm")
    ceiling = 0.0
    targets = set()
    for rule in cfg.reward_function.reward_functions:
        kind = rule["_target_"].rsplit(".", 1)[-1]
        if kind in targets:
            raise ValueError("duplicate reward component")
        targets.add(kind)
        if kind == "DiagnosisReward":
            ceiling += max(_number(rule[k], k) for k in ("correct_reward", "incorrect_reward", "undiagnosed_reward"))
        elif kind == "FormatReward":
            ceiling += max(_number(rule[k], k) for k in ("valid_reward", "invalid_reward"))
        elif kind == "TestCostReward":
            values = [rule.test_cost] if rule.single_cost else list(rule.test_cost.values())
            if any(_number(v, "cost reward") > 0 for v in values):
                raise ValueError("positive cost reward invalidates this bound")
        else:
            raise ValueError("unsupported reward component")
    if not {"DiagnosisReward", "FormatReward"} <= targets:
        raise ValueError("diagnosis and format rewards required")
    rewards = [_number(v, "episode reward") for v in rewards]
    if not rewards:
        raise ValueError("episode rewards must not be empty")
    return {"arm": arm, "upper_bound": ceiling, "holds": all(v <= ceiling + 1e-12 for v in rewards),
            "scope": "episode-total diagnosis + format + nonpositive provided-test cost"}
