"""The audit card of SPEC 9.1: one record per decision point, a summary, and its markdown rendering."""
from __future__ import annotations

import math

import numpy as np

from .core import FLOOR, MAX_ITER, TOL, project
from .decision import coherent_voi, identity_utility
from .identity import realized_gain
from .measures import drift, responsiveness
from .scaling import VARIANT_BOUNDS, tau_budget, tau_ratio

REQUIRED = ("id", "classes", "prior", "outcomes", "weights", "posteriors")
DISPLAY_ZERO = 1e-12    # the markdown card shows smaller magnitudes as 0; the JSON keeps the number


def _position(sequence, item, what):
    if item not in sequence:
        raise ValueError(f"{what} {item!r} is not among {list(sequence)!r}")
    return sequence.index(item)


def audit_record(point, tol=TOL, max_iter=MAX_ITER):
    """Audit one decision point (SPEC 9.1; quantities of SPEC 5.2 to 5.6).

    point is a mapping with id, classes, prior, outcomes, weights, posteriors (one row per
    outcome) and optionally budget, real_outcome, y_true (a class name) and utility (one row per
    diagnosis, one column per class; 1[correct] when absent). Returns a flat record of plain
    numbers: drift (L1 norm and largest component), responsiveness, KL_inc and its convergence,
    tau_B under the shrink-only interval [1, 16] and the two-sided interval [1/8, 16] when a
    budget is given (the two-sided value is the regime-map diagnostic), tau_ratio, coherent VOI,
    and the realized gain when the real outcome and the label are given. When the projection does
    not converge the raw reader is used (SPEC 5.3), so the coherent VOI and the realized gain are
    reported as None.
    """
    missing = [key for key in REQUIRED if key not in point]
    if missing:
        raise ValueError(f"decision point is missing {missing}")
    classes, outcomes = list(point["classes"]), list(point["outcomes"])
    prior, weights, posteriors = point["prior"], point["weights"], point["posteriors"]
    projection = project(prior, weights, posteriors, tol, max_iter)
    if len(classes) != projection.prior.size or len(outcomes) != projection.weights.size:
        raise ValueError("classes must match the prior, and outcomes the weights")
    d = drift(prior, weights, posteriors)
    record = {
        "id": point["id"],
        "n_classes": len(classes),
        "n_outcomes": len(outcomes),
        "drift_l1": float(np.abs(d).sum()),
        "drift_max": float(np.abs(d).max()),
        "responsiveness": responsiveness(weights, posteriors),
        "kl_inc": projection.kl_inc,
        "converged": projection.converged,
        "iterations": projection.iterations,
        "max_violation": projection.max_violation,
        "budget": None,
        "tau_b_shrink_only": None,
        "tau_b_shrink_only_path": None,
        "tau_b_two_sided": None,
        "tau_b_two_sided_path": None,
        "tau_ratio": None,
        "utility": "supplied" if point.get("utility") is not None else "identity",
        "coherent_voi": None,
        "realized_gain": None,
    }
    if point.get("budget") is not None:
        record["budget"] = float(point["budget"])
        for variant, bounds in VARIANT_BOUNDS.items():
            found = tau_budget(prior, weights, posteriors, record["budget"], bounds, tol=tol, max_iter=max_iter)
            key = "tau_b_" + variant.replace("-", "_")
            record[key], record[key + "_path"] = found.tau, found.path
    ratio = tau_ratio(prior, weights, posteriors, tol=tol, max_iter=max_iter).tau
    record["tau_ratio"] = None if math.isnan(ratio) else ratio
    label = point.get("y_true")
    real = point.get("real_outcome")
    y = None if label is None else _position(classes, label, "label")
    o = None if real is None else _position(outcomes, real, "real outcome")
    utility = identity_utility(len(classes)) if point.get("utility") is None else point["utility"]
    voi = coherent_voi(projection.prior, projection.weights, projection.corrected, utility)
    if projection.converged:
        record["coherent_voi"] = voi
        if y is not None and o is not None:
            record["realized_gain"] = realized_gain(projection.u, projection.log_z[o], y)
    return record


def audit_point(reader, world_model, history, test, k=None, rng=None, budget=None):
    """Audit any reader and world model at one history and test (SPEC 9.1, behind the interfaces of SPEC 16).

    Calls reader.belief(history), world_model.outcomes(history, test, k, rng) and
    reader.read(history, test, outcome) for each proposed outcome, and returns the record of
    audit_record with the test as its id. k = None asks the world model for exact enumeration.
    budget is the information budget B(h, a) in nats, if one is declared.
    """
    predictive = world_model.outcomes(history, test, k, rng)
    return audit_record(
        {
            "id": str(test),
            "classes": list(reader.classes),
            "prior": reader.belief(history),
            "outcomes": list(predictive.outcomes),
            "weights": predictive.weights,
            "posteriors": [reader.read(history, test, outcome) for outcome in predictive.outcomes],
            "budget": budget,
        }
    )


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _geomean(values):
    values = [v for v in values if v is not None]
    return float(np.exp(np.mean(np.log(values)))) if values else None


def summarize(records):
    """Summary means of an audit card (SPEC 9.1).

    Scales (tau_B, tau_ratio) are averaged geometrically. KL_inc, coherent VOI and realized gain
    are averaged over converged points only; each mean is None when it has nothing to average.
    """
    converged = [r for r in records if r["converged"]]
    return {
        "n_points": len(records),
        "n_converged": len(converged),
        "mean_drift_l1": _mean(r["drift_l1"] for r in records),
        "mean_drift_max": _mean(r["drift_max"] for r in records),
        "mean_responsiveness": _mean(r["responsiveness"] for r in records),
        "mean_kl_inc": _mean(r["kl_inc"] for r in converged),
        "mean_coherent_voi": _mean(r["coherent_voi"] for r in converged),
        "n_labeled": sum(r["realized_gain"] is not None for r in records),
        "mean_realized_gain": _mean(r["realized_gain"] for r in records),
        "n_budget": sum(r["budget"] is not None for r in records),
        "geomean_tau_b_shrink_only": _geomean(r["tau_b_shrink_only"] for r in records),
        "geomean_tau_b_two_sided": _geomean(r["tau_b_two_sided"] for r in records),
        "n_tau_ratio": sum(r["tau_ratio"] is not None for r in records),
        "geomean_tau_ratio": _geomean(r["tau_ratio"] for r in records),
    }


def _number(value):
    if value is None:
        return "n/a"
    return "0" if abs(value) < DISPLAY_ZERO else f"{value:.4g}"


def _tau(value, path):
    return "n/a" if value is None else f"{value:.3g} ({path})"


def render_markdown(records, summary, source):
    """The audit card as markdown (SPEC 9.1)."""
    lines = [
        "# CEE audit card",
        "",
        f"Source: {source}. Decision points: {summary['n_points']}.",
        f"Method constants: floor {FLOOR:.0e} then renormalize; projection tolerance {TOL:.0e}, at most {MAX_ITER} iterations.",
        "Information is in nats. tau_B is the budget-implied scale: above 1 the reader reacts more than the",
        "budget justifies, below 1 less. The two-sided interval [1/8, 16] is the regime-map diagnostic; the",
        "shrink-only interval is [1, 16]. tau_ratio is a secondary diagnostic and never a repair. A point",
        "whose projection did not converge uses the raw reader, so it has no coherent VOI or realized gain.",
        f"Magnitudes below {DISPLAY_ZERO:.0e} are shown as 0.",
        "",
        "## Decision points",
        "",
        "| id | C | K | drift L1 | drift max | responsiveness | KL_inc | converged | tau_B shrink-only | tau_B two-sided | tau_ratio | coherent VOI | realized gain |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        cells = [
            str(r["id"]),
            str(r["n_classes"]),
            str(r["n_outcomes"]),
            _number(r["drift_l1"]),
            _number(r["drift_max"]),
            _number(r["responsiveness"]),
            _number(r["kl_inc"]),
            f"{'yes' if r['converged'] else 'no'} ({r['iterations']})",
            _tau(r["tau_b_shrink_only"], r["tau_b_shrink_only_path"]),
            _tau(r["tau_b_two_sided"], r["tau_b_two_sided_path"]),
            _number(r["tau_ratio"]),
            _number(r["coherent_voi"]),
            _number(r["realized_gain"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    rows = [
        ("decision points", summary["n_points"], summary["n_points"]),
        ("converged projections", summary["n_converged"], summary["n_points"]),
        ("mean drift, L1", summary["mean_drift_l1"], summary["n_points"]),
        ("mean drift, largest component", summary["mean_drift_max"], summary["n_points"]),
        ("mean responsiveness", summary["mean_responsiveness"], summary["n_points"]),
        ("mean KL_inc", summary["mean_kl_inc"], summary["n_converged"]),
        ("mean coherent VOI", summary["mean_coherent_voi"], summary["n_converged"]),
        ("mean realized gain", summary["mean_realized_gain"], summary["n_labeled"]),
        ("geometric mean tau_B, shrink-only", summary["geomean_tau_b_shrink_only"], summary["n_budget"]),
        ("geometric mean tau_B, two-sided", summary["geomean_tau_b_two_sided"], summary["n_budget"]),
        ("geometric mean tau_ratio", summary["geomean_tau_ratio"], summary["n_tau_ratio"]),
    ]
    lines += ["", "## Summary", "", "| quantity | value | points |", "|---|---|---|"]
    lines += [f"| {name} | {value if isinstance(value, int) else _number(value)} | {n} |" for name, value, n in rows]
    return "\n".join(lines) + "\n"
