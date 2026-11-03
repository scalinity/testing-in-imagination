"""Episode metrics, named as in LA-CDM Table 1 where they correspond.

- ``mean_acc``: mean of per-class accuracies (Table 1 "Mean").
- ``micro_f1`` / ``macro_f1``: undiagnosed episodes count as false negatives
  of the true class and are no class's false positive.
- ``avg_cost_usd``: tests that returned data only (LA-CDM's basis, F-2).
- ``utility``: mean of 1[correct] - lam * cost_norm, the agents' objective.
"""

from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Mapping, Sequence

from extension.envs.synthetic_diagnosis.actions import A_DIAG
from extension.planners.agents import EpisodeResult


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def episode_metrics(results: Sequence[EpisodeResult], lam: float) -> Dict[str, float]:
    n = len(results)
    out: Dict[str, float] = {"n": n}
    class_acc, class_f1 = [], []
    for y in A_DIAG:
        tp = sum(r.y_true == y and r.y_pred == y for r in results)
        fp = sum(r.y_true != y and r.y_pred == y for r in results)
        fn = sum(r.y_true == y and r.y_pred != y for r in results)
        support = tp + fn
        acc = tp / support if support else 0.0
        out[f"acc_{y}"] = acc
        class_acc.append(acc)
        class_f1.append(_f1(tp, fp, fn))
    correct = sum(r.correct for r in results)
    diagnosed = sum(r.y_pred is not None for r in results)
    out["mean_acc"] = sum(class_acc) / len(class_acc)
    out["micro_f1"] = _f1(correct, diagnosed - correct, n - correct)
    out["macro_f1"] = sum(class_f1) / len(class_f1)
    out["overall_acc"] = correct / n
    out["avg_cost_usd"] = sum(r.cost_usd for r in results) / n
    out["avg_tests"] = sum(r.n_tests for r in results) / n
    out["avg_requests"] = sum(r.n_requests for r in results) / n
    out["undiagnosed_rate"] = (n - diagnosed) / n
    out["utility"] = sum(r.correct - lam * r.cost_norm for r in results) / n
    return out


def aggregate(per_seed: Iterable[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean and sample std of each metric across seeds."""
    per_seed = list(per_seed)
    keys: List[str] = list(per_seed[0])
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [m[k] for m in per_seed]
        out[k] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return out
