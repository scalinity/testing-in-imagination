"""Holm adjustment inside one named family of planned contrasts."""
from __future__ import annotations

import math
from numbers import Real


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values for one named family: {name: adjusted p-value}.

    Call it once per family; families are never pooled, and an experiment's single primary
    contrast stays outside every family, uncorrected. A value of None marks a planned member that
    was not run: it stays in the family size, so it still costs the others, and its adjusted value
    is None. This is the rule of replication.stats.holm_adjust.
    """
    for name, p in pvalues.items():
        if p is not None and (isinstance(p, bool) or not isinstance(p, Real)
                              or not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError(f"p-value of {name!r} must be a number in [0, 1] or None")
    adjusted = {name: None for name in pvalues}
    run = sorted(((p, name) for name, p in pvalues.items() if p is not None), key=lambda item: item[0])
    previous = 0.0
    for i, (p, name) in enumerate(run):
        previous = min(1.0, max(previous, float((len(pvalues) - i) * p)))
        adjusted[name] = previous
    return adjusted
