"""V1, the hypothesis-fallback reference of a policy, from that policy's unguarded rows."""
from __future__ import annotations

from tiicee.clinical import use_prior_studies


def v1_rows(rows) -> list:
    """tim's v1_rows rule (tim/analysis.py), applied by tim's own function: a missing final answer
    becomes the last valid hypothesis; an answer that was given never changes."""
    use_prior_studies()
    from tim.analysis import v1_rows as rule
    return rule(rows)
