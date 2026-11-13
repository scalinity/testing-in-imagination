"""The one statistics module of study TII-CEE-1 (SPEC_RESEARCH_ADVANCEMENT.md section 15).

Every experiment takes its intervals, tests and adjustments from here; none computes its own.

- The unit is the admission on charts and the patient in the twin and on public data. Rows of one
  unit (orders, prefixes, renderings, items) are never resampled independently.
- Intervals are percentile bootstrap intervals, 95% and 10,000 resamples unless a caller says
  otherwise, paired wherever two arms share units, and class-stratified for T5.
- T5 contrasts use the two-sided paired sign-flip randomization test of tim/analysis.py, with its
  null distribution computed exactly; the result names the null it tests.
- One primary contrast per experiment is uncorrected. Other contrasts are Holm-adjusted inside
  their named family, one family per call.
- Every function that draws takes an explicit numpy Generator; nothing reads global random state.
  Results are plain JSON-ready dicts; an interval sits under "ci" beside its "level".
"""
from tiicee.stats.holm import holm
from tiicee.stats.intervals import hierarchical_interval, paired_interval, paired_interval_policies, seed_interval
from tiicee.stats.rank import cell_spearman_interval, spearman, spearman_difference_interval
from tiicee.stats.t5 import mde, sign_flip_p, t5, t5_contrast, t5_power

__all__ = ["cell_spearman_interval", "hierarchical_interval", "holm", "mde", "paired_interval",
           "paired_interval_policies", "seed_interval", "sign_flip_p", "spearman",
           "spearman_difference_interval", "t5", "t5_contrast", "t5_power"]
