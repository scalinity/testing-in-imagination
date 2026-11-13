"""The CEE library of SPEC 16: conservation of expected evidence as a small numpy-only instrument.

See README.md in this directory for the map from functions to specification sections and for the
floors and tolerances in force. The command-line audit is `python -m tiicee.cee.audit`.
"""
from .card import audit_point, audit_record
from .core import FLOOR, MAX_ITER, TOL, Projection, apply_to_real, entropy, floor_norm, kl, project
from .decision import (
    coherent_voi,
    commit,
    expected_charged_cost,
    identity_utility,
    net_value,
    phi,
    raw_voi_score,
    recall_aligned_utility,
    select,
)
from .gate import bootstrap_upper_bound, gate_allows
from .identity import GainTerms, gain_terms, realized_gain
from .measures import EntropyChange, drift, entropy_change_surrogate, imagined_mean, reaction_kl, responsiveness
from .multistep import MultiStepAccounting, multistep_accounting
from .scaling import (
    TAU_RATIO_GRID,
    VARIANT_BOUNDS,
    ScaleResult,
    TauBudget,
    TauRatio,
    apply_scale_to_real,
    projected_responsiveness,
    scale,
    tau_budget,
    tau_ratio,
    temper,
)
from .tracks import Step, stateful_step, stateless_step

__all__ = [
    "FLOOR",
    "TOL",
    "MAX_ITER",
    "floor_norm",
    "kl",
    "entropy",
    "Projection",
    "project",
    "apply_to_real",
    "imagined_mean",
    "drift",
    "responsiveness",
    "reaction_kl",
    "EntropyChange",
    "entropy_change_surrogate",
    "GainTerms",
    "gain_terms",
    "realized_gain",
    "VARIANT_BOUNDS",
    "TAU_RATIO_GRID",
    "temper",
    "projected_responsiveness",
    "TauBudget",
    "tau_budget",
    "ScaleResult",
    "scale",
    "apply_scale_to_real",
    "TauRatio",
    "tau_ratio",
    "phi",
    "coherent_voi",
    "raw_voi_score",
    "identity_utility",
    "recall_aligned_utility",
    "expected_charged_cost",
    "net_value",
    "select",
    "commit",
    "gate_allows",
    "bootstrap_upper_bound",
    "Step",
    "stateful_step",
    "stateless_step",
    "MultiStepAccounting",
    "multistep_accounting",
    "audit_record",
    "audit_point",
]
