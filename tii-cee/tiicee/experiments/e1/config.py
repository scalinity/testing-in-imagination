"""Every parameter of E1 in one place. card() is the frozen card; tiny() is for tests and timing only."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional

READERS = ("r0", "independence", "entrench", "fixed", "over", "under", "entrench+over", "fixed+under")
R4 = READERS[1:]
DRIFT_FAMILIES = ("entrench", "fixed")
SCALE_FAMILIES = ("over", "under")
SCALE_REPAIR_FAMILIES = ("over", "under", "independence")
BUDGETS = ("exact", "exact_x0.5", "exact_x2", "exact_noise", "w1", "w2cc", "heldout")
STRESS_SET = ("exact", "exact_x0.5", "exact_x2", "exact_noise")
SCALE_VARIANTS = ("shrink-only", "two-sided")
TRACKS = ("stateless", "stateful")


@dataclass(frozen=True)
class Config:
    name: str = "card"
    seeds: tuple = tuple(range(20))
    variants: tuple = ("base", "no_correlation", "strong_correlation", "duplicated_panel")
    primary_variant: str = "base"
    reduced_tests: Optional[tuple] = None       # None: the six tests of specification 8.1
    horizon_reduced: int = 6
    horizon_full: int = 8
    n_train: int = 5000                         # fitting stream, training split: W1, W2, W2-cc, the held-out budget
    n_eval: int = 2000                          # generation stream: full-twin decision points
    n_gate: int = 1000                          # fitting stream, after the training patients: CEE-Gate bounds
    eval_split: str = "validation"              # E1 makes method choices, so its cases are validation cases
    lam: float = 1.0
    lam_frontier: tuple = (0.0, 0.5, 1.0, 2.0)
    threshold: float = 0.8
    threshold_sensitivity: tuple = (0.7, 0.9)
    readers: tuple = READERS
    world_models: tuple = ("W0", "W1", "W2")    # W2-cc is W2's fitted model read class-conditionally: same predictive
    mirror: tuple = (0.1, 0.2, 0.4)
    temperatures: tuple = (0.5, 2.0)
    matched_kl: float = 0.02                    # nats; both members of the matched pair sit at this KL from the truth
    noise_sigma: float = 0.7
    heldout_folds: int = 5
    heldout_l2: float = 1.0
    heldout_level: float = 0.90
    admission_limit: float = 0.10
    spearman_margin: float = 0.20
    no_harm: float = 1e-9
    n_boot: int = 10_000
    analysis_seed: int = 20261113               # bootstrap draws of the rule evaluation

    def as_dict(self) -> dict:
        return asdict(self)


def card() -> Config:
    return Config()


def tiny(**changes) -> Config:
    """A few seeds, three reduced-twin tests, a handful of patients. Never a result."""
    base = Config(
        name="tiny", seeds=(0, 1), variants=("base", "duplicated_panel"),
        reduced_tests=("CT", "Liver Function Panel", "Ultrasound"), horizon_reduced=3, horizon_full=3,
        n_train=400, n_eval=30, n_gate=60, n_boot=200, heldout_folds=5,
    )
    return replace(base, **changes)
