"""The twin's generative model, taken from the vendored snapshot of Edwin Salcedo's synthetic study.

    y ~ prior                                   diagnosis
    e | y ~ phenotype_given_class[y]            presenting phenotype (atypical when e != y)
    s_i | y, e ~ finding_lik[i, y, e]           finding of item i, one of the classes or "nonspecific"
    available_i | y ~ Bernoulli(availability[i, y])

A test reads one item. Two tests that read the same item return the same finding and the same
availability; that is the only way two results are tied beyond the shared phenotype. A requested
test returns its finding when the item is in the chart and UNAVAILABLE otherwise.

In the snapshot the finding depends on the phenotype only and availability on neither; the two
extra axes exist so that every variant is the base model with one array changed:

    base                      the snapshot, with its availability table
    no_correlation            each test reads its own phenotype draw: the single-test distributions
                              given the class are unchanged, and tests are independent given the class
    strong_correlation        atypical-presentation rate 0.20 instead of 0.07
    duplicated_panel          one extra test that reads the liver function panel again
    informative_missingness   availability depends on the diagnosis. HELD OUT: no method choice may
                              use it (specification section 8.1)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from tiicee.interfaces import UNAVAILABLE
from tiicee.twin import source

# Specification section 8.1, in the snapshot's names.
REDUCED_TESTS = (
    "Physical Examination", "CT", "Ultrasound", "Complete Blood Count", "Liver Function Panel", "Urinalysis",
)
P_ATYPICAL_STRONG = 0.20          # the stress value of cee_check.py
DUPLICATED_TEST = "Liver Function Panel"
MISSINGNESS_STRENGTH = 1.0

_VARIANTS = ("base", "no_correlation", "strong_correlation", "duplicated_panel", "informative_missingness")
_HELD_OUT = frozenset({"informative_missingness"})


class HeldOutVariant(ValueError):
    """A held-out variant was requested without allow_held_out=True."""


@dataclass(frozen=True, eq=False)
class TwinModel:
    name: str
    variant: str
    held_out: bool
    classes: tuple
    phenotypes: tuple
    findings: tuple
    tests: tuple
    items: tuple                        # the distinct measurements
    item_of: tuple                      # for each test, the index of the item it reads
    prior: np.ndarray                   # [class]
    phenotype_given_class: np.ndarray   # [class, phenotype]
    finding_lik: np.ndarray             # [item, class, phenotype, finding]
    availability: np.ndarray            # [item, class]
    cost_usd: np.ndarray                # [test]
    cost_norm: np.ndarray               # [test], on the existing study's scale (its twelve tests sum to one)
    step_limit: int                     # actions per episode in the snapshot, the diagnosis included
    source: str                         # digest of the snapshot the parameters came from

    def __post_init__(self):
        n_class, n_pheno, n_find = len(self.classes), len(self.phenotypes), len(self.findings)
        n_tests, n_items = len(self.tests), len(self.items)
        shapes = {
            "prior": (n_class,), "phenotype_given_class": (n_class, n_pheno),
            "finding_lik": (n_items, n_class, n_pheno, n_find), "availability": (n_items, n_class),
            "cost_usd": (n_tests,), "cost_norm": (n_tests,),
        }
        for field, shape in shapes.items():
            array = np.array(getattr(self, field), dtype=float)
            if array.shape != shape:
                raise ValueError(f"{field} has shape {array.shape}, expected {shape}")
            if not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f"{field} must be finite and non-negative")
            array.setflags(write=False)
            object.__setattr__(self, field, array)
        for field in ("classes", "phenotypes", "findings", "tests", "items", "item_of"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        for field in ("classes", "phenotypes", "findings", "tests", "items"):
            if len(set(getattr(self, field))) != len(getattr(self, field)):
                raise ValueError(f"{field} has repeated names")
        if UNAVAILABLE in self.findings:
            raise ValueError("UNAVAILABLE is not a finding")
        if len(self.item_of) != n_tests or set(self.item_of) != set(range(n_items)):
            raise ValueError("item_of must give one item per test and reach every item")
        for field in ("prior", "phenotype_given_class", "finding_lik"):
            if np.abs(getattr(self, field).sum(-1) - 1.0).max() > 1e-9:
                raise ValueError(f"{field} is not normalized")
        if (self.availability > 1).any():
            raise ValueError("availability must lie in [0, 1]")

    @property
    def outcomes(self) -> tuple:
        """What a request can return: the findings, then UNAVAILABLE."""
        return self.findings + (UNAVAILABLE,)

    def outcome_lik(self) -> np.ndarray:
        """P(outcome | item, class, phenotype), UNAVAILABLE last. Shape [item, class, phenotype, outcome]."""
        available = self.availability[:, :, None, None] * self.finding_lik
        absent = np.broadcast_to((1.0 - self.availability)[:, :, None, None], self.finding_lik.shape[:3] + (1,))
        return np.concatenate([available, absent], axis=-1)

    def class_lik(self) -> np.ndarray:
        """P(outcome | test, class) with the phenotype summed out. Shape [test, class, outcome].

        This is the class-conditional independence model: exact for one test, wrong for several
        whenever the phenotype ties them together.
        """
        by_item = np.einsum("ye,iyeo->iyo", self.phenotype_given_class, self.outcome_lik())
        return by_item[list(self.item_of)]


def restrict(model: TwinModel, tests: Sequence[str], name: Optional[str] = None) -> TwinModel:
    """The same model with only the named tests, in the given order."""
    unknown = [t for t in tests if t not in model.tests]
    if unknown:
        raise ValueError(f"not tests of {model.name}: {unknown}")
    rows = [model.tests.index(t) for t in tests]
    kept = list(dict.fromkeys(model.item_of[r] for r in rows))
    return replace(
        model,
        name=name or f"{model.name}[{', '.join(tests)}]",
        tests=tuple(tests),
        items=tuple(model.items[i] for i in kept),
        item_of=tuple(kept.index(model.item_of[r]) for r in rows),
        finding_lik=model.finding_lik[kept],
        availability=model.availability[kept],
        cost_usd=model.cost_usd[rows],
        cost_norm=model.cost_norm[rows],
    )


def _snapshot_model(missingness: bool) -> TwinModel:
    actions, profiles = source.actions, source.profiles
    classes, tests, findings = tuple(actions.A_DIAG), tuple(actions.A_TEST), tuple(profiles.FINDINGS)
    phenotype = profiles.phenotype_given_y(profiles.P_ATYPICAL)
    by_phenotype = np.array([[[profiles.LIKELIHOOD[t][e][s] for s in findings] for e in classes] for t in tests])
    availability = np.array([profiles.AVAILABILITY[t] if missingness else 1.0 for t in tests])
    return TwinModel(
        name="full", variant="base", held_out=False,
        classes=classes, phenotypes=classes, findings=findings,
        tests=tests, items=tests, item_of=tuple(range(len(tests))),
        prior=np.array([profiles.PRIOR_MIMIC_CDM[c] for c in classes]),
        phenotype_given_class=np.array([[phenotype[y][e] for e in classes] for y in classes]),
        finding_lik=np.repeat(by_phenotype[:, None], len(classes), axis=1),
        availability=np.repeat(availability[:, None], len(classes), axis=1),
        cost_usd=np.array([actions.COST_USD[t] for t in tests]),
        cost_norm=np.array([actions.COST_NORM[t] for t in tests]),
        step_limit=int(profiles.V1_MAX_STEPS),
        source=source.snapshot_digest(),
    )


def _no_correlation(model: TwinModel) -> TwinModel:
    by_class = np.einsum("ye,iyes->iys", model.phenotype_given_class, model.finding_lik)
    return replace(model, finding_lik=np.repeat(by_class[:, :, None], len(model.phenotypes), axis=2))


def _strong_correlation(model: TwinModel) -> TwinModel:
    phenotype = source.profiles.phenotype_given_y(P_ATYPICAL_STRONG)
    return replace(model, phenotype_given_class=np.array(
        [[phenotype[y][e] for e in model.phenotypes] for y in model.classes]))


def _duplicated_panel(model: TwinModel) -> TwinModel:
    original = model.tests.index(DUPLICATED_TEST)
    return replace(
        model,
        tests=model.tests + (f"{DUPLICATED_TEST} (repeat)",),
        item_of=model.item_of + (model.item_of[original],),
        cost_usd=np.append(model.cost_usd, model.cost_usd[original]),
        cost_norm=np.append(model.cost_norm, model.cost_norm[original]),
    )


def _informative_missingness(model: TwinModel) -> TwinModel:
    """A test is charted more often for the diseases it detects well.

    availability[i, y] = a_i + strength * min(a_i, 1 - a_i) * (hit[i, y] - sum_y' prior[y'] hit[i, y']),
    where hit[i, y] is the chance that item i shows finding y in a typical patient of class y. The
    prior-weighted mean over classes is still a_i, so the expected cost of a request at the empty
    history is the base model's; only the dependence on the diagnosis is new.
    """
    n_class = len(model.classes)
    hit = model.finding_lik[:, np.arange(n_class), np.arange(n_class), np.arange(n_class)]
    base = model.availability
    room = np.minimum(base, 1.0 - base)
    shift = hit - (hit @ model.prior)[:, None]
    return replace(model, availability=base + MISSINGNESS_STRENGTH * room * shift)


_TRANSFORMS = {
    "base": lambda model: model,
    "no_correlation": _no_correlation,
    "strong_correlation": _strong_correlation,
    "duplicated_panel": _duplicated_panel,
    "informative_missingness": _informative_missingness,
}


def variants(allow_held_out: bool = False) -> tuple:
    """Variant names. Held-out variants are listed only on request: method selection never sees them."""
    return tuple(v for v in _VARIANTS if allow_held_out or v not in _HELD_OUT)


def build(variant: str = "base", *, reduced: bool, missingness: bool = True, allow_held_out: bool = False) -> TwinModel:
    """A variant of the twelve-test twin (reduced=False) or of the six-test twin of section 8.1 (reduced=True).

    missingness=False puts every test in every chart, the snapshot's always-available setting.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {_VARIANTS}")
    if variant in _HELD_OUT and not allow_held_out:
        raise HeldOutVariant(f"{variant} is held out; pass allow_held_out=True for evaluation only")
    if variant == "informative_missingness" and not missingness:
        raise ValueError("informative_missingness needs missingness=True")
    model = _snapshot_model(missingness)
    if reduced:
        model = restrict(model, REDUCED_TESTS)
    name = ("reduced" if reduced else "full") + f"/{variant}" + ("" if missingness else "/always-available")
    return replace(_TRANSFORMS[variant](model), name=name, variant=variant, held_out=variant in _HELD_OUT)
