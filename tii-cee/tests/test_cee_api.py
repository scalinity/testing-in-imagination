"""The public surface of tiicee.cee (SPEC 16, "CEE library").

A snapshot, so that an addition is a decision. In particular there is no function that minimizes
KL_inc over tau alone: SPEC 5.5 says that rule is degenerate and must not be used for anything.
"""
import ast
import sys
from pathlib import Path

import tiicee.cee as cee

EXPECTED = {
    # the floor and the projection (5, 5.3)
    "FLOOR", "TOL", "MAX_ITER", "floor_norm", "kl", "entropy", "Projection", "project", "apply_to_real",
    # audit quantities (5.2)
    "imagined_mean", "drift", "responsiveness", "reaction_kl", "EntropyChange", "entropy_change_surrogate",
    # the identity (5.4)
    "GainTerms", "gain_terms", "realized_gain",
    # scale (5.5, 5.11)
    "VARIANT_BOUNDS", "TAU_RATIO_GRID", "temper", "projected_responsiveness", "TauBudget", "tau_budget",
    "ScaleResult", "scale", "apply_scale_to_real", "TauRatio", "tau_ratio",
    # value of information, select, commit (5.6, 5.7)
    "phi", "coherent_voi", "raw_voi_score", "identity_utility", "recall_aligned_utility",
    "expected_charged_cost", "net_value", "select", "commit",
    # gate (5.8)
    "gate_allows", "bootstrap_upper_bound",
    # tracks (5.9)
    "Step", "stateful_step", "stateless_step",
    # multi-step accounting (5.12)
    "MultiStepAccounting", "multistep_accounting",
    # audit (9.1)
    "audit_record", "audit_point",
}


def test_public_names_are_exactly_the_declared_ones():
    assert set(cee.__all__) == EXPECTED
    assert len(cee.__all__) == len(set(cee.__all__))
    for name in EXPECTED:
        assert hasattr(cee, name), name


def test_every_public_callable_cites_the_specification():
    for name in EXPECTED:
        obj = getattr(cee, name)
        if callable(obj):
            assert obj.__doc__ and ("SPEC" in obj.__doc__ or "PROOF" in obj.__doc__), name


def test_the_library_imports_numpy_and_the_standard_library_only():
    # SPEC 9.1: a small library with no dependency on either environment.
    for path in sorted(Path(cee.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                assert root in sys.stdlib_module_names or root in ("numpy", "tiicee"), (path.name, root)
