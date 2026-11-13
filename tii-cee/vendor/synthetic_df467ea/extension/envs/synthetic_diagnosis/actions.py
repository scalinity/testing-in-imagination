"""Frozen LA-CDM action inventories and costs (Appendix Table 4 + PE)."""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# Exact LA-CDM test action strings (A_test)
A_TEST: Tuple[str, ...] = (
    "Physical Examination",
    "CT",
    "MRI",
    "Radiograph",
    "Ultrasound",
    "Complete Blood Count",
    "Basic Metabolic Panel",
    "Comprehensive Metabolic Panel",
    "Renal Function Panel",
    "Liver Function Panel",
    "Urinalysis",
    "Electrolyte Panel",
)

# Exact LA-CDM diagnosis action strings (A_diag)
A_DIAG: Tuple[str, ...] = (
    "appendicitis",
    "cholecystitis",
    "diverticulitis",
    "pancreatitis",
)

A_TEST_SET: FrozenSet[str] = frozenset(A_TEST)
A_DIAG_SET: FrozenSet[str] = frozenset(A_DIAG)

# Absolute USD costs from paper Appendix Table 4;
# Physical Examination ≈ $300 from archaeology D-2.
COST_USD: Dict[str, float] = {
    "Physical Examination": 300.0,
    "CT": 1306.0,
    "MRI": 4866.0,
    "Radiograph": 434.0,
    "Ultrasound": 1288.0,
    "Complete Blood Count": 71.0,
    "Basic Metabolic Panel": 298.0,
    "Comprehensive Metabolic Panel": 636.0,
    "Renal Function Panel": 394.0,
    "Liver Function Panel": 413.0,
    "Urinalysis": 50.0,
    "Electrolyte Panel": 134.0,
}

_TOTAL_COST = sum(COST_USD[a] for a in A_TEST)

# Normalized costs that sum to 1.0 (LA-CDM style)
COST_NORM: Dict[str, float] = {
    a: COST_USD[a] / _TOTAL_COST for a in A_TEST
}


def is_test(action: str) -> bool:
    return action in A_TEST_SET


def is_diagnosis(action: str) -> bool:
    return action in A_DIAG_SET


def is_valid_action(action: str) -> bool:
    return is_test(action) or is_diagnosis(action)
