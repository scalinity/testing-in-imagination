"""Holm adjustment within one named family, against hand calculations."""
import numpy as np
import pytest

from tiicee import stats


def test_holm_by_hand():
    # Sorted: 0.01, 0.03, 0.04 with multipliers 3, 2, 1: 0.03, 0.06, 0.04, then made monotone.
    assert stats.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06}, abs=1e-15)
    # 4 * 0.001, 3 * 0.01, 2 * 0.02, 1 * 0.5
    assert stats.holm({"w": 0.5, "x": 0.02, "y": 0.001, "z": 0.01}) == pytest.approx(
        {"w": 0.5, "x": 0.04, "y": 0.004, "z": 0.03}, abs=1e-15)


def test_holm_keeps_the_callers_order_and_caps_at_one():
    adjusted = stats.holm({"late": 0.6, "early": 0.5})
    assert list(adjusted) == ["late", "early"] and adjusted == {"late": 1.0, "early": 1.0}
    assert stats.holm({"only": 0.03}) == {"only": 0.03}
    assert stats.holm({}) == {}


def test_tied_p_values_share_an_adjusted_value():
    assert stats.holm({"a": 0.02, "b": 0.02, "c": 0.5}) == pytest.approx({"a": 0.06, "b": 0.06, "c": 0.5}, abs=1e-15)


def test_an_unrun_member_stays_in_the_family_size():
    adjusted = stats.holm({"run": 0.01, "unrun": None, "other": 0.02})
    assert adjusted["unrun"] is None
    assert adjusted["run"] == pytest.approx(0.03, abs=1e-15) and adjusted["other"] == pytest.approx(0.04, abs=1e-15)


def test_adjustment_never_lowers_a_p_value_and_one_family_does_not_see_another():
    rng = np.random.default_rng(0)
    family = {f"c{i}": float(p) for i, p in enumerate(rng.uniform(0, 0.2, size=6))}
    adjusted = stats.holm(family)
    assert all(adjusted[name] >= p for name, p in family.items())
    order = sorted(family, key=family.get)
    assert [adjusted[name] for name in order] == sorted(adjusted.values())
    first, second = dict(list(family.items())[:3]), dict(list(family.items())[3:])
    assert {**stats.holm(first), **stats.holm(second)} != adjusted  # families are adjusted separately, never pooled


def test_holm_accepts_numpy_floats_and_refuses_invalid_p_values():
    assert stats.holm({"a": np.float64(0.01), "b": np.float32(0.5)}) == pytest.approx({"a": 0.02, "b": 0.5}, abs=1e-12)
    for bad in (1.2, -0.1, float("nan"), "0.05", True):
        with pytest.raises(ValueError, match="p-value"):
            stats.holm({"a": 0.01, "b": bad})
