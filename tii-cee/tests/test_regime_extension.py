"""The D-REGIME extension: its readers against the twin's own, its slopes, its search against
regime_check's, and its criteria."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tiicee import regime_check
from tiicee import regime_extension as ext
from tiicee.experiments.e1 import points
from tiicee.experiments.e1.config import tiny
from tiicee.experiments.e1.context import Context
from tiicee.interfaces import UNAVAILABLE
from tiicee.twin.readers import BayesReader, SimulatedReader


@pytest.fixture(scope="module")
def ctx():
    return Context(tiny(n_train=300, n_eval=6, n_gate=40, n_boot=300), 0, "base")


@pytest.fixture(scope="module")
def first(ctx):
    """(test index, test name) at each acquisition rule's first decision point."""
    m = ctx.twin("full").model
    out = {}
    for selector in points.SELECTORS:
        t = int(points.Trajectories(ctx, selector).steps[0].tests[0])
        out[selector] = (t, m.tests[t])
    return out


def test_abnormal_is_every_class_pointing_finding(ctx):
    m = ctx.twin("full").model
    mask = ext.abnormal(m)
    assert [o for o, k in zip(m.outcomes, mask) if k] == list(m.classes)
    assert [o for o, k in zip(m.outcomes, mask) if not k] == ["nonspecific", UNAVAILABLE]


def test_equal_powers_are_the_r4_reader_and_one_is_r0(ctx, first):
    twin = ctx.twin("full")
    m = twin.model
    for t, test in first.values():
        for p in ext.POWERS:
            mine = ext.reactions(m, t, np.full(len(m.outcomes), p))
            theirs = np.array([SimulatedReader(twin, power=p).read((), test, o) for o in m.outcomes])
            assert np.allclose(mine, theirs, atol=1e-12)
        bayes = np.array([BayesReader(twin).read((), test, o) for o in m.outcomes])
        assert np.allclose(ext.reactions(m, t, np.ones(len(m.outcomes))), bayes, atol=1e-12)


def test_slopes_recover_each_readers_powers_and_ignore_drift(ctx, first):
    twin = ctx.twin("full")
    m, mask = twin.model, ext.abnormal(twin.model)
    for t, test in first.values():
        w0 = ext.weights(ctx, "W0", test)
        for a, b in ext.READERS:
            found = ext.slopes(m, t, ext.reactions(m, t, np.where(mask, a, b)), w0, mask)
            assert found == pytest.approx({"p_abnormal": a, "p_normal": b}, abs=1e-9), (test, a, b)
        for drift in ("entrench", "fixed"):          # a class factor at the empty history: the shared class term takes it
            reader = SimulatedReader(twin, power=2.0, drift=drift, delta=0.8)
            post = np.array([reader.read((), test, o) for o in m.outcomes])
            assert ext.slopes(m, t, post, w0, mask) == pytest.approx({"p_abnormal": 2.0, "p_normal": 2.0}, abs=1e-9)


def test_shifted_world_models_scale_the_class_pointing_findings(ctx, first):
    m = ctx.twin("full").model
    mask = ext.abnormal(m)
    for _, test in first.values():
        w0 = ext.weights(ctx, "W0", test)
        assert w0.sum() == pytest.approx(1.0)
        for world, factor in ext.SHIFTS.items():
            w = ext.weights(ctx, world, test)
            assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
            ratio = (w[mask].sum() / w[~mask].sum()) / (w0[mask].sum() / w0[~mask].sum())
            assert ratio == pytest.approx(factor)
            assert np.allclose(w[mask] / w[mask].sum(), w0[mask] / w0[mask].sum())
        with pytest.raises(ValueError, match="unknown world model"):
            ext.weights(ctx, "W1", test)


@pytest.mark.parametrize("world", ext.JUDGED)
def test_the_search_agrees_with_the_validated_first_point(ctx, first, world):
    """r0 and (2, 2) against regime_check.first_point, the path D-REGIME was validated on."""
    twin = ctx.twin("full")
    m, mask = twin.model, ext.abnormal(twin.model)
    for selector, (t, test) in first.items():
        traj = points.Trajectories(ctx, selector)
        budget = twin.true_information((), test)
        w = ext.weights(ctx, world, test)
        r0 = np.array([BayesReader(twin).read((), test, o) for o in m.outcomes])
        for name, post in (("r0", r0), ("over", ext.reactions(m, t, np.where(mask, 2.0, 2.0)))):
            mine, theirs = ext.point(m.prior, w, post, budget), regime_check.first_point(traj, name, world)
            assert abs(mine["log_tau_b"] - theirs["log_tau_b"]) <= 2 * regime_check.RESOLUTION
            # the small fixture's W2 (300 training patients) leaves CT's projection unconverged on both paths
            assert mine["converged"] == bool(theirs["search_converged"]) and mine["path"] == "solved"


def synthetic_rows(cfg, shift=None):
    """Rows at log tau_B = 0.6 log a + 0.4 log b with a small jitter, plus an offset per world model and
    seed that R0 shares, so only a reading paired with R0 by seed cancels it. A shift is a number or a
    function of the seed."""
    rng = np.random.default_rng(0)
    rows = []
    for seed in cfg.seeds:
        for w, world in enumerate(ext.WORLD_MODELS):
            offset = 0.5 * w + 0.3 * seed
            for selector in points.SELECTORS:
                for name, a, b in [("r0", 1.0, 1.0)] + [(ext.reader_name(a, b), a, b) for a, b in ext.READERS]:
                    extra = (shift or {}).get((world, name), 0.0)
                    extra = extra(seed) if callable(extra) else extra
                    value = offset + 0.6 * np.log(a) + 0.4 * np.log(b) + 0.0005 * rng.standard_normal() + extra
                    rows.append({"seed": seed, "variant": cfg.primary_variant, "selector": selector, "world_model": world,
                                 "reader": name, "test": "CT", "abnormal_share": 0.5, "log_tau_b": value,
                                 "path": "solved", "converged": True, "p_abnormal": a, "p_normal": b})
    return rows


def test_criteria_support_ordered_readers_and_fail_when_moved():
    cfg = tiny(seeds=tuple(range(6)), n_boot=500)
    good = ext.evaluate(synthetic_rows(cfg), cfg)
    assert good["outcome"] == "supported"
    assert all(good["world_models"][w]["judged"] == (w in ext.JUDGED) for w in ext.WORLD_MODELS)
    relative = good["world_models"]["W2"]["readers"]["a0.5:b1"]["relative_to_r0"]   # the shared offset cancels
    assert np.log(relative["estimate"]) == pytest.approx(0.6 * np.log(0.5), abs=0.005)

    def side_fails_alone(world, name, shift):
        found = ext.evaluate(synthetic_rows(cfg, {(world, name): shift}), cfg)
        here, other = found["world_models"][world], found["world_models"]["W2" if world == "W0" else "W0"]
        assert found["outcome"] == "scope limit", name
        assert here["readers"][name]["side_holds"] is False and not here["sides_hold"] and here["ordering"]["holds"]
        assert other["sides_hold"] and other["ordering"]["holds"]

    third, half_band = 0.4 * np.log(2), 0.5 * ext.BAND
    side_fails_alone("W2", "a1:b0.5", third - half_band)          # inside the band, below R0
    side_fails_alone("W0", "a1:b2", -third + half_band)            # inside the band, above R0
    side_fails_alone("W0", "a1:b2", lambda seed: -third + ext.BAND + 0.02 * (-1) ** seed)    # straddles +band
    side_fails_alone("W2", "a1:b0.5", lambda seed: third - ext.BAND + 0.02 * (-1) ** seed)   # straddles -band

    # (2, 1) moved above (2, 2): every side still holds, the line over b at a = 2 does not rise
    hump = ext.evaluate(synthetic_rows(cfg, {("W0", "a2:b1"): 0.5 * np.log(2)}), cfg)
    assert hump["outcome"] == "scope limit"
    assert hump["world_models"]["W0"]["sides_hold"] and not hump["world_models"]["W0"]["ordering"]["holds"]
    broken = [(line["vary"], line["fixed"]) for line in hump["world_models"]["W0"]["ordering"]["lines"] if not line["rises"]]
    assert broken == [("b", 2.0)]

    # a shifted world model is reported, never decided on
    shifted = ext.evaluate(synthetic_rows(cfg, {("W0-abnormal-x2", "a2:b2"): -3.0}), cfg)
    assert shifted["outcome"] == "supported" and not shifted["world_models"]["W0-abnormal-x2"]["sides_hold"]


def test_mixed_readers_and_r0s_twin_are_not_judged():
    assert [(a, b) for a, b in ext.READERS if ext.expected_side(a, b) == 0] == [(0.5, 2.0), (1.0, 1.0), (2.0, 0.5)]


def test_command_line_runs_every_seed_and_decides(tmp_path):
    cfg = tiny()
    ext.main(["cells", "--out", str(tmp_path), "--config", "tiny"])
    files = sorted((tmp_path / "rows").glob("*.json"))
    assert [f.name for f in files] == [f"seed{s:02d}_base.json" for s in cfg.seeds]
    rows = [r for f in files for r in json.loads(f.read_text())]
    assert len(rows) == len(cfg.seeds) * len(points.SELECTORS) * len(ext.WORLD_MODELS) * (1 + len(ext.READERS))
    ext.main(["evaluate", "--out", str(tmp_path), "--config", "tiny"])
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["outcome"] in ("supported", "scope limit") and set(verdict["world_models"]) == set(ext.WORLD_MODELS)
    for row in rows:                    # cell() gives a to the class-pointing findings and b to the rest
        if row["reader"] != "r0":
            a, b = (float(x[1:]) for x in row["reader"].split(":"))
            assert (row["p_abnormal"], row["p_normal"]) == pytest.approx((a, b), abs=1e-9), row["reader"]
    for world in ext.WORLD_MODELS:
        for a, b in ext.READERS:
            entry = verdict["world_models"][world]["readers"][ext.reader_name(a, b)]
            assert (entry["a"], entry["b"]) == (a, b)
            assert (entry["p_abnormal"], entry["p_normal"]) == pytest.approx((a, b), abs=1e-9)
    assert len(verdict["code"]["regime_extension_sha256"]) == 64
