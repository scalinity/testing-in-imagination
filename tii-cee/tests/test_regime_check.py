"""D-REGIME's check: the first-decision-point tau_B against the library, its exact values under W0, and the criteria."""
from __future__ import annotations

import numpy as np
import pytest

from tiicee import cee, regime_check
from tiicee.experiments.e1 import config, figure1, points
from tiicee.experiments.e1.config import tiny
from tiicee.experiments.e1.context import Context
from tiicee.twin import world_models as wm
from tiicee.twin.readers import BayesReader, SimulatedReader

RESOLUTION = regime_check.RESOLUTION


def reader_object(twin, name):
    if name == "r0":
        return BayesReader(twin)
    if name.startswith("grid:"):
        _, power, delta, drift = name.split(":")
        drift = None if float(delta) == 0 else drift
        return SimulatedReader(twin, power=float(power), drift=drift, delta=float(delta))
    options = {"over": {"power": 2.0}, "fixed+under": {"drift": "fixed", "power": 0.5}, "entrench": {"drift": "entrench"}}
    return SimulatedReader(twin, **options[name])


def dense(model, predictive):
    out = np.zeros(len(model.outcomes))
    for outcome, weight in zip(predictive.outcomes, predictive.weights):
        out[model.outcomes.index(outcome)] += weight
    return out


@pytest.fixture(scope="module")
def ctx():
    return Context(tiny(n_train=300, n_eval=6, n_gate=40, n_boot=300), 0, "base")


@pytest.fixture(scope="module")
def trajectories(ctx):
    return {name: points.Trajectories(ctx, name) for name in points.SELECTORS}


@pytest.mark.parametrize("reader_name", ["r0", "over", "fixed+under", "entrench", "grid:0.25:0.8:entrench", "grid:3:0.2:fixed"])
@pytest.mark.parametrize("world_name", ["W0", "W2"])
def test_first_point_is_the_library_search_at_the_empty_history(ctx, trajectories, reader_name, world_name):
    twin = ctx.twin("full")
    m = twin.model
    world = wm.w0(twin) if world_name == "W0" else ctx.fitted("full", world_name)
    reader = reader_object(twin, reader_name)
    for traj in trajectories.values():
        test = m.tests[int(traj.steps[0].tests[0])]
        weights = dense(m, world.outcomes((), test, None, None))
        posteriors = np.array([reader.read((), test, o) for o in m.outcomes])
        found = cee.tau_budget(reader.belief(()), weights, posteriors, twin.true_information((), test), cee.VARIANT_BOUNDS["two-sided"])
        row = regime_check.first_point(traj, reader_name, world_name)
        assert row["n_points"] == traj.n and row["log_tau_b_range"] == 0.0
        # r' crosses the budget exactly at a grid point under W0, so the two searches may bracket it from either side
        assert abs(row["log_tau_b"] - np.log(found.tau)) <= RESOLUTION
        assert row["search_converged"] == float(found.converged)


@pytest.mark.parametrize("selector", points.SELECTORS)
def test_under_w0_the_first_point_reads_the_likelihood_power(trajectories, selector):
    traj = trajectories[selector]
    for name, power in [("r0", 1.0), ("independence", 1.0), ("entrench", 1.0), ("fixed", 1.0), ("over", 2.0), ("under", 0.5),
                        ("entrench+over", 2.0), ("fixed+under", 0.5)]:
        assert abs(regime_check.first_point(traj, name, "W0")["log_tau_b"] - np.log(power)) <= RESOLUTION, name
    for power in figure1.POWERS:
        for delta, drift in ((0.0, "none"), (0.8, "entrench"), (0.8, "fixed")):
            row = regime_check.first_point(traj, regime_check.grid_name(power, delta, drift), "W0")
            assert abs(row["log_tau_b"] - np.log(power)) <= RESOLUTION, (power, delta, drift)
            assert row["clipped"] == 0.0 and row["fallback"] == 0.0


def synthetic_rows(cfg, shift=None):
    """Rows whose log tau_B is log power plus a small seed jitter, with optional per-reader shifts."""
    rng = np.random.default_rng(0)
    rows = []
    for seed in cfg.seeds:
        for world in regime_check.WORLD_MODELS:
            for name in regime_check.reader_names():
                power = float(name.split(":")[1]) if name.startswith("grid:") else regime_check.POWER.get(name, 1.0)
                for selector in points.SELECTORS:
                    value = np.log(power) + 0.001 * rng.standard_normal() + (shift or {}).get((world, name), 0.0)
                    rows.append({"seed": seed, "variant": cfg.primary_variant, "selector": selector, "world_model": world, "reader": name,
                                 "test": "CT", "log_tau_b": value, "log_tau_b_range": 0.0, "kl_inc_at_tau_b": 0.0,
                                 "clipped": 0.0, "fallback": 0.0, "search_converged": 1.0, "n_points": 1})
    return rows


def test_criteria_hold_on_readers_at_their_power_and_fail_when_moved():
    cfg = tiny(seeds=tuple(range(6)), n_boot=500)
    assert regime_check.evaluate(synthetic_rows(cfg), cfg)["validated"]

    under_wrong_side = regime_check.evaluate(synthetic_rows(cfg, {("W2", "under"): np.log(2.2)}), cfg)
    assert not under_wrong_side["validated"]
    assert under_wrong_side["primary"]["W0"]["sides_hold"] and not under_wrong_side["primary"]["W2"]["sides_hold"]

    independence_off = regime_check.evaluate(synthetic_rows(cfg, {("W0", "independence"): 0.05}), cfg)
    assert not independence_off["primary"]["W0"]["families"]["independence"]["holds"]
    within_resolution = regime_check.evaluate(synthetic_rows(cfg, {("W0", "independence"): 0.5 * RESOLUTION}), cfg)
    assert within_resolution["primary"]["W0"]["families"]["independence"]["holds"]

    hump = regime_check.evaluate(synthetic_rows(cfg, {("W2", regime_check.grid_name(4.0, 0.2, "fixed")): -np.log(4.0)}), cfg)
    assert hump["primary"]["W0"]["ordering"]["holds"] and not hump["primary"]["W2"]["ordering"]["holds"]
    broken = [line for line in hump["primary"]["W2"]["ordering"]["lines"] if not line["rises"]]
    assert [(line["drift"], line["delta"]) for line in broken] == [("fixed", 0.2)]

    r0_off = regime_check.evaluate(synthetic_rows(cfg, {("W0", "r0"): 0.3}), cfg)
    assert r0_off["validated"] and not r0_off["primary"]["W0"]["families"]["r0"]["holds"]


def test_grid_has_81_readers_and_covers_every_r4_family_by_parameters():
    names = regime_check.reader_names()
    assert len(names) == 1 + len(config.R4) + 81 and len(set(names)) == len(names)
    assert len(regime_check.ordering(synthetic_rows(tiny()), "W0", tiny().seeds)["lines"]) == 9
