"""Decision-point quantities of parts B and C on the full twin against the library and the twin's objects, one point at a time."""
from __future__ import annotations

import numpy as np
import pytest

from tiicee import cee
from tiicee.experiments.e1 import points
from tiicee.experiments.e1.config import tiny
from tiicee.experiments.e1.context import Context
from tiicee.twin import world_models as wm
from tiicee.twin.env import stream
from tiicee.twin.readers import BayesReader, SimulatedReader

OPTIONS = {"independence": {}, "entrench": {"drift": "entrench"}, "over": {"power": 2.0}, "fixed+under": {"drift": "fixed", "power": 0.5}}


def reader_object(twin, name):
    return BayesReader(twin) if name == "r0" else SimulatedReader(twin, **OPTIONS[name])


def dense(model, predictive):
    out = np.zeros(len(model.outcomes))
    for outcome, weight in zip(predictive.outcomes, predictive.weights):
        out[model.outcomes.index(outcome)] += weight
    return out


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)       # no ties in these tests
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


@pytest.fixture(scope="module", params=["base", "duplicated_panel"])
def ctx(request):
    return Context(tiny(n_train=300, n_eval=10, n_gate=40, n_boot=300), 0, request.param)


@pytest.fixture(scope="module")
def trajectories(ctx):
    return {name: points.Trajectories(ctx, name) for name in points.SELECTORS}


def histories(traj, i):
    m = traj.model
    out = [()]
    for step in traj.steps:
        out.append(out[-1] + ((m.tests[step.tests[i]], m.outcomes[step.outcomes[i]]),))
    return out


def test_trajectories_follow_the_patients_and_the_acquisition_rules(ctx, trajectories):
    twin, cfg = ctx.twin("full"), ctx.cfg
    m = twin.model
    patients = twin.sample_patients(cfg.n_eval, ctx.seed, cfg.eval_split)
    assert cfg.eval_split == "validation"
    for name, traj in trajectories.items():
        assert traj.fingerprints == [p.fingerprint for p in patients]            # every arm sees these cases
        assert traj.horizon == 3 and traj.n == cfg.n_eval
        for i, patient in enumerate(patients):
            assert m.classes[traj.y[i]] == patient.diagnosis
            for t, history in enumerate(histories(traj, i)[:-1]):
                step = traj.steps[t]
                test = m.tests[step.tests[i]]
                if name == "rand":
                    assert step.tests[i] == ctx.order("full")[t]
                else:
                    best = max(twin.eligible(history), key=lambda a: (twin.true_information(history, a), -m.tests.index(a)))
                    assert test == best
                assert m.outcomes[step.outcomes[i]] == twin.observe(patient, test)
                np.testing.assert_allclose(step.posterior[i], twin.true_posterior(history), atol=1e-12)
                assert [m.tests[a] for a in np.flatnonzero(~step.asked[i])] == list(twin.eligible(history))
    calibration = points.Trajectories(ctx, "rand", calibration=True)
    extra = twin.sample_patients(cfg.n_train + cfg.n_gate, ctx.seed, "train", purpose="fitting")[cfg.n_train:]
    assert calibration.fingerprints == [p.fingerprint for p in extra] and calibration.n == cfg.n_gate


def test_sandbox_configuration_has_every_test_available_and_an_order_per_patient():
    cfg = tiny(n_train=300, n_eval=10, n_gate=40, n_boot=300)
    plain = Context(cfg, 0, "base", missingness=False)
    assert (plain.twin("full").model.availability == 1.0).all() and (plain.twin("reduced").model.availability == 1.0).all()
    traj = points.Trajectories(plain, "patient")
    orders = np.stack([step.tests for step in traj.steps], axis=1)
    assert all(len(set(row)) == traj.horizon for row in orders.tolist()) and len({tuple(row) for row in orders.tolist()}) > 1
    assert all((step.outcomes != len(traj.model.outcomes) - 1).all() for step in traj.steps)           # nothing is unavailable
    np.testing.assert_array_equal(orders, np.stack([s.tests for s in points.Trajectories(plain, "patient").steps], axis=1))
    rows = points.sandbox_rows(cfg, 0, spearman)
    assert [(r["variant"], r["reader"]) for r in rows] == list(points.SANDBOX_CELLS) and all(r["n_patients"] == 10 for r in rows)
    assert all(np.isfinite([r["mc_log_loss_raw"], r["mc_log_loss_stateful"]]).all() for r in rows)


def point_reference(ctx, traj, reader_name, world, i, t, lam):
    """One decision point through the library: scores and true net values of every eligible test, and the audit of the one taken."""
    twin = ctx.twin("full")
    m = twin.model
    reader = reader_object(twin, reader_name)
    history = histories(traj, i)[t]
    taken = m.tests[traj.steps[t].tests[i]]
    belief = reader.belief(history)
    truth_now = twin.true_posterior(history)
    utility = cee.identity_utility(len(m.classes))
    out = {"raw": [], "coherent": []}
    for test in twin.eligible(history):
        weights = dense(m, world.outcomes(history, test, None, None))
        posteriors = np.array([reader.read(history, test, o) for o in m.outcomes])
        projection = cee.project(belief, weights, posteriors)
        cost = cee.expected_charged_cost(1.0 - weights[-1], float(m.cost_norm[m.tests.index(test)]))
        floored = cee.floor_norm(posteriors)
        corrected = projection.corrected if projection.converged else floored
        scores = {"raw": cee.entropy_change_surrogate(belief, weights, posteriors).score - lam * cost,
                  "coherent": cee.coherent_voi(projection.prior, projection.weights, corrected, utility) - lam * cost}
        p_star = dense(m, twin.true_predictive(history, test))
        for arm, held in (("raw", floored), ("coherent", corrected)):
            value = -truth_now[np.argmax(cee.floor_norm(belief))] - lam * twin.expected_charged_cost(history, test)
            for o, outcome in enumerate(m.outcomes):
                if p_star[o] > 0:
                    value += p_star[o] * twin.true_posterior(history + ((test, outcome),))[np.argmax(held[o])]
            out[arm].append((scores[arm], value))
        if test == taken:
            record = cee.audit_record({"id": test, "classes": m.classes, "prior": belief, "outcomes": m.outcomes, "weights": weights,
                                       "posteriors": posteriors, "budget": twin.true_information(history, test)})
            terms = cee.gain_terms(projection.prior, projection.weights, projection.u, projection.log_z, truth_now, p_star)
            out.update(record=record, terms=terms, unavailable=cee.reaction_kl(belief, posteriors[-1]))
            if t:
                out["reexposure"] = cee.reaction_kl(belief, reader.read(history, *history[-1]))
    for arm in ("raw", "coherent"):
        score, value = max(out[arm], key=lambda sv: sv[0])
        best = max(0.0, max(v for _, v in out[arm]))
        out[f"regret_{arm}"] = best - (value if score > 0 else 0.0)
    return out


@pytest.mark.parametrize("reader_name,world_name", [("independence", "W0"), ("entrench", "W2"), ("over", "W1"), ("r0", "W1"), ("fixed+under", "W2")])
def test_part_b_cell_is_the_mean_of_the_library_audit_over_the_decision_points(ctx, trajectories, reader_name, world_name):
    twin = ctx.twin("full")
    world = wm.w0(twin) if world_name == "W0" else ctx.fitted("full", world_name)
    for name, traj in trajectories.items():
        gate = np.zeros(len(twin.model.tests))
        row = points.part_b_cell(traj, reader_name, world_name, gate, spearman)
        refs = [point_reference(ctx, traj, reader_name, world, i, t, ctx.cfg.lam) for t in range(traj.horizon) for i in range(traj.n)]
        assert row["n_points"] == len(refs)
        mean = lambda values: float(np.mean(list(values)))
        assert row["regret_raw"] == pytest.approx(mean(r["regret_raw"] for r in refs), abs=1e-10)
        assert row["regret_coherent"] == pytest.approx(mean(r["regret_coherent"] for r in refs), abs=1e-10)
        assert row["drift_l1"] == pytest.approx(mean(r["record"]["drift_l1"] for r in refs), abs=1e-10)
        assert row["responsiveness"] == pytest.approx(mean(r["record"]["responsiveness"] for r in refs), abs=1e-10)
        converged = [r for r in refs if r["record"]["converged"]]
        assert row["converged"] == pytest.approx(len(converged) / len(refs))
        assert row["kl_inc"] == pytest.approx(mean(r["record"]["kl_inc"] for r in converged), abs=1e-10)
        assert row["log_tau_b"] == pytest.approx(mean(np.log(r["record"]["tau_b_two_sided"]) for r in refs), abs=1e-9)
        assert row["tau_b_clipped"] == pytest.approx(mean(r["record"]["tau_b_two_sided_path"] != "solved" for r in refs))
        ratios = [r["record"]["tau_ratio"] for r in refs if r["record"]["tau_ratio"] is not None]
        assert row["tau_ratio_defined"] == pytest.approx(len(ratios) / len(refs))
        if ratios:
            assert row["log_tau_ratio"] == pytest.approx(mean(np.log(v) for v in ratios), abs=1e-12)
        assert row["probe_unavailable"] == pytest.approx(mean(r["unavailable"] for r in refs), abs=1e-12)
        assert row["probe_reexposure"] == pytest.approx(mean(r["reexposure"] for r in refs if "reexposure" in r), abs=1e-12)
        for field, attribute in (("gain_incoherence", "incoherence"), ("gain_prior", "prior_calibration"), ("gain_world_model", "world_model"), ("gain", "total")):
            assert row[field] == pytest.approx(mean(getattr(r["terms"], attribute) for r in converged), abs=1e-10), field
        assert row["identity_error"] < 1e-10                     # the gain identity holds at every point


def test_probes_separate_drift_from_double_counting(ctx, trajectories):
    traj = trajectories["rand"]
    gate = np.zeros(len(traj.model.tests))
    rows = {name: points.part_b_cell(traj, name, "W0", gate, spearman) for name in ("r0", "independence", "over", "entrench")}
    assert rows["r0"]["probe_unavailable"] < 1e-12 and rows["r0"]["probe_reexposure"] < 1e-12
    assert rows["independence"]["probe_unavailable"] < 1e-12 < rows["independence"]["probe_reexposure"]    # no drift, but double counting
    assert rows["over"]["probe_unavailable"] < 1e-12
    assert rows["entrench"]["probe_unavailable"] > 1e-4                                                     # drift moves whatever the result
    assert abs(rows["r0"]["gain"]) < 1e-9 and rows["r0"]["kl_inc"] < 1e-9


def test_monte_carlo_episode_log_loss_follows_the_tracks(ctx, trajectories):
    twin = ctx.twin("full")
    m = twin.model
    traj = trajectories["ig"]
    reader, world = reader_object(twin, "entrench"), ctx.fitted("full", "W2")
    row = points.part_b_cell(traj, "entrench", "W2", np.zeros(len(m.tests)), spearman)
    losses = {"raw": [], "stateless": [], "stateful": []}
    for i in range(traj.n):
        path = histories(traj, i)
        stateful = cee.floor_norm(reader.belief(()))
        for t in range(traj.horizon):
            test, outcome = path[t + 1][-1]
            weights = dense(m, world.outcomes(path[t], test, None, None))
            posteriors = [reader.read(path[t], test, o) for o in m.outcomes]
            index = m.outcomes.index(outcome)
            stateless = cee.stateless_step(reader.belief(path[t]), weights, posteriors, index).belief
            stateful = cee.stateful_step(stateful, reader.belief(path[t]), weights, posteriors, index).belief
        final = {"raw": cee.floor_norm(reader.belief(path[-1])), "stateless": stateless, "stateful": stateful}
        for arm, belief in final.items():
            losses[arm].append(-np.log(belief[traj.y[i]]))
    for arm, values in losses.items():
        assert row[f"mc_log_loss_{arm}"] == pytest.approx(np.mean(values), abs=1e-10)


def test_gate_bounds_are_bootstrap_bounds_of_the_factual_terms(ctx):
    twin, cfg = ctx.twin("full"), ctx.cfg
    m = twin.model
    calibration = points.Trajectories(ctx, "rand", calibration=True)
    reader, world = reader_object(twin, "entrench"), ctx.fitted("full", "W1")
    bounds = points.gate_bounds(calibration, "entrench", "W1")
    terms = {}
    for i in range(calibration.n):
        path = histories(calibration, i)
        for t in range(calibration.horizon):
            test, outcome = path[t + 1][-1]
            weights = dense(m, world.outcomes(path[t], test, None, None))
            projection = cee.project(reader.belief(path[t]), weights, [reader.read(path[t], test, o) for o in m.outcomes])
            log_u = np.log(projection.u)
            prior = log_u[calibration.y[i]] - projection.prior @ log_u
            model = -(projection.log_z[m.outcomes.index(outcome)] - weights @ projection.log_z)
            terms.setdefault(m.tests.index(test), []).append((i, prior, model))
    assert sorted(terms) == sorted(ctx.order("full")[:calibration.horizon].tolist())
    for a in range(len(m.tests)):
        if a not in terms:
            assert bounds[a] == np.inf
            continue
        patient = np.array([r[0] for r in terms[a]])
        rng = stream(ctx.seed, "montecarlo", "train", a)
        expected = sum(cee.bootstrap_upper_bound(np.array([r[k] for r in terms[a]]), patient, rng, level=0.90, n_boot=cfg.n_boot, absolute=True)
                       for k in (1, 2))
        assert bounds[a] == pytest.approx(expected, abs=1e-10) and bounds[a] > 0
    gated = points.part_b_cell(points.Trajectories(ctx, "rand"), "entrench", "W1", bounds, spearman)
    opened = points.part_b_cell(points.Trajectories(ctx, "rand"), "entrench", "W1", np.zeros(len(m.tests)), spearman)
    assert gated["gate_allowed"] <= opened["gate_allowed"] and opened["gate_allowed"] == pytest.approx(opened["converged"])
    assert opened["gate_gain"] == pytest.approx(opened["gain"], abs=1e-12)


def test_part_c_cell_matches_the_controlled_error_objects(ctx, trajectories):
    twin, cfg = ctx.twin("full"), ctx.cfg
    m = twin.model
    traj = trajectories["rand"]
    utility = cee.identity_utility(len(m.classes))
    for reader_name in ("independence", "r0"):
        reader = reader_object(twin, reader_name)
        ignored, responsive = wm.matched_pair(twin, reader, kl=cfg.matched_kl)
        worlds = {"mirror(0.2)": wm.mirror_corrupt(twin, 0.2), "temperature(2)": wm.temperature(twin, 2.0),
                  "matched_ignored": ignored, "matched_responsive": responsive}
        got = points.part_c_cell(traj, reader_name)
        assert got["errors"] == ("mirror(0.1)", "mirror(0.2)", "mirror(0.4)", "temperature(0.5)", "temperature(2)", "matched_ignored", "matched_responsive")
        truth = wm.w0(twin)
        for name, world in worlds.items():
            e = got["errors"].index(name)
            for i in range(0, traj.n, 3):
                for t in range(traj.horizon):
                    history = histories(traj, i)[t]
                    taken = m.tests[traj.steps[t].tests[i]]
                    belief, truth_now = reader.belief(history), twin.true_posterior(history)
                    rows = []
                    for test in twin.eligible(history):
                        p_star = dense(m, twin.true_predictive(history, test))
                        weights = dense(m, world.outcomes(history, test, None, None))
                        posteriors = [reader.read(history, test, o) for o in m.outcomes]
                        projection = cee.project(belief, weights, posteriors)
                        cost = cee.expected_charged_cost(1.0 - weights[-1], float(m.cost_norm[m.tests.index(test)]))
                        score = cee.coherent_voi(projection.prior, projection.weights, projection.corrected, utility) - cfg.lam * cost
                        value = -truth_now[np.argmax(cee.floor_norm(belief))] - cfg.lam * twin.expected_charged_cost(history, test)
                        loss = 0.0
                        for o, outcome in enumerate(m.outcomes):
                            if p_star[o] > 0:
                                after = twin.true_posterior(history + ((test, outcome),))
                                value += p_star[o] * after[np.argmax(projection.corrected[o])]
                                loss -= p_star[o] * after @ np.log(projection.corrected[o])
                        top = int(np.argmax(weights))
                        rows.append({"test": test, "score": score, "value": value, "loss": loss, "kl": float(cee.kl(p_star, weights)),
                                     "calibration": abs(weights[top] - p_star[top]),
                                     "decision": abs((p_star - weights) @ projection.corrected.max(1)),
                                     "reading": abs((p_star - weights) @ projection.log_z)})
                    chosen = max(rows, key=lambda r: r["score"])
                    regret = max(0.0, max(r["value"] for r in rows)) - (chosen["value"] if chosen["score"] > 0 else 0.0)
                    assert got["regret"][t, i, e] == pytest.approx(regret, abs=1e-9), (name, i, t)
                    assert got["kl"][t, i, e] == pytest.approx(np.mean([r["kl"] for r in rows]), abs=1e-9)
                    assert got["kl_max"][t, i, e] == pytest.approx(max(r["kl"] for r in rows), abs=1e-9)
                    assert got["calibration"][t, i, e] == pytest.approx(np.mean([r["calibration"] for r in rows]), abs=1e-9)
                    assert got["e_w"][t, i, e] == pytest.approx(np.mean([r["decision"] for r in rows]), abs=1e-9)
                    assert got["e_w_max"][t, i, e] == pytest.approx(max(r["decision"] for r in rows), abs=1e-9)
                    at = next(r for r in rows if r["test"] == taken)
                    p_star = dense(m, twin.true_predictive(history, taken))
                    clean = cee.project(belief, dense(m, truth.outcomes(history, taken, None, None)), [reader.read(history, taken, o) for o in m.outcomes])
                    clean_loss = -sum(p_star[o] * twin.true_posterior(history + ((taken, outcome),)) @ np.log(clean.corrected[o])
                                      for o, outcome in enumerate(m.outcomes) if p_star[o] > 0)
                    assert got["log_loss_change"][t, i, e] == pytest.approx(at["loss"] - clean_loss, abs=1e-9)
                    assert got["kl_taken"][t, i, e] == pytest.approx(at["kl"], abs=1e-9)
                    assert got["e_w_taken"][t, i, e] == pytest.approx(at["reading"], abs=1e-9)
        pair = [got["errors"].index("matched_ignored"), got["errors"].index("matched_responsive")]
        fresh = ctx.variant == "base"                         # every unrequested test there reads a new item
        if fresh:
            np.testing.assert_allclose(got["kl"][:, :, pair[0]], got["kl"][:, :, pair[1]], rtol=0, atol=1e-6)     # equal KL by construction
            np.testing.assert_allclose(got["kl"][:, :, pair[0]], cfg.matched_kl, rtol=0, atol=1e-6)
        # The ignored member keeps the reader's imagined mean belief, so the reading error (log Z) nearly vanishes.
        # The decision error uses max q', which is not linear in q, so nothing of the kind is promised for it.
        assert got["e_w_taken"][:, :, pair[0]].mean() < 0.05 * got["e_w_taken"][:, :, pair[1]].mean()
