"""Decision-point quantities on the full twin, along the trajectories of the evaluation patients.

The card's second source of exactness: at each step of each patient's trajectory the quantities of
parts B and C are exact over the outcome distribution at that decision point. A decision point is a
history and the test the acquisition rule requests there; the regrets look at every test not yet
requested. Episode log loss at the horizon is Monte Carlo over the patients and is labeled so.

Definitions used here (the card names the quantities; these are the formulas):

    true net value of a test, for a reader and layer
        sum_o P*(o) P*(Y = argmax of the belief held after o) - P*(Y = argmax of the belief now)
        - lambda x P*(available) x cost
    one-step regret of a score   max(0, best true net value) - true net value of the test the score
        picks (the highest positive score; no positive score means stop, worth 0)
    raw score         entropy-change surrogate under the world model - lambda x expected charged cost
    coherent score    coherent VOI of the projected pair - lambda x expected charged cost
    probes            KL(read(h, a, UNAVAILABLE) || b) and KL(read(h, previous result again) || b)
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tiicee.cee import bootstrap_upper_bound, entropy, floor_norm, kl
from tiicee.experiments.e1 import batch, filters
from tiicee.experiments.e1.context import Context
from tiicee.twin.env import stream

SELECTORS = ("rand", "ig")


class Trajectories:
    """Patients of one (seed, variant) followed for the full twin's horizon under one acquisition rule."""

    def __init__(self, ctx: Context, selector: str, calibration: bool = False):
        cfg, twin = ctx.cfg, ctx.twin("full")
        m = twin.model
        self.ctx, self.selector, self.model, self.horizon = ctx, selector, m, ctx.horizon("full")
        if calibration:         # CEE-Gate bounds: fitting-stream patients that no fit and no evaluation uses
            patients = twin.sample_patients(cfg.n_train + cfg.n_gate, ctx.seed, "train", purpose="fitting")[cfg.n_train:]
        else:
            patients = twin.sample_patients(cfg.n_eval, ctx.seed, cfg.eval_split)
        self.fingerprints = [p.fingerprint for p in patients]
        self.n = n = len(patients)
        self.y = np.array([m.classes.index(p.diagnosis) for p in patients])
        observed = np.array([[m.outcomes.index(twin.observe(p, t)) for t in m.tests] for p in patients])
        truth = filters.LatentFilter(m)
        rows = np.arange(n)
        w, seen = truth.start(n), filters.new_seen(n, m)
        asked = np.zeros((n, len(m.tests)), dtype=bool)
        order = ctx.order("full")
        if selector == "patient":   # the sandbox's rule: a fresh random order for every patient
            n_tests = len(m.tests)
            own = stream(ctx.seed, "behavior", cfg.eval_split, n_tests, 1).permuted(np.tile(np.arange(n_tests), (n, 1)), axis=1)
        self.steps = []
        for t in range(self.horizon):
            joint_all = truth.joint_all(w, seen)
            joint_all = joint_all / joint_all.sum((2, 3), keepdims=True)                # P*(class, outcome | history, test)
            if selector == "rand":
                tests = np.full(n, order[t])
            elif selector == "patient":
                tests = own[:, t]
            elif selector == "ig":
                info = batch.information(joint_all.reshape(-1, *joint_all.shape[2:])).reshape(n, -1)
                tests = np.argmax(np.where(asked, -1.0, info), axis=1)
            else:
                raise ValueError(f"unknown acquisition rule {selector!r}")
            outcomes = observed[rows, tests]
            posterior = w.sum(2) / w.sum((1, 2))[:, None]
            self.steps.append(SimpleNamespace(tests=tests, outcomes=outcomes, seen=seen, asked=asked.copy(),
                                              joint_all=joint_all, posterior=posterior))
            w = truth.advance(w, seen, tests, outcomes, keep_impossible=False)
            w = w / w.sum((1, 2), keepdims=True)
            seen = filters.record(seen, truth.item_of, tests, outcomes)
            asked[rows, tests] = True
        self._states = {}

    def states(self, kind: str, name: str):
        """(the vectorized reader or world model, its state before every step and after the last)."""
        if (kind, name) not in self._states:
            size = "full"
            component = filters.reader(self.model, name) if kind == "reader" else self.ctx.world_filter(size, name)
            states = [component.start(self.n)]
            for step in self.steps:
                if kind == "reader":
                    states.append(component.advance(states[-1], step.seen, step.tests, step.outcomes))
                else:
                    states.append(component.advance(states[-1], step.seen, step.tests, step.outcomes, keep_impossible=True))
            self._states[(kind, name)] = (component, states)
        return self._states[(kind, name)]

    def pairs(self, t: int, reader_name: str) -> SimpleNamespace:
        """Everything a reader brings to step t, for every (patient, unrequested test) pair, patient by patient."""
        step = self.steps[t]
        reader, states = self.states("reader", reader_name)
        rows, tests = np.nonzero(~step.asked)
        k = len(self.model.tests) - t
        bf = floor_norm(reader.belief(states[t]))
        q_raw = reader.reactions_all(states[t], step.seen)[rows, tests]
        joint = step.joint_all[rows, tests]
        p_star = joint.sum(1)
        taken = np.arange(self.n) * k + np.argmax(tests.reshape(self.n, k) == step.tests[:, None], axis=1)
        return SimpleNamespace(
            rows=rows, tests=tests, k=k, taken=taken, bf=bf, bf_m=bf[rows], q_raw=q_raw, q=floor_norm(q_raw),
            joint=joint, p_star=p_star, b_star=step.posterior, absent=len(self.model.outcomes) - 1,
            cost_true=(1.0 - p_star[:, -1]) * self.model.cost_norm[tests],
            hit_now=step.posterior[np.arange(self.n), np.argmax(bf, axis=1)][rows],
        )


def world_predictive(traj: Trajectories, t: int, world_name: str, pr) -> np.ndarray:
    world, states = traj.states("world", world_name)
    return filters.predictive(world.joint_all(states[t], traj.steps[t].seen))[pr.rows, pr.tests]


def true_net_value(pr, posteriors, lam: float) -> np.ndarray:
    """[patient, unrequested test]: what requesting the test is truly worth to an agent that will hold these posteriors."""
    choice = np.argmax(posteriors, axis=2)                                  # the class committed to after each outcome
    hit = np.take_along_axis(pr.joint, choice[:, None, :], axis=1)[:, 0, :].sum(1)
    return (hit - pr.hit_now - lam * pr.cost_true).reshape(-1, pr.k)


def regret(score, value) -> np.ndarray:
    """One-step regret of choosing by score (the highest positive one, else stop) against the best true net value."""
    rows = np.arange(len(score))
    best = np.argmax(score, axis=1)
    achieved = np.where(score[rows, best] > 0, value[rows, best], 0.0)
    return np.maximum(value.max(1), 0.0) - achieved


def expected_log_loss(joint, posteriors) -> np.ndarray:
    """-sum over classes and outcomes of P*(y, o | h, a) log bel_o(y)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(joint > 0, joint * np.log(np.swapaxes(posteriors, 1, 2)), 0.0).sum((1, 2))


def _project(pr, weights):
    projection = batch.project(pr.bf_m, weights, pr.q)
    posteriors = np.where(projection.converged[:, None, None], projection.corrected, pr.q)   # raw reader on non-convergence
    return projection, posteriors


# ---------------------------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------------------------
def gate_bounds(calibration: Trajectories, reader_name: str, world_name: str) -> np.ndarray:
    """CEE-Gate thresholds eps_prior + eps_wm per test (specification 5.8), from the calibration patients.

    Each term is estimated as on charts, from the factual result and label of each decision point:
    prior term log u(y) - sum_y b log u, world-model term -(log Z_real - sum_o P_hat log Z_o). The
    bound is the upper 90% bootstrap bound of the absolute mean, resampling patients. A test with no
    calibration decision point gets an infinite threshold: the gate never opens there.
    """
    cfg, m = calibration.ctx.cfg, calibration.model
    rows = np.arange(calibration.n)
    prior_term, world_term, test = [], [], []
    for t, step in enumerate(calibration.steps):
        pr = calibration.pairs(t, reader_name)
        weights = world_predictive(calibration, t, world_name, pr)[pr.taken]
        projection = batch.project(pr.bf, weights, pr.q[pr.taken])
        log_u = np.log(projection.u)
        prior_term.append(log_u[rows, calibration.y] - (pr.bf * log_u).sum(1))
        world_term.append(-(projection.log_z[rows, step.outcomes] - (weights * projection.log_z).sum(1)))
        test.append(step.tests)
    prior_term, world_term, test = (np.concatenate(v) for v in (prior_term, world_term, test))
    patient = np.tile(rows, calibration.horizon)
    bounds = np.full(len(m.tests), np.inf)
    for a in np.unique(test):
        here = test == a
        rng = stream(calibration.ctx.seed, "montecarlo", "train", int(a))
        bounds[a] = sum(bootstrap_upper_bound(values[here], patient[here], rng, level=0.90, n_boot=cfg.n_boot, absolute=True)
                        for values in (prior_term, world_term))
    return bounds


def part_b_cell(traj: Trajectories, reader_name: str, world_name: str, gate: np.ndarray, spearman) -> dict:
    """One reader under one world model: the decision-point quantities of part B, averaged over the decision points."""
    cfg, n = traj.ctx.cfg, traj.n
    rows = np.arange(n)
    reader, r_states = traj.states("reader", reader_name)
    collected = {}

    def keep(name, values):
        collected.setdefault(name, []).append(np.asarray(values, dtype=float))

    stateless = stateful = floor_norm(reader.belief(r_states[0]))
    for t, step in enumerate(traj.steps):
        pr = traj.pairs(t, reader_name)
        weights = world_predictive(traj, t, world_name, pr)
        projection, posteriors = _project(pr, weights)
        cost = (1.0 - weights[:, -1]) * traj.model.cost_norm[pr.tests]
        raw_score = entropy(pr.bf_m) - (weights * entropy(pr.q)).sum(1) - cfg.lam * cost
        coherent_score = (weights * posteriors.max(2)).sum(1) - pr.bf_m.max(1) - cfg.lam * cost
        keep("regret_raw", regret(raw_score.reshape(n, pr.k), true_net_value(pr, pr.q, cfg.lam)))
        keep("regret_coherent", regret(coherent_score.reshape(n, pr.k), true_net_value(pr, posteriors, cfg.lam)))

        j = pr.taken                                                        # the decision point itself
        w, q, joint, p_star = weights[j], pr.q[j], pr.joint[j], pr.p_star[j]
        u, log_z, ok = projection.u[j], projection.log_z[j], projection.converged[j]
        log_u = np.log(u)
        keep("converged", ok)
        keep("drift_l1", np.abs(np.einsum("no,noc->nc", w, q) - pr.bf).sum(1))
        keep("responsiveness", batch.responsiveness(w, q))
        keep("kl_inc", np.where(ok, projection.kl_inc[j], np.nan))
        budget = batch.information(joint)
        search, at_tau = batch.scale(pr.bf, w, q, budget, "two-sided")
        keep("log_tau_b", np.log(search.tau))
        keep("tau_b_clipped", search.path != 0)
        keep("kl_inc_at_tau_b", np.where(at_tau.converged, at_tau.kl_inc, np.nan))
        ratio = batch.tau_ratio(pr.bf, w, q)
        keep("log_tau_ratio", np.log(ratio))
        keep("probe_unavailable", kl(q[:, -1], pr.bf))
        if t:
            previous = traj.steps[t - 1]
            again = reader.reactions(r_states[t], step.seen, previous.tests)[rows, previous.outcomes]
            keep("probe_reexposure", kl(floor_norm(again), pr.bf))
        prior_term = ((pr.b_star - pr.bf) * log_u).sum(1)
        world_term = -((p_star - w) * log_z).sum(1)
        gain = projection.kl_inc[j] + prior_term + world_term
        for name, values in (("gain_incoherence", projection.kl_inc[j]), ("gain_prior", prior_term), ("gain_world_model", world_term), ("gain", gain)):
            keep(name, np.where(ok, values, np.nan))
        raw_loss, stateless_loss = expected_log_loss(joint, q), expected_log_loss(joint, posteriors[j])
        keep("log_loss_raw", raw_loss)
        keep("log_loss_stateless", stateless_loss)
        keep("identity_error", np.where(ok, np.abs((raw_loss - stateless_loss) - gain), 0.0))
        tracked = batch.stateful_step(stateful, pr.bf, w, q)
        keep("log_loss_stateful", expected_log_loss(joint, tracked.beliefs))
        allowed = ok & (projection.kl_inc[j] > gate[step.tests])
        keep("gate_allowed", allowed)
        keep("gate_gain", np.where(allowed, gain, np.nan))
        keep("gate_harmful", allowed & (gain < 0))
        keep("ungated_harmful", ok & (gain < 0))
        stateless, stateful = posteriors[j][rows, step.outcomes], tracked.beliefs[rows, step.outcomes]

    flat = {name: np.concatenate(values) for name, values in collected.items()}
    with np.errstate(invalid="ignore"):
        row = {name: float(np.nanmean(values)) if np.isfinite(values).any() else None for name, values in flat.items()}
    row["identity_error"] = float(flat["identity_error"].max())
    row["n_points"] = int(len(flat["converged"]))
    defined = np.isfinite(flat["log_tau_ratio"])
    row["tau_ratio_defined"] = float(defined.mean())
    row["tau_ratio_spearman"] = spearman(flat["log_tau_ratio"][defined], flat["log_tau_b"][defined]) if defined.sum() > 2 else None
    row["tau_ratio_abs_log_ratio"] = float(np.abs(flat["log_tau_ratio"][defined] - flat["log_tau_b"][defined]).mean()) if defined.any() else None
    final = floor_norm(reader.belief(r_states[-1]))
    for name, belief in (("raw", final), ("stateless", stateless), ("stateful", stateful)):       # Monte Carlo over patients
        row[f"mc_log_loss_{name}"] = float(-np.log(belief[rows, traj.y]).mean())
        row[f"mc_accuracy_{name}"] = float((np.argmax(belief, axis=1) == traj.y).mean())
    return row


SANDBOX_CELLS = (("base", "independence"), ("base", "entrench"), ("base", "fixed"), ("base", "over"), ("strong_correlation", "independence"))


def sandbox_rows(cfg, seed: int, spearman) -> list:
    """Episode log loss at the horizon in the sandbox's configuration (cee_check.py part 4), Monte Carlo over patients.

    Every test in every chart, a fresh random order per patient, W0 as the world model, the
    stateful track. Atypical rate 0.07 for four readers and 0.20 for the independence reader.
    """
    rows = []
    for variant in dict.fromkeys(v for v, _ in SANDBOX_CELLS):
        traj = Trajectories(Context(cfg, seed, variant, missingness=False), "patient")
        never = np.full(len(traj.model.tests), np.inf)
        for reader in (r for v, r in SANDBOX_CELLS if v == variant):
            cell = part_b_cell(traj, reader, "W0", never, spearman)
            rows.append({"seed": seed, "variant": variant, "reader": reader, "n_patients": traj.n, "horizon": traj.horizon,
                         **{k: cell[k] for k in ("mc_log_loss_raw", "mc_log_loss_stateful", "mc_accuracy_raw", "mc_accuracy_stateful")}})
    return rows


def regime_coordinates(traj: Trajectories, reader_name: str, world_name: str) -> dict:
    """The Figure 1 coordinates of a reader alone: mean log tau_B (two-sided, exact budget) and mean KL_inc at tau_B."""
    log_tau, kl_inc = [], []
    for t in range(traj.horizon):
        pr = traj.pairs(t, reader_name)
        j = pr.taken
        weights = world_predictive(traj, t, world_name, pr)[j]
        search, at_tau = batch.scale(pr.bf, weights, pr.q[j], batch.information(pr.joint[j]), "two-sided")
        log_tau.append(np.log(search.tau))
        kl_inc.append(at_tau.kl_inc[at_tau.converged])
    return {"log_tau_b": float(np.concatenate(log_tau).mean()), "kl_inc_at_tau_b": float(np.concatenate(kl_inc).mean())}


# ---------------------------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------------------------
def error_names(cfg) -> tuple:
    return tuple(f"mirror({a:g})" for a in cfg.mirror) + tuple(f"temperature({t:g})" for t in cfg.temperatures) + (
        "matched_ignored", "matched_responsive")


def error_predictives(traj: Trajectories, t: int, pr) -> dict:
    """The predictive of every controlled error of the card at every pair of step t, and how many matched points took the slow route."""
    cfg = traj.ctx.cfg
    out = {f"mirror({a:g})": world_predictive(traj, t, f"mirror({a:g})", pr) for a in cfg.mirror}
    out.update({f"temperature({temp:g})": filters.temperature(pr.p_star, temp) for temp in cfg.temperatures})
    out["matched_ignored"], out["matched_responsive"], slow = filters.matched_pair(pr.p_star, pr.q_raw, cfg.matched_kl)
    return out, slow


def part_c_cell(traj: Trajectories, reader_name: str) -> dict:
    """One reader against every controlled error: per history and error, the regret of CEE-Select and what could explain it.

    Returns arrays [step, patient, error]. Per-history predictors are means over the unrequested
    tests (the *_max entries hold the largest instead); the log-loss change and its predictors are
    taken at the test the trajectory requests.
    """
    cfg, n = traj.ctx.cfg, traj.n
    names = error_names(cfg)
    fields = ("regret", "kl", "calibration", "e_w", "kl_max", "e_w_max", "log_loss_change", "kl_taken", "calibration_taken", "e_w_taken")
    out = {f: np.empty((traj.horizon, n, len(names))) for f in fields}
    slow = 0
    for t in range(traj.horizon):
        pr = traj.pairs(t, reader_name)
        truth_posteriors = _project(pr, pr.p_star)[1]
        baseline = expected_log_loss(pr.joint[pr.taken], truth_posteriors[pr.taken])
        predictives, slow_here = error_predictives(traj, t, pr)
        slow += slow_here
        for e, name in enumerate(names):
            weights = predictives[name]
            projection, posteriors = _project(pr, weights)
            cost = (1.0 - weights[:, -1]) * traj.model.cost_norm[pr.tests]
            score = (weights * posteriors.max(2)).sum(1) - pr.bf_m.max(1) - cfg.lam * cost
            out["regret"][t, :, e] = regret(score.reshape(n, pr.k), true_net_value(pr, posteriors, cfg.lam))
            divergence = kl(pr.p_star, weights)
            top = np.argmax(weights, axis=1)[:, None]
            calibration = np.abs(np.take_along_axis(weights, top, 1) - np.take_along_axis(pr.p_star, top, 1))[:, 0]
            gap = pr.p_star - weights
            decision_error = np.abs((gap * posteriors.max(2)).sum(1))            # E_W with the committed utility
            reading_error = np.abs((gap * projection.log_z).sum(1))              # E_W with log Z
            for field, values in (("kl", divergence), ("calibration", calibration), ("e_w", decision_error)):
                out[field][t, :, e] = values.reshape(n, pr.k).mean(1)
            out["kl_max"][t, :, e] = divergence.reshape(n, pr.k).max(1)
            out["e_w_max"][t, :, e] = decision_error.reshape(n, pr.k).max(1)
            out["log_loss_change"][t, :, e] = expected_log_loss(pr.joint[pr.taken], posteriors[pr.taken]) - baseline
            out["kl_taken"][t, :, e] = divergence[pr.taken]
            out["calibration_taken"][t, :, e] = calibration[pr.taken]
            out["e_w_taken"][t, :, e] = reading_error[pr.taken]
    out["errors"], out["slow_matched_points"] = names, slow
    return out
