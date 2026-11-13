"""Exact multi-step quantities on the reduced twin: the exact layer of part B, and part D.

Everything here is an expectation over the latent case and every outcome path, computed on the
dense tree of tiicee.experiments.e1.tree. Belief arms:

    raw          the reader, floored once
    stateless    CEE-Read, stateless track (specification 5.9)
    stateful     CEE-Read, stateful track
    scaled       either track with CEE-Scale in front, for a scale variant and a budget source (part D)

Acquisition is one of the card's two fixed rules, so the tree does not depend on the arm:
"rand" (the seed's random order) or "ig" (greedy exact information under W0).
"""
from __future__ import annotations

import numpy as np

from tiicee.cee import floor_norm
from tiicee.experiments.e1 import batch, filters, tree
from tiicee.experiments.e1.config import BUDGETS, SCALE_VARIANTS, TRACKS
from tiicee.experiments.e1.context import Context

SELECTORS = ("rand", "ig")


class Scene:
    """One acquisition rule on one reduced twin: the tree, and each reader's and world model's state on it."""

    def __init__(self, ctx: Context, selector: str):
        self.ctx, self.selector = ctx, selector
        self.twin = ctx.twin("reduced")
        self.model = self.twin.model
        self.horizon = ctx.horizon("reduced")
        if selector == "rand":
            order = ctx.order("reduced")
            choose = lambda level: tree.choose_order(level, order)
        elif selector == "ig":
            choose = lambda level: tree.choose_information(level, self.twin)
        else:
            raise ValueError(f"unknown acquisition rule {selector!r}")
        self.levels = tree.skeleton(self.twin, self.horizon, choose)
        truth = filters.LatentFilter(self.model)
        self.joint = [truth.joint(lv.mass, lv.seen, lv.tests) for lv in self.levels[:-1]]       # P*(class, outcome, history)
        self.exact_budget = [batch.information(j) for j in self.joint]
        self._readers, self._worlds = {}, {}

    def reader(self, name: str):
        """(the vectorized reader, its state at every level)."""
        if name not in self._readers:
            reader = filters.reader(self.model, name)
            states = [reader.start(1)]
            for lv in self.levels[:-1]:
                states.append(tree.flat(reader.expand(states[-1], lv.seen, lv.tests)))
            self._readers[name] = (reader, states)
        return self._readers[name]

    def world(self, name: str):
        """(the vectorized world model, its state at every level, its probability of every path at every level)."""
        if name not in self._worlds:
            world = self.ctx.world_filter("reduced", name)
            states, path = [world.start(1)], [np.ones(1)]
            for lv in self.levels[:-1]:
                p = filters.predictive(world.joint(states[-1], lv.seen, lv.tests))
                path.append(tree.flat(path[-1][:, None] * p))
                states.append(tree.flat(world.expand(states[-1], lv.seen, lv.tests, keep_impossible=True)))
            self._worlds[name] = (world, states, path)
        return self._worlds[name]

    def own_budget(self, name: str, t: int) -> np.ndarray:
        """B-model of a class-conditional world model at level t: its own I(Y; O | h, a)."""
        world, states, _ = self.world(name)
        lv = self.levels[t]
        return batch.information(world.joint(states[t], lv.seen, lv.tests))

    def weight(self, t: int) -> np.ndarray:
        """P*(history) of each node of level t."""
        return self.levels[t].class_mass.sum(1)


def _voi(target, weights, proposals) -> np.ndarray:
    """Coherent value of information with 1[correct] utility; the raw posteriors where the projection did not converge."""
    projection = batch.project(target, weights, proposals)
    posteriors = np.where(projection.converged[:, None, None], projection.corrected, proposals)
    return (weights * posteriors.max(2)).sum(1) - target.max(1)


def _commit_values(scene: Scene, t: int, bf, q_all, world_name: str, beliefs: dict) -> dict:
    """Per track, each node's coherent VOI and expected charged cost for every unrequested test: ([n, k], [n, k])."""
    lv = scene.levels[t]
    world, states, _ = scene.world(world_name)
    rows, tests = np.nonzero(~lv.asked)
    p = filters.predictive(world.joint_all(states[t], lv.seen))[rows, tests]
    q = floor_norm(q_all[rows, tests])
    cost = ((1.0 - p[:, -1]) * scene.model.cost_norm[tests]).reshape(lv.n, -1)
    out = {}
    for track in TRACKS:
        if track == "stateless":
            target, proposals = bf[rows], q
        else:
            target = beliefs[track][rows]
            proposals = target[:, None, :] * q / bf[rows][:, None, :]
            proposals = proposals / proposals.sum(-1, keepdims=True)
        out[track] = (_voi(target, p, proposals).reshape(lv.n, -1), cost)
    return out


def part_b_cell(scene: Scene, reader_name: str, world_name: str) -> dict:
    """Raw, stateless and stateful beliefs of one reader under one world model: exact log loss, accuracy, utility."""
    cfg = scene.ctx.cfg
    reader, r_states = scene.reader(reader_name)
    world, w_states, _ = scene.world(world_name)
    n_out = len(scene.model.outcomes)
    thresholds = (cfg.threshold,) + tuple(cfg.threshold_sensitivity)
    arms = ("raw",) + TRACKS
    stoppers = {(f"threshold_{thr:g}", None, arm): tree.Stopper() for thr in thresholds for arm in arms}
    stoppers.update({("step_limit", None, arm): tree.Stopper() for arm in arms})
    stoppers.update({("cee_commit", lam, track): tree.Stopper() for lam in cfg.lam_frontier for track in TRACKS})
    horizon_rows, fallback = [], {track: 0.0 for track in TRACKS}
    beliefs = None
    for t, lv in enumerate(scene.levels):
        last = t == scene.horizon
        bf = floor_norm(reader.belief(r_states[t]))
        if beliefs is None:
            beliefs = {"raw": bf, "stateless": bf, "stateful": bf}
        beliefs["raw"] = bf
        for arm in arms:
            horizon_rows.append({"arm": arm, "horizon": t, "log_loss": tree.log_loss(lv, beliefs[arm]),
                                 "accuracy": tree.accuracy(lv, beliefs[arm])})
        everywhere = np.ones(lv.n, dtype=bool)
        for (rule, lam, arm), stopper in stoppers.items():
            if rule.startswith("threshold"):
                stop = everywhere if last else beliefs[arm].max(1) >= float(rule.split("_")[1])
                stopper.step(lv, beliefs[arm], stop, n_out)
            elif rule == "step_limit":
                stopper.step(lv, beliefs[arm], everywhere if last else ~everywhere, n_out)
        if last:
            for (rule, lam, arm), stopper in stoppers.items():
                if rule == "cee_commit":
                    stopper.step(lv, beliefs[arm], everywhere, n_out)
            break
        values = _commit_values(scene, t, bf, reader.reactions_all(r_states[t], lv.seen), world_name, beliefs)
        for (rule, lam, arm), stopper in stoppers.items():
            if rule == "cee_commit":
                voi, cost = values[arm]
                stopper.step(lv, beliefs[arm], (voi - lam * cost).max(1) <= 0, n_out)
        p = filters.predictive(world.joint(w_states[t], lv.seen, lv.tests))
        q = floor_norm(reader.reactions(r_states[t], lv.seen, lv.tests))
        steps = {"stateless": batch.stateless_step(bf, p, q), "stateful": batch.stateful_step(beliefs["stateful"], bf, p, q)}
        for track, step in steps.items():
            fallback[track] += float(scene.weight(t)[step.fallback].sum())
            beliefs[track] = tree.flat(step.beliefs)
    commit_rows = [{"commit": rule, "lam": lam, "arm": arm, "accuracy": s.accuracy, "cost": s.cost, "requests": s.requests}
                   for (rule, lam, arm), s in stoppers.items()]
    return {"horizon": horizon_rows, "commit": commit_rows,
            "fallback": {track: share / scene.horizon for track, share in fallback.items()}}


# ---------------------------------------------------------------------------------------------
# Part D
# ---------------------------------------------------------------------------------------------
def budgets(scene: Scene, t: int, heldout=None) -> dict:
    """Every budget source of the card at the decision points of level t: name -> (budget [n], eligible [n])."""
    exact = scene.exact_budget[t]
    everywhere = np.ones(len(exact), dtype=bool)
    out = {
        "exact": (exact, everywhere),
        "exact_x0.5": (0.5 * exact, everywhere),
        "exact_x2": (2.0 * exact, everywhere),
        "exact_noise": (exact * scene.ctx.noise(len(exact), t), everywhere),
        "w1": (scene.own_budget("W1", t), everywhere),
        "w2cc": (scene.own_budget("W2", t), everywhere),
    }
    if heldout is not None:
        out["heldout"] = heldout.at(scene.levels[t])
    return out


def _scaled_step(track: str, c, bf, p, q, budget, eligible, variant: str):
    """One step of a track with CEE-Scale where the budget is eligible and CEE-Read alone elsewhere.

    Returns (beliefs after each outcome [n, outcome, class], tau [n], path [n] with -1 where no scale ran, fallback [n]).
    """
    n = len(bf)
    beliefs = np.empty(q.shape)
    tau, path, fallback = np.ones(n), np.full(n, -1), np.zeros(n, dtype=bool)
    for rows, scaling in ((np.flatnonzero(eligible), True), (np.flatnonzero(~eligible), False)):
        if not rows.size:
            continue
        arg = (budget[rows], variant) if scaling else None
        if track == "stateless":
            step = batch.stateless_step(bf[rows], p[rows], q[rows], arg)
        else:
            step = batch.stateful_step(c[rows], bf[rows], p[rows], q[rows], arg)
        beliefs[rows], fallback[rows] = step.beliefs, step.fallback
        if scaling:
            tau[rows], path[rows] = step.search.tau, step.search.path
    return beliefs, tau, path, fallback


def part_d_cell(scene: Scene, reader_name: str, world_name: str, heldout=None, budget_names=BUDGETS) -> dict:
    """Raw, projected and scaled-and-projected beliefs of one reader under one world model, for every scale arm."""
    cfg = scene.ctx.cfg
    reader, r_states = scene.reader(reader_name)
    world, w_states, path_prob = scene.world(world_name)
    n_out = len(scene.model.outcomes)
    names = [b for b in budget_names if b != "heldout" or heldout is not None]
    arms = [("raw", "none", "none")] + [(track, "none", "none") for track in TRACKS]
    arms += [(track, variant, b) for track in TRACKS for variant in SCALE_VARIANTS for b in names]
    beliefs = {}
    stoppers = {(rule, arm): tree.Stopper() for rule in (f"threshold_{cfg.threshold:g}", "step_limit") for arm in arms}
    horizon_rows = []
    shares = ("scaled", "clipped_low", "clipped_high", "non_monotone", "fallback")
    tau_stats = {arm: dict.fromkeys(("log_tau", "weight") + shares, 0.0) for arm in arms if arm[1] != "none"}
    for t, lv in enumerate(scene.levels):
        last = t == scene.horizon
        bf = floor_norm(reader.belief(r_states[t]))
        if not beliefs:
            beliefs = {arm: bf for arm in arms}
        beliefs[arms[0]] = bf
        for arm in arms:
            row = {"track": arm[0], "scale": arm[1], "budget": arm[2], "horizon": t,
                   "log_loss": tree.log_loss(lv, beliefs[arm]), "accuracy": tree.accuracy(lv, beliefs[arm])}
            row.update(tree.accounting(lv, path_prob[t], beliefs[arm]))
            horizon_rows.append(row)
            everywhere = np.ones(lv.n, dtype=bool)
            stoppers[(f"threshold_{cfg.threshold:g}", arm)].step(lv, beliefs[arm], everywhere if last else beliefs[arm].max(1) >= cfg.threshold, n_out)
            stoppers[("step_limit", arm)].step(lv, beliefs[arm], everywhere if last else ~everywhere, n_out)
        if last:
            break
        p = filters.predictive(world.joint(w_states[t], lv.seen, lv.tests))
        q = floor_norm(reader.reactions(r_states[t], lv.seen, lv.tests))
        available = budgets(scene, t, heldout)
        weight = scene.weight(t)
        nowhere = np.zeros(lv.n, dtype=bool)
        nxt = {}
        for arm in arms[1:]:
            track, variant, name = arm
            budget, eligible = (None, nowhere) if variant == "none" else available[name]
            children, tau, path, fallback = _scaled_step(track, beliefs[arm], bf, p, q, budget, eligible, variant)
            nxt[arm] = tree.flat(children)
            if variant != "none":
                stats = tau_stats[arm]
                stats["weight"] += float(weight.sum())
                stats["scaled"] += float(weight[path >= 0].sum())
                stats["log_tau"] += float((weight * np.log(tau)).sum())
                stats["clipped_low"] += float(weight[path == 1].sum())
                stats["clipped_high"] += float(weight[path == 2].sum())
                stats["non_monotone"] += float(weight[path == 3].sum())
                stats["fallback"] += float(weight[fallback].sum())
        beliefs = {arms[0]: bf, **nxt}
    commit_rows = [{"commit": rule, "track": arm[0], "scale": arm[1], "budget": arm[2], "accuracy": s.accuracy, "cost": s.cost,
                    "requests": s.requests} for (rule, arm), s in stoppers.items()]
    tau_rows = [{"track": arm[0], "scale": arm[1], "budget": arm[2],
                 "mean_log_tau": s["log_tau"] / s["weight"], **{k: s[k] / s["weight"] for k in shares}}
                for arm, s in tau_stats.items()]
    return {"horizon": horizon_rows, "commit": commit_rows, "tau": tau_rows}
