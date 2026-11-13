"""The decision rules of the E1 card, as deterministic code over the per-seed result tables.

Written and committed before the full run. Each rule states the card's sentence and the reading the
code implements where the sentence leaves a choice open. No rule looks at anything but the tables.

Shared readings:
- "Expected log loss" in a rule is the exact expected log loss at the final horizon of the reduced
  twin, on the primary variant (base). A seed's value is the mean over the card's two acquisition
  rules, S_rand and S_ig. The other variants are evaluated the same way and reported, not decided on.
- "Improves with the paired 95% interval excluding 0" means the bootstrap interval of the mean
  per-seed difference lies entirely on the improving side.
- A rule whose tables lack one of the configured seeds, one of the two acquisition rules, or any other
  cell it reads is not evaluated: it returns status "incomplete" and verdict None, never a partial verdict.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from tiicee.experiments.e1 import intervals as stats_api
from tiicee.experiments.e1 import part_a
from tiicee.experiments.e1.reduced import SELECTORS
from tiicee.experiments.e1.config import DRIFT_FAMILIES, SCALE_FAMILIES, SCALE_REPAIR_FAMILIES, STRESS_SET, Config

DECLARED_ORDER = ("selection", "reading", "commitment")

# cee_check.py REFERENCE, the values reported on November 12, 2026 (4,000 patients, horizon 8, seed 0):
# (variant, reader) -> (raw log loss, projected log loss, raw accuracy, projected accuracy)
SANDBOX = {
    ("base", "independence"): (0.543, 0.524, 0.882, 0.868),
    ("base", "entrench"): (0.761, 0.524, 0.842, 0.868),
    ("base", "fixed"): (0.695, 0.524, 0.817, 0.868),
    ("base", "over"): (0.947, 1.60, 0.882, 0.796),
    ("strong_correlation", "independence"): (1.038, 0.903, 0.756, 0.723),
}
SANDBOX_PATIENTS = 4000

# Carried beside every verdict that rests on a paired interval over seeds (RESEARCH_LEDGER.md, D-STATS item 2).
SEED_INTERVAL_CAVEAT = ("The specified 95% percentile bootstrap over twenty seeds has simulated coverage of about 92% for normal "
                        "and 90% for skewed differences, so a verdict that rests on it runs slightly liberal. The method is the "
                        "card's and is not changed.")
# Each rule draws its bootstrap streams from a fixed block of keys, so its intervals do not depend on whether an
# earlier rule was complete. The blocks number the intervals in the order of a complete evaluation, which is the
# order the committed E1 verdicts were drawn in: repair and controls have 2 world models x 2 families each;
# scale_present has two variants x 3 families x 2 comparators; usable_outside one variant of those.
KEY_BLOCKS = {"track_selection": 0, "repair_present": 1, "negative_controls": 5, "variant_selection": 9,
              "scale_present": 10, "usable_outside": 22}
RESTS_ON_SEED_INTERVAL = ("track_selection", "repair_present", "negative_controls", "variant_selection", "scale_present", "usable_outside")


def _keep(rows, **equal):
    return [r for r in rows if all(r[k] == v for k, v in equal.items())]


def _final(rows):
    """The rows of the final horizon (none for no rows)."""
    if not rows:
        return []
    last = max(r["horizon"] for r in rows)
    return [r for r in rows if r["horizon"] == last]


def _by_seed(rows, value="log_loss") -> dict:
    """seed -> mean of a column over the two acquisition rules. A seed is left out unless it has exactly one row
    for each acquisition rule, so a missing rule makes the seed missing (and the rule incomplete), never a mean
    over the one that is there."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["seed"]].append(r)
    out = {}
    for seed, found in grouped.items():
        if sorted(r["selector"] for r in found) == sorted(SELECTORS):
            out[seed] = float(np.mean([r[value] for r in found]))
    return out


def _difference(a: dict, b: dict, cfg: Config):
    """Per-seed a - b in the configured seed order, or None if a seed is missing."""
    if not all(seed in a and seed in b for seed in cfg.seeds):
        return None
    return [a[seed] - b[seed] for seed in cfg.seeds]


class Evaluator:
    def __init__(self, cfg: Config, interval=None):
        self.cfg = cfg
        self._interval = interval or (lambda differences, key: stats_api.seed_interval(differences, cfg, key))
        self.families = tuple(r for r in cfg.readers if r != "r0")      # the R4 family of the run

    def interval(self, differences, key):
        """{estimate, lo, hi} of the mean per-seed difference, on the bootstrap stream `key`; None for a missing seed."""
        if differences is None:
            return None
        out = self._interval(differences, key)
        return {"estimate": float(out["estimate"]), "lo": float(out["lo"]), "hi": float(out["hi"])}

    # ------------------------------------------------------------------ part B
    def _b(self, rows, variant, reader, world, arm):
        return _by_seed(_keep(_final(_keep(rows, variant=variant)), reader=reader, world_model=world, arm=arm))

    def track_selection(self, b_horizon, variant=None) -> dict:
        """Card: the track with lower exact expected log loss at the final horizon, averaged over the two drift
        families under W2; a tie (the paired 95% interval over seeds includes 0) goes to stateless."""
        variant = variant or self.cfg.primary_variant
        per_track = {}
        for track in ("stateless", "stateful"):
            families = [self._b(b_horizon, variant, family, "W2", track) for family in DRIFT_FAMILIES]
            per_track[track] = {seed: float(np.mean([f[seed] for f in families])) for seed in families[0] if all(seed in f for f in families)}
        found = self.interval(_difference(per_track["stateful"], per_track["stateless"], self.cfg), KEY_BLOCKS["track_selection"])
        if found is None:
            return {"status": "incomplete", "verdict": None}
        return {"status": "evaluated", "verdict": "stateful" if found["hi"] < 0 else "stateless",
                "stateful_minus_stateless": found, "tie": found["lo"] <= 0 <= found["hi"]}

    def repair_present(self, b_horizon, track, variant=None) -> dict:
        """Card: repair present if both drift families improve with the paired 95% interval excluding 0 under W0
        and under W2. Kills the claim: the drift families do not improve under W0."""
        variant = variant or self.cfg.primary_variant
        cells = {}
        for i, (world, family) in enumerate((w, f) for w in ("W0", "W2") for f in DRIFT_FAMILIES):
            gain = self.interval(_difference(self._b(b_horizon, variant, family, world, "raw"),
                                             self._b(b_horizon, variant, family, world, track), self.cfg),
                                 KEY_BLOCKS["repair_present"] + i)
            if gain is None:
                return {"status": "incomplete", "verdict": None}
            cells[f"{family}|{world}"] = {**gain, "improves": gain["lo"] > 0}
        under_truth = all(cells[f"{family}|W0"]["improves"] for family in DRIFT_FAMILIES)
        return {"status": "evaluated", "verdict": all(c["improves"] for c in cells.values()), "track": track,
                "raw_minus_corrected": cells, "killed": not under_truth}

    def no_harm(self, b_horizon, variant=None) -> dict:
        """Card: no-harm if R0 with W0 changes by at most 1e-9. Reading: the largest absolute change in exact
        expected log loss over both tracks, every horizon, both acquisition rules and every seed."""
        variant = variant or self.cfg.primary_variant
        rows = _keep(b_horizon, variant=variant, reader="r0", world_model="W0")
        raw = {(r["seed"], r["selector"], r["horizon"]): r["log_loss"] for r in rows if r["arm"] == "raw"}
        tracks = {(r["seed"], r["selector"], r["horizon"], r["arm"]) for r in rows if r["arm"] != "raw"}
        complete = ({(seed, selector) for seed, selector, _ in raw} == {(s, x) for s in self.cfg.seeds for x in SELECTORS}
                    and tracks == {(*k, arm) for k in raw for arm in ("stateless", "stateful")})
        if not complete:
            return {"status": "incomplete", "verdict": None}
        changes = [abs(r["log_loss"] - raw[(r["seed"], r["selector"], r["horizon"])]) for r in rows if r["arm"] != "raw"]
        return {"status": "evaluated", "verdict": max(changes) <= self.cfg.no_harm, "largest_change": max(changes), "limit": self.cfg.no_harm}

    def negative_controls(self, b_horizon, track, variant=None) -> dict:
        """Card: confirmed if both scale families (over- and under-reactive) worsen under projection alone.
        Reading: under W0, the mean over seeds of corrected minus raw is positive for both; the interval is reported."""
        variant = variant or self.cfg.primary_variant
        cells = {}
        for i, (world, family) in enumerate((w, f) for w in ("W0", "W2") for f in SCALE_FAMILIES):
            harm = self.interval(_difference(self._b(b_horizon, variant, family, world, track),
                                             self._b(b_horizon, variant, family, world, "raw"), self.cfg),
                                 KEY_BLOCKS["negative_controls"] + i)
            if harm is None:
                return {"status": "incomplete", "verdict": None}
            cells[f"{family}|{world}"] = {**harm, "worsens": harm["estimate"] > 0}
        return {"status": "evaluated", "verdict": all(cells[f"{family}|W0"]["worsens"] for family in SCALE_FAMILIES),
                "track": track, "corrected_minus_raw": cells}

    def gate_valid(self, b_points, variant=None) -> dict:
        """Card: gate valid if allowed applications have positive mean gain for every family.
        Reading: for each R4 family and each world model, the exact expected gain averaged over every decision
        point the gate allows (all seeds, both acquisition rules) is positive. A cell where the gate allows
        nothing is reported and does not fail the rule; if it allows nothing in every cell the rule is vacuous and
        its verdict is None. Every (family, world model) cell must hold every seed and both acquisition rules."""
        variant = variant or self.cfg.primary_variant
        rows = _keep(b_points, variant=variant)
        full = {(s, x) for s in self.cfg.seeds for x in SELECTORS}
        cells, valid, any_allowed = {}, True, False
        for family in self.families:
            for world in self.cfg.world_models:
                here = _keep(rows, reader=family, world_model=world)
                if {(r["seed"], r["selector"]) for r in here} != full or not sum(r["n_points"] for r in here):
                    return {"status": "incomplete", "verdict": None}
                allowed = sum(r["gate_allowed"] * r["n_points"] for r in here)
                gain = sum((r["gate_gain"] or 0.0) * r["gate_allowed"] * r["n_points"] for r in here)
                total = sum(r["n_points"] for r in here)
                cell = {"allowed_share": allowed / total, "mean_gain_allowed": gain / allowed if allowed else None,
                        "mean_gain_ungated": float(np.mean([r["gain"] for r in here if r["gain"] is not None])),
                        "harmful_share_gated": sum(r["gate_harmful"] * r["n_points"] for r in here) / total,
                        "harmful_share_ungated": sum(r["ungated_harmful"] * r["n_points"] for r in here) / total}
                cells[f"{family}|{world}"] = cell
                valid &= cell["mean_gain_allowed"] is None or cell["mean_gain_allowed"] > 0
                any_allowed |= allowed > 0
        if not any_allowed:
            return {"status": "evaluated", "verdict": None, "vacuous": True, "cells": cells}
        return {"status": "evaluated", "verdict": valid, "cells": cells}

    def sandbox_cross_check(self, b_sandbox) -> dict:
        """Card: episode log loss at horizon 8 on the full twin is a Monte Carlo cross-check against the sandbox.
        Reading: run in the sandbox's own configuration and compared with its reported numbers. A cell agrees
        when the mean over seeds is within three standard errors of the reported value, the standard error
        combining the spread of our per-seed means with the same spread scaled to the sandbox's 4,000 patients.
        Descriptive: the library floors probabilities at 1e-6 and stops IPF at 1e-8, the sandbox did neither."""
        if {r["seed"] for r in b_sandbox} < set(self.cfg.seeds) or len(self.cfg.seeds) < 2:
            return {"status": "incomplete", "verdict": None}
        cells = {}
        for (variant, reader), reference in SANDBOX.items():
            rows = _keep(b_sandbox, variant=variant, reader=reader)
            if sorted(r["seed"] for r in rows) != sorted(self.cfg.seeds):
                return {"status": "incomplete", "verdict": None}
            entry = {}
            for column, reported in zip(("mc_log_loss_raw", "mc_log_loss_stateful", "mc_accuracy_raw", "mc_accuracy_stateful"), reference):
                values = np.array([r[column] for r in rows])
                ours = values.std(ddof=1) / np.sqrt(len(values))
                theirs = values.std(ddof=1) * np.sqrt(rows[0]["n_patients"] / SANDBOX_PATIENTS)
                gap, error = values.mean() - reported, np.hypot(ours, theirs)
                z = gap / error if error > 0 else (0.0 if gap == 0 else float(np.copysign(np.inf, gap)))    # no spread across seeds
                entry[column] = {"ours": float(values.mean()), "sandbox": reported, "z": float(z), "agrees": bool(abs(z) <= 3)}
            cells[f"{variant}|{reader}"] = entry
        return {"status": "evaluated", "label": "Monte Carlo", "cells": cells,
                "verdict": all(c[k]["agrees"] for c in cells.values() for k in ("mc_log_loss_raw", "mc_log_loss_stateful"))}

    # ------------------------------------------------------------------ part C
    def e1c_supported(self, c_intervals) -> dict:
        """Card: supported if the Spearman correlation of regret with |E_W| exceeds that with KL by at least 0.20
        with the interval of the difference excluding 0 (bootstrap over patients within seeds), and the matched
        pair's regrets differ with the interval excluding 0. Kills the claim: the margin is not met and the
        matched pair does not differ."""
        if not c_intervals:
            return {"status": "incomplete", "verdict": None}
        gap, pair = c_intervals["spearman_difference"], c_intervals["matched_pair"]
        margin = gap["estimate"] >= self.cfg.spearman_margin and gap["ci"][0] > 0
        differs = pair["ci"][0] > 0 or pair["ci"][1] < 0
        return {"status": "evaluated", "verdict": margin and differs, "margin_met": margin, "matched_pair_differs": differs,
                "killed": not margin and not differs, "spearman_difference": gap, "matched_pair": pair}

    # ------------------------------------------------------------------ part D
    def _d(self, rows, variant, reader, world, track, scale, budget):
        return _by_seed(_keep(_final(_keep(rows, variant=variant)), reader=reader, world_model=world, track=track, scale=scale, budget=budget))

    def variant_selection(self, d_horizon, track, variant=None) -> dict:
        """Card: the variant with the lower worst-case exact expected log loss over the stress set, averaged over
        reader families under W2; a tie (paired 95% interval over seeds includes 0) goes to shrink-only.
        Reading: for each family the worst case is the largest log loss over the four stress budgets
        (the exact budget, half, double, and with lognormal noise), then the families are averaged."""
        variant = variant or self.cfg.primary_variant
        worst = {}
        for scale in ("shrink-only", "two-sided"):
            families = []
            for family in self.families:
                stress = [self._d(d_horizon, variant, family, "W2", track, scale, budget) for budget in STRESS_SET]
                families.append({seed: max(s[seed] for s in stress) for seed in stress[0] if all(seed in s for s in stress)})
            worst[scale] = {seed: float(np.mean([f[seed] for f in families])) for seed in families[0] if all(seed in f for f in families)}
        found = self.interval(_difference(worst["two-sided"], worst["shrink-only"], self.cfg), KEY_BLOCKS["variant_selection"])
        if found is None:
            return {"status": "incomplete", "verdict": None}
        return {"status": "evaluated", "verdict": "two-sided" if found["hi"] < 0 else "shrink-only",
                "two_sided_minus_shrink_only": found, "tie": found["lo"] <= 0 <= found["hi"], "track": track}

    def _scale_improves(self, d_horizon, variant, world, track, scale, budget, first_key):
        cells = {}
        for i, family in enumerate(SCALE_REPAIR_FAMILIES):
            scaled = self._d(d_horizon, variant, family, world, track, scale, budget)
            for j, (name, comparator) in enumerate((("raw", self._d(d_horizon, variant, family, world, "raw", "none", "none")),
                                                    ("projection", self._d(d_horizon, variant, family, world, track, "none", "none")))):
                gain = self.interval(_difference(comparator, scaled, self.cfg), first_key + 2 * i + j)
                if gain is None:
                    return None
                cells[f"{family}|{name}"] = {**gain, "improves": gain["lo"] > 0}
        return cells

    def scale_present(self, d_horizon, track, variant=None) -> dict:
        """Card: scale repair present if, with the exact budget under W0, the over-reactive, under-reactive and
        independence families each improve on both raw and projection-only with the paired 95% interval
        excluding 0. Kills the claim: no improvement even with the exact budget.
        Reading: decided on the two-sided variant, the only one that can amplify an under-reactive reader;
        shrink-only is evaluated the same way and reported."""
        variant = variant or self.cfg.primary_variant
        by_scale = {scale: self._scale_improves(d_horizon, variant, "W0", track, scale, "exact",
                                                KEY_BLOCKS["scale_present"] + 2 * len(SCALE_REPAIR_FAMILIES) * k)
                    for k, scale in enumerate(("two-sided", "shrink-only"))}
        if any(cells is None for cells in by_scale.values()):
            return {"status": "incomplete", "verdict": None}
        decided = by_scale["two-sided"]
        return {"status": "evaluated", "verdict": all(c["improves"] for c in decided.values()), "track": track,
                "comparator_minus_scaled": by_scale, "killed": not any(c["estimate"] > 0 for c in decided.values())}

    def admission(self, d_heldout, variant=None) -> dict:
        """Card: the eligibility test admits an overestimated budget in at most 10% of stress cases.
        Reading: a stress case is one (seed, test, stratum) cell with data, with its conservative held-out budget
        multiplied by one member of the stress set (1, 0.5, 2, lognormal noise). The case is overestimated when
        the stressed budget exceeds the exact mean information of the cell, and admitted when it does not exceed
        the mean reduction on the fold kept apart. The rate is admitted-and-overestimated cases over all cases;
        the same count over overestimated cases only is reported beside it."""
        variant = variant or self.cfg.primary_variant
        rows = [r for r in _keep(d_heldout, variant=variant) if r["bound"] is not None]
        if {r["seed"] for r in rows} < set(self.cfg.seeds):
            return {"status": "incomplete", "verdict": None}
        cases = [(r[f"overestimated_{name}"], r[f"admitted_{name}"]) for r in rows for name in ("x1", "x0.5", "x2", "noise")]
        over = sum(o for o, _ in cases)
        both = sum(o and a for o, a in cases)
        rate = both / len(cases)
        return {"status": "evaluated", "verdict": rate <= self.cfg.admission_limit, "rate": rate, "limit": self.cfg.admission_limit,
                "cases": len(cases), "overestimated": over, "admitted_and_overestimated": both,
                "rate_among_overestimated": both / over if over else None,
                "eligible_share": float(np.mean([r["eligible"] for r in rows]))}

    def usable_outside(self, d_horizon, d_heldout, track, scale, variant=None) -> dict:
        """Card (gate G2c): usable outside the twin if the scale-repair condition holds for the selected variant
        under W2 with the estimated conservative budget, and the admission rate is at most 10%."""
        variant = variant or self.cfg.primary_variant
        cells = self._scale_improves(d_horizon, variant, "W2", track, scale, "heldout", KEY_BLOCKS["usable_outside"])
        admitted = self.admission(d_heldout, variant)
        if cells is None or admitted["status"] != "evaluated":
            return {"status": "incomplete", "verdict": None}
        improves = all(c["improves"] for c in cells.values())
        return {"status": "evaluated", "verdict": improves and admitted["verdict"], "track": track, "scale": scale,
                "improves": improves, "comparator_minus_scaled": cells, "admission": admitted}

    # ------------------------------------------------------------------ part A
    def attribution(self, a_values, lam=None, variant=None) -> dict:
        """Card: descriptive; the claim is killed if the spread of a component's share across the six orders
        exceeds that share's value in the declared order (selection, reading, commitment).
        Reading: values are averaged over seeds first; a share is an increment over V* - V; the spread is the
        largest minus the smallest share of the component over the six orders; an agent is unstable if any of its
        three components fails the comparison."""
        lam = self.cfg.lam if lam is None else lam
        rows = _keep(a_values, variant=variant or self.cfg.primary_variant, lam=lam)
        for family in self.families:
            for agent in ("raw", "repaired"):
                if sorted(r["seed"] for r in _keep(rows, reader=family, agent=agent)) != sorted(self.cfg.seeds):
                    return {"status": "incomplete", "verdict": None}
        agents, unstable = {}, []
        keys = [part_a.key(c) for c in part_a.COMBINATIONS] + ["v_star"]
        for reader, agent in sorted({(r["reader"], r["agent"]) for r in rows}):
            mean = {k: float(np.mean([r[k] for r in _keep(rows, reader=reader, agent=agent)])) for k in keys}
            found = part_a.attribution(mean)
            entry = {"regret": found["regret"], "v": mean[keys[0]], "v_star": mean["v_star"]}
            if found["regret"] > 1e-9:
                shares = {order: {c: inc / found["regret"] for c, inc in increments.items()} for order, increments in found["orders"].items()}
                declared = shares[DECLARED_ORDER]
                spread = {c: max(s[c] for s in shares.values()) - min(s[c] for s in shares.values()) for c in DECLARED_ORDER}
                failing = [c for c in DECLARED_ORDER if spread[c] > declared[c]]
                entry.update(shares=declared, spread=spread, unstable_components=failing,
                             orders={" > ".join(order): s for order, s in shares.items()})
                if failing:
                    unstable.append(f"{reader}|{agent}")
            agents[f"{reader}|{agent}"] = entry
        return {"status": "evaluated", "verdict": not unstable, "killed_for": unstable, "agents": agents, "lam": lam}


def evaluate(tables: dict, cfg: Config, interval=None, c_intervals=None) -> dict:
    """Every rule of the card, in dependency order. tables maps a table name to its rows; a missing table leaves
    its rules unevaluated. The repaired agent's CEE layer follows from the track, variant and G2c verdicts."""
    ev = Evaluator(cfg, interval)
    out = {}
    track = scale = None
    if "B_reduced_horizon" in tables:
        b = tables["B_reduced_horizon"]
        out["track_selection"] = ev.track_selection(b)
        track = out["track_selection"]["verdict"]
        if track:
            out["repair_present"] = ev.repair_present(b, track)
            out["negative_controls"] = ev.negative_controls(b, track)
        out["no_harm"] = ev.no_harm(b)
    if "B_points" in tables:
        out["gate_valid"] = ev.gate_valid(tables["B_points"])
    if "B_sandbox" in tables:
        out["sandbox_cross_check"] = ev.sandbox_cross_check(tables["B_sandbox"])
    out["e1c_supported"] = ev.e1c_supported(c_intervals)
    if "D_horizon" in tables and track:
        d = tables["D_horizon"]
        out["variant_selection"] = ev.variant_selection(d, track)
        scale = out["variant_selection"]["verdict"]
        out["scale_present"] = ev.scale_present(d, track)
        if scale and "D_heldout" in tables:
            out["usable_outside"] = ev.usable_outside(d, tables["D_heldout"], track, scale)
    if track:
        g2c = out.get("usable_outside", {"status": "incomplete"})
        if g2c["status"] == "evaluated":
            layer = part_a.Layer(track, scale, "heldout") if g2c["verdict"] else part_a.Layer(track)
            out["selected_layer"] = {"status": "evaluated", "track": layer.track, "scale": layer.scale, "budget": layer.budget,
                                     "label": layer.label(),
                                     "reason": "CEE-Scale with the held-out budget is used only if it is usable outside the twin (G2c)"}
        else:                                       # the track is known; whether CEE-Scale joins it is not
            out["selected_layer"] = {"status": "incomplete", "track": track, "scale": None, "budget": None, "label": None,
                                     "reason": "the layer waits on G2c (usable_outside), which is not evaluated"}
    if "A_values" in tables:
        out["attribution"] = ev.attribution(tables["A_values"])
    for name in RESTS_ON_SEED_INTERVAL:
        if name in out:
            out[name]["interval_caveat"] = SEED_INTERVAL_CAVEAT
    return out
