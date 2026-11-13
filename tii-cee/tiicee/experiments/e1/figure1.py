"""The twin's inputs to Figure 1 (specification 1.7; monitor/DATA_CONTRACT.md, "Figure 1 files").

regime_map.json      one marker per simulated reader family, and the region where projection alone is harmful
regret_attribution.json   shares of exact regret by component for the twin agents, raw and repaired

The marker intervals are 95% percentile bootstrap intervals over seeds, which run slightly narrow at
twenty seeds (rules.SEED_INTERVAL_CAVEAT).

Marker coordinates: tau_B is the two-sided diagnostic under the exact budget and kl_inc the location
incoherence of the pair tempered at tau_B, both under W0, averaged over the decision points of the
full twin (both acquisition rules) and then over seeds, the scale geometrically. delta_log_loss is
the exact change in expected log loss at the final horizon of the reduced twin, repaired minus raw,
under W2 with the CEE layer the rules selected.

Harmful region. A grid of simulated readers spans the two ways a reader can be wrong: a likelihood
power from 1/4 to 4 (scale) and a drift from 0 to 0.8 of either kind (location). Each grid reader gets
the same two coordinates as a marker, and one exact number: the change in expected log loss under
projection alone (the selected track, W0, final horizon of the reduced twin). The region is traced in
the grid, where the readers are ordered, and then mapped to the plane: on each side of tau_B = 1, and
for each kind of drift, it is the polygon through the grid readers from which harm is unbroken out to
the most extreme power. Its outline passes only through computed points and nothing is extrapolated
beyond the grid. A harmful reader with a harmless one beyond it is left outside; harmful_summary
counts those, and the count is written beside the figure files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tiicee.experiments.e1 import filters, points, reduced, tables
from tiicee.experiments.e1 import intervals as stats_api
from tiicee.experiments.e1.config import READERS, Config
from tiicee.experiments.e1.context import Context
from tiicee.experiments.e1.manifest import now

POWERS = (0.25, 1 / 3, 0.5, 2 / 3, 1.0, 1.5, 2.0, 3.0, 4.0)
DELTAS = (0.0, 0.1, 0.2, 0.4, 0.8)
DRIFTS = ("entrench", "fixed")
LABELS = {"r0": "R0 (Bayes)", "independence": "Independence", "entrench": "Entrenching drift", "fixed": "Fixed-class drift",
          "over": "Over-reactive", "under": "Under-reactive", "entrench+over": "Entrenching + over-reactive",
          "fixed+under": "Fixed-class + under-reactive"}


def grid_readers() -> list:
    """(power, delta, drift) of every grid reader; a zero drift has no kind and is listed under each for the tracing."""
    return [(p, d, k) for k in DRIFTS for d in DELTAS for p in POWERS]


def harm_cell(cfg: Config, seed: int, variant: str) -> list:
    """Coordinates and exact harm of every grid reader for one seed."""
    ctx = Context(cfg, seed, variant)
    trajectories = [points.Trajectories(ctx, s) for s in points.SELECTORS]
    scenes = [reduced.Scene(ctx, s) for s in reduced.SELECTORS]
    rows, done = [], {}
    for power, delta, drift in grid_readers():
        name = filters.grid_name(power, delta, drift if delta > 0 else "none")
        if name not in done:
            coordinates = [points.regime_coordinates(traj, name, "W0") for traj in trajectories]
            harm = {"stateless": [], "stateful": []}
            for scene in scenes:
                final = [r for r in reduced.part_b_cell(scene, name, "W0")["horizon"] if r["horizon"] == scene.horizon]
                raw = next(r["log_loss"] for r in final if r["arm"] == "raw")
                for track in harm:
                    harm[track].append(next(r["log_loss"] for r in final if r["arm"] == track) - raw)
            done[name] = {"log_tau_b": float(np.mean([c["log_tau_b"] for c in coordinates])),
                          "kl_inc_at_tau_b": float(np.mean([c["kl_inc_at_tau_b"] for c in coordinates])),
                          "harm_stateless": float(np.mean(harm["stateless"])), "harm_stateful": float(np.mean(harm["stateful"]))}
        rows.append({"seed": seed, "variant": variant, "power": power, "delta": delta, "drift": drift, **done[name]})
    return rows


def grid_means(rows: list) -> dict:
    """(power, delta, drift) -> the seed means of each column."""
    cells = {}
    for r in rows:
        cells.setdefault((r["power"], r["delta"], r["drift"]), []).append(r)
    return {key: {c: float(np.mean([r[c] for r in group])) for c in ("log_tau_b", "kl_inc_at_tau_b", "harm_stateless", "harm_stateful")}
            for key, group in cells.items()}


def harmful_polygon(rows: list, track: str) -> list:
    """The harmful region as one closed outline in (tau_b, kl_inc), as described in the module docstring.

    For each side of power 1 and each drift kind, take at every drift size the grid reader nearest
    to power 1 that is harmful together with every reader beyond it. Those boundary readers, and the
    extreme-power readers behind them, bound the harmful readers of that side. The four outlines are
    joined into one polygon through the origin of the plane's vertical axis at tau_b = 1, where no
    area is enclosed. An empty list means no grid reader is harmful.
    """
    means = grid_means(rows)
    column = f"harm_{track}"

    def point(power, delta, drift):
        cell = means[(power, delta, drift)]
        return {"tau_b": float(np.exp(cell["log_tau_b"])), "kl_inc": cell["kl_inc_at_tau_b"]}

    lobes = []
    for side in (sorted(p for p in POWERS if p > 1), sorted((p for p in POWERS if p < 1), reverse=True)):   # nearest to 1 first
        for drift in DRIFTS:
            inner, outer = [], []
            for delta in DELTAS:
                harmful = [means[(p, delta, drift)][column] > 0 for p in side]
                first = next((i for i in range(len(side)) if all(harmful[i:])), None)      # harmful from here outward
                if first is not None:
                    inner.append(point(side[first], delta, drift))
                    outer.append(point(side[-1], delta, drift))
            if inner:
                lobes.append(inner + outer[::-1])
    if not lobes:
        return []
    anchor = {"tau_b": 1.0, "kl_inc": 0.0}
    polygon = []
    for lobe in lobes:
        polygon += [anchor] + lobe
    return polygon + [anchor]


def harmful_summary(rows: list, track: str) -> dict:
    """How many distinct grid readers projection alone harms, and how many of them the outline encloses."""
    means = grid_means(rows)
    column = f"harm_{track}"
    harmful, enclosed = set(), set()
    for side in (sorted(p for p in POWERS if p > 1), sorted((p for p in POWERS if p < 1), reverse=True)):
        for drift in DRIFTS:
            for delta in DELTAS:
                flags = [means[(p, delta, drift)][column] > 0 for p in side]
                first = next((i for i in range(len(side)) if all(flags[i:])), len(side))
                for i, p in enumerate(side):
                    name = (p, delta, drift if delta > 0 else "none")
                    if flags[i]:
                        harmful.add(name)
                        if i >= first:
                            enclosed.add(name)
    at_one = {(1.0, d, k if d > 0 else "none") for k in DRIFTS for d in DELTAS if means[(1.0, d, k)][column] > 0}
    return {"track": track, "grid_readers": len({(p, d, k if d > 0 else "none") for p, d, k in means}),
            "harmful": len(harmful | at_one), "enclosed": len(enclosed), "harmful_at_power_one": len(at_one),
            "powers": list(POWERS), "deltas": list(DELTAS), "drifts": list(DRIFTS)}


def _interval(values_by_seed: dict, cfg: Config, key: int) -> dict:
    return stats_api.seed_interval([values_by_seed[s] for s in cfg.seeds], cfg, key)


def markers(table: dict, verdicts: dict, cfg: Config) -> list:
    layer = verdicts["selected_layer"]
    out = []
    for i, reader in enumerate(r for r in READERS if r in cfg.readers):
        cells = [r for r in table["B_points"] if r["variant"] == cfg.primary_variant and r["world_model"] == "W0" and r["reader"] == reader]
        by_seed = lambda column: {s: float(np.mean([r[column] for r in cells if r["seed"] == s])) for s in cfg.seeds}
        tau, kl_inc = _interval(by_seed("log_tau_b"), cfg, 1000 + 2 * i), _interval(by_seed("kl_inc_at_tau_b"), cfg, 1001 + 2 * i)
        if layer["scale"] is None:
            rows = [r for r in table["B_reduced_horizon"] if r["variant"] == cfg.primary_variant and r["world_model"] == "W2" and r["reader"] == reader]
            last = max(r["horizon"] for r in rows)
            repaired = [r["log_loss"] for r in rows if r["horizon"] == last and r["arm"] == layer["track"]]
            raw = [r["log_loss"] for r in rows if r["horizon"] == last and r["arm"] == "raw"]
        else:
            rows = [r for r in table["D_horizon"] if r["variant"] == cfg.primary_variant and r["world_model"] == "W2" and r["reader"] == reader]
            last = max(r["horizon"] for r in rows) if rows else None
            repaired = [r["log_loss"] for r in rows if r["horizon"] == last and (r["track"], r["scale"], r["budget"]) == (layer["track"], layer["scale"], layer["budget"])]
            raw = [r["log_loss"] for r in rows if r["horizon"] == last and r["track"] == "raw"]
        out.append({
            "id": f"E1:{reader}", "label": LABELS[reader], "experiment": "E1", "kind": "simulated",
            "tau_b": float(np.exp(tau["estimate"])), "tau_b_lo": float(np.exp(tau["lo"])), "tau_b_hi": float(np.exp(tau["hi"])),
            "kl_inc": kl_inc["estimate"], "kl_inc_lo": kl_inc["lo"], "kl_inc_hi": kl_inc["hi"],
            "delta_log_loss": float(np.mean(repaired) - np.mean(raw)) if repaired and raw else None,
        })
    return out


def regime_map(table: dict, verdicts: dict, cfg: Config) -> dict:
    return {"schema": "tii-fig1-regime/1", "generated_at": now(),
            "harmful_region": harmful_polygon(table["F_harmful_grid"], verdicts["selected_layer"]["track"]) if table.get("F_harmful_grid") else [],
            "markers": markers(table, verdicts, cfg)}


def regret_attribution(verdicts: dict) -> dict:
    agents = []
    for name, entry in verdicts["attribution"]["agents"].items():
        reader, variant = name.split("|")
        if "shares" in entry:
            agents.append({"id": f"E1:{reader}:{variant}", "label": f"{LABELS[reader]}, {variant}", "variant": variant, **entry["shares"]})
    return {"schema": "tii-fig1-regret/1", "generated_at": now(), "agents": agents}


def write(out: Path, figure: Path, cfg: Config) -> list:
    """Write the Figure 1 files that the tables under `out` support. Returns the paths written."""
    out, figure = Path(out), Path(figure)
    verdicts = json.loads((out / "verdicts.json").read_text())
    table = {name: tables.read_table(out, name) for name in ("B_points", "B_reduced_horizon", "F_harmful_grid")}
    table["D_horizon"] = tables.read_table(out, "D_final")        # the final-horizon rows are all the markers need
    figure.mkdir(parents=True, exist_ok=True)
    written = []
    if verdicts.get("selected_layer", {}).get("status") == "evaluated" and table["B_points"]:
        (figure / "regime_map.json").write_text(json.dumps(regime_map(table, verdicts, cfg), indent=1) + "\n")
        written.append(figure / "regime_map.json")
        if table["F_harmful_grid"]:
            summary = harmful_summary(table["F_harmful_grid"], verdicts["selected_layer"]["track"])
            (out / "figure1_notes.json").write_text(json.dumps({"harmful_region": summary}, indent=1) + "\n")
    if verdicts.get("attribution", {}).get("status") == "evaluated":
        (figure / "regret_attribution.json").write_text(json.dumps(regret_attribution(verdicts), indent=1) + "\n")
        written.append(figure / "regret_attribution.json")
    return written
