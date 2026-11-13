"""One (seed, variant) cell of E1: every table of parts B, C and D for that cell, written under the output directory."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from tiicee.experiments.e1 import heldout, intervals, points, reduced, tables
from tiicee.experiments.e1.config import R4, Config
from tiicee.experiments.e1.context import Context

PARTS = ("B_reduced", "B_points", "C", "D")
D_WORLDS = ("W0", "W2")
C_FIELDS = ("regret", "e_w", "kl", "calibration", "e_w_max", "kl_max", "log_loss_change", "e_w_taken", "kl_taken", "calibration_taken")


def _rho(x, y):
    """Spearman correlation, or None when one side does not vary."""
    return intervals.spearman(x, y) if np.ptp(x) > 0 and np.ptp(y) > 0 else None


def part_c_rows(out: dict, common: dict) -> list:
    """Per error, the means of part C's per-history quantities; and across errors, the rank correlations of the card."""
    rows = []
    for e, name in enumerate(out["errors"]):
        rows.append({**common, "error": name, **{f: float(out[f][:, :, e].mean()) for f in C_FIELDS}})
    flat = {f: out[f].ravel() for f in C_FIELDS}
    change = np.abs(flat["log_loss_change"])
    rows.append({**common, "error": "all", **{f: float(flat[f].mean()) for f in C_FIELDS},
                 "rho_regret_e_w": _rho(flat["regret"], flat["e_w"]), "rho_regret_kl": _rho(flat["regret"], flat["kl"]),
                 "rho_regret_calibration": _rho(flat["regret"], flat["calibration"]),
                 "rho_regret_e_w_max": _rho(flat["regret"], flat["e_w_max"]), "rho_regret_kl_max": _rho(flat["regret"], flat["kl_max"]),
                 "rho_change_e_w": _rho(change, flat["e_w_taken"]), "rho_change_kl": _rho(change, flat["kl_taken"]),
                 "rho_change_calibration": _rho(change, flat["calibration_taken"]),
                 "share_regret_positive": float((flat["regret"] > 0).mean()), "slow_matched_points": out["slow_matched_points"]})
    return rows


def run_cell(cfg: Config, seed: int, variant: str, out: Path, scratch: Optional[Path] = None, parts=PARTS) -> dict:
    """Compute and write the tables of one cell. Returns the seconds each part took."""
    ctx = Context(cfg, seed, variant)
    common = {"seed": seed, "variant": variant}
    written = {name: [] for name in ("B_reduced_horizon", "B_reduced_commit", "B_reduced_fallback", "B_points", "B_sandbox", "C_summary",
                                     "D_horizon", "D_commit", "D_tau", "D_heldout")}
    seconds = dict.fromkeys(parts, 0.0)

    if "B_reduced" in parts or "D" in parts:
        estimate = None
        if "D" in parts:
            start = time.perf_counter()
            estimate = heldout.HeldOut(ctx)
            written["D_heldout"] = [{**common, **record} for record in estimate.records]
            seconds["D"] += time.perf_counter() - start
        for selector in reduced.SELECTORS:
            scene = reduced.Scene(ctx, selector)
            if "B_reduced" in parts:
                start = time.perf_counter()
                for reader in cfg.readers:
                    for world in cfg.world_models:
                        key = {**common, "selector": selector, "reader": reader, "world_model": world}
                        cell = reduced.part_b_cell(scene, reader, world)
                        written["B_reduced_horizon"] += [{**key, **row} for row in cell["horizon"]]
                        written["B_reduced_commit"] += [{**key, **row} for row in cell["commit"]]
                        written["B_reduced_fallback"].append({**key, **cell["fallback"]})
                seconds["B_reduced"] += time.perf_counter() - start
            if "D" in parts:
                start = time.perf_counter()
                for reader in (r for r in cfg.readers if r in R4):
                    for world in D_WORLDS:
                        key = {**common, "selector": selector, "reader": reader, "world_model": world}
                        cell = reduced.part_d_cell(scene, reader, world, heldout=estimate)
                        for table in ("horizon", "commit", "tau"):
                            written[f"D_{table}"] += [{**key, **row} for row in cell[table]]
                seconds["D"] += time.perf_counter() - start

    if "B_points" in parts and variant == cfg.primary_variant:      # the Monte Carlo cross-check against the sandbox, once per seed
        start = time.perf_counter()
        written["B_sandbox"] = points.sandbox_rows(cfg, seed, intervals.spearman)
        seconds["B_points"] += time.perf_counter() - start
    if "B_points" in parts or "C" in parts:
        for selector in points.SELECTORS:
            traj = points.Trajectories(ctx, selector)
            if "B_points" in parts:
                start = time.perf_counter()
                calibration = points.Trajectories(ctx, selector, calibration=True)
                for reader in cfg.readers:
                    for world in cfg.world_models:
                        gate = points.gate_bounds(calibration, reader, world)
                        row = points.part_b_cell(traj, reader, world, gate, intervals.spearman)
                        written["B_points"].append({**common, "selector": selector, "reader": reader, "world_model": world, **row})
                seconds["B_points"] += time.perf_counter() - start
            if "C" in parts:
                start = time.perf_counter()
                for reader in cfg.readers:
                    result = part_c = points.part_c_cell(traj, reader)
                    written["C_summary"] += part_c_rows(result, {**common, "selector": selector, "reader": reader})
                    if scratch is not None and variant == cfg.primary_variant:      # the rows part C's intervals resample
                        target = Path(scratch) / "C_rows" / f"seed{seed:02d}_{selector}_{reader.replace('+', '_')}.npz"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(target, errors=np.array(part_c["errors"]),
                                            **{f: part_c[f] for f in ("regret", "e_w", "kl", "calibration", "log_loss_change", "e_w_taken", "kl_taken")})
                seconds["C"] += time.perf_counter() - start

    for table, rows in written.items():
        if rows:
            tables.write(tables.cell_file(out, table, seed, variant), rows)
    return seconds
