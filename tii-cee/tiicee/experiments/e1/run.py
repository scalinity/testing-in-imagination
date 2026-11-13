"""Command line of E1.

    python -m tiicee.experiments.e1.run cells       --out DIR --scratch DIR [--parts B_reduced B_points C D]
    python -m tiicee.experiments.e1.run c-intervals --out DIR --scratch DIR
    python -m tiicee.experiments.e1.run rules       --out DIR
    python -m tiicee.experiments.e1.run part-a      --out DIR
    python -m tiicee.experiments.e1.run harm        --out DIR
    python -m tiicee.experiments.e1.run figure      --out DIR --figure DIR
    python -m tiicee.experiments.e1.run manifest    --out DIR --phase NAME

Order of a full run: cells, c-intervals, rules, part-a (needs the selected layer), harm, rules again,
figure, manifest. Common options: --config card|tiny, --seeds, --variants, --jobs, --force.
A finished cell leaves a marker and is skipped on a later call unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tiicee.experiments.e1 import cells, config, figure1, heldout, intervals, manifest, part_a, rules, tables
from tiicee.experiments.e1.context import Context

RULE_TABLES = ("B_reduced_horizon", "B_points", "B_sandbox", "D_horizon", "D_heldout", "A_values")


def _marker(out: Path, stage: str, seed: int, variant: str) -> Path:
    return Path(out) / "tables" / "_done" / f"{stage}_seed{seed:02d}_{variant}.json"


def _cell_job(args):
    cfg, seed, variant, out, scratch, parts = args
    seconds = cells.run_cell(cfg, seed, variant, Path(out), Path(scratch) if scratch else None, parts)
    return seed, variant, seconds


def _part_a_job(args):
    cfg, seed, variant, out, layer = args
    ctx = Context(cfg, seed, variant)
    layer = part_a.Layer(*layer)
    estimate = heldout.HeldOut(ctx) if layer.scale is not None else None
    start = time.perf_counter()
    rows = []
    for lam in cfg.lam_frontier:
        for reader in (r for r in cfg.readers if r != "r0"):
            for agent in ("raw", "repaired"):
                values = part_a.agent_values(ctx, reader, agent, layer, lam, estimate)
                rows.append({"seed": seed, "variant": variant, "lam": lam, "reader": reader, "agent": agent,
                             "layer": layer.label() if agent == "repaired" else "none", **values})
    tables.write(tables.cell_file(out, "A_values", seed, variant), rows)
    return seed, variant, {"A": time.perf_counter() - start}


def _harm_job(args):
    cfg, seed, variant, out = args
    start = time.perf_counter()
    tables.write(tables.cell_file(out, "F_harmful_grid", seed, variant), figure1.harm_cell(cfg, seed, variant))
    return seed, variant, {"harm": time.perf_counter() - start}


def _run(stage: str, job, jobs: list, out: Path, workers: int, force: bool) -> None:
    todo = [j for j in jobs if force or not _marker(out, stage, j[1], j[2]).exists()]
    print(f"{stage}: {len(todo)} of {len(jobs)} cells to run, {workers} at a time", flush=True)
    start = time.perf_counter()
    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(job, todo)
            for seed, variant, seconds in results:
                _finish(out, stage, seed, variant, seconds)
    else:
        for item in todo:
            _finish(out, stage, *job(item))
    print(f"{stage}: done in {time.perf_counter() - start:.0f} s", flush=True)


def _finish(out: Path, stage: str, seed: int, variant: str, seconds: dict) -> None:
    path = _marker(out, stage, seed, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "variant": variant, "seconds": seconds, "finished_at": manifest.now()}) + "\n")
    print(f"  seed {seed:2d} {variant}: " + ", ".join(f"{k} {v:.0f} s" for k, v in seconds.items()), flush=True)


def c_intervals(cfg: config.Config, scratch: Path) -> dict:
    """Part C's two intervals, over every decision point of the primary variant (bootstrap over patients within seeds)."""
    names = ("regret", "e_w", "kl")
    columns = {name: [] for name in names}
    unit, seed_of, ignored, responsive, pair_unit, pair_seed = [], [], [], [], [], []
    for path in sorted((Path(scratch) / "C_rows").glob("*.npz")):
        seed = int(path.name[4:6])
        if seed not in cfg.seeds:
            continue
        data = np.load(path)
        errors = list(data["errors"])
        steps, n, n_err = data["regret"].shape
        patient = np.broadcast_to(seed * 1_000_000 + np.arange(n)[None, :, None], (steps, n, n_err))
        for name in names:
            columns[name].append(data[name].ravel())
        unit.append(patient.ravel())
        seed_of.append(np.full(patient.size, seed))
        ignored.append(data["regret"][:, :, errors.index("matched_ignored")].ravel())
        responsive.append(data["regret"][:, :, errors.index("matched_responsive")].ravel())
        pair_unit.append(patient[:, :, 0].ravel())
        pair_seed.append(np.full(steps * n, seed))
    if not unit:
        raise SystemExit(f"no part C rows under {scratch}")
    stack = {name: np.concatenate(v) for name, v in columns.items()}
    gap = intervals.spearman_difference(stack["regret"], stack["e_w"], stack["kl"], np.concatenate(unit), np.concatenate(seed_of), cfg, 2000)
    pair = intervals.paired(np.concatenate(responsive), np.concatenate(ignored), np.concatenate(pair_unit), np.concatenate(pair_seed), cfg, 2001)
    return {"spearman_difference": gap, "matched_pair": pair, "rows": int(len(stack["regret"])),
            "definition": "rho(regret, |E_W|) - rho(regret, KL) over (patient, step, acquisition rule, reader, controlled error) rows; "
                          "matched pair: regret under the responsive member minus regret under the ignored member"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="E1, the exact synthetic study")
    parser.add_argument("stage", choices=("cells", "c-intervals", "rules", "part-a", "harm", "figure", "manifest"))
    parser.add_argument("--config", choices=("card", "tiny"), default="card")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scratch")
    parser.add_argument("--figure")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--parts", nargs="*", default=list(cells.PARTS))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--phase", default="phase 2: full run")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = config.card() if args.config == "card" else config.tiny()
    out = Path(args.out)
    seeds = args.seeds if args.seeds else list(cfg.seeds)
    variants = args.variants if args.variants else list(cfg.variants)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    command = "python -m tiicee.experiments.e1.run " + " ".join(sys.argv[1:] if argv is None else argv)

    if args.stage == "cells":
        _run("cells_" + "_".join(args.parts), _cell_job, [(cfg, s, v, str(out), args.scratch, tuple(args.parts)) for s in seeds for v in variants],
             out, args.jobs, args.force)
    elif args.stage == "c-intervals":
        found = c_intervals(cfg, Path(args.scratch))
        (out / "c_intervals.json").write_text(json.dumps(found, indent=1) + "\n")
        print(json.dumps({k: found[k] for k in ("rows",)}))
    elif args.stage == "rules":
        loaded = {name: tables.read_table(out, name) for name in RULE_TABLES}
        loaded = {name: rows for name, rows in loaded.items() if rows}
        path = out / "c_intervals.json"
        verdicts = rules.evaluate(loaded, cfg, c_intervals=json.loads(path.read_text()) if path.exists() else None)
        (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1) + "\n")
        for name, verdict in verdicts.items():
            print(f"  {name}: {verdict.get('verdict', verdict.get('label'))}" + ("" if verdict.get("status", "evaluated") == "evaluated" else "  (not evaluated)"))
    elif args.stage == "part-a":
        layer = json.loads((out / "verdicts.json").read_text())["selected_layer"]
        chosen = (layer["track"], layer["scale"], layer["budget"])
        _run("part_a", _part_a_job, [(cfg, s, cfg.primary_variant, str(out), chosen) for s in seeds], out, args.jobs, args.force)
    elif args.stage == "harm":
        _run("harm", _harm_job, [(cfg, s, cfg.primary_variant, str(out)) for s in seeds], out, args.jobs, args.force)
    elif args.stage == "figure":
        for path in figure1.write(out, Path(args.figure), cfg):
            print(f"  wrote {path}")
    elif args.stage == "manifest":
        timings = [json.loads(p.read_text()) for p in sorted((out / "tables" / "_done").glob("*.json"))] if (out / "tables" / "_done").exists() else []
        extra = {"cell_seconds": {}, "verdicts_sha256": manifest.sha256(out / "verdicts.json") if (out / "verdicts.json").exists() else None}
        for item in timings:
            for part, seconds in item["seconds"].items():
                extra["cell_seconds"][part] = extra["cell_seconds"].get(part, 0.0) + seconds
        manifest.write(manifest.build(cfg, out, args.phase, command, extra), out / "manifest.json")
        print(f"  wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
