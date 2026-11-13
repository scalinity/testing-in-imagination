"""The manifest of an E1 run: what was run, on what, from which code, under which card and authorization."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from tiicee.experiments.e1 import tables as table_files
from tiicee.experiments.e1.config import Config
from tiicee.twin import source

PACKAGE = Path(__file__).resolve().parents[3]           # tii-cee/
REPOSITORY = PACKAGE.parent
CARD = REPOSITORY / "reports" / "research-advancement" / "EXPERIMENT_CARDS" / "E1.md"
AUTHORIZATION = [REPOSITORY / "reports" / "research-advancement" / name for name in ("AUTHORIZATION.md", "AUTHORIZATION_ADDENDUM_1.md")]
BOUND = ("tiicee/experiments", "tiicee/twin", "tiicee/cee", "tiicee/stats", "tiicee/interfaces.py", "vendor/synthetic_df467ea")

SEEDS_BY_PURPOSE = {
    "generation": "evaluation patients: stream(seed, generation, <eval_split>, case index), cases 0..n_eval-1",
    "fitting": "training patients for W1, W2, W2-cc and the held-out budget: stream(seed, fitting, train, case index), "
               "cases 0..n_train-1; CEE-Gate calibration patients: the next n_gate cases of the same stream",
    "behavior": "S_rand order: stream(seed, behavior, <eval_split>, number of tests); random prefix orders of the "
                "held-out budget: stream(seed, behavior, train, number of tests)",
    "corruption": "lognormal budget noise per tree level: stream(seed, corruption, <eval_split>, level); stress factor "
                  "of the eligibility test: stream(seed, corruption, train)",
    "montecarlo": "bootstrap of the held-out bound: stream(seed, montecarlo, train, test, stratum); of the gate bounds: "
                  "stream(seed, montecarlo, train, test); of the rule intervals: stream(analysis_seed, montecarlo, "
                  "<eval_split>, interval number)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now() -> str:
    """The current time in US Eastern, ISO 8601."""
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _git(*args):
    try:
        done = subprocess.run(["git", "-C", str(REPOSITORY), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def bound_files() -> dict:
    """SHA-256 of every source file the run executes or reads, by path relative to tii-cee/."""
    files = {}
    for entry in BOUND:
        root = PACKAGE / entry
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".md", "") and "__pycache__" not in p.parts)
        for path in paths:
            files[path.relative_to(PACKAGE).as_posix()] = sha256(path)
    return files


def code_identity() -> dict:
    """The commit, and whether the bound paths differ from it (read-only git)."""
    status = _git("status", "--porcelain", "--", *[f"tii-cee/{entry}" for entry in BOUND])
    paths = [f"tii-cee/{entry}" for entry in BOUND]
    return {"commit": _git("rev-parse", "HEAD"), "bound_paths_commit": _git("log", "-1", "--format=%H", "--", *paths),
            "bound_paths_clean": status == "" if status is not None else None,
            "uncommitted": status.splitlines() if status else []}


def build(cfg: Config, out: Path, phase: str, command: str, extra: dict = None) -> dict:
    out = Path(out)
    tables = {}
    for path in table_files.files(out):
        relative = path.relative_to(out).as_posix()
        tables[relative] = {"sha256": sha256(path), "rows": len(table_files.read(path)), "bytes": path.stat().st_size,
                            "committed": not relative.startswith("large/")}
    manifest = {
        "study_id": "TII-CEE-1",
        "experiment_id": "E1",
        "phase": phase,
        "kind": "new intervention; evaluation provenance: exact (twin); CPU only",
        "written_at": now(),
        "command": command,
        "config": cfg.as_dict(),
        "seeds_by_purpose": {"user_seeds": list(cfg.seeds), "analysis_seed": cfg.analysis_seed, **SEEDS_BY_PURPOSE},
        "snapshot": {"digest": source.snapshot_digest(), "commit": source.SNAPSHOT_COMMIT, "author": source.SNAPSHOT_AUTHOR},
        "code": code_identity(),
        "bound_files": bound_files(),
        "card": {"path": CARD.relative_to(REPOSITORY).as_posix(), "sha256": sha256(CARD)},
        "authorization": {p.name: sha256(p) for p in AUTHORIZATION},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "tables": tables,
        "notes": [
            "W2-cc is W2's fitted model read class-conditionally; its predictive is W2's, so part B's W2 rows are also its rows.",
            "Evaluation patients are the validation split: E1 selects the track and the scale variant.",
            "Full-twin episode log loss (mc_log_loss_*) is Monte Carlo over the evaluation patients; everything else is exact.",
            "Tables are gzipped CSV. Those under large/ are not committed; their SHA-256 is recorded here. D_final holds the "
            "final-horizon rows of D_horizon, which is what the rules and the figure step read of part D.",
        ],
    }
    manifest.update(extra or {})
    return manifest


def write(manifest: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
