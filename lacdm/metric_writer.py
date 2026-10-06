"""Write metrics as JSON under lacdm/artifacts/, stamped with where they came from.

Every file records its `source` and the upstream commit, so numbers computed from
a test fixture can never be mistaken for numbers from a logged run.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LACDM_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = LACDM_DIR / "artifacts"
SOURCES = ("fixture", "run")


def upstream_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(LACDM_DIR / "upstream"), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def write_metrics(name: str, metrics: dict, source: str, out_dir: Path = ARTIFACTS_DIR) -> Path:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    payload = {
        "source": source,
        "upstream_sha": upstream_sha(),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
