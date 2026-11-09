"""tim publish: re-validate a staged bundle, scan it for canaries, copy it under results/ (SPEC §1.5)."""
import json
from pathlib import Path
import re
import shutil

from tim.events import IDENTIFIER, SchemaError, TimError, load_schema, validate, validate_event
from tim.paths import OWN, SCHEMAS

PHASES = {"analyze-a": "phase-a", "analyze-b": "phase-b"}
CANARY = re.compile(rb"TIM-CANARY", re.IGNORECASE)


def destination(results_root: Path, source: str, stage: str) -> Path:
    kind = "credentialed" if source == "mimic" else "synthetic"
    return Path(results_root) / kind / PHASES.get(stage, stage)


def admission_ids(work_dir: Path) -> set[str]:
    """Every admission ID a stage has loaded: natural canaries that must never be public."""
    folder = Path(work_dir) / "private" / "ids"
    return {line for path in folder.glob("*.txt") for line in path.read_text().split()} if folder.exists() else set()


def _schema_name(stem: str):
    """Aggregate files are <schema>.json or <schema>.<job>.json."""
    name, _, job = stem.partition(".")
    if name in ("events", "_defs") or not (SCHEMAS / f"{name}.json").is_file():
        return None
    return name if not job or IDENTIFIER.fullmatch(job) else None


def _check_file(staging: Path, path: Path) -> None:
    from replication.manifest import assert_aggregate
    relative = path.relative_to(staging)
    try:
        text = path.read_bytes().decode("utf-8")
        if relative.parts[0] == "events" and len(relative.parts) == 2 and relative.suffix == ".jsonl" \
                and IDENTIFIER.fullmatch(relative.stem):
            for line in text.splitlines():
                validate_event(json.loads(line))
        elif len(relative.parts) == 1 and relative.suffix == ".json" and _schema_name(relative.stem):
            value = json.loads(text)
            validate(value, load_schema(_schema_name(relative.stem)))
            assert_aggregate(value)
        else:
            raise SchemaError("$", "unexpected_file")
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SchemaError("$", "json") from None


def check(work_dir: Path, stage: str) -> list[Path]:
    staging = Path(work_dir) / "staging" / stage
    files = sorted(p for p in staging.rglob("*") if p.is_file()) if staging.is_dir() else []
    if not files:
        raise TimError("schema_violation", "nothing is staged for this stage")
    for path in files:
        _check_file(staging, path)
    ids = admission_ids(work_dir)
    for path in files:
        data = path.read_bytes()
        if CANARY.search(data) or ids & set(re.findall(r"\d+", data.decode("utf-8"))):
            raise TimError("canary_found", "a canary or admission identifier is in the bundle")
    return files


def publish(settings, stage: str, results_root: Path = OWN / "results") -> Path:
    check(settings.work_dir, stage)
    target = destination(results_root, settings.source, stage)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(settings.work_dir / "staging" / stage, target)
    return target
