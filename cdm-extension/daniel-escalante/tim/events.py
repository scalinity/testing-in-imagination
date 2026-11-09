"""Public bundle writer: a fixed-schema event stream plus aggregate JSON, validated on write.

Validation is a positive allowlist (SPEC §1.4). Errors name the JSON path built from
schema-declared keys and a rule code, never the offending value.
"""
from contextlib import contextmanager
from functools import lru_cache
import json
import math
from pathlib import Path
import re

from tim.enums import ENUMS
from tim.paths import SCHEMAS

IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
NOT_ALLOWLIST = "schema not an allowlist"


class SchemaError(ValueError):
    def __init__(self, path: str, code: str):
        super().__init__(f"schema violation at {path}: {code}")
        self.path, self.code = path, code


class TimError(RuntimeError):
    """A failure with a public error code; its detail text stays in the private log."""

    def __init__(self, code: str, detail: str = ""):
        if code not in ENUMS["error_code"]:
            raise ValueError("unknown error code")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def _enum(name, path):
    if name not in ENUMS:
        raise SchemaError(path, NOT_ALLOWLIST)
    return ENUMS[name]


def validate(value, schema: dict, path: str = "$") -> None:
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate(value, option, path)
                return
            except SchemaError:
                pass
        raise SchemaError(path, "anyOf")
    kind = schema.get("type")
    if kind == "object":
        keyed = "keys" in schema
        if not keyed and schema.get("additionalProperties", True) is not False:
            raise SchemaError(path, NOT_ALLOWLIST)
        if not isinstance(value, dict):
            raise SchemaError(path, "type")
        allowed = _enum(schema["keys"], path) if keyed else schema.get("properties", {})
        for key in value:
            if key not in allowed:
                raise SchemaError(path, "unexpected_key")
        for key in schema.get("required", ()):
            if key not in value:
                raise SchemaError(f"{path}.{key}", "required")
        for key, item in value.items():
            validate(item, schema["values"] if keyed else allowed[key], f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise SchemaError(path, "type")
        if len(value) > schema.get("maxItems", len(value)):
            raise SchemaError(path, "maxItems")
        for index, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{index}]")
    elif kind in ("integer", "number"):
        types = int if kind == "integer" else (int, float)
        if isinstance(value, bool) or not isinstance(value, types):
            raise SchemaError(path, "type")
        if not math.isfinite(value):
            raise SchemaError(path, "nonfinite")
        if value < schema.get("minimum", value):
            raise SchemaError(path, "minimum")
        if value > schema.get("maximum", value):
            raise SchemaError(path, "maximum")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise SchemaError(path, "type")
    elif kind == "null":
        if value is not None:
            raise SchemaError(path, "type")
    elif kind == "string" and "enum" in schema:
        names = _enum(schema["enum"], path)
        if not isinstance(value, str):
            raise SchemaError(path, "type")
        if value not in names:
            raise SchemaError(path, "enum")
    elif kind == "string" and schema.get("format") == "identifier":
        if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
            raise SchemaError(path, "identifier")
    else:
        raise SchemaError(path, NOT_ALLOWLIST)


def _lint(schema, path="schema"):
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            _lint(option, path)
        return
    kind = schema.get("type")
    if kind == "object":
        if "keys" in schema:
            _enum(schema["keys"], path)
            _lint(schema["values"], path)
        elif schema.get("additionalProperties", True) is not False:
            raise SchemaError(path, NOT_ALLOWLIST)
        else:
            for key, item in schema.get("properties", {}).items():
                _lint(item, f"{path}.{key}")
    elif kind == "array":
        _lint(schema["items"], path)
    elif kind == "string":
        if "enum" in schema:
            _enum(schema["enum"], path)
        elif schema.get("format") != "identifier":
            raise SchemaError(path, NOT_ALLOWLIST)
    elif kind not in ("integer", "number", "boolean", "null"):
        raise SchemaError(path, NOT_ALLOWLIST)


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    schema = json.loads((SCHEMAS / f"{name}.json").read_text())
    if name == "events":
        for event_type, item in schema.items():
            _enum("event_type", "schema")
            if event_type not in ENUMS["event_type"]:
                raise SchemaError("schema", NOT_ALLOWLIST)
            _lint(item, f"schema.{event_type}")
    else:
        _lint(schema)
    return schema


def validate_event(event) -> None:
    if not isinstance(event, dict) or event.get("type") not in ENUMS["event_type"]:
        raise SchemaError("$.type", "enum")
    validate(event, load_schema("events")[event["type"]])


def dumps(value) -> str:
    """Canonical JSON for public files, so repeated runs are byte-identical."""
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


class Bundle:
    """Staging bundle for one stage: events/<job>.jsonl plus <name>.json aggregates."""

    def __init__(self, work_dir: Path, stage: str, job: str):
        if stage not in ENUMS["stage"] or not IDENTIFIER.fullmatch(job):
            raise ValueError("unknown stage or invalid job label")
        self.dir = Path(work_dir) / "staging" / stage
        self.events_path = self.dir / "events" / f"{job}.jsonl"

    def emit(self, event_type: str, **fields) -> None:
        event = {"type": event_type, **fields}
        validate_event(event)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")

    def aggregate(self, name: str, value) -> Path:
        from replication.manifest import assert_aggregate
        validate(value, load_schema(name))
        assert_aggregate(value)
        path = self.dir / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(value))
        return path


def error_code(exc: BaseException) -> str:
    if isinstance(exc, TimError):
        return exc.code
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return "out_of_memory"
    if isinstance(exc, SchemaError):
        return "schema_violation"
    return "internal"


@contextmanager
def guarded(bundle: Bundle):
    """Record an exception as a code and class name; the message stays in the private log."""
    try:
        yield
    except BaseException as exc:
        name = type(exc).__name__
        bundle.emit("error", code=error_code(exc),
                    exception_class=name if name in ENUMS["exception_class"] else "Other")
        raise
