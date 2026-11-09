import json

import pytest

from tim.events import Bundle, SchemaError, TimError, guarded, load_schema, validate

SECRET = "TIM-CANARY-7f3a patient narrative"

OBJECT = {
    "type": "object", "additionalProperties": False, "required": ["count"],
    "properties": {
        "count": {"type": "integer", "minimum": 0},
        "rate": {"type": "number"},
        "flag": {"type": "boolean"},
        "split": {"type": "string", "enum": "split"},
        "commit": {"type": "string", "format": "identifier"},
        "by_class": {"type": "object", "keys": "disease", "values": {"type": "integer"}},
        "maybe": {"anyOf": [{"type": "null"}, {"type": "number"}]},
        "items": {"type": "array", "items": {"type": "integer"}},
    },
}


def test_allowlisted_values_pass():
    validate({"count": 3, "rate": 0.5, "flag": True, "split": "val", "commit": "df3b764",
              "by_class": {"appendicitis": 2}, "maybe": None, "items": [1, 2]}, OBJECT)


@pytest.mark.parametrize("value, code", [
    ({"count": 1, SECRET: 1}, "unexpected_key"),
    ({"count": 1, "split": SECRET}, "enum"),
    ({"count": 1, "commit": SECRET}, "identifier"),
    ({"count": 1, "by_class": {SECRET: 1}}, "unexpected_key"),
    ({"count": float("nan")}, "type"),
    ({"count": True}, "type"),
    ({"count": 1, "rate": float("inf")}, "nonfinite"),
    ({}, "required"),
    ({"count": -1}, "minimum"),
])
def test_rejections_name_the_rule_never_the_value(value, code):
    with pytest.raises(SchemaError) as caught:
        validate(value, OBJECT)
    assert caught.value.code == code
    assert "TIM-CANARY" not in str(caught.value) and "narrative" not in str(caught.value)


def test_free_text_string_schemas_are_refused():
    with pytest.raises(SchemaError, match="allowlist"):
        validate("x", {"type": "string"})
    with pytest.raises(SchemaError, match="allowlist"):
        validate({"a": 1}, {"type": "object", "properties": {"a": {"type": "integer"}}})


def test_every_schema_file_loads():
    for name in ("events", "validate"):
        assert load_schema(name)


def test_bundle_writes_only_valid_events(tmp_path):
    bundle = Bundle(tmp_path, "smoke", "job0")
    bundle.emit("progress", completed=1, total=2)
    with pytest.raises(SchemaError):
        bundle.emit("progress", completed=SECRET, total=2)
    lines = bundle.events_path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"type": "progress", "completed": 1, "total": 2}]


def test_guarded_records_code_and_class_only(tmp_path):
    bundle = Bundle(tmp_path, "smoke", "job0")
    with pytest.raises(ValueError):
        with guarded(bundle):
            raise ValueError(SECRET)
    with pytest.raises(TimError):
        with guarded(bundle):
            raise TimError("access_denied", SECRET)

    class Custom(Exception):
        pass

    with pytest.raises(Custom):
        with guarded(bundle):
            raise Custom(SECRET)
    text = bundle.events_path.read_text()
    assert "TIM-CANARY" not in text
    events = [json.loads(line) for line in text.splitlines()]
    assert events == [
        {"type": "error", "code": "internal", "exception_class": "ValueError"},
        {"type": "error", "code": "access_denied", "exception_class": "TimError"},
        {"type": "error", "code": "internal", "exception_class": "Other"},
    ]


def test_aggregates_are_validated_and_reject_patient_keys(tmp_path):
    bundle = Bundle(tmp_path, "validate", "main")
    with pytest.raises(SchemaError):
        bundle.aggregate("validate", {"patient_id": 1})
    assert not (bundle.dir / "validate.json").exists()
