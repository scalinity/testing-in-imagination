"""Code-defined enumerations: the only free strings a public bundle may hold."""

DISEASES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
SPLITS = ("train", "val", "test")
PHASE_A_STAGES = ("validate", "smoke", "bench-a", "val-a", "analyze-val-a", "addendum-a",
                  "test-a", "analyze-a", "test-a2", "analyze-a2")
PHASE_B_STAGES = ("hpi-audit", "wm-fit", "inventory", "leakage", "bench-b", "val-b",
                  "analyze-val-b", "addendum-b", "test-b", "analyze-b")
STAGES = (*PHASE_A_STAGES, *PHASE_B_STAGES, "canary-probe")
LADDER = ("official", "V0", "V0-rerun", "V0-shadow", "V1", "V2-raw", "V2", "V3-raw", "V3")
COMPARISONS = ("V3-official", "V1-official", "V3-V1", "V2-V1", "V3-V2", "V2raw-official", "V3raw-official",
               "V3-V0rerun", "V1-V0", "V2-V0", "V3-V0")
TEST_METHODS = ("exact", "draws")
SIGNIFICANCE = ("statistically significant", "not statistically significant")
CHANGES = ("improved", "regressed", "unchanged")
VARIANTS = ("V0", "V0-shadow", "V1", "V2", "V3", "V0-rerun")
TERMINATIONS = ("diagnosis", "step_limit", "malformed", "length_completion", "length_observation")
GUARD_EVENTS = ("g1_replace", "g2_commit", "replacement_failed", "scorer_error", "wm_rerank", "wm_stop")
ACTION_KINDS = ("test", "diagnosis", "malformed", "unknown_action", "unknown_test", "repeated_test",
                "unknown_diagnosis", "overflow")
PREDICTION_SOURCES = ("diagnosis", "hypothesis_fallback", "none")
ERROR_CODES = ("internal", "interrupted", "out_of_memory", "access_denied", "admission_refused",
               "approval_missing", "approval_mismatch", "schema_violation", "canary_found",
               "baseline_mismatch", "runtime_mismatch", "budget_exceeded", "resume_mismatch")
EXCEPTION_CLASSES = ("ValueError", "RuntimeError", "KeyError", "TypeError", "IndexError", "AttributeError",
                     "AssertionError", "FileNotFoundError", "FileExistsError", "PermissionError", "OSError",
                     "ImportError", "ModuleNotFoundError", "ZeroDivisionError", "NotImplementedError",
                     "TimeoutError", "MemoryError", "OutOfMemoryError", "KeyboardInterrupt", "SystemExit",
                     "TimError", "SchemaError")
EVENT_TYPES = ("job_start", "progress", "job_end", "error")
STATUSES = ("pass", "fail", "blocked")
SOURCES = ("synthetic", "mimic")
DIGEST_NAMES = ("train.csv", "val.csv", "test.csv", "lab_test_mapping.csv", "official_rows")

ENUMS = {
    "disease": DISEASES,
    "outcome": (*DISEASES, "undiagnosed"),
    "split": SPLITS,
    "stage": STAGES,
    "variant": VARIANTS,
    "termination": TERMINATIONS,
    "guard_event": GUARD_EVENTS,
    "action_kind": ACTION_KINDS,
    "prediction_source": PREDICTION_SOURCES,
    "error_code": ERROR_CODES,
    "exception_class": (*EXCEPTION_CLASSES, "Other"),
    "event_type": EVENT_TYPES,
    "status": STATUSES,
    "source": SOURCES,
    "digest_name": DIGEST_NAMES,
    "metric": ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"),
    "ladder": LADDER,
    "comparison": COMPARISONS,
    "test_method": TEST_METHODS,
    "significance": SIGNIFICANCE,
    "change": CHANGES,
}
