import json

import pytest

from tim.events import Bundle, SchemaError, TimError, dumps
from tim.publish import destination, publish
from tim.settings import Settings


@pytest.fixture
def settings(environ, tmp_path):
    return Settings.load(env_file=tmp_path / "none.env", environ=environ)


def staged(settings):
    bundle = Bundle(settings.work_dir, "smoke", "job0")
    bundle.emit("progress", completed=2, total=2)
    return bundle


def test_publish_copies_a_valid_bundle(settings, tmp_path):
    staged(settings)
    out = publish(settings, "smoke", tmp_path / "results")
    assert out == tmp_path / "results" / "synthetic" / "smoke"
    assert json.loads((out / "events" / "job0.jsonl").read_text()) == {"type": "progress", "completed": 2, "total": 2}


def test_analysis_bundles_publish_under_their_phase(tmp_path):
    assert destination(tmp_path, "mimic", "analyze-a") == tmp_path / "credentialed" / "phase-a"
    assert destination(tmp_path, "mimic", "val-a") == tmp_path / "credentialed" / "val-a"


@pytest.mark.parametrize("tamper, error", [
    (lambda d: (d / "notes.txt").write_text("hello"), SchemaError),
    (lambda d: (d / "events" / "job0.jsonl").write_text('{"type": "progress", "completed": "TIM-CANARY-1", "total": 1}\n'), SchemaError),
    (lambda d: (d / "validate.json").write_text(dumps({"source": "TIM-CANARY-77aa"})), SchemaError),
    (lambda d: (d / "events" / "job0.jsonl").write_text('{"type": "progress", "completed": 90000017, "total": 1}\n'), TimError),
])
def test_publish_refuses_and_leaves_results_untouched(settings, tmp_path, tamper, error):
    bundle = staged(settings)
    (settings.work_dir / "private" / "ids").mkdir(parents=True)
    (settings.work_dir / "private" / "ids" / "val.txt").write_text("90000017\n")
    tamper(bundle.dir)
    with pytest.raises(error):
        publish(settings, "smoke", tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_canary_text_is_refused_even_inside_a_valid_identifier(settings, tmp_path):
    bundle = staged(settings)
    bundle.emit("job_start", stage="smoke", source="synthetic", commit="TIM-CANARY-0b1d")
    with pytest.raises(TimError) as caught:
        publish(settings, "smoke", tmp_path / "results")
    assert caught.value.code == "canary_found"
