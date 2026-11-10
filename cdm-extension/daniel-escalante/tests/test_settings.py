import pytest

from tim.paths import OWN
from tim.settings import Settings


def load(environ, **changes):
    return Settings.load(env_file=OWN / "missing.env", environ={**environ, **changes})


def test_loads_private_directories(environ, env_dirs):
    settings = load(environ)
    assert settings.baseline_dir == env_dirs["baseline"].resolve()
    assert settings.work_dir == env_dirs["work"].resolve()
    assert settings.admission_dir == env_dirs["admissions"].resolve()
    assert settings.synthetic


def test_runtime_artifacts_never_inside_the_repository(environ):
    with pytest.raises(ValueError, match="outside the repository"):
        load(environ, TIM_WORK_DIR=str(OWN / "work"))


@pytest.mark.parametrize("name", ["TIM_WORK_DIR", "TIM_ADMISSION_DIR"])
def test_runtime_artifacts_never_inside_the_baseline(environ, env_dirs, name):
    with pytest.raises(ValueError, match="baseline"):
        load(environ, **{name: str(env_dirs["baseline"] / "tim")})


def test_admissions_do_not_live_under_the_work_directory(environ, env_dirs):
    with pytest.raises(ValueError, match="admission"):
        load(environ, TIM_ADMISSION_DIR=str(env_dirs["work"] / "admissions"))


def test_every_directory_is_required(environ):
    reduced = {k: v for k, v in environ.items() if k != "TIM_ADMISSION_DIR"}
    with pytest.raises(ValueError, match="TIM_ADMISSION_DIR"):
        Settings.load(env_file=OWN / "missing.env", environ=reduced)


def test_baseline_settings_point_the_replication_at_the_baseline(environ, env_dirs):
    baseline = load(environ, CDM_RUNTIME="blackwell").baseline()
    assert baseline.work_dir == env_dirs["baseline"].resolve()
    assert baseline.source == "synthetic" and baseline.runtime == "blackwell"
