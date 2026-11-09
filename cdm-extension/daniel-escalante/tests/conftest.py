from pathlib import Path
import sys

import pytest

OWN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OWN))

import tim.paths  # noqa: E402,F401  (puts the replication and upstream on sys.path)


@pytest.fixture
def env_dirs(tmp_path):
    """Private directories outside the repository, as a credentialed run would use."""
    dirs = {name: tmp_path / name for name in ("data", "baseline", "work", "admissions")}
    for path in dirs.values():
        path.mkdir()
    return dirs


@pytest.fixture
def environ(env_dirs):
    return {
        "CDM_DATA_SOURCE": "synthetic",
        "CDM_DATA_DIR": str(env_dirs["data"]),
        "CDM_WORK_DIR": str(env_dirs["baseline"]),
        "TIM_WORK_DIR": str(env_dirs["work"]),
        "TIM_ADMISSION_DIR": str(env_dirs["admissions"]),
    }
