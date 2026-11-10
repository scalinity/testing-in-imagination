import hashlib
import os
import stat

from tim import cli, fixture
from tim.settings import Settings


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def set_writable(root, writable):
    for path in [root, *root.rglob("*")]:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWUSR if writable else mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def test_every_phase_a_stage_runs_against_a_read_only_baseline(environ, env_dirs, tmp_path):
    baseline = fixture.build(env_dirs["baseline"])
    before = snapshot(baseline)
    set_writable(baseline, False)
    try:
        settings = Settings.load(env_file=tmp_path / "none.env", environ=environ)
        for stage in cli.SYNTHETIC_ORDER:
            cli.run_command(settings, stage)
        assert not os.access(baseline / "prepared", os.W_OK)
    finally:
        set_writable(baseline, True)
    assert snapshot(baseline) == before
