import json
import os
from pathlib import Path
import subprocess

from tim.paths import OWN

MARKER = "TIM-CANARY-7f3a"
CHANNELS = ("print", "pandas", "stderr", "subprocess", "assertion", "exception")


def run_tim(environ, *args):
    env = {**os.environ, **environ}
    return subprocess.run(["bash", str(OWN / "scripts" / "tim.sh"), *args], env=env,
                          capture_output=True, text=True, timeout=120)


def files_under(path: Path):
    return [p for p in path.rglob("*") if p.is_file()] if path.exists() else []


def test_every_leak_channel_lands_only_in_the_private_log(environ, env_dirs):
    result = run_tim(environ, "canary-probe")
    assert result.returncode != 0
    assert MARKER not in result.stdout + result.stderr
    logs = files_under(env_dirs["work"] / "private-logs")
    assert len(logs) == 1
    text = logs[0].read_text()
    for channel in CHANNELS:
        assert f"{MARKER} {channel}" in text
    staged = files_under(env_dirs["work"] / "staging")
    assert staged
    for path in staged + files_under(OWN / "results"):
        assert MARKER not in path.read_text(errors="replace")
    events = [json.loads(line) for line in (env_dirs["work"] / "staging/canary-probe/events/main.jsonl").read_text().splitlines()]
    assert events[-1] == {"type": "error", "code": "internal", "exception_class": "ValueError"}


def test_the_probe_refuses_credentialed_mode(environ, env_dirs):
    result = run_tim({**environ, "CDM_DATA_SOURCE": "mimic"}, "canary-probe")
    assert result.returncode != 0
    events = (env_dirs["work"] / "staging/canary-probe/events/main.jsonl").read_text()
    assert '"code": "access_denied"' in events
