import json
import subprocess

import pytest

from tim import access, approvals
from tim.events import TimError
from tim.settings import Settings


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


FOLDER = approvals.FOLDER


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / FOLDER / "approvals").mkdir(parents=True)
    (root / "tim").mkdir()
    (root / "tim" / "code.py").write_text("x = 1\n")
    git(root, "init", "-q")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "code")
    return root


@pytest.fixture
def settings(environ, tmp_path):
    return Settings.load(env_file=tmp_path / "none.env", environ={**environ, "CDM_DATA_SOURCE": "mimic"})


def approve(repo, stage, change=None):
    request = json.loads((repo / FOLDER / "approvals" / f"{stage}.request.json").read_text())
    approval = {**request, "approved_by": "Daniel Escalante", **(change or {})}
    (repo / FOLDER / "approvals" / f"{stage}.json").write_text(json.dumps(approval))
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", str(FOLDER))
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "approve")


def test_a_committed_matching_approval_passes(repo, settings):
    approvals.request(settings, "val-a", root=repo, code_paths=["tim"])
    approve(repo, "val-a")
    approvals.check(settings, "val-a", root=repo, code_paths=["tim"])


@pytest.mark.parametrize("change", [{"jobs": []}, {"protocol_sha256": "0" * 64}, {"config_sha256": "1" * 64},
                                    {"approved_by": "someone else"}])
def test_any_mismatch_refuses(repo, settings, change):
    approvals.request(settings, "val-a", root=repo, code_paths=["tim"])
    approve(repo, "val-a", change)
    with pytest.raises(TimError) as caught:
        approvals.check(settings, "val-a", root=repo, code_paths=["tim"])
    assert caught.value.code == "approval_mismatch"


def test_an_uncommitted_approval_refuses(repo, settings):
    approvals.request(settings, "val-a", root=repo, code_paths=["tim"])
    request = json.loads((repo / FOLDER / "approvals" / "val-a.request.json").read_text())
    (repo / FOLDER / "approvals" / "val-a.json").write_text(json.dumps({**request, "approved_by": "Daniel Escalante"}))
    with pytest.raises(TimError) as caught:
        approvals.check(settings, "val-a", root=repo, code_paths=["tim"])
    assert caught.value.code == "approval_missing"


def test_code_changed_after_the_approved_commit_refuses(repo, settings):
    approvals.request(settings, "val-a", root=repo, code_paths=["tim"])
    approve(repo, "val-a")
    (repo / "tim" / "code.py").write_text("x = 2\n")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qam", "change")
    with pytest.raises(TimError) as caught:
        approvals.check(settings, "val-a", root=repo, code_paths=["tim"])
    assert caught.value.code == "approval_mismatch"


def test_smoke_runs_within_its_cap_and_then_needs_approval(repo, settings):
    approvals.check(settings, "smoke", root=repo, code_paths=["tim"])
    access.Ledger(settings.admission_dir).record("smoke_seconds", key="k", seconds=1801.0)
    with pytest.raises(TimError) as caught:
        approvals.check(settings, "smoke", root=repo, code_paths=["tim"])
    assert caught.value.code == "approval_missing"


def test_requests_carry_the_job_list_and_budget(repo, settings):
    path = approvals.request(settings, "val-a", root=repo, code_paths=["tim"])
    request = json.loads(path.read_text())
    assert [job["variant"] for job in request["jobs"]] == ["V0", "V0-shadow", "V2", "V3"]
    assert all(job["split"] == "val" and job["patients"] == 240 for job in request["jobs"])
    assert request["gpu_hours_total"] == pytest.approx(sum(request["gpu_hours"].values()))
    assert request["code_commit"] == git(repo, "rev-parse", "HEAD").strip()
