"""The Mac-side local loop (WATCHER_DESIGN.md section 5), with stub ssh, rsync, osascript and agent
commands first on PATH. Nothing here opens a connection: the stub ssh runs the "remote" command
against a local directory, and the stub rsync hands the copy to the real rsync between two local
directories, so the loop's own rsync options are exercised.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="local_loop.sh is for macOS")

LOCAL = Path(__file__).resolve().parents[1] / "local"
PROMPT = ("Read STATUS.md, RESEARCH_LEDGER.md, BLOCKED.md, and the newest gate records. Continue study "
          "TII-CEE-1 under the version 3.1 kickoff from the first unfinished item. Do not repeat completed work.")
VIA_MASTER = ["-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ProxyCommand=false"]
RECORD = 'printf "%s" "$(basename "$0")" >> "$STUB_LOG"; printf "\\t%s" "$@" >> "$STUB_LOG"; printf "\\n" >> "$STUB_LOG"\n'
STUBS = {
    # `ssh -O check` answers for the master; anything else must carry the three options and is then
    # run locally, standing in for the remote side.
    "ssh": RECORD + """if [ "$1" = "-O" ]; then [ -f "$STUB_DIR/master_up" ]; exit $?; fi
[ -f "$STUB_DIR/master_up" ] || exit 255
[ "$1 $2 $3 $4 $5 $6" = "-o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false" ] || exit 99
shift 7
exec /bin/bash -c "$*"
""",
    "rsync": RECORD + """args=()
while [ $# -gt 0 ]; do
    case "$1" in -e) shift 2; continue ;; hpg:*) args+=("${1#hpg:}") ;; *) args+=("$1") ;; esac
    shift
done
exec /usr/bin/rsync "${args[@]}"
""",
    "osascript": RECORD,
    "agent": RECORD + 'printf "cwd\\t%s\\n" "$PWD" >> "$STUB_LOG"\n'
                      'while [ -f "$STUB_DIR/agent_hold" ]; do sleep 0.05; done\n',
}
EVENTS = ["2026-11-13T03:10:00-0500\tstage completed\tstage=noop_a experiment=benches attempts=1 gpu_hours=0",
          "2026-11-13T03:10:00-0500\tgate decided\tgate=G_noop result=met path=met skipped=0"]


class Mac:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.bin, self.remote, self.repo = tmp_path / "bin", tmp_path / "remote", tmp_path / "repo"
        self.state, self.approved = tmp_path / "loop state", tmp_path / "approved"
        for directory in (self.bin, self.remote / "state", self.remote / "export" / "approved", self.repo):
            directory.mkdir(parents=True)
        for name, body in STUBS.items():
            (self.bin / name).write_text("#!/bin/bash\n" + body)
            (self.bin / name).chmod(0o755)
        self.log = tmp_path / "calls.log"
        self.conf = tmp_path / "local_loop.conf"
        self.configure()
        (self.remote / "state" / "STATUS.md").write_text("# TII-CEE-1 watcher status\n\ncycle: 3\n")
        (tmp_path / "master_up").touch()
        self.env = {"HOME": str(tmp_path / "home"), "PATH": "%s:/usr/bin:/bin:/usr/sbin:/sbin" % self.bin,
                    "TII_LOCAL_LOOP_CONF": str(self.conf), "STUB_LOG": str(self.log), "STUB_DIR": str(tmp_path)}

    def configure(self, **changes):
        settings = {"TII_HPG_HOST": '"hpg"', "TII_REMOTE_ROOT": '"%s"' % self.remote, "TII_REPO": '"%s"' % self.repo,
                    "TII_AGENT_CMD": '("%s/agent" --headless)' % self.bin, "TII_LOOP_STATE": '"%s"' % self.state,
                    "TII_APPROVED_DIR": '"%s"' % self.approved}
        settings.update(changes)
        self.conf.write_text("".join("%s=%s\n" % item for item in settings.items()))

    def add_events(self, *lines):
        with open(self.remote / "state" / "events.tsv", "a") as handle:
            handle.write("".join(line + "\n" for line in lines))

    def run(self):
        result = subprocess.run(["/bin/bash", str(LOCAL / "local_loop.sh")], env=self.env,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        return result

    def calls(self, command=None):
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        return [line.split("\t") for line in lines if command is None or line.split("\t")[0] == command]

    def notifications(self):
        return [tuple(call[-2:]) for call in self.calls("osascript")]

    def wait_for_session_to_end(self):
        deadline = time.time() + 20
        while (self.state / "agent.pid").exists():
            assert time.time() < deadline, "the stub agent session did not end"
            time.sleep(0.05)


@pytest.fixture
def mac(tmp_path):
    made = Mac(tmp_path)
    yield made
    (tmp_path / "agent_hold").unlink(missing_ok=True)


def test_without_a_master_it_exits_quietly_having_only_checked(mac):
    (mac.tmp / "master_up").unlink()
    mac.add_events(*EVENTS)
    result = mac.run()
    assert (result.stdout, result.stderr) == ("", "")
    assert mac.calls() == [["ssh", "-O", "check", "hpg"]]
    assert not (mac.state / "mirror" / "events.tsv").exists() and not (mac.state / "agent.pid").exists()


def test_new_events_are_mirrored_notified_and_start_exactly_one_session(mac):
    mac.add_events(*EVENTS)
    (mac.remote / "export" / "approved" / "noop_a.json").write_text('{"count": 1}')
    (mac.remote / "export" / "approved" / "E2").mkdir()
    (mac.remote / "export" / "approved" / "E2" / "summary.json").write_text('{"n": 9}')
    (mac.remote / "export" / "approved" / "rows.csv").write_text("not an aggregate\n")
    (mac.approved).mkdir()
    (mac.approved / "withdrawn.json").write_text("{}")  # no longer approved on the cluster
    mac.run()
    mac.wait_for_session_to_end()

    assert (mac.state / "mirror" / "events.tsv").read_text().splitlines() == EVENTS
    assert (mac.state / "mirror" / "STATUS.md").read_text() == (mac.remote / "state" / "STATUS.md").read_text()
    assert mac.notifications() == [("TII: stage completed", "stage=noop_a experiment=benches attempts=1 gpu_hours=0"),
                                   ("TII: gate decided", "gate=G_noop result=met path=met skipped=0")]
    # Text reaches osascript as arguments after "--", never inside the script source.
    assert mac.calls("osascript")[0][1:8] == ["-e", "on run argv", "-e",
                                               "display notification (item 2 of argv) with title (item 1 of argv)",
                                               "-e", "end run", "--"]
    # One session, in the repository, with the configured command and the fixed prompt last.
    assert mac.calls("agent") == [["agent", "--headless", PROMPT]]
    assert mac.calls("cwd") == [["cwd", str(mac.repo.resolve())]] or mac.calls("cwd") == [["cwd", str(mac.repo)]]
    # Only approved JSON is copied, and the copy mirrors the cluster.
    assert sorted(str(path.relative_to(mac.approved)) for path in mac.approved.rglob("*") if path.is_file()) == [
        "E2/summary.json", "noop_a.json"]

    # Every call after the check goes through the existing master and cannot open a connection.
    ssh = mac.calls("ssh")
    assert ssh[0] == ["ssh", "-O", "check", "hpg"] and len(ssh) == 3
    assert all(call[1:8] == VIA_MASTER + ["hpg"] for call in ssh[1:])
    rsync = mac.calls("rsync")[0]
    assert rsync[rsync.index("-e") + 1] == "ssh " + " ".join(VIA_MASTER)
    assert rsync[-2:] == ["hpg:%s/export/approved/" % mac.remote, "%s/" % mac.approved]
    assert not (mac.state / "lock").exists()


def test_events_are_relayed_once_and_a_quiet_queue_starts_nothing(mac):
    mac.add_events(*EVENTS)
    mac.run()
    mac.wait_for_session_to_end()
    mac.run()
    mac.run()
    assert len(mac.notifications()) == 2 and len(mac.calls("agent")) == 1
    assert (mac.state / "events.seen").read_text().strip() == "2"

    mac.add_events("2026-11-13T04:00:00-0500\tstudy completed\tstages=3 completed=3 skipped=0")
    mac.run()  # relayed, but not one of the four kinds that call for a session
    assert mac.notifications()[-1] == ("TII: study completed", "stages=3 completed=3 skipped=0")
    assert len(mac.notifications()) == 3 and len(mac.calls("agent")) == 1


def test_running_session_prevents_a_second(mac):
    (mac.tmp / "agent_hold").touch()
    mac.add_events(EVENTS[0])
    mac.run()
    pid = int((mac.state / "agent.pid").read_text())
    os.kill(pid, 0)  # the session is running
    deadline = time.time() + 20
    while not mac.calls("cwd"):  # the stub agent has started and recorded itself
        assert time.time() < deadline
        time.sleep(0.05)
    calls = len(mac.calls())

    mac.add_events(EVENTS[1], "2026-11-13T03:25:00-0500\tfailure\tstage=x experiment=E2 outcome=other attempts=2")
    mac.run()
    mac.run()
    assert len(mac.calls()) == calls  # nothing at all while a session runs: no ssh, no second agent
    assert len(mac.calls("agent")) == 1

    (mac.tmp / "agent_hold").unlink()
    mac.wait_for_session_to_end()
    mac.run()  # catches up: both waiting events once, and one session for them
    mac.wait_for_session_to_end()
    assert [title for title, _ in mac.notifications()] == ["TII: stage completed", "TII: gate decided", "TII: failure"]
    assert len(mac.calls("agent")) == 2


def test_free_text_in_the_queue_is_dropped_unseen_and_later_events_still_arrive(mac):
    mac.add_events(EVENTS[0], "2026-11-13T03:11:00-0500\tfailure\tthe patient in bed 4 had chest pain",
                   "no tabs at all", "2026-11-13T03:12:00-0500\tnote\tstage=a", EVENTS[1])
    mac.run()
    mac.wait_for_session_to_end()
    assert (mac.state / "mirror" / "events.tsv").read_text().splitlines() == EVENTS
    assert len(mac.notifications()) == 2
    loop_log = (mac.tmp / "home" / "Library" / "Logs" / "tii-local-loop.log").read_text()
    assert "chest pain" not in mac.log.read_text() + loop_log  # neither shown nor logged
    assert "2 relayed, 3 dropped" in loop_log
    assert (mac.state / "events.seen").read_text().strip() == "5"
    mac.run()
    assert len(mac.notifications()) == 2


def test_queue_that_started_over_is_read_from_its_first_line(mac):
    mac.add_events(*EVENTS)
    mac.run()
    mac.wait_for_session_to_end()
    (mac.remote / "state" / "events.tsv").write_text(
        "2026-11-14T09:00:00-0500\tblocked\titem=W-001 key=authorization code=authorization_missing\n")
    mac.run()  # notices the shorter queue
    mac.run()  # and reads it from the top
    mac.wait_for_session_to_end()
    assert mac.notifications()[-1] == ("TII: blocked", "item=W-001 key=authorization code=authorization_missing")
    assert len(mac.notifications()) == 3 and len(mac.calls("agent")) == 2


def test_notifications_can_be_left_to_the_keeper(mac):
    mac.configure(TII_NOTIFY="0")
    mac.add_events(*EVENTS)
    mac.run()
    mac.wait_for_session_to_end()
    assert not mac.calls("osascript") and len(mac.calls("agent")) == 1


def test_only_one_run_of_the_loop_at_a_time(mac):
    mac.add_events(*EVENTS)
    (mac.state / "lock").mkdir(parents=True)
    mac.run()
    assert mac.calls() == [] and (mac.state / "lock").is_dir()
    old = time.time() - 3600
    os.utime(mac.state / "lock", (old, old))
    mac.run()  # a lock left by a run that never finished is taken over
    mac.wait_for_session_to_end()
    assert len(mac.calls("agent")) == 1 and not (mac.state / "lock").exists()


def test_launchd_template_and_example_settings():
    import plistlib
    text = (LOCAL / "com.tii.local-loop.plist").read_text()
    plist = plistlib.loads(text.replace("__HOME__", "/Users/someone").encode())
    assert plist["Label"] == "com.tii.local-loop" and plist["StartInterval"] == 600
    assert plist["ProgramArguments"] == ["/bin/bash", "/Users/someone/.tii/local-loop/local_loop.sh"]
    assert plist["StandardOutPath"].startswith("/Users/someone/Library/Logs/")
    assert plist["StandardErrorPath"].startswith("/Users/someone/Library/Logs/")
    assert plist["AbandonProcessGroup"] is True  # the session must outlive the loop script
    assert text.count("__HOME__") == 3

    shown = subprocess.run(["/bin/bash", "-c", '. "$1"; printf "%s\\n" "$TII_HPG_HOST" "$TII_REMOTE_ROOT" '
                            '"${#TII_AGENT_CMD[@]}" "${TII_AGENT_CMD[1]}" "$TII_NOTIFY"',
                            "-", str(LOCAL / "local_loop.conf.example")], capture_output=True, text=True,
                           env={"HOME": "/Users/someone", "PATH": "/usr/bin:/bin"})
    assert shown.stdout.split("\n")[:5] == ["hpg", "/blue/cai6734/daniel.escalante/private/tii-cee-1", "2", "-p", "1"]
    assert PROMPT in (LOCAL / "local_loop.sh").read_text()
    design = (LOCAL.parents[1] / "reports" / "research-advancement" / "WATCHER_DESIGN.md").read_text()
    assert PROMPT in " ".join(design.split())  # the prompt is the one the design quotes
