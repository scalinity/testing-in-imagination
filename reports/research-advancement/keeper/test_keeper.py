import os
from pathlib import Path
import subprocess
import tempfile
import unittest

KEEPER = Path(__file__).with_name("hpg_keeper.sh")


class KeeperTests(unittest.TestCase):
    def test_missing_master_shows_recovery_without_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            conf = home / ".tii/keeper"
            conf.mkdir(parents=True)
            (conf / "keeper.conf").write_text(
                'TII_HPG_HOST="hpg-tail"\n'
                'TII_RECOVERY_HINT="Use hpg-public, then login7 and tsup."\n'
            )
            ssh = home / "ssh"
            ssh.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$HOME/calls"\nexit 1\n')
            ssh.chmod(0o700)
            osa = home / "osa"
            osa.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HOME/notice"\n')
            osa.chmod(0o700)
            env = dict(os.environ, HOME=tmp, TII_SSH_BIN=str(ssh), TII_OSASCRIPT_BIN=str(osa))
            subprocess.run(["/bin/bash", str(KEEPER)], env=env, check=True, timeout=10)
            self.assertEqual((home / "calls").read_text().strip(), "-O check hpg-tail")
            notice = (home / "notice").read_text()
            self.assertIn("ssh -fN hpg-tail", notice)
            self.assertIn("hpg-public, then login7 and tsup", notice)

    def check_event_cursor(self, rows, expected_notices):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            conf = home / ".tii/keeper"
            conf.mkdir(parents=True)
            (conf / "keeper.conf").write_text('TII_HPG_HOST="hpg-tail"\nTII_REMOTE_EVENTS="/events"\n')
            (home / "rows").write_text(rows)
            ssh = home / "ssh"
            ssh.write_text("""#!/bin/bash
if [ "$1" = "-O" ] || [ "${!#}" = "true" ]; then exit 0; fi
cmd=${!#}
first=${cmd#*tail -n +}
first=${first%% *}
/usr/bin/tail -n +"$first" "$HOME/rows" | /usr/bin/head -n 20
""")
            ssh.chmod(0o700)
            osa = home / "osa"
            osa.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$HOME/notice"\n')
            osa.chmod(0o700)
            env = dict(os.environ, HOME=tmp, TII_SSH_BIN=str(ssh), TII_OSASCRIPT_BIN=str(osa))
            for _ in range(2):
                subprocess.run(["/bin/bash", str(KEEPER)], env=env, check=True, timeout=10)
            cursor = home / "Library/Application Support/tii-keeper/events.seen"
            self.assertEqual(int(cursor.read_text()), len(rows.splitlines()))
            notices = (home / "notice").read_text() if (home / "notice").exists() else ""
            self.assertEqual(notices.count("TII: done"), expected_notices)
            self.assertNotIn("TII: should be skipped", notices)

    def test_skipped_lines_are_consumed_without_replaying_notifications(self):
        self.check_event_cursor("\nnow\t\tshould be skipped\nnow\tdone\tcompleted\n", 1)

    def test_only_malformed_lines_advance_cursor(self):
        self.check_event_cursor("malformed\n\n", 0)


if __name__ == "__main__":
    unittest.main()
