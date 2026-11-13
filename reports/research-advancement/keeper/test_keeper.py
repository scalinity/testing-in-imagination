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


if __name__ == "__main__":
    unittest.main()
