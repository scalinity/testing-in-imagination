import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HELPER = Path(__file__).with_name("hpg_tsup.bash")


class StartupTests(unittest.TestCase):
    def run_helper(self, host="login7", owner="login7", delayed=False):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            state = home / ".local/share/tailscale"
            state.mkdir(parents=True)
            if owner is not None:
                (state / "owner").write_text(owner + "\n")
            scripts = {
                "hostname": '#!/bin/bash\nprintf "%s\\n" "$TEST_HOST"\n',
                "pgrep": "#!/bin/bash\nexit 0\n",
                "sleep": "#!/bin/bash\nexit 0\n",
                "tailscale": '''#!/bin/bash
printf '%s\n' "$*" >> "$HOME/calls"
if [ "$2" = "up" ]; then exit 0; fi
[ "$DELAYED" = "yes" ] || exit 0
[ "${3:-}" = "--json" ] || exit 1
n=$(cat "$HOME/count" 2>/dev/null || echo 0)
echo $((n + 1)) > "$HOME/count"
[ "$n" -ge 2 ]
''',
            }
            for name, text in scripts.items():
                p = home / name
                p.write_text(text)
                p.chmod(0o700)
            env = dict(os.environ, HOME=tmp, PATH=tmp + ":/usr/bin:/bin", TEST_HOST=host,
                       TS_SOCK=str(state / "tailscaled.sock"), DELAYED="yes" if delayed else "no")
            r = subprocess.run(["/bin/bash", "-c", 'source "$1"; tsup', "test", str(HELPER)],
                               env=env, capture_output=True, text=True, timeout=10)
            calls = (home / "calls").read_text() if (home / "calls").exists() else ""
            return r, calls

    def test_other_node_is_refused_before_touching_shared_daemon(self):
        r, calls = self.run_helper(host="login8")
        self.assertEqual(r.returncode, 1)
        self.assertIn("login7", r.stderr)
        self.assertEqual(calls, "")

    def test_missing_owner_is_refused(self):
        r, calls = self.run_helper(owner=None)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(calls, "")

    def test_healthy_daemon_is_reused_without_up(self):
        r, calls = self.run_helper()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn(" up ", calls)

    def test_waits_for_api_readiness_before_up(self):
        r, calls = self.run_helper(delayed=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(calls.count("status --json"), 3)
        self.assertEqual(calls.count("up --hostname=hipergator"), 1)


if __name__ == "__main__":
    unittest.main()
