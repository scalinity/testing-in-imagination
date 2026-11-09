"""Command dispatch for scripts/run_tim.py; scripts/tim.sh is the only caller."""
import argparse
import subprocess
import sys

from tim.events import Bundle, TimError, guarded
from tim.settings import Settings

CANARY = "TIM-CANARY-7f3a"


def run_stage(settings, stage, job, body, **start):
    bundle = Bundle(settings.work_dir, stage, job)
    bundle.emit("job_start", stage=stage, source=settings.source, **start)
    with guarded(bundle):
        body(settings, bundle)
    bundle.emit("job_end", status="pass")
    return bundle


def canary_probe(settings, bundle):
    """Leak a marker through every channel; scripts/tim.sh must catch them all (SPEC §1.8)."""
    if not settings.synthetic:
        raise TimError("access_denied", "the canary probe runs on synthetic settings only")
    import pandas as pd
    import traceback
    print(f"multiline output\n{CANARY} print\nend of output")
    print(pd.DataFrame({"history": [f"{CANARY} pandas"]}))
    sys.stderr.write(f"{CANARY} stderr\n")
    subprocess.run([sys.executable, "-c", f"import sys; sys.stderr.write('{CANARY} subprocess\\n')"], check=True)
    try:
        assert False, f"{CANARY} assertion"
    except AssertionError:
        traceback.print_exc()
    bundle.emit("progress", completed=1, total=1)
    raise ValueError(f"{CANARY} exception")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tim")
    parser.add_argument("command")
    parser.add_argument("target", nargs="?")
    parser.add_argument("--job", default="main")
    args = parser.parse_args(argv)
    settings = Settings.load()
    if args.command == "canary-probe":
        run_stage(settings, "canary-probe", args.job, canary_probe)
    elif args.command == "publish":
        from tim.publish import publish
        print(f"published: {publish(settings, args.target)}")
    else:
        parser.error("unknown command")
