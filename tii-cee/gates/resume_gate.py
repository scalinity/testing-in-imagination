#!/usr/bin/env python3
"""Gate G_resume: training in parts follows the uninterrupted admission sequence (D-CLIN-RESUME).

Contract: gates/noop_gate.py. Reads the Nemotron bench's approved report
(export/approved/bench-nemotron.json, written by tiicee/clinical/bench.py).

Met only when all hold:
  reference.outcome == "ok" and training.outcome == "ok"
  reference.follows_plan and training.follows_plan are true
  sequences_identical is true: the training paused at the epoch boundary and inside epoch two
      trained on the same admission at every optimizer step as the uninterrupted reference
  visits_identical is true: and on every admission the same number of times
Anything missing, null or false is not met. E8, E9 and E10 training stages wait on this gate;
if it is not met, no policy of the training track is trained, and the result is reported.
"""
import json
import sys


def decide(files):
    checks = {"report_present": False, "reference_ok": False, "training_ok": False,
              "reference_follows_plan": False, "training_follows_plan": False,
              "sequences_identical": False, "visits_identical": False}
    if len(files) == 1:
        try:
            with open(files[0]) as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            report = None
        if isinstance(report, dict):
            checks["report_present"] = True
            reference = report.get("reference") or {}
            training = report.get("training") or {}
            checks["reference_ok"] = reference.get("outcome") == "ok"
            checks["training_ok"] = training.get("outcome") == "ok"
            checks["reference_follows_plan"] = reference.get("follows_plan") is True
            checks["training_follows_plan"] = training.get("follows_plan") is True
            checks["sequences_identical"] = report.get("sequences_identical") is True
            checks["visits_identical"] = report.get("visits_identical") is True
    met = all(checks.values())
    return {"result": "met" if met else "not_met",
            "path": "training_track_proceeds" if met else "training_track_held",
            "evidence": {name: int(value) for name, value in checks.items()}}


if __name__ == "__main__":
    print(json.dumps(decide(sys.argv[1:]), sort_keys=True))
