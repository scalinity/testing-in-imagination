#!/usr/bin/env python3
"""No-op test gate, and the contract every gate rule script follows.

The watcher runs a rule as `python <rule.py> <file> ...` with the files the gate lists under
"reads" in graph.json (approved aggregates, or private aggregates it is allowed to read), once,
when the gate's input stages are complete. The rule prints one JSON object and exits 0:

    {"result": "met" | "not_met", "path": <name of the pre-registered path>, "evidence": {...}}

"path" defaults to the result. "evidence" holds names and numbers only; the watcher refuses
anything else. A rule is deterministic: the same files give the same output. It is committed
before its inputs exist and runs under the watcher's Python 3.9, standard library only.

This rule: met when every file exists and reports a count of at least 1.
"""
import json
import sys


def decide(files):
    counts = []
    for name in files:
        try:
            with open(name) as handle:
                counts.append(int(json.load(handle)["count"]))
        except (OSError, ValueError, KeyError, TypeError):
            counts.append(None)
    met = bool(counts) and all(count is not None and count >= 1 for count in counts)
    return {"result": "met" if met else "not_met",
            "evidence": {"files": len(files), "missing": counts.count(None),
                         "lowest_count": min([c for c in counts if c is not None], default=None)}}


if __name__ == "__main__":
    print(json.dumps(decide(sys.argv[1:]), sort_keys=True))
