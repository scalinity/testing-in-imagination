"""Command-line audit (SPEC 9.1).

    uv run python -m tiicee.cee.audit FILE.json [--json]

FILE holds a list of decision points, each an object with id, classes, prior, outcomes, weights,
posteriors (one row per outcome), and optionally budget, real_outcome, y_true and utility. The
command prints an audit card in markdown, or the same content as JSON with --json. To audit a
reader and a world model directly, use audit_point.
"""
from __future__ import annotations

import argparse
import json
import sys

from .card import audit_point, audit_record, render_markdown, summarize
from .core import FLOOR, MAX_ITER, TOL

__all__ = ["audit_point", "audit_record", "main"]


def main(argv=None):
    """Run the audit command (SPEC 9.1). Returns 0, or 2 when a decision point is malformed."""
    parser = argparse.ArgumentParser(prog="python -m tiicee.cee.audit", description="Print a CEE audit card for a file of decision points.")
    parser.add_argument("file", help="JSON file holding a list of decision points")
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    args = parser.parse_args(argv)
    with open(args.file) as handle:
        points = json.load(handle)
    records = []
    for position, point in enumerate(points):
        try:
            records.append(audit_record(point))
        except (ValueError, TypeError, AttributeError) as error:
            name = point.get("id", position) if isinstance(point, dict) else position
            print(f"error: decision point {name!r}: {error}", file=sys.stderr)
            return 2
    summary = summarize(records)
    if args.json:
        payload = {"source": args.file, "floor": FLOOR, "tol": TOL, "max_iter": MAX_ITER, "points": records, "summary": summary}
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(records, summary, args.file), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
