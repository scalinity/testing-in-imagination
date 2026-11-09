#!/usr/bin/env python3
"""Run a tim command; call it through scripts/tim.sh so output is captured privately."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tim.paths  # noqa: E402,F401
from tim.cli import main  # noqa: E402

main()
