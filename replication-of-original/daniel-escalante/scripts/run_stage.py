#!/usr/bin/env python3
"""Run a replication stage from any working directory."""
from pathlib import Path
import sys

own = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(own))
sys.path.insert(0, str(own.parents[1] / "lacdm/upstream"))
sys.path.insert(0, str(own.parents[1]))

from replication.pipeline import main

main()
