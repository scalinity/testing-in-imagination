"""Repository locations; puts the replication and the pinned upstream on the import path."""
from pathlib import Path
import sys

OWN = Path(__file__).resolve().parents[1]
ROOT = OWN.parents[1]
REPLICATION = ROOT / "replication-of-original" / "daniel-escalante"
UPSTREAM = ROOT / "lacdm" / "upstream"
SCHEMAS = OWN / "schemas"

for _path in (UPSTREAM, REPLICATION):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
