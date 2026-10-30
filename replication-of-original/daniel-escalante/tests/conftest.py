from pathlib import Path
import sys

OWN = Path(__file__).resolve().parents[1]
ROOT = OWN.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lacdm/upstream"))
sys.path.insert(0, str(OWN))
