"""The frozen protocol (prereg/protocol.json) and the synthetic stand-in constants."""
import hashlib
import json

from tim.enums import DISEASES
from tim.paths import OWN

PATH = OWN / "prereg" / "protocol.json"
# The synthetic run exercises the pipeline at fixture scale with fewer resamples; numbers are meaningless.
SYNTHETIC = {"t5_reference": None, "training_majority": "appendicitis",
             "test_class_sizes": {d: 6 for d in DISEASES}, "bootstrap_resamples": 200,
             "sensitivity_simulations": 300, "premise_minimum_nonanswers": 10}


def load(settings) -> tuple[dict, str]:
    """(constants, sha256 of protocol.json); credentialed stages require the committed file."""
    from tim.access import check_admission_path
    from tim.events import TimError
    if not PATH.exists():
        if settings.synthetic:
            return SYNTHETIC, "0" * 64
        raise TimError("approval_missing", "commit prereg/protocol.json before any credentialed stage")
    data = PATH.read_bytes()
    protocol = json.loads(data)
    check_admission_path(settings, protocol)
    constants = SYNTHETIC if settings.synthetic else protocol["analysis"]
    return constants, hashlib.sha256(data).hexdigest()
