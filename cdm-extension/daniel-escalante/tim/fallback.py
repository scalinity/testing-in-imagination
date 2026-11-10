"""G3 hypothesis fallback, applied only at finalization (SPEC §3.2, §3.3 step 8)."""
from tim.enums import DISEASES


def finalize(raw_prediction, hypotheses, g3: bool):
    """(finalized_prediction, prediction_source) from upstream's diagnosis and the hypothesis history."""
    if raw_prediction is not None:
        return raw_prediction, "diagnosis"
    if g3:
        valid = [h for h in hypotheses if h in DISEASES]
        if valid:
            return valid[-1], "hypothesis_fallback"
    return None, "none"
