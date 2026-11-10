"""HPI audit (counts only): how often the history summary the agent sees names a test."""
import re

from tim.enums import DISEASES
from tim.inventory import quantiles


def hpi_audit(items_by_split, tests) -> dict:
    report = {}
    for split, items in items_by_split.items():
        histories = [i["patient_history"] for i in items]
        report[split] = {
            "n": len(items),
            "word_count": quantiles([len(h.split()) for h in histories]),
            "mentions": {t: sum(bool(re.search(rf"\b{re.escape(t)}\b", h, re.IGNORECASE)) for h in histories)
                         for t in tests},
            "class_counts": {d: sum(i["condition"] == d for i in items) for d in DISEASES},
        }
    return report
