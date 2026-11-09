"""Initial-state action/outcome inventory (SPEC 4.8): one record per patient and test at t = 0.
Records are private JSONL (train and val only); only the aggregate card is public."""
import numpy as np

from tim.availability import SENTINEL
from tim.enums import DISEASES

QUANTILES = {"min": 0, "q25": 25, "median": 50, "q75": 75, "max": 100}


def records(items, tests, split, source) -> list[dict]:
    """These are initial-state pairs, not sequential-history training tuples."""
    out = []
    for item in items:
        for test in tests:
            result = item["test_results"][test]
            available = result != SENTINEL
            out.append({"episode_id": str(item["patient_id"]), "t": 0, "h_t": item["patient_history"], "a_t": test,
                        "o_t1": result if available else "UNAVAILABLE", "y_true": item["condition"],
                        "label": int(available), "source": source,
                        "meta": {"split": split, "available_tests_at_t": list(tests), "seed": 0}})
    return out


def quantiles(values) -> dict:
    return {name: float(np.percentile(values, q)) for name, q in QUANTILES.items()}


def card(records_by_split, tests) -> dict:
    counts, rates, words = {}, {}, {}
    for split, rows in records_by_split.items():
        counts[split] = {t: {"available": sum(r["label"] for r in rows if r["a_t"] == t),
                             "unavailable": sum(1 - r["label"] for r in rows if r["a_t"] == t)} for t in tests}
        rates[split] = {d: {t: float(np.mean([r["label"] for r in rows if r["a_t"] == t and r["y_true"] == d]))
                            for t in tests} for d in DISEASES if any(r["y_true"] == d for r in rows)}
        histories = {r["episode_id"]: r["h_t"] for r in rows}
        words[split] = quantiles([len(h.split()) for h in histories.values()])
    return {"counts": counts, "availability_rate": rates, "hpi_words": words}
