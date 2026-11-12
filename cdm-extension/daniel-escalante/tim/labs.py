"""Lab-panel outcomes for the content world model (SPEC_CONTENT_WM.md §1.2).

Each item of a panel becomes one discrete outcome: a train-tertile bin of its numeric value (per unit),
a token from the train vocabulary of non-numeric values, "other", or "absent" when the panel is in the
chart but the item is not. A panel is in the chart exactly when any of its items is reported, the rule
upstream's dataset uses. Bin edges and vocabularies are patient-derived, so they stay private.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import re

NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*(.*?)\s*$")
MIN_NUMERIC, MIN_TOKEN = 30, 10
BINS = ("low", "mid", "high")
OTHER, ABSENT = "other", "absent"


def parse(value: str):
    """(number, unit) for a value that starts with a number, else the whitespace-normalized upper-case token."""
    match = NUMBER.match(value)
    if match:
        return float(match.group(1)), match.group(2)
    return " ".join(value.split()).upper()


@dataclass(frozen=True)
class Bins:
    panels: dict  # panel -> tuple of item ids, in the release mapping's order
    edges: dict   # (item, unit) -> (lower tertile, upper tertile)
    tokens: dict  # item -> sorted tuple of tokens seen at least MIN_TOKEN times (sorted: stable pickles)

    def item_outcome(self, item, value) -> str:
        if value is None:
            return ABSENT
        parsed = parse(value)
        if isinstance(parsed, str):
            return f"t:{parsed}" if parsed in self.tokens.get(item, ()) else OTHER
        number, unit = parsed
        edges = self.edges.get((item, unit))
        if edges is None:
            return OTHER
        return BINS[0] if number <= edges[0] else BINS[1] if number <= edges[1] else BINS[2]

    def outcome(self, panel, record):
        """The panel's outcome tuple, or None when the panel is not in the chart."""
        items = self.panels[panel]
        if not any(item in record for item in items):
            return None
        return tuple(self.item_outcome(item, record.get(item)) for item in items)


def fit(records, panels, min_numeric=MIN_NUMERIC, min_token=MIN_TOKEN) -> Bins:
    """Bins from train admissions' lab records (item id -> value string) only."""
    import numpy as np
    numbers, tokens = defaultdict(list), defaultdict(Counter)
    for record in records:
        for item, value in record.items():
            parsed = parse(value)
            if isinstance(parsed, str):
                tokens[item][parsed] += 1
            else:
                numbers[(item, parsed[1])].append(parsed[0])
    edges = {key: tuple(float(q) for q in np.quantile(values, [1 / 3, 2 / 3]))
             for key, values in numbers.items() if len(values) >= min_numeric}
    vocab = {item: tuple(sorted(t for t, n in counts.items() if n >= min_token)) for item, counts in tokens.items()}
    return Bins({panel: tuple(ids) for panel, ids in panels.items()}, edges, vocab)
