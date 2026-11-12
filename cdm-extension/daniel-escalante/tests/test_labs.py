"""Lab-panel outcomes (SPEC_CONTENT_WM.md §1.2): train-tertile bins per item and unit, a train vocabulary
of non-numeric values, "other" and "absent". Invented values only."""
from tim import labs

PANELS = {"Complete Blood Count": [1, 2], "Urinalysis": [3]}


def train():
    """Item 1: 1..30 u; item 2: 1..30 g plus 3 values in a rare unit; item 3: NEG x12, TRACE x2."""
    records = [{1: f"{v} u", 2: f"{v}.0 g"} for v in range(1, 31)]
    records += [{2: "5 mg"}, {2: "6 mg"}, {2: "7 mg"}]
    records += [{3: "NEG"}] * 12 + [{3: " trace "}] * 2
    return records


def test_parse_reads_a_leading_number_with_its_unit_or_else_a_token():
    assert labs.parse("12.5 K/uL") == (12.5, "K/uL")
    assert labs.parse(" -0.3 mg/dL ") == (-0.3, "mg/dL")
    assert labs.parse("7") == (7.0, "")
    assert labs.parse(".5 u") == (0.5, "u")
    assert labs.parse(" trace  ") == "TRACE"
    assert labs.parse("<0.5 mg/dL") == "<0.5 MG/DL"


def test_numeric_values_fall_in_train_tertile_bins_per_unit():
    bins = labs.fit(train(), PANELS, min_numeric=10, min_token=10)
    assert bins.edges[(1, "u")] == (10.666666666666666, 20.333333333333332)
    outcome = bins.outcome("Complete Blood Count", {1: "3 u", 2: "25 g"})
    assert outcome == ("low", "high")
    assert bins.outcome("Complete Blood Count", {1: "15 u"}) == ("mid", "absent")


def test_rare_units_rare_tokens_and_unseen_values_are_other():
    bins = labs.fit(train(), PANELS, min_numeric=10, min_token=10)
    assert (2, "mg") not in bins.edges  # 3 values, below the minimum
    assert bins.outcome("Complete Blood Count", {1: "3 kg", 2: "5 mg"}) == ("other", "other")
    assert bins.outcome("Urinalysis", {3: "neg"}) == ("t:NEG",)
    assert bins.outcome("Urinalysis", {3: "TRACE"}) == ("other",)  # 2 occurrences, below the minimum
    assert bins.outcome("Urinalysis", {3: "MODERATE"}) == ("other",)


def test_a_panel_is_unavailable_exactly_when_none_of_its_items_is_reported():
    bins = labs.fit(train(), PANELS, min_numeric=10, min_token=10)
    assert bins.outcome("Urinalysis", {1: "3 u"}) is None
    assert bins.outcome("Complete Blood Count", {}) is None


def test_bins_depend_only_on_the_records_they_were_fit_on():
    fitted = labs.fit(train(), PANELS, min_numeric=10, min_token=10)
    extended = labs.fit(train() + [{1: "999 u"}] * 40, PANELS, min_numeric=10, min_token=10)
    assert fitted.edges[(1, "u")] != extended.edges[(1, "u")]
    assert labs.fit(train(), PANELS, min_numeric=10, min_token=10).edges == fitted.edges
