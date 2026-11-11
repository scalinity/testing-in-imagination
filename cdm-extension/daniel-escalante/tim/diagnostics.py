"""Exploratory Phase-B validation diagnostics (amendment A4): the oracle contrasts of Family D and the
tests the reranker added or dropped, judged against the dataset authors' guideline tests."""
import ast

from tim.enums import CONFIGURED_TESTS, DISEASES, PHASE_B_ARMS, TEST_CODES

# Hager et al. (2024), the authors of MIMIC-IV-Ext-CDM: required laboratory item IDs per pathology in
# github.com/paulhager/MIMIC-Clinical-Decision-Making-Framework@e1b4a52 (evaluators/, tools/utils.py).
INFLAMMATION = (51301, 51755, 51300, 50889, 51652)
REQUIRED_ITEMS = {
    "appendicitis": INFLAMMATION,
    "cholecystitis": (*INFLAMMATION, 50861, 50878, 50883, 50927),
    "diverticulitis": INFLAMMATION,
    "pancreatitis": (*INFLAMMATION, 50867, 50956, 51480, 50810, 51221, 51638, 51006, 52647, 51000, 50893,
                     50824, 52623, 50983, 52610, 50971, 50822),
}
# The abdominal modalities each evaluator scores; HIDA and EUS are not actions here.
IMAGING = {"appendicitis": ("Ultrasound", "CT", "MRI"), "cholecystitis": ("Ultrasound", "MRI"),
           "diverticulitis": ("CT", "Ultrasound", "MRI"), "pancreatitis": ("Ultrasound", "CT")}
LABS = CONFIGURED_TESTS[:7]
CATEGORY = {**dict.fromkeys(LABS, "lab"), **dict.fromkeys(CONFIGURED_TESTS[7:11], "imaging"),
            "Physical Examination": "exam"}
COUNTS = ("extra", "extra_provided", "extra_relevant", "extra_relevant_provided",
          "dropped", "dropped_provided", "dropped_relevant", "dropped_relevant_provided")
# D3: the arms the dose-response needs that validation lacks (B-marginal is its alpha = 0.5 point).
DOSE_RESPONSE_ARMS = tuple(arm for arm in PHASE_B_ARMS if arm == "B-marginal" or arm.startswith("B-a"))
NAMED = {t.lower(): t for t in CONFIGURED_TESTS}  # rows name tests in lower case, as the cost table does


def _ids(value) -> list[int]:
    return [int(x) for x in (ast.literal_eval(value) if isinstance(value, str) else value)]


def _item(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == number and number.is_integer() else None


def panel_items(frame) -> dict[str, set[int]]:
    """Each lab panel's item IDs in the release mapping, plus the IDs each listed item maps to."""
    maps_to = {_item(r["itemid"]): _ids(r["corresponding_ids"]) for r in frame.to_dict("records")
               if _item(r["itemid"]) is not None}
    out = {}
    for panel in LABS:
        listed = _ids(frame.loc[frame["label"] == panel, "corresponding_ids"].item())
        out[panel] = set(listed) | {x for item in listed for x in maps_to.get(item, ())}
    return out


def relevance(frame) -> dict[str, dict[str, bool]]:
    """test name -> disease -> diagnosis-relevant, by the rule in amendment A4."""
    panels = panel_items(frame)

    def relevant(test, disease):
        if test in panels:
            return bool(panels[test] & set(REQUIRED_ITEMS[disease]))
        return test == "Physical Examination" or test in IMAGING[disease]
    return {test: {d: relevant(test, d) for d in DISEASES} for test in CONFIGURED_TESTS}


def changes(arm_rows, p0_rows, relevant) -> dict:
    """Tests the arm requested that P0 did not (extra) and the reverse (dropped), once per patient."""
    if [r["patient_id"] for r in arm_rows] != [r["patient_id"] for r in p0_rows]:
        raise ValueError("arm and P0 rows must list the same patients in the same order")
    by_test = {test: dict.fromkeys(COUNTS, 0) for test in CONFIGURED_TESTS}
    for arm, p0 in zip(arm_rows, p0_rows):
        for kind, run, other in (("extra", arm, p0), ("dropped", p0, arm)):
            for name in set(run["requested_tests"]) - set(other["requested_tests"]):
                test = NAMED[name.lower()]
                provided, useful = name in run["provided_tests"], relevant[test][run["label"]]
                counts = by_test[test]
                counts[kind] += 1
                counts[f"{kind}_provided"] += provided
                counts[f"{kind}_relevant"] += useful
                counts[f"{kind}_relevant_provided"] += provided and useful

    def total(tests):
        return {name: sum(by_test[t][name] for t in tests) for name in COUNTS}
    return {"by_test": {TEST_CODES[t]: c for t, c in by_test.items()},
            "by_category": {c: total([t for t in CONFIGURED_TESTS if CATEGORY[t] == c])
                            for c in ("lab", "imaging", "exam")},
            "total": total(CONFIGURED_TESTS)}


def diagnostics(rows, p0, cfg, source, relevant, resamples=10_000) -> dict:
    """diagnose-val-b: Family D (Holm) and the extra and dropped tests of every val-b arm against P0."""
    from tim import analysis
    family = analysis.holm({
        name: analysis.compare(rows["B-oracle"], other, cfg, name, "val", resamples=resamples)
        for name, other in (("B-oracle-B-uniform", rows["B-uniform"]), ("B-oracle-P0", p0))})
    return {"source": source, "family_d": family,
            "relevance": {TEST_CODES[t]: by_disease for t, by_disease in relevant.items()},
            "extra_tests": {arm: changes(r, p0, relevant) for arm, r in rows.items()}}
