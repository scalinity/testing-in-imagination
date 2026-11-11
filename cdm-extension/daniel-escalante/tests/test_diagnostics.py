"""Exploratory Phase-B validation diagnostics (prereg/amendments/A4-validation-diagnostics.md)."""
import numpy as np
import pandas as pd
import pytest

from tim import analysis, diagnostics, render
from tim.enums import CONFIGURED_TESTS, DISEASES, TEST_CODES
from tim.events import TimError, load_schema, validate

PANELS = {"Complete Blood Count": [51301], "Basic Metabolic Panel": [50824], "Comprehensive Metabolic Panel": [50878],
          "Renal Function Panel": [51006], "Liver Function Panel": [50861], "Urinalysis": [51486],
          "Electrolyte Panel": [99999]}


def mapping():
    """Release-style mapping: panel rows with a blank item ID list their items; item 99999 maps to lipase."""
    rows = [{"itemid": "", "label": panel, "corresponding_ids": repr(items)} for panel, items in PANELS.items()]
    rows += [{"itemid": i, "label": f"item {i}", "corresponding_ids": repr([i])} for i in (51301, 50824, 50878,
                                                                                         51006, 50861, 51486, 50956)]
    rows.append({"itemid": 99999, "label": "item alias", "corresponding_ids": repr([50956])})
    frame = pd.DataFrame(rows)
    frame["itemid"] = pd.to_numeric(frame["itemid"])  # blank panel IDs read as NaN, as pd.read_csv does
    return frame


def relevant_to(table, test):
    return {d for d in DISEASES if table[test][d]}


def test_relevance_follows_the_dataset_authors_required_tests():
    table = diagnostics.relevance(mapping())
    assert set(table) == set(CONFIGURED_TESTS)
    assert relevant_to(table, "Complete Blood Count") == set(DISEASES)  # white blood cells: inflammation
    assert relevant_to(table, "Liver Function Panel") == {"cholecystitis"}
    assert relevant_to(table, "Comprehensive Metabolic Panel") == {"cholecystitis"}
    assert relevant_to(table, "Basic Metabolic Panel") == {"pancreatitis"}  # sodium: severity
    assert relevant_to(table, "Renal Function Panel") == {"pancreatitis"}  # urea nitrogen: severity
    assert relevant_to(table, "Electrolyte Panel") == {"pancreatitis"}  # an alias of lipase is followed
    assert relevant_to(table, "Urinalysis") == set()
    assert relevant_to(table, "CT") == {"appendicitis", "diverticulitis", "pancreatitis"}
    assert relevant_to(table, "MRI") == {"appendicitis", "cholecystitis", "diverticulitis"}
    assert relevant_to(table, "Ultrasound") == set(DISEASES)
    assert relevant_to(table, "Radiograph") == set()
    assert relevant_to(table, "Physical Examination") == set(DISEASES)


def row(pid, label, requested, provided, prediction=None):
    return {"patient_id": pid, "label": label, "prediction": prediction, "requested_tests": requested,
            "provided_tests": provided, "ece_triplets": []}


def test_extra_and_dropped_tests_are_counted_once_per_patient_by_test_category_and_total():
    table = diagnostics.relevance(mapping())
    # Rows name tests in lower case, as the replication's cost table does.
    p0 = [row(1, "appendicitis", ["physical examination", "radiograph", "complete blood count"],
              ["physical examination", "complete blood count"]),
          row(2, "pancreatitis", ["ultrasound"], [])]
    arm = [row(1, "appendicitis", ["physical examination", "ct", "complete blood count", "ct"],
               ["physical examination", "ct", "complete blood count"]),
           row(2, "pancreatitis", ["basic metabolic panel"], ["basic metabolic panel"])]
    result = diagnostics.changes(arm, p0, table)
    assert result["total"] == {"extra": 2, "extra_provided": 2, "extra_relevant": 2, "extra_relevant_provided": 2,
                               "dropped": 2, "dropped_provided": 0, "dropped_relevant": 1,
                               "dropped_relevant_provided": 0}
    assert set(result["by_test"]) == set(TEST_CODES.values())
    assert result["by_test"]["CT"]["extra"] == 1 and result["by_test"]["BMP"]["extra_relevant"] == 1
    assert result["by_test"]["XR"]["dropped"] == 1 and result["by_test"]["XR"]["dropped_relevant"] == 0
    assert result["by_test"]["US"]["dropped_relevant"] == 1 and result["by_test"]["PE"]["extra"] == 0
    assert result["by_category"]["imaging"]["extra"] == 1 and result["by_category"]["imaging"]["dropped"] == 2
    assert result["by_category"]["lab"]["extra"] == 1 and result["by_category"]["exam"] == dict.fromkeys(
        diagnostics.COUNTS, 0)


def test_changes_refuse_rows_for_different_patients():
    with pytest.raises(ValueError):
        diagnostics.changes([row(1, "appendicitis", [], [])], [row(2, "appendicitis", [], [])], {})


def arms(seed=0, n=48):
    rng = np.random.default_rng(seed)
    labels = [DISEASES[i % 4] for i in range(n)]

    def run():
        return [row(1000 + i, label, ["physical examination"], ["physical examination"],
                    prediction=label if rng.random() < 0.75 else None) for i, label in enumerate(labels)]
    return {"B-uniform": run(), "B-learned-s0": run(), "B-oracle": run()}, run()


def _cfg():
    from replication.config import compose
    return compose("primary")


def test_family_d_is_holm_adjusted_and_reuses_the_published_bootstrap_seed():
    rows, p0 = arms()
    result = diagnostics.diagnostics(rows, p0, _cfg(), "synthetic", diagnostics.relevance(mapping()), resamples=200)
    validate(result, load_schema("diagnostics"))
    family = result["family_d"]
    assert set(family) == {"B-oracle-B-uniform", "B-oracle-P0"}
    assert all(c["p_holm"] is not None and c["p_holm"] >= c["p_value"] for c in family.values())
    published = analysis.compare(rows["B-oracle"], p0, _cfg(), "B-oracle-P0", "val", tested=False, resamples=200)
    assert family["B-oracle-P0"]["ci95"] == published["ci95"]
    assert set(result["extra_tests"]) == set(rows)
    assert result["relevance"]["CBC"] == dict.fromkeys(DISEASES, True)


def test_the_diagnostics_stage_reads_validation_only():
    from tim.access import mode
    assert mode("diagnose-val-b", "val") == "full"
    for split in ("train", "test"):
        with pytest.raises(TimError):
            mode("diagnose-val-b", split)


# ---- check H (amendment A5) ---------------------------------------------------------------------

def hrow(pid, label, requested, provided, hypotheses, termination="diagnosis"):
    return {**row(pid, label, requested, provided), "hypotheses": hypotheses, "termination": termination}


def test_the_timeline_skips_no_hypothesis_after_an_unavailable_result():
    # ct is unavailable, so upstream stores no hypothesis on the next step.
    r = hrow(1, "appendicitis", ["ct", "complete blood count", "ultrasound"], ["complete blood count", "ultrasound"],
             ["pancreatitis", "appendicitis", "cholecystitis"])
    assert diagnostics.timeline(r) == [("ct", "pancreatitis", None),
                                       ("complete blood count", "pancreatitis", "appendicitis"),
                                       ("ultrasound", "appendicitis", "cholecystitis")]
    assert diagnostics.timeline({**r, "hypotheses": r["hypotheses"][:2]}) is None  # one hypothesis short
    assert diagnostics.timeline({**r, "termination": "length_observation"}) is None


def test_check_h_counts_added_and_dropped_relevant_results_by_the_hypothesis_held_then():
    table = diagnostics.relevance(mapping())
    p0 = [hrow(1, "appendicitis", ["radiograph", "ultrasound"], ["ultrasound"], ["appendicitis", "appendicitis"]),
          hrow(2, "pancreatitis", ["ultrasound"], ["ultrasound"], ["pancreatitis", "pancreatitis"])]
    arm = [hrow(1, "appendicitis", ["complete blood count", "ct"], ["complete blood count", "ct"],
                ["pancreatitis", "appendicitis", "appendicitis"]),
           hrow(2, "pancreatitis", ["ultrasound", "complete blood count"], ["ultrasound", "complete blood count"],
                ["pancreatitis"], termination="length_observation")]
    result = diagnostics.hypothesis_check({"B-oracle": arm}, p0, table, "synthetic")
    from replication.manifest import assert_aggregate
    validate(result, load_schema("hypothesis_check"))
    assert_aggregate(result)  # the replication's export guard, as Bundle.aggregate applies it
    assert result["alignment"] == {"B-oracle": {"admissions": 2, "aligned": 1}, "P0": {"admissions": 2, "aligned": 2}}
    assert result["extra"]["B-oracle"] == {"tests": 2, "unaligned": 1, "before_correct": 1, "before_other": 1,
                                           "before_unknown": 0, "changed": 1, "changed_to_correct": 1,
                                           "changed_from_correct": 0}
    assert result["dropped"]["B-oracle"] == {"tests": 1, "unaligned": 0, "before_correct": 1, "before_other": 0,
                                             "before_unknown": 0, "changed": 0, "changed_to_correct": 0,
                                             "changed_from_correct": 0}
    totals = diagnostics.changes(arm, p0, table)["total"]
    assert result["extra"]["B-oracle"]["tests"] + result["extra"]["B-oracle"]["unaligned"] == \
        totals["extra_relevant_provided"]


# ---- the dose-response stage (amendment A5) --------------------------------------------------------

def test_the_dose_response_stage_runs_the_nine_d3_arms_on_validation_only():
    from tim.access import mode
    from tim.runner import STAGE_SPECS
    assert STAGE_SPECS["dose-val-b"] == {"split": "val", "jobs": list(diagnostics.DOSE_RESPONSE_ARMS)}
    assert len(diagnostics.DOSE_RESPONSE_ARMS) == 9 and mode("dose-val-b", "val") == "full"
    with pytest.raises(TimError):
        mode("dose-val-b", "test")


# ---- DIAGNOSTICS.md ----------------------------------------------------------------------------

def contrast(delta, ci, p_holm):
    return {"delta_t5": delta, "ci95": ci, "p_value": p_holm, "p_holm": p_holm, "method": "exact",
            "discordant": 9, "significance": analysis.significance(p_holm, ci), "mcnemar_p": None}


def bundle(oracle_uniform, source="mimic", ua_extra=0):
    counts = dict.fromkeys(diagnostics.COUNTS, 0)

    def changed():
        by_test = {code: dict(counts) for code in TEST_CODES.values()}
        by_test["UA"] = {**counts, "extra": ua_extra, "extra_provided": ua_extra}
        return {"by_test": by_test, "by_category": {c: dict(counts) for c in ("lab", "imaging", "exam")},
                "total": {**counts, "extra": ua_extra, "extra_provided": ua_extra}}
    relevance = {code: dict.fromkeys(DISEASES, code not in ("UA", "XR")) for code in TEST_CODES.values()}
    return {"source": source,
            "family_d": {"B-oracle-B-uniform": oracle_uniform, "B-oracle-P0": contrast(1.4672, [-0.72, 3.83], 0.4)},
            "relevance": relevance, "extra_tests": {arm: changed() for arm in ("B-uniform", "B-learned-s0", "B-oracle")}}


def summary(t5, unavailable):
    return {"n": 236, "metrics": {"T5": t5}, "requests": {"requested": 0, "provided": 0, "unavailable": unavailable,
                                                "repeated": 0}, "events": {"wm_rerank": 0, "wm_stop": 0}}


VAL_B = {"variants": {"P0": summary(83.12, 207), "B-uniform": summary(84.70, 203),
                      "B-learned-s0": summary(82.92, 153), "B-oracle": summary(84.59, 66)},
         "descriptive": {"B-oracle-P0": {"ci95": [-0.72, 3.83]}}}
H_COUNTS = {"tests": 41, "unaligned": 3, "before_correct": 29, "before_other": 9, "before_unknown": 3, "changed": 7,
            "changed_to_correct": 4, "changed_from_correct": 1}
HYPOTHESIS = {"source": "mimic",
              "alignment": {arm: {"admissions": 236, "aligned": 230} for arm in ("P0", "B-uniform", "B-learned-s0", "B-oracle")},
              "extra": {arm: dict(H_COUNTS) for arm in ("B-uniform", "B-learned-s0", "B-oracle")},
              "dropped": {arm: dict(H_COUNTS) for arm in ("B-uniform", "B-learned-s0", "B-oracle")}}


def rendered(oracle_uniform=None, **kwargs):
    oracle_uniform = oracle_uniform or contrast(-0.1066, [-2.9, 2.7], 0.95)
    return render.render_diagnostics(bundle(oracle_uniform, **kwargs), VAL_B, threshold=4.1537, dose_hours=4.122,
                                     hypothesis=HYPOTHESIS)


@pytest.mark.parametrize("oracle_uniform, verdict", [
    (contrast(5.1234, [2.1, 8.0], 0.004), "signal bottleneck"),
    (contrast(-4.5678, [-8.0, -1.2], 0.01), "planner bottleneck"),
    (contrast(-0.1066, [-2.9, 2.7], 0.95), "planner bottleneck"),
    (contrast(1.2345, [-1.5, 4.6], 0.4), "inconclusive"),
])
def test_the_provisional_verdict_follows_the_preregistered_rule(oracle_uniform, verdict):
    first = next(line for line in rendered(oracle_uniform).splitlines() if line.startswith("**Provisional"))
    assert first.startswith(f"**Provisional: {verdict}.**")
    for caveat in ("rests on one validation contrast", "has not run",
                   "written after the validation point estimates were known"):
        assert caveat in first
    assert f"{oracle_uniform['delta_t5']:+.2f}" in first
    if verdict != "signal bottleneck":
        assert "4.15" in first
    if oracle_uniform["significance"] != "statistically significant":
        assert "not statistically significant" in first


def test_diagnostics_report_states_scope_and_lists_the_dose_response_as_pending():
    text = rendered()
    assert "exploratory" in text.lower() and "the 236 validation admissions" in text and "240" not in text
    assert "not a reproduction" in text.lower() and "not blind" in text and "matches the published" in text
    pending = text[text.index("## Pending"):]
    assert "confirmatory" in pending and "9 validation runs" in pending
    assert "about 4.1 GPU-hours" in pending and "an estimate" in pending
    assert not text.startswith("> **SYNTHETIC**")
    assert rendered(source="synthetic").startswith("> **SYNTHETIC**")


def test_check_h_is_reported_as_an_untested_hypothesis_with_counts():
    section = rendered().split("## Do the added results repeat")[1].split("\n## ")[0]
    assert "untested hypothesis" in section and "aggregate" in section
    assert "| B-oracle | 41 | 3 | 29 | 9 | 3 | 7 | 4 | 1 |" in section


def test_sensitivity_s1_adds_urinalysis_and_radiographs_to_the_relevant_counts():
    text = rendered(ua_extra=5)
    section = text.split("## Sensitivity S1")[1].split("\n## ")[0]
    assert "exploratory" in section and "| B-oracle | 5 | 5 | 0 | 0 | +5 | +0 |" in section
    assert "urinalysis and radiographs" in text.split("## Limitations")[1]
    with pytest.raises(ValueError):
        b = bundle(contrast(-0.1066, [-2.9, 2.7], 0.95))
        b["relevance"]["UA"]["appendicitis"] = True
        render.render_diagnostics(b, VAL_B, threshold=4.1537, dose_hours=4.122, hypothesis=HYPOTHESIS)
