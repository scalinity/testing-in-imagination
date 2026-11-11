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


# ---- check H (amendment A5): the hypothesis held when each added result was requested -----------

H_COUNTS = ("tests", "unaligned", "before_correct", "before_other", "before_unknown", "changed",
            "changed_to_correct", "changed_from_correct")


def timeline(row):
    """(test, hypothesis held when it was requested, hypothesis stored after its result) per request,
    or None when the row's stored hypotheses cannot be aligned with its requests. Upstream stores a
    hypothesis at the start of every step except the step after an unavailable result."""
    requested, provided, hypotheses = row["requested_tests"], set(row["provided_tests"]), row["hypotheses"]
    if (row.get("termination") != "diagnosis" or len(set(requested)) != len(requested)
            or len(hypotheses) != 1 + sum(test in provided for test in requested)):
        return None
    steps, current = [], 0
    for test in requested:
        if test in provided:
            steps.append((test, hypotheses[current], hypotheses[current + 1]))
            current += 1
        else:
            steps.append((test, hypotheses[current], None))
    return steps


def _tally(pairs, relevant) -> dict:
    """Check-H counts for the tests each run requested that the other did not, relevant and provided."""
    counts = dict.fromkeys(H_COUNTS, 0)
    for run, other in pairs:
        label = run["label"]
        chosen = {name for name in set(run["requested_tests"]) - set(other["requested_tests"])
                  if name in run["provided_tests"] and relevant[NAMED[name.lower()]][label]}
        steps = timeline(run)
        if steps is None:
            counts["unaligned"] += len(chosen)
            continue
        for test, before, after in steps:
            if test not in chosen:
                continue
            counts["tests"] += 1
            counts["before_correct" if before == label else "before_unknown" if before == "unknown"
                   else "before_other"] += 1
            if after != before:
                counts["changed"] += 1
                counts["changed_to_correct"] += after == label
                counts["changed_from_correct"] += before == label
    return counts


def hypothesis_check(rows, p0, relevant, source) -> dict:
    """Check H: every val-b arm's added relevant results, and P0's dropped ones, by hypothesis held then."""
    for arm_rows in rows.values():
        if [r["patient_id"] for r in arm_rows] != [r["patient_id"] for r in p0]:
            raise ValueError("arm and P0 rows must list the same patients in the same order")

    def aligned(run_rows):
        return {"admissions": len(run_rows), "aligned": sum(timeline(r) is not None for r in run_rows)}
    return {"source": source,
            "alignment": {"P0": aligned(p0), **{arm: aligned(r) for arm, r in rows.items()}},
            "extra": {arm: _tally(zip(r, p0), relevant) for arm, r in rows.items()},
            "dropped": {arm: _tally(zip(p0, r), relevant) for arm, r in rows.items()}}


# ---- D3: the corruption dose-response (amendment A4), fixed in code before any dose-val-b run ------

DOSE_PRIMARY = (("B-oracle", 0.0), ("B-a0.1-T1-n0", 0.1), ("B-a0.1-T1-n1", 0.1), ("B-a0.3-T1-n0", 0.3),
                ("B-a0.3-T1-n1", 0.3), ("B-marginal", 0.5))  # T = 1; B-marginal is alpha = 0.5 (SPEC 4.3)
DOSE_SECONDARY = (("B-oracle", 0.0), ("B-a0.1-T0.25-n0", 0.1), ("B-a0.1-T0.25-n1", 0.1),
                  ("B-a0.3-T0.25-n0", 0.3), ("B-a0.3-T0.25-n1", 0.3))  # T = 0.25
DOSE_SEED_TAG = "tim-dose-response"  # seed = stream_seed(269, "val", DOSE_SEED_TAG)


def dose_response(rows, points, seed, resamples=10_000) -> dict:
    """OLS slope of each arm's T5 on alpha (one point per arm), with a 95% class-stratified percentile
    bootstrap that draws patients once per resample and applies the draw to every arm."""
    import numpy as np
    from tim.analysis import TOLERANCE  # float residue below this is zero
    arms = [arm for arm, _ in points]
    ids = [r["patient_id"] for r in rows[arms[0]]]
    if any([r["patient_id"] for r in rows[arm]] != ids for arm in arms):
        raise ValueError("every arm must list the same patients in the same order")
    labels = np.array([r["label"] for r in rows[arms[0]]])
    correct = np.array([[r["prediction"] == r["label"] for r in rows[arm]] for arm in arms], dtype=float)
    classes = [np.flatnonzero(labels == d) for d in DISEASES if (labels == d).any()]
    x = np.array([alpha for _, alpha in points], dtype=float)
    centred = x - x.mean()

    def slope(t5):  # t5: arms x anything
        return centred @ t5 / (centred @ centred)

    t5 = 100 * np.mean([correct[:, members].mean(axis=1) for members in classes], axis=0)
    rng, recall = np.random.default_rng(seed), np.zeros((len(arms), resamples))
    for members in classes:
        draws = members[rng.integers(0, len(members), size=(resamples, len(members)))]
        recall += correct[:, draws].mean(axis=2)
    low, high = (float(v) for v in np.percentile(slope(100 * recall / len(classes)), [2.5, 97.5]))
    levels = sorted(set(x.tolist()))
    level_t5 = [float(np.mean(t5[x == level])) for level in levels]
    return {"points": [{"arm": arm, "alpha": float(a), "t5": float(v)} for arm, a, v in zip(arms, x, t5)],
            "levels": [{"alpha": level, "t5": value} for level, value in zip(levels, level_t5)],
            "slope": float(slope(t5)), "ci95": [low, high],
            "reading": "decreasing" if high < -TOLERANCE else "increasing" if low > TOLERANCE else "flat",
            "levels_non_increasing": all(b <= a + 1e-9 for a, b in zip(level_t5, level_t5[1:])),
            "resamples": resamples}
