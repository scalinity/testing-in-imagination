"""The corruption dose-response of amendment A4, D3: OLS slope of validation T5 on alpha with a
class-stratified percentile bootstrap that draws patients once per resample for every arm."""
import numpy as np
import pytest

from tim import analysis, diagnostics
from tim.enums import DISEASES


def rows(correct_by_patient, n_per_class=12):
    """Rows for one arm; correct_by_patient(i) says whether patient i is diagnosed correctly."""
    labels = [DISEASES[i % 4] for i in range(4 * n_per_class)]
    return [{"patient_id": 5000 + i, "label": label, "prediction": label if correct_by_patient(i) else None,
             "requested_tests": [], "provided_tests": [], "ece_triplets": []} for i, label in enumerate(labels)]


def arms(rate_by_arm, n_per_class=12):
    """Each arm is correct on the first rate share of every class's patients."""
    return {arm: rows(lambda i, r=rate: (i // 4) < round(r * n_per_class), n_per_class)
            for arm, rate in rate_by_arm.items()}


def test_the_point_estimate_is_the_ols_slope_of_each_arms_t5_on_alpha():
    points = (("B-oracle", 0.0), ("B-a0.1-T1-n0", 0.1), ("B-a0.1-T1-n1", 0.1), ("B-marginal", 0.5))
    data = arms({"B-oracle": 1.0, "B-a0.1-T1-n0": 0.75, "B-a0.1-T1-n1": 0.5, "B-marginal": 0.25})
    result = diagnostics.dose_response(data, points, seed=7, resamples=200)
    t5 = {p["arm"]: p["t5"] for p in result["points"]}
    from replication.config import compose
    for arm, arm_rows in data.items():
        assert t5[arm] == pytest.approx(analysis.summarize(arm_rows, compose("primary"))["metrics"]["T5"])
    expected = np.polyfit([a for _, a in points], [t5[arm] for arm, _ in points], 1)[0]
    assert result["slope"] == pytest.approx(expected)
    assert result["levels"] == [{"alpha": 0.0, "t5": 100.0}, {"alpha": 0.1, "t5": 62.5}, {"alpha": 0.5, "t5": 25.0}]
    assert result["levels_non_increasing"] is True
    assert result["ci95"][0] <= result["slope"] <= result["ci95"][1]
    assert result["reading"] == "decreasing" and result["resamples"] == 200


def test_one_draw_of_patients_is_shared_by_every_arm():
    # Identical arms: with shared draws every resampled slope is exactly 0; independent draws would spread.
    same = lambda i: i % 3 != 0
    data = {arm: rows(same) for arm in ("B-oracle", "B-a0.3-T1-n0", "B-marginal")}
    result = diagnostics.dose_response(data, (("B-oracle", 0.0), ("B-a0.3-T1-n0", 0.3), ("B-marginal", 0.5)),
                                       seed=3, resamples=300)
    assert result["slope"] == pytest.approx(0, abs=1e-9) and result["reading"] == "flat"
    assert result["ci95"] == pytest.approx([0, 0], abs=1e-9)


def test_an_increasing_curve_reads_increasing_and_the_bootstrap_is_reproducible():
    points = (("B-oracle", 0.0), ("B-a0.3-T1-n0", 0.3), ("B-marginal", 0.5))
    data = arms({"B-oracle": 0.25, "B-a0.3-T1-n0": 0.75, "B-marginal": 1.0}, n_per_class=20)
    first = diagnostics.dose_response(data, points, seed=11, resamples=500)
    assert first["reading"] == "increasing" and first["levels_non_increasing"] is False
    assert diagnostics.dose_response(data, points, seed=11, resamples=500) == first


def test_rows_for_different_patients_are_refused():
    data = arms({"B-oracle": 1.0, "B-marginal": 0.5})
    data["B-marginal"] = data["B-marginal"][1:] + data["B-marginal"][:1]
    with pytest.raises(ValueError):
        diagnostics.dose_response(data, (("B-oracle", 0.0), ("B-marginal", 0.5)), seed=1, resamples=50)


def test_the_preregistered_arms_levels_and_seed_tag():
    assert diagnostics.DOSE_PRIMARY == (("B-oracle", 0.0), ("B-a0.1-T1-n0", 0.1), ("B-a0.1-T1-n1", 0.1),
                                        ("B-a0.3-T1-n0", 0.3), ("B-a0.3-T1-n1", 0.3), ("B-marginal", 0.5))
    assert diagnostics.DOSE_SECONDARY == (("B-oracle", 0.0), ("B-a0.1-T0.25-n0", 0.1), ("B-a0.1-T0.25-n1", 0.1),
                                          ("B-a0.3-T0.25-n0", 0.3), ("B-a0.3-T0.25-n1", 0.3))
    assert diagnostics.DOSE_SEED_TAG == "tim-dose-response"
    used = {arm for arm, _ in diagnostics.DOSE_PRIMARY + diagnostics.DOSE_SECONDARY} - {"B-oracle"}
    assert used == set(diagnostics.DOSE_RESPONSE_ARMS)


def test_the_dose_analysis_stage_reads_validation_only():
    from tim.access import mode
    from tim.events import TimError
    assert mode("analyze-dose-val-b", "val") == "full"
    with pytest.raises(TimError):
        mode("analyze-dose-val-b", "test")
