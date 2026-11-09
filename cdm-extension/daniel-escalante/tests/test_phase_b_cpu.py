import pytest

from tim import audit, fixture, inventory, leakage
from tim.enums import DISEASES

TESTS = list(fixture.TESTS)


def items(n=48):
    out = []
    for i in range(n):
        label = DISEASES[i % 4]
        cue = fixture.CUES[label]
        present = {"CT", "Complete Blood Count"} if i % 2 else {"Ultrasound"}
        out.append({"patient_id": 90000000 + i, "condition": label,
                    "patient_history": f"Invented admission {i % 6}: pain with {cue}. CT ordered" if i % 3 == 0
                    else f"Invented admission {i % 6}: pain with {cue}.",
                    "test_results": {t: ("text\n" if t in present else "not available.\n") for t in TESTS}})
    return out


def test_leakage_probes_fit_on_train_and_report_mean_class_recall():
    train, val = items(48), items(24)
    report = leakage.probes(train, val, TESTS)
    assert set(report) == {"hpi", "hpi_bits", "bits"}
    for probe in report.values():
        assert 0 <= probe["T5"] <= 100 and set(probe["recall"]) == set(DISEASES)
        assert probe["C"] in (0.01, 0.1, 1, 10)
    assert report["hpi"]["T5"] > 50  # the invented cues name the disease


def test_inventory_records_one_row_per_patient_and_test_at_t0():
    rows = inventory.records(items(4), TESTS, "train", source="mimic_real")
    assert len(rows) == 4 * len(TESTS)
    row = rows[1]
    assert row["t"] == 0 and row["source"] == "mimic_real" and row["meta"]["split"] == "train"
    assert row["a_t"] == TESTS[1] and row["o_t1"] in ("text\n", "UNAVAILABLE")
    assert row["label"] == int(row["o_t1"] != "UNAVAILABLE")
    assert row["meta"]["available_tests_at_t"] == TESTS and row["y_true"] in DISEASES


def test_inventory_card_holds_counts_rates_and_quantiles_only():
    card = inventory.card({"train": inventory.records(items(8), TESTS, "train", "mimic_real")}, TESTS)
    assert card["counts"]["train"]["CT"] == {"available": 4, "unavailable": 4}
    assert card["availability_rate"]["train"]["appendicitis"]["CT"] == pytest.approx(0.0)
    assert set(card["hpi_words"]["train"]) == {"min", "q25", "median", "q75", "max"}


def test_hpi_audit_counts_test_mentions_without_text():
    report = audit.hpi_audit({"train": items(12)}, TESTS)
    assert report["train"]["n"] == 12
    assert report["train"]["mentions"]["CT"] == 4
    assert report["train"]["class_counts"] == {d: 3 for d in DISEASES}
