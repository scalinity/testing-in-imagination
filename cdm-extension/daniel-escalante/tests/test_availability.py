import math

import numpy as np
import pytest

from tim import availability as av

TESTS = ["Complete Blood Count", "CT", "MRI"]


def item(pid, history, present):
    return {"patient_id": pid, "patient_history": history, "condition": "appendicitis",
            "test_results": {t: ("value\n" if t in present else "not available.\n") for t in TESTS}}


def test_labels_follow_the_retrieval_rule_exactly():
    assert av.bits(item(1, "h", {"CT"}), TESTS) == {"Complete Blood Count": 0, "CT": 1, "MRI": 0}


def test_the_mask_draw_is_the_specs_hash():
    import hashlib
    expected = int(hashlib.sha256(b"tim-mask|1|test|900|CT").hexdigest()[:16], 16) / 16 ** 16
    assert av.mask_draw(1, "test", 900, "CT") == expected


@pytest.mark.parametrize("temperature", [1, 0.25])
def test_alpha_zero_gives_the_oracle_for_every_temperature(temperature):
    for y in (0, 1):
        p = av.corrupted(y, 0.3, 0.0, temperature, 0, "val", 7, "CT")
        assert p == pytest.approx(av.clip(y))


def test_alpha_half_at_t1_gives_the_marginal():
    for y in (0, 1):
        assert av.corrupted(y, 0.3, 0.5, 1, 0, "val", 7, "CT") == pytest.approx(0.3)


def test_masks_are_nested_across_alpha_and_shared_across_temperature():
    patients = range(400)
    flipped = {a: {p for p in patients if av.flipped(0, "val", p, "CT", a)} for a in (0.1, 0.3)}
    assert flipped[0.1] <= flipped[0.3] and 0 < len(flipped[0.1]) < len(flipped[0.3])
    for p in patients:
        q1 = av.corrupted(1, 0.4, 0.3, 1, 0, "val", p, "CT")
        q4 = av.corrupted(1, 0.4, 0.3, 0.25, 0, "val", p, "CT")
        assert (q1 > 0.5) == (q4 > 0.5)


def test_overconfidence_sharpens_without_reordering():
    q = av.corrupted(1, 0.4, 0.1, 1, 0, "val", 3, "CT")
    sharp = av.corrupted(1, 0.4, 0.1, 0.25, 0, "val", 3, "CT")
    assert abs(sharp - 0.5) > abs(q - 0.5)
    assert 0.001 <= sharp <= 0.999


def test_learned_model_uses_base_rates_for_rare_tests_and_clips():
    rng = np.random.default_rng(0)
    items = []
    for i in range(60):
        has_ct = i % 2 == 0
        history = ("pain scan ordered radiology " if has_ct else "mild pain clinic visit ") + f"note{i % 5}"
        present = {"CT"} if has_ct else set()
        if i < 3:
            present.add("MRI")
        present.add("Complete Blood Count")
        items.append(item(i, history, present))
    model = av.fit_learned(items, TESTS, seed=0, c=1.0)
    assert model.kinds == {"Complete Blood Count": "base_rate", "CT": "logistic", "MRI": "base_rate"}
    p = model.predict(items[0]["patient_history"])
    assert set(p) == set(TESTS) and all(0.001 <= v <= 0.999 for v in p.values())
    assert p["CT"] > model.predict(items[1]["patient_history"])["CT"]


def test_c_is_chosen_once_by_cross_validated_log_loss():
    items = [item(i, ("scan " if i % 2 else "clinic ") * 3 + f"w{i % 7}", {"CT"} if i % 2 else set()) for i in range(80)]
    assert av.choose_c(items, TESTS) in (0.01, 0.1, 1, 10)


def test_quality_metrics():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.35, 0.8]
    q = av.quality(y, p, base_rate=0.5)
    assert q["auroc"] == pytest.approx(0.75)
    assert q["brier"] == pytest.approx(np.mean([(a - b) ** 2 for a, b in zip(p, y)]))
    assert q["brier_skill"] == pytest.approx(1 - q["brier"] / 0.25)
    assert 0 <= q["ece"] <= 1
