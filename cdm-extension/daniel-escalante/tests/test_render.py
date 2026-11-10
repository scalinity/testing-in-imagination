import json

from tim import render
from tim.paths import OWN, REPLICATION

METRICS = json.loads((OWN / "results" / "synthetic" / "phase-a" / "metrics.json").read_text())


def with_family(v1_sig, v3_v1_sig, primary_sig):
    m = json.loads(json.dumps(METRICS))
    word = {True: "statistically significant", False: "not statistically significant"}
    m["family_a"]["V1-official"]["significance"] = word[v1_sig]
    m["family_a"]["V3-V1"]["significance"] = word[v3_v1_sig]
    m["primary"]["significance"] = word[primary_sig]
    return render.render(m, "disclosure")


def test_the_negative_result_plan_sentence_appears_when_its_case_holds():
    text = with_family(v1_sig=True, v3_v1_sig=False, primary_sig=True)
    assert "the hypothesis agent already held the missing answers, and the online guards added nothing measurable" in text
    assert "the hypothesis agent already held" not in with_family(True, True, True)


def test_a_null_primary_names_the_components():
    text = with_family(v1_sig=False, v3_v1_sig=False, primary_sig=False)
    assert "V3 − official is not statistically significant" in text


def test_amendments_are_listed_as_protocol_notes():
    text = with_family(True, False, True)
    assert "prereg/amendments/A1-code-fixes.md" in text and "prereg/amendments/A2-reference-scoring.md" in text


METRICS_B = json.loads((OWN / "results" / "synthetic" / "phase-b" / "metrics.json").read_text())
PAPER = json.loads((REPLICATION / "results" / "paper_values.json").read_text())


def credentialed(metrics):
    m = json.loads(json.dumps(metrics))
    m["source"] = "mimic"
    return m


def test_the_writeup_and_slide_render_their_numbers_from_metrics():
    a = credentialed(METRICS)
    a["variants"]["V3"]["metrics"]["T5"] = 12.3456
    a["primary"]["delta_t5"] = 4.5678
    for text in (render.render_writeup(a, credentialed(METRICS_B), PAPER, "disclosure"),
                 render.render_slide(a, credentialed(METRICS_B), PAPER)):
        assert "12.35" in text and "+4.57" in text and "81.30" in text


def test_the_writeup_and_slide_say_they_are_not_a_reproduction():
    a, b = credentialed(METRICS), credentialed(METRICS_B)
    for text in (render.render_writeup(a, b, PAPER, "disclosure"), render.render_slide(a, b, PAPER)):
        assert "not a reproduction of the paper" in text
        assert not text.startswith("> **SYNTHETIC**")


def test_a_null_primary_is_never_worded_as_an_improvement():
    a = credentialed(METRICS)
    a["primary"]["significance"] = "not statistically significant"
    writeup = render.render_writeup(a, credentialed(METRICS_B), PAPER, "disclosure")
    assert "the difference is not statistically significant" in writeup and "raises" not in writeup
    a["primary"]["significance"] = "statistically significant"
    a["primary"]["delta_t5"] = 3.0
    assert "raises" in render.render_writeup(a, credentialed(METRICS_B), PAPER, "disclosure")


def test_synthetic_metrics_carry_the_banner_on_both_outputs():
    assert render.render_writeup(METRICS, METRICS_B, PAPER, "disclosure").startswith("> **SYNTHETIC**")
    assert render.render_slide(METRICS, METRICS_B, PAPER).startswith("> **SYNTHETIC**")
