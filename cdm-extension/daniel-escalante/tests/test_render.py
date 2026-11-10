import json

from tim import render
from tim.paths import OWN

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
