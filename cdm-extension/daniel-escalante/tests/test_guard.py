import pytest

from tim.actions import Action, Vocabulary
from tim.fallback import finalize
from tim.guard import FIXED_THOUGHT, Limits, fit, trigger

TESTS = ["Complete Blood Count", "Basic Metabolic Panel", "CT", "Physical Examination"]
DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
VOCAB = Vocabulary(TESTS, DISEASES)
LIMITS = Limits()


def completion(action, value, thought="look again"):
    return f"Thought: {thought}\nAction: {action}\nAction Input: {value}\n"


def tokenizer():
    from replication.shim import MockTokenizer, ensure_pad_token
    return ensure_pad_token(MockTokenizer())


@pytest.mark.parametrize("text, requested, kind, name", [
    (completion("Test", "CT"), [], "test", "CT"),
    (completion("test", "ct"), [], "test", "CT"),
    (completion("Test", "Brain MRI"), [], "unknown_test", None),
    (completion("Test", "CT"), ["ct"], "repeated_test", None),
    (completion("Consult", "Surgery"), [], "unknown_action", None),
    (completion("Diagnosis", "gastritis"), [], "unknown_diagnosis", None),
    (completion("Diagnosis", "Appendicitis"), [], "diagnosis", "appendicitis"),
    ("Thought: unsure\nno action here\n", [], "malformed", None),
])
def test_proposals_are_classified_with_upstreams_parser(text, requested, kind, name):
    proposal = VOCAB.classify(text, requested)
    assert proposal.kind == kind
    assert (proposal.action.name if proposal.action else None) == name


def test_untouched_turn_has_no_trigger():
    proposal = VOCAB.classify(completion("Test", "CT"), [])
    assert trigger(3, 500, 20, proposal, [], VOCAB, g1=True, g2=True) is None


def test_step_13_commits_a_test_proposal_to_a_diagnosis():
    proposal = VOCAB.classify(completion("Test", "CT"), [])
    found = trigger(13, 500, 20, proposal, [], VOCAB, g1=True, g2=True)
    assert found.event == "g2_commit" and found.from_kind == "test"
    assert [a.name for a in found.candidates] == DISEASES


def test_step_13_keeps_a_valid_diagnosis():
    proposal = VOCAB.classify(completion("Diagnosis", "pancreatitis"), [])
    assert trigger(13, 500, 20, proposal, [], VOCAB, g1=True, g2=True) is None


def test_completion_overflow_commits_even_a_valid_diagnosis():
    proposal = VOCAB.classify(completion("Diagnosis", "pancreatitis"), [])
    found = trigger(2, 4090, 20, proposal, [], VOCAB, g1=True, g2=True)
    assert found.event == "g2_commit" and found.from_kind == "overflow"


def test_beyond_the_step_limit_upstream_terminates():
    proposal = VOCAB.classify(completion("Test", "CT"), [])
    assert trigger(14, 500, 20, proposal, [], VOCAB, g1=True, g2=True) is None


@pytest.mark.parametrize("text, requested, kind, expected", [
    ("Thought: unsure\n", [], "malformed", TESTS + DISEASES),
    (completion("Consult", "Surgery"), [], "unknown_action", TESTS + DISEASES),
    (completion("Test", "Brain MRI"), ["ct"], "unknown_test", ["Complete Blood Count", "Basic Metabolic Panel", "Physical Examination"] + DISEASES),
    (completion("Test", "CT"), ["ct"], "repeated_test", ["Complete Blood Count", "Basic Metabolic Panel", "Physical Examination"] + DISEASES),
    (completion("Diagnosis", "gastritis"), [], "unknown_diagnosis", DISEASES),
    (completion("Test", "CT"), [t.lower() for t in TESTS], "repeated_test", DISEASES),
])
def test_validity_guard_candidates(text, requested, kind, expected):
    proposal = VOCAB.classify(text, requested)
    found = trigger(4, 500, 20, proposal, requested, VOCAB, g1=True, g2=True)
    assert found.event == "g1_replace" and found.from_kind == kind
    assert [a.name for a in found.candidates] == expected


def test_guards_off_never_trigger():
    proposal = VOCAB.classify("Thought: unsure\n", [])
    assert trigger(13, 4090, 20, proposal, [], VOCAB, g1=False, g2=False) is None


def test_g2_takes_precedence_over_g1():
    proposal = VOCAB.classify("Thought: unsure\n", [])
    assert trigger(13, 500, 20, proposal, [], VOCAB, g1=True, g2=True).event == "g2_commit"


def test_replacement_keeps_the_proposals_thought_when_it_fits():
    text, ids = fit(Action("test", "CT"), "check the scan", "g1_replace", 100, tokenizer(), LIMITS)
    assert text == "Thought: check the scan\nAction: Test\nAction Input: CT\n"
    assert ids == tokenizer().encode(text)


def test_replacement_falls_back_to_the_fixed_thought():
    long_thought = " ".join(["word"] * 300)
    text, ids = fit(Action("diagnosis", "appendicitis"), long_thought, "g2_commit", 100, tokenizer(), LIMITS)
    assert text == f"Thought: {FIXED_THOUGHT['g2_commit']}\nAction: Diagnosis\nAction Input: appendicitis"
    assert ids[-1] == tokenizer().eos_token_id


def test_replacement_failed_when_nothing_fits():
    assert fit(Action("diagnosis", "appendicitis"), None, "g2_commit", 4093, tokenizer(), LIMITS) is None


def test_continuations_are_canonical():
    assert Action("test", "CT").continuation("<|eot_id|>") == " Test\nAction Input: CT\nObservation:"
    assert Action("diagnosis", "pancreatitis").continuation("<|eot_id|>") == " Diagnosis\nAction Input: pancreatitis<|eot_id|>"


@pytest.mark.parametrize("raw, hypotheses, g3, expected", [
    ("appendicitis", ["pancreatitis"], True, ("appendicitis", "diagnosis")),
    (None, ["pancreatitis", "unknown", "cholecystitis", "unknown"], True, ("cholecystitis", "hypothesis_fallback")),
    (None, ["unknown"], True, (None, "none")),
    (None, [], True, (None, "none")),
    (None, ["pancreatitis"], False, (None, "none")),
])
def test_hypothesis_fallback(raw, hypotheses, g3, expected):
    assert finalize(raw, hypotheses, g3) == expected
