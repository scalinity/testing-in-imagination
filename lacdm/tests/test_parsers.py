"""Phase E item 16: action, diagnosis and hypothesis parsers.

Decision-agent parsing is Environment.parse_actions_and_check_validity
(environment.py:186). Hypothesis parsing is inside _generate_hypothesis
(environment.py:382-390).
"""

import pytest

from conftest import ScriptedModel, act, completion


# --- action parser ----------------------------------------------------------


def test_parses_test_action_and_lowercases(env, make_trajectory):
    trajectory = make_trajectory()
    trajectory.last_completion = completion("Test", "Complete Blood Count")
    assert env.parse_actions_and_check_validity(trajectory) == (
        True, "test", "complete blood count",
    )


def test_accepts_blank_line_between_fields(env, make_trajectory):
    trajectory = make_trajectory()
    trajectory.last_completion = "Thought: x\n\nAction: Test\n\nAction Input: Urinalysis\n"
    assert env.parse_actions_and_check_validity(trajectory) == (True, "test", "urinalysis")


def test_trailing_observation_keyword_is_not_captured(env, make_trajectory):
    # Generation stops at "Observation:" (environment.py:74-84).
    trajectory = make_trajectory()
    trajectory.last_completion = "Thought: x\nAction: Test\nAction Input: Ultrasound\nObservation:"
    assert env.parse_actions_and_check_validity(trajectory)[2] == "ultrasound"


@pytest.mark.parametrize(
    "text",
    [
        "Action: Test\nAction Input: CT\n",  # no Thought
        "Thought: x\nAction Input: CT\n",  # no Action
        "Thought: x\nAction: Test\nAction Input CT\n",  # missing colon
        "",
    ],
)
def test_malformed_completion_ends_episode_as_invalid(env, make_trajectory, tokenizer, text):
    trajectory = make_trajectory()
    trajectory.last_completion = text
    env.step(trajectory, 0, tokenizer)
    assert trajectory.completed and trajectory.invalid
    assert trajectory.diagnosis is None


def test_over_length_trajectory_is_invalid(env, make_trajectory):
    trajectory = make_trajectory()
    trajectory.token_mask = [False] * (env.max_length + 1)
    trajectory.last_completion = completion("Test", "CT")
    assert env.parse_actions_and_check_validity(trajectory) == (False, None, None)


@pytest.mark.parametrize("action_input", ["CT.", "CT ", "a CT scan", "Ultrasound (RUQ)"])
def test_action_input_is_matched_exactly_not_normalized(env, make_trajectory, tokenizer, action_input):
    # Characterization: only case is normalized. Punctuation or trailing space
    # makes a well-formed request an "Unknown test" error rather than a test.
    trajectory = act(env, make_trajectory(), tokenizer, "Test", action_input)
    assert trajectory.error
    assert not trajectory.completed
    assert trajectory.request_count == 0
    assert "Error: Unknown test." in trajectory.text


def test_valid_test_action_appends_observation(env, make_trajectory, tokenizer):
    trajectory = act(env, make_trajectory(), tokenizer, "TEST", "complete BLOOD count")
    assert not trajectory.error
    assert trajectory.provided_tests == ["complete blood count"]
    assert "Observation: synthetic Complete Blood Count result." in trajectory.text


def test_unknown_action_is_an_error_not_an_invalid_episode(env, make_trajectory, tokenizer):
    trajectory = act(env, make_trajectory(), tokenizer, "Consult", "surgery")
    assert trajectory.error
    assert not trajectory.completed
    assert "Error: Unknown action. Valid actions are: Test, Diagnosis." in trajectory.text


# --- diagnosis parser -------------------------------------------------------


def test_diagnosis_completes_episode(env, make_trajectory, tokenizer):
    trajectory = act(env, make_trajectory(), tokenizer, "Diagnosis", "Pancreatitis")
    assert trajectory.diagnosis == "pancreatitis"
    assert trajectory.completed and not trajectory.invalid
    assert not trajectory.error


@pytest.mark.parametrize("disease", ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"])
def test_every_configured_disease_is_accepted(env, make_trajectory, tokenizer, disease):
    assert act(env, make_trajectory(), tokenizer, "Diagnosis", disease).diagnosis == disease


def test_unknown_diagnosis_keeps_episode_running(env, make_trajectory, tokenizer):
    trajectory = act(env, make_trajectory(), tokenizer, "Diagnosis", "gastritis")
    assert trajectory.diagnosis is None
    assert trajectory.error and not trajectory.completed
    assert "Error: Unknown diagnosis." in trajectory.text


# --- hypothesis parser ------------------------------------------------------


@pytest.mark.parametrize(
    "output, hypothesis, confidence, suffix",
    [
        ("Hypothesis: Cholecystitis\nConfidence: 8", "cholecystitis", 0.8,
         "The current hypothesis is: cholecystitis (confidence: 8)\n"),
        ("Hypothesis: gastritis\nConfidence: 9", "unknown", 0.9,
         "The current hypothesis is: unknown\n"),
        ("Hypothesis: appendicitis", "appendicitis", None,
         "The current hypothesis is: appendicitis\n"),
        ("I think appendicitis.", "unknown", None,
         "The current hypothesis is: unknown\n"),
    ],
)
def test_hypothesis_and_confidence_parsing(env, make_trajectory, output, hypothesis, confidence, suffix):
    trajectory = make_trajectory()
    env._generate_hypothesis([trajectory], ScriptedModel(output), None, None)
    assert trajectory.hypothesis == [hypothesis]
    assert trajectory.confidence == [confidence]
    assert trajectory.text == "PROMPT" + suffix
