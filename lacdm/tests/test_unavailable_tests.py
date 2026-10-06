"""Phase E item 18: the unavailable-test branch, executed against upstream code.

Turns REPRODUCTION.md §6b findings F-1, F-2 and F-3 (read from source) into
executed checks. Line numbers refer to lacdm/upstream at the pinned SHA.
"""

import re

import pytest
from hydra.utils import instantiate

from conftest import ScriptedModel, act


@pytest.fixture
def reward_funcs(cfg):
    return [instantiate(rf) for rf in cfg.reward_function.reward_functions]


def rewards_for(reward_funcs, env, trajectory, condition="appendicitis"):
    (kwargs,) = env._collect_reward_function_kwargs([trajectory])
    batched = {k: [v] for k, v in kwargs.items()}
    return [f(prompts=[None], completions=[None], condition=[condition], **batched)[0] for f in reward_funcs]


# --- F-1: one error flag for four different situations ----------------------


@pytest.mark.parametrize(
    "action, action_input, situation",
    [
        ("Consult", "surgery", "unknown action (line 325)"),
        ("Test", "PET scan", "unknown test (line 330)"),
        ("Test", "CT", "test unavailable for this record (line 336)"),
        ("Diagnosis", "gastritis", "unknown diagnosis (line 341)"),
    ],
)
def test_f1_every_situation_sets_the_same_error_flag(env, make_trajectory, tokenizer, action, action_input, situation):
    trajectory = act(env, make_trajectory(), tokenizer, action, action_input)
    assert trajectory.error is True, situation
    assert not trajectory.completed and not trajectory.invalid


def test_f1_reward_functions_cannot_tell_model_error_from_missing_data(env, make_trajectory, tokenizer, reward_funcs):
    malformed = act(env, make_trajectory(), tokenizer, "Test", "PET scan")
    missing = act(env, make_trajectory(), tokenizer, "Test", "CT")

    (malformed_kwargs,) = env._collect_reward_function_kwargs([malformed])
    (missing_kwargs,) = env._collect_reward_function_kwargs([missing])
    # The only trace of the difference is request_count, which no reward function reads.
    assert malformed_kwargs["request_count"] == 0 and missing_kwargs["request_count"] == 1
    malformed_kwargs.pop("request_count"), missing_kwargs.pop("request_count")
    assert malformed_kwargs == missing_kwargs
    assert rewards_for(reward_funcs, env, malformed) == rewards_for(reward_funcs, env, missing)


# --- F-2: unavailable tests are free ----------------------------------------


def test_f2_unavailable_test_is_requested_but_not_provided(env, make_trajectory, tokenizer):
    trajectory = act(env, make_trajectory(), tokenizer, "Test", "CT")
    assert trajectory.requested_tests == ["ct"]
    assert trajectory.provided_tests == []
    assert trajectory.request_count == 1
    assert "Observation: not available." in trajectory.text


def test_f2_unavailable_test_costs_nothing(env, make_trajectory, tokenizer, reward_funcs):
    cost = reward_funcs[2]
    unavailable = act(env, make_trajectory(unavailable=("CT",)), tokenizer, "Test", "CT")
    available = act(env, make_trajectory(unavailable=()), tokenizer, "Test", "CT")
    assert rewards_for([cost], env, unavailable) == [0]
    assert rewards_for([cost], env, available) == [pytest.approx(-0.1282)]


def test_f2_repeated_requests_are_charged_once(env, make_trajectory, tokenizer, reward_funcs):
    trajectory = make_trajectory()
    for step in range(3):
        act(env, trajectory, tokenizer, "Test", "Ultrasound", step=step)
    assert trajectory.request_count == 3
    assert trajectory.provided_tests == ["ultrasound"]
    assert rewards_for([reward_funcs[2]], env, trajectory) == [pytest.approx(-0.1264)]


# --- F-3: the only penalty is a stale hypothesis and a spent step -----------


def test_f3_error_turn_skips_hypothesis_generation(env, make_trajectory, tokenizer):
    stale, fresh = make_trajectory(), make_trajectory()
    model = ScriptedModel(
        "Hypothesis: appendicitis\nConfidence: 7",
        "Hypothesis: appendicitis\nConfidence: 7",
        "Hypothesis: cholecystitis\nConfidence: 6",
    )
    env._generate_hypothesis([stale, fresh], model, None, None)
    act(env, stale, tokenizer, "Test", "CT")  # unavailable
    act(env, fresh, tokenizer, "Test", "Urinalysis")  # available

    env._generate_hypothesis([stale, fresh], model, None, None)

    assert len(model.prompts) == 3  # 2 on turn one, then only the fresh trajectory
    assert stale.error is False  # flag is reset (environment.py:377)
    assert stale.hypothesis == ["appendicitis"]  # not re-generated, not re-appended
    assert fresh.hypothesis == ["appendicitis", "cholecystitis"]


def test_f3_stale_turn_shows_confidence_on_a_different_scale(env, make_trajectory, tokenizer):
    # Characterization, not in the ledger before this test: a fresh turn shows the
    # raw 0-10 integer, but the stale path reuses the stored value, already divided
    # by 10 (environment.py:376 vs 391). The decision prompt describes a 0-10 scale.
    trajectory = make_trajectory()
    model = ScriptedModel("Hypothesis: appendicitis\nConfidence: 7")
    env._generate_hypothesis([trajectory], model, None, None)
    act(env, trajectory, tokenizer, "Test", "CT")  # unavailable
    env._generate_hypothesis([trajectory], model, None, None)

    shown = re.findall(r"The current hypothesis is: .*", trajectory.text)
    assert shown == [
        "The current hypothesis is: appendicitis (confidence: 7)",
        "The current hypothesis is: appendicitis (confidence: 0.7)",
    ]


def test_f3_unavailable_requests_burn_the_step_budget_and_nothing_else(env, make_trajectory, tokenizer, reward_funcs):
    trajectory = make_trajectory()
    step = 0
    while not trajectory.completed:
        act(env, trajectory, tokenizer, "Test", "MRI", step=step)  # unavailable every time
        step += 1

    # step() rejects only when step > max_steps (environment.py:311), so
    # max_steps = 13 allows 14 actions; the 15th call ends the episode.
    assert env.max_steps == 13
    assert trajectory.request_count == env.max_steps + 1
    assert trajectory.invalid and trajectory.diagnosis is None
    assert trajectory.provided_tests == []
    assert rewards_for(reward_funcs, env, trajectory) == [0.0, 0.0, 0]
