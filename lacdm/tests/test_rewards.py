"""Phase E item 17: reward math, instantiated from config the way train.py does it.

The trainer's episode reward is the unweighted sum of the reward functions:
no config sets reward_weights, so they default to ones (cdm_grpo_trainer.py:183,620).
"""

import inspect
import math

import pytest
from hydra.utils import instantiate

from conftest import compose_config
from src.reward_function.diagnosis_reward import DiagnosisReward

# Paper §4: R_diag gives a fixed negative r_neg for a wrong diagnosis. The class
# default (diagnosis_reward.py:13) agrees; the shipped config overrides it (D-1).
PAPER_INCORRECT_REWARD = -1.0
SHIPPED_INCORRECT_REWARD = 0.0


@pytest.fixture
def reward_funcs(cfg):
    return [instantiate(rf) for rf in cfg.reward_function.reward_functions]


def episode_reward(reward_funcs, diagnosis, invalid, provided_tests, condition="appendicitis"):
    kwargs = dict(
        prompts=[None], completions=[None], condition=[condition],
        diagnosis=[diagnosis], invalid=[invalid], provided_tests=[provided_tests],
    )
    return sum(f(**kwargs)[0] for f in reward_funcs)


# --- D-1: shipped 0.0 vs paper -1.0 -----------------------------------------


def test_shipped_config_uses_zero_not_the_papers_negative_reward(cfg):
    shipped = cfg.reward_function.reward_functions[0]
    assert shipped._target_.endswith("DiagnosisReward")
    assert shipped.incorrect_reward == SHIPPED_INCORRECT_REWARD
    assert shipped.incorrect_reward != PAPER_INCORRECT_REWARD


def test_class_default_matches_the_paper_and_config_overrides_it():
    default = inspect.signature(DiagnosisReward.__init__).parameters["incorrect_reward"].default
    assert default == PAPER_INCORRECT_REWARD


def test_backbone_overlay_does_not_touch_rewards():
    # The overlay must not silently "fix" D-1: rewards compose identically with or without it.
    assert compose_config().reward_function == compose_config("+backbone=nemotron_nano_8b").reward_function


def test_no_test_cost_ablation_also_ships_zero():
    cfg = compose_config("+backbone=nemotron_nano_8b", "reward_function=no_test_cost")
    assert cfg.reward_function.reward_functions[0].incorrect_reward == SHIPPED_INCORRECT_REWARD


def test_shipped_diagnosis_reward_cannot_tell_wrong_from_undiagnosed(reward_funcs):
    diagnosis = reward_funcs[0]
    rewards = diagnosis([None] * 3, [None] * 3, ["appendicitis"] * 3, ["appendicitis", "pancreatitis", None])
    assert rewards == [1.0, 0.0, 0.0]


def test_paper_diagnosis_reward_separates_wrong_from_undiagnosed():
    paper = DiagnosisReward(correct_reward=1.0, incorrect_reward=PAPER_INCORRECT_REWARD, undiagnosed_reward=0.0)
    assert paper([None] * 2, [None] * 2, ["appendicitis"] * 2, ["pancreatitis", None]) == [-1.0, 0.0]


@pytest.mark.parametrize(
    "incorrect_reward, correct, wrong, undiagnosed",
    [(SHIPPED_INCORRECT_REWARD, 2.0, 1.0, 0.0), (PAPER_INCORRECT_REWARD, 2.0, 0.0, 0.0)],
)
def test_episode_totals_under_each_setting(cfg, incorrect_reward, correct, wrong, undiagnosed):
    # An episode can only end undiagnosed by being invalid (timeout or malformed
    # output), which also zeroes FormatReward. So under the paper's value a wrong
    # guess TIES with running out the clock; under the shipped value it beats it.
    funcs = [instantiate(rf) for rf in cfg.reward_function.reward_functions]
    funcs[0].incorrect_reward = incorrect_reward
    assert episode_reward(funcs, "appendicitis", False, []) == correct
    assert episode_reward(funcs, "pancreatitis", False, []) == wrong
    assert episode_reward(funcs, None, True, []) == undiagnosed


def test_diagnosis_match_is_case_insensitive(reward_funcs):
    assert reward_funcs[0]([None], [None], ["Appendicitis"], ["APPENDICITIS"]) == [1.0]


# --- format and cost --------------------------------------------------------


def test_format_reward(reward_funcs):
    assert reward_funcs[1]([None] * 2, [None] * 2, invalid=[False, True]) == [1.0, 0.0]


def test_cost_table_covers_exactly_the_environment_vocabulary(cfg, test_names):
    # TestCostReward indexes the table by the lowercased test name; a missing key
    # would raise KeyError mid-training (test_cost_reward.py:25).
    cost_keys = set(cfg.reward_function.reward_functions[2].test_cost)
    assert cost_keys == {name.lower() for name in test_names}
    assert set(cfg.evaluation.test_costs_usd) == cost_keys


def test_costs_sum_to_minus_the_correct_reward(cfg):
    # Paper's design: an all-tests pathway to a correct diagnosis nets zero.
    total = sum(cfg.reward_function.reward_functions[2].test_cost.values())
    assert total == pytest.approx(-cfg.reward_function.reward_functions[0].correct_reward, abs=1e-9)


def test_all_tests_correct_diagnosis_nets_only_the_format_reward(reward_funcs, cfg):
    every_test = list(cfg.reward_function.reward_functions[2].test_cost)
    assert episode_reward(reward_funcs, "appendicitis", False, every_test) == pytest.approx(1.0)


def test_cost_reward_sums_per_test_costs(reward_funcs):
    rewards = reward_funcs[2]([None], [None], provided_tests=[["ct", "urinalysis"]])
    assert rewards == [pytest.approx(-0.1282 - 0.0049)]


def test_physical_examination_usd_matches_ledger_derivation(cfg):
    # D-2 derived $300 by arithmetic; the upstream eval config states the same.
    assert cfg.evaluation.test_costs_usd["physical examination"] == 300


# --- confidence calibration -------------------------------------------------


@pytest.fixture
def conf_cal(cfg):
    return instantiate(cfg.reward_function.conf_cal_reward_func)


def test_shipped_conf_cal_settings(conf_cal):
    assert (conf_cal.oof_reward, conf_cal.reward_scale, conf_cal.correctness_bonus) == (-1.0, 1.0, 0.0)


@pytest.mark.parametrize(
    "confidence, correct, expected",
    [
        ("10", True, 1.0),
        ("0", True, -1.0),  # log(0) floored at log(0.001), the bottom of the range
        ("10", False, -1.0),
        ("0", False, 1.0),
        # Hand-derived: -1 + 2*(ln p - ln .001)/(0 - ln .001) = 1 + 2 ln p / ln 1000
        ("5", True, 1 + 2 * math.log(0.5) / math.log(1000)),
        ("8", False, 1 + 2 * math.log(0.2) / math.log(1000)),
    ],
)
def test_log_score_is_rescaled_to_minus_one_one(conf_cal, confidence, correct, expected):
    label = "appendicitis"
    hypothesis = label if correct else "pancreatitis"
    assert conf_cal([confidence], hypothesis, label) == [pytest.approx(expected)]


@pytest.mark.parametrize("confidence", ["11", "-1", "7.5", " 7", "seven", ""])
def test_out_of_format_confidence_gets_oof_reward(conf_cal, confidence):
    assert conf_cal([confidence], "appendicitis", "appendicitis") == [-1.0]
