import json
from pathlib import Path
import tomllib

from omegaconf import OmegaConf
import pytest

from replication.config import (ARMS, MODEL, OWN, REVISION, ROOT, Settings,
                                UPSTREAM, check_resolved_models, compose,
                                external_path, read_env, stream_seed, upstream_defaults)


@pytest.mark.parametrize("arm", (*ARMS, "zero_shot"))
def test_all_composed_backbones_and_rewards(arm):
    cfg = compose(arm)
    assert cfg.model.model_name == MODEL
    assert len(cfg.model.revision) == 40 and cfg.model.revision == REVISION
    assert cfg.environment.system_prompt == "detailed thinking off"
    assert cfg.reward_function.reward_functions[0].incorrect_reward == (-1 if arm == "incorrect_minus1" else 0)
    assert cfg.training.learning_rate == 1e-5
    assert cfg.training.per_device_train_batch_size == 2
    assert cfg.training.seed == 269
    assert cfg.training.num_generations == 8
    assert cfg.training.optimizer == "adam"
    assert cfg.training.log_completions is False
    assert cfg.wandb.mode == "disabled"
    assert cfg.training.report_to == "none"
    check_resolved_models(cfg)


def test_primary_reward_and_schedule_match_shipped_behavior():
    original, cfg = upstream_defaults(), compose()
    assert original.reward_function == cfg.reward_function
    assert original.training.loss_schedule == cfg.training.loss_schedule
    assert [phase.steps for phase in cfg.training.loss_schedule.phases] == [100, 100, 100]
    assert compose("decision_only").environment.generate_hypothesis is False
    assert len(compose("no_cost").reward_function.reward_functions) == 2


@pytest.mark.parametrize("model", ["Qwen/Qwen2.5-7B-Instruct", "DeepSeek/test", "01-ai/Yi-9B", "internlm/test", "release/rebranded-derivative"])
def test_banned_and_unapproved_names_rejected(model):
    for args in (("--model", model), (f"--model={model}",)):
        with pytest.raises(ValueError):
            check_resolved_models(compose(), cli_args=args)
    with pytest.raises(ValueError):
        check_resolved_models({"model": {"model_name": model}})


def test_rng_streams_are_split_and_stage_specific():
    assert len({stream_seed(269, split, stage) for split in ("train", "val", "test") for stage in ("data", "policy", "bootstrap")}) == 9
    assert stream_seed(269, "test", "policy") == stream_seed(269, "test", "policy")


def test_runtime_preserves_exact_dependency_requirements():
    original = tomllib.loads((UPSTREAM / "pyproject.toml").read_text())
    local = tomllib.loads((OWN / "runtime/pyproject.toml").read_text())
    assert local["project"]["dependencies"] == original["project"]["dependencies"]
    assert local["tool"]["uv"]["sources"]["xgrammar"]["rev"] == "989222175c2a30fb7987d8bcce35bec1bf6817f2"
    assert (OWN / "runtime/uv.lock").exists()


def test_private_settings_and_literal_env(tmp_path, monkeypatch):
    for key in tuple(__import__("os").environ):
        if key.startswith("CDM_"):
            monkeypatch.delenv(key)
    env = tmp_path / ".env"
    env.write_text(f'CDM_DATA_SOURCE=synthetic\nCDM_WORK_DIR="{tmp_path / "work dir"}"\n')
    settings = Settings.load(env)
    assert settings.mock and settings.work_dir == tmp_path / "work dir"
    env.write_text('CDM_DATA_SOURCE=mimic\n')
    with pytest.raises(ValueError, match="CDM_DATA_DIR"):
        Settings.load(env)
    with pytest.raises(ValueError):
        external_path(ROOT / "artifacts")
    env.write_text('CDM_WORK_DIR="$(touch should-never-exist)"\n')
    assert read_env(env)["CDM_WORK_DIR"] == "$(touch should-never-exist)"
