"""The D-5 backbone overlay composes onto upstream's configs as intended."""

import json
import re

import pytest
import yaml
from omegaconf import OmegaConf

from conftest import LACDM_DIR, compose_config

NEMOTRON = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
EXCLUDED_MODELS = re.compile(r"qwen|deepseek|\byi[-_]|01-ai", re.IGNORECASE)


@pytest.mark.parametrize("experiment", ["la_cdm", "decision_agent_only"])
def test_overlay_replaces_the_backbone_for_every_experiment(experiment):
    cfg = compose_config("+backbone=nemotron_nano_8b", f"experiment={experiment}")
    assert cfg.model.model_name == NEMOTRON
    assert cfg.environment.system_prompt == "detailed thinking off"
    assert not EXCLUDED_MODELS.search(OmegaConf.to_yaml(cfg))


def test_upstream_alone_still_defaults_to_qwen():
    # Guards the overlay's reason to exist: if upstream changes its default,
    # the D-5 note needs revisiting.
    assert compose_config().model.model_name == "Qwen/Qwen2.5-7B-Instruct"


def test_revision_placeholder_is_declared(cfg):
    assert "revision" in cfg.model


def test_overlay_changes_only_backbone_keys():
    base = OmegaConf.to_container(compose_config())
    overlaid = OmegaConf.to_container(compose_config("+backbone=nemotron_nano_8b"))
    base["model"]["model_name"] = NEMOTRON
    base["model"]["revision"] = None
    base["environment"]["system_prompt"] = "detailed thinking off"
    assert overlaid == base


@pytest.mark.parametrize("path", sorted((LACDM_DIR / "configs").rglob("*.yaml")), ids=lambda p: p.name)
def test_lacdm_configs_name_no_excluded_model(path):
    values = json.dumps(yaml.safe_load(path.read_text()))  # values only; comments may cite upstream
    assert not EXCLUDED_MODELS.search(values)
