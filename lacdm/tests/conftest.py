"""Shared fixtures for the Phase E tests.

The tests exercise the real upstream code in lacdm/upstream. No GPU, no model
weights, no MIMIC data: every patient record here is synthetic.
"""

import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

LACDM_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = LACDM_DIR / "upstream"

if not (UPSTREAM_DIR / "src" / "environment" / "environment.py").exists():
    raise RuntimeError(
        "lacdm/upstream is empty. Run: git submodule update --init lacdm/upstream"
    )

# environment.py imports trl, which pulls in the CUDA stack. Its only use there is
# chat-templating hypothesis prompts, so concatenating message contents suffices.
if "trl" not in sys.modules:
    trl = types.ModuleType("trl")
    trl.data_utils = types.ModuleType("trl.data_utils")
    trl.data_utils.maybe_apply_chat_template = lambda example, tokenizer: {
        "prompt": "".join(m["content"] for m in example["prompt"])
    }
    sys.modules["trl"] = trl
    sys.modules["trl.data_utils"] = trl.data_utils

sys.path.insert(0, str(UPSTREAM_DIR))
sys.path.insert(0, str(LACDM_DIR.parent))

from src.environment.environment import Environment, Trajectory  # noqa: E402

NOT_AVAILABLE = "not available.\n"  # upstream's marker, src/dataset/mimic_cdm_dataset.py:48


def compose_config(*overrides: str):
    """Compose upstream's defaults with the lacdm/configs overlay, as a run would."""
    with initialize_config_dir(config_dir=str(UPSTREAM_DIR / "configs"), version_base=None):
        return compose(
            "defaults",
            overrides=[f"hydra.searchpath=[file://{LACDM_DIR / 'configs'}]", *overrides],
        )


@pytest.fixture(scope="session")
def cfg():
    return compose_config("+backbone=nemotron_nano_8b")


@pytest.fixture(scope="session")
def test_names(cfg):
    """Test vocabulary in upstream's order (src/utils/setup.py:136)."""
    return list(cfg.dataset.lab_tests + cfg.dataset.imaging_tests + cfg.dataset.other_tests)


@pytest.fixture
def env(cfg, test_names):
    """Environment built the way src/utils/setup.py:prepare_environment builds it."""
    prompt = (UPSTREAM_DIR / cfg.environment.prompt_template_file).read_text()
    hypothesis_prompt = (UPSTREAM_DIR / cfg.environment.hypothesis_prompt_template_file).read_text()
    return Environment(
        prompt_template=prompt,
        max_length=cfg.environment.max_length,
        max_steps=cfg.environment.max_steps,
        disease_list=list(cfg.dataset.disease_list),
        test_list=test_names,
        generate_hypothesis=cfg.environment.generate_hypothesis,
        hypothesis_prompt_template=hypothesis_prompt,
        generate_confidence_calibration=cfg.environment.generate_confidence_calibration,
        seed=cfg.training.seed,
    )


@pytest.fixture
def make_trajectory(test_names):
    """Synthetic record: CT and MRI unavailable, every other test returns a result."""

    def _make(unavailable=("CT", "MRI")):
        results = {
            name: NOT_AVAILABLE if name in unavailable else f"synthetic {name} result.\n"
            for name in test_names
        }
        trajectory = Trajectory("PROMPT", results, "synthetic history")
        trajectory.token_mask = [False] * 10
        return trajectory

    return _make


class WordTokenizer:
    """Stands in for a HF tokenizer; step() only needs len(encode(text))."""

    def encode(self, text, **kwargs):
        return text.split()


@pytest.fixture
def tokenizer():
    return WordTokenizer()


def completion(action: str, action_input: str) -> str:
    """A decision-agent completion in the format prompts/decision_agent.txt asks for."""
    return f"Thought: synthetic reasoning\nAction: {action}\nAction Input: {action_input}\n"


def act(env, trajectory, tokenizer, action, action_input, step=0):
    trajectory.last_completion = completion(action, action_input)
    env.step(trajectory, step, tokenizer)
    return trajectory


class ScriptedModel:
    """Returns canned hypothesis-agent outputs and records the prompts it was sent."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.prompts = []

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        self.prompts.extend(prompts)
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=self.texts.pop(0))], prompt=p, prompt_token_ids=[0])
            for p in prompts
        ]
