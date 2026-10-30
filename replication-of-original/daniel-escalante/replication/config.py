"""The only reader of data-source settings and the upstream configuration overlay."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shlex
import tempfile

from hydra import compose as hydra_compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

OWN = Path(__file__).resolve().parents[1]
ROOT = OWN.parents[1]
UPSTREAM = ROOT / "lacdm/upstream"
UPSTREAM_SHA = "3f435a10f569c3255172c1ef8fe7880c6edb7fee"
MODEL = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
REVISION = "54641c1611fcff44fa4865626462445e0a153fc7"
CLASSES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
ARMS = ("primary", "decision_only", "no_cost", "incorrect_minus1")
LABEL = "re-implementation on a different backbone"


def stream_seed(seed: int, split: str, stage: str) -> int:
    """Stable independent streams; Python's salted hash must never seed a run."""
    value = f"{seed}\0{split}\0{stage}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def external_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("data and work directories must be outside the repository")
    return path


def read_env(path: Path) -> dict[str, str]:
    """Read literal KEY=value settings, without shell execution or interpolation."""
    settings = {}
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.removeprefix("export ").partition("=")
            if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key.strip()):
                raise ValueError(f"invalid .env setting at line {number}")
            parsed = shlex.split(value, comments=True)
            if len(parsed) > 1:
                raise ValueError(f"quote spaces in .env line {number}")
            settings[key.strip()] = parsed[0] if parsed else ""
    return settings


@dataclass(frozen=True)
class Settings:
    source: str
    data_dir: Path
    work_dir: Path
    seed: int = 269
    seeds: tuple[int, ...] = (269,)
    synthetic_n: int = 2400
    bootstrap_resamples: int = 10000

    @property
    def mock(self) -> bool:
        return self.source == "synthetic"

    @classmethod
    def load(cls, env_file: Path | None = None):
        values = read_env(env_file or OWN / ".env") | dict(os.environ)
        source = values.get("CDM_DATA_SOURCE", "synthetic")
        if source not in ("synthetic", "mimic"):
            raise ValueError("CDM_DATA_SOURCE must be synthetic or mimic")
        work = external_path(values.get("CDM_WORK_DIR", str(Path(tempfile.gettempdir()) / "testing-in-imagination-replication")))
        data_value = values.get("CDM_DATA_DIR")
        if source == "mimic" and not data_value:
            raise ValueError("mimic requires CDM_DATA_DIR in the untracked .env")
        data = external_path(data_value or work / "raw-synthetic")
        if data == work or data in work.parents:
            raise ValueError("CDM_WORK_DIR cannot be inside the input data directory")
        seeds = tuple(int(s) for s in values.get("CDM_TRAINING_SEEDS", "269").split(","))
        if not seeds or len(set(seeds)) != len(seeds) or seeds[0] != 269:
            raise ValueError("training seeds must be distinct, with primary seed 269 first")
        n = int(values.get("CDM_SYNTHETIC_N", "2400"))
        if n < 120:
            raise ValueError("at least 120 synthetic admissions are needed for the full split and statistics pipeline")
        return cls(source=source, data_dir=data, work_dir=work, seeds=seeds, synthetic_n=n)


def assert_model(model: str) -> None:
    # An exact allowlist also rejects derivatives with uninformative release names.
    if model != MODEL:
        raise ValueError("only the approved replication backbone is permitted")


def check_resolved_models(config, cli_args=()) -> None:
    config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
    banned = re.compile(r"qwen|deepseek|internlm|(?:^|[/_\-\s])yi(?:$|[/_\-\s\d])", re.I)

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("model_name", "model", "tokenizer_name") and isinstance(item, str):
                    assert_model(item)
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, str) and banned.search(value):
            raise ValueError("banned backbone family in resolved configuration or CLI")

    walk(config)
    walk(list(cli_args))
    for i, arg in enumerate(cli_args):
        if arg == "--model" and i + 1 < len(cli_args):
            assert_model(cli_args[i + 1])
        elif arg.startswith("--model="):
            assert_model(arg.split("=", 1)[1])


def compose(arm="primary", data_dir=None, output_dir=None, seed=269):
    if arm not in (*ARMS, "zero_shot"):
        raise ValueError(f"unknown arm: {arm}")
    overrides = [f"hydra.searchpath=[file://{ROOT / 'lacdm/configs'}]", "+backbone=nemotron_nano_8b"]
    if arm == "decision_only":
        overrides.extend(["environment=decision_agent_only", "experiment=decision_agent_only"])
    if arm == "no_cost":
        overrides.append("reward_function=no_test_cost")
    with initialize_config_dir(config_dir=str(UPSTREAM / "configs"), version_base=None):
        cfg = hydra_compose("defaults", overrides=overrides)
    with open_dict(cfg):
        cfg.model.revision = REVISION
        cfg.training.seed = seed
        cfg.training.log_completions = False
        cfg.training.report_to = "none"
        cfg.wandb.mode = "disabled"
        cfg.wandb.save_code = False
        cfg.training.optimizer = "adam"
        cfg.training.temperature = 1.0
        cfg.evaluation.temperature = 0.0
        for key in ("prompt_template_file", "hypothesis_prompt_template_file"):
            cfg.environment[key] = str(UPSTREAM / cfg.environment[key])
        if arm == "incorrect_minus1":
            cfg.reward_function.reward_functions[0].incorrect_reward = -1.0
        if data_dir is not None:
            for split in ("train", "val", "test"):
                cfg.dataset[f"{split}_data_file"] = str(Path(data_dir) / f"{split}.csv")
            cfg.dataset.lab_test_mapping_file = str(Path(data_dir) / "lab_test_mapping.csv")
        if output_dir is not None:
            cfg.training.output_dir = str(output_dir)
            cfg.evaluation.metrics_output_file = str(Path(output_dir) / "metrics.json")
    check_resolved_models(cfg)
    return cfg


def upstream_defaults():
    with initialize_config_dir(config_dir=str(UPSTREAM / "configs"), version_base=None):
        return hydra_compose("defaults")


def overrides_against_defaults(cfg) -> list[dict]:
    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, path))
            else:
                result[path] = item
        return result
    before = flatten(OmegaConf.to_container(upstream_defaults(), resolve=True))
    after = flatten(OmegaConf.to_container(cfg, resolve=True))
    return [{"key": key, "upstream": before.get(key), "resolved": value}
            for key, value in after.items() if before.get(key) != value]
