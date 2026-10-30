"""Execute prompt, loader and episode hooks without GPU packages or downloads."""

from copy import deepcopy
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir


OWN = Path(__file__).resolve().parents[1]
ROOT = OWN.parents[1]
sys.path.insert(0, str(OWN))

from replication import gpu, shim


@pytest.fixture
def upstream(monkeypatch):
    # Config and path insertion are supplied by the coordinator on integration.
    # The source checkout also supports isolated-worktree tests with empty submodules.
    configured = sys.modules.get("replication.config")
    path = Path(configured.UPSTREAM) if configured else ROOT / "lacdm" / "upstream"
    if not (path / "src" / "environment" / "environment.py").exists():
        path = ROOT.parents[1] / "lacdm" / "upstream"
    if not (path / "src" / "environment" / "environment.py").exists():
        pytest.fail("The pinned upstream submodule is not initialized")
    monkeypatch.syspath_prepend(str(path))
    if configured is None:
        module = ModuleType("replication.config")
        module.UPSTREAM = path
        module.ROOT = ROOT
        module.stream_seed = lambda seed, split, stage: seed + {"train": 1, "val": 2, "test": 3}[split]
        monkeypatch.setitem(sys.modules, "replication.config", module)
    return path


@pytest.fixture
def cfg(upstream):
    with initialize_config_dir(config_dir=str(upstream / "configs"), version_base=None):
        value = compose(config_name="defaults")
    from omegaconf import open_dict
    with open_dict(value):
        value.model.model_name = shim.APPROVED_MODEL
        value.model.revision = "a" * 40
    return value


def patient(cfg, condition="appendicitis", history="Right lower quadrant pain with McBurney tenderness.", missing=("CT",)):
    tests = list(cfg.dataset.lab_tests) + list(cfg.dataset.imaging_tests) + list(cfg.dataset.other_tests)
    return {"patient_id": 100, "patient_history": history, "condition": condition,
            "test_results": {name: "not available.\n" if name in missing else "Observed tenderness.\n" for name in tests}}


def test_actual_prompts_have_system_in_all_three_agent_paths(cfg):
    class RecordingTokenizer(shim.MockTokenizer):
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(deepcopy(messages))
            return super().apply_chat_template(messages, **kwargs)

    class RecordingPolicy(shim.MockPolicy):
        def __init__(self):
            self.tokenizer = shim.ensure_pad_token(RecordingTokenizer())
            self.prompts = []

        def generate(self, prompts, **kwargs):
            self.prompts.extend(prompts)
            return super().generate(prompts, **kwargs)

    policy = RecordingPolicy()
    result = shim._evaluate_with_model(cfg, [patient(cfg)], policy, shim.generation_settings(False))
    assert result[0]["prediction"] == "appendicitis"
    assert len(policy.tokenizer.calls) >= 2
    assert all(messages[0] == {"role": "system", "content": "detailed thinking off"}
               for messages in policy.tokenizer.calls)
    confidence = [prompt for prompt in policy.prompts if prompt.endswith("Confidence: ")]
    assert confidence
    assert all("<|start_header_id|>system<|end_header_id|>\n\ndetailed thinking off" in prompt
               for prompt in confidence)
    # The wrapper and scoped TRL substitute never replace the tokenizer template.
    assert policy.tokenizer.calls[0][1]["role"] == "user"


def test_mock_runs_upstream_and_preserves_missing_test_cost(cfg):
    row, = shim.evaluate_dataset(cfg, [patient(cfg)], "test", 269)
    assert row["provided_tests"] == ["physical examination"]
    assert row["requested_tests"] == ["ct", "physical examination"]
    assert row["unavailable_requests"] == ["ct"]
    assert row["reward"] == pytest.approx(2 - .0294)
    assert row["hypotheses"] == ["appendicitis", "appendicitis"]
    assert row["confidences"] == [.7, .7]
    assert shim.evaluate_dataset(cfg, [patient(cfg)], "test", 269, trained=True) == [row]


def test_actual_out_of_range_hypothesis_confidence_does_not_drop_patient(cfg):
    from replication.stats import summary
    class HighConfidence(shim.MockPolicy):
        def generate(self, prompts, **kwargs):
            outputs = super().generate(prompts, **kwargs)
            for output in outputs:
                response = output.outputs[0]
                response.text = response.text.replace("Confidence: 7", "Confidence: 11")
                response.token_ids = self.tokenizer.encode(response.text) + [self.tokenizer.eos_token_id]
            return outputs
    rows = shim._evaluate_with_model(cfg, [patient(cfg)], HighConfidence(), shim.generation_settings(False))
    assert all(triplet[1] == 1.1 for triplet in rows[0]["ece_triplets"])
    measured = summary(rows, cfg)
    assert measured["n"] == 1 and measured["hypothesis"]["accuracy"] == 1
    assert measured["hypothesis"]["invalid_confidence_count"] > 0
    assert measured["hypothesis"]["ece"] is None


def test_no_label_leak_and_identical_arm_policies(cfg):
    before, = shim.evaluate_dataset(cfg, [patient(cfg)], "test", 269)
    changed, = shim.evaluate_dataset(cfg, [patient(cfg, condition="pancreatitis")], "test", 269)
    assert before["prediction"] == changed["prediction"] == "appendicitis"
    assert before["requested_tests"] == changed["requested_tests"]
    assert before["reward"] - changed["reward"] == pytest.approx(1)
    cfg.environment.generate_hypothesis = False
    cfg.environment.generate_confidence_calibration = False
    cfg.environment.prompt_template_file = "prompts/decision_agent_only.txt"
    without_ha, = shim.evaluate_dataset(cfg, [patient(cfg)], "test", 269)
    assert before["prediction"] == without_ha["prediction"]
    assert before["requested_tests"] == without_ha["requested_tests"]
    assert without_ha["hypotheses"] == without_ha["confidences"] == []


@pytest.mark.parametrize("history,prediction", [
    ("Right upper quadrant pain and Murphy sign", "cholecystitis"),
    ("Left lower quadrant pain and sigmoid tenderness", "diverticulitis"),
    ("Epigastric pain radiating to the back", "pancreatitis"),
])
def test_mock_uses_clinical_text(cfg, history, prediction):
    row, = shim.evaluate_dataset(cfg, [patient(cfg, history=history)], "val", 269)
    assert row["prediction"] == prediction


def test_reward_arms_use_actual_upstream_functions(cfg):
    item = patient(cfg, condition="pancreatitis")
    shipped, = shim.evaluate_dataset(cfg, [item], "test", 269)
    cfg.reward_function.reward_functions[0].incorrect_reward = -1
    paper, = shim.evaluate_dataset(cfg, [item], "test", 269)
    assert shipped["reward"] - paper["reward"] == pytest.approx(1)
    cfg.reward_function.reward_functions = cfg.reward_function.reward_functions[:2]
    free, = shim.evaluate_dataset(cfg, [item], "test", 269)
    assert free["reward"] == pytest.approx(0)


def test_f1_f3_f4_and_fourteen_action_budget_preserved(cfg):
    env = shim.prepare_environment(cfg)
    module = import_module("src.environment.environment")
    trajectory = module.Trajectory("{first_hypothesis}", patient(cfg)["test_results"], "Right lower quadrant pain")
    policy = shim.MockPolicy()
    prompts = env._generate_hypothesis([trajectory], policy, None, policy.tokenizer)
    assert "confidence: 7" in prompts[0]
    trajectory.last_completion = "Thought: Check imaging\nAction: Test\nAction Input: CT\n"
    env.step(trajectory, 0, policy.tokenizer)
    assert trajectory.error is True
    assert not trajectory.completed
    prompts = env._generate_hypothesis([trajectory], policy, None, policy.tokenizer)
    assert len(trajectory.hypothesis) == 1
    assert "confidence: 0.7" in prompts[0]
    assert trajectory.error is False
    for step in range(1, 15):
        env.step(trajectory, step, policy.tokenizer)
    assert trajectory.request_count == 14
    assert trajectory.completed and trajectory.invalid
    assert trajectory.provided_tests == []


@pytest.fixture
def fake_loaders(monkeypatch):
    calls = []

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("model", name, kwargs))
            return SimpleNamespace(
                config=SimpleNamespace(_name_or_path=name), name_or_path=name,
                enable_input_require_grads=lambda: None, gradient_checkpointing_enable=lambda: None)

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return shim.MockTokenizer()

    class LLM:
        def __init__(self, *args, **kwargs):
            calls.append(("llm", args, kwargs))
            self.tokenizer = shim.MockTokenizer()

        def get_tokenizer(self):
            return self.tokenizer

        def generate(self, prompts, sampling_params=None, **kwargs):
            calls.append(("generate", prompts, {"sampling_params": sampling_params, **kwargs}))
            policy = shim.MockPolicy()
            policy.tokenizer = self.tokenizer
            return policy.generate(prompts, sampling_params=sampling_params)

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = Model
    transformers.AutoTokenizer = Tokenizer
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    vllm = ModuleType("vllm")
    vllm.LLM = LLM
    trainer = ModuleType("src.trainer.cdm_grpo_trainer")
    trainer.LLM = LLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "src.trainer.cdm_grpo_trainer", trainer)
    return transformers, vllm, trainer, calls


@pytest.fixture
def upstream_setup(cfg, fake_loaders, monkeypatch):
    shim._environment_module()
    transformers, _, _, _ = fake_loaders
    torch = ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.cuda = SimpleNamespace(device_count=lambda: 1)
    utils = ModuleType("torch.utils")
    data = ModuleType("torch.utils.data")
    data.Dataset = type("Dataset", (), {})
    torch.utils = utils
    utils.data = data
    for name, module in (("torch", torch), ("torch.utils", utils), ("torch.utils.data", data)):
        monkeypatch.setitem(sys.modules, name, module)
    wandb = ModuleType("wandb")
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    transformers.PreTrainedModel = type("PreTrainedModel", (), {})
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    peft.get_peft_model = lambda model, config: model
    peft.prepare_model_for_kbit_training = lambda model: model
    peft.PeftModel = SimpleNamespace(from_pretrained=lambda model, path: model)
    monkeypatch.setitem(sys.modules, "peft", peft)
    accelerate = ModuleType("accelerate")
    accelerate.PartialState = lambda: SimpleNamespace(process_index=0)
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    trl = ModuleType("trl")
    trl.GRPOConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "trl", trl)
    module = import_module("src.utils.setup")
    try:
        yield module
    finally:
        sys.modules.pop("src.utils.setup", None)


def test_actual_upstream_policy_loader_uses_revision(cfg, upstream_setup, fake_loaders):
    _, _, _, calls = fake_loaders
    with shim.pinned_loaders(cfg.model.model_name, cfg.model.revision, token="fixture-token"):
        model = upstream_setup.load_model(cfg)
    kind, name, kwargs = calls[0]
    assert kind == "model" and name == shim.APPROVED_MODEL
    assert kwargs["revision"] == "a" * 40
    assert kwargs["token"] == "fixture-token"
    assert model.config._name_or_path == shim.APPROVED_MODEL


def test_gpu_evaluation_uses_actual_singleton_episodes_and_greedy_sampler(
        cfg, upstream_setup, fake_loaders, monkeypatch, tmp_path):
    import json

    _, vllm, _, calls = fake_loaders
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    request_module = ModuleType("vllm.lora.request")
    request_module.LoRARequest = lambda *args: args
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_module)
    monkeypatch.setattr(gpu, "_cuda_runtime", lambda: None)
    seen_splits = []
    def dataset(value, split):
        seen_splits.append(split)
        first = patient(cfg)
        second = patient(cfg, condition="pancreatitis", history="Epigastric pain and lipase elevation")
        second["patient_id"] = 101
        return [first, second]
    monkeypatch.setattr(upstream_setup, "prepare_dataset", dataset)
    fake_loaders[0].set_seed = lambda seed: None
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": shim.APPROVED_MODEL, "revision": "a" * 40}))
    rows = gpu.evaluate(cfg, tmp_path, "test", 269, adapter_dir=adapter)
    assert [row["prediction"] for row in rows] == ["appendicitis", "pancreatitis"]
    assert seen_splits == ["test"]
    generation_calls = [kwargs for kind, _, kwargs in calls if kind == "generate"]
    assert generation_calls
    assert all(kwargs["sampling_params"].temperature == 0 for kwargs in generation_calls)
    assert all(kwargs["lora_request"][2] == str(adapter) for kwargs in generation_calls)
    assert all(len(prompts) <= 1 for kind, prompts, _ in calls if kind == "generate")


def test_train_uses_adam_fixed_budget_train_val_only_and_resume(
        cfg, upstream_setup, fake_loaders, monkeypatch, tmp_path):
    transformers, vllm, upstream_trainer, _ = fake_loaders
    events = {}
    optimizer = SimpleNamespace(param_groups=[{}])
    torch = SimpleNamespace(optim=SimpleNamespace(Adam=lambda params, **kwargs:
                           events.update(optimizer="Adam", optimizer_kwargs=kwargs) or optimizer))
    monkeypatch.setattr(gpu, "_cuda_runtime", lambda: torch)
    weights = SimpleNamespace(
        peft_config={"default": SimpleNamespace(revision=None)}, config=SimpleNamespace(),
        parameters=lambda: [SimpleNamespace(requires_grad=True)])
    monkeypatch.setattr(upstream_setup, "load_model", lambda cfg: weights)
    monkeypatch.setattr(upstream_setup, "prepare_dataset", lambda cfg, split:
                        events.setdefault("splits", []).append(split) or [patient(cfg)])
    monkeypatch.setattr(upstream_setup, "setup_wandb", lambda cfg: None)
    transformers.set_seed = lambda seed: events.update(seed=seed)
    trainer_utils = ModuleType("transformers.trainer_utils")
    trainer_utils.get_last_checkpoint = lambda path: str(tmp_path / "checkpoint-3")
    import json
    import random
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    (checkpoint / "environment-rng.json").write_text(json.dumps({"source": "mimic", "rng_state": random.Random(269).getstate()}))
    monkeypatch.setitem(sys.modules, "transformers.trainer_utils", trainer_utils)
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    upstream_trainer.CDMGRPOTrainer = object
    wandb = sys.modules["wandb"]
    wandb.run = SimpleNamespace(id="fixture-run-id")
    wandb.finish = lambda **kwargs: events.update(finish=kwargs)

    class Trainer:
        def __init__(self, **kwargs):
            events["trainer_kwargs"] = kwargs
            self.state = SimpleNamespace(log_history=[])

        def train(self, **kwargs):
            events["resume"] = kwargs
            assert self.sampling_params.temperature == 1

        def evaluate(self):
            return {"eval_all/accuracy": .5}

        def save_model(self, path):
            events["adapter_dir"] = path

    monkeypatch.setattr(gpu, "_trainer_class", lambda *args: Trainer)
    tokenizer = shim.MockTokenizer()
    tokenizer.save_pretrained = lambda path: events.update(tokenizer_dir=path)
    transformers.AutoTokenizer.from_pretrained = lambda *args, **kwargs: shim.ensure_pad_token(tokenizer)
    # Fake constructors stop before CUDA; the actual setup.configure_grpo is used.
    result = gpu.train(cfg, tmp_path / "training", tmp_path / "data", 269)
    assert events["optimizer"] == "Adam"
    assert events["optimizer_kwargs"] == {"lr": 1e-5}
    assert events["splits"] == ["train", "val"]
    assert events["resume"] == {"resume_from_checkpoint": str(tmp_path / "checkpoint-3")}
    args = events["trainer_kwargs"]["args"]
    assert args.num_generations == 8 and args.per_device_train_batch_size == 2
    assert events["trainer_kwargs"]["optimizers"] == (optimizer, None)
    assert weights.peft_config["default"].revision == "a" * 40
    assert result["selection"]["method"] == "fixed_budget"
    assert result["wandb_run_ids"] == ["fixture-run-id"]
    assert events["finish"] == {"exit_code": 0}


def test_every_loader_path_is_pinned_and_padding_is_set(fake_loaders):
    transformers, vllm, trainer, calls = fake_loaders
    model = shim.APPROVED_MODEL
    original = vllm.LLM
    with shim.pinned_loaders(model, "a" * 40, token="fixture-token"):
        # policy, ZeRO reference, tokenizer and the existing trainer alias
        for _ in range(2):
            transformers.AutoModelForCausalLM.from_pretrained(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        engine = trainer.LLM(model=model)
        assert tokenizer.pad_token_id == tokenizer.eos_token_id
        assert engine.get_tokenizer().padding_side == "left"
        with pytest.raises(ValueError, match="Unapproved"):
            transformers.AutoModelForCausalLM.from_pretrained("wrong/model")
        with pytest.raises(ValueError, match="Conflicting"):
            vllm.LLM(model=model, revision="b" * 40)
        with pytest.raises(ValueError, match="Unapproved"):
            vllm.LLM(model=model, tokenizer="wrong/tokenizer")
    assert trainer.LLM is vllm.LLM is original
    for kind, _, kwargs in calls:
        assert kwargs["revision"] == "a" * 40
        assert kwargs["hf_token" if kind == "llm" else "token"] == "fixture-token"
        if kind == "llm":
            assert kwargs["tokenizer_revision"] == "a" * 40


def test_trainer_imported_during_loader_scope_does_not_retain_stale_revision(fake_loaders, monkeypatch):
    _, vllm, _, calls = fake_loaders
    original = vllm.LLM
    monkeypatch.delitem(sys.modules, "src.trainer.cdm_grpo_trainer")
    trainer = ModuleType("src.trainer.cdm_grpo_trainer")
    with shim.pinned_loaders(shim.APPROVED_MODEL, "a" * 40):
        trainer.LLM = vllm.LLM
        monkeypatch.setitem(sys.modules, "src.trainer.cdm_grpo_trainer", trainer)
    assert trainer.LLM is original
    with shim.pinned_loaders(shim.APPROVED_MODEL, "b" * 40):
        trainer.LLM(model=shim.APPROVED_MODEL)
    assert calls[-1][2]["revision"] == "b" * 40


def test_actual_summarizer_factory_uses_loader_pins(cfg, fake_loaders, monkeypatch):
    transformers, _, _, calls = fake_loaders
    module = import_module("scripts.create_data_with_summaries") if "scripts.create_data_with_summaries" in sys.modules else None
    # Import the actual upstream function with its optional GPU imports stubbed.
    torch = ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    tqdm = ModuleType("tqdm")
    tqdm.tqdm = lambda value, **kwargs: value
    monkeypatch.setitem(sys.modules, "tqdm", tqdm)
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    original = transformers.AutoModelForCausalLM.from_pretrained
    transformers.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: SimpleNamespace(eval=lambda: None)
    # Keep fake loader recording while providing the model's eval method.
    def load(*args, **kwargs):
        original(*args, **kwargs)
        return SimpleNamespace(eval=lambda: None)
    transformers.AutoModelForCausalLM.from_pretrained = load
    monkeypatch.setattr(gpu, "_cuda_runtime", lambda: torch)
    gpu._summarizer.cache_clear()
    try:
        _, tokenizer, instruction = gpu._summarizer(shim.APPROVED_MODEL, "a" * 40)
        assert "medical expert" in instruction
        assert tokenizer.pad_token_id == tokenizer.eos_token_id
        assert {kind for kind, _, _ in calls} == {"model", "tokenizer"}
        assert all(kwargs["revision"] == "a" * 40 for _, _, kwargs in calls)
    finally:
        gpu._summarizer.cache_clear()
        if module is None:
            sys.modules.pop("scripts.create_data_with_summaries", None)


def test_padding_preserves_existing_token_and_rejects_missing_eos():
    tokenizer = shim.MockTokenizer()
    tokenizer.pad_token_id = 123
    tokenizer.pad_token = "existing"
    shim.ensure_pad_token(tokenizer)
    assert tokenizer.pad_token_id == 123
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="EOS"):
        shim.ensure_pad_token(tokenizer)


def test_validation_is_greedy_training_sampled_and_context_restores():
    original = SimpleNamespace(temperature=1.0, top_p=.8, n=1, stop=["Observation:"])
    trainer = SimpleNamespace(sampling_params=original)
    with gpu.generation_mode(trainer, False):
        assert trainer.sampling_params.temperature == 0
        assert trainer.sampling_params.stop == original.stop
    assert trainer.sampling_params is original
    assert shim.generation_settings(True)["temperature"] == 1


def test_actual_trainer_wrapper_switches_generation_mode(cfg):
    class Base:
        def _run_environment(self):
            return self.sampling_params.temperature

    trainer = gpu._trainer_class(Base, cfg, 269)()
    trainer.sampling_params = SimpleNamespace(temperature=1, top_p=1, n=1)
    trainer.model = SimpleNamespace(training=False)
    assert trainer._run_environment() == 0
    trainer.model.training = True
    assert trainer._run_environment() == 1
    assert trainer.sampling_params.temperature == 1


def test_schedule_has_full_hundred_step_phases_and_resume(cfg):
    phases = cfg.training.loss_schedule.phases
    assert gpu._phase_weights(phases, 99)[0]["da_loss_weight"] == 1
    assert gpu._phase_weights(phases, 100)[0]["ha_sft_loss_weight"] == 1
    assert gpu._phase_weights(phases, 199)[0]["ha_sft_loss_weight"] == 1
    assert gpu._phase_weights(phases, 200)[0]["ha_conf_cal_loss_weight"] == 1
    assert gpu._phase_weights(phases, 300)[0]["da_loss_weight"] == 1


def test_ha_examples_keep_their_independent_count(cfg):
    class Base:
        def _prepare_inputs(self, inputs):
            return {"prompt_ids": SimpleNamespace(shape=(8, 10)),
                    "ha_prompt_label_ids": SimpleNamespace(shape=(3, 10))}

    trainer = gpu._trainer_class(Base, cfg, 269)()
    trainer.args = SimpleNamespace(num_generations=8)
    trainer.environment = SimpleNamespace(generate_hypothesis=True)
    assert trainer._prepare_inputs(None)["ha_prompt_label_ids"].shape[0] == 3


def test_wrong_models_and_mutable_revisions_rejected():
    with pytest.raises(ValueError, match="Unapproved"):
        shim.validate_model("wrong/model", "a" * 40)
    for revision in (None, "main", "a" * 39):
        with pytest.raises(ValueError, match="40-character"):
            shim.validate_model(shim.APPROVED_MODEL, revision)


def test_summary_uses_split_rng_explicit_model_and_system_template(cfg, monkeypatch):
    seen = {"seeds": [], "messages": [], "generate": []}

    class Tokens:
        shape = (1, 5)

        def to(self, device):
            return self

    tokenizer = SimpleNamespace(
        pad_token_id=0,
        apply_chat_template=lambda messages, **kwargs: seen["messages"].append(messages) or Tokens(),
        decode=lambda tokens, **kwargs: "brief observed history")
    class Output:
        def __getitem__(self, key):
            return "summary tokens"
    weights = SimpleNamespace(device="fake-device", generate=lambda tokens, **kwargs:
                              seen["generate"].append(kwargs) or Output())
    monkeypatch.setattr(gpu, "_summarizer", lambda model, revision: (weights, tokenizer, "Summarize clinical symptoms."))
    torch = ModuleType("torch")
    torch.inference_mode = nullcontext
    transformers = ModuleType("transformers")
    transformers.set_seed = seen["seeds"].append
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    for split in ("val", "test"):
        assert gpu.summarize("observed history", split, shim.APPROVED_MODEL, "a" * 40) == "brief observed history"
    assert seen["seeds"][0] != seen["seeds"][1]
    assert all(messages[0]["content"].startswith("detailed thinking off\n") for messages in seen["messages"])
    assert all(kwargs["do_sample"] is True for kwargs in seen["generate"])


def test_policy_weight_transfer_failure_is_not_suppressed(cfg):
    def fail(**kwargs):
        raise ValueError("weight shape mismatch")

    engine_model = SimpleNamespace(load_weights=fail)
    class Base:
        def _move_model_to_vllm(self):
            try:
                engine_model.load_weights(weights=[("weight", None)])
            except Exception:
                pass

    trainer = gpu._trainer_class(Base, cfg, 269)()
    trainer.llm = SimpleNamespace(llm_engine=SimpleNamespace(model_executor=SimpleNamespace(
        driver_worker=SimpleNamespace(model_runner=SimpleNamespace(model=engine_model)))))
    with pytest.raises(RuntimeError, match="weight shape mismatch"):
        trainer._move_model_to_vllm()
    assert engine_model.load_weights is fail
