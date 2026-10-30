"""Local launcher hooks around the pinned clinical environment."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from importlib import import_module, util
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from zlib import crc32


SYSTEM_MESSAGE = "detailed thinking off"
APPROVED_MODEL = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"


def validate_model(model, revision):
    if model != APPROVED_MODEL:
        raise ValueError(f"Unapproved model: {model}")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Model revision must be an immutable 40-character SHA")


def ensure_pad_token(tokenizer):
    """Reuse EOS for padding without changing the model vocabulary."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor a usable EOS token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def generation_settings(training: bool) -> dict:
    """vLLM parameters; one completion per each of eight repeated training inputs."""
    return {"temperature": 1.0 if training else 0.0, "top_p": 1.0, "n": 1}


@contextmanager
def pinned_loaders(model, revision, token=None):
    """Pin policy, reference, tokenizer, summarizer and vLLM loader paths."""
    validate_model(model, revision)
    transformers = import_module("transformers")
    vllm = import_module("vllm")
    original_llm = vllm.LLM

    def loader(original, tokenizer=False):
        def load(name=None, *args, **kwargs):
            name = kwargs.pop("pretrained_model_name_or_path", name)
            validate_model(name, revision)
            if kwargs.get("revision", revision) != revision:
                raise ValueError("Conflicting model revision")
            kwargs["revision"] = revision
            kwargs["token"] = token
            result = original(name, *args, **kwargs)
            return ensure_pad_token(result) if tokenizer else result
        return load

    def llm(*args, **kwargs):
        name = args[0] if args else kwargs.get("model")
        validate_model(name, revision)
        if kwargs.get("revision", revision) != revision:
            raise ValueError("Conflicting vLLM revision")
        tokenizer_name = kwargs.get("tokenizer", model)
        validate_model(tokenizer_name, revision)
        if kwargs.get("tokenizer_revision", revision) != revision:
            raise ValueError("Conflicting tokenizer revision")
        kwargs.update(revision=revision, tokenizer_revision=revision, hf_token=token)
        result = original_llm(*args, **kwargs)
        ensure_pad_token(result.get_tokenizer())
        return result

    # Class-method patches also cover aliases imported before this context.
    try:
        with ExitStack() as stack:
            for cls, is_tokenizer in ((transformers.AutoModelForCausalLM, False),
                                      (transformers.AutoTokenizer, True)):
                stack.enter_context(patch.object(cls, "from_pretrained", loader(cls.from_pretrained, is_tokenizer)))
            stack.enter_context(patch.object(vllm, "LLM", llm))
            module = sys.modules.get("src.trainer.cdm_grpo_trainer")
            if module is not None and hasattr(module, "LLM"):
                stack.enter_context(patch.object(module, "LLM", llm))
            yield
    finally:
        # A trainer first imported inside the context captures the patched alias.
        module = sys.modules.get("src.trainer.cdm_grpo_trainer")
        if module is not None and getattr(module, "LLM", None) is llm:
            module.LLM = original_llm


def _with_system(messages):
    messages = [dict(message) for message in messages]
    if messages and messages[0]["role"] == "system":
        if SYSTEM_MESSAGE not in messages[0]["content"]:
            messages[0]["content"] = SYSTEM_MESSAGE + "\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": SYSTEM_MESSAGE})
    return messages


class _SystemTokenizer:
    """Scope the hypothesis hook to a call, without mutating the tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, messages, *args, **kwargs):
        return self.tokenizer.apply_chat_template(_with_system(messages), *args, **kwargs)


def _environment_module():
    # Only the chat helper is needed on CPU. Preserve actual role-aware templating.
    if "trl" in sys.modules or util.find_spec("trl") is not None:
        return import_module("src.environment.environment")
    trl = ModuleType("trl")
    data_utils = ModuleType("trl.data_utils")

    def maybe_apply_chat_template(example, tokenizer):
        result = dict(example)
        result["prompt"] = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True)
        return result

    data_utils.maybe_apply_chat_template = maybe_apply_chat_template
    trl.data_utils = data_utils
    # Restore only these two optional imports, retaining the loaded environment.
    sys.modules["trl"] = trl
    sys.modules["trl.data_utils"] = data_utils
    try:
        return import_module("src.environment.environment")
    finally:
        sys.modules.pop("trl.data_utils", None)
        sys.modules.pop("trl", None)


def _upstream_path():
    from replication.config import UPSTREAM

    path = Path(UPSTREAM)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def prepare_environment(cfg):
    """Import upstream mechanics unchanged and instrument terminal trajectories."""
    upstream = _upstream_path()
    base = _environment_module().Environment

    class Environment(base):
        def construct_prompt(self, inputs):
            result = super().construct_prompt(inputs)
            for item in result:
                item["prompt"] = _with_system(item["prompt"])
            return result

        def _generate_hypothesis(self, trajectories, model, sampling_params, processing_class):
            return super()._generate_hypothesis(
                trajectories, model, sampling_params, _SystemTokenizer(processing_class))

        def _collect_reward_function_kwargs(self, trajectories):
            self.captured = [{
                "provided_tests": list(t.provided_tests),
                "requested_tests": list(t.requested_tests),
                "unavailable_requests": [name for name in t.requested_tests
                                         if t.test_results[name] == "not available.\n"],
            } for t in trajectories]
            return super()._collect_reward_function_kwargs(trajectories)

    def read_template(value):
        path = Path(value)
        return (path if path.is_absolute() else upstream / path).read_text()

    return Environment(
        prompt_template=read_template(cfg.environment.prompt_template_file),
        max_length=cfg.environment.max_length,
        max_steps=cfg.environment.max_steps,
        disease_list=list(cfg.dataset.disease_list),
        test_list=list(cfg.dataset.lab_tests) + list(cfg.dataset.imaging_tests) + list(cfg.dataset.other_tests),
        generate_hypothesis=cfg.environment.generate_hypothesis,
        hypothesis_prompt_template=(read_template(cfg.environment.hypothesis_prompt_template_file)
                                    if cfg.environment.generate_hypothesis else None),
        generate_confidence_calibration=cfg.environment.generate_confidence_calibration,
        seed=cfg.training.seed,
    )


class MockTokenizer:
    """Reversible CPU tokens with explicit chat roles, for environment mechanics."""

    eos_token = "<|eot_id|>"
    eos_token_id = 0
    pad_token = None
    pad_token_id = None

    def encode(self, text, **kwargs):
        return [crc32(word.encode()) + 1 for word in re.findall(r"\S+|\n", text)]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text) for text in texts]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        text = "<|begin_of_text|>" + "".join(
            f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
            for m in messages)
        if add_generation_prompt:
            text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return self.encode(text) if tokenize else text


class MockPolicy:
    """Observation-only scripted policy; training flags do not simulate learning."""

    def __init__(self):
        self.tokenizer = ensure_pad_token(MockTokenizer())

    def get_tokenizer(self):
        return self.tokenizer

    def _observed(self, prompt):
        for marker in ("Here is the available information on the patient:",
                       "Here is some context on the patient:"):
            if marker in prompt:
                return prompt.split(marker, 1)[1]
        raise ValueError("Mock policy requires an upstream patient prompt")

    def _diagnosis(self, observed):
        # Remove the hypothesis hint: each arm uses the same observable clinical cues.
        observed = re.sub(r"The current hypothesis is: [^\n]*", "", observed.lower())
        signals = {
            "appendicitis": ("appendicitis", "right lower", "rlq", "mcburney"),
            "cholecystitis": ("cholecystitis", "right upper", "ruq", "murphy"),
            "diverticulitis": ("diverticulitis", "left lower", "llq", "sigmoid"),
            "pancreatitis": ("pancreatitis", "epigastric", "lipase", "radiating to the back"),
        }
        return max(signals, key=lambda disease: sum(observed.count(word) for word in signals[disease]))

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        outputs = []
        for prompt in prompts:
            observed = self._observed(prompt)
            diagnosis = self._diagnosis(observed)
            if prompt.endswith("Confidence: "):
                text = "7"
            elif "Here is the available information on the patient:" in prompt:
                text = f"Hypothesis: {diagnosis}\nConfidence: 7"
            else:
                requested = re.findall(r"Action: Test\nAction Input: ([^\n]+)", observed)
                next_test = next((name for name in ("CT", "Physical Examination") if name not in requested), None)
                action, value = ("Test", next_test) if next_test else ("Diagnosis", diagnosis)
                text = f"Thought: Assess the available clinical findings.\nAction: {action}\nAction Input: {value}\n"
            outputs.append(SimpleNamespace(
                prompt=prompt, prompt_token_ids=self.tokenizer.encode(prompt),
                outputs=[SimpleNamespace(text=text, token_ids=self.tokenizer.encode(text) + [self.tokenizer.eos_token_id])]))
        return outputs


def _evaluate_with_model(cfg, dataset, model, sampling_params):
    from hydra.utils import instantiate

    environment = prepare_environment(cfg)
    tokenizer = ensure_pad_token(model.get_tokenizer())
    rewards = [instantiate(item) for item in cfg.reward_function.reward_functions]
    rows = []
    # Singleton episodes avoid upstream's inputs[0]['condition'] batch-label bug.
    for index in range(len(dataset)):
        item = dict(dataset[index])
        inputs = environment.construct_prompt([item])
        prompts = [tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)]
        output = environment.run(prompts, inputs, model, sampling_params, tokenizer)
        kwargs = output.reward_function_kwargs[0]
        reward_kwargs = {key: [value] for key, value in kwargs.items()}
        reward = sum(function(prompts=prompts, completions=[None],
                              condition=[item["condition"]], **reward_kwargs)[0] for function in rewards)
        rows.append({
            "patient_id": int(item["patient_id"]), "label": item["condition"],
            "prediction": kwargs["diagnosis"], **environment.captured[0], "reward": float(reward),
            "hypotheses": output.ha_hypotheses_lists[0], "confidences": output.ha_confidences_lists[0],
        })
    return rows


def evaluate_dataset(cfg, dataset, split, seed, trained=False):
    """Execute actual upstream episodes on CPU; return rows without writing data."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    from replication.config import stream_seed

    cfg = deepcopy(cfg)
    cfg.training.seed = stream_seed(seed, split, "evaluation")
    return _evaluate_with_model(cfg, dataset, MockPolicy(), generation_settings(False))
