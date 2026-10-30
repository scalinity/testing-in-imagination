"""CUDA entry points using pinned upstream setup and trainer implementations."""

from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from importlib import import_module
import os
import json
from pathlib import Path
import sys
from unittest.mock import patch

from replication.shim import (
    SYSTEM_MESSAGE, _evaluate_with_model, _upstream_path, ensure_pad_token, generation_settings,
    pinned_loaders, prepare_environment, validate_model,
)


def _cuda_runtime():
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("CUDA execution requires Python 3.11 or 3.12")
    os.environ["VLLM_USE_V1"] = "0"
    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real training and evaluation")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Pinned trainer gathers prompts but not patient records: use one training process")
    return torch


def _external(path):
    from replication.config import ROOT

    path = Path(path).resolve()
    if path.is_relative_to(Path(ROOT).resolve()):
        raise ValueError("Patient data and model artifacts must remain outside the repository")
    return path


def _configuration(cfg, data_dir, seed, output_dir=None):
    from omegaconf import OmegaConf, open_dict

    validate_model(cfg.model.model_name, cfg.model.revision)
    _upstream_path()
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_dir = _external(data_dir)
    with open_dict(result):
        result.training.seed = seed
        for split in ("train", "val", "test"):
            result.dataset[f"{split}_data_file"] = str(data_dir / f"{split}.csv")
        result.dataset.lab_test_mapping_file = str(data_dir / "lab_test_mapping.csv")
        if output_dir is not None:
            result.training.output_dir = str(_external(output_dir))
    return result


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _sampling(vllm, cfg, training, seed):
    return vllm.SamplingParams(
        **generation_settings(training), seed=seed,
        max_tokens=cfg.training.max_completion_length,
        stop=["Observation:"], include_stop_str_in_output=False,
    )


@contextmanager
def generation_mode(trainer, training):
    """Use greedy sampling on every validation call, restoring sampled training."""
    original = trainer.sampling_params
    params = deepcopy(original)
    for name, value in generation_settings(training).items():
        setattr(params, name, value)
    trainer.sampling_params = params
    try:
        yield
    finally:
        trainer.sampling_params = original


def _phase_weights(phases, step, repeat=True):
    total = sum(int(phase["steps"]) for phase in phases)
    if total <= 0 or any(int(phase["steps"]) <= 0 for phase in phases):
        raise ValueError("Loss schedule phases must have positive lengths")
    position = step % total if repeat else min(step, total - 1)
    for phase in phases:
        if position < int(phase["steps"]):
            return {
                "da_loss_weight": float(phase["name"] == "da"),
                "ha_sft_loss_weight": float(phase["name"] == "ha_sft"),
                "ha_conf_cal_loss_weight": float(phase["name"] == "ha_conf_cal"),
            }, float(phase["learning_rate"])
        position -= int(phase["steps"])


class _WeightTransferFailure(BaseException):
    """Escape the upstream per-weight Exception suppression, then re-raise normally."""


def _trainer_class(base, cfg, seed):
    class Trainer(base):
        def _run_environment(self, *args, **kwargs):
            with generation_mode(self, self.model.training):
                return super()._run_environment(*args, **kwargs)

        def _prepare_inputs(self, inputs):
            prepared = super()._prepare_inputs(inputs)
            group = self.args.num_generations
            if prepared["prompt_ids"].shape[0] != group:
                raise RuntimeError("Pinned trainer requires exactly one patient's eight-rollout group")
            # Upstream slices all objectives using the DA group size, although HA
            # prompts are deduplicated independently. Reject lost/empty HA slices.
            if self.environment.generate_hypothesis:
                count = prepared["ha_prompt_label_ids"].shape[0]
                if count != group:
                    raise RuntimeError(
                        f"Pinned HA minibatch integration is blocked: {count} deduplicated "
                        f"HA examples for {group} DA rollouts. A separate HA batching design "
                        "must be validated before training; no examples were silently dropped.")
            return prepared

        def compute_loss(self, model, inputs, *args, **kwargs):
            # Set phase once by optimizer step. Validation never consumes a phase,
            # and resuming restores the cycle from Trainer's saved global_step.
            schedule = self.loss_scheduler
            if schedule is not None:
                phases = [vars(phase) for phase in schedule.phases]
                weights, rate = _phase_weights(phases, self.state.global_step, schedule.repeat)
                for name, value in weights.items():
                    setattr(self, name, value)
                if self.optimizer is not None:
                    for group in self.optimizer.param_groups:
                        group["lr"] = rate
            self.loss_scheduler = None
            try:
                return super().compute_loss(model, inputs, *args, **kwargs)
            finally:
                self.loss_scheduler = schedule

        def _move_model_to_vllm(self):
            engine_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            original = engine_model.load_weights

            def checked(*args, **kwargs):
                try:
                    return original(*args, **kwargs)
                except Exception as exc:
                    raise _WeightTransferFailure(str(exc)) from exc

            try:
                with patch.object(engine_model, "load_weights", checked):
                    return super()._move_model_to_vllm()
            except _WeightTransferFailure as exc:
                raise RuntimeError(f"Policy-to-vLLM weight transfer failed: {exc}") from exc

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            # Running full episodes singly avoids the upstream inputs[0] label bug
            # and does not calculate GRPO advantages from greedy duplicates.
            from replication.config import stream_seed
            from src.utils.metrics import calculate_classification_metrics

            dataset = self.eval_dataset if eval_dataset is None else eval_dataset
            if isinstance(dataset, dict):
                dataset = dataset["val"]
            previous_mode = self.model.training
            self.model.eval()
            try:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
                vllm = import_module("vllm")
                val_seed = stream_seed(seed, "val", "evaluation")
                val_cfg = deepcopy(cfg)
                val_cfg.training.seed = val_seed
                rows = _evaluate_with_model(val_cfg, dataset, self.llm, _sampling(vllm, cfg, False, val_seed))
                self.eval_preds = [row["prediction"] for row in rows]
                self.eval_labels = [row["label"] for row in rows]
                self.eval_provided_tests = [row["provided_tests"] for row in rows]
                self.eval_ece_triplets = [(hypothesis, confidence, row["label"])
                                         for row in rows for hypothesis, confidence in
                                         zip(row["hypotheses"], row["confidences"])]
                values = calculate_classification_metrics(
                    self.eval_preds, self.eval_labels, self.eval_ece_triplets)
                metrics = {f"{metric_key_prefix}_{name}": value for name, value in values.items()}
                self.log(metrics)
                self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
                return metrics
            finally:
                self.model.train(previous_mode)

    return Trainer


def train(cfg, output_dir, data_dir, seed):
    """Train on train only; select the final checkpoint of a fixed epoch budget."""
    torch = _cuda_runtime()
    cfg = _configuration(cfg, data_dir, seed, output_dir)
    output_dir = _external(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(output_dir)
    with pinned_loaders(cfg.model.model_name, cfg.model.revision, _token()):
        setup = import_module("src.utils.setup")
        upstream = import_module("src.trainer.cdm_grpo_trainer")
        transformers = import_module("transformers")
        wandb = import_module("wandb")
        transformers.set_seed(seed)
        if cfg.training.num_generations != 8 or cfg.training.per_device_train_batch_size != 2:
            raise ValueError("Parity requires eight rollouts and batch-two forward passes")
        if cfg.training.load_best_model_at_end:
            raise ValueError("This launcher selects the final fixed-budget checkpoint")
        setup.setup_wandb(cfg)
        success = False
        try:
            train_data = setup.prepare_dataset(cfg, "train")
            val_data = setup.prepare_dataset(cfg, "val")
            model = setup.load_model(cfg)
            for adapter in model.peft_config.values():
                adapter.revision = cfg.model.revision
            tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.model_name)
            model.config.pad_token_id = tokenizer.pad_token_id
            args = setup.configure_grpo(cfg)
            # One training process/device; an additional GPU can serve vLLM.
            args._n_gpu = 1
            args.remove_unused_columns = False
            args.vllm_max_model_len = cfg.environment.max_length + cfg.training.max_completion_length
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                         lr=cfg.training.learning_rate)
            trainer_type = _trainer_class(upstream.CDMGRPOTrainer, cfg, seed)
            trainer = trainer_type(
                model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
                processing_class=tokenizer, environment=prepare_environment(cfg),
                reward_funcs=setup.prepare_reward_functions(cfg),
                conf_cal_reward_func=setup.prepare_conf_cal_reward_func(cfg),
                loss_type=cfg.training.loss_type, loss_schedule=cfg.training.loss_schedule,
                test_costs_usd=dict(cfg.evaluation.test_costs_usd), optimizers=(optimizer, None),
            )
            vllm = import_module("vllm")
            trainer.sampling_params = _sampling(vllm, cfg, True, seed)
            trainer._last_loaded_step = -1
            checkpoint = import_module("transformers.trainer_utils").get_last_checkpoint(str(output_dir))
            trainer.train(resume_from_checkpoint=checkpoint)
            val_metrics = trainer.evaluate()
            adapter_dir = output_dir / "adapter"
            trainer.save_model(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            result = {
                "adapter_dir": str(adapter_dir), "optimizer": "Adam",
                "selection": {"method": "fixed_budget", "epochs": cfg.training.num_train_epochs,
                              "checkpoint": "final", "validation_metrics": val_metrics},
                "resume_from_checkpoint": checkpoint,
                "wandb_ids": [wandb.run.id] if wandb.run is not None else [],
            }
            success = True
            return result
        finally:
            wandb.finish(exit_code=0 if success else 1)


def evaluate(cfg, data_dir, split, seed, adapter_dir=None):
    """One greedy singleton episode per patient using the locked final policy."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    _cuda_runtime()
    from replication.config import stream_seed

    cfg = _configuration(cfg, data_dir, stream_seed(seed, split, "evaluation"))
    if adapter_dir is not None:
        adapter_dir = _external(adapter_dir)
        metadata = json.loads((adapter_dir / "adapter_config.json").read_text())
        validate_model(metadata.get("base_model_name_or_path"), metadata.get("revision"))
        if metadata["revision"] != cfg.model.revision:
            raise ValueError("Adapter and evaluation backbone revisions differ")
    with pinned_loaders(cfg.model.model_name, cfg.model.revision, _token()):
        setup = import_module("src.utils.setup")
        transformers = import_module("transformers")
        vllm = import_module("vllm")
        transformers.set_seed(cfg.training.seed)
        dataset = setup.prepare_dataset(cfg, split)
        # vLLM LoRA uses the same pinned base without reloading a second HF model.
        engine = vllm.LLM(
            model=cfg.model.model_name, enable_lora=adapter_dir is not None,
            max_lora_rank=max(8, int(cfg.model.lora_r)),
            gpu_memory_utilization=cfg.training.vllm_gpu_memory_utilization,
            max_model_len=cfg.environment.max_length + cfg.training.max_completion_length, seed=cfg.training.seed,
        )
        if adapter_dir is not None:
            request = import_module("vllm.lora.request").LoRARequest("policy", 1, str(adapter_dir))

            class AdaptedPolicy:
                def get_tokenizer(self):
                    return engine.get_tokenizer()

                def generate(self, *args, **kwargs):
                    return engine.generate(*args, **kwargs, lora_request=request)

            policy = AdaptedPolicy()
        else:
            policy = engine
        return _evaluate_with_model(cfg, dataset, policy, _sampling(vllm, cfg, False, cfg.training.seed))


@lru_cache(maxsize=1)
def _summarizer(model, revision):
    _cuda_runtime()
    _upstream_path()
    with pinned_loaders(model, revision, _token()):
        upstream = import_module("scripts.create_data_with_summaries")
        weights, tokenizer = upstream.load_summarizer(model)
    return weights, ensure_pad_token(tokenizer), upstream.SYSTEM_PROMPT


def summarize(text, split, model, revision, seed=269):
    """Summarize with an explicit pinned model and a split/text-specific RNG."""
    validate_model(model, revision)
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    from replication.config import stream_seed

    weights, tokenizer, instruction = _summarizer(model, revision)
    torch = import_module("torch")
    transformers = import_module("transformers")
    transformers.set_seed(stream_seed(seed, split, "summary:" + sha256(text.encode()).hexdigest()))
    messages = [{"role": "system", "content": SYSTEM_MESSAGE + "\n" + instruction},
                {"role": "user", "content": text}]
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(weights.device)
    with torch.inference_mode():
        output = weights.generate(tokens, max_new_tokens=512, do_sample=True, temperature=1.0,
                                  top_p=1.0, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0, tokens.shape[-1]:], skip_special_tokens=True)
