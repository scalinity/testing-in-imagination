"""Compare actual accumulated gradients with a whole-group objective on CPU."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from replication.config import compose
from replication.gpu import _trainer_class


@pytest.mark.parametrize("count", [1, 3, 8, 11])
@pytest.mark.parametrize("phase", [0, 100, 200])
def test_independent_objective_batches_match_reference_gradient(count, phase):
    cfg = compose()
    seen = {"da": [], "ha": [], "conf": []}
    targets_da = torch.arange(1, 9, dtype=torch.float64)
    targets_ha = torch.arange(1, count + 1, dtype=torch.float64)
    # Unequal label lengths exercise the token weighting across HA microbatches.
    label_mask = torch.tensor([[1.] * (1 + i % 3) + [0.] * (2 - i % 3) for i in range(count)], dtype=torch.float64)
    prepared = {"prompt_ids": targets_da[:, None], "ha_conf_cal_prompt_ids": targets_da[:, None],
                "ha_prompt_label_ids": targets_ha[:, None], "ha_label_mask": label_mask,
                "ha_attention_mask": label_mask}

    class Base:
        def _prepare_inputs(self, inputs):
            return prepared

        def compute_loss_context_manager(self):
            return nullcontext()

        def _compute_loss_da(self, model, inputs, last):
            target = inputs["prompt_ids"].flatten()
            seen["da"].extend(target.tolist())
            assert len(target) <= 2
            return ((model - target) ** 2).mean()

        def _compute_loss_ha(self, model, inputs, last):
            target = inputs["ha_prompt_label_ids"].flatten()
            weight = inputs["ha_label_mask"].sum(dim=1)
            seen["ha"].extend(target.tolist())
            assert len(target) <= 2
            return (((model - target) ** 2) * weight).sum() / weight.sum()

        def _compute_loss_ha_conf_cal(self, model, inputs, last):
            target = inputs["ha_conf_cal_prompt_ids"].flatten()
            seen["conf"].extend(target.tolist())
            assert len(target) <= 2
            return ((model - target) ** 2).mean()

    class Scalar(torch.nn.Parameter):
        def train(self):
            return self

    parameter = Scalar(torch.tensor(0.7, dtype=torch.float64))
    trainer = _trainer_class(Base, cfg, 269)()
    trainer.args = SimpleNamespace(num_generations=8, per_device_train_batch_size=2)
    trainer.environment = SimpleNamespace(generate_hypothesis=True, generate_confidence_calibration=True)
    trainer.state = SimpleNamespace(global_step=phase)
    trainer.loss_scheduler = SimpleNamespace(phases=[SimpleNamespace(**dict(p)) for p in cfg.training.loss_schedule.phases], repeat=True)
    trainer.optimizer = torch.optim.Adam([parameter], lr=1e-5)
    trainer.accelerator = SimpleNamespace(backward=lambda value: value.backward())
    trainer.da_loss_weight, trainer.ha_sft_loss_weight, trainer.ha_conf_cal_loss_weight = 1., 0., 0.
    loss = trainer.training_step(parameter, None)
    reference = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    if phase == 100:
        weights = label_mask.sum(dim=1)
        mean = (((reference - targets_ha) ** 2) * weights).sum() / weights.sum()
        assert seen["ha"] == list(range(1, count + 1))
        assert seen["da"] == seen["conf"] == []
    else:
        mean = ((reference - targets_da) ** 2).mean()
        assert seen["da" if phase == 0 else "conf"] == list(range(1, 9))
        assert seen["ha"] == []
    # Upstream accumulates four batch-two losses without dividing gradients by four.
    (4 * mean).backward()
    assert parameter.grad.item() == pytest.approx(reference.grad.item())
    assert loss.item() == pytest.approx(mean.item())


def test_checkpoint_restores_actual_environment_pair_rng(tmp_path):
    from replication.gpu import _environment_rng_callback, _restore_environment_rng
    from replication.shim import prepare_environment
    environment = prepare_environment(compose())
    patients = [SimpleNamespace(hypothesis=["appendicitis", "cholecystitis", "pancreatitis"],
                               hypothesis_prompt=["invented A", "invented B", "invented C"])]
    for _ in range(13):
        environment._sample_pair(patients)
    callback = _environment_rng_callback(object, environment, "synthetic")
    callback.on_save(SimpleNamespace(output_dir=str(tmp_path)), SimpleNamespace(global_step=13), None)
    expected = [environment._sample_pair(patients) for _ in range(30)]
    restored = prepare_environment(compose())
    _restore_environment_rng(restored, tmp_path / "checkpoint-13", "synthetic")
    assert [restored._sample_pair(patients) for _ in range(30)] == expected
    with pytest.raises(RuntimeError, match="lacks the environment RNG"):
        _restore_environment_rng(restored, tmp_path / "missing", "synthetic")


def test_training_rollouts_use_independent_reconstructable_request_seeds():
    import random
    from replication.gpu import _SeededTrainingPolicy
    class Engine:
        def generate(self, prompts, sampling_params, **kwargs):
            assert len(prompts) == len(sampling_params)
            self.seeds = [param.seed for param in sampling_params]
            return [random.Random(param.seed).random() for param in sampling_params]
    prompts = ["identical invented prompt"] * 8
    params = SimpleNamespace(seed=269, temperature=1.0, n=1)
    engine = Engine()
    policy = _SeededTrainingPolicy(engine, 269, 100)
    first = policy.generate(prompts, params)
    assert len(set(engine.seeds)) == 8 and len(set(first)) == 8
    second = policy.generate(prompts, params)
    assert first != second
    restored = _SeededTrainingPolicy(Engine(), 269, 100)
    assert restored.generate(prompts, params) == first
    assert restored.generate(prompts, params) == second
    assert _SeededTrainingPolicy(Engine(), 269, 101).generate(prompts, params) != first
    assert params.seed == 269  # Shared configuration is not mutated.
