"""The training plan (D-CLIN-RESUME): drawn before the first step, checked at every step, continued
on resume. Stubs stand for the Trainer loop and TRL's sampler (test_clinical_support.TrainerBase);
the pieces under test are the real ones train.fit wires around replication.gpu.train."""
import importlib.util
import json
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from test_clinical_support import CallbackBase, RepeatRandomSampler, TrainerBase
from tiicee.clinical import plan, train, use_prior_studies

use_prior_studies()
from replication import gpu  # noqa: E402  (imports with the standard library alone)

IDS = list(range(700, 730))  # 30 admissions: 30 optimizer steps an epoch, 60 in all
needs_torch = pytest.mark.skipif(not importlib.util.find_spec("torch"), reason="needs torch (tim's CPU environment)")


def part(directory, seed=5, pause_at=None, follow=True, boundary=True, at_step=lambda step: None, **behaviour):
    """One process of a training, wired as train.fit wires gpu.train: the schedule, the observed
    environment, the stop-aware callback over gpu's own sidecar callback, the planned trainer class."""
    checkpoint = train.last_checkpoint(gpu, directory)
    schedule = plan.Schedule(directory, checkpoint)
    environment = plan.observed(lambda cfg: SimpleNamespace(construct_prompt=lambda inputs: inputs, _rng=random.Random(seed)),
                                schedule)(None)
    made = []
    callback = train.stop_aware(gpu._environment_rng_callback, lambda step, total: step == pause_at, made,
                                lambda step: schedule.save(directory / f"checkpoint-{step}"), boundary)(
        CallbackBase, environment, "mimic")
    args = SimpleNamespace(output_dir=str(directory), seed=seed, num_generations=8, num_train_epochs=2,
                           gradient_accumulation_steps=1)
    trainer_class = plan.planned(lambda base, cfg, seed: base, schedule, follow)(type("Base", (TrainerBase,), behaviour), None, seed)
    trainer = trainer_class([{"patient_id": patient_id} for patient_id in IDS], environment, callback, args, at_step)
    if checkpoint:
        gpu._restore_environment_rng(environment, checkpoint, "mimic")
    trainer.train(resume_from_checkpoint=checkpoint)
    schedule.finish(made[-1].final_step)
    return schedule, trainer.state


def whole(directory, **options):
    """Every part of a training, a new process after each pause. Returns (schedule, state, pauses)."""
    directory.mkdir(exist_ok=True)
    pauses = []
    while True:
        try:
            return (*part(directory, **options), pauses)
        except train.Paused as paused:
            pauses.append(paused.step)


@pytest.fixture
def reference(tmp_path):
    """The uninterrupted run with the trainer's sampler untouched: observed, never steered."""
    return whole(tmp_path / "reference", follow=False, boundary=False)


def test_the_plan_is_what_an_untouched_uninterrupted_run_trains_on(reference):
    schedule, state, pauses = reference
    assert pauses == [] and state.global_step == 60 and state.validations == [30, 60]
    assert state.sampler == "RepeatRandomSampler"  # the trainer iterated the sampler it built, as built
    assert schedule.sequence == schedule.plan["steps"] and len(schedule.sequence) == 60
    assert all(Counter(schedule.sequence[start:start + 30]) == Counter(IDS) for start in (0, 30))  # each admission once an epoch
    assert schedule.sequence[:30] != schedule.sequence[30:]  # and the second epoch has its own order
    assert (schedule.plan["epochs"], schedule.plan["group"]) == (2, 8) and len(schedule.plan["orders"][1]) == 240


def test_the_plan_is_saved_before_the_first_optimizer_step(tmp_path):
    seen = []
    whole(tmp_path / "run", at_step=lambda step: seen.append((tmp_path / "run" / "plan.json").is_file()))
    assert len(seen) == 60 and all(seen)


@pytest.mark.parametrize("replays", [False, True])
def test_a_training_in_parts_trains_on_exactly_the_uninterrupted_sequence(tmp_path, reference, replays):
    """Paused at the epoch boundary and again inside the second epoch, under a Trainer that replays
    finished epochs' iterations on resume and under one that does not."""
    schedule, state, pauses = whole(tmp_path / "parts", pause_at=45, replays_epochs_on_resume=replays)
    assert pauses == [30, 45] and state.sampler == "Following"
    assert schedule.sequence == reference[0].sequence and schedule.plan == reference[0].plan
    assert state.digest == reference[1].digest          # what the run left behind saw the same order
    assert state.validations == reference[1].validations == [30, 60]  # no validation pass lost to a pause
    assert json.loads((tmp_path / "parts" / "checkpoint-45" / "sequence.json").read_text()) == schedule.sequence[:45]


def test_without_the_continuation_a_resumed_epoch_departs_and_is_caught(tmp_path, reference):
    """The hazard itself: a new process's sampler starts its generator over, so the resumed second
    epoch would follow the first epoch's order. The step check stops it at once."""
    with pytest.raises(plan.Departed, match="optimizer step 31 "):
        whole(tmp_path / "unsteered", follow=False, replays_epochs_on_resume=False)
    replayed = whole(tmp_path / "replayed", follow=False, replays_epochs_on_resume=True)  # no false alarm
    assert replayed[0].sequence == reference[0].sequence and replayed[2] == [30]


def test_a_step_on_another_admission_than_planned_ends_the_training_there(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(plan.Departed, match="optimizer step 17 ") as departed:
        part(tmp_path / "run", stray_at=17)
    assert not any(str(patient_id) in str(departed.value) for patient_id in IDS)  # steps are named, admissions never


def test_a_later_part_must_draw_the_plan_saved_before_the_first_step(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(train.Paused):
        part(tmp_path / "run", seed=5)
    with pytest.raises(plan.Departed, match="differs"):
        part(tmp_path / "run", seed=6)


def test_a_finished_training_must_cover_its_whole_plan(tmp_path, reference):
    schedule = reference[0]
    schedule.finish(60)
    with pytest.raises(plan.Departed, match="does not match"):
        schedule.finish(59)       # the trainer stopped a step short of the plan
    schedule.sequence.pop()
    with pytest.raises(plan.Departed, match="does not match"):
        schedule.finish(60)       # the record misses a step
    with pytest.raises(plan.Departed):
        plan.Schedule(tmp_path / "nowhere").finish(0)  # no plan was ever settled


def test_a_checkpoint_must_carry_a_record_that_follows_the_plan(tmp_path, reference):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(train.Paused):
        part(run, boundary=False, pause_at=20)
    checkpoint = run / "checkpoint-20"
    assert plan.Schedule(run, checkpoint).sequence == reference[0].sequence[:20]
    record = json.loads((checkpoint / "sequence.json").read_text())
    for damaged in (record[:19], record + [record[0]], [record[1], record[0], *record[2:]]):
        (checkpoint / "sequence.json").write_text(json.dumps(damaged))
        with pytest.raises(plan.Departed, match="does not follow"):
            plan.Schedule(run, checkpoint)
    (checkpoint / "sequence.json").unlink()
    with pytest.raises(plan.Departed, match="no record"):
        plan.Schedule(run, checkpoint)
    (checkpoint / "sequence.json").write_text(json.dumps(record))
    (run / "plan.json").unlink()
    with pytest.raises(plan.Departed, match="does not follow"):
        plan.Schedule(run, checkpoint)


@pytest.fixture
def process_generators():
    """Tests that make a sampler misuse the process-wide generators put them back afterwards."""
    state = random.getstate()
    yield
    random.setstate(state)


def sampler_of(build):
    """A sampler class whose order comes from build(sampler); 6 admissions, eight rollouts each."""
    return type("Sampler", (), {"__init__": lambda self: None, "__len__": lambda self: 48,
                                "__iter__": lambda self: iter([index for index in build(self) for _ in range(8)])})


def test_draw_refuses_an_order_the_plan_could_not_rely_on(process_generators):
    drawn = plan.draw(lambda: RepeatRandomSampler(range(6), 8, 3), lambda index: 900 + index, 2, 8)
    assert sorted(drawn["steps"][:6]) == sorted(drawn["steps"][6:]) == list(range(900, 906))

    own_entropy = sampler_of(lambda self: random.Random().sample(range(6), 6))  # its own generator, seeded by nothing
    with pytest.raises(plan.Departed, match="not fixed by its seed"):
        plan.draw(own_entropy, int, 2, 8)
    unseeded = sampler_of(lambda self: random.sample(range(6), 6))  # the process-wide generator, never seeded
    with pytest.raises(plan.Departed, match="not fixed by its seed|process-wide"):
        plan.draw(unseeded, int, 2, 8)

    def reseeding(self):  # the same order every time, but by resetting the process-wide generator
        random.seed(11)
        return random.sample(range(6), 6)

    with pytest.raises(plan.Departed, match="process-wide"):
        plan.draw(sampler_of(reseeding), int, 2, 8)
    with pytest.raises(plan.Departed, match="mixes admissions"):
        plan.draw(lambda: RepeatRandomSampler(range(6), 4, 3), int, 2, 8)   # four rollouts where eight are grouped
    with pytest.raises(plan.Departed, match="whole number"):
        plan.draw(lambda: RepeatRandomSampler(range(5), 4, 3), int, 2, 8)   # twenty indices: not whole groups
    with pytest.raises(plan.Departed, match="whole number"):
        plan.draw(lambda: RepeatRandomSampler(range(0), 8, 3), int, 2, 8)


def test_the_trainer_must_run_whole_epochs_and_one_batch_per_step():
    schedule = SimpleNamespace(settle=lambda drawn: pytest.fail("no plan should be settled"))
    trainer_class = plan.planned(lambda base, cfg, seed: base, schedule)(TrainerBase, None, 1)
    for args in (dict(num_train_epochs=1.5, gradient_accumulation_steps=1), dict(num_train_epochs=2, gradient_accumulation_steps=2)):
        trainer = trainer_class([{"patient_id": 1}], None, None, SimpleNamespace(num_generations=8, seed=1, **args))
        with pytest.raises(plan.Departed, match="whole epochs"):
            trainer._get_train_sampler()


def test_following_changes_only_the_order_of_the_samplers_iteration():
    orders, step = [[0] * 8 + [1] * 8, [1] * 8 + [0] * 8], [0]
    sampler = RepeatRandomSampler(range(2), 8, 9)
    followed = plan.following(sampler, orders, 8, lambda: step[0])
    assert followed is sampler and isinstance(followed, RepeatRandomSampler) and len(followed) == 16
    assert list(followed) == list(followed) == orders[0]  # however often it is iterated within the epoch
    step[0] = 1
    assert list(followed) == orders[0]
    step[0] = 2                                           # the second epoch begins at step 2
    assert list(followed) == orders[1]
    step[0] = 4                                           # past the end: still the last epoch's order
    assert list(followed) == orders[1]


@needs_torch
def test_following_drives_a_real_torch_dataloader_and_draw_watches_torchs_generator():
    import torch
    from torch.utils.data import DataLoader, Sampler

    class TorchRepeatSampler(Sampler):  # the same shape as TRL's, over torch's generators
        def __init__(self, count, seed=None):
            self.count, self.generator = count, torch.Generator() if seed is not None else None
            if seed is not None:
                self.generator.manual_seed(seed)

        def __iter__(self):
            return iter([index for index in torch.randperm(self.count, generator=self.generator).tolist() for _ in range(8)])

        def __len__(self):
            return self.count * 8

    dataset = [{"patient_id": 300 + index} for index in range(5)]
    drawn = plan.draw(lambda: TorchRepeatSampler(5, seed=4), lambda index: dataset[index]["patient_id"], 2, 8)
    step = [0]
    loader = DataLoader(dataset, batch_size=8, sampler=plan.following(TorchRepeatSampler(5, seed=4), drawn["orders"], 8, lambda: step[0]),
                        collate_fn=lambda items: items)
    assert len(loader) == 5
    for epoch in (1, 0, 1):  # in any order, and again: the order belongs to the epoch, not to the iteration count
        step[0] = epoch * 5 + 2
        batches = [[item["patient_id"] for item in batch] for batch in loader]
        assert [batch[0] for batch in batches] == drawn["steps"][epoch * 5:epoch * 5 + 5]
        assert all(len(set(batch)) == 1 and len(batch) == 8 for batch in batches)
    state = torch.random.get_rng_state()
    with pytest.raises(plan.Departed, match="not fixed by its seed|process-wide"):
        plan.draw(lambda: TorchRepeatSampler(5), int, 2, 8)  # torch's process-wide generator
    torch.random.set_rng_state(state)
