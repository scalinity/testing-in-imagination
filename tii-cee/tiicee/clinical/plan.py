"""The admission each optimizer step trains on, fixed before the first step (RESEARCH_LEDGER.md,
D-CLIN-RESUME): every training, interrupted or not, consumes exactly the sequence an uninterrupted
run with the same seed would.

An uninterrupted training takes its order from one sampler: built once per process by
_get_train_sampler (TRL's GRPOTrainer; called by lacdm/upstream/src/trainer/cdm_grpo_trainer.py:1137)
and iterated once per epoch by the Trainer's epoch loop. A process that resumes builds a new sampler,
whose first iteration is the first epoch's order again whatever epoch is being resumed.

  draw        builds that sampler afresh, as the trainer itself does, and iterates it once per
              epoch: the plan. It also checks what the plan relies on: the order is fixed by the
              seed, drawing it moves no process-wide generator, and each group of rollouts is one
              admission.
  Schedule    saves the plan before the first optimizer step, requires every later part to draw
              the same plan, checks each optimizer step against it as the step builds its prompts,
              and keeps the sequence trained on so far inside every checkpoint.
  following   the continuation: the trainer's own sampler object, whose iteration yields the
              planned order of the epoch that holds the trainer's current step. It does not count
              its own iterations, so it is right for a resumed process and for a Trainer that
              replays earlier epochs' iterations before resuming as well as one that does not.
  planned     wraps replication.gpu._trainer_class (gpu.py:160-307, used at gpu.py:344): the trainer
              class it returns is subclassed to override _get_train_sampler with the above.
  observed    wraps replication.gpu.prepare_environment (used at gpu.py:345): the training
              environment reports each batch to the Schedule before prompting for it. The trainer
              prompts once per optimizer step (cdm_grpo_trainer.py:716, from gpu.py:200).

A departure raises Departed and the policy is not published. numpy only; torch is looked at only
if the process has already loaded it.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

PLAN, SEQUENCE = "plan.json", "sequence.json"


class Departed(RuntimeError):
    """The training does not follow its plan, or cannot be shown to. Messages name steps, never admissions."""


def _process_generators():
    """State of the process-wide generators a sampler could draw from instead of its own."""
    state = [random.getstate(), np.random.get_state()[1].tobytes(), np.random.get_state()[2]]
    if "torch" in sys.modules:
        state.append(sys.modules["torch"].random.get_rng_state().numpy().tobytes())
    return state


def draw(fresh, patient_id, epochs, group) -> dict:
    """The plan of an uninterrupted run. fresh() builds the trainer's sampler; patient_id(index)
    names the admission at a dataset index; group is the number of rollouts per optimizer step."""
    before = _process_generators()
    sampler = fresh()
    orders = [[int(index) for index in sampler] for _ in range(epochs)]
    again = fresh()
    if [[int(index) for index in again] for _ in range(epochs)] != orders:
        raise Departed("the sampler's order is not fixed by its seed")
    if _process_generators() != before:
        raise Departed("the sampler draws from a process-wide generator")
    steps = []
    for order in orders:
        if not order or len(order) != len(sampler) or len(order) % group:
            raise Departed("an epoch is not a whole number of rollout groups")
        for start in range(0, len(order), group):
            members = set(order[start:start + group])
            if len(members) != 1:
                raise Departed("a rollout group mixes admissions")
            steps.append(patient_id(members.pop()))
    return {"epochs": epochs, "group": group, "orders": orders, "steps": steps}


class Schedule:
    """One training's plan and the record of what it has trained on, across its parts."""

    def __init__(self, directory, checkpoint=None):
        self.path = Path(directory) / PLAN
        self.plan = json.loads(self.path.read_text()) if self.path.is_file() else None
        self.sequence = []
        if checkpoint is not None:
            try:
                self.sequence = json.loads((Path(checkpoint) / SEQUENCE).read_text())
            except (OSError, ValueError):
                raise Departed("the checkpoint carries no record of the admissions trained on") from None
            if self.plan is None or len(self.sequence) != int(Path(checkpoint).name.split("-")[1]) \
                    or self.sequence != self.plan["steps"][:len(self.sequence)]:
                raise Departed("the checkpoint's record does not follow the plan")

    def settle(self, drawn) -> None:
        """Before the first optimizer step of each part: the plan a fresh sampler gives now. The
        first part saves it; a later part must have drawn the same."""
        if self.plan is None:
            temporary = self.path.with_name(PLAN + ".tmp")
            temporary.write_text(json.dumps(drawn))
            temporary.replace(self.path)
            self.plan = drawn
        elif drawn != self.plan:
            raise Departed("the plan drawn now differs from the one saved before the first step")

    def note(self, patient_ids) -> None:
        """An optimizer step is about to prompt for this batch: it must be the planned admission."""
        step = len(self.sequence)
        if self.plan is None or step >= len(self.plan["steps"]) or patient_ids != [self.plan["steps"][step]]:
            raise Departed(f"optimizer step {step + 1} is not on its planned admission")
        self.sequence.append(patient_ids[0])

    def save(self, checkpoint) -> None:
        (Path(checkpoint) / SEQUENCE).write_text(json.dumps(self.sequence))

    def finish(self, steps) -> None:
        """The finished training, whole: every planned step trained on, in order, and nothing else."""
        if self.plan is None or steps != len(self.plan["steps"]) or self.sequence != self.plan["steps"]:
            raise Departed("the finished training does not match its plan")


def following(sampler, orders, group, global_step):
    """The same sampler object, iterating in the planned order of the epoch that holds the
    trainer's current step. Only its iteration changes; its length and attributes stay its own."""
    per_epoch = len(orders[0]) // group

    class Following(type(sampler)):
        def __iter__(self):
            return iter(orders[min(global_step() // per_epoch, len(orders) - 1)])

    sampler.__class__ = Following
    return sampler


def planned(make_trainer_class, schedule, follow=True):
    """replication.gpu._trainer_class, whose trainer settles the plan before its first step and,
    unless follow is false, trains in the plan's order. follow=False leaves the trainer's sampler
    exactly as TRL built it: the bench's uninterrupted reference, which is only observed."""

    def build(base, cfg, seed):
        class Planned(make_trainer_class(base, cfg, seed)):
            def _get_train_sampler(self, *args, **kwargs):
                def fresh():
                    return super(Planned, self)._get_train_sampler(*args, **kwargs)

                epochs = int(self.args.num_train_epochs)
                if epochs != self.args.num_train_epochs or self.args.gradient_accumulation_steps != 1:
                    raise Departed("the plan needs whole epochs and one batch per optimizer step")
                group = self.args.num_generations  # the training batch size (cdm_grpo_trainer.py:1129)
                drawn = draw(fresh, lambda index: int(self.train_dataset[index]["patient_id"]), epochs, group)
                schedule.settle(drawn)
                if not follow:
                    return fresh()
                return following(fresh(), drawn["orders"], group, lambda: self.state.global_step)

        return Planned

    return build


def observed(prepare_environment, schedule):
    """replication.gpu.prepare_environment, whose environment reports every training batch to the
    schedule before building its prompts. What the trainer gets back is unchanged."""

    def build(cfg):
        environment = prepare_environment(cfg)
        construct_prompt = environment.construct_prompt

        def noting(inputs):
            schedule.note(sorted({int(item["patient_id"]) for item in inputs}))
            return construct_prompt(inputs)

        environment.construct_prompt = noting
        return environment

    return build
