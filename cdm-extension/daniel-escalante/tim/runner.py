"""Guarded evaluation inside the baseline's own evaluator (SPEC §3.6).

The GPU path calls replication.gpu.evaluate with its module-level _evaluate_with_model patched
to make_evaluator(...): engine creation, adapter merge and sampling stay the baseline's own.
"""
from dataclasses import dataclass, field
import hashlib
import json
import re
import traceback

from tim.actions import PATTERN, Vocabulary
from tim.fallback import finalize
from tim.guard import FIXED_THOUGHT, Limits, fit, trigger
from tim.scoring import best


@dataclass(frozen=True)
class Variant:
    name: str
    g1: bool = False
    g2: bool = False
    g3: bool = False
    shadow: bool = False

    @property
    def scored(self) -> bool:
        return self.g1 or self.g2 or self.shadow


VARIANTS = {
    "V0": Variant("V0"),
    "V0-rerun": Variant("V0-rerun"),
    "V0-shadow": Variant("V0-shadow", shadow=True),
    "V2": Variant("V2", g2=True, g3=True),
    "V3": Variant("V3", g1=True, g2=True, g3=True),
}


@dataclass
class EpisodeState:
    turns: int = 0
    termination: str | None = None
    events: list = field(default_factory=list)
    scorer_errors: int = 0
    replacement_failed: int = 0


def replace(trajectory, text, ids) -> None:
    """Swap the latest completion for a replacement in text, token mask and completion ids."""
    old_ids, old_text = len(trajectory.completion_ids), len(trajectory.last_completion)
    trajectory.token_mask = trajectory.token_mask[:len(trajectory.token_mask) - old_ids] + [True] * len(ids)
    trajectory.text = trajectory.text[:len(trajectory.text) - old_text] + text
    trajectory.completion_ids = list(ids)
    trajectory.last_completion = text


def guarded_class(base, variant, vocab, scorer, limits=Limits()):
    class Guarded(base):
        def run(self, prompts_text, inputs, model, sampling_params, processing_class):
            self.tim_state, self.tim_tokenizer = EpisodeState(), processing_class
            return super().run(prompts_text, inputs, model, sampling_params, processing_class)

        def _event(self, t, kind, from_kind, to_kind):
            self.tim_state.events.append({"t": t, "type": kind, "from_kind": from_kind, "to_kind": to_kind})

        def _score(self, t, prompt_ids, thought, candidates, from_kind):
            try:
                return scorer.score(prompt_ids, thought, list(candidates))
            except Exception:
                traceback.print_exc()  # private log only
                self.tim_state.scorer_errors += 1
                self._event(t, "scorer_error", from_kind, from_kind)
                return None

        def _guard(self, trajectory, t):
            requested = list(trajectory.requested_tests)
            proposal = vocab.classify(trajectory.last_completion, requested)
            prompt_ids = trajectory.output.prompt_token_ids
            if variant.shadow:
                self._score(t, prompt_ids, proposal.thought or FIXED_THOUGHT["g1_replace"],
                            vocab.tests + vocab.diseases, proposal.kind)
            found = trigger(t, len(prompt_ids), len(trajectory.completion_ids), proposal, requested, vocab,
                            variant.g1, variant.g2, limits)
            if found is None:
                return
            thought = proposal.thought if proposal.thought is not None else FIXED_THOUGHT[found.event]
            scores = self._score(t, prompt_ids, thought, found.candidates, found.from_kind)
            if scores is None:
                return
            action = found.candidates[best(scores)]
            fitted = fit(action, proposal.thought, found.event, len(prompt_ids), self.tim_tokenizer, limits)
            if fitted is None:
                self.tim_state.replacement_failed += 1
                self._event(t, "replacement_failed", found.from_kind, action.kind)
                return
            replace(trajectory, *fitted)
            self._event(t, found.event, found.from_kind, action.kind)

        def step(self, trajectory, step, tokenizer):
            self.tim_state.turns += 1
            if step <= limits.max_steps:
                self._guard(trajectory, step)
            over = len(trajectory.token_mask) > self.max_length
            matched = re.search(PATTERN, trajectory.last_completion) is not None
            super().step(trajectory, step, tokenizer)
            if trajectory.completed and self.tim_state.termination is None:
                self.tim_state.termination = (
                    "diagnosis" if not trajectory.invalid else
                    "step_limit" if step > limits.max_steps else
                    "length_completion" if over else
                    "malformed" if not matched else
                    "length_observation")

    Guarded.__name__ = Guarded.__qualname__ = f"Guarded{base.__name__}"
    return Guarded


class Recorder:
    """Passes generation through and keeps a digest of every generated token sequence."""

    def __init__(self, model):
        self.model, self.digests = model, []

    def get_tokenizer(self):
        return self.model.get_tokenizer()

    def generate(self, prompts, *args, **kwargs):
        outputs = self.model.generate(prompts, *args, **kwargs)
        for output in outputs:
            ids = ",".join(str(i) for i in output.outputs[0].token_ids)
            self.digests.append(hashlib.sha256(ids.encode()).hexdigest())
        return outputs

    def __getattr__(self, name):
        return getattr(self.model, name)


def make_evaluator(variant, dataset, scorer_factory, log=None, limits=Limits()):
    """A drop-in for replication.shim._evaluate_with_model over tim.access's patient selection."""

    def evaluate(cfg, _baseline_dataset, model, sampling_params):
        from hydra.utils import instantiate
        from replication.shim import ensure_pad_token, prepare_environment
        environment = prepare_environment(cfg)
        tokenizer = ensure_pad_token(model.get_tokenizer())
        vocab = Vocabulary(environment.test_list, environment.disease_list)
        scorer = scorer_factory(cfg, tokenizer) if variant.scored else None
        environment.__class__ = guarded_class(type(environment), variant, vocab, scorer, limits)
        rewards = [instantiate(item) for item in cfg.reward_function.reward_functions]
        recorder, rows = Recorder(model), []
        for index in range(len(dataset)):
            item = dict(dataset[index])
            patient_id = int(item["patient_id"])
            if log is not None and patient_id in log.done:
                continue
            inputs = environment.construct_prompt([item])
            prompts = [tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)]
            recorder.digests = []
            output = environment.run(prompts, inputs, recorder, sampling_params, tokenizer)
            kwargs = output.reward_function_kwargs[0]
            reward_kwargs = {key: [value] for key, value in kwargs.items()}
            reward = sum(function(prompts=prompts, completions=[None],
                                  condition=[item["condition"]], **reward_kwargs)[0] for function in rewards)
            finalized, source = finalize(kwargs["diagnosis"], output.ha_hypotheses_lists[0], variant.g3)
            state = environment.tim_state
            row = {
                "patient_id": patient_id, "label": item["condition"], "prediction": finalized,
                **environment.captured[0], "reward": float(reward),
                "hypotheses": output.ha_hypotheses_lists[0], "confidences": output.ha_confidences_lists[0],
                "ece_triplets": list(output.ece_calc_triplets),
                "variant": variant.name, "raw_prediction": kwargs["diagnosis"], "finalized_prediction": finalized,
                "prediction_source": source, "termination": state.termination, "n_turns": state.turns,
                "events": state.events, "scorer_error_count": state.scorer_errors,
                "replacement_failed_count": state.replacement_failed,
                "generation_digest": hashlib.sha256(",".join(recorder.digests).encode()).hexdigest(),
            }
            row = json.loads(json.dumps(row))
            rows.append(row)
            if log is not None:
                log.append(patient_id, row)
        return rows

    return evaluate


def split_cfg(settings, split):
    """The baseline's configuration with the evaluation seed gpu.evaluate derives for this split."""
    from replication.config import stream_seed
    from tim.access import baseline_cfg
    cfg = baseline_cfg(settings)
    cfg.training.seed = stream_seed(stream_seed(269, split, "policy"), split, "evaluation")
    return cfg
