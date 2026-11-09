"""Guarded evaluation inside the baseline's own evaluator (SPEC §3.6).

The GPU path calls replication.gpu.evaluate with its module-level _evaluate_with_model patched
to make_evaluator(...): engine creation, adapter merge and sampling stay the baseline's own.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
import re
import traceback

import math

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
    planner: dict | None = None

    @property
    def scored(self) -> bool:
        return self.g1 or self.g2 or self.shadow or self.planner is not None


def load_config() -> dict:
    import yaml
    from tim.paths import OWN
    return yaml.safe_load((OWN / "configs" / "variants.yaml").read_text())


CONFIG = load_config()
VARIANTS = {name: Variant(name, **flags) for name, flags in {**CONFIG["variants"], **CONFIG["arms"]}.items()}
STAGE_SPECS = {**CONFIG["stages"], **CONFIG["stages_b"]}


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


def guarded_class(base, variant, vocab, scorer, limits=Limits(), predictor=None):
    theta, beta = CONFIG["planner"]["theta_stop"], CONFIG["planner"]["beta"]

    class Guarded(base):
        def run(self, prompts_text, inputs, model, sampling_params, processing_class):
            self.tim_state, self.tim_tokenizer = EpisodeState(), processing_class
            self.tim_p = predictor(inputs[0]) if variant.planner is not None else None
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
                generated = trajectory.output.outputs[0].token_ids
                if hasattr(scorer, "observe") and len(generated):
                    scorer.observe(prompt_ids, generated[0])
            found = trigger(t, len(prompt_ids), len(trajectory.completion_ids), proposal, requested, vocab,
                            variant.g1, variant.g2, limits)
            if found is None:
                if self.tim_p is not None and proposal.kind == "test" and t < limits.max_steps:
                    self._plan(trajectory, t, proposal, requested, prompt_ids)
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

        def _plan(self, trajectory, t, proposal, requested, prompt_ids):
            """SPEC 4.4: stop when no untried test is likely charted, else rerank by score + beta log p."""
            untried = vocab.untried(requested)
            if max(self.tim_p[a.name] for a in untried) < theta:
                event, candidates = "wm_stop", vocab.diseases
            else:
                event, candidates = "wm_rerank", untried
            scores = self._score(t, prompt_ids, proposal.thought, candidates, "test")
            if scores is None:
                return
            if event == "wm_rerank":
                scores = [s + beta * math.log(self.tim_p[a.name]) for s, a in zip(scores, candidates)]
            action = candidates[best(scores)]
            if action == proposal.action:
                return
            fitted = fit(action, proposal.thought, event, len(prompt_ids), self.tim_tokenizer, limits)
            if fitted is None:
                self.tim_state.replacement_failed += 1
                self._event(t, "replacement_failed", "test", action.kind)
                return
            replace(trajectory, *fitted)
            self._event(t, event, "test", action.kind)

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
    """Passes generation through and keeps a digest of each hypothesis and decision generation.

    Confidence-calibration generations (prompts ending in "Confidence: ") are left out: upstream
    discards them and picks their prompts with an RNG that a resumed run restarts (SPEC §5.5).
    """

    def __init__(self, model):
        self.model, self.digests = model, []

    def get_tokenizer(self):
        return self.model.get_tokenizer()

    def generate(self, prompts, *args, **kwargs):
        outputs = self.model.generate(prompts, *args, **kwargs)
        for prompt, output in zip(prompts, outputs):
            if prompt.endswith("Confidence: "):
                continue
            ids = ",".join(str(i) for i in output.outputs[0].token_ids)
            self.digests.append(hashlib.sha256(ids.encode()).hexdigest())
        return outputs

    def __getattr__(self, name):
        return getattr(self.model, name)


def make_evaluator(variant, dataset, scorer_factory, log=None, limits=Limits(), predictor=None):
    """A drop-in for replication.shim._evaluate_with_model over tim.access's patient selection."""

    def evaluate(cfg, _baseline_dataset, model, sampling_params):
        from hydra.utils import instantiate
        from replication.shim import ensure_pad_token, prepare_environment
        environment = prepare_environment(cfg)
        tokenizer = ensure_pad_token(model.get_tokenizer())
        vocab = Vocabulary(environment.test_list, environment.disease_list)
        scorer = scorer_factory(cfg, tokenizer) if variant.scored else None
        environment.__class__ = guarded_class(type(environment), variant, vocab, scorer, limits, predictor)
        rewards = [instantiate(item) for item in cfg.reward_function.reward_functions]
        recorder, rows = Recorder(model), []
        if log is not None and hasattr(log, "started"):
            log.started()
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


class Progress:
    """Episode log plus public progress events, one per completed patient."""

    def __init__(self, episodes, bundle, total, already, clock):
        self.episodes, self.bundle, self.total, self.count = episodes, bundle, total, already
        self.done, self.clock, self.setup_done = episodes.done, clock, None

    def started(self):
        self.setup_done = self.clock()

    def append(self, patient_id, row):
        self.episodes.append(patient_id, row)
        self.count += 1
        self.bundle.emit("progress", completed=self.count, total=self.total)


def _gpu_rows(settings, split, data_dir, evaluator_for):
    """Run the baseline's own GPU evaluator with tim's evaluator patched in (SPEC 3.6)."""
    from unittest.mock import patch
    from replication import gpu, pipeline
    from replication.config import stream_seed
    from replication.manifest import file_digest
    from tim.events import TimError
    baseline = settings.baseline()
    pipeline._assert_runtime(baseline)
    selection = pipeline.read_json(baseline.work_dir / "models/parity/269/primary/selection.json")
    for path, expected in selection["run_outputs"].items():
        if file_digest(path) != expected:
            raise TimError("baseline_mismatch", "the selected primary model changed")
    with patch.object(gpu, "_evaluate_with_model", evaluator_for(selection["adapter_dir"])):
        return gpu.evaluate(pipeline.cfg_for(baseline, "primary"), data_dir, split=split,
                            seed=stream_seed(269, split, "policy"), adapter_dir=selection["adapter_dir"])


def run_job(settings, stage, variant_name, bundle, protocol_hash, commit, clock):
    """One (stage, variant) job: admission, resume, guarded evaluation, public counts."""
    from tim import access
    from tim.scoring import BenchScorer
    split = STAGE_SPECS[stage]["split"]
    variant, limits = VARIANTS[variant_name], Limits(**CONFIG["limits"])
    dataset = access.open_split(settings, stage, split)
    episodes = access.Episodes(settings, protocol_hash, stage, variant_name, split, commit, len(dataset))
    done = episodes.open()
    if done:
        bundle.emit("progress", completed=len(done), total=len(dataset), resumed=True)
    log = Progress(episodes, bundle, len(dataset), len(done), clock)
    bench = [] if stage.startswith("bench") and variant.shadow else None

    def wrap(scorer):
        if bench is None:
            return scorer
        bench.append(BenchScorer(scorer))
        return bench[-1]

    start = clock()
    if stage == "smoke" and not settings.synthetic:
        episodes.ledger.record("smoke_started", key=episodes.key, job=os.environ.get("SLURM_JOB_ID", "local"))
    try:
        if episodes.complete:
            rows = []  # every patient is logged; only the finish record was missing
        elif settings.synthetic:
            from replication.shim import generation_settings
            from tim.fixture import MockScorer, ScriptedPolicy
            evaluator = make_evaluator(variant, dataset, lambda cfg, tok: wrap(MockScorer(tok)), log, limits)
            rows = evaluator(split_cfg(settings, split), None, ScriptedPolicy(), generation_settings(False))
        else:
            from tim.scoring import load_scorer
            rows = _gpu_rows(settings, split, dataset.data_dir, lambda adapter: make_evaluator(
                variant, dataset, lambda cfg, tok: wrap(load_scorer(cfg, adapter, tok)), log, limits))
        episodes.finish()
    finally:
        episodes.close()
        if stage == "smoke" and not settings.synthetic:
            episodes.ledger.record("smoke_seconds", key=episodes.key, job=os.environ.get("SLURM_JOB_ID", "local"),
                                   seconds=clock() - start)
    rows, seconds = done + rows, clock() - start
    if stage.startswith("bench"):
        measured = bench[0].measurements() if bench else {
            "scoring_calls": 0, "max_abs_fast_minus_reference": None, "fast_within_tolerance": None,
            "next_token_checks": 0, "next_token_agreement": None}
        bundle.aggregate("bench", {"variant": variant_name, "patients": len(rows), "seconds": seconds,
                                   "setup_seconds": (log.setup_done or start) - start,
                                   "decision_turns": sum(r["n_turns"] for r in rows), **measured}, job=variant_name)
    return {"patients": len(rows), "seconds": seconds}
