"""Candidate scorer: teacher-forced log-probability of each canonical continuation (SPEC §3.4).

The core takes model-agnostic callables so the CPU tests can check it against brute force:
forward(ids, keep) -> logits of the last `keep` positions; prefill(ids) -> (last logits, state);
extend(state, batch) -> logits for every position of each batch row.
"""
from contextlib import contextmanager

import torch


def _sum_logprobs(logits, targets) -> float:
    targets = torch.as_tensor(targets, device=logits.device)
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets[:, None]).sum().item()


def reference_scores(forward, context, continuations) -> list[float]:
    """One full-sequence forward per candidate."""
    scores = []
    for continuation in continuations:
        logits = forward(torch.tensor(context + continuation), len(continuation) + 1)[:-1]
        scores.append(_sum_logprobs(logits, continuation))
    return scores


def fast_scores(prefill, extend, context, continuations) -> list[float]:
    """One prefill of the shared context, then one right-padded batch of continuations."""
    last, state = prefill(torch.tensor(context))
    width = max(len(c) for c in continuations)
    batch = torch.tensor([c + [c[-1]] * (width - len(c)) for c in continuations])  # padding is never read
    logits = extend(state, batch)
    first = torch.log_softmax(last.float(), dim=-1)
    scores = []
    for row, continuation in enumerate(continuations):
        score = first[continuation[0]].item()
        if len(continuation) > 1:
            score += _sum_logprobs(logits[row, :len(continuation) - 1], continuation[1:])
        scores.append(score)
    return scores


def best(scores) -> int:
    """Index of the highest score; ties go to the earlier candidate (configured tests, then diseases)."""
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def encode(tokenizer, prompt_ids, thought, candidates):
    """Context = the decision prompt's ids as vLLM saw them + "Thought: {thought}\\nAction:"."""
    context = list(prompt_ids) + tokenizer.encode(f"Thought: {thought}\nAction:", add_special_tokens=False)
    continuations = [tokenizer.encode(a.continuation(tokenizer.eos_token), add_special_tokens=False)
                     for a in candidates]
    return context, continuations


@contextmanager
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


class HFScorer:
    """A separate Hugging Face instance with the merged weights; vLLM never scores."""

    def __init__(self, model, tokenizer, path="fast"):
        self.model, self.tokenizer, self.path = model.eval(), tokenizer, path
        self.device = next(model.parameters()).device

    def _forward(self, ids, keep):
        return self.model(input_ids=ids[None].to(self.device), logits_to_keep=keep).logits[0]

    def _prefill(self, ids):
        out = self.model(input_ids=ids[None].to(self.device), use_cache=True, logits_to_keep=1)
        return out.logits[0, -1], out.past_key_values

    def _extend(self, cache, batch):
        cache.batch_repeat_interleave(len(batch))
        return self.model(input_ids=batch.to(self.device), past_key_values=cache, use_cache=True).logits

    def next_token(self, prompt_ids) -> int:
        """The HF model's greedy next token for a decision prompt (bench agreement check)."""
        with deterministic(), torch.inference_mode():
            return int(self._prefill(torch.tensor(list(prompt_ids)))[0].argmax())

    def score(self, prompt_ids, thought, candidates, path=None) -> list[float]:
        context, continuations = encode(self.tokenizer, prompt_ids, thought, candidates)
        with deterministic(), torch.inference_mode():
            if (path or self.path) == "fast":
                return fast_scores(self._prefill, self._extend, context, continuations)
            return reference_scores(self._forward, context, continuations)


class BenchScorer:
    """Scores with the fast path and checks it against the reference path on every call (SPEC 3.4)."""

    def __init__(self, inner, tolerance=1e-3):
        self.inner, self.tolerance = inner, tolerance
        self.calls, self.max_diff, self.checks, self.agreements = 0, None, 0, 0

    def score(self, prompt_ids, thought, candidates, path=None):
        fast = self.inner.score(prompt_ids, thought, candidates, path="fast")
        reference = self.inner.score(prompt_ids, thought, candidates, path="reference")
        diff = max(abs(a - b) for a, b in zip(fast, reference))
        self.max_diff = diff if self.max_diff is None else max(self.max_diff, diff)
        self.calls += 1
        return fast

    def observe(self, prompt_ids, first_token):
        self.checks += 1
        self.agreements += int(self.inner.next_token(prompt_ids) == first_token)

    def measurements(self) -> dict:
        return {"scoring_calls": self.calls, "max_abs_fast_minus_reference": self.max_diff,
                "fast_within_tolerance": None if self.max_diff is None else self.max_diff <= self.tolerance,
                "next_token_checks": self.checks,
                "next_token_agreement": self.agreements / self.checks if self.checks else None}


def load_scorer(cfg, adapter_dir, tokenizer, path="fast"):
    """Merged policy weights in bf16 via upstream load_model, PeftModel and merge_and_unload (GPU)."""
    from copy import deepcopy
    from importlib import import_module
    from omegaconf import open_dict
    adapter_cfg = deepcopy(cfg)
    with open_dict(adapter_cfg):
        adapter_cfg.evaluation.base_model = False
        adapter_cfg.evaluation.adapter_dir = str(adapter_dir)
    model = import_module("src.utils.setup").load_model(adapter_cfg, inference_mode=True).merge_and_unload()
    return HFScorer(model, tokenizer, path)
