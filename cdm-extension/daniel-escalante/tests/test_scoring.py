import pytest
import torch

from tim.actions import Action
from tim.scoring import best, encode, fast_scores, reference_scores


class TinyLM:
    """Causal toy LM: a running mean of embeddings; its state is the prefix sum and length."""

    def __init__(self, vocab=13, dim=6, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(vocab, dim, generator=generator)
        self.head = torch.randn(dim, vocab, generator=generator)

    def _hidden(self, ids, total=None, count=0):
        e = self.emb[ids]
        running = torch.cumsum(e, 0) + (0 if total is None else total)
        n = torch.arange(1, len(ids) + 1)[:, None] + count
        return torch.tanh(running / n + e)

    def full(self, ids):
        return self._hidden(torch.as_tensor(ids)) @ self.head

    def forward(self, ids, keep):
        return self.full(ids)[-keep:]

    def prefill(self, ids):
        return self.full(ids)[-1], (self.emb[ids].sum(0), len(ids))

    def extend(self, state, batch):
        total, count = state
        return torch.stack([self._hidden(row, total, count) @ self.head for row in batch])


def brute_force(model, context, continuation):
    total = 0.0
    for j, token in enumerate(continuation):
        logits = model.full(context + continuation[:j])[-1]
        total += torch.log_softmax(logits.float(), -1)[token].item()
    return total


CONTEXT = [1, 4, 2, 7, 3]
CONTINUATIONS = [[5], [6, 2, 9], [8, 8], [12, 0, 3, 4]]


def test_reference_path_equals_brute_force():
    model = TinyLM()
    got = reference_scores(model.forward, CONTEXT, CONTINUATIONS)
    assert got == pytest.approx([brute_force(model, CONTEXT, c) for c in CONTINUATIONS], abs=1e-5)


def test_fast_path_equals_brute_force_with_ragged_continuations():
    model = TinyLM(seed=3)
    got = fast_scores(model.prefill, model.extend, CONTEXT, CONTINUATIONS)
    assert got == pytest.approx([brute_force(model, CONTEXT, c) for c in CONTINUATIONS], abs=1e-5)


def test_ties_go_to_the_fixed_order():
    assert best([-1.0, -0.5, -0.5, -3.0]) == 1
    assert best([-2.0, -2.0]) == 0
    assert best([-4.0, -0.1]) == 1


def test_inputs_follow_the_canonical_strings():
    from replication.shim import MockTokenizer, ensure_pad_token
    tokenizer = ensure_pad_token(MockTokenizer())
    context, continuations = encode(tokenizer, [11, 12], "check it", [Action("test", "CT"), Action("diagnosis", "appendicitis")])
    assert context == [11, 12] + tokenizer.encode("Thought: check it\nAction:")
    assert continuations == [tokenizer.encode(" Test\nAction Input: CT\nObservation:"),
                             tokenizer.encode(" Diagnosis\nAction Input: appendicitis<|eot_id|>")]
