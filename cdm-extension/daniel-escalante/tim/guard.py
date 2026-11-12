"""The commit guard's decision table (SPEC §3.3): G2 budget commit, then G1 validity guard."""
from dataclasses import dataclass

from tim.actions import Action

FIXED_THOUGHT = {"g1_replace": "I will choose the next step.",
                 "wm_rerank": "I will choose the next step.",
                 "wm_stop": "I must now give my final diagnosis.",
                 "g2_commit": "I must now give my final diagnosis.",
                 "eig_rerank": "I will choose the next step."}
INVALID = ("malformed", "unknown_action", "unknown_test", "repeated_test", "unknown_diagnosis")


@dataclass(frozen=True)
class Limits:
    max_steps: int = 13
    max_length: int = 4096   # MAX: trajectory tokens
    context: int = 4352      # CTX: model context
    completion: int = 256    # COMP: completion tokens


@dataclass(frozen=True)
class Trigger:
    event: str
    from_kind: str
    candidates: tuple[Action, ...]


def trigger(t, prompt_tokens, completion_tokens, proposal, requested, vocab, g1, g2, limits=Limits()):
    """Which guard acts on decision turn t, with its candidate set; None executes c unchanged."""
    if t > limits.max_steps:
        return None  # upstream terminates; unreachable with G2 on
    overflow = prompt_tokens + completion_tokens > limits.max_length
    if g2 and (t == limits.max_steps or overflow):
        if proposal.kind == "diagnosis" and not overflow:
            return None
        return Trigger("g2_commit", "overflow" if overflow else proposal.kind, tuple(vocab.diseases))
    if g1 and proposal.kind in INVALID:
        if proposal.kind == "unknown_diagnosis":
            return Trigger("g1_replace", proposal.kind, tuple(vocab.diseases))
        return Trigger("g1_replace", proposal.kind, tuple(vocab.untried(requested) + vocab.diseases))
    return None


def fit(action, thought, event, prompt_tokens, tokenizer, limits=Limits()):
    """Replacement text and trajectory ids, with the proposal's thought or else the fixed one.

    Returns None when neither fits MAX, CTX and COMP (the turn then records replacement_failed).
    """
    stop = len(tokenizer.encode("Observation:", add_special_tokens=False)) if action.kind == "test" else 0
    for candidate in (thought, FIXED_THOUGHT[event]):
        if candidate is None:
            continue
        text = action.completion(candidate)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if action.kind == "diagnosis":
            ids = ids + [tokenizer.eos_token_id]
        generated = len(ids) + stop
        if (generated <= limits.completion and prompt_tokens + len(ids) <= limits.max_length
                and prompt_tokens + generated <= limits.context):
            return text, ids
    return None
