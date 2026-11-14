"""Reader probe conformance (tiicee/clinical/probe.py) on a CPU rollout made as the rollout's own tests
make it: the replay rebuilds every prompt and V0 row, the reading picks the stored hypothesis, and each
kind of departure is counted or refused."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from test_clinical_rollout import adapter, baseline, rollout_stage  # noqa: F401  (fixtures)
from test_clinical_support import approved, context, cpu_stack, fake_evaluate, rollout_environment
from tiicee.clinical import probe, rollout
from tiicee.pipeline import export_guard

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


def digest(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


class Scoring:
    """The engine's part in the reading, over the scripted policy's tokenizer: prompt log probabilities
    in which the continuation of the disease `favoured` names for a prompt's ids is the most likely."""

    def __init__(self, favoured):
        from tim import fixture
        self.tokenizer = fixture.ScriptedPolicy().get_tokenizer()
        self.favoured, self.calls = favoured, []
        self.continuation = {d: self.tokenizer.encode(f"Hypothesis: {d}") for d in DISEASES}

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, requests, sampling_params=None, use_tqdm=False):
        self.calls.append(sampling_params)
        outputs = []
        for request in requests:
            ids = request["prompt_token_ids"]
            (d, tail), = [(d, c) for d, c in self.continuation.items() if ids[-len(c):] == c]
            start, best = len(ids) - len(tail), self.favoured(digest(ids[:-len(tail)]))
            outputs.append(SimpleNamespace(prompt_logprobs=[
                {token: SimpleNamespace(logprob=0.0 if j < start else (-0.1 if d == best else -2.0))}
                for j, token in enumerate(ids)]))
        return outputs


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, values):
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


@pytest.fixture
def rolled(tmp_path, baseline, adapter, monkeypatch):  # noqa: F811
    """A finished CPU rollout of the scripted policy, and the stored hypothesis at each prompt digest."""
    from replication import gpu
    from tim import fixture
    rollout_environment(monkeypatch)
    monkeypatch.setattr(gpu, "evaluate", fake_evaluate(fixture.ScriptedPolicy()))
    ctx = rollout_stage(tmp_path / "rollout", baseline, adapter)
    rollout.run(ctx)
    stored = {step["prompt_ids_sha256"]: probe.stored_hypothesis(step["text"], DISEASES)
              for entry in lines(ctx.out_dir / "steps.jsonl") for step in entry["steps"] if step["role"] == "hypothesis"}
    return SimpleNamespace(out=ctx.out_dir, baseline=baseline, adapter=adapter, stored=stored, gpu=gpu)


def run_probe(tmp_path, rolled, favoured=None, monkeypatch=None):
    engine = Scoring(favoured or (lambda key: rolled.stored.get(key) or DISEASES[0]))
    monkeypatch.setattr(rolled.gpu, "evaluate", fake_evaluate(engine))
    roles = {"split": rolled.baseline / "val.csv", "lab_test_mapping": rolled.baseline / "lab_test_mapping.csv",
             "adapter": rolled.adapter, "steps": rolled.out / "steps.jsonl", "v0_rows": rolled.out / "rows.jsonl"}
    ctx = context(tmp_path / "probe", "p-probe", ["--backbone", "nemotron"], roles, {"policy": 269})
    probe.run(ctx)
    return ctx, approved(ctx), engine


def test_the_stored_hypothesis_is_read_as_the_environment_reads_it():
    assert probe.stored_hypothesis("Hypothesis: Appendicitis\nConfidence: 7", DISEASES) == "appendicitis"
    assert probe.stored_hypothesis("Hypothesis: unsure\nConfidence: 3", DISEASES) is None
    assert probe.stored_hypothesis("Hypothesis: appendicitis \nConfidence: 7", DISEASES) is None  # upstream reads "unknown"
    assert probe.stored_hypothesis("I think appendicitis", DISEASES) is None


@cpu_stack
def test_a_probe_of_a_cpu_rollout_conforms_and_leaves_the_rollout_untouched(tmp_path, rolled, monkeypatch):
    before = {path.name: path.read_bytes() for path in rolled.out.iterdir()}
    ctx, report, engine = run_probe(tmp_path, rolled, monkeypatch=monkeypatch)
    steps = sum(len(entry["steps"]) for entry in lines(rolled.out / "steps.jsonl"))
    assert report["outcome"] == "conforms" and report["rate"] == 1.0
    assert report["steps"] == report["replayed"] == report["rebuilt_identical"] == report["token_digests_identical"] == steps
    assert report["admissions"] == report["v0_identical"] == 24 and report["admissions_with_unused_steps"] == 0
    assert 0 < report["aligned"] < report["hypothesis_steps"]  # the no_hypothesis script answers "unsure"
    assert report["matched"] == report["canonical_tokens"] == report["aligned"]
    assert {key: report[key] for key in ("stage", "backbone", "seed")} == {"stage": "p-probe", "backbone": "nemotron", "seed": 269}
    export_guard.check_aggregate(report)
    scores = lines(ctx.out_dir / "scores.jsonl")
    assert len(scores) == report["hypothesis_steps"] and all(set(s["logprobs"]) == set(DISEASES) for s in scores)
    (params,) = engine.calls  # one batch, scoring only
    assert params == {"max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1}
    assert {path.name: path.read_bytes() for path in rolled.out.iterdir()} == before  # the probe never writes a trajectory


@cpu_stack
def test_a_reading_that_disagrees_on_more_than_one_percent_is_investigated(tmp_path, rolled, monkeypatch):
    aligned = sorted(key for key, value in rolled.stored.items() if value)
    flipped = set(aligned[:max(1, len(aligned) // 50)])  # more than one percent of aligned steps

    def favoured(key):
        right = rolled.stored.get(key) or DISEASES[0]
        return DISEASES[(DISEASES.index(right) + 1) % 4] if key in flipped else right

    _, report, _ = run_probe(tmp_path, rolled, favoured, monkeypatch)
    assert report["outcome"] == "investigate" and report["matched"] == report["aligned"] - len(flipped)
    assert report["rate"] < probe.RATE


@cpu_stack
@pytest.mark.parametrize("field, counted", [("prompt", "rebuilt_identical"), ("prompt_ids_sha256", "token_digests_identical")])
def test_a_logged_prompt_or_digest_the_replay_does_not_reproduce_is_investigated(tmp_path, rolled, monkeypatch, field, counted):
    entries = lines(rolled.out / "steps.jsonl")
    step = next(s for s in entries[0]["steps"] if s["role"] == "decision")
    step[field] = step[field] + "x" if field == "prompt" else digest([0])
    write(rolled.out / "steps.jsonl", entries)
    _, report, _ = run_probe(tmp_path, rolled, monkeypatch=monkeypatch)
    assert report["outcome"] == "investigate" and report[counted] == report["steps"] - 1
    assert report["rate"] == 1.0  # a decision prompt: the reading is unaffected


@cpu_stack
def test_a_replayed_row_that_differs_from_the_stored_one_is_investigated(tmp_path, rolled, monkeypatch):
    rows = lines(rolled.out / "rows.jsonl")
    rows[5]["reward"] += 1.0
    write(rolled.out / "rows.jsonl", rows)
    _, report, _ = run_probe(tmp_path, rolled, monkeypatch=monkeypatch)
    assert report["outcome"] == "investigate" and report["v0_identical"] == 23


@cpu_stack
def test_a_logged_step_the_replay_never_asks_for_is_investigated(tmp_path, rolled, monkeypatch):
    entries = lines(rolled.out / "steps.jsonl")
    entries[3]["steps"].append(dict(entries[3]["steps"][-1]))
    write(rolled.out / "steps.jsonl", entries)
    _, report, _ = run_probe(tmp_path, rolled, monkeypatch=monkeypatch)
    assert report["outcome"] == "investigate" and report["admissions_with_unused_steps"] == 1


@cpu_stack
def test_a_log_missing_a_generation_or_an_admission_is_an_error(tmp_path, rolled, monkeypatch):
    entries = lines(rolled.out / "steps.jsonl")
    entries[3]["steps"].pop()
    write(rolled.out / "steps.jsonl", entries)
    with pytest.raises(RuntimeError, match="more generations than the rollout logged"):
        run_probe(tmp_path, rolled, monkeypatch=monkeypatch)
    write(rolled.out / "steps.jsonl", entries[:-1])
    with pytest.raises(ValueError, match="prompt log does not cover"):
        run_probe(tmp_path / "again", rolled, monkeypatch=monkeypatch)
