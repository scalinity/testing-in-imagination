"""Synthetic stand-in for the baseline work directory, built by tim's own writers.

Layout matches the parts tim reads: prepared/{train,val,test}.csv, lab_test_mapping.csv and
the official primary rows at episodes/parity/269/primary-test.json. Histories and scripted
generations carry a canary marker; each patient follows one scripted behaviour (KINDS) so
every decision-table path of SPEC §3.3 occurs. Everything is invented and deterministic.
"""
import json
from pathlib import Path
import re
from types import SimpleNamespace

from tim.enums import DISEASES

CANARY = "TIM-CANARY-5e1c"
SIZES = {"train": 48, "val": 24, "test": 24}
FIRST_ID = {"train": 90000001, "val": 90001001, "test": 90002001}
MAX, COMP = 4096, 256
KINDS = ("direct", "wrong", "malformed", "repeat", "unknown_action", "unknown_test", "unknown_diagnosis",
         "exhaust", "overflow", "obs_overflow", "truncated", "too_long", "no_hypothesis")
CUES = {
    "appendicitis": "right lower quadrant pain near McBurney point",
    "cholecystitis": "right upper quadrant pain with a Murphy sign",
    "diverticulitis": "left lower quadrant pain over the sigmoid",
    "pancreatitis": "epigastric pain radiating to the back with high lipase",
}
LAB_TESTS = ("Complete Blood Count", "Basic Metabolic Panel", "Comprehensive Metabolic Panel",
             "Renal Function Panel", "Liver Function Panel", "Urinalysis", "Electrolyte Panel")
TESTS = ("Physical Examination", "CT", "MRI", "Radiograph", "Ultrasound", *LAB_TESTS)
# Prompt length targets (MockTokenizer tokens) for the length paths.
PADDED = {"overflow": 3950, "obs_overflow": 3950, "too_long": 4092}


def kind_of(index: int) -> str:
    return KINDS[index % len(KINDS)]


def _cue(index: int, label: str, kind: str) -> str:
    return DISEASES[(DISEASES.index(label) + 1) % 4] if kind == "wrong" else label


def _tokenizer():
    from replication.shim import MockTokenizer, ensure_pad_token
    return ensure_pad_token(MockTokenizer())


def _template_tokens(cfg) -> int:
    """Tokens of the first decision prompt around an empty history, hypothesis line included."""
    from replication.shim import prepare_environment
    env = prepare_environment(cfg)
    item = env.construct_prompt([{"patient_history": ""}])[0]
    text = _tokenizer().apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
    text = text.replace("{first_hypothesis}", "\n\nThe current hypothesis is: appendicitis (confidence: 7)\n")
    return len(_tokenizer().encode(text))


def _record(split: str, index: int, template: int) -> dict:
    label, kind = DISEASES[index % 4], kind_of(index)
    cue = _cue(index, label, kind)
    history = f"Invented admission {CANARY}: two days of abdominal pain with {CUES[cue]}. [script:{kind}]"
    if kind in PADDED:
        history += " filler" * (PADDED[kind] - template - len(_tokenizer().encode(history)))
    findings = " ".join(["Invented"] + ["finding"] * (400 if kind == "obs_overflow" else 8))
    radiology = [{"Modality": "CT", "Report": f"TECHNIQUE:\nInvented.\nFINDINGS:\n{findings}\nIMPRESSION:\nNone."}]
    if index % 3 == 0:
        radiology.append({"Modality": "Ultrasound", "Report": "FINDINGS:\nInvented sonographic finding.\n"})
    labs = {50001 + 10 * j: f"{(index + j) % 9 + 1}.0 invented units" for j in range(7) if (index + j) % 3}
    exam = "not available.\n" if index % 5 == 4 else f"Invented examination consistent with {CUES[cue]}."
    return {"Patient ID": FIRST_ID[split] + index, "Patient History": history, "Patient History Summary": history,
            "Physical Examination": exam, "Laboratory Tests": repr(labs), "Radiology": repr(radiology),
            "Label": label}


def build(baseline_dir: Path) -> Path:
    import pandas as pd
    from replication.config import compose, stream_seed
    from replication.data import load_dataset
    from replication.shim import _evaluate_with_model, generation_settings
    baseline_dir = Path(baseline_dir)
    prepared = baseline_dir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    mapping = []
    for j, panel in enumerate(LAB_TESTS):
        mapping.append({"itemid": 50000 + 10 * j, "label": panel, "corresponding_ids": repr([50001 + 10 * j])})
        mapping.append({"itemid": 50001 + 10 * j, "label": f"Invented measure {j + 1}",
                        "corresponding_ids": repr([50001 + 10 * j])})
    pd.DataFrame(mapping).to_csv(prepared / "lab_test_mapping.csv", index=False)
    cfg = compose("primary", prepared, baseline_dir / "models/parity/269/primary", 269)
    template = _template_tokens(cfg)
    for split, size in SIZES.items():
        frame = pd.DataFrame([_record(split, i, template) for i in range(size)])
        frame.to_csv(prepared / f"{split}.csv", index=False)
    cfg.training.seed = stream_seed(stream_seed(269, "test", "policy"), "test", "evaluation")
    rows = _evaluate_with_model(cfg, load_dataset(prepared, "test", cfg), ScriptedPolicy(), generation_settings(False))
    path = baseline_dir / "episodes/parity/269/primary-test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": "synthetic", "payload": rows}, sort_keys=True))
    return baseline_dir


class ScriptedPolicy:
    """Deterministic CPU stand-in for the vLLM engine, keyed by each history's script tag."""

    def __init__(self):
        self.tokenizer = _tokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def _completion(self, kind, turn, cue, prompt_tokens):
        thought = f"{CANARY} weigh the findings."
        action = None
        if kind in ("direct", "wrong"):
            action = (("Test", "CT"), ("Test", "Physical Examination"), ("Diagnosis", cue))[min(turn, 2)]
        elif kind in ("repeat", "obs_overflow"):
            action = ("Test", "CT")
        elif kind == "unknown_action":
            action = ("Consult", "Surgery")
        elif kind == "unknown_test":
            action = ("Test", "Brain MRI")
        elif kind == "unknown_diagnosis":
            action = ("Diagnosis", "gastritis")
        elif kind == "exhaust":
            action = ("Test", TESTS[turn] if turn < len(TESTS) else "CT")
        elif kind == "overflow":
            thought += " reasoning" * (MAX - prompt_tokens + 10)
            action = ("Diagnosis", cue)
        elif kind == "truncated":
            thought += " reasoning" * 300
            action = ("Diagnosis", cue)
        elif kind == "too_long":
            action = ("Diagnosis", cue)
        if action is None:
            return f"Thought: {thought}\nI cannot decide yet.\n", False
        text = f"Thought: {thought}\nAction: {action[0]}\nAction Input: {action[1]}"
        return (text, True) if action[0] == "Diagnosis" else (text + "\n", False)

    def _decision(self, prompt):
        observed = prompt.split("Here is some context on the patient:", 1)[1]
        kind = re.search(r"\[script:(\w+)\]", observed).group(1)
        cue = max(CUES, key=lambda disease: observed.count(CUES[disease]))
        prompt_tokens = len(self.tokenizer.encode(prompt))
        text, diagnosis = self._completion(kind, observed.count("\nAction: "), cue, prompt_tokens)
        ids = self.tokenizer.encode(text)
        if len(ids) > COMP:
            end = [m.end() for m in re.finditer(r"\S+|\n", text)][COMP - 1]
            return text[:end], ids[:COMP]
        stop = [self.tokenizer.eos_token_id] if diagnosis else self.tokenizer.encode("Observation:")
        return text, ids + stop

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        outputs = []
        for prompt in prompts:
            if prompt.endswith("Confidence: "):
                text = "7"
                ids = self.tokenizer.encode(text)
            elif "Here is the available information on the patient:" in prompt:
                observed = prompt.split("Here is the available information on the patient:", 1)[1]
                cue = max(CUES, key=lambda disease: observed.count(CUES[disease]))
                text = "Hypothesis: unsure\nConfidence: 3" if "[script:no_hypothesis]" in observed \
                    else f"Hypothesis: {cue}\nConfidence: 7"
                ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            else:
                text, ids = self._decision(prompt)
            outputs.append(SimpleNamespace(prompt=prompt, prompt_token_ids=self.tokenizer.encode(prompt),
                                           outputs=[SimpleNamespace(text=text, token_ids=ids)]))
        return outputs


class MockScorer:
    """Deterministic CPU scorer: a disease scores how often the prompt names it; a test scores 0.5."""

    def __init__(self, tokenizer):
        self.tokenizer, self.calls = tokenizer, 0

    def next_token(self, prompt_ids):
        return -1  # no language model on CPU; bench agreement is measured on GPU only

    def score(self, prompt_ids, thought, candidates, path=None):
        self.calls += 1
        return [float(list(prompt_ids).count(self.tokenizer.encode(a.name)[0])) if a.kind == "diagnosis" else 0.5
                for a in candidates]
