"""The two backbones: frozen names and revisions, the Granite adaptation, and the tokenizer facts.

Granite cannot be downloaded here, so its tokenizer and chat template are stood in for by stubs
with the behaviours the adaptation has to handle; the same checks run on the real tokenizer inside
the bench and every rollout on the cluster.
"""
import sys
import time
from datetime import datetime
from types import ModuleType, SimpleNamespace

import pytest

from test_clinical_support import context, cpu_stack, fake_train, prepared
from tiicee.clinical import ROOT, backbone, folds, train, use_prior_studies
from tiicee.clinical.backbone import BACKBONES

use_prior_studies()
from replication import gpu, shim  # noqa: E402  (import with the standard library alone)

CARDS = ROOT / "reports" / "research-advancement" / "EXPERIMENT_CARDS"
NEMOTRON, GRANITE = BACKBONES["nemotron"], BACKBONES["granite"]
FIXED = "Knowledge Cutoff Date: April 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant."


class TokenizerBase:
    """transformers.PreTrainedTokenizerBase's part in this: a chat template that takes options."""

    eos_token, eos_token_id, pad_token_id = "<|end_of_text|>", 0, 0

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **options):
        self.options = options
        messages = list(conversation)
        if messages[0]["role"] != "system":  # Granite's default: a system prompt carrying today's date
            default = "Knowledge Cutoff Date: April 2024.\nToday's Date: " + self.today() + ".\nYou are Granite, developed by IBM."
            default += " Write your thoughts between <think></think>." if options.get("thinking") else " You are a helpful AI assistant."
            messages.insert(0, {"role": "system", "content": default})
        text = "".join(f"<|start_of_role|>{m['role']}<|end_of_role|>{m['content']}<|end_of_text|>\n" for m in messages)
        return text + ("<|start_of_role|>assistant<|end_of_role|>" if add_generation_prompt else "")

    def today(self):
        return time.strftime("%B %d, %Y")

    def encode(self, text, add_special_tokens=True):
        return [self.eos_token_id if word == self.eos_token else sum(word.encode()) + 1
                for word in text.replace(self.eos_token, " " + self.eos_token).replace("\n", " \n ").split(" ") if word]


@pytest.fixture
def transformers(monkeypatch):
    module = ModuleType("transformers")
    module.PreTrainedTokenizerBase = TokenizerBase
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


def cfg(system):
    return SimpleNamespace(environment=SimpleNamespace(system_prompt=system))


def test_backbones_are_the_ones_the_cards_froze():
    for card, chosen in (("E8.md", NEMOTRON), ("E9.md", GRANITE)):
        text = " ".join((CARDS / card).read_text().split())
        assert chosen.model in text and chosen.revision in text
    assert (NEMOTRON.model, backbone.ARM) == (shim.APPROVED_MODEL, "primary") and NEMOTRON.overlay is None
    overlay = GRANITE.overlay.read_text()
    assert GRANITE.overlay.parent == ROOT / "tii-cee" / "clinical_configs" / "backbone"
    assert f"model_name: {GRANITE.model}\n" in overlay and f"revision: {GRANITE.revision}\n" in overlay
    assert "detailed thinking off" not in overlay.split("system_prompt:")[1]


def test_granite_is_allowed_only_inside_its_runtime(transformers):
    with pytest.raises(ValueError):
        shim.validate_model(GRANITE.model, GRANITE.revision)
    with backbone.runtime(GRANITE, cfg(FIXED)):
        shim.validate_model(GRANITE.model, GRANITE.revision)  # the pinned loaders' check
        with pytest.raises(ValueError):
            shim.validate_model(NEMOTRON.model, NEMOTRON.revision)
        with pytest.raises(ValueError):
            shim.validate_model("ibm-granite/granite-3.3-2b-instruct", GRANITE.revision)
    shim.validate_model(NEMOTRON.model, NEMOTRON.revision)
    assert (shim.APPROVED_MODEL, shim.SYSTEM_MESSAGE) == (NEMOTRON.model, "detailed thinking off")


def test_granite_gets_the_fixed_system_message_and_never_nemotrons_switch(transformers):
    user = [{"role": "user", "content": "decide"}]
    with backbone.runtime(GRANITE, cfg(FIXED)):
        messages = shim._with_system(user)  # what the replication's environment puts before every agent turn
        hypothesis = shim._SystemTokenizer(TokenizerBase()).apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    assert messages == [{"role": "system", "content": FIXED}, *user]
    assert FIXED in hypothesis and "detailed thinking off" not in hypothesis and "Today's Date" not in hypothesis
    assert shim._with_system(user)[0]["content"] == "detailed thinking off"  # restored for Nemotron


def test_granite_templates_are_applied_with_reasoning_off_on_every_path(transformers):
    tokenizer, user = TokenizerBase(), [{"role": "user", "content": "decide"}]
    original = TokenizerBase.apply_chat_template
    with backbone.runtime(GRANITE, cfg(FIXED)):
        tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        assert tokenizer.options == {"thinking": False}
        shim._SystemTokenizer(tokenizer).apply_chat_template(user, tokenize=False)
        assert tokenizer.options == {"thinking": False}
        tokenizer.apply_chat_template(user, thinking=True)  # an explicit choice is passed through
        assert tokenizer.options == {"thinking": True}
    tokenizer.apply_chat_template(user)
    assert tokenizer.options == {} and TokenizerBase.apply_chat_template is original


def test_nemotron_runs_on_the_replications_constants_unchanged():
    with backbone.runtime(NEMOTRON, cfg("detailed thinking off")):
        assert (shim.APPROVED_MODEL, shim.SYSTEM_MESSAGE) == (NEMOTRON.model, "detailed thinking off")
    with pytest.raises(ValueError, match="frozen recipe"):
        with backbone.runtime(NEMOTRON, cfg("some other system message")):
            pass


def test_tokenizer_facts_on_a_well_behaved_tokenizer():
    facts = backbone.tokenizer_facts(TokenizerBase(), FIXED)
    assert facts["observation_stop_ids"] == TokenizerBase().encode("Observation:") and facts["observation_stop_strips"]
    assert facts["pad_is_eos"] and facts["label_ends_with_eos"] and facts["assistant_turn_ends_with_eos"]
    assert facts["system_message_sent"] and not facts["date_strings"]
    assert facts["fixed_is_default_without_date"] and not facts["thinking_changes"]
    assert facts["clock_changes"] in (None, False)  # None without transformers: the check reports itself unavailable


def test_tokenizer_facts_flag_what_the_adaptation_must_handle():
    class MergesNewline(TokenizerBase):  # "\nObservation:" becomes one token: upstream's strip would not match
        def encode(self, text, add_special_tokens=True):
            return super().encode(text.replace("\nObservation:", " <nl-observation>"))

    class SeparatePad(TokenizerBase):
        pad_token_id = 7

    class OtherTerminator(TokenizerBase):  # the template ends a turn with a token that is not EOS
        def apply_chat_template(self, conversation, **options):
            return super().apply_chat_template(conversation, **options).replace("<|end_of_text|>", "<|eot|>")

    class ThinkingWithSystem(TokenizerBase):
        def apply_chat_template(self, conversation, **options):
            return super().apply_chat_template(conversation, **options) + ("<think>" if options.get("thinking") else "")

    assert not backbone.tokenizer_facts(MergesNewline(), FIXED)["observation_stop_strips"]
    assert not backbone.tokenizer_facts(SeparatePad(), FIXED)["pad_is_eos"]
    assert not backbone.tokenizer_facts(OtherTerminator(), FIXED)["assistant_turn_ends_with_eos"]
    assert backbone.tokenizer_facts(ThinkingWithSystem(), FIXED)["thinking_changes"]
    assert not backbone.tokenizer_facts(TokenizerBase(), "Some other fixed message.")["fixed_is_default_without_date"]

    class DatedAlways(TokenizerBase):  # a template that writes the date even with a system message
        def apply_chat_template(self, conversation, **options):
            return "Today Date: " + time.strftime("%d %b %Y") + "\n" + super().apply_chat_template(conversation, **options)

    assert backbone.tokenizer_facts(DatedAlways(), FIXED)["date_strings"]


def test_the_clock_check_moves_the_templates_clock(monkeypatch):
    utils, module = ModuleType("transformers.utils"), ModuleType("transformers.utils.chat_template_utils")
    module.datetime, utils.chat_template_utils = datetime, module
    for name, value in (("transformers", ModuleType("transformers")), ("transformers.utils", utils),
                        ("transformers.utils.chat_template_utils", module)):
        monkeypatch.setitem(sys.modules, name, value)

    class Dated(TokenizerBase):  # reads the clock the way transformers' strftime_now does, in an unusual format
        def apply_chat_template(self, conversation, **options):
            return module.datetime.now().strftime("%A week %W") + super().apply_chat_template(conversation, **options)

    assert backbone.tokenizer_facts(Dated(), FIXED)["clock_changes"] is True
    assert backbone.tokenizer_facts(TokenizerBase(), FIXED)["clock_changes"] is False
    assert module.datetime is datetime  # the template's clock is put back


@cpu_stack
def test_the_granite_overlay_changes_the_backbone_and_nothing_else_in_the_recipe(tmp_path):
    from omegaconf import OmegaConf

    def flat(value, prefix=""):
        return {k: v for key, item in value.items() for k, v in
                (flat(item, f"{prefix}{key}.") if isinstance(item, dict) else {prefix + key: item}).items()}

    data, out = tmp_path / "data", tmp_path / "out"
    primary, granite = (flat(OmegaConf.to_container(backbone.compose(chosen, data, out, 270), resolve=True))
                        for chosen in (NEMOTRON, GRANITE))
    assert {key for key in primary if primary[key] != granite[key]} == \
        {"model.model_name", "model.revision", "environment.system_prompt"} and set(primary) == set(granite)
    assert (granite["model.model_name"], granite["model.revision"]) == (GRANITE.model, GRANITE.revision)
    assert granite["environment.system_prompt"] == FIXED and granite["training.seed"] == 270
    assert primary["environment.system_prompt"] == "detailed thinking off"
    with pytest.raises(ValueError, match="approved"):  # the replication's own allowlist is untouched outside compose
        gpu._configuration(backbone.compose(GRANITE, data, out, 270), data, 270)


@cpu_stack
def test_granite_training_runs_inside_the_adaptation_and_leaves_no_trace(tmp_path, monkeypatch, transformers):
    source = prepared(tmp_path)
    monkeypatch.setenv("CDM_WORK_DIR", str(source.parent))
    data_ctx = context(tmp_path, "E9_data")
    folds.run_data(data_ctx)
    seen = {}

    def during(step):
        if step == 1:
            from replication.shim import prepare_environment
            seen["model"], seen["system"] = shim.APPROVED_MODEL, shim.SYSTEM_MESSAGE
            environment = prepare_environment(backbone.compose(GRANITE, data_ctx.out_dir, tmp_path / "x", 269))
            seen["prompt"] = environment.construct_prompt([{"patient_history": "invented"}])[0]["prompt"]

    roles = {"data": data_ctx.out_dir}
    with pytest.raises(ValueError, match="backbone_overlay"):  # the overlay must be among the stage's hashed inputs
        train.run(context(tmp_path, "unbound", ["--backbone", "granite"], roles, {"training": 269}))
    with pytest.raises(ValueError, match="backbone_overlay"):
        train.run(context(tmp_path, "misbound", ["--backbone", "granite"], {**roles, "backbone_overlay": data_ctx.out_dir}, {"training": 269}))
    roles["backbone_overlay"] = GRANITE.overlay
    ctx = context(tmp_path, "E9_train_269", ["--backbone", "granite"], roles, {"training": 269})
    monkeypatch.setattr(gpu, "train", fake_train(during))  # its gpu._configuration call accepts Granite here
    train.run(ctx)
    assert (seen["model"], seen["system"]) == (GRANITE.model, FIXED)
    assert seen["prompt"][0] == {"role": "system", "content": FIXED} and len(seen["prompt"]) == 2
    assert "detailed thinking off" not in str(seen["prompt"])
    assert (shim.APPROVED_MODEL, shim.SYSTEM_MESSAGE) == (NEMOTRON.model, "detailed thinking off")
    import json
    policy = json.loads((ctx.out_dir / "policy.json").read_text())
    assert (policy["backbone"], policy["revision"], policy["recipe"]) == (GRANITE.model, GRANITE.revision, "primary")
    assert json.loads((ctx.out_dir / "model" / "adapter" / "adapter_config.json").read_text())["revision"] == GRANITE.revision
