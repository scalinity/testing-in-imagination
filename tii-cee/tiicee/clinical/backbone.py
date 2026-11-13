"""The two backbones of the training track, and the E9 adaptation of the replication to Granite.

The replication names Nemotron in three places that are not hyperparameters: the model allowlist
(replication.config.MODEL, replication.shim.APPROVED_MODEL), the system message
(replication.shim.SYSTEM_MESSAGE, "detailed thinking off") and the Hydra backbone overlay. None is
edited. For Granite, and only while a stage that asked for Granite is inside `runtime`:
  - replication.shim.APPROVED_MODEL is the Granite name, so the replication's pinned loaders accept
    Granite at its pinned revision and nothing else (Nemotron included);
  - replication.shim.SYSTEM_MESSAGE is the overlay's fixed system message, so "detailed thinking
    off" is never sent and Granite's chat template never writes its dated default system prompt;
  - transformers' apply_chat_template receives thinking=False unless a caller passes it, which is
    Granite's reasoning mode off on every path (trainer, hypothesis agent, evaluator).
These are replacements of module attributes in this process, undone when `runtime` exits.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from tiicee.clinical import CONFIGS, REPLICATION, ROOT, TIM, bound, sha256, use_prior_studies

ARM = "primary"  # the recipe of EXPERIMENT_CARDS/E8.md; no other arm is trained here
# The default the replication's and tim's launchers export for every GPU stage (replication
# scripts/stage.sh, tim scripts/gpu.slurm). The pipeline's job script does not, so GPU stages set it.
ALLOCATOR = ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass(frozen=True)
class Backbone:
    key: str
    model: str
    revision: str
    overlay: Path | None  # None: the replication's own composition


BACKBONES = {b.key: b for b in (
    Backbone("nemotron", "nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "54641c1611fcff44fa4865626462445e0a153fc7", None),
    Backbone("granite", "ibm-granite/granite-3.3-8b-instruct", "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
             CONFIGS / "backbone" / "granite_3_3_8b.yaml"),
)}


def compose(backbone, data_dir, output_dir, seed):
    """The replication's "primary" configuration; for Granite, with the backbone overlay on top."""
    use_prior_studies()
    from replication import config
    cfg = config.compose(ARM, data_dir, output_dir, seed)
    if backbone.overlay is not None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(cfg, OmegaConf.load(backbone.overlay))
        # The replication's own check of the resolved configuration, with its allowlist naming
        # this backbone: still an exact match, still refusing the banned families.
        with patch.object(config, "MODEL", backbone.model):
            config.check_resolved_models(cfg)
    if (cfg.model.model_name, cfg.model.revision) != (backbone.model, backbone.revision):
        raise ValueError("the composed configuration does not name the requested backbone and revision")
    return cfg


@contextmanager
def runtime(backbone, cfg):
    """Run the replication's GPU entry points for this backbone (see the module docstring)."""
    use_prior_studies()
    from replication import shim
    system = cfg.environment.system_prompt
    if backbone.overlay is None:
        if (shim.APPROVED_MODEL, shim.SYSTEM_MESSAGE) != (backbone.model, system):
            raise ValueError("the replication's backbone constants differ from the frozen recipe")
        yield
        return
    from transformers import PreTrainedTokenizerBase
    original = PreTrainedTokenizerBase.apply_chat_template

    def apply_chat_template(self, conversation, *args, **kwargs):
        kwargs.setdefault("thinking", False)
        return original(self, conversation, *args, **kwargs)

    with patch.object(shim, "APPROVED_MODEL", backbone.model), patch.object(shim, "SYSTEM_MESSAGE", system), \
            patch.object(PreTrainedTokenizerBase, "apply_chat_template", apply_chat_template):
        yield


def chosen(ctx, key):
    """The backbone a stage asked for. A Granite stage's manifest must bind the overlay (role
    "backbone_overlay"), so the file that names the backbone, its revision and the system message
    is hashed with the stage's other inputs."""
    backbone = BACKBONES[key]
    if backbone.overlay is not None:
        entry = ctx.manifest["bound"].get("backbone_overlay")
        if entry is None or bound(ctx, "backbone_overlay").resolve() != backbone.overlay.resolve():
            raise ValueError("a Granite stage must bind the backbone overlay as role backbone_overlay")
    return backbone


def require_frozen_code():
    """Refuse to run on changed prior-study code: the upstream pin check that
    replication.manifest.make_manifest makes before every run of the replication, and no tracked
    change in the replication, tim or the lacdm overlay."""
    use_prior_studies()
    from replication.config import UPSTREAM, UPSTREAM_SHA
    from replication.manifest import command
    if command(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"]) != UPSTREAM_SHA \
            or command(["git", "-C", str(UPSTREAM), "status", "--porcelain"]):
        raise RuntimeError("lacdm/upstream is not at its pin, or has local changes")
    if command(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no", "--",
                str(REPLICATION), str(TIM), str(ROOT / "lacdm")]):
        raise RuntimeError("the replication, tim or lacdm has tracked changes")


def recipe_record(cfg, backbone, seed) -> dict:
    """Identity of what a stage ran: recipe arm, backbone and revision, seed, prompts, configuration."""
    from omegaconf import OmegaConf
    from replication.config import UPSTREAM_SHA
    from replication.manifest import runtime_info
    templates = (cfg.environment.prompt_template_file, cfg.environment.hypothesis_prompt_template_file)
    return {"recipe": ARM, "backbone": backbone.model, "revision": backbone.revision, "seed": seed,
            "upstream_sha": UPSTREAM_SHA,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "UNSET"),
            "templates": {Path(path).name: sha256(Path(path).read_bytes()) for path in templates},
            "config": OmegaConf.to_container(cfg, resolve=True), "runtime": runtime_info()}


def _clock_changes(render):
    """Whether a rendered prompt depends on the clock, by rendering again with the template's
    clock moved. None when transformers' template clock is not where 4.51 keeps it; date_strings
    then stands alone."""
    try:
        from transformers.utils import chat_template_utils as module
    except ImportError:
        return None
    clock = getattr(module, "datetime", None)
    if not isinstance(clock, type):
        return None

    class Moved(clock):
        @classmethod
        def now(cls, tz=None):
            return cls(2001, 2, 3, 4, 5, 6)

    with patch.object(module, "datetime", Moved):
        moved = render()
    return moved != render()


def tokenizer_facts(tokenizer, system_message) -> dict:
    """What upstream's and the replication's token handling assume of a tokenizer, measured on it
    (EXPERIMENT_CARDS/E9.md, adaptation items 2 and 3). Ids and flags only.

    observation_stop_strips: upstream removes a trailing "Observation:" from a completion's ids by
    comparing them with tokenizer.encode("Observation:") (environment.py, Trajectory.update_output);
    the comparison only matches if the word encodes the same way after a newline.
    """
    def render(messages, **options):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)

    def ids(text):
        return list(tokenizer.encode(text, add_special_tokens=False))

    turn = [{"role": "system", "content": system_message}, {"role": "user", "content": "x"}]
    prompt, stop, eos = render(turn), ids("Observation:"), tokenizer.eos_token_id
    closed = tokenizer.apply_chat_template(turn + [{"role": "assistant", "content": "y"}], tokenize=False,
                                           add_generation_prompt=False)
    now = datetime.now()
    dates = ("Today's Date", "Today Date", *(now.strftime(form) for form in ("%B %d, %Y", "%d %b %Y", "%Y-%m-%d")))
    return {
        "observation_stop_ids": stop,
        "observation_stop_strips": bool(stop) and ids("Action Input: CT\nObservation:")[-len(stop):] == stop,
        "eos_id": eos, "pad_id": tokenizer.pad_token_id, "pad_is_eos": tokenizer.pad_token_id == eos,
        "label_ends_with_eos": ids("Hypothesis: appendicitis\nConfidence: 0" + tokenizer.eos_token)[-1:] == [eos],
        "assistant_turn_ends_with_eos": closed.rstrip().endswith(tokenizer.eos_token),
        "system_message_sent": system_message in prompt,
        "date_strings": any(text in prompt for text in dates),
        "clock_changes": _clock_changes(lambda: render(turn)),
        "thinking_changes": render(turn, thinking=True) != prompt,
        "fixed_is_default_without_date": system_message in re.sub(r"Today's Date: [^\n]*\n", "", render(turn[1:])),
    }
