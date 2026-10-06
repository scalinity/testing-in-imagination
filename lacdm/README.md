# lacdm/

Reproduction scaffold for LA-CDM (Bani-Harouni et al., ICLR 2026). It holds three
things, and none of them is a copy of the upstream code:

| Path | What it is |
|---|---|
| `upstream/` | Git submodule → [dharouni/LA-CDM](https://github.com/dharouni/LA-CDM), pinned (below). Never vendored: the repo has no licence (B-3). |
| `configs/` | Hydra overlay that layers onto `upstream/configs` without editing it. |
| `tests/` | Phase E unit tests that execute the upstream parsers, rewards, environment and metrics on synthetic fixtures. |
| `metric_writer.py` | Writes metrics JSON to `artifacts/`, stamped with `source` and the upstream SHA. |
| `artifacts/` | Local metric output. Gitignored except `.gitkeep`. |

## Upstream pin

`lacdm/upstream` @ **`3f435a10f569c3255172c1ef8fe7880c6edb7fee`** (2026-08-24, `main`
HEAD as of 2026-10-06). It is the same commit as `baseline/LA-CDM`, so every
`file:line` citation in `REPRODUCTION.md` holds for both.

```sh
git submodule update --init lacdm/upstream
```

## Running the tests (Mac)

```sh
uv venv --python 3.12 lacdm/.venv
uv pip install --python lacdm/.venv/bin/python -r lacdm/requirements-test.txt
lacdm/.venv/bin/python -m pytest lacdm/tests
```

Python 3.12 sits inside upstream's `>=3.11,<3.13`, and the test dependencies use
upstream's exact pins where it pins them. `trl` is stubbed in `tests/conftest.py`: its
one use in `environment.py` is chat templating, and importing it pulls in the CUDA
stack.

## Mac vs HiPerGator

| Work | Mac | HiPerGator |
|---|---|---|
| Phase E tests (parsers, rewards, unavailable-test path, metrics) | yes | yes |
| Hydra composition of the overlay | yes | yes |
| Loading the 8B backbone, vLLM rollouts, GRPO training | no: `vllm`, `bitsandbytes`, `deepspeed` and `cupy-cuda12x` have no macOS-arm64 build (B-1) | yes |
| Anything touching MIMIC-IV-Ext-CDM | no (B-2, DUA) | yes, once credentialed |

Nothing in this directory produces a Table 1 number. `artifacts/` files written by the
tests carry `"source": "fixture"`.

## D-5: backbone

The paper and upstream default to `Qwen/Qwen2.5-7B-Instruct`. Qwen, DeepSeek and Yi
models are excluded from this project, so the overlay swaps in
**`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`**:

```sh
cd lacdm/upstream
python -m src.train \
  "hydra.searchpath=[file://$(git rev-parse --show-superproject-working-tree)/lacdm/configs]" \
  +backbone=nemotron_nano_8b
```

`model.model_name` is upstream's only Hydra-level Qwen default. The HF model, the
tokenizer and the vLLM engine all derive from it. The data-prep scripts are not Hydra:
`prepare_data.py` defaults to Mixtral-8x7B, and `create_data_with_summaries.py`
requires `--model`. Qwen appears there only in a docstring and in help text.

The rewards are not overridden. The shipped `incorrect_reward: 0.0` stays, against the
paper's −1.0 (D-1). `tests/test_rewards.py` fails if the overlay ever changes that.

A result on this backbone is a **re-implementation on a different model**, not a
reproduction of Table 1, and must be labelled that way.

### What the overlay cannot do yet

All four items below must be resolved before a HiPerGator run. Each one is a fact
about upstream code or the model's published files, checked on 2026-10-06:

1. **"detailed thinking off" is declared, not applied.** Nemotron toggles reasoning
   with a system message. Upstream sends only user messages
   (`environment.py:164,368`), and it has no system-prompt hook. With no system message,
   Nemotron's chat template emits an *empty* system block, and the model card does not
   say which mode that gives. `environment.system_prompt` needs a launcher shim that
   prepends it to every decision, hypothesis and confidence prompt.
2. **No pad token.** Nemotron's `tokenizer_config.json` has `pad_token: null`.
   Upstream pads policy completions with `processing_class.pad_token_id`
   (`cdm_grpo_trainer.py:451,477,509`), and it only backfills a pad token for *reward*
   tokenizers (`:198-199`). Qwen ships one, so upstream never needed this.
3. **`model.revision` is a placeholder.** Upstream passes no revision to
   `from_pretrained` (`setup.py:53`), `AutoTokenizer` (`cdm_grpo_trainer.py:162`) or
   vLLM (`:287`). The shim has to thread it. The HF `main` SHA observed on 2026-10-06 was
   `54641c1611fcff44fa4865626462445e0a153fc7`. That is a candidate, not yet pinned.
4. **Sampling.** The model card recommends greedy decoding with reasoning off. GRPO
   needs 8 sampled generations per prompt, so training cannot follow that advice.
   Evaluation could.

The LoRA target names (`q,k,v,o,gate,up,down_proj`) are the Llama module names as well,
so the LoRA config carries over unchanged. Licence: NVIDIA Open Model License plus the
Llama 3.1 Community License.
