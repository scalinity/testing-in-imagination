# replication-of-original/

Independent replications of the original **LA-CDM** paper — Bani-Harouni et al., ICLR 2026 ([arXiv:2506.13474](https://arxiv.org/abs/2506.13474) · [code](https://github.com/dharouni/LA-CDM)). Each team member runs the paper on their **own** PhysioNet-credentialed access; we compare results later against the paper and against each other.

The shared claims ledger is [`../REPRODUCTION.md`](../REPRODUCTION.md) (claim IDs T1–T8, B1–B2, C1–C4, I1–I5; findings D-1…D-5, F-1…F-4). Reference those IDs in your notes so results line up.

## Layout

One folder per person, `lowercase-hyphen` name:

```
replication-of-original/
├── README.md            # this file
├── .gitignore           # blocks data / checkpoints / raw generations
├── daniel-escalante/
├── hugo-resendez/
└── edwin-salcedo/
```

Suggested structure inside your folder:

```
<your-name>/
├── README.md        # short summary + run log table (below)
├── RUN_LOG.md       # or notes.md — what you ran, what broke, what you changed
├── configs/         # Hydra overrides / config files (no data paths, no secrets)
├── scripts/         # launch scripts, SLURM job files, small helpers
└── results/         # small metric files only (*.json, *.md, short logs)
```

## Rules

1. **Only edit your own folder.** Don't change teammates' folders or `lacdm/upstream`.
2. **Never commit restricted data.** No MIMIC-IV / MIMIC-IV-Ext CDM files, patient-level outputs, model generations containing patient text, or credentials/tokens. Each person must use their own PhysioNet-credentialed access; restricted data can't be shared (PhysioNet DUA). Keep data paths in local untracked config or env vars (e.g. `.env`, which is gitignored).
3. **No large artifacts.** Don't commit checkpoints, LoRA weights, or W&B dumps — record the path / W&B run id instead.
4. **Running the GitHub code ≠ reproduction.** A run counts as *reproduced* only when it matches a paper claim under a logged **config, seed, upstream commit hash, model, hardware, and metrics**. Record gaps honestly; never invent or round toward paper numbers.
5. **Upstream pin:** this repo pins `dharouni/LA-CDM` as submodule `lacdm/upstream` @ `3f435a10` (`3f435a10f569c3255172c1ef8fe7880c6edb7fee`). Log the commit you actually ran.
6. **Backbone (D-5):** on HiPerGator use `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. Chinese-origin models (Qwen, DeepSeek, Yi, InternLM, and derivatives) are **not allowed** — the paper's `Qwen/Qwen2.5-7B-Instruct` default must be overridden everywhere (incl. `prepare_data.py`, see D-3). Record any deviation from the paper's model in your log.

## Run log template

Copy into your folder's `README.md` or `RUN_LOG.md`, one row per run/metric:

| Date | Person | Upstream commit | Model | Config | Seed | Dataset version | Metric | Value | Paper value | Notes |
|------|--------|-----------------|-------|--------|------|-----------------|--------|-------|-------------|-------|
| 2026-MM-DD | | `3f435a10` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | `configs/…` | 269 | MIMIC-IV-Ext CDM v… | Mean Acc (T5) | | 81.3 | |

Also note hardware (GPU, CUDA), split seed (D-4), and W&B run id where relevant.

## Workflow

- `git pull` before you start and before you push.
- Work on a branch and open a PR, or commit directly to `main` — either is fine as long as you only touch your own folder.
- Clear commit messages, e.g. `hugo-resendez: zero-shot eval, seed 269, Nemotron 8B`.
- Run `git status` before committing to make sure no data or outputs are staged.
