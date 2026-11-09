# Decisions

Implementation choices that SPEC.md leaves open, and why. SPEC.md wins on any conflict.

## Execution

- **Stages run on HiPerGator through the agent's SSH session** (SPEC §1.1). The agent reads only published bundles, SLURM job metadata and error codes; private logs, rows, HPI text, generations, inventories and world-model files are never read into the agent's conversation, which the PhysioNet DUA treats as a third-party service.
- **One RTX PRO 6000 per GPU job** (partition hpg-rtx6000, `gpu:rtx_pro_6000:1`), whichever course account the job is billed to. Greedy bf16 decoding is not bit-stable across GPU architectures, so mixing hardware would break the V0-rerun identity (SPEC §5.6) and the shadow control (SPEC §5.5).
- **Accounts:** `cai6734` and `cai5724` each allow 8 GPUs; a stage's jobs are spread across the accounts in `SLURM_ACCOUNTS`. `cai5720` and `cai5731` have no GPU allowance. CPU stages use a burst QOS (`SLURM_CPU_QOS`).

## Environment

- The CPU environment resolves to exactly the replication's lock, transitive packages included (`constraint-dependencies` in pyproject.toml).
- GPU jobs run on the replication's own runtime (`$CDM_WORK_DIR/runtime-env/bin/python`) with `tim`, the replication and the pinned upstream on `sys.path`; nothing is installed into it.

## Data boundary

- **Schema validation is implemented in `tim.events`** (about 60 lines) rather than through a `jsonschema` dependency, which neither environment may gain. Schema files may only use: objects with `additionalProperties: false` or enumeration-keyed maps (`keys`/`values`), arrays, integers, numbers (finite), booleans, null, `anyOf`, and strings that name a code-defined enumeration (`tim/enums.py`) or the identifier pattern. A schema outside that vocabulary fails to load.
- Validation errors carry the schema path and a rule code, never the rejected value.

## Replication reuse

- SPEC §1.10's list governs every access to `$CDM_WORK_DIR`. Other replication functions that neither write nor touch that directory are reused unchanged where SPEC asks for the replication's behaviour: `stats` (metric code), `manifest.assert_aggregate`, `config.stream_seed` and `config.read_env`, `shim.prepare_environment`, `shim.ensure_pad_token` and the CPU `shim.MockPolicy`. No replication function that writes is called anywhere.
