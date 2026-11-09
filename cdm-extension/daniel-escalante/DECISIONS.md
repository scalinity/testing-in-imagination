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

## Admissions, resume and approvals

- **Run keys.** A test run is keyed by protocol hash, stage, variant and arm only, so no code change can reopen an admitted test run. A validation run's key adds the code commit, so a validation re-run after a fix (SPEC §3.4) is a new run.
- **Resume.** The episode log in `$TIM_WORK_DIR` is checked against per-patient digests in the admission ledger, and the admission marker records the log's path. A changed `TIM_WORK_DIR` is refused (`resume_mismatch`) instead of re-evaluating patients. Upstream's confidence-pair RNG advances once per patient; on resume it restarts, which changes only the discarded confidence-calibration generations, never a row.
- **Approvals gate GPU stages** (SPEC §2): each binds the protocol, addendum and config digests, the job list, a GPU-hour budget, and a code commit with no change since in `tim/`, `scripts/`, `configs/`, `schemas/`, the lock files, the replication package or the upstream submodule. CPU analysis stages are gated by the access table and by completed runs. SLURM time limits are twice the approved per-job GPU-hours plus 30 minutes.
- **Estimates.** Before bench-a, GPU-hours come from the SPEC §9 table. After it, each job is estimated as bench setup time plus the bench per-patient time scaled to 240 patients; V0, V0-rerun and V2 use V3's measurement as an upper bound.

## Guard mechanics

- Candidates are scored with the proposal's thought (the fixed thought when the proposal has none). If the chosen replacement does not fit with that thought, the fixed thought is substituted without rescoring.
- A replacement rewrites the trajectory's text, token mask and completion ids together. Later prompts re-tokenize the whole text, so the mask can drift by a token from the generated ids, exactly as it already does upstream for generated completions.
- **Shadow control and bench.** V0-shadow scores the full candidate set (12 tests and 4 diseases) on every decision turn and discards the result. In bench-a, V0-shadow's scorer runs both the fast and the reference path on every call and checks the HF greedy next token against vLLM's first generated token; V3's bench job uses the fast path alone, so its timing is realistic.
- **`CUBLAS_WORKSPACE_CONFIG=:4096:8` is set in every GPU job**, not only scored ones. It is process-wide, so setting it uniformly keeps V0 and V0-shadow identically configured (SPEC §5.5). Any effect on generation relative to the official run shows up in the V0-rerun agreement check (SPEC §5.6), which is that check's purpose.
- GPU jobs request 4 CPUs and 16 GB as SPEC §8 specifies; smoke confirms that the vLLM engine and the bf16 scorer copy fit in that host memory.

## Statistics

- The sensitivity simulation computes each simulated test's p-value exactly: ΔT5's weights take one value per class, so the sign-flip null is a sum of four scaled binomials. This equals exact enumeration and keeps 10,000 simulations per bisection step fast. The same function cross-checks the primary test's implementation in the CPU tests.
- The minimum detectable effect holds each class's validation discordance fixed and bisects on the improve/regress balance; at an equal 10% or 20% discordance it reproduces SPEC §5.7's orientation figures (about 6.4 and 9.1 points).

## Synthetic run

- `scripts/tim.sh all` uses 200 bootstrap resamples and 300 sensitivity simulations at fixture scale (24 patients per split) so the CPU run finishes in seconds. Credentialed stages use the protocol's 10,000 each.
- Wall-clock fields are 0 in synthetic mode, so two runs are byte-identical.

## Review fixes (amendment A1)

- **Split-local loading.** `open_split` reads only the authorized split and gives each stage a private data directory (`$TIM_WORK_DIR/private/prepared/<stage>/`) holding that split and the lab mapping; GPU jobs pass it to `replication.gpu.evaluate` as its data directory. The replication's `load_dataset` is no longer called, because its disjointness audit reads every split.
- **Locks.** A run's writer holds an exclusive `flock` in `$TIM_ADMISSION_DIR/locks/` from admission or resume until it finishes, the same mechanism the replication's `stage_lock` uses on this file system.
- **Generation digest.** Only hypothesis and decision generations enter the digest the shadow control compares.
- **Repeated requests.** `requests.repeated` counts, per patient, requests beyond the first for the same test; `requests.unavailable` counts every request for an unavailable test, so the two can overlap.
- **Reference-path scoring (amendment A2).** bench-a found the prefix-reuse path outside the 1e-3 tolerance against the full-sequence reference, so decisions use the reference path; the bench comparison still runs both paths and reports the largest difference.

## Approvals in practice

- From 2026-11-09 (after bench-a and val-a, which Daniel committed himself) Daniel has authorized the agent to draft and commit each stage's approval file on his behalf; `approved_by` still names him as the approver of record. Every request and its summary is still posted before the stage runs, and the code's own gates still apply: test-a refuses to start unless Addendum A records a passed premise check and a passed (or amended) shadow control.
