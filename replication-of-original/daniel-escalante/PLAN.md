# Replication plan

Build a re-implementation on a different backbone against upstream
`3f435a10f569c3255172c1ef8fe7880c6edb7fee`. T1–T8, B1–B2, C1–C4 and I1–I5
remain unverified on MIMIC until credentialed data and CUDA runs complete.

## Architecture and stages

One configuration module reads the untracked `.env`. `CDM_DATA_SOURCE=synthetic|mimic`
selects data and execution: synthetic defaults to a deterministic mock policy;
mimic requires CUDA and real summarization/training. All working data, episode rows,
checkpoints and manifests live in `CDM_WORK_DIR` outside the repository. Only aggregate
metrics and the write-up may be exported to `results/`.

- S0 validates runtime, upstream identity, immutable model revision, packages and
  the fully composed upstream Hydra configuration. The pin has no `uv.lock`:
  generate a local runtime project from its exact dependency declarations, lock
  it locally and use `uv sync --frozen` there; never change upstream.
  The index no longer serves `xgrammar==0.1.16`; retain that exact version using
  publisher source commit `989222175c2a30fb7987d8bcce35bec1bf6817f2` in the local lock.
  Source-build installation on HiPerGator remains unverified.
- S1 validates raw files before GPU admission; invokes upstream seed-269 splitter,
  copies/converts the lab mapping, summarizes HPI with the explicit approved model,
  and builds a separate subject-grouped sensitivity split when subject IDs exist.
- S2 installs a local launcher shim for system messages, padding, revision forwarding
  and sampled training versus greedy validation/evaluation. F-1–F-4 remain intact.
- S3 evaluates the untrained policy on the same held-out patients as later arms.
  No ReAct implementation ships at the pin: B2 is an honest gap, not a substitute.
- S4 trains primary, decision-only, no-cost and incorrect-reward −1 arms independently
  on train, with val-only checkpoint selection. Shipped incorrect reward 0 is primary.
  Keep the upstream cyclic 100/100/100 schedule, Adam lr 1e-5, batch 2, seed 269.
- S5 evaluates each locked final model once on test, with identical patient order,
  prompt/token/action budgets. Interrupted/failed seeds remain in the run ledger.
- S6 computes upstream metrics, paired 10,000-resample intervals, exact discordance
  tests, leakage and cost decomposition, and renders a source-labelled write-up.

Each script accepts `--dry-run` and writes a manifest. `scripts/run_all.sh` runs S0–S6
with content-validated resumable receipts. Synthetic runs execute the actual upstream
dataset parser, environment, rewards and metric functions; mock updates demonstrate
orchestration only, never learning. No model downloads or GPU rentals in CPU mode.

## Data contract

Four `<condition>_hadm_info_first_diag.pkl` dictionaries keyed by integer admission ID.
Consumed fields: `Patient History` (string), `Physical Examination` (string),
`Laboratory Tests` (integer item ID → string), `Radiology` (list of dictionaries with
`Modality` and `Report` strings). `Label` is assigned by the splitter, `Patient ID`
comes from the dictionary key, `Patient History Summary` is produced in S1.
Optional `subject_id` supports admission-overlap audits and grouped splitting;
absence must be reported as unassessable, never zero overlap.
`lab_test_mapping.csv` or `.pkl` must resolve `itemid`, `label`, `corresponding_ids`
through the upstream parser. The configured 7 lab panels, 4 imaging modalities and
physical exam are the only action vocabulary; normalized costs sum to 1 including
physical exam. Validate roughly 2,400 admissions and the expected 957/648/257/538
class mix before mimic execution. Synthetic records are invented independently,
seeded by (seed, split, stage), with missing tests and repeated subjects.

## Verification and review

Tests cover corrupted schemas, actual dataset parsing, disjoint admission IDs on every
load, subject leakage and grouped sensitivity, split-specific RNG, provided/requested
costs, reward variants, model allowlisting, actual prompt paths, loader revisions and
padding, statistics against hand-computed cases, complete manifests, safe exports,
tracked-file data detection, deterministic resume and a full synthetic shell run.
Run the existing `lacdm/tests` unchanged. Independently review the integrated change
and resolve findings before PR merge. Preserve all failed runs and report unrun CUDA
checks explicitly.

## Constraints and questions settled from the record

- D-4 conflicts with `scripts/split_dataset.py`: the executable pin hardcodes
  `RANDOM_STATE=269`; use 269 for parity and report the ledger inconsistency.
- No upstream license is listed. Import the pinned submodule; do not copy its source.
- `metric_writer.py` only accepts fixture/run sources. Reuse it through a local adapter
  that stamps the actual synthetic/mimic source; shared-file change is out of scope.
- Paper T5 mean accuracy is the mean of four class recalls; also report pooled accuracy
  so class imbalance and the upstream micro-F1 treatment of abstentions stay visible.
- Extra seeds are optional and must be declared before training. Never omit a seed;
  one seed does not establish between-training-run uncertainty.
- SLURM account, partition and storage paths are site-specific untracked settings.
  CUDA memory capacity and scheduler admission remain HiPerGator verification work.
- A source-only implementation can close wiring blockers, not establish CUDA numerical
  acceptance. No synthetic result will be called a reproduced paper result.

## Implementation ownership

`replication/config.py`, `manifest.py`, `pipeline.py`, shell/SLURM entry points and
integration tests form the coordinator. Separate bounded lanes own `data.py` and
`test_data.py`; `shim.py`, `gpu.py` and `test_shim.py`; and `stats.py`, `reporting.py`,
`paper_values.json`, `REPORT_TEMPLATE.md` and their tests. No shared-file edits.
All lanes work from this plan and return patches for inspection before integration.
