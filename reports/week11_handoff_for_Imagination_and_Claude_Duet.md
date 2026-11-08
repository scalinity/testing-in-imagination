# Week 11 handoff: Testing in Imagination (CAI6734), Daniel Escalante

## 1. Your role and scope

You are helping Daniel Escalante (GitHub `scalinity`) write his **Week 11 individual progress report** for CAI6734, project "Testing in Imagination". The report covers **November 4–10, 2026** and is due **Tuesday, November 10, 2026, 11:59 PM EST**. The project is an independent replication of LA-CDM (Bani-Harouni et al., ICLR 2026; upstream `dharouni/LA-CDM` pinned at `3f435a10`) on the credentialed **MIMIC-IV-Ext-CDM v1.1** dataset. Daniel's code lives in `https://github.com/scalinity/testing-in-imagination` under `replication-of-original/daniel-escalante/`; the local clone is `/Users/danny/Documents/AIBHS/Applied Generative AI in Medicine/Testing in Imagination`.

Everything below was recorded from command output between **2026-11-05, 6:51 PM EST** and **2026-11-08, 5:20 AM EST**. Treat it as the evidence base. The credentialed run is **complete**: all 22 SLURM jobs of the final graph finished, every stage receipt passed, and the aggregate metrics were exported and checked.

Week 9 and Week 10 already covered PhysioNet credentialing, CITI training, the data use agreement and the replication pipeline tested on **synthetic** data. **Week 11 is about the first real-data access, the first full training run on it, and its held-out results.** Do not re-report Week 9/10 items as new.

## 2. What is in flight

Nothing. The final graph, from commit `7fcd9e5`, finished at **5:10 AM EST on Sunday, November 8**. Do not suggest new runs, seeds or arms; any further GPU work needs Daniel's decision first (Section 4.8).

| Stage | SLURM job(s) | Final state | Receipt / duration |
|---|---|---|---|
| S0: environment and data check | login node | PASS, 4:04 PM EST Nov 6 | `s0.json` |
| GPU smoke (2 patients per split) | 50010187 | PASS | code-path check only |
| HPI summaries, 8 shards | 50010230–50010237 | COMPLETED 0:0 | 2,400 summaries |
| S1: seed-269 split | 50010238 | COMPLETED 0:0 | `s1.json` |
| S2: per-arm config wiring | 50010239 | COMPLETED 0:0 | `s2.json` |
| S3: zero-shot baseline (B1) | 50010240 | COMPLETED 0:0 | receipt `f323a460` |
| Train primary / decision_only / no_cost / incorrect_minus1 | 50010241 / 50010242 / 50010243 / 50010244 | all COMPLETED 0:0 | 25 h 5 min / 27 h 58 min / 25 h 42 min / 24 h 53 min |
| S4: verify the four trained arms | 50010245 | COMPLETED 0:0, 9:05 PM EST Nov 7 | receipt `05601581` |
| Test evaluation, one per arm | 50010246–50010249 | all COMPLETED 0:0, 3:44–4:58 AM EST Nov 8 | 24–33 min each |
| S5: verify the four evaluations | 50010250 | COMPLETED 0:0 | receipt `a03cb630` |
| S6: report and statistics | 50010251 | COMPLETED 0:0, 5:10 AM EST Nov 8 | receipt `800c18a6` |

## 3. How Daniel can re-check the status (ask him to run these and paste the output)

On HiPerGator (`ssh daniel.escalante@hpg.rc.ufl.edu`):

```bash
sacct -j $(seq -f %.0f 50010230 50010251 | paste -sd, -) -X -P --format=JobID,JobName,State,ExitCode,Elapsed
ls /blue/cai6734/daniel.escalante/private/cdm-work/mimic-v1.1-hpg-bw-7fcd9e5/receipts/
```

A stage counts as done only when `sacct` shows `COMPLETED` with `ExitCode 0:0` **and** its receipt exists; the receipts directory now holds `s0.json` through `s6.json`. On a Mac, `seq` needs `-f %.0f`, or it prints 8-digit job IDs in scientific notation. Model metrics come only from `results/mimic/metrics.json` and `results/mimic/REPORT.md`, written by `bash scripts/stage.sh export` after S6. Both files are private: the repository's `**/mimic*/` ignore rule keeps them out of git, and the README run log carries the reported values.

## 4. Evidence

### 4.1 Data access, relocation and checksums

- Downloaded with `wget -r -N -c -np --user scalinity --ask-password https://physionet.org/files/mimic-iv-ext-cdm/1.1/` (13 files: 11 data files + `LICENSE.txt` + `SHA256SUMS.txt`, 19 MB).
- **Mac private copy:** `/Users/danny/Documents/AIBHS/private/mimic-iv-ext-cdm/1.1/`, copied at 6:53 PM EST, Nov 5. **SHA256: 12/12 listed digests match**; `SHA256SUMS.txt` itself has sha256 `47f45e1bd4f46b053f5e971274be34f7cbd4ce656a0d6605dbeec11e13f2b2f6`. Files are read-only for the owner (`0400`), the folder is `0500`, and its parent is `0700`. The folder is outside every git repository and is not cloud-synced.
- **The Downloads copy was deleted** at 6:53:34 PM EST, after a byte-for-byte comparison with the private copy.
- **HiPerGator private copy:** `/blue/cai6734/daniel.escalante/private/mimic-iv-ext-cdm/1.1/`, **12/12 SHA256 match** at 7:22 PM EST. The project root `/blue/cai6734/daniel.escalante` is `drwx------` with ACL `group::---`, so classmates in the shared `cai6734` group cannot enter it.
- **No data is in git.** `bash scripts/stage.sh check-git` printed `tracked-file check: PASS` before every commit, including the final one (`b546c2e`). The root `.gitignore` blocks `data/`, `**/mimic*/`, `**/*physionet*`, `**/*.csv` and `.env`; the replication `.gitignore` adds `*.pkl`, `*.parquet`, checkpoints and logs.
- One verification trap: PhysioNet writes `SHA256SUMS.txt` with a single space, which macOS `shasum -c` rejects while still exiting 0. Every digest was therefore checked line by line.

### 4.2 CSV release → condition pickles

- Upstream's splitter reads four `<condition>_hadm_info_first_diag.pkl` files; v1.1 ships CSVs. Upstream defers this step to Hager et al.'s `ConvertPhysionet.py` (`paulhager/MIMIC-Clinical-Decision-Making-Dataset@4d33f98`, MIT). `scripts/build_condition_pickles.py` reimplements the same field mapping, with tests.
- The outputs are **byte-identical on the Mac (arm64) and on HiPerGator (x86_64)**: appendicitis `d56c6849…`, cholecystitis `38e10506…`, diverticulitis `a160d4a9…`, pancreatitis `9b1f8bba…`, lab mapping `dcf3b693…`.

### 4.3 Real-data validation and split (aggregates only)

- `bash scripts/stage.sh validate`: **PASS** on the Mac and on HiPerGator. **2,400 admissions**, classes **957 / 648 / 257 / 538** (appendicitis / cholecystitis / diverticulitis / pancreatitis), exactly the paper's mix.
- Lab mapping: 1,209 rows, of which 118 are test groups with a blank `itemid`. Labs: 138,788 rows plus 4,403 microbiology rows = **143,191**, matching upstream's stated total. Radiology: 5,960 reports (upstream states 5,959; the one-report gap is unexplained). 373 reports use 10 modalities outside CT/MRI/Radiograph/Ultrasound; the upstream parser skips them.
- **No `subject_id` in v1.1**, so subject leakage is unassessable and the subject-grouped sensitivity split cannot run. The report says "unassessable", never zero overlap.
- Seed-269 parity split: **1,920 / 240 / 240**, zero admission overlap (confirmed again in the S6 leakage audit). Test classes 96 / 65 / 25 / 54.

### 4.4 Stage history on real v1.1

Three run graphs were submitted. The first two were cancelled for the reasons listed under Setbacks in Section 5; the third completed.

| Graph | Submitted | Outcome |
|---|---|---|
| `b7190c5` (summaries and arms on separate GPUs) | about 9 PM EST, Nov 5 | S1–S3 PASS (S3 receipt `70fc5c16`, 1 h 32 min). Training jobs 49874617–49874620 cancelled before their first scheduled validation, which would have crashed (setback 1) |
| `4d22c70` | about 11:30 PM EST, Nov 5 | S0–S3 PASS (S3 receipt `17f05b4b`). Training cancelled after a CUDA out-of-memory error at step 333 (setback 2) and a crash in three arms at step 1,340 (setback 3) |
| `7fcd9e5` | 4:04 PM EST, Nov 6 | All 22 jobs COMPLETED 0:0; S0–S6 receipts PASS; finished 5:10 AM EST, Nov 8 |

Run configuration: the primary arm keeps the shipped `incorrect_reward = 0` (D-1); −1 is the separate `incorrect_minus1` ablation. Arms: primary, decision_only (hypothesis and confidence agents disabled), no_cost (cost reward disabled), incorrect_minus1, plus the zero-shot baseline. Training and split seed 269. Each arm trains 2 epochs × 1,920 patients = 3,840 optimizer steps on one RTX PRO 6000, with validation at each epoch end and the final fixed-budget checkpoint selected. Backbone `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` at `54641c16` for both policy and summarizer (D-5, D-3). W&B is disabled. Work directory: `/blue/cai6734/daniel.escalante/private/cdm-work/mimic-v1.1-hpg-bw-7fcd9e5`.

### 4.5 HiPerGator GPUs, runtime and scheduling

- HiPerGator has no A100s. Its GPU partitions are `hpg-turin` (L4, 24 GB), `hpg-rtx6000` (RTX PRO 6000, 96 GB) and `hpg-b200` (B200). The class QOS `cai6734` allows 32 CPUs, 250 GB and 8 GPUs **for the whole class**. The burst QOS `cai6734-b` allows 288 CPUs and 2,250 GB but **no GPUs**.
- The pinned torch 2.6.0+cu124 build stops at `sm_90` (probe job 49867089), so it cannot run on the Blackwell GPUs. `runtime-blackwell/` keeps every upstream package and changes 28 pins to the CUDA 12.8 set of the course's `/blue/cai6734/envs/lacdm-bw` environment: torch 2.7.0+cu128, vLLM 0.9.2, TRL 0.15.2, transformers 4.51.1. Upstream code stays at `3f435a10`. Every manifest records this dependency deviation.
- Trained adapters are evaluated through merged weights, as upstream does, because vLLM's LoRA kernel fails on sm_120 (smoke job 49868789; fix `d434552`).
- Daniel authorized all 8 class GPUs for this graph: 8 summary shards in parallel, then the four arms on one GPU each.
- **Memory:** the training jobs peaked at 6.8 GB of host memory, so the host-memory request of `incorrect_minus1` and the four evaluation jobs was lowered from 40 GB to 16 GB with `scontrol` while they were still queued.
- **Class memory cap:** after training, other class users' CPU-only jobs held 240–246 of the class's 250 GB, so S4 was blocked (`QOSGrpMemLimit`). S4, S5 and S6 need no GPU and were moved to the burst QOS `cai6734-b` (S4 then ran at once). The four evaluation jobs need GPUs and waited 6 h 40 min to 7 h 20 min for class memory.

### 4.6 Code changes pushed to `main` (Week 11)

| Commit | Change |
|---|---|
| `53ae1fc` | Accept the v1.1 release: pickle builder; lab-mapping group rows; unconfigured modalities |
| `ec2311f` | Audit counts unconfigured modalities instead of echoing them (review fix) |
| `9fd3401` | Pickle builder fails without echoing admission IDs (review fix) |
| `8f4e3c2` | Accept integral float item IDs in pickled lab mappings (review fix) |
| `77f9892` | Docs: first real-data run log rows; HiPerGator GPU facts |
| `d711f21` | CUDA 12.8 Blackwell runtime (`CDM_RUNTIME=pinned\|blackwell`) |
| `d434552` | Evaluate trained adapters through merged weights, as upstream does |
| `e65cb03` | Tie every stage to the runtime S0 built; truthful packaging note (review fixes) |
| `6a9d112` | Docs: Blackwell S0, smoke, submitted S1–S3 |
| `a863585` | The first version of this handoff |
| `b7190c5` | Run summaries and arms on separate GPUs (parallel SLURM graph) |
| `14ac0b0` | Docs: log the parallel S1–S6 graph |
| `4d22c70` | Validate once per epoch and stop tracking a best metric (setback 1) |
| `595276a` | Docs: log the cancelled `b7190c5` training and the `4d22c70` graph |
| `7fcd9e5` | Keep episode braces literal; reclaim fragmented GPU memory (setbacks 2 and 3) |
| `39aaa20` | Docs: log the cancelled `4d22c70` training and the `7fcd9e5` graph |
| `b546c2e` | Docs: completed run and held-out metrics in the README run log and `REPRODUCTION.md` |

Plus the commit that rewrites this file. Tests: **256 pass** at `7fcd9e5`. Each fix had an independent cross-vendor review (Codex GPT-6.1 Sol, xhigh, read-only); after the fixes, no finding remained open.

### 4.7 Metrics (real data, held-out test set)

Source: `results/mimic/REPORT.md` and `metrics.json` from the aggregate export after S6 (`check-git` PASS; a scan found no patient-level fields). Primary arm, parity split, seed 269, **240 test admissions**, a **different backbone and summarizer** from the paper. Intervals are 95% class-stratified patient bootstraps (10,000 resamples). No match verdict is given: no tolerance was agreed, and the backbone differs.

| Claim | Metric | Paper | Observed | 95% interval |
|---|---|---|---|---|
| T1 | Appendicitis recall (%) | 93.1 | 90.62 | [84.38, 95.83] |
| T2 | Cholecystitis recall (%) | 83.6 | 72.31 | [61.54, 83.08] |
| T3 | Diverticulitis recall (%) | 75.0 | 72.00 | [56.00, 88.00] |
| T4 | Pancreatitis recall (%) | 73.5 | 57.41 | [44.44, 70.37] |
| T5 | Mean of four class recalls (%) | 81.3 | 73.09 | [66.65, 79.19] |
| T6 | Micro F1 (%) | 84.1 | 81.15 | [76.47, 85.53] |
| T7 | Macro F1 (%) | 81.3 | 78.20 | [72.32, 83.44] |
| T8 | Average provided-test cost (USD) | 1,295.61 | 1,439.58 | [1,341.50, 1,537.65] |
| B1 | Zero-shot mean recall (%) | 64.5 | 12.71 | [8.04, 17.86] |
| B2 | ReAct | 74.9 | gap | no ReAct at the upstream pin |

- Observed minus paper, T5: −8.21 points [−14.65, −2.11]; T6: −2.95 [−7.63, 1.43]; T7: −3.10 [−8.98, 2.14]. Pooled accuracy 76.25%.
- **C1** hypothesis accuracy, zero-shot → primary: 29.59% → 81.67% (paper 75.7 → 81.9). **C2** ECE: 0.3022 → 0.1281 (paper 0.069 → 0.037). These are held-out endpoints, not a training curve.
- **C3** no_cost → primary: cost $1,321.33 → $1,439.58, mean recall 70.81 → 73.09%. Unlike the paper, the cost reward did not lower cost: the paired cost difference is +$118.25 [−0.25, 239.86], and McNemar on pooled accuracy gives Holm p = 0.551.
- **C4** decision_only → primary mean recall: 57.49 → 73.09% (paired +15.59 [6.34, 24.60]; McNemar Holm p = 0.00056). The paper's exact Table 3 values are not in the ledger.
- **D-1 ablation** (incorrect reward −1): mean recall 59.38%, cost $1,619.27; primary minus ablation T5 +13.71 [5.32, 21.93], McNemar Holm p = 0.0031.
- Only one training seed ran, so between-seed uncertainty is unassessable; the intervals cover test-patient sampling only.
- Validation hypothesis accuracy (primary, never test data): 31.21% at the pre-training point, 77.58% at step 1,920, 75.15% at the final checkpoint.

### 4.8 Open items and decisions

1. **The gaps to the paper are not yet explained.** T2 and T4 recall, T8 cost, and especially B1 (12.71% vs 64.5%) sit well below the paper. The report's cost table counts 631 missing test requests for zero-shot across 240 admissions, against 267 for the trained primary arm, and $8,002.34 of requested cost per admission against $3,499.42. Any investigation must use aggregates only.
2. **Single seed.** Between-seed uncertainty needs more training seeds, which means more multi-day GPU jobs; that is Daniel's decision under the class GPU rules.
3. **The subject-grouped sensitivity split is unassessable**, because v1.1 has no `subject_id`.
4. **The Blackwell runtime is a dependency deviation** (28 pins), and the backbone and summarizer differ (D-5, D-3); both must be named whenever results are reported.
5. **The ReAct baseline (B2) remains a gap.**
6. **HiPerGator storage** (not backed up): work directories `-7fcd9e5` 5.7 GB (of which intermediate checkpoints 4 × 1.3 GB; final adapters 97 MB each), `-4d22c70` 15 GB, `-b7190c5` 187 MB, `-e65cb03` 158 MB, `-d434552` 80 KB. The class allocation has 955 GB free. Nothing is deleted without Daniel's approval.
7. The one-report radiology count difference (5,960 vs 5,959) is unexplained.

## 5. Suggested Week 11 report bullets (same structure as Week 10)

**Progress**
- First credentialed-data access: downloaded MIMIC-IV-Ext CDM v1.1 and verified all 12 listed SHA256 digests. It's stored owner-only outside the repository, on the Mac and in private HiPerGator storage; no data in git.
- Converted the CSV release into the four condition pickles upstream expects, byte-identically on two machines. Real-data validation passed: 2,400 admissions, 957/648/257/538; seed-269 split 1,920/240/240 with zero admission overlap.
- Set up HiPerGator under `cai6734` with a documented CUDA 12.8 runtime for its Blackwell GPUs, and parallelized the pipeline: 8 GPUs for summaries, then four training arms on one GPU each.
- **Completed the first full real-data run:** four arms × 3,840 training steps (about 25–28 h each), test evaluation on 240 held-out patients, and all stage receipts S0–S6 passing.
- **First held-out results** (primary, different 8B backbone): mean class recall 73.09% (paper 81.3%), micro F1 81.15% (84.1%), macro F1 78.20% (81.3%), cost $1,439.58 ($1,295.61). Training lifted mean recall from 12.71% zero-shot, and hypothesis accuracy from 29.59% to 81.67%.
- The hypothesis agent helped: decision-only reached 57.49% mean recall (McNemar Holm p = 0.00056). The −1 incorrect-reward ablation also scored lower, at 59.38% (p = 0.0031).
- 256 tests passing; every fix independently reviewed.

**Setbacks (and how I addressed them)**
- **Validation crash before it happened:** the independent review found that upstream's `metric_for_best_model: loss` would raise `KeyError` at the first scheduled validation, because the per-patient validation reports no `eval_loss`. I cancelled the four training jobs (49874617–49874620) before they reached it. The fix (`4d22c70`) turns best-metric tracking off and validates once per epoch instead of every 100 steps, saving about 50 GPU-hours per arm. The fix was reviewed, then smoke-tested (job 49889947: scheduled validation at step 2 ran with no best metric). On full data, every arm passed both epoch validations.
- **GPU out-of-memory at step 333** (decision_only, job 49889996): 12.31 GiB needed, 12.27 GiB free, 4.70 GiB lost to fragmentation. I resumed from the checkpoint with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (job 49924900), which passed step 333. Then I built the setting into every stage in `7fcd9e5` and recorded it in the manifests.
- **Crash at step 1,340 in all three hypothesis arms** (jobs 49889995, 49889997, 49889998): upstream's `environment.py:407` runs `str.format` over the whole episode text, and that step's patient record contains braces. Identical failures at the same step pointed to a data-driven, deterministic bug. I wrote a failing test, made the first-hypothesis injection a literal substitution (`7fcd9e5`), and had it reviewed with no findings. GPU smoke 50010187 passed, and on full data all arms passed step 1,340 and finished.
- **Shared class memory cap:** after training, other class jobs held up to 246 of the class's 250 GB. I moved the CPU-only verification stages to the class burst QOS; the GPU evaluations waited about 7 h, then finished in under 35 min each.
- v1.1 has no `subject_id`, so the subject-grouped leakage check is reported as unassessable.

**Next steps**
1. Write up the comparison to the paper, naming the backbone, summarizer and runtime deviations, and investigate the gaps (cholecystitis and pancreatitis recall, cost, the zero-shot baseline) from aggregates only.
2. Decide with the team whether further GPU time goes to more training seeds (for between-seed uncertainty) or to the stock Llama-3.1-8B fallback backbone (D-5).
3. Clean up superseded HiPerGator work directories and checkpoints, keeping the final adapters, receipts and aggregates.

**Notes**
- All numbers are aggregates from the S6 export; no PHI is anywhere in the repository or reports. The synthetic results stay labelled SYNTHETIC.
- Results come from one training seed on a different backbone, so they are a replication attempt rather than a reproduction of the paper's numbers.
- HiPerGator storage is not backed up; code is in git and data can be re-downloaded from PhysioNet.

## 6. Constraints (non-negotiable)

- **PhysioNet DUA:** never ask Daniel to paste, upload or quote any MIMIC row, note text, generated summary, episode transcript or model generation into this chat or any other online service. Aggregate counts, job IDs, states and the exported metrics are fine. The DUA explicitly bars sending credentialed data to third-party model services.
- Never suggest sharing data or private folders with Hugo Resendez, Edwin Salcedo or anyone else; each person needs their own credentialed access.
- Never invent or estimate a metric. A number appears in the report only if it is printed in `results/mimic/metrics.json`, `results/mimic/REPORT.md`, a receipt, or command output Daniel pastes. Every value in Section 4.7 comes from the S6 export.
- State times in US Eastern (EST after November 1, 2026).

## 7. Verification rules for anything you draft

- Tag each claim with its source from Section 4 (commit SHA, job ID, receipt ID or report section).
- Never describe the results as reproducing the paper: the backbone, summarizer and runtime differ, and no tolerance was agreed.
- If Daniel's pasted output contradicts this file, the pasted output wins; say which line changed.

## 8. Close-out

1. When Daniel asks for the report, produce the Progress / Setbacks / Next steps / Notes sections from Section 5, trimmed to the course's length.
2. Keep the metric table from Section 4.7 exactly as written; do not round differently or drop the intervals.
3. End every reply with one line: `Status as of <time EST>: run 7fcd9e5 complete (S0–S6 PASS), model metrics: results/mimic/REPORT.md (private), README run log rows of 2026-11-08`.
