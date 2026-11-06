# Week 11 handoff: Testing in Imagination (CAI6734), Daniel Escalante

## 1. Your role and scope

You are helping Daniel Escalante (GitHub `scalinity`) track and write his **Week 11 individual progress report** for CAI6734, project "Testing in Imagination". The report covers **November 4–10, 2026** and is due **Tuesday, November 10, 2026, 11:59 PM EST**. The project is an independent replication of LA-CDM (Bani-Harouni et al., ICLR 2026; upstream `dharouni/LA-CDM` pinned at `3f435a10`) on the credentialed **MIMIC-IV-Ext-CDM v1.1** dataset. Daniel's code lives in `https://github.com/scalinity/testing-in-imagination` under `replication-of-original/daniel-escalante/`; the local clone is `/Users/danny/Documents/AIBHS/Applied Generative AI in Medicine/Testing in Imagination`.

Everything below was recorded on **2026-11-05 between 6:51 PM and 8:38 PM EST** from command output. Treat it as the evidence base. Anything later, especially SLURM job states, comes only from Daniel pasting fresh command output; never assume a job finished.

Week 9 and Week 10 already covered PhysioNet credentialing, CITI training, the data use agreement and the replication pipeline tested on **synthetic** data. **Week 11 is about the first real-data access and what actually ran on it.** Do not re-report Week 9/10 items as new.

## 2. What is in flight (track it; do not act on it)

| Stage | SLURM job | Partition / GPU | Time limit | State at 8:37 PM EST, Nov 5 |
|---|---|---|---|---|
| S1: seed-269 split + HPI summaries | 49872927 | `hpg-rtx6000`, 1× RTX PRO 6000 | 24 h | RUNNING on `c0800a-s5` since about 8:29 PM EST |
| S2: per-arm config wiring | 49872929 | `hpg-default`, CPU | 1 h | PENDING (after S1) |
| S3: zero-shot baseline (B1) on the 240 test patients | 49872930 | `hpg-rtx6000`, 1× RTX PRO 6000 | 12 h | PENDING (after S2) |
| S4–S6: training, final evaluation, report | not submitted | — | — | Needs Daniel's decision (Section 4.8) |

- S1's duration was not measured; the GPU smoke suggests roughly 6–10 hours (about 10–15 s per summary × 2,400).
- A separate Claude Code session runs on HiPerGator under Daniel's account. It must follow the same data rule as you (Section 6).
- **Do not** recommend submitting S4–S6 until Daniel approves the GPU budget: UF's class rules require asking before any multi-GPU or >24 h job, and the class shares one 8-GPU cap.
- **Do not** describe S1–S3 as complete, or quote any model metric, until Daniel pastes the output of the checks in Section 3.

## 3. How Daniel checks status (ask him to run these and paste the output)

On HiPerGator (`ssh daniel.escalante@hpg.rc.ufl.edu`):

```bash
squeue -u daniel.escalante -o "%.10i %.8j %.8T %.10M %R"
sacct -j 49872927,49872929,49872930 -X -P --format=JobID,JobName,State,Start,End,Elapsed,ExitCode
ls /blue/cai6734/daniel.escalante/private/cdm-work/mimic-v1.1-hpg-bw-e65cb03/receipts/
```

A stage counts as done only when `sacct` shows `COMPLETED` with `ExitCode 0:0` **and** `receipts/s1.json` (or `s2.json`, `s3.json`) exists. A missing receipt means the stage did not finish, whatever the job state says. S3's zero-shot numbers become reportable only after S6 runs and `bash scripts/stage.sh export` writes `results/mimic/metrics.json`; until then there are **no model metrics**.

## 4. Evidence

### 4.1 Data access, relocation and checksums

- Downloaded with `wget -r -N -c -np --user scalinity --ask-password https://physionet.org/files/mimic-iv-ext-cdm/1.1/` (13 files: 11 data files + `LICENSE.txt` + `SHA256SUMS.txt`, 19 MB).
- **Mac private copy:** `/Users/danny/Documents/AIBHS/private/mimic-iv-ext-cdm/1.1/`, copied at 6:53 PM EST. **SHA256: 12/12 listed digests match**; `SHA256SUMS.txt` itself has sha256 `47f45e1bd4f46b053f5e971274be34f7cbd4ce656a0d6605dbeec11e13f2b2f6`. Files are read-only for the owner (`0400`), the folder is `0500`, and its parent is `0700`. The folder is outside every git repository and is not cloud-synced (no iCloud Documents sync; Google Drive is a separate folder).
- **The Downloads copy was deleted** at 6:53:34 PM EST, after a byte-for-byte comparison with the private copy.
- **HiPerGator private copy:** `/blue/cai6734/daniel.escalante/private/mimic-iv-ext-cdm/1.1/`, **12/12 SHA256 match** at 7:22 PM EST. The project root `/blue/cai6734/daniel.escalante` is `drwx------` with ACL `group::---`, so classmates in the shared `cai6734` group cannot enter it.
- **No data is in git.** `bash scripts/stage.sh check-git` printed `tracked-file check: PASS` before every commit. The root `.gitignore` blocks `data/`, `**/mimic*/`, `**/*physionet*`, `**/*.csv` and `.env`, and the replication `.gitignore` adds `*.pkl`, `*.parquet`, checkpoints and logs. The private `.env` files on the Mac and on HiPerGator are git-ignored.
- One verification trap: PhysioNet writes `SHA256SUMS.txt` with a single space, which macOS `shasum -c` rejects while still exiting 0. Every digest was therefore checked line by line.

### 4.2 CSV release → condition pickles

- Upstream's splitter reads four `<condition>_hadm_info_first_diag.pkl` files; v1.1 ships CSVs. Upstream defers this step to Hager et al.'s `ConvertPhysionet.py` (`paulhager/MIMIC-Clinical-Decision-Making-Dataset@4d33f98`, MIT). Running that external repository was blocked by the session's safety check, so `scripts/build_condition_pickles.py` reimplements the same field mapping, with tests.
- The outputs are **byte-identical on the Mac (arm64) and on HiPerGator (x86_64)**: appendicitis `d56c6849…`, cholecystitis `38e10506…`, diverticulitis `a160d4a9…`, pancreatitis `9b1f8bba…`, lab mapping `dcf3b693…`.

### 4.3 Real-data validation and split preflight (aggregates only)

- `bash scripts/stage.sh validate`: **PASS** on the Mac and on HiPerGator. **2,400 admissions**, classes **957 / 648 / 257 / 538** (appendicitis / cholecystitis / diverticulitis / pancreatitis), exactly the paper's mix.
- Lab mapping: 1,209 rows, of which 118 are test groups with a blank `itemid`. Labs: 138,788 rows plus 4,403 microbiology rows = **143,191**, matching upstream's stated total. Radiology: 5,960 reports (upstream states 5,959; the one-report gap is unexplained). 373 reports use 10 modalities outside CT/MRI/Radiograph/Ultrasound (e.g. MRCP, ERCP); the upstream parser skips them.
- **No `subject_id` in v1.1**, so subject leakage is unassessable and the subject-grouped sensitivity split cannot run. The pipeline reports this as unassessable, never as zero overlap.
- Seed-269 parity split (pinned upstream splitter, CPU preflight): **1,920 / 240 / 240**, zero admission overlap. Test classes 96 / 65 / 25 / 54. The real upstream parser read all 2,400 admissions. Test-set availability: MRI 6/240, CT 174/240, radiograph 94/240, ultrasound 113/240, liver function panel 217/240, urinalysis 199/240, other panels and physical exam 240/240.

### 4.4 Stage status on real v1.1

| Stage | Status | Evidence |
|---|---|---|
| S0 on the Mac | Blocked by design | 7:03 PM EST: data validation passed, then the macOS platform gate stopped it (no CUDA/SLURM on the M5 Pro). Failure record `s0-2026-11-05T190329-0500` |
| S0, pinned stack, HiPerGator | PASS | 7:35 PM EST, commit `8f4e3c2`, receipt `92a05616…`: torch 2.6.0+cu124, vLLM 0.8.1, `xgrammar==0.1.16` built from source, model revision verified. Superseded |
| S0, Blackwell stack, HiPerGator (run in use) | PASS | 8:28:08 PM EST, commit `e65cb03` (clean), receipt `816da388513af068`, manifest `ad47f0c98002227b`, lock `f12ce556…`, 28 substituted pins |
| S1 | RUNNING | SLURM 49872927 |
| S2, S3 | QUEUED | SLURM 49872929, 49872930 |
| S4, S5, S6 | NOT SUBMITTED | Section 4.8 |

Config for the run: primary arm keeps the shipped `incorrect_reward = 0` (D-1); −1 is the separate ablation arm. Arms: primary, decision-only, no-cost, incorrect = −1, plus the zero-shot baseline. Training seed and split seed are 269. Backbone `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` at `54641c16` (D-5, D-3). W&B is disabled in every arm. The work directory is `/blue/cai6734/daniel.escalante/private/cdm-work/mimic-v1.1-hpg-bw-e65cb03`.

### 4.5 HiPerGator GPUs and the runtime decision

- HiPerGator has no A100s. Its GPU partitions are `hpg-turin` (L4, 24 GB), `hpg-rtx6000` (RTX PRO 6000, 96 GB) and `hpg-b200` (B200). Account `cai6734` has 32 CPUs, 250 GB and 8 GPUs **for the whole class**, and no burst QOS for GPUs.
- Probe job 49867089 on an L4 showed the pinned torch 2.6.0+cu124 build stops at `sm_90`, so it cannot run on the Blackwell GPUs. On an L4, upstream's vLLM memory setting (0.6 × 24 GB) cannot hold the 16 GB model.
- Decision (Daniel approved the Blackwell path): `runtime-blackwell/` keeps every upstream package and changes 28 pins to the CUDA 12.8 set of the course-provided `/blue/cai6734/envs/lacdm-bw` environment: torch 2.7.0+cu128, vLLM 0.9.2, xgrammar 0.1.19, transformers 4.51.1. TRL 0.15.2 and numpy are unchanged, and upstream code stays at `3f435a10`. This is a documented dependency deviation recorded in every manifest.
- GPU smoke on RTX PRO 6000 (2 patients per split, run only to check that the code works):
  - Job 49868701 was cancelled by mistake while running.
  - Job 49868789 passed summarize and train but failed evaluating a trained adapter inside vLLM's Triton LoRA kernel (`PassManager::run failed`, sm_120).
  - Fix `d434552` evaluates adapters as upstream does (merged weights copied into vLLM).
  - Job 49872040 **passed all steps**: summarize 43 s; train 146 s for 4 optimizer steps (~36.5 s/step, all three loss phases); zero-shot and adapter evaluation both completed.
- The model was downloaded at the pinned revision (job 49868688, 64 s, 15 GB) into private `/blue` storage. No Hugging Face token was needed.

### 4.6 Code changes pushed to `main` (all on 2026-11-05)

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

Plus the commit that adds this file. Tests: **253 pass** on the Mac at `e65cb03`; 247 passed on HiPerGator at `8f4e3c2` (the HiPerGator suite was not re-run at `e65cb03`). There were two independent cross-vendor reviews (Codex GPT-6.1 Sol, xhigh, read-only). They found 5 warnings and no critical issues, and all 5 were fixed with tests.

### 4.7 Metrics

**No model metrics exist yet.** No zero-shot or trained Table-1 value (T1–T8, B1) has been produced on real data. The only real-data numbers are the cohort, mapping and split aggregates in 4.3. The smoke run's rewards and losses come from 2 patients and four steps; they only show the code runs and must never be reported as results. The synthetic rows dated 2026-10-30 remain SYNTHETIC.

### 4.8 Open blockers and decisions

1. **S4 compute budget (Daniel's decision).** Upstream trains 2 epochs × 1,920 patients = **3,840 optimizer steps per arm**, with a 240-patient validation every 100 steps. At the measured ~36.5 s/step that is about 39 h of steps per arm before validation, and four arms on one GPU is on the order of a week. The current `s4.slurm` asks for 48 h, so training needs either resumable multi-job submission (checkpoints every 100 steps) or arms run in parallel, within the class's 8-GPU cap.
2. **S1 duration is unmeasured** (estimate 6–10 h; limit 24 h).
3. **The subject-grouped sensitivity split is unassessable**, because v1.1 has no `subject_id`.
4. **The Blackwell runtime is a dependency deviation** (28 pins); it must be named whenever results are reported.
5. **The ReAct baseline (B2) remains a gap**: the pinned upstream ships none.
6. The one-report radiology count difference (5,960 vs 5,959) is unexplained.
7. Colleagues: on GitHub, neither Hugo nor Edwin has pushed replication work (template READMEs only). Edwin's PR #1 is the synthetic world-model extension. The course staff's `/blue/cai6734/LA-CDM` checkout uses a different backbone (`unsloth/Llama-3.1-8B-Instruct`) with W&B offline.

## 5. Suggested Week 11 report bullets (same structure as Week 10)

**Progress**
- First credentialed-data access: downloaded MIMIC-IV-Ext CDM v1.1 and verified all 12 listed SHA256 digests. It's stored owner-only outside the repository, on the Mac and in private HiPerGator storage; the Downloads copy was deleted; no data in git.
- Converted the CSV release into the four condition pickles upstream expects, byte-identically on two machines.
- Real-data validation passed: 2,400 admissions, 957/648/257/538. Seed-269 split 1,920/240/240 with zero admission overlap; the upstream parser read every admission.
- Real data exposed two validator assumptions built on synthetic data (blank-ID lab groups; unconfigured imaging modalities). Both were fixed to match upstream behaviour, with tests.
- Set up HiPerGator under account `cai6734`. Diagnosed that the pinned CUDA 12.4 stack cannot run on its Blackwell GPUs. Added a documented CUDA 12.8 runtime, following the course's `lacdm-bw` environment.
- S0 passed on HiPerGator. A GPU smoke on an RTX PRO 6000 passed the summarize, training and evaluation paths. S1–S3 (split and summaries, config wiring, zero-shot baseline) were submitted and S1 is running.
- Two independent code reviews; 5 warnings fixed; 253 tests passing.

**Setbacks (and how I addressed them)**
- The pinned stack has no kernels for HiPerGator's Blackwell GPUs, and L4s are too small → added a CUDA 12.8 runtime with every substituted version recorded.
- vLLM's LoRA kernel failed on RTX PRO 6000 → switched S5 to upstream's merged-weight evaluation, which also matches upstream more closely.
- v1.1 has no `subject_id` → the subject-grouped leakage check is unassessable and is reported as such.
- Upstream's 2-epoch training is about 3,840 steps per arm → S4 needs a GPU budget decision before submission.

**Next steps**
1. Confirm S1–S3 receipts. After S6, export the zero-shot (B1) aggregates and log them in the README run log.
2. Decide S4's plan (resumable multi-job vs parallel arms) within the class GPU cap, then submit S4–S6.
3. Keep every reported number aggregate-only, with the runtime deviation named.

**Notes**
- No model metrics yet; synthetic results stay labelled SYNTHETIC; no PHI anywhere in the repository or reports.
- HiPerGator storage is not backed up; code is in git and data can be re-downloaded from PhysioNet.

## 6. Constraints (non-negotiable)

- **PhysioNet DUA:** never ask Daniel to paste, upload or quote any MIMIC row, note text, generated summary, episode transcript or model generation into this chat or any other online service. Aggregate counts, job IDs, states and metrics files are fine. The DUA explicitly bars sending credentialed data to third-party model services.
- Never suggest sharing data or private folders with Hugo Resendez, Edwin Salcedo or anyone else; each person needs their own credentialed access.
- Never invent or estimate a metric. A number appears in the report only if it is printed in `results/mimic/metrics.json`, a receipt, or command output Daniel pastes.
- State times in US Eastern (EST after November 1, 2026).

## 7. Verification rules for anything you draft

- Tag each claim with its source from Section 4 (commit SHA, job ID, receipt ID or command).
- Mark anything not yet observed as **not run** or **pending**, never as passed.
- If Daniel's pasted output contradicts this file, the pasted output wins; say which line changed.

## 8. Close-out

1. Ask Daniel for the Section 3 command output when he next wants the report updated.
2. Update the status table in Section 2 from that output only.
3. When he asks for the report, produce the Progress / Setbacks / Next steps / Notes sections from Section 5, adjusted for the new status.
4. End every reply with one line: `Status as of <time EST>: S1 <state>, S2 <state>, S3 <state>, S4–S6 not submitted, model metrics: <none | path>`.
