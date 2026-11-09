# Week 11 finalization close-out: LA-CDM replication (run `7fcd9e5`)

Daniel Escalante · CAI6734 Testing in Imagination · closed Sunday, November 8, 2026 (all times EST)

Scope: close the six open items from the completed credentialed run (graph `7fcd9e5`, parity split, training seed 269, 240 test admissions) without starting any new training. Every number below comes from `results/mimic/metrics.json`, `results/mimic/REPORT.md`, `results/mimic/supplementary.json`, a stage receipt or command output. Aggregates and counts only; no patient row, identifier, note text, summary or model generation appears here or in the repository.

These results are a replication attempt, not a reproduction: the policy and summarizer differ (D-5, D-3), the CUDA 12.8 Blackwell runtime changes 28 upstream pins, decoding differs (D-6), there is one training seed, and no tolerance was agreed.

## 1. What was checked or fixed, item by item

### Item 1: zero-shot gap (B1 12.71% vs paper 64.5%): no code bug; decoding ruled out

- **Line-by-line comparison with upstream `3f435a10`.** The prompt, chat template, regex action parser, action names, stop string (`Observation:`) and 256-token generation budget all match. Two differences:
  - **System message:** ours is `detailed thinking off`, which the D-5 backbone requires.
  - **Decoding:** upstream's `configure_grpo(eval_only=True)` keeps the training temperature, 1.0, and `CDMGRPOTrainer` samples with it during evaluation. This replication pre-specified greedy decoding. This is recorded as new deviation **D-6**.
- **Invalid episodes per arm (greedy, counts only).** An undiagnosed episode ended without a valid diagnosis, through a malformed action, the 14-action budget or the 4,096-token trajectory limit. The episodes do not record which of these happened, so parse failures cannot be separated from budget or length endings. Grouping by how many valid tests an episode requested before ending bounds them. Zero valid requests before the end is the only group that can be a first-action parse failure, and only zero-shot has a sizeable one.

  | Arm | Undiagnosed / 240 | 0 valid requests | 1–3 | 4–7 | 8–13 | 14+ | Repeated requests | Valid requests |
  |---|---|---|---|---|---|---|---|---|
  | zero_shot | 194 | 23 | 27 | 66 | 56 | 22 | 613 | 1,521 |
  | primary | 29 | 0 | 6 | 13 | 8 | 2 | 121 | 661 |
  | decision_only | 19 | 0 | 8 | 6 | 4 | 1 | 108 | 747 |
  | no_cost | 31 | 1 | 6 | 13 | 9 | 2 | 154 | 618 |
  | incorrect_minus1 | 65 | 0 | 18 | 22 | 12 | 13 | 299 | 937 |

  Greedy zero-shot got 27 of the 46 diagnoses it gave right. 904 of its 1,095 hypotheses were cholecystitis.
- **Decoding check:** a post-hoc test, labelled as such, with greedy kept as the primary analysis.
  - **Code:** `4b79683` (1:24 PM) added `decode-arm` / `decode-report` and the confusion counts, test-first.
  - **Review:** an independent Codex review (`gpt-6.1-sol`, xhigh; NativePeers group `92e4959e-c97c-4afe-a537-24229fd8691c`, task `88f12023-6694-40b5-86a1-4bbdbba9c1e1`) raised five warnings, F001–F005: the admission race, missing subject clusters, the Holm family shrinking in partial reports, multi-seed resume, and protocol/seed labels. All five were fixed test-first in `d3ef9de` (1:36 PM).
  - **Jobs:** every locked model was evaluated once on the same 240 test patients at temperature 1.0, each request with its own seed. All finished `COMPLETED 0:0`:

    | Job | Arm | Code | Time (EST) |
    |---|---|---|---|
    | 50282940 | zero_shot | `4b79683` | 1:29–2:11 PM |
    | 50282941 | primary | `d3ef9de` | 2:24–2:40 PM |
    | 50282942 | decision_only | `d3ef9de` | 2:25–2:45 PM |
    | 50282943 | no_cost | `d3ef9de` | 2:25–2:42 PM |
    | 50282944 | incorrect_minus1 | `d3ef9de` | 2:25–2:41 PM |
    | 50282959 | report (burst QOS) | `d3ef9de` | 2:46–2:58 PM |

  - **Result:** temperature 1.0 lowered every arm. Zero-shot mean recall fell from 12.71 to 9.19, a paired difference of −3.52 [−9.38, 2.37]. Decoding therefore does not explain the B1 gap. The backbone and summarizer (D-5, D-3) remain the untested candidates; testing them needs a new backbone run, which was out of scope.
- **Outcome:** no bug was found, so nothing was re-run under item 1's fix-and-rerun clause. The original greedy evaluations, S5, S6 and their receipts stand unchanged.

### Item 2: cost reward is wired into primary training

- **Composed configs:** primary and decision_only include `TestCostReward` (normalized costs sum to −1.0). no_cost omits it, and incorrect_minus1 includes it with the incorrect reward −1.
- **Trainer logs:** across the 3,840 primary steps, the logged total reward equals the sum of its components (maximum residual 0.0).
- **Mean per 960-step block (primary):**
  - cost: −0.087, −0.099, −0.104, −0.106;
  - diagnosis: 0.166, 0.296, 0.453, 0.535;
  - format: 0.259, 0.416, 0.594, 0.692.
- **Reading:** the reward is present but small next to the diagnosis reward, and it acts only in the decision-agent phase (100 of every 300 updates). C3 is not observed under either decoding (section 3).

### Item 3: confusion matrices: added to `REPORT.md`

`REPORT.md`'s supplementary section now has 4×4 counts plus an *undiagnosed* column for all five arms under both decodings. The greedy (primary-analysis) matrices requested:

| Primary, greedy | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
|---|---|---|---|---|---|
| appendicitis (96) | 87 | 0 | 2 | 2 | 5 |
| cholecystitis (65) | 2 | 47 | 0 | 10 | 6 |
| diverticulitis (25) | 0 | 0 | 18 | 3 | 4 |
| pancreatitis (54) | 1 | 7 | 1 | 31 | 14 |

| Zero-shot, greedy | appendicitis | cholecystitis | diverticulitis | pancreatitis | undiagnosed |
|---|---|---|---|---|---|
| appendicitis (96) | 11 | 10 | 1 | 0 | 74 |
| cholecystitis (65) | 0 | 10 | 0 | 0 | 55 |
| diverticulitis (25) | 0 | 0 | 6 | 0 | 19 |
| pancreatitis (54) | 1 | 6 | 1 | 0 | 46 |

### Item 4: aggregate files tracked

- **Exception:** a narrow root `.gitignore` exception admits exactly three files: `results/mimic/metrics.json`, `REPORT.md` and `supplementary.json`. `git check-ignore` confirms any other file in that folder, and every other `mimic*` or `data/` path, stays ignored.
- **Scans, run before staging:**
  - the existing patient-field scan (`assert_aggregate`) passed on both JSON files;
  - no 8-digit identifier-shaped token appears in `REPORT.md` or `supplementary.json`. One appears in `metrics.json`, inside the 64-character hex S6 manifest digest, so it is not an identifier.
- **Before replacing `REPORT.md`:** the S6 part above the supplementary marker was confirmed byte-identical, and `metrics.json` is byte-identical to the S6 export (written 5:13 AM).
- **Commit:** `bash scripts/stage.sh check-git` printed `tracked-file check: PASS` on the staged files, then `2478dae` was committed (7:34 PM).

### Item 5: HiPerGator cleanup

See section 5. All four listed directories were deleted; `-7fcd9e5` is untouched.

### Item 6: documentation, `aeab9a8` (7:34 PM)

- **`REPRODUCTION.md`:**
  - a Status and Reason for every claim: T1–T8, B1, B2 and C1–C4, each as within CI, below CI, above CI, gap or unassessable;
  - a D-6 row and a decoding-check table;
  - an expanded honest-gaps list;
  - the definition-of-done status: B met and exceeded; A unassessable because no tolerance was agreed.
- **README, "Interpretation and open checks":**
  - metric definitions;
  - what the counts show: undiagnosed episodes, cholecystitis/pancreatitis confusion, the cost reward and the decoding check;
  - open checks: single seed, no `subject_id`, B2 ReAct, radiology 5,960 vs 5,959, and D-1, D-3, D-5, Blackwell and D-6.
- **README, "Running an independent second replication":** five steps for Hugo or Edwin from their own credentialed access. No data or private folder is shared.
- **README, GPU budget:** about 105 GPU-hours per additional training seed (about 29 h on four GPUs), about 211 for two more seeds, and about 110 for a different backbone. No new seed or backbone was trained.
- **README, run log:** new rows for the code audit, the cost-reward check, the confusion counts (job 50282958), the cleanup and the decoding check. The day-one `.env` example now matches the run, and the verification paragraph says 264 tests.
- **Tests:** 264 passed at `d3ef9de` (181 replication, 83 `lacdm/tests`).

## 2. Changed metrics: old → new

The primary analysis (greedy) is **unchanged**: no bug was found, so no evaluation, S5 or S6 was re-run, and `metrics.json` is byte-identical to the S6 export. The only new metrics are the supplementary decoding check. They are reported beside greedy, not in place of it.

| Arm | Metric | Greedy (unchanged) | Temperature 1.0 [95% CI] | Paired difference [95% CI] |
|---|---|---|---|---|
| zero_shot | B1 / T5 mean recall | 12.71 | 9.19 [5.55, 13.44] | −3.52 [−9.38, 2.37] |
| primary | T1 appendicitis recall | 90.62 | 65.62 [56.25, 75.00] | −25.00 [−35.42, −14.58] |
| primary | T2 cholecystitis recall | 72.31 | 66.15 [53.85, 76.92] | −6.15 [−20.00, 7.69] |
| primary | T3 diverticulitis recall | 72.00 | 48.00 [28.00, 68.00] | −24.00 [−44.00, −4.00] |
| primary | T4 pancreatitis recall | 57.41 | 57.41 [44.44, 70.37] | 0.00 [−16.67, 16.67] |
| primary | T5 mean recall | 73.09 | 59.30 [52.23, 66.19] | −13.79 [−21.70, −6.24] |
| primary | T6 micro F1 | 81.15 | 70.12 [64.27, 75.63] | −11.04 [−17.22, −5.21] |
| primary | T7 macro F1 | 78.20 | 66.91 [59.94, 73.16] | −11.29 [−18.43, −4.52] |
| primary | T8 provided-test cost (USD) | 1,439.58 | 961.71 [875.56, 1,046.85] | −477.87 [−601.46, −355.99] |
| decision_only | T5 | 57.49 | 40.64 [33.86, 47.27] | −16.85 [−25.05, −8.68] |
| no_cost | T5 | 70.81 | 61.46 [54.52, 68.28] | −9.35 [−17.60, −1.02] |
| incorrect_minus1 | T5 | 59.38 | 55.75 [48.63, 62.62] | −3.63 [−12.94, 5.66] |

Undiagnosed episodes, greedy → 1.0:

| Arm | Undiagnosed | Repeated requests | Valid requests |
|---|---|---|---|
| zero_shot | 194 → 202 | 613 → 44 | 1,521 → 406 |
| primary | 29 → 55 | 121 → 26 | 661 → 400 |
| decision_only | 19 → 50 | 108 → 26 | 747 → 479 |
| no_cost | 31 → 50 | 154 → 25 | 618 → 372 |
| incorrect_minus1 | 65 → 78 | 299 → 29 | 937 → 380 |

Hypothesis accuracy, zero-shot → primary: 28.49% → 75.41% at 1.0 (greedy 29.59% → 81.67%).

Comparisons at 1.0 (exact McNemar, Holm family of four):

| Comparison | Difference [95% CI] | Holm p | Reading |
|---|---|---|---|
| primary − decision_only | T5 +18.66 [9.94, 27.21] | 0.000049 | C4 holds |
| primary − no_cost | cost +$82.94 [−30.33, 195.84] | 0.538 | C3 still not observed |
| primary − incorrect_minus1 | T5 +3.55 [−5.68, 12.76] | 0.474 | the greedy D-1 difference does not hold under sampling |
| primary − zero_shot | T5 +50.10 [41.96, 57.98] | < 0.000001 | |

## 3. Final metric table (primary analysis: greedy, run `7fcd9e5`)

Receipts: S3 `f323a460` · S4 `05601581` · S5 `a03cb630` · S6 `800c18a6`. Intervals are 95% class-stratified patient bootstrap (10,000 resamples).

| ID | Metric | Paper | Ours [95% CI] | Status |
|---|---|---|---|---|
| T1 | Appendicitis recall | 93.1 | 90.62 [84.38, 95.83] | within CI |
| T2 | Cholecystitis recall | 83.6 | 72.31 [61.54, 83.08] | below CI |
| T3 | Diverticulitis recall | 75.0 | 72.00 [56.00, 88.00] | within CI |
| T4 | Pancreatitis recall | 73.5 | 57.41 [44.44, 70.37] | below CI |
| T5 | Mean class recall | 81.3 | 73.09 [66.65, 79.19] | below CI |
| T6 | Micro F1 | 84.1 | 81.15 [76.47, 85.53] | within CI |
| T7 | Macro F1 | 81.3 | 78.20 [72.32, 83.44] | within CI |
| T8 | Avg provided-test cost (USD) | 1,295.61 | 1,439.58 [1,341.50, 1,537.65] | above CI (costlier) |
| B1 | Zero-shot mean recall | 64.5 | 12.71 [8.04, 17.86] | below CI; unexplained (no bug; decoding ruled out) |
| B2 | ReAct mean recall | 74.9 | — | gap (no implementation at the pin) |
| C1 | Hypothesis accuracy, zero-shot → primary | 75.7 → 81.9 | 29.59 → 81.67 [77.24, 85.89] | endpoint within CI; start below; +52.08 [46.40, 57.73] |
| C2 | ECE, zero-shot → primary | 0.069 → 0.037 | 0.3022 → 0.1281 [0.0861, 0.1749] | improves; endpoint above paper; −0.174 [−0.232, −0.114] |
| C3 | Cost reward lowers cost (no_cost → primary) | $1,427.85 → $1,295.61 | $1,321.33 → $1,439.58 | not observed; +118.25 [−0.25, 239.86], Holm p 0.551 |
| C4 | Hypothesis agent helps (decision_only → primary) | Table 3 | 57.49 → 73.09 | direction observed; +15.59 [6.34, 24.60], Holm p 0.00056; paper value unassessable |
| D-1 | incorrect −1 ablation, T5 | — | 59.38 (primary − ablation +13.71 [5.32, 21.93], Holm p 0.0031) | greedy only; not significant at 1.0 |

## 4. Remaining gaps

- **B1 zero-shot gap** (12.71 vs 64.5) is unexplained. There is no code bug against `3f435a10`, and upstream's sampled decoding scores lower still. The backbone and summarizer (D-5, D-3) are untested; testing them needs a new backbone run (about 110 GPU-hours), which was out of scope.
- **One training seed (269).** Between-seed uncertainty is unassessable; every interval covers test-patient sampling only. One more seed costs about 105 GPU-hours.
- **No `subject_id` in v1.1.** Subject leakage of the parity split and the subject-grouped sensitivity split are unassessable, never reported as zero.
- **B2 ReAct** has no implementation at the upstream pin; no substitute is labelled ReAct.
- **Radiology count:** v1.1 has 5,960 reports where upstream states 5,959; unexplained.
- **C4 paper value:** Table 3 is not in the ledger, so only the direction is assessed.
- **Parse failures vs budget endings** are not recorded per episode. Only the bounding counts in section 1 are available.
- **Deviations named in every report:**
  - D-1: shipped incorrect reward 0 as primary;
  - D-3 and D-5: Nemotron Nano 8B as summarizer and policy;
  - Blackwell runtime: 28 pins, torch 2.7.0+cu128, vLLM 0.9.2;
  - D-6: greedy evaluation.

## 5. HiPerGator deletions

All four sat under `/blue/cai6734/daniel.escalante/private/cdm-work/`:

| Directory | Unique size | Notes |
|---|---|---|
| `mimic-v1.1-hpg-bw-4d22c70` | 5.3 GB | `4d22c70` graph, training cancelled |
| `mimic-v1.1-hpg-bw-b7190c5` | 187 MB | `b7190c5` graph, training cancelled |
| `mimic-v1.1-hpg-bw-e65cb03` | 158 MB | `e65cb03` S0 and GPU smoke graph |
| `mimic-v1.1-hpg-bw-d434552` | 80 KB | early `d434552` graph |

- **Checks before deleting:** zero references to any target in `-7fcd9e5`'s metadata, zero symlinks pointing at them, and the untracked `.env` names only the kept directory.
- **Deletion:** removed at 1:30 PM EST, November 8. About 5.6 GB of unique data was freed. Measured alone, the directories show 15 GB / 9.4 GB / 9.4 GB / 80 KB, because they share hard-linked runtime files with the kept directory.
- **Kept:**
  - `mimic-v1.1-hpg-bw-7fcd9e5` (15 GB) is intact.
  - `mimic-v1.1-hpg` (56 KB), `mimic-v1.1-hpg-bw` (52 KB) and `smoke-bw` (746 MB) were not on the list and remain.

## 6. Updated Week 11 bullets (November 4–10)

**Progress**
- First credentialed-data access: downloaded MIMIC-IV-Ext-CDM v1.1 and verified all 12 SHA256 digests. The copy is stored owner-only outside every repository, on the Mac and in private HiPerGator storage.
- Converted the CSV release into the four condition pickles upstream expects, byte-identically on two machines. Real-data validation passed: 2,400 admissions (957/648/257/538), seed-269 split 1,920/240/240 with zero admission overlap.
- Set up HiPerGator under `cai6734` with a documented CUDA 12.8 runtime for its Blackwell GPUs, and parallelized the pipeline: summaries on 8 GPUs, then four training arms on one GPU each.
- **Completed the first full real-data run** (graph `7fcd9e5`): four arms × 3,840 training steps (about 25–28 h each), test evaluation on 240 held-out patients, and all stage receipts S0–S6 passing.
- **Held-out results** (primary, different 8B backbone):

  | Metric | Ours | Paper |
  |---|---|---|
  | Mean class recall | 73.09% | 81.3% |
  | Micro F1 | 81.15% | 84.1% |
  | Macro F1 | 78.20% | 81.3% |
  | Cost | $1,439.58 | $1,295.61 |

  Training lifted mean recall from 12.71% zero-shot, and hypothesis accuracy from 29.59% to 81.67%. Decision-only reached 57.49% (Holm p = 0.00056).
- **Finalized the replication.**
  - Every paper claim has a status against our 95% interval, with its reason.
  - Confusion matrices show undiagnosed episodes driving most errors (29 for primary, 194 for zero-shot).
  - The cost reward is confirmed wired into training but small.
  - The aggregate-only export is tracked in git through a narrow exception.
- 264 tests passing; every code change independently reviewed (Codex review of the decoding check: 5 warnings, all fixed test-first).

**Setbacks (and how I addressed them)**
- **Validation crash caught before it happened:** upstream's `metric_for_best_model: loss` would raise `KeyError` at the first validation. I cancelled the training jobs before they reached it and turned best-metric tracking off (`4d22c70`). That fix also validates once per epoch, saving about 50 GPU-hours per arm.
- **GPU out-of-memory at step 333** (decision_only): resumed with expandable allocator segments, then built the setting into every stage (`7fcd9e5`).
- **Crash at step 1,340 in three arms:** upstream's `str.format` over episode text containing braces. I wrote a failing test and made the first-hypothesis injection a literal substitution (`7fcd9e5`).
- **Shared class memory cap:** moved the CPU-only stages to the burst QOS; the GPU evaluations waited about 7 h for class memory.
- **Zero-shot baseline far below the paper (12.71% vs 64.5%):**
  - audited our code against upstream line by line: no bug;
  - found that upstream samples at temperature 1.0 where we decode greedily (D-6);
  - re-ran every locked model at 1.0 as a labelled check. Every arm scored lower (zero-shot 9.19%), so decoding does not explain the gap. The backbone and summarizer remain the untested explanation.

**Next steps**
1. Decide with the team whether further GPU time goes to more training seeds (about 105 GPU-hours each) or to the stock Llama-3.1-8B backbone (about 110 GPU-hours), the only remaining test of the zero-shot gap.
2. Write the paper comparison into the final report, naming D-1, D-3, D-5, D-6 and the Blackwell runtime.
3. Support an independent second replication by Hugo or Edwin from their own credentialed access, following the README steps.

**Notes**
- All numbers are aggregates from the S6 export and the post-hoc supplement. No PHI is in the repository or reports, and synthetic results stay labelled SYNTHETIC.
- One training seed on a different backbone: a replication attempt, not a reproduction of the paper's numbers.
- HiPerGator storage is not backed up. Code and aggregates are in git, and data can be re-downloaded from PhysioNet.
