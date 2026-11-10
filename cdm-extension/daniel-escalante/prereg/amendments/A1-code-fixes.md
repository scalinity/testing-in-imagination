# Amendment A1: code fixes after the protocol freeze

Date: 2026-11-09 (US Eastern). Written before any test access. No scientific choice changes: variants, constants, seeds, contrasts, families, tests and wording rules are as frozen in `prereg/protocol.json`.

An independent review of the Phase-A code (at commit `e3083a9`) found the defects below; each is fixed with a CPU test.

1. **Split-local loading.** `tim.access.open_split` called `replication.data.load_dataset`, whose `assert_disjoint` audit parses all three prepared CSVs. As a result, the smoke job (SLURM 50442537, commit `f3e67c0`) opened and parsed the train and test CSVs while loading its two validation patients. That audit checks record structure and admission-ID disjointness and returns only pass or fail: no test-derived value was computed, nothing from it was logged publicly or published, and no test episode ran. `open_split` now reads only the authorized split file, writes a private copy for the stage (only the listed patients for smoke and bench), and GPU jobs give that directory to the baseline evaluator, which reads only that split. Cross-split integrity rests on the digests the validate stage records.
2. **One writer per run.** A run now holds an exclusive lock on its key while it writes, so a second job cannot resume an admission whose first job is still running.
3. **Crash reconciliation.** A row written before its ledger record is recovered; a truncated final line is dropped and its patient re-run (it was never recorded as complete); a run with every row but no completion record finishes without evaluating anyone.
4. **Validation stop rules at test admission.** Approvals for `test-a` now require a committed Addendum A and refuse test access unless the premise check held and the shadow control passed, or a committed amendment explains a shadow mismatch (SPEC §5.3, §5.5).
5. **Shadow-control digest.** The per-patient generation digest now covers only hypothesis and decision generations, as SPEC §5.5 specifies. It had also covered upstream's confidence-calibration generations, whose prompts come from an RNG that a resumed run restarts.
6. **Repeated requests.** The published repeated-request count is now the per-patient number of requests beyond the first for the same test, available or not. It had excluded repeats of unavailable tests. The validate bundle is republished with this definition.
7. **Smoke accounting.** A smoke attempt now records its GPU time even when it fails; an attempt killed before recording is charged its scheduler-reported time (or the whole cap when that is unavailable), and unapproved smoke jobs are limited to the remaining allowance.

Infrastructure changes since the freeze, none affecting results: jobs receive `HF_HOME` (the model cache the baseline run used); GPU jobs go to the least busy course account first; the bench-a request scales smoke's measured time.

Smoke is re-run on the fixed code before bench-a, because item 1 changes the GPU data path.
