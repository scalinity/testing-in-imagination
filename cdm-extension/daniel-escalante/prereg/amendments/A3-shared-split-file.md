# Amendment A3: concurrent jobs share a private split file

Date: 2026-11-09 (US Eastern). Written before the affected arms' test access. No scientific choice changes.

`tim.access.open_split` gives a stage a private copy of its split (`$TIM_WORK_DIR/private/prepared/<stage>/`), which every job of that stage rewrites when it starts. When the 14 test-b jobs started together, one job truncated the file while another was reading it: test-b's B-learned-s1 job (SLURM 50526631) read a partial file and failed with a KeyError inside `open_split`, before admission, so it used no test access.

The other 13 jobs were handled as follows:
- The 9 that had not started (B-learned-s2 and the eight corruption arms) were cancelled before they ran. None was admitted.
- The 4 that had started (B-uniform, B-marginal, B-learned-s0, B-oracle) were admitted with the full 240 test admissions, per the admission ledger's recorded totals. They continued unchanged.

Fix: `open_split` now writes the split copy, the lab mapping and the loaded-ID list through a uniquely named temporary file and renames it into place, so a concurrent reader always sees a complete file; a CPU test watches the final path during a write. The 10 arms that never ran are resubmitted at the fixed commit under a new test-b approval. Earlier stages ran with at most four concurrent jobs and every one of their runs recorded the expected patient total.
