# Gate G0, addendum: launch check 7.4c closed

Evaluated November 13, 2026, about 6:10 AM EST. Closes the one item G0.md left open. Immutable once
committed.

## Evidence: the live no-op graph on HiPerGator

Installed 4:42 AM from code export 462600e: the scrontab entry of WATCHER_DESIGN.md section 4 and a
three-stage no-op graph with gate G_noop (WATCHER_DESIGN section 7, test 1).

| Time (EST) | Cycle | Node | What the watcher did |
|---|---|---|---|
| 4:46 | 1 | c0700a-s1 | authorization ok, addendum signed; sacct and squeue answered; submitted noop_a (job 51145986) through sbatch |
| 5:02 | 2 | c0700a-s7 | noop_a COMPLETED (ran 4:49:23 to 4:49:28); gate G_noop evaluated by its committed rule script, met; record written to gates/ and copied through the export guard to export/approved/gates/ |
| 5:16 | 3 | c0706a-s6 | submitted noop_b (CPU, routed to cai5724-b because Daniel's own pending jobs requested 320 CPUs on cai6734-b) |
| 5:32 | 4 | c0704a-s16 | noop_b COMPLETED; submitted noop_c |
| 5:45 | 5 | c0704a-s6 | noop_c COMPLETED; final status "study_completed" (3 of 3, 0 failed, 0 refused, 0 blocked); event queued; the tagged scrontab entry removed from inside the job, `scrontab -l` empty afterwards |

## Result

Launch check 7.4c ANSWERED: a scrontab job on HiPerGator can call sacct, squeue, sbatch and
scrontab; every cycle ran on a compute node; nothing long-lived ran on a login node; one sacct and
one squeue per cycle. The daemon-server fallback (its documentation page now 404) is not needed.
G0 is MET in full.

## Found by the live test and handled before any GPU stage

- squeue's %b shows N/A for GPUs requested per job (Daniel's v2-* jobs on hpg-b200), so GPU use was
  undercounted; the watcher's queue call moves to a fixed-width --Format with tres-per-job and
  tres-alloc, verified live on Slurm 25.11.8 (line length 956).
- A race between a stage's temporary progress file and the watcher's surface check, found under
  load in the local tests; fixed by preparing writes outside the watched folders.
- The class GPU QOS's 250 GB memory cap binds (QOSGrpMemLimit); the queue-reason routing covers it.
These ship in the next deployment; no GPU stage runs on export 462600e.
