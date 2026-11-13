# Watcher, pipeline, and local loop: design for TII-CEE-1

Companion to SPEC_RESEARCH_ADVANCEMENT.md version 3.1, section 20. November 13, 2026.
This is a design for the executing agent to implement and test. Nothing here has been run on HiPerGator.

## 1. What UF Research Computing's documentation says (read November 13, 2026)

| Source | What it says | Consequence |
|---|---|---|
| docs.rc.ufl.edu/scheduler/scrontab | scrontab schedules recurring Slurm jobs; `scrontab -e` and `-l`; `#SCRON` lines act as directives and `#SBATCH` lines are ignored; jobs must request resources; an example uses `--dependency=singleton`; all recurrences share one job ID; jobs do not run in a login shell | The watcher is a scrontab entry |
| docs.rc.ufl.edu/services/non_batch | Long-running processes cannot run on login servers; a daemon server exists for small recurring or long-running tasks | Fallback host for the watcher |
| HiPerGator usage policies (it.ufl.edu/rc/documentation/policies/hipergator-usage-policies) | Login nodes are for managing jobs and data and editing files; short tests only; processes that frequently poll the scheduler are not allowed; more than 10,000 jobs at once needs approval; no burst QOS for GPU jobs | One queue query and one accounting query per cycle; nothing long-lived on login nodes |
| docs.rc.ufl.edu/access/ssh_multiplexing | Multiplexing is the documented way to avoid MFA on every connection; `ControlMaster auto`, `ControlPersist 8h`; on macOS use `ControlPath ~/.ssh/control/%C` if the socket path is too long; no maximum session length stated; nothing on keepalives | Basis for the connection keeper |

The agent re-reads these pages in M0 (specification 7.4c), records the dates, and stops to raise a BLOCKED item if any of them now says otherwise. The 15-minute cycle is my choice of a conservative interval; the documents give no number.

## 2. Layout on HiPerGator

Under the study's storage root, `$TII` = `/blue/cai6734/daniel.escalante/private/tii-cee-1` (from AUTHORIZATION.md, section 3; UNVERIFIED against the live cluster until the launch check):

    $TII/pipeline/graph.json        stage graph: id, experiment, command, resources, depends_on, gate, allowance key
    $TII/state/stages/<id>.json     status, attempts, job ids, checkpoint pointer, manifest digest, GPU-hours
    $TII/state/lock/                directory lock for the watcher cycle (atomic mkdir)
    $TII/state/runlog.jsonl         append-only: one line per event and per job attempt
    $TII/state/events.tsv           notification queue: time, kind, text. Names and counts only
    $TII/state/STATUS.md            rewritten each cycle; aggregate content only
    $TII/state/BLOCKED.md           mirror of the repository's BLOCKED.md
    $TII/gates/<gate>.json          immutable gate records
    $TII/export/approved/           aggregates that passed the export guard; the only tree the Mac copies
    $TII/private/                   everything patient-level; never listed, printed, or copied by the agent

`events.tsv`, `STATUS.md`, and `export/approved/` are the whole surface the Mac reads.

## 3. Stage contract

Every stage is started as `run_stage <id>` inside a Slurm job and obeys:

1. If `<id>.done` exists and its output digest matches, exit 0.
2. Verify the manifest: code revision, experiment card, fitted artifacts, inputs, SHA-256 of AUTHORIZATION.md. On mismatch exit 20 and write the reason. Never retried.
3. If a checkpoint exists, resume from it; otherwise start.
4. Write outputs to a temporary name; rename when complete; then write `<id>.done` with the digest.
5. Request at most 24 hours with a warning signal ten minutes before the limit. On the signal: checkpoint, exit 10.
6. Granularity of checkpoints: training, every saved step; rollouts and replays, every completed admission, keyed by case fingerprint so nothing is scored twice.

Exit codes: 0 done; 10 continue; 20 identity refusal; 30 authorization or data-use refusal; 40 out of memory detected by the stage; anything else is a failure.

Test stages add: a one-shot ledger entry written before the first test row is scored; a second start with a different manifest is refused; a resume under the same manifest is the same run.

## 4. The watcher cycle

Installed as one tagged scrontab entry:

    #SCRON --job-name=tii-watcher
    #SCRON --account=cai6734 --qos=cai6734-b
    #SCRON --time=00:10:00 --mem=1gb --cpus-per-task=1
    #SCRON --dependency=singleton
    #SCRON --output=<TII>/state/watcher_%j.log --open-mode=append
    */15 * * * * <TII>/pipeline/watcher_cycle.sh    # tii-watcher

The watcher needs no GPU, so it runs on the CPU burst QOS the repository already uses for CPU stages. The exact directive spelling is checked against the live scrontab; if it marks a line `#BAD`, fix it before anything else. Because all recurrences share one job ID, the script writes its own timestamped lines.

Each cycle, in order:

1. Take the lock with `mkdir`; if it exists and is younger than 20 minutes, exit; if older, record a stale lock and take it.
2. Recompute the SHA-256 of AUTHORIZATION.md and compare with the bound value. Check the expiry date. If either fails: mark every not-yet-submitted stage that needs authorization as BLOCKED, queue one event, and continue with step 3 for jobs already running.
3. One `sacct` call for this study's job names since the last cycle and one `squeue --me` call. No other scheduler calls in the cycle except the `sbatch` calls of step 5.
4. For each stage whose job ended, classify from the scheduler state and the exit code:

   | Outcome | Action | Retries |
   |---|---|---|
   | exit 0 | mark DONE; queue "stage completed" | |
   | exit 10, or TIMEOUT with a newer checkpoint | resubmit | not counted |
   | TIMEOUT with no progress, PREEMPTED, NODE_FAIL | resubmit from checkpoint | 3 |
   | OUT_OF_MEMORY or exit 40 | resubmit with 1.5 times the memory | 1 |
   | exit 20 or 30 | mark REFUSED; BLOCKED item; queue event | 0 |
   | any other failure | resubmit once, then FAILED | 1 |

   Every attempt is appended to `runlog.jsonl` with job ID, state, elapsed time, GPU-hours from the scheduler, and exit code, and is charged to the experiment's allowance. A stage that exhausts its retries is FAILED, an event is queued, and only its dependents wait.
5. Submit every stage whose dependencies are DONE and whose gate, if any, is met, while the GPUs in use stay at or below the limit and the experiment's GPU-hours plus the stage's estimate stay at or below its allowance. If the allowance would be exceeded: no submission, a BLOCKED item with the numbers.
6. For each gate whose inputs are complete and which has no record yet, run its rule script over `export/approved/` and the private aggregates it is allowed to read, write `gates/<gate>.json` with input digests, the rule, the result, and the path taken, mark skipped stages SKIPPED BY GATE, and queue "gate decided".
7. Rewrite `STATUS.md`: milestone, counts by status, running and queued jobs, GPU-hours by experiment against estimate and allowance, open BLOCKED items, last gate records.
8. If no stage is pending, running, retryable, or waiting on a gate that can still be evaluated: write the final status, queue "study completed" or "all remaining items blocked", and disable the tagged scrontab entry only, leaving any other entry as it was. Verify with `scrontab -l`.
9. Release the lock.

The gate scripts are deterministic and committed before the gate's inputs exist. The watcher makes no other judgment. Analysis, writing, new code, and the paper are the agent's work in the local loop.

If a scrontab job cannot call `sbatch`, `squeue`, or `sacct`, the same script runs from cron on the daemon server, which the documentation provides for exactly this kind of task.

## 5. The local loop on the Mac

A launchd agent, `com.tii.local-loop`, every 600 seconds, built and installed by the agent with Daniel's authorization:

1. Take a local lock; exit if an agent session for this study is already running.
2. Check the SSH master with `ssh -O check hpg`. If it is not there, exit quietly. The loop never opens a connection.
3. Through the master, with a command that cannot fall back to a new connection, read the new lines of `events.tsv` and `STATUS.md`.
4. Copy `export/approved/` with rsync over the same master. No other path is ever copied.
5. If a new "stage completed", "gate decided", "failure", or "blocked" event exists, start or resume one headless agent session in the repository with the fixed continuation prompt:
   "Read STATUS.md, RESEARCH_LEDGER.md, BLOCKED.md, and the newest gate records. Continue study TII-CEE-1 under the version 3.1 kickoff from the first unfinished item. Do not repeat completed work."
   The command line for the headless session is stored in one configuration file and is checked against the installed tool's own help when the loop is built; it is not assumed here.
6. Release the lock.

The session's own work continues to obey the kickoff: it reaches HiPerGator only through the master, and only the surface of section 2 crosses the channel.

## 6. Notifications

The watcher queues events; the keeper relays them as macOS notifications when the connection is up (keeper/KEEPER_INSTALL.md, with `TII_REMOTE_EVENTS` set to `/blue/cai6734/daniel.escalante/private/tii-cee-1/state/events.tsv`), and the local loop does the same if the keeper's relay is off. Events: stage completed; gate decided; blocked; failure after retries; study completed; all remaining items blocked. The keeper itself raises "sign in again" and "reconnected". Every event is also a line in `STATUS.md`.

## 7. Tests before any real job

1. A three-stage no-op graph with a gate: runs to completion, writes the gate record, disables the watcher.
2. Kill a stage mid-run: it resumes from its checkpoint and scores nothing twice.
3. Change a bound artifact: exit 20, no retry, BLOCKED item.
4. Set an allowance below the next estimate: no submission, BLOCKED item.
5. Rename AUTHORIZATION.md: nothing new is submitted; restoring it resumes.
6. Start two cycles at once: one exits on the lock.
7. Drop the Mac's connection for an hour: HiPerGator work proceeds; the local loop catches up afterward.
8. Confirm by inspection that `events.tsv`, `STATUS.md`, and `export/approved/` contain no patient-level content.
