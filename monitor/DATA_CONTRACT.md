# Data contract between the HiPerGator watcher and the TII monitor

The watcher (tii-cee/tiicee/pipeline) writes these files under the study root
`$TII = /blue/cai6734/daniel.escalante/private/tii-cee-1`. The monitor reads them through the
existing SSH master only, with `ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false
daniel.escalante@hpg.rc.ufl.edu cat|tail ...`. It never opens a connection, never calls a Slurm
command, and never reads any other path on HiPerGator.

Every file below holds only what may cross the SSH channel (specification 20.10): scheduler
metadata, stage and gate names, approved exit codes, counts, timings, GPU-hours, and aggregates that
passed the export guard. The watcher's surface_check enforces it. Times are ISO 8601 with offset.

## state/status.json (rewritten atomically every watcher cycle)

```json
{
  "schema": "tii-status/1",
  "generated_at": "2026-11-13T03:15:02-05:00",
  "milestone": "M0",
  "watcher": {"cycle": 12, "started_at": "...", "finished_at": "...", "interval_s": 900,
              "lock": "ok | held | stale_taken", "completed": false},
  "authorization": {"ok": true, "sha256_ok": true, "signature_ok": true, "expires": "2027-03-31",
                    "addenda": [{"name": "AUTHORIZATION_ADDENDUM_1.md", "sha256_ok": true, "signed": true}]},
  "gpu": {"limit_by_account": {"cai6734": 8, "cai5724": 8}, "in_use_by_account": {"cai6734": 3, "cai5724": 0}},
  "counts": {"PLANNED": 10, "QUEUED": 2, "RUNNING": 3, "COMPLETED": 5, "FAILED": 0,
             "REFUSED": 0, "BLOCKED": 0, "SKIPPED BY GATE": 0, "INTERRUPTED": 0},
  "experiments": [{"id": "E8", "status": "RUNNING", "estimate_gpu_hours": 105,
                   "allowance_gpu_hours": 210, "spent_gpu_hours": 12.4}],
  "stages": [{"id": "e8-seed1-part1", "experiment": "E8", "kind": "normal", "status": "RUNNING",
              "depends_on": ["preflight"], "gate": null, "attempts": 1, "job_id": "51234567",
              "account": "cai6734", "gpus": 1, "submitted_at": "...", "started_at": "...",
              "ended_at": null, "elapsed_s": 3600, "gpu_hours": 1.0, "exit_code": null,
              "last_state": "RUNNING", "estimate_gpu_hours": 25.6}],
  "gates": [{"id": "G2a", "decided": false, "result": null, "path_taken": null, "decided_at": null}],
  "blocked": [{"id": "BLOCKED-003", "title": "short name only", "opened_at": "...", "status": "OPEN"}]
}
```

## state/progress/<stage_id>.json (written by a running stage, at most once a minute)

```json
{"stage": "e4-cohort-x", "unit": "admission", "done": 812, "total": 2160, "updated_at": "..."}
```

## state/runlog.jsonl (append-only; one line per event and per job attempt)

```json
{"time": "...", "event": "submitted | started | ended | resubmitted | refused | gate_decided | blocked",
 "stage": "...", "experiment": "E8", "attempt": 1, "job_id": "...", "account": "cai6734",
 "state": "COMPLETED | TIMEOUT | PREEMPTED | NODE_FAIL | OUT_OF_MEMORY | FAILED | CANCELLED",
 "exit_code": 0, "elapsed_s": 86400, "gpu_hours": 24.0}
```

## state/events.tsv (append-only notification queue)

`time<TAB>kind<TAB>text`, kind one of: stage completed, gate decided, blocked, failure,
study completed, all remaining items blocked. Text holds names and counts only.

## state/STATUS.md, gates/<gate>.json, export/approved/

As WATCHER_DESIGN.md describes. The monitor may list export/approved/ and read its JSON files,
which already passed the export guard.
