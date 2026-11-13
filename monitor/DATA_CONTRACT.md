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

## Figure 1 files (local, written by the study's figure step)

Figure 1 (specification 1.7) is drawn only from result manifests, so its three data files are
written by the study's figure step into the repository, aggregates only, and the monitor reads them
locally, never over SSH:

- `tii-cee/results/figure1/regime_map.json`: `{"schema": "tii-fig1-regime/1", "generated_at", "harm_grid": [{"tau_b", "kl_inc", "harmful", "delta_log_loss", "power", "drift", "drift_kind"}] (the exact twin grid readers, drawn as points by harm sign; no boundary is drawn), "tau_b_definition": "first decision point", "markers": [{"id", "label", "experiment": "E1|E2|E4|E7|E8|E9", "kind": "simulated|language_model|trained_policy|second_backbone|public_dataset", "tau_b", "tau_b_lo", "tau_b_hi", "kl_inc", "kl_inc_lo", "kl_inc_hi", "params_b" (optional), "dataset" (optional), "delta_log_loss"}]}`. `kl_inc` is the location incoherence after scaling, in nats per decision; `tau_b` is the budget-implied scale (log axis, 1 = matched).
- `tii-cee/results/figure1/gain_accounting.json`: `{"schema": "tii-fig1-gain/1", "generated_at", "cells": [{"id", "reader", "setting", "experiment", "predicted_gain", "predicted_lo", "predicted_hi", "realized_gain", "realized_lo", "realized_hi"}]}`.
- `tii-cee/results/figure1/regret_attribution.json`: `{"schema": "tii-fig1-regret/1", "generated_at", "agents": [{"id", "label", "variant": "raw|repaired", "selection", "reading", "commitment", "range": {"selection": [min, max], "reading": [min, max], "commitment": [min, max]}, "stable", "same_as"}]}`: shares in the declared order (they sum to 1 but individual shares may fall outside [0, 1]), their range over all six orders, whether the attribution passed its stability rule, and the id of an identical agent where two are the same. An unstable agent is drawn as its ranges, not as parts of a whole.

A missing file means that panel's experiments have not produced results; the monitor says which.
