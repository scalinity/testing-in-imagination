// Writes the demo fixtures. Everything here is FICTIONAL: invented jobs, invented numbers,
// invented worker activity. The files follow monitor/DATA_CONTRACT.md so that demo mode runs
// through the same parsers as real mode. Run with `pnpm fixtures`; the output is deterministic.

import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const ROOT = import.meta.dirname;
const DEMO = "FICTIONAL demo fixture for the TII monitor. Not study data.";
const HOUR = 3600e3;
const MIN = 60e3;

// The demo's clock: a Tuesday afternoon inside the plan's Nov 20-26 window.
const NOW = Date.parse("2026-11-24T14:10:00-05:00");
const at = (text) => Date.parse(`2026-${text}:00-05:00`); // "11-19T17:40"

/** ISO 8601 with the Eastern offset the watcher writes. */
const iso = (ms) => new Date(ms - 5 * HOUR).toISOString().replace(/\.\d+Z$/, "-05:00");
const keeperStamp = (ms) => new Date(ms - 5 * HOUR).toISOString().replace(/\.\d+Z$/, "-0500");

let seed = 20261124;
const rand = () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const pick = (list) => list[Math.floor(rand() * list.length)];

function write(path, content) {
  const full = join(ROOT, path);
  mkdirSync(dirname(full), { recursive: true });
  writeFileSync(full, typeof content === "string" ? content : `${JSON.stringify(content, null, 1)}\n`);
}

for (const dir of ["hpg", "mac"]) rmSync(join(ROOT, dir), { recursive: true, force: true });
write("demo.json", { _demo: DEMO, now: iso(NOW) });

// ---------------------------------------------------------------- HiPerGator: stages and attempts

let nextJob = 90000001;
const stages = [];

/** attempts: [{ sub, start, end, state, exit }] with times in ms; a missing end is still running. */
function stage(id, experiment, account, attempts, extra = {}) {
  stages.push({ id, experiment, account, gpus: 1, kind: "normal", depends_on: ["preflight"], gate: null, estimate_gpu_hours: 25.6, ...extra, attempts: attempts.map((a) => ({ ...a, job: String(nextJob++) })) });
}
const ran = (sub, queueMin, hours, state = "COMPLETED", exit = 0) => ({ sub, start: sub + queueMin * MIN, end: sub + queueMin * MIN + hours * HOUR, state, exit });
const running = (sub, queueMin) => ({ sub, start: sub + queueMin * MIN });
const queued = (sub) => ({ sub });

// A training runs past the 24-hour limit: the stage checkpoints (exit 10) and the watcher
// resubmits it, so one stage carries several attempts.
const continued = (sub, queueMin, tailHours) => {
  const first = ran(sub, queueMin, 23.83, "FAILED", 10);
  return [first, ran(first.end + 14 * MIN, 5, tailHours)];
};

// E8: four seeds of the primary recipe.
for (let s = 1; s <= 4; s++) stage(`e8-seed${s}`, "E8", "cai6734", continued(at("11-19T17:40") + s * 4 * MIN, 12 + s * 6, 1.7 + s * 0.1));

// E10: five cross-fitted fold policies, then the cohort X rollouts.
for (let f = 1; f <= 5; f++) stage(`e10-fold${f}`, "E10", f <= 4 ? "cai6734" : "cai5724", continued(at("11-19T18:05") + f * 5 * MIN, 20 + f * 4, 1.9));
stage("e10-rollouts-x", "E10", "cai6734", [ran(at("11-22T10:00"), 9, 4.2)], { depends_on: ["e10-fold1", "e10-fold5"], estimate_gpu_hours: 4.2 });

// E9: three seeds on the second backbone, with the retries a real cluster produces.
stage("e9-seed1", "E9", "cai5724", [ran(at("11-19T18:20"), 25, 6.1, "NODE_FAIL", 1), ran(at("11-20T01:05"), 18, 23.83, "FAILED", 10), ran(at("11-21T01:30"), 6, 0.4, "OUT_OF_MEMORY", 40), ran(at("11-21T02:10"), 9, 8.6)], { estimate_gpu_hours: 34 });
stage("e9-seed2", "E9", "cai5724", [ran(at("11-19T18:30"), 40, 11.2, "PREEMPTED", 1), ran(at("11-20T06:40"), 55, 23.83, "FAILED", 10), ran(at("11-21T07:45"), 12, 9.1)], { estimate_gpu_hours: 34 });
stage("e9-seed3", "E9", "cai5724", [ran(at("11-22T20:10"), 30, 24, "TIMEOUT", 1), running(at("11-23T21:00"), 14)], { estimate_gpu_hours: 34 });

// E1 and E3 run on CPU.
for (const [part, start, hours] of [["a", "11-20T09:00", 5.5], ["b", "11-20T15:00", 19], ["c", "11-21T11:00", 7.5], ["d", "11-21T19:30", 26]]) {
  stage(`e1-${part}`, "E1", "cai6734", [ran(at(start), 2, hours)], { gpus: 0, estimate_gpu_hours: 0, depends_on: [] });
}
stage("e3-crossfit", "E3", "cai6734", [running(at("11-23T20:00"), 3)], { gpus: 0, estimate_gpu_hours: 0, depends_on: [] });

// E2: the reader panel, released by gate G2a.
for (let r = 1; r <= 9; r++) {
  const id = `e2-reader-0${r}`;
  const extra = { gate: "G2a", depends_on: ["e1-d"], estimate_gpu_hours: 10 };
  if (r <= 3) stage(id, "E2", "cai6734", [ran(at("11-23T02:10") + r * 6 * MIN, 10 + r * 3, 8.4 + r * 0.7)], extra);
  else if (r <= 7) stage(id, "E2", "cai6734", [running(at("11-24T05:30") + r * 9 * MIN, 12 + r * 2)], extra);
  else stage(id, "E2", "cai6734", [queued(at("11-24T13:40"))], extra);
}

// E7: public datasets. One dataset exhausted its retry; one stage was refused.
stage("e7-ckd", "E7", "cai5724", [ran(at("11-23T03:00"), 14, 11.3)], { gate: "G2a", estimate_gpu_hours: 12 });
stage("e7-hepc", "E7", "cai5724", [running(at("11-24T03:15"), 21)], { gate: "G2a", estimate_gpu_hours: 12 });
stage("e7-pima", "E7", "cai5724", [running(at("11-24T08:50"), 34)], { gate: "G2a", estimate_gpu_hours: 12 });
stage("e7-ddxplus", "E7", "cai5724", [ran(at("11-23T03:20"), 22, 2.2, "FAILED", 1), ran(at("11-23T06:05"), 17, 2.4, "FAILED", 1)], { gate: "G2a", estimate_gpu_hours: 60, final: "FAILED" });
stage("e4-probe-bench", "E4", "cai6734", [{ sub: at("11-24T09:45"), end: at("11-24T09:46"), refused: true, exit: 20 }], { gate: "G2a", estimate_gpu_hours: 3, final: "REFUSED" });

const KNOWN_UNTIL = NOW - 6 * MIN; // the watcher's last cycle
const gpuHours = (a, s) => (a.start ? (((a.end ?? KNOWN_UNTIL) - a.start) / HOUR) * s.gpus : 0);

function stageStatus(s) {
  const last = s.attempts[s.attempts.length - 1];
  if (s.final) return s.final;
  if (last.end == null) return last.start == null ? "QUEUED" : "RUNNING";
  return last.exit === 0 ? "COMPLETED" : "INTERRUPTED";
}

const runlog = [];
for (const s of stages) {
  s.attempts.forEach((a, i) => {
    const base = { stage: s.id, experiment: s.experiment, attempt: i + 1, job_id: a.job, account: s.account };
    runlog.push({ time: a.sub, event: i === 0 ? "submitted" : "resubmitted", ...base });
    if (a.refused) return runlog.push({ time: a.end, event: "refused", ...base, exit_code: a.exit });
    if (a.start != null) runlog.push({ time: a.start, event: "started", ...base });
    if (a.end != null) {
      runlog.push({ time: a.end, event: "ended", ...base, state: a.state, exit_code: a.exit, elapsed_s: Math.round((a.end - a.start) / 1000), gpu_hours: Number(gpuHours(a, s).toFixed(2)) });
    }
  });
}

const gateRecords = [
  { id: "G0", at: at("11-18T16:30"), result: "met", path: "M1 and M2 proceed" },
  { id: "G1", at: at("11-19T17:15"), result: "met", path: "E1, E3 and the training track proceed" },
  { id: "G2a", at: at("11-23T01:45"), result: "met", path: "The measurement tier runs: E2, E4, E7" },
  { id: "G2b", at: at("11-23T01:45"), result: "met", path: "Method arms stay eligible" },
  { id: "G2c", at: at("11-23T01:45"), result: "not met", path: "CEE-Scale is reported from the twin as found and not applied elsewhere" },
];
for (const g of gateRecords) runlog.push({ time: g.at, event: "gate_decided", stage: g.id, gate: g.id, result: g.result });
runlog.push({ time: at("11-23T08:45"), event: "blocked", stage: "e7-ddxplus", experiment: "E7" });
runlog.push({ time: at("11-24T09:46"), event: "blocked", stage: "e4-probe-bench", experiment: "E4" });
runlog.sort((a, b) => a.time - b.time);
write("hpg/state/runlog.jsonl", `${runlog.map((l) => JSON.stringify({ ...l, time: iso(l.time) })).join("\n")}\n`);

const LIMITS = { E2: [90, 200], E4: [180, 400], E5: [400, 800], E6: [145, 300], E7: [150, 300], E8: [105, 210], E9: [120, 300], E10: [133, 270] };
const experimentIds = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11"];
const experiments = experimentIds.map((id) => {
  const mine = stages.filter((s) => s.experiment === id);
  const states = mine.map(stageStatus);
  let status = "PLANNED";
  if (id === "E11") status = "BLOCKED";
  else if (states.length && states.every((s) => s === "COMPLETED")) status = "COMPLETED";
  else if (states.some((s) => s === "RUNNING")) status = "RUNNING";
  else if (states.some((s) => s === "REFUSED")) status = "BLOCKED";
  return {
    id,
    status,
    estimate_gpu_hours: LIMITS[id]?.[0] ?? 0,
    allowance_gpu_hours: LIMITS[id]?.[1] ?? 0,
    spent_gpu_hours: Number(mine.reduce((sum, s) => sum + s.attempts.reduce((x, a) => x + gpuHours(a, s), 0), 0).toFixed(1)),
  };
});

const statusStages = stages.map((s) => {
  const last = s.attempts[s.attempts.length - 1];
  return {
    id: s.id,
    experiment: s.experiment,
    kind: s.kind,
    status: stageStatus(s),
    depends_on: s.depends_on,
    gate: s.gate,
    attempts: s.attempts.length,
    job_id: last.job,
    account: s.account,
    gpus: s.gpus,
    submitted_at: iso(last.sub),
    started_at: last.start ? iso(last.start) : null,
    ended_at: last.end ? iso(last.end) : null,
    elapsed_s: last.start ? Math.round(((last.end ?? KNOWN_UNTIL) - last.start) / 1000) : 0,
    gpu_hours: Number(s.attempts.reduce((x, a) => x + gpuHours(a, s), 0).toFixed(2)),
    exit_code: last.end ? last.exit : null,
    last_state: last.refused ? "REFUSED" : last.end ? last.state : last.start ? "RUNNING" : "PENDING",
    estimate_gpu_hours: s.estimate_gpu_hours,
  };
});

const counts = { PLANNED: 14, QUEUED: 0, RUNNING: 0, COMPLETED: 0, FAILED: 0, REFUSED: 0, BLOCKED: 1, "SKIPPED BY GATE": 0, INTERRUPTED: 0 };
for (const s of statusStages) counts[s.status] += 1;
const inUse = { cai6734: 0, cai5724: 0 };
for (const s of statusStages) if (s.status === "RUNNING") inUse[s.account] += s.gpus;

write("hpg/state/status.json", {
  _demo: DEMO,
  schema: "tii-status/1",
  generated_at: iso(KNOWN_UNTIL),
  milestone: "M2",
  watcher: { cycle: 1052, started_at: iso(KNOWN_UNTIL - 40e3), finished_at: iso(KNOWN_UNTIL), interval_s: 900, lock: "ok", completed: false },
  authorization: { ok: true, sha256_ok: true, signature_ok: true, expires: "2027-03-31", addenda: [{ name: "AUTHORIZATION_ADDENDUM_1.md", sha256_ok: true, signed: true }] },
  gpu: { limit_by_account: { cai6734: 8, cai5724: 8 }, in_use_by_account: inUse },
  counts,
  experiments,
  stages: statusStages,
  gates: ["G0", "G1", "G2a", "G2b", "G2c", "G3", "G4", "G5"].map((id) => {
    const g = gateRecords.find((x) => x.id === id);
    return { id, decided: Boolean(g), result: g?.result ?? null, path_taken: g?.path ?? null, decided_at: g ? iso(g.at) : null };
  }),
  blocked: [
    { id: "BLOCKED-002", title: "DEMO no second cohort is listed for E11", opened_at: iso(at("11-13T03:00")), status: "OPEN" },
    { id: "BLOCKED-004", title: "DEMO manifest mismatch refused e4-probe-bench", opened_at: iso(at("11-24T09:46")), status: "OPEN" },
  ],
});

const progress = [
  ["e2-reader-04", "patient", 612, 1000, 2],
  ["e2-reader-05", "patient", 540, 1000, 1],
  ["e2-reader-06", "patient", 471, 1000, 3],
  ["e2-reader-07", "patient", 398, 1000, 1],
  ["e9-seed3", "step", 4100, 6000, 2],
  ["e7-hepc", "patient", 388, 615, 4],
  ["e7-pima", "patient", 301, 768, 1],
  ["e3-crossfit", "fold", 3, 5, 38],
];
for (const [id, unit, done, total, minutesAgo] of progress) {
  write(`hpg/state/progress/${id}.json`, { _demo: DEMO, stage: id, unit, done, total, updated_at: iso(KNOWN_UNTIL - minutesAgo * MIN) });
}

for (const g of gateRecords) {
  write(`hpg/gates/${g.id}.json`, { _demo: DEMO, gate: g.id, decided_at: iso(g.at), rule: "see specification 14.1", result: g.result, path_taken: g.path, inputs: [{ name: "demo-input", sha256: "0".repeat(64) }] });
}

const events = [
  [at("11-23T01:45"), "gate decided", "G2a met; G2b met; G2c not met"],
  [at("11-23T08:45"), "failure", "e7-ddxplus failed after 2 attempts"],
  [at("11-23T15:10"), "stage completed", "e7-ckd"],
  [at("11-24T07:05"), "stage completed", "e2-reader-03"],
  [at("11-24T09:46"), "blocked", "BLOCKED-004 e4-probe-bench refused, exit 20"],
  [at("11-24T13:40"), "stage completed", "2 stages queued: e2-reader-08, e2-reader-09"],
];
write("hpg/state/events.tsv", `${events.map(([t, kind, text]) => `${iso(t)}\t${kind}\t${text}`).join("\n")}\n`);
write("hpg/export/approved/E1/summary.json", { _demo: DEMO, experiment: "E1", note: "fictional aggregate" });
write("hpg/export/approved/E8/policies.json", { _demo: DEMO, experiment: "E8", note: "fictional aggregate" });
write("hpg/export/approved/E10/cohort_x_counts.json", { _demo: DEMO, experiment: "E10", note: "fictional aggregate" });

// ---------------------------------------------------------------- Mac: Figure 1 files

const readers = [
  ["sim-entrenching-drift", "simulated", "E1", 1.05, 0.21, -0.19],
  ["sim-fixed-class-drift", "simulated", "E1", 0.96, 0.15, -0.13],
  ["sim-over-reactive", "simulated", "E1", 3.4, 0.03, 0.09],
  ["sim-under-reactive", "simulated", "E1", 0.36, 0.02, 0.04],
  ["sim-double-counting", "simulated", "E1", 2.3, 0.08, 0.06],
  ["demo-lm-1b", "language_model", "E2", 4.6, 0.07, 0.05, 1],
  ["demo-lm-3b", "language_model", "E2", 3.1, 0.1, 0.02, 3],
  ["demo-lm-8b", "language_model", "E2", 1.9, 0.13, -0.06, 8],
  ["demo-lm-14b", "language_model", "E2", 1.4, 0.16, -0.11, 14],
  ["demo-policy-seed1", "trained_policy", "E8", 1.6, 0.11, -0.07],
  ["demo-policy-seed2", "trained_policy", "E8", 1.75, 0.09, -0.05],
  ["demo-backbone-2", "second_backbone", "E9", 2.2, 0.06, -0.01],
];
write("mac/figure1/regime_map.json", {
  _demo: DEMO,
  schema: "tii-fig1-regime/1",
  generated_at: iso(at("11-24T11:30")),
  harmful_region: [{ tau_b: 2.1, kl_inc: 0 }, { tau_b: 9, kl_inc: 0 }, { tau_b: 9, kl_inc: 0.12 }, { tau_b: 4, kl_inc: 0.1 }, { tau_b: 2.6, kl_inc: 0.05 }],
  markers: readers.map(([id, kind, experiment, tau, kl, delta, params]) => ({
    id,
    label: id,
    experiment,
    kind,
    tau_b: tau,
    tau_b_lo: Number((tau / 1.12).toFixed(3)),
    tau_b_hi: Number((tau * 1.12).toFixed(3)),
    kl_inc: kl,
    kl_inc_lo: Number((kl * 0.8).toFixed(3)),
    kl_inc_hi: Number((kl * 1.2).toFixed(3)),
    ...(params ? { params_b: params } : {}),
    delta_log_loss: delta,
  })),
});
write("mac/figure1/gain_accounting.json", {
  _demo: DEMO,
  schema: "tii-fig1-gain/1",
  generated_at: iso(at("11-24T11:30")),
  cells: readers.map(([id, , experiment, , kl, delta]) => {
    const predicted = Number((kl - 0.04 + (rand() - 0.5) * 0.02).toFixed(3));
    const realized = Number((-delta + (rand() - 0.5) * 0.03).toFixed(3));
    return { id, reader: id, setting: experiment === "E1" ? "twin, W2" : "twin text", experiment, predicted_gain: predicted, predicted_lo: predicted - 0.02, predicted_hi: predicted + 0.02, realized_gain: realized, realized_lo: realized - 0.03, realized_hi: realized + 0.03 };
  }),
});
write("mac/figure1/regret_attribution.json", {
  _demo: DEMO,
  schema: "tii-fig1-regret/1",
  generated_at: iso(at("11-24T11:30")),
  agents: [
    { id: "drift", label: "drift", variant: "raw", selection: 0.22, reading: 0.61, commitment: 0.17 },
    { id: "drift", label: "drift", variant: "repaired", selection: 0.48, reading: 0.2, commitment: 0.32 },
    { id: "scale", label: "scale", variant: "raw", selection: 0.18, reading: 0.52, commitment: 0.3 },
    { id: "scale", label: "scale", variant: "repaired", selection: 0.31, reading: 0.4, commitment: 0.29 },
  ],
});

// ---------------------------------------------------------------- Mac: study documents that change

write(
  "mac/docs/BLOCKED.md",
  `# BLOCKED items (FICTIONAL demo fixture)

## BLOCKED-002: DEMO no second cohort is listed for E11 (opened November 13, 2026)

- What waits: E11 only.
- Why it is blocked: fictional demo text. No other cohort is listed.
- Request, one step: fictional demo text. Add a cohort in a new authorization, or do nothing.
- Meanwhile: E7 and E2 are the evidence beyond the study's own agent.
- Status: OPEN.

## BLOCKED-003: DEMO a closed example (opened November 20, 2026)

- What waits: nothing now.
- Status: CLOSED November 21, 2026. Fictional demo text.

## BLOCKED-004: DEMO manifest mismatch refused e4-probe-bench (opened November 24, 2026)

- What waits: E4's probe bench, and so E4.
- Why it is blocked: fictional demo text. The stage exited 20 because a bound artifact changed.
- Request, one step: fictional demo text. Confirm which checkpoint digest is the intended one.
- Meanwhile: E2 and E7 continue; E4 waits.
- Status: OPEN.
`,
);
write(
  "mac/docs/STATUS.md",
  `# STATUS (FICTIONAL demo fixture)

Updated November 24, 2026, about 1:55 PM EST. Fictional demo text.

## 1. Where the study is

- Milestone: M0 and M1 COMPLETE. M2 IN PROGRESS. M3 IN PROGRESS. M4 and M5 NOT STARTED.
`,
);
write(
  "mac/docs/RESEARCH_LEDGER.md",
  `# Research ledger (FICTIONAL demo fixture)

## 4. Experiment register

| ID | Status | Card | Notes |
|---|---|---|---|
| E0 | COMPLETED (VERIFIED_CODE, PASS) | EXPERIMENT_CARDS/E0.md | Fictional demo note |
| E1 (A to D) | COMPLETED | EXPERIMENT_CARDS/E1.md | Fictional demo note |
| E2 | RUNNING | EXPERIMENT_CARDS/E2.md | Fictional demo note |
| E3 | RUNNING | EXPERIMENT_CARDS/E3.md | Fictional demo note |
| E4 | PLANNED | EXPERIMENT_CARDS/E4.md | Fictional: waits on E9 and BLOCKED-004 |
| E5 | PLANNED | Frozen at G3 | Waits on G3 |
| E6 | PLANNED | Frozen at G4 | Waits on G4 |
| E7 | RUNNING | EXPERIMENT_CARDS/E7.md | Fictional demo note |
| E8 | COMPLETED | EXPERIMENT_CARDS/E8.md | Fictional demo note |
| E9 | RUNNING | EXPERIMENT_CARDS/E9.md | Fictional demo note |
| E10 | COMPLETED | EXPERIMENT_CARDS/E10.md | Fictional demo note |
| E11 | BLOCKED | None | Fictional: BLOCKED-002 |

## 5. Job ledger
`,
);
write("mac/docs/GATE_DECISIONS/G0.md", "# Gate G0: M0 acceptance (FICTIONAL demo fixture)\n\nEvaluated November 18, 2026, about 4:30 PM EST. Fictional demo text.\n\n## Result\n\nMET.\n\n## Path taken\n\nM1 and M2 proceed.\n");

// ---------------------------------------------------------------- Mac: connection

const keeper = [];
for (let t = NOW - 26 * HOUR; t <= NOW - 2 * MIN; t += 5 * MIN + Math.round(rand() * 4000)) {
  const line = (text, dt = 0) => keeper.push(`${keeperStamp(t + dt)} ${text}`);
  const hoursAgo = (NOW - t) / HOUR;
  if (hoursAgo < 12 && hoursAgo > 8.6) continue; // the Mac slept: no lines at all
  if (hoursAgo < 5.0 && hoursAgo > 4.92) line("master present but did not answer (1 in a row)");
  else if (hoursAgo <= 4.92 && hoursAgo > 4.83) line("master present but did not answer (2 in a row)");
  else if (hoursAgo <= 4.83 && hoursAgo > 4.75) {
    line("closed unresponsive master");
    line("notified: sign-in needed", 1000);
  } else if (hoursAgo <= 4.75 && hoursAgo > 4.0) line("down, already notified");
  else if (hoursAgo <= 4.0 && hoursAgo > 3.91) line("reconnected");
  else {
    line("ok");
    if (rand() < 0.04) line("relayed 2 event(s)", 2000);
  }
}
write("mac/keeper.log", `${keeper.join("\n")}\n`);
write("mac/ssh-check.txt", "Master running (pid=4242)\n");
const launchd = (label, state, extra) => `gui/501/${label} = {\n\tactive count = ${state === "running" ? 1 : 0}\n\tpath = /Users/demo/Library/LaunchAgents/${label}.plist\n\ttype = LaunchAgent\n\tstate = ${state}\n\n${extra}\n\tresource coalition = {\n\t\tstate = active\n\t}\n}\n`;
write("mac/launchctl/com.tii.hpg-keeper.txt", launchd("com.tii.hpg-keeper", "not running", "\truns = 3140\n\tlast exit code = 0\n\trun interval = 300 seconds"));
write("mac/launchctl/com.tii.caffeinate.txt", launchd("com.tii.caffeinate", "running", "\tpid = 4343\n\truns = 1"));
write("mac/launchctl/com.tii.local-loop.txt", launchd("com.tii.local-loop", "not running", "\truns = 1571\n\tlast exit code = 0\n\trun interval = 600 seconds"));

// ---------------------------------------------------------------- Mac: git

const commits = [];
const AREA_FILES = { "tii-cee/": ["tiicee/cee/projection.py", "tiicee/twin/env.py", "tiicee/pipeline/watcher.py", "tests/test_cee_identity.py"], "reports/research-advancement/": ["RESEARCH_LEDGER.md", "STATUS.md", "EXPERIMENT_CARDS/E5.md"], "monitor/": ["src/App.svelte", "src-tauri/src/remote.rs"] };
for (let t = at("11-13T02:40"), i = 0; t < NOW - 20 * MIN; t += (2 + rand() * 9) * HOUR, i++) {
  const hash = Array.from({ length: 40 }, () => "0123456789abcdef"[Math.floor(rand() * 16)]).join("");
  const areas = Object.keys(AREA_FILES).filter(() => rand() < 0.55);
  if (!areas.length) areas.push("reports/research-advancement/");
  const numstat = areas.map((a) => `${Math.round(20 + rand() * (a === "tii-cee/" ? 700 : 260))}\t${Math.round(rand() * 90)}\t${a}${pick(AREA_FILES[a])}`);
  commits.push(`\u001e${hash}\u001f${Math.round(t / 1000)}\u001fdemo: fictional commit ${i + 1}\n\n${numstat.join("\n")}\n`);
}
write("mac/git/log.txt", commits.reverse().join(""));
write("mac/git/diff-numstat.txt", "84\t12\ttii-cee/tiicee/pipeline/figures.py\n31\t4\treports/research-advancement/STATUS.md\n");
write("mac/git/untracked-numstat.txt", "412\t0\ttii-cee/tiicee/analysis/e2_panel.py\n96\t0\ttii-cee/tests/test_e2_panel.py\n");

// ---------------------------------------------------------------- Mac: Claude Code transcripts

const SESSION = "0000demo-0000-4000-8000-000000000001";
const TOOLS = [
  ["Bash", (i) => ({ command: `uv run pytest tests/test_demo_${i}.py -q`, description: pick(["Run the E2 panel tests", "Rebuild the regime map figure", "Check the ledger against the manifests", "Run the fixture suite"]) })],
  ["Read", () => ({ file_path: `/Users/demo/repo/tii-cee/tiicee/${pick(["cee/projection.py", "analysis/e2_panel.py", "twin/env.py", "pipeline/figures.py"])}` })],
  ["Edit", () => ({ file_path: `/Users/demo/repo/tii-cee/tiicee/${pick(["analysis/e2_panel.py", "pipeline/figures.py"])}`, old_string: "a", new_string: "b" })],
  ["Write", () => ({ file_path: `/Users/demo/repo/reports/research-advancement/${pick(["STATUS.md", "RESEARCH_LEDGER.md"])}`, content: "fictional" })],
  ["Grep", () => ({ pattern: pick(["tau_B", "KL_inc", "realized_gain"]), path: "/Users/demo/repo/tii-cee" })],
  ["SendMessage", () => ({ to: "team-lead", summary: "Fictional demo status update", message: "fictional" })],
];
const WEIGHTS = [5, 4, 3, 1, 2, 0.4];

function transcript(path, { model, from, to, rate, inflight, errorRate = 0.03, title }) {
  const lines = [];
  if (title) lines.push({ type: "ai-title", aiTitle: title, sessionId: SESSION });
  let n = 0;
  for (let t = from; t < to; t += (15 + rand() * rate) * 1000) {
    let r = rand() * WEIGHTS.reduce((a, b) => a + b, 0);
    const k = WEIGHTS.findIndex((w) => (r -= w) < 0);
    const [name, input] = TOOLS[Math.max(0, k)];
    const id = `toolu_demo_${path.length}_${n++}`;
    lines.push({ type: "assistant", timestamp: new Date(t).toISOString(), sessionId: SESSION, message: { model, role: "assistant", content: [{ type: "tool_use", id, name, input: input(n) }] } });
    const took = (name === "Bash" ? 4 + rand() * 40 : 0.3 + rand() * 2) * 1000;
    lines.push({ type: "user", timestamp: new Date(Math.min(t + took, to)).toISOString(), sessionId: SESSION, message: { role: "user", content: [{ type: "tool_result", tool_use_id: id, content: "fictional output, never read by the monitor", is_error: rand() < errorRate }] } });
    t += took;
  }
  if (inflight) {
    lines.push({ type: "assistant", timestamp: new Date(to - 95e3).toISOString(), sessionId: SESSION, message: { model, role: "assistant", content: [{ type: "tool_use", id: "toolu_demo_inflight", name: "Bash", input: { command: "uv run python -m tiicee.pipeline.figures --panel a", description: inflight } }] } });
  }
  write(path, `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
}

const sub = (name) => `mac/transcripts/${SESSION}/subagents/agent-a${name}-00000000demo0000`;
transcript(`mac/transcripts/${SESSION}.jsonl`, { model: "claude-demo-lead", from: NOW - 3.2 * HOUR, to: NOW - 40e3, rate: 240, title: "DEMO lead session" });
for (const [name, description, opts] of [
  ["demo-e2-analysis", "Fictional worker: analyse the E2 reader panel", { model: "claude-demo-large", from: NOW - 2.4 * HOUR, to: NOW - 20e3, rate: 60 }],
  ["demo-paper-figures", "Fictional worker: build Figure 1 from manifests", { model: "claude-demo-large", from: NOW - 70 * MIN, to: NOW - 5e3, rate: 45, inflight: "Rebuild panel A of Figure 1" }],
  ["demo-ledger", "Fictional worker: keep the ledger in step with the run log", { model: "claude-demo-small", from: NOW - 2.9 * HOUR, to: NOW - 6.5 * MIN, rate: 150 }],
  ["demo-reviewer", "Fictional worker: review round one", { model: "claude-demo-small", from: NOW - 3.1 * HOUR, to: NOW - 52 * MIN, rate: 90, errorRate: 0.14 }],
]) {
  transcript(`${sub(name)}.jsonl`, opts);
  write(`${sub(name)}.meta.json`, { agentType: name, name, description, model: "demo" });
}

write(
  "README.md",
  `# Demo fixtures

Everything under this folder is fictional. \`generate.mjs\` invents the jobs, the numbers, the worker
activity and the dates, in the formats of \`../DATA_CONTRACT.md\`, so that \`TII Monitor --demo\` can show
every panel in every state. None of it is study data, and the monitor reads it only in demo mode.

- \`hpg/\` mirrors the study root on HiPerGator: \`state/\`, \`gates/\`, \`export/approved/\`.
- \`mac/\` stands in for the Mac's own sources: transcripts, keeper log, launchd output, git output,
  the study documents that change, and the three Figure 1 files.
- \`demo.json\` holds the demo's clock.

Regenerate with \`pnpm fixtures\`.
`,
);
console.log(`wrote fixtures: ${stages.length} stages, ${runlog.length} run log lines, ${keeper.length} keeper lines, ${commits.length} commits`);
