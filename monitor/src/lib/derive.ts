// Turns the raw sources into what the panels draw. Pure functions: no state, no fetching.

import type { ClusterStatus, Docs, GateRecord, Remote, RunlogLine, StageStatus } from "./types";
import { parseTime } from "./time";

const KNOWN_STATUS = ["SKIPPED BY GATE", "NOT STARTED", "IN PROGRESS", "COMPLETED", "COMPLETE", "PLANNED", "QUEUED", "RUNNING", "INTERRUPTED", "FAILED", "REFUSED", "BLOCKED"];

/** Splits "COMPLETED (VERIFIED_CODE, PASS)" into the status word and what follows it. */
export function canonical(raw: string | undefined): { status: string; rest: string } {
  const text = (raw ?? "").trim();
  const hit = KNOWN_STATUS.find((k) => text.toUpperCase().startsWith(k));
  if (!hit) return { status: text.toUpperCase() || "UNKNOWN", rest: "" };
  return { status: hit === "COMPLETE" ? "COMPLETED" : hit, rest: text.slice(hit.length).trim() };
}

export interface Resolved {
  /** One of the specification's status words, or "UNKNOWN" when no source names it. */
  status: string;
  /** Whatever the source wrote after the status word, e.g. "(VERIFIED_CODE, PASS)". */
  detail?: string;
  /** Which source said so, in words. */
  source: string;
  /** What the other source says, when the two differ. */
  other?: string;
  note?: string;
}

const norm = (id: string | undefined) => (id ?? "").toUpperCase().replace(/[^A-Z0-9]/g, "");

/** True when a register or status id ("E1 (A to D)", "E1-B", "e1b") belongs to experiment `id`. */
export function belongsTo(candidate: string | undefined, id: string): boolean {
  const c = norm(candidate);
  const want = norm(id);
  if (!c.startsWith(want)) return false;
  const next = c.charAt(want.length);
  return next === "" || !/[0-9]/.test(next);
}

/** Experiment status: the cluster's status.json when it names the experiment, else the ledger. */
export function experimentStatus(id: string, remote: Remote | null, docs: Docs | null): Resolved {
  const fromCluster = remote?.status?.experiments?.filter((e) => belongsTo(e.id, id)) ?? [];
  const row = docs?.register.find((r) => belongsTo(r.id, id));
  const ledger = row ? canonical(row.status) : null;
  if (fromCluster.length) {
    const status = combine(fromCluster.map((e) => canonical(e.status).status));
    return {
      status,
      source: "state/status.json on HiPerGator",
      other: row && ledger!.status !== status ? `RESEARCH_LEDGER.md says ${row.status}` : undefined,
      note: row?.notes,
    };
  }
  if (row) return { status: ledger!.status, detail: ledger!.rest, source: "RESEARCH_LEDGER.md, experiment register", note: row.notes };
  return { status: "UNKNOWN", source: "no source names this experiment yet" };
}

/** One status for several parts: anything live or wrong wins over planned and done. */
export function combine(statuses: string[]): string {
  const order = ["FAILED", "REFUSED", "BLOCKED", "RUNNING", "INTERRUPTED", "QUEUED", "PLANNED", "SKIPPED BY GATE", "COMPLETED"];
  const present = statuses.filter(Boolean);
  if (!present.length) return "UNKNOWN";
  if (present.every((s) => s === present[0])) return present[0];
  // Some finished and some not started: the experiment is under way.
  if (present.includes("COMPLETED") && present.every((s) => s === "COMPLETED" || s === "PLANNED" || s === "SKIPPED BY GATE")) {
    return present.includes("PLANNED") ? "RUNNING" : "COMPLETED";
  }
  return order.find((s) => present.includes(s)) ?? present[0];
}

export interface GateState {
  /** Decided whole: the cluster says so, or a record named exactly after the gate exists. */
  decided: boolean;
  /** Only partial records exist so far (for example G1-twin.md). */
  partial: boolean;
  result: string;
  path: string;
  decidedAt: number | null;
  source: string;
  /** Records in GATE_DECISIONS/ that belong to this gate. */
  records: GateRecord[];
}

const gateOf = (fileName: string) => fileName.replace(/\.[a-z]+$/i, "").split(/[-_. ]/)[0].toUpperCase();
const isWhole = (fileName: string) => fileName.replace(/\.[a-z]+$/i, "").toUpperCase() === gateOf(fileName);

export function gateState(id: string, remote: Remote | null, docs: Docs | null): GateState {
  const g = remote?.status?.gates?.find((x) => norm(x.id) === norm(id));
  const file = Object.values(remote?.gates ?? {}).find((f) => norm(f.name.replace(/\.json$/i, "")) === norm(id));
  const records = (docs?.gate_records ?? []).filter((r) => gateOf(r.name) === id.toUpperCase());
  const whole = records.find((r) => isWhole(r.name));
  const record = file?.json ?? {};
  const decided = Boolean(g?.decided) || Boolean(file) || Boolean(whole);
  const newest = records.reduce<number | null>((t, r) => (r.mtime_ms != null && (t == null || r.mtime_ms > t) ? r.mtime_ms : t), null);
  return {
    decided,
    partial: !decided && records.length > 0,
    result: String(g?.result ?? record.result ?? whole?.result ?? ""),
    path: String(g?.path_taken ?? record.path_taken ?? whole?.path ?? ""),
    decidedAt: parseTime(g?.decided_at ?? record.decided_at) ?? (file ? file.mtime_ms : newest),
    source: g?.decided
      ? "state/status.json on HiPerGator"
      : file
        ? `gates/${file.name} on HiPerGator`
        : records.length
          ? `GATE_DECISIONS/${records.map((r) => r.name).join(", ")} in the repository`
          : "no gate record exists yet",
    records,
  };
}

/** A gate result reads as met unless it says otherwise. Empty means not stated. */
export function gateMet(result: string, path: string): boolean | null {
  const text = `${result} ${path}`.toLowerCase();
  if (!text.trim()) return null;
  if (/\bnot[ _-]?met\b|\bfail|\bfalse\b|\bno\b|skipped/.test(text)) return false;
  if (/\bmet\b|\bpass|\btrue\b|\byes\b|proceed|eligible|run/.test(text)) return true;
  return null;
}

export interface Attempt {
  stage: string;
  experiment: string;
  attempt: number;
  jobId: string;
  account: string;
  submitted: number | null;
  started: number | null;
  ended: number | null;
  state: string;
  exitCode: number | null;
  elapsedS: number | null;
  gpuHours: number | null;
  refused: boolean;
  /** GPUs held while running, when known. */
  gpus: number | null;
}

export interface Marker {
  t: number;
  kind: "gate" | "blocked" | "refused";
  label: string;
}

export interface StageRow {
  id: string;
  experiment: string;
  status: string;
  stage: StageStatus | null;
  attempts: Attempt[];
  first: number | null;
  last: number | null;
}

export interface ClusterWork {
  rows: StageRow[];
  markers: Marker[];
  /** The latest moment the cluster's files speak for: status.json's generated_at. */
  knownUntil: number | null;
  first: number | null;
  attemptsWithoutGpus: number;
}

/** Job attempts from the append-only run log, completed with status.json where the log is silent. */
export function clusterWork(remote: Remote | null): ClusterWork {
  const status: ClusterStatus | null = remote?.status ?? null;
  const stages = new Map<string, StageStatus>();
  for (const s of status?.stages ?? []) if (s.id) stages.set(s.id, s);
  const knownUntil = parseTime(status?.generated_at) ?? remote?.status_mtime_ms ?? null;

  const byKey = new Map<string, Attempt>();
  const markers: Marker[] = [];
  const get = (line: RunlogLine): Attempt => {
    const stage = line.stage ?? "?";
    const attempt = line.attempt ?? 1;
    const key = `${stage}#${attempt}`;
    let a = byKey.get(key);
    if (!a) {
      a = {
        stage,
        experiment: line.experiment ?? stages.get(stage)?.experiment ?? "",
        attempt,
        jobId: "",
        account: "",
        submitted: null,
        started: null,
        ended: null,
        state: "",
        exitCode: null,
        elapsedS: null,
        gpuHours: null,
        refused: false,
        gpus: null,
      };
      byKey.set(key, a);
    }
    if (line.job_id) a.jobId = String(line.job_id);
    if (line.account) a.account = line.account;
    if (line.experiment) a.experiment = line.experiment;
    return a;
  };

  for (const line of remote?.runlog ?? []) {
    const t = parseTime(line.time);
    if (t == null) continue;
    switch (line.event) {
      case "submitted":
      case "resubmitted":
        get(line).submitted ??= t;
        break;
      case "started":
        get(line).started = t;
        break;
      case "ended": {
        const a = get(line);
        a.ended = t;
        a.state = (line.state ?? "").toUpperCase();
        a.exitCode = line.exit_code ?? null;
        a.elapsedS = line.elapsed_s ?? null;
        a.gpuHours = line.gpu_hours ?? null;
        if (a.started == null && a.elapsedS != null) a.started = t - a.elapsedS * 1000;
        break;
      }
      case "refused": {
        const a = get(line);
        a.refused = true;
        a.ended ??= t;
        a.exitCode = line.exit_code ?? a.exitCode;
        markers.push({ t, kind: "refused", label: `${line.stage ?? "stage"} refused` });
        break;
      }
      case "gate_decided":
        markers.push({ t, kind: "gate", label: `${line.gate ?? line.stage ?? "gate"} decided${line.result ? `: ${line.result}` : ""}` });
        break;
      case "blocked":
        markers.push({ t, kind: "blocked", label: `${line.stage ?? "item"} blocked` });
        break;
    }
  }

  // A stage that status.json shows in flight, but the run log has no line for yet.
  for (const s of stages.values()) {
    const submitted = parseTime(s.submitted_at);
    if (!s.id || submitted == null) continue;
    const attempt = s.attempts ?? 1;
    if (!byKey.has(`${s.id}#${attempt}`)) {
      byKey.set(`${s.id}#${attempt}`, {
        stage: s.id,
        experiment: s.experiment ?? "",
        attempt,
        jobId: s.job_id ?? "",
        account: s.account ?? "",
        submitted,
        started: parseTime(s.started_at),
        ended: parseTime(s.ended_at),
        state: s.ended_at ? (s.last_state ?? "").toUpperCase() : "",
        exitCode: s.exit_code ?? null,
        elapsedS: s.elapsed_s ?? null,
        gpuHours: s.gpu_hours ?? null,
        refused: false,
        gpus: null,
      });
    }
  }

  let attemptsWithoutGpus = 0;
  const rowsById = new Map<string, StageRow>();
  for (const a of byKey.values()) {
    const s = stages.get(a.stage) ?? null;
    if (s?.gpus != null) a.gpus = s.gpus;
    else if (a.gpuHours != null && a.elapsedS) a.gpus = Math.round(a.gpuHours / (a.elapsedS / 3600));
    else if (a.started != null) attemptsWithoutGpus += 1;
    if (!a.account && s?.account) a.account = s.account;
    let row = rowsById.get(a.stage);
    if (!row) {
      row = { id: a.stage, experiment: a.experiment || s?.experiment || "", status: (s?.status ?? "").toUpperCase(), stage: s, attempts: [], first: null, last: null };
      rowsById.set(a.stage, row);
    }
    row.attempts.push(a);
  }
  for (const row of rowsById.values()) {
    row.attempts.sort((x, y) => x.attempt - y.attempt);
    const times = row.attempts.flatMap((a) => [a.submitted, a.started, a.ended]).filter((t): t is number => t != null);
    row.first = times.length ? Math.min(...times) : null;
    row.last = times.length ? Math.max(...times) : null;
    if (!row.status) {
      const lastAttempt = row.attempts[row.attempts.length - 1];
      row.status = lastAttempt.refused ? "REFUSED" : lastAttempt.ended == null ? (lastAttempt.started == null ? "QUEUED" : "RUNNING") : "";
    }
  }
  const rank = (r: StageRow) => (r.status === "RUNNING" ? 0 : r.status === "QUEUED" ? 1 : 2);
  const rows = [...rowsById.values()].sort((a, b) => rank(a) - rank(b) || (b.last ?? 0) - (a.last ?? 0));
  const firsts = rows.map((r) => r.first).filter((t): t is number => t != null);
  return { rows, markers, knownUntil, first: firsts.length ? Math.min(...firsts) : null, attemptsWithoutGpus };
}

export interface Step {
  t: number;
  n: number;
}

/** GPUs in use over time for one account, as steps, from attempt start and end times. */
export function gpuSteps(work: ClusterWork, account: string): Step[] {
  const changes: [number, number][] = [];
  for (const row of work.rows) {
    for (const a of row.attempts) {
      if (a.account !== account || a.started == null || !a.gpus) continue;
      changes.push([a.started, a.gpus]);
      const end = a.ended ?? work.knownUntil;
      if (end != null && end > a.started) changes.push([end, -a.gpus]);
    }
  }
  changes.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  const steps: Step[] = [];
  let n = 0;
  for (const [t, delta] of changes) {
    n += delta;
    if (steps.length && steps[steps.length - 1].t === t) steps[steps.length - 1].n = n;
    else steps.push({ t, n });
  }
  return steps;
}

export interface Budget {
  id: string;
  label: string;
  spent: number | null;
  estimate: number | null;
  allowance: number | null;
  spentSource: string;
  limitSource: string;
}

/** GPU-hours per experiment: spent from the cluster, estimate and allowance from the cluster or AUTHORIZATION.md. */
export function budgets(remote: Remote | null, docs: Docs | null): Budget[] {
  const out: Budget[] = [];
  const cluster = remote?.status?.experiments ?? [];
  for (const row of docs?.allowances ?? []) {
    if (!row.allowance || /^total$/i.test(row.label)) continue;
    const match = row.label.match(/^E\d+/);
    const id = match ? match[0] : row.label.split(" ")[0];
    const fromCluster = cluster.filter((e) => belongsTo(e.id, id) || (!match && norm(e.id).startsWith(norm(id))));
    const spent = fromCluster.length ? fromCluster.reduce((sum, e) => sum + (e.spent_gpu_hours ?? 0), 0) : null;
    const estimate = fromCluster.find((e) => e.estimate_gpu_hours != null)?.estimate_gpu_hours ?? row.estimate;
    const allowance = fromCluster.find((e) => e.allowance_gpu_hours != null)?.allowance_gpu_hours ?? row.allowance;
    out.push({
      id,
      label: row.label,
      spent,
      estimate,
      allowance,
      spentSource: fromCluster.length ? "state/status.json on HiPerGator" : "no spending reported yet",
      limitSource: fromCluster.some((e) => e.allowance_gpu_hours != null) ? "state/status.json on HiPerGator" : "AUTHORIZATION.md section 4",
    });
  }
  return out;
}

export interface OpenBlocked {
  id: string;
  title: string;
  opened: string;
  waits: string;
  request: string;
  meanwhile: string;
  why: string;
  source: string;
}

/** Open BLOCKED items from BLOCKED.md and from the cluster's status.json, joined by id. */
export function openBlocked(remote: Remote | null, docs: Docs | null): OpenBlocked[] {
  const out = new Map<string, OpenBlocked>();
  for (const b of docs?.blocked ?? []) {
    if (b.open) out.set(b.id, { id: b.id, title: b.title, opened: b.opened, waits: b.waits, request: b.request, meanwhile: b.meanwhile, why: b.why, source: "BLOCKED.md" });
  }
  for (const b of remote?.status?.blocked ?? []) {
    if (!b.id || (b.status ?? "OPEN").toUpperCase() !== "OPEN") continue;
    const have = out.get(b.id);
    if (have) have.source = "BLOCKED.md and state/status.json on HiPerGator";
    else if (!docs?.blocked.some((d) => d.id === b.id)) {
      out.set(b.id, { id: b.id, title: b.title ?? "", opened: b.opened_at ?? "", waits: "", request: "", meanwhile: "", why: "", source: "state/status.json on HiPerGator (not yet in BLOCKED.md)" });
    }
  }
  return [...out.values()].sort((a, b) => a.id.localeCompare(b.id));
}

/** Stage counts for one experiment from status.json: done, live, total. */
export function stageCounts(id: string, remote: Remote | null): { done: number; live: number; total: number } | null {
  const stages = (remote?.status?.stages ?? []).filter((s) => belongsTo(s.experiment, id));
  if (!stages.length) return null;
  const up = (s: StageStatus) => (s.status ?? "").toUpperCase();
  return {
    done: stages.filter((s) => up(s) === "COMPLETED").length,
    live: stages.filter((s) => up(s) === "RUNNING").length,
    total: stages.length,
  };
}
