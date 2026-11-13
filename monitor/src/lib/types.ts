// Shapes sent by the Rust side (src-tauri/src). Cluster files are passed through as written by
// the watcher, so every field of those is optional here and read defensively.

export interface Call {
  t_ms: number;
  end_ms: number | null;
  tool: string;
  kind: "read" | "write" | "run" | "talk" | "other";
  label: string;
  detail: string;
  error: boolean;
}

export interface FileTouch {
  path: string;
  reads: number;
  writes: number;
  last_ms: number;
}

export interface Worker {
  id: string;
  session: string;
  is_lead: boolean;
  name: string;
  description: string;
  model: string | null;
  state: "working" | "idle" | "quiet";
  first_ms: number | null;
  last_ms: number | null;
  current: Call | null;
  last_call: Call | null;
  calls_total: number;
  calls_in_window: number;
  errors_total: number;
  kinds: [number, number, number, number, number];
  files: FileTouch[];
  files_total: number;
  /** Per time bucket: calls by kind [read, write, run, talk, other] and errors. */
  buckets: number[][];
  samples: Record<string, string[]>;
  recent: Call[];
  bad_lines: number;
  bytes: number;
  file: string;
}

export interface Workers {
  window_s: number;
  bucket_s: number;
  start_ms: number;
  end_ms: number;
  workers: Worker[];
  dir: string;
  dir_exists: boolean;
  older_sessions_hidden: number;
  scanned_ms: number | null;
}

export interface KeeperEntry {
  t_ms: number;
  kind: "ok" | "reconnected" | "relayed" | "no_answer" | "closed" | "signin" | "down" | "other";
  text: string;
}

export interface LaunchdJob {
  label: string;
  installed: boolean;
  state: string;
  pid: number | null;
  last_exit: number | null;
  runs: number | null;
  interval_s: number | null;
  note: string;
}

export interface Link {
  master_up: boolean;
  master_detail: string;
  keeper: KeeperEntry[];
  keeper_log: string;
  keeper_log_exists: boolean;
  keeper_log_mtime_ms: number | null;
  keeper_unparsed: number;
  launchd: LaunchdJob[];
}

export interface Delta {
  ins: number;
  del: number;
  files: number;
}

export interface Commit {
  hash: string;
  t_ms: number;
  subject: string;
  areas: [Delta, Delta, Delta, Delta];
}

export interface Git {
  ok: boolean;
  error: string;
  branch: string;
  head_branch: string;
  start_commit: string;
  areas: [string, string, string, string];
  commits: Commit[];
  modified: [Delta, Delta, Delta, Delta];
  untracked: [Delta, Delta, Delta, Delta];
}

export interface FileInfo {
  path: string;
  exists: boolean;
  mtime_ms: number | null;
}

export interface SpecExperiment {
  id: string;
  question: string;
  design: string;
  measurement: string;
  rule: string;
  kills: string;
  compute: string;
}

export interface SpecGate {
  id: string;
  rule: string;
  if_met: string;
  if_not: string;
}

export interface GpuRow {
  label: string;
  counted: string;
  estimate: number | null;
  allowance: number | null;
  basis: string;
  authorized: string;
}

export interface BlockedItem {
  id: string;
  title: string;
  opened: string;
  open: boolean;
  status: string;
  waits: string;
  why: string;
  request: string;
  meanwhile: string;
}

export interface Card {
  id: string;
  title: string;
  frozen: string;
  question: string;
  path: string;
  mtime_ms: number | null;
}

export interface FigureFile {
  name: string;
  path: string;
  exists: boolean;
  mtime_ms: number | null;
  json: any;
  error: string;
}

export interface GateRecord {
  name: string;
  mtime_ms: number | null;
  title: string;
  evaluated: string;
  result: string;
  path: string;
}

export interface GpuLimit {
  account: string;
  gpus: number;
}

export interface Docs {
  spec: FileInfo;
  spec_version: string;
  spec_sha256: string;
  experiments: SpecExperiment[];
  gates: SpecGate[];
  gpu_estimates: GpuRow[];
  plan: { date: string; what: string }[];
  authorization: FileInfo;
  allowances: GpuRow[];
  addendum: FileInfo;
  gpu_limits: GpuLimit[];
  blocked_file: FileInfo;
  blocked: BlockedItem[];
  status_file: FileInfo;
  status_updated: string;
  status_milestone_line: string;
  milestones: { id: string; state: string }[];
  ledger_file: FileInfo;
  register: { id: string; status: string; card: string; notes: string }[];
  cards: Card[];
  gate_records_dir: FileInfo;
  gate_records: GateRecord[];
  figure: FigureFile[];
  problems: string[];
}

// state/status.json, schema tii-status/1 (monitor/DATA_CONTRACT.md)
export interface StageStatus {
  id?: string;
  experiment?: string;
  kind?: string;
  status?: string;
  depends_on?: string[];
  gate?: string | null;
  attempts?: number;
  job_id?: string | null;
  account?: string | null;
  gpus?: number;
  submitted_at?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  elapsed_s?: number | null;
  gpu_hours?: number | null;
  exit_code?: number | null;
  last_state?: string | null;
  estimate_gpu_hours?: number | null;
}

export interface ClusterStatus {
  schema?: string;
  generated_at?: string;
  milestone?: string;
  watcher?: {
    cycle?: number;
    started_at?: string;
    finished_at?: string;
    interval_s?: number;
    lock?: string;
    completed?: boolean;
  };
  authorization?: {
    ok?: boolean;
    sha256_ok?: boolean;
    signature_ok?: boolean;
    expires?: string;
    addenda?: { name?: string; sha256_ok?: boolean; signed?: boolean }[];
  };
  gpu?: { limit_by_account?: Record<string, number>; in_use_by_account?: Record<string, number> };
  counts?: Record<string, number>;
  experiments?: {
    id?: string;
    status?: string;
    estimate_gpu_hours?: number;
    allowance_gpu_hours?: number;
    spent_gpu_hours?: number;
  }[];
  stages?: StageStatus[];
  gates?: { id?: string; decided?: boolean; result?: string | null; path_taken?: string | null; decided_at?: string | null }[];
  blocked?: { id?: string; title?: string; opened_at?: string; status?: string }[];
}

export interface RunlogLine {
  time?: string;
  event?: string;
  stage?: string;
  experiment?: string;
  attempt?: number;
  job_id?: string;
  account?: string;
  state?: string;
  exit_code?: number | null;
  elapsed_s?: number | null;
  gpu_hours?: number | null;
  gate?: string;
  result?: string;
}

export interface JsonFile {
  name: string;
  size: number;
  mtime_ms: number;
  json: any;
}

export interface Remote {
  phase: "never_read" | "ok" | "not_started" | "error" | "";
  detail: string;
  last_attempt_ms: number | null;
  last_ok_ms: number | null;
  skipped: string | null;
  skipped_ms: number | null;
  status: ClusterStatus | null;
  status_note: string | null;
  status_mtime_ms: number | null;
  runlog: RunlogLine[];
  runlog_bad_lines: number;
  events: { time: string; kind: string; text: string }[];
  progress: Record<string, JsonFile>;
  gates: Record<string, JsonFile>;
  exports: { path: string; size: number; mtime_ms: number }[];
  exports_listed: boolean;
  calls: { command: string; exit: number | null; ms: number; bytes: number; note: string }[];
}

export interface RemoteMeta {
  host: string;
  root: string;
  reading: boolean;
  interval_s: number;
  min_interval_s: number;
  max_interval_s: number;
  next_read_in_s: number | null;
  manual_wait_s: number;
}

export interface Checked {
  transcripts: number | null;
  git: number | null;
  docs: number | null;
  keeper: number | null;
  master: number | null;
  launchd: number | null;
}

export interface StateReply {
  now_ms: number;
  demo: boolean;
  repo: string;
  study_start_ms: number;
  revs: { link: number; git: number; docs: number; remote: number };
  checked: Checked;
  workers: Workers;
  link: Link | null;
  git: Git | null;
  docs: Docs | null;
  remote: Remote | null;
  remote_meta: RemoteMeta;
}
