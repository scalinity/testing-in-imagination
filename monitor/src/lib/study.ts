// What is fixed about study TII-CEE-1 and is not data: how the plan board is laid out, which
// node waits on which, and what the study's words mean. Everything here is transcribed from
// reports/research-advancement/ (specification version 3.1, AUTHORIZATION.md, WATCHER_DESIGN.md)
// and cites its section. Statuses, dates' text, rules and numbers are read live, not from here.

import { easternMidnight } from "./time";

/** Term and its meaning, in plain words, as the study's documents define it. */
export const GLOSSARY = {
  cee: ["Conservation of expected evidence", "An agent's reactions to the results it imagines must average to what it believes now (spec 1.2)."],
  reader: ["Reader", "Any procedure that turns what has been observed so far into a probability for each disease (spec 5.1)."],
  worldModel: ["World model", "The model that predicts what a test would show before it is ordered. Imagined results are predictions, never observations (spec 5.1, 18)."],
  twin: ["Twin", "The synthetic environment where the truth is known, so quantities can be computed exactly instead of estimated (spec 8.1)."],
  drift: ["Drift", "The average of the reader's imagined reactions minus its current belief. A reader with drift moves in a direction that does not depend on what the result says (spec 5.2)."],
  responsiveness: ["Responsiveness", "How much the reader's belief moves across the results it imagines, in nats. At zero the reader does not move whatever the result says (spec 5.2)."],
  klInc: ["Location incoherence (KL_inc)", "How far the reader's imagined reactions are from conserving expected evidence, in nats per decision. Zero exactly when they already average to the current belief (spec 5.3)."],
  projection: ["Projection (CEE-Read)", "One weight per disease that makes the imagined reactions average back to the current belief. The same weights then correct how the real result is read. It fixes drift only, not the size of reactions (spec 5.3, 5.10)."],
  tauB: ["Budget-implied scale (tau_B)", "The size of the reader's reactions relative to the information the test can carry. 1 is matched; above 1 the reader reacts more than the test can justify; below 1, less (spec 5.5)."],
  budget: ["Information budget", "An estimate of how many nats a test's result can tell about the diagnosis, given the history (spec 5.11)."],
  ceeScale: ["CEE-Scale", "Flattens or amplifies the reader's reactions until they carry exactly the information budget. It repairs wrong scale and repeated evidence, and fails if the budget is overestimated (spec 5.11)."],
  gainIdentity: ["Gain identity", "An exact accounting: the expected improvement in log score from the correction equals the incoherence, plus a prior-calibration term, minus a world-model-error term. It predicts the gain and names the term that blocks it (spec 5.4)."],
  realizedGain: ["Realized gain", "Per labeled case, the improvement in log score from the correction, measured on the real result alone. Its average estimates the predicted gain without any counterfactual (spec 5.4)."],
  voi: ["Coherent value of information", "Expected utility after a test under the corrected beliefs, minus utility now. Never negative, because the corrected beliefs average to the current one (spec 5.6)."],
  regret: ["Regret attribution", "How much of the gap to the best possible agent comes from choosing tests (selection), from reading results (reading), and from deciding when to stop (commitment) (spec 1.2, 9.2)."],
  regimeMap: ["Regime map", "Figure 1, panel A: each reader placed by the size of its reactions (tau_B) and by its remaining incoherence, which shows whether it drifts, mis-sizes evidence, or does not respond (spec 1.7, 10.2)."],
  logLoss: ["Log loss", "A score of how much probability the belief put on the true diagnosis; lower is better. Lower log loss is not higher accuracy (spec 18)."],
  t5: ["T5", "Mean class recall: the average of the four per-disease recalls, in percent (REPRODUCTION.md)."],
  v1: ["V1", "The existing reference agent: the trained policy with the hypothesis fallback. Test T5 84.45. It is the bar any successor has to beat (spec 1.3, 2.3)."],
  cohorts: ["Cohorts P, S, X, B", "P: the existing primary policy on its 240 logged validation admissions. S: each new seed policy (E8) on its own 240. X: all 2,160 training-plus-validation admissions, each replayed by the fold policy (E10) that never trained on it. B: each second-backbone policy (E9) on its own validation admissions (spec 10.3)."],
  arms: ["Arms A1 to A4", "A1: reading and commitment with the layer, by replay. A1s: the same with CEE-Scale. A2: adds coherent test selection. A3: A2 with raw scores, to isolate the layer. A4: A2 with the gate, run only if E4 shows at least 20% harmful applications (spec 11.1)."],
  gate: ["Gate", "A decision rule fixed in advance. The agent evaluates it from recorded evidence, writes an immutable record, and takes the pre-registered path. No gate waits for a person, and a gate that is not met is a path, not a failure (spec 14.1)."],
  blocked: ["BLOCKED item", "Something only Daniel can resolve. It names what waits, the exact one-step request, and the path taken meanwhile. Independent work continues (spec 20.3)."],
  allowance: ["Allowance", "A ceiling against runaway spending, about twice the estimate. If the next submission would exceed it, nothing is submitted for that experiment and a BLOCKED request is raised. Failed jobs and retries count against it (AUTHORIZATION.md section 4)."],
  estimate: ["Estimate", "Expected GPU-hours for the experiment. An estimate, not a cap on the science (spec 14.3)."],
  watcher: ["Watcher", "A recurring job on HiPerGator, every 15 minutes: checks the authorization, reads job states, resubmits within retry limits, advances the stage graph, evaluates gates, and rewrites the status files (spec 20.6)."],
  stage: ["Stage", "One resumable unit of an experiment, run as one Slurm job of at most 24 hours. A stage that hits the limit checkpoints and continues in a new attempt (WATCHER_DESIGN.md section 3)."],
  keeper: ["Connection keeper", "A launchd job on this Mac that checks the SSH master on a timer and sends a trivial command through it to hold the session open. It never signs in; if the session drops it tells Daniel once (spec 20.8)."],
  localLoop: ["Local loop", "A launchd job on this Mac, every 10 minutes: through the existing connection it reads the status and new events, copies approved aggregates, and starts an agent session when a stage completed or a gate was decided (spec 20.7)."],
  master: ["SSH master", "Daniel's one signed-in connection to HiPerGator. Every read goes through it; nothing here can open a new connection (WATCHER_DESIGN.md section 1)."],
} as const satisfies Record<string, readonly [string, string]>;

export type TermKey = keyof typeof GLOSSARY;
export const terms = (...keys: TermKey[]): [string, string][] => keys.map((k) => [GLOSSARY[k][0], GLOSSARY[k][1]]);

/** Short names for the chips, from AUTHORIZATION.md section 4 and the cards' titles. */
export const EXPERIMENT_NAME: Record<string, string> = {
  E0: "instrument fixtures",
  E1: "exact synthetic study",
  E2: "reader panel on twin text",
  E3: "structured readers on charts",
  E4: "clinical evidence replay",
  E5: "closed-loop arms",
  E6: "frozen one-shot test",
  E7: "public datasets",
  E8: "four seeds",
  E9: "second backbone",
  E10: "cross-fitted policies",
  E11: "independent cohort",
};

export const EXPERIMENT_TERMS: Record<string, TermKey[]> = {
  E0: ["cee", "klInc", "gainIdentity"],
  E1: ["twin", "regret", "projection", "ceeScale"],
  E2: ["twin", "regimeMap", "tauB", "klInc"],
  E3: ["reader", "tauB", "realizedGain"],
  E4: ["cohorts", "arms", "logLoss", "t5", "v1"],
  E5: ["arms", "t5", "v1", "cohorts"],
  E6: ["arms", "t5", "v1"],
  E7: ["gainIdentity", "realizedGain"],
  E8: ["cohorts"],
  E9: ["cohorts"],
  E10: ["cohorts"],
  E11: ["blocked"],
};

export const MILESTONE_NAME: Record<string, string> = {
  M0: "Evidence, verification, freeze, and launch checks (spec 7)",
  M1: "Integrity repairs the claim depends on (spec 8)",
  M2: "The CEE instrument and the exact synthetic study (spec 9)",
  M3: "Readers that matter: the panel, clinical replay, and public data (spec 10)",
  M4: "Closed-loop evaluation, the one-shot test, and the training track (spec 11)",
  M5: "Robustness, independent evidence, the paper, and the submission decision (spec 12)",
};

/**
 * Columns of the plan board. Each is one window of the specification's dates table (17.2), in
 * the table's own order. `label` must equal the table's date cell, so that the live table can be
 * matched to it and a change in the specification shows up as a mismatch. `x0`/`x1` are
 * fractions of the board's width: widths follow content, not days, and each column carries its
 * own day ticks to show that.
 */
export interface PlanWindow {
  key: string;
  /** Date cells of spec 17.2 that this column covers. */
  rows: string[];
  label: string;
  from: number;
  to: number;
  x0: number;
  x1: number;
}

const d = easternMidnight;

export const WINDOWS: PlanWindow[] = [
  { key: "w0", rows: ["Fri Nov 13, 2026"], label: "Nov 13", from: d(2026, 11, 13), to: d(2026, 11, 14), x0: 0, x1: 0.03 },
  { key: "w1", rows: ["Nov 14–19"], label: "Nov 14–19", from: d(2026, 11, 14), to: d(2026, 11, 20), x0: 0.03, x1: 0.215 },
  { key: "w2", rows: ["Nov 20–26"], label: "Nov 20–26", from: d(2026, 11, 20), to: d(2026, 11, 27), x0: 0.215, x1: 0.475 },
  { key: "w3", rows: ["Nov 27–Dec 3"], label: "Nov 27–Dec 3", from: d(2026, 11, 27), to: d(2026, 12, 4), x0: 0.475, x1: 0.64 },
  { key: "w4", rows: ["Dec 4–10"], label: "Dec 4–10", from: d(2026, 12, 4), to: d(2026, 12, 11), x0: 0.64, x1: 0.745 },
  { key: "w5", rows: ["Dec 10–12"], label: "Dec 10–12", from: d(2026, 12, 11), to: d(2026, 12, 13), x0: 0.745, x1: 0.815 },
  { key: "w6", rows: ["Dec 13–Jan 2"], label: "Dec 13–Jan 2", from: d(2026, 12, 13), to: d(2027, 1, 3), x0: 0.815, x1: 0.908 },
  {
    key: "w7",
    rows: ["Sun Jan 3, 2027", "Fri Jan 8", "Mon Jan 11", "Thu Jan 14", "Jan 15–20", "Fri Jan 22"],
    label: "Jan 3–22",
    from: d(2027, 1, 3),
    to: d(2027, 1, 23),
    x0: 0.908,
    x1: 1,
  },
];

/** Single-day rows of spec 17.2, drawn as flags on the calendar rail. */
export const FLAGS: { row: string; at: number; name: string }[] = [
  { row: "Fri Dec 4", at: d(2026, 12, 4), name: "Course checkpoint" },
  { row: "Sun Jan 3, 2027", at: d(2027, 1, 3), name: "First complete draft" },
  { row: "Fri Jan 8", at: d(2027, 1, 8), name: "Full draft" },
  { row: "Mon Jan 11", at: d(2027, 1, 11), name: "Go or no-go" },
  { row: "Thu Jan 14", at: d(2027, 1, 14), name: "Abstract registration" },
  { row: "Fri Jan 22", at: d(2027, 1, 22), name: "Submission" },
];

/** Position on the board for a time: linear inside its window. */
export function xOfTime(ms: number): number {
  if (ms <= WINDOWS[0].from) return 0;
  for (const w of WINDOWS) {
    if (ms < w.to) return w.x0 + ((ms - w.from) / (w.to - w.from)) * (w.x1 - w.x0);
  }
  return 1;
}

export type NodeKind = "experiment" | "gate" | "milestone";

export interface BoardNode {
  id: string;
  kind: NodeKind;
  row: number;
  /** Chips: left and right edge. Gates: `x0` is the centre. Fractions of board width. */
  x0: number;
  x1: number;
  /** Date cell of spec 17.2 that names this node, for the hover. */
  planRow: string;
  /** What the plan says about its timing, in the monitor's words. */
  planNote: string;
}

export const ROWS = 9;

export const NODES: BoardNode[] = [
  { id: "M0", kind: "milestone", row: 0, x0: 0.04, x1: 0.072, planRow: "Nov 14–19", planNote: "M0 and its launch checks are planned for Nov 14–19." },
  { id: "G0", kind: "gate", row: 0, x0: 0.092, x1: 0.092, planRow: "Nov 14–19", planNote: "The plan gives no date for G0. It is drawn in the window that holds M0." },
  { id: "E0", kind: "experiment", row: 0, x0: 0.112, x1: 0.158, planRow: "Nov 14–19", planNote: "Planned for Nov 14–19." },
  { id: "M1", kind: "milestone", row: 4, x0: 0.112, x1: 0.144, planRow: "Nov 14–19", planNote: "Twin repairs and the preflight of spec 8.3 are planned for Nov 14–19." },
  { id: "G1", kind: "gate", row: 4, x0: 0.17, x1: 0.17, planRow: "Nov 14–19", planNote: "The plan gives no date for G1. It is drawn in the window that holds the repairs." },
  { id: "E1", kind: "experiment", row: 1, x0: 0.232, x1: 0.348, planRow: "Nov 20–26", planNote: "Parts A to D are planned for Nov 20–26." },
  { id: "E3", kind: "experiment", row: 4, x0: 0.232, x1: 0.278, planRow: "Nov 20–26", planNote: "Planned for Nov 20–26." },
  { id: "E8", kind: "experiment", row: 6, x0: 0.232, x1: 0.456, planRow: "Nov 14–19", planNote: "Submitted by Thu Nov 19; training finishes in Nov 20–26." },
  { id: "E9", kind: "experiment", row: 7, x0: 0.232, x1: 0.456, planRow: "Nov 14–19", planNote: "Submitted by Thu Nov 19; training finishes in Nov 20–26." },
  { id: "E10", kind: "experiment", row: 8, x0: 0.232, x1: 0.456, planRow: "Nov 14–19", planNote: "Submitted by Thu Nov 19; training finishes in Nov 20–26." },
  { id: "G2a", kind: "gate", row: 1, x0: 0.4, x1: 0.4, planRow: "Nov 20–26", planNote: "Due by Wed Nov 25." },
  { id: "G2c", kind: "gate", row: 2, x0: 0.4, x1: 0.4, planRow: "Nov 20–26", planNote: "Due by Wed Nov 25." },
  { id: "G2b", kind: "gate", row: 3, x0: 0.4, x1: 0.4, planRow: "Nov 20–26", planNote: "Due by Wed Nov 25." },
  { id: "E2", kind: "experiment", row: 1, x0: 0.442, x1: 0.626, planRow: "Nov 20–26", planNote: "Running in Nov 20–26; finishes in Nov 27–Dec 3." },
  { id: "E7", kind: "experiment", row: 2, x0: 0.442, x1: 0.626, planRow: "Nov 20–26", planNote: "Running in Nov 20–26; finishes in Nov 27–Dec 3." },
  { id: "E4", kind: "experiment", row: 5, x0: 0.502, x1: 0.566, planRow: "Nov 27–Dec 3", planNote: "Planned for Nov 27–Dec 3, on cohorts P, S, X, B." },
  { id: "G3", kind: "gate", row: 5, x0: 0.612, x1: 0.612, planRow: "Nov 27–Dec 3", planNote: "Due by Thu Dec 3." },
  { id: "E5", kind: "experiment", row: 5, x0: 0.658, x1: 0.712, planRow: "Dec 4–10", planNote: "Planned for Dec 4–10." },
  { id: "G4", kind: "gate", row: 5, x0: 0.732, x1: 0.732, planRow: "Dec 4–10", planNote: "Due by Thu Dec 10." },
  { id: "E6", kind: "experiment", row: 5, x0: 0.756, x1: 0.804, planRow: "Dec 10–12", planNote: "Planned for Dec 10–12, once." },
  { id: "M5", kind: "milestone", row: 5, x0: 0.828, x1: 0.86, planRow: "Dec 13–Jan 2", planNote: "Robustness, analysis, figures and drafting are planned for Dec 13–Jan 2." },
  { id: "E11", kind: "experiment", row: 7, x0: 0.828, x1: 0.878, planRow: "Dec 13–Jan 2", planNote: "No date in the plan. It belongs to M5 and runs only if a second cohort is listed." },
  { id: "G5", kind: "gate", row: 5, x0: 0.946, x1: 0.946, planRow: "Mon Jan 11", planNote: "The plan gives no date for G5. Its recommendation is the go or no-go of Mon Jan 11, so it is drawn there." },
];

export interface BoardEdge {
  from: string;
  to: string;
  /** Waypoints between the two nodes as [x fraction, row] pairs; rows may be fractional. */
  via?: [number, number][];
  /** A dashed edge changes what the target may do; it does not release it. */
  modifies?: boolean;
  /** Vertical offset in pixels where the edge meets its target chip. */
  enterDy?: number;
  /** Where the edge leaves or enters: default is right side out, left side in. */
  out?: "right" | "bottom" | "top";
  into?: "left" | "top" | "bottom";
  /** Drawn last with a gap under it, so that it reads as passing over the lines it crosses. */
  over?: boolean;
  why: string;
}

const BUS_G1 = 0.202;
const BUS_E1 = 0.37;
const BUS_G2A = 0.422;
const BUS_TRAIN = 0.474;

export const EDGES: BoardEdge[] = [
  { from: "M0", to: "G0", why: "G0 is M0's acceptance (spec 7.4, 14.1)." },
  { from: "G0", to: "E0", why: "If G0 is met, M1 and M2 proceed; E0 is M2's instrument (spec 14.1, 9.1)." },
  { from: "G0", to: "M1", out: "bottom", via: [[0.092, 4]], why: "If G0 is met, M1 and M2 proceed (spec 14.1)." },
  { from: "M1", to: "G1", why: "G1: the repairs needed by E0 to E3 and by training are verified (spec 14.1)." },
  { from: "G1", to: "E1", via: [[BUS_G1, 4], [BUS_G1, 1]], why: "If G1 is met, E1, E3 and the training track proceed (spec 14.1)." },
  { from: "G1", to: "E3", why: "If G1 is met, E1, E3 and the training track proceed (spec 14.1)." },
  { from: "G1", to: "E8", via: [[BUS_G1, 4], [BUS_G1, 6]], why: "If G1 is met, E1, E3 and the training track proceed (spec 14.1)." },
  { from: "G1", to: "E9", via: [[BUS_G1, 4], [BUS_G1, 7]], why: "If G1 is met, E1, E3 and the training track proceed (spec 14.1)." },
  { from: "G1", to: "E10", via: [[BUS_G1, 4], [BUS_G1, 8]], why: "If G1 is met, E1, E3 and the training track proceed (spec 14.1)." },
  { from: "E0", to: "G2a", into: "top", via: [[0.4, 0]], why: "G2a needs E0 passed (spec 14.1)." },
  { from: "E1", to: "G2a", why: "G2a needs E1 completed and interpretable (spec 14.1)." },
  { from: "E1", to: "G2c", via: [[BUS_E1, 1], [BUS_E1, 2]], why: "G2c is decided by E1 part D: is scale repair usable outside the twin (spec 14.1)." },
  { from: "E1", to: "G2b", via: [[BUS_E1, 1], [BUS_E1, 3]], why: "G2b is decided by E1 part B: is projection repair present (spec 14.1)." },
  { from: "G2a", to: "E2", why: "If G2a is met, the measurement tier runs: E2, E4, E7 (spec 14.1)." },
  { from: "G2a", to: "E7", via: [[BUS_G2A, 1], [BUS_G2A, 2]], why: "If G2a is met, the measurement tier runs: E2, E4, E7 (spec 14.1)." },
  { from: "G2a", to: "E4", via: [[BUS_G2A, 1], [BUS_G2A, 5]], enterDy: -5, why: "If G2a is met, the measurement tier runs: E2, E4, E7 (spec 14.1)." },
  { from: "G2c", to: "E7", modifies: true, via: [[BUS_G2A, 2]], why: "If G2c is met, CEE-Scale may be applied on charts (E4) and public data (E7). If not, it is reported from the twin only (spec 14.1)." },
  { from: "G2b", to: "E5", into: "top", over: true, via: [[0.686, 3]], why: "If G2b is met, the method arms stay eligible. If not, E5 is skipped and the paper is the mechanism paper (spec 14.1)." },
  { from: "E8", to: "E4", via: [[BUS_TRAIN, 6], [BUS_TRAIN, 5]], why: "E4's cohort S replays each E8 seed policy (spec 10.3)." },
  { from: "E9", to: "E4", via: [[BUS_TRAIN, 7], [BUS_TRAIN, 5]], why: "E4's cohort B replays each E9 second-backbone policy (spec 10.3)." },
  { from: "E10", to: "E4", via: [[BUS_TRAIN, 8], [BUS_TRAIN, 5]], why: "E4's cohort X is replayed by the E10 fold policies (spec 10.3)." },
  { from: "E4", to: "G3", why: "G3: the E4 rule is met, and spec 8.3 and 8.5 are verified (spec 14.1)." },
  { from: "G3", to: "E5", why: "If G3 is met, E5 runs and A1 is eligible for the test. If not, E5 and E6 are skipped and the test split stays untouched (spec 14.1)." },
  { from: "E5", to: "G4", why: "G4: G3 met; E5 completed or skipped by G2b; the test protocol frozen and committed (spec 14.1)." },
  { from: "G4", to: "E6", why: "If G4 is met, E6 runs once (spec 14.1)." },
  { from: "E6", to: "M5", why: "M5 follows the GPU critical path: robustness, analysis, the paper (spec 12, 17.2)." },
  { from: "E11", to: "M5", out: "top", into: "bottom", why: "E11 is M5's independent evidence, used only if a second credentialed cohort is listed (spec 12)." },
  { from: "M5", to: "G5", why: "G5 is M5's assessment: the recommendation, the manuscript, and the remaining requirements (spec 12, 14.1)." },
];

/** Points where lines join (a dot), as opposed to merely crossing (no dot). */
export const JUNCTIONS: [number, number][] = [
  [BUS_G1, 4],
  [BUS_G1, 6],
  [BUS_G1, 7],
  [BUS_E1, 1],
  [BUS_E1, 2],
  [BUS_G2A, 1],
  [BUS_G2A, 2],
  [BUS_TRAIN, 6],
  [BUS_TRAIN, 7],
];

/** Which experiments each Figure 1 panel waits for (spec 1.7, 9.2, 10.2 to 10.4). */
export const FIGURE_PANELS = [
  {
    file: "regime_map.json",
    letter: "A",
    name: "The regime map",
    needs: ["E1", "E2", "E4", "E7"],
    what: "One marker per reader and setting: simulated families from E1, language-model readers from E2, the trained policies from E4, public datasets from E7. The shaded region, from E1's exact twin results, is where projection alone is harmful.",
  },
  {
    file: "gain_accounting.json",
    letter: "B",
    name: "The accounting",
    needs: ["E1", "E2", "E4", "E7"],
    what: "One point per reader-by-setting cell: the gain the identity predicts against the gain that was realized, with intervals. Points on the diagonal mean the accounting holds.",
  },
  {
    file: "regret_attribution.json",
    letter: "C",
    name: "The attribution",
    needs: ["E1"],
    what: "For the twin agents, raw and repaired: the share of exact regret from selection, reading, and commitment.",
  },
] as const;

/** Stage exit codes (WATCHER_DESIGN.md section 3). */
export const EXIT_CODE: Record<number, string> = {
  0: "done",
  10: "checkpointed at the time limit; continues in a new attempt",
  20: "identity refusal: a bound artifact changed; never retried",
  30: "authorization or data-use refusal; never retried",
  40: "out of memory, detected by the stage",
};

/** What the watcher does with each end state (WATCHER_DESIGN.md section 4). */
export const END_STATE: Record<string, { letter: string; meaning: string }> = {
  COMPLETED: { letter: "", meaning: "Finished. Exit 0 marks the stage done; exit 10 means it checkpointed and continues." },
  TIMEOUT: { letter: "T", meaning: "Hit the time limit. Resubmitted from its checkpoint; counted against three retries only if it made no progress." },
  PREEMPTED: { letter: "P", meaning: "Preempted by the scheduler. Resubmitted from its checkpoint, up to three times." },
  NODE_FAIL: { letter: "N", meaning: "The node failed. Resubmitted from its checkpoint, up to three times." },
  OUT_OF_MEMORY: { letter: "M", meaning: "Ran out of memory. Resubmitted once with 1.5 times the memory." },
  FAILED: { letter: "F", meaning: "Failed for another reason. Resubmitted once; a second failure marks the stage FAILED and only its dependents wait." },
  CANCELLED: { letter: "C", meaning: "Cancelled." },
};

export const STATUS_WORD: Record<string, string> = {
  PLANNED: "planned",
  QUEUED: "queued",
  RUNNING: "running",
  COMPLETED: "completed",
  FAILED: "failed",
  REFUSED: "refused",
  BLOCKED: "blocked",
  INTERRUPTED: "interrupted",
  "SKIPPED BY GATE": "skipped by gate",
  SKIPPED: "skipped",
};

export const STATUS_MEANING: Record<string, string> = {
  PLANNED: "Not submitted yet.",
  QUEUED: "Submitted and waiting for a node.",
  RUNNING: "At least one job is running.",
  COMPLETED: "Finished. A negative or inconclusive result still completes it when it answers its question honestly (spec 1.5).",
  FAILED: "Retries are exhausted. Only its dependents wait.",
  REFUSED: "A manifest or authorization check refused to run it. Never retried; it becomes a BLOCKED item.",
  BLOCKED: "Waits on something only Daniel can resolve.",
  INTERRUPTED: "Stopped part-way; it resumes from its checkpoint.",
  "SKIPPED BY GATE": "A gate's pre-registered path skipped it. Reported with the record that skipped it (spec 14.2).",
  SKIPPED: "Not run, by a recorded decision; nothing waits on it.",
};
