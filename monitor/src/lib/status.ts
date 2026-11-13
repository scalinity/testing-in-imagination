// How a status looks. Hue is reserved for state; every state also has a glyph and a word.

import { STATUS_MEANING, STATUS_WORD } from "./study";

export type Look = "planned" | "queued" | "running" | "done" | "fail" | "wait" | "retry" | "skip" | "unknown";
export type GlyphName = "ring" | "queued" | "dot" | "check" | "cross" | "pause" | "retry" | "skip" | "clock" | "none";

const LOOK: Record<string, Look> = {
  PLANNED: "planned",
  QUEUED: "queued",
  RUNNING: "running",
  COMPLETED: "done",
  COMPLETE: "done",
  FAILED: "fail",
  REFUSED: "fail",
  BLOCKED: "wait",
  INTERRUPTED: "retry",
  "SKIPPED BY GATE": "skip",
  "NOT STARTED": "planned",
  "IN PROGRESS": "running",
};

const GLYPH: Record<Look, GlyphName> = {
  planned: "ring",
  queued: "queued",
  running: "dot",
  done: "check",
  fail: "cross",
  wait: "pause",
  retry: "retry",
  skip: "skip",
  unknown: "none",
};

export const look = (status: string): Look => LOOK[status.toUpperCase()] ?? "unknown";
export const glyphFor = (status: string): GlyphName => GLYPH[look(status)];
export const word = (status: string): string => STATUS_WORD[status.toUpperCase()] ?? (status ? status.toLowerCase() : "no source yet");
export const meaning = (status: string): string => STATUS_MEANING[status.toUpperCase()] ?? "";
