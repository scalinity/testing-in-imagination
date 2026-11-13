// Sentences that name a value's source and its age, for the foot of every tooltip.

import { store } from "./store.svelte";
import { dayClock, parseTime, span } from "./time";
import type { FileInfo } from "./types";

const when = (ms: number | null | undefined) => (ms == null ? "at an unknown time" : `${dayClock(ms)} (${span(store.now - ms)} ago)`);

/** For anything drawn from the cluster's files. */
export function clusterSource(file = "state/status.json"): string {
  const r = store.remote;
  if (!r || r.phase === "never_read" || r.phase === "") return `${file} on HiPerGator: not read yet.`;
  const written = file === "state/status.json" ? parseTime(r.status?.generated_at) : null;
  const parts = [`${file} on HiPerGator`];
  if (written != null) parts.push(`written by the watcher ${when(written)}`);
  parts.push(`read through the SSH master ${when(r.last_ok_ms)}`);
  return `${parts.join(", ")}.`;
}

/** For anything drawn from a file in the repository. */
export function fileSource(file: FileInfo | undefined | null, section = ""): string {
  if (!file) return "The study's documents have not been read yet.";
  const name = section ? `${file.path}, ${section}` : file.path;
  if (!file.exists) return `${name}: this file does not exist.`;
  return `${name}, last changed ${when(file.mtime_ms)}, checked ${span(store.now - (store.checked?.docs ?? store.now))} ago.`;
}

export const checkedAgo = (ms: number | null | undefined) => (ms == null ? "not checked yet" : `checked ${span(store.now - ms)} ago`);
