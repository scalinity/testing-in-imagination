// One poll loop, at module level: asks Rust for the snapshot every second and keeps a clock
// that ticks between polls, so every age on screen moves on its own.

import type { Checked, Docs, Git, Link, Remote, RemoteMeta, StateReply, Workers } from "./types";

const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

class Store {
  ready = $state(false);
  demo = $state(false);
  repo = $state("");
  studyStart = $state(0);
  /** Worker timeline width in seconds. */
  window = $state(3600);
  /** Window size, kept current by a resize listener. */
  vw = $state(typeof window === "undefined" ? 1440 : window.innerWidth);
  vh = $state(typeof window === "undefined" ? 900 : window.innerHeight);
  /** Whether Figure 1 is shown enlarged. */
  figureOpen = $state(false);

  /** Tighter geometry for a small display, so that everything still fits one screen. */
  get compact(): boolean {
    return this.vw < 1400 || this.vh < 860;
  }

  workers = $state.raw<Workers | null>(null);
  link = $state.raw<Link | null>(null);
  git = $state.raw<Git | null>(null);
  docs = $state.raw<Docs | null>(null);
  remote = $state.raw<Remote | null>(null);
  remoteMeta = $state.raw<RemoteMeta | null>(null);
  checked = $state.raw<Checked | null>(null);

  /** Backend clock at the last successful poll, and the local time it arrived. */
  private serverNow = $state(0);
  private polledAt = $state(0);
  private tick = $state(0);
  pollError = $state("");
  polls = $state(0);

  /** The clock every age is measured against. Keeps ticking if polls stop. */
  get now(): number {
    return this.serverNow + Math.max(0, this.tick - this.polledAt);
  }
  /** Milliseconds since the window opened. */
  get uptime(): number {
    return this.tick;
  }
  /** Milliseconds since the backend last answered. */
  get sincePoll(): number {
    return Math.max(0, this.tick - this.polledAt);
  }

  private revs = { link: 0, git: 0, docs: 0, remote: 0 };
  private busy = false;
  private snapshotStamp = 0;

  apply(reply: StateReply) {
    this.demo = reply.demo;
    this.repo = reply.repo;
    this.studyStart = reply.study_start_ms;
    this.workers = reply.workers;
    this.checked = reply.checked;
    this.remoteMeta = reply.remote_meta;
    if (reply.link) this.link = reply.link;
    if (reply.git) this.git = reply.git;
    if (reply.docs) this.docs = reply.docs;
    if (reply.remote) this.remote = reply.remote;
    this.revs = { ...reply.revs };
    this.serverNow = reply.now_ms;
    this.polledAt = performance.now();
    this.tick = this.polledAt;
    this.pollError = "";
    this.polls += 1;
    this.ready = true;
  }

  async poll() {
    if (this.busy) return;
    this.busy = true;
    try {
      if (inTauri) {
        const { invoke } = await import("@tauri-apps/api/core");
        this.apply(await invoke<StateReply>("get_state", { known: this.revs, window: this.window }));
      } else {
        // Development in a plain browser: snapshots written by `pnpm snapshot`. The file is
        // re-read every second and applied when it has changed, so in-place updates can be
        // exercised without the app.
        const mode = new URLSearchParams(location.search).get("mode") === "demo" ? "demo" : "real";
        const res = await fetch(`/snapshot?mode=${mode}`);
        if (!res.ok) throw new Error("no snapshot; run `pnpm snapshot` first");
        const reply = (await res.json()) as StateReply;
        if (reply.now_ms !== this.snapshotStamp) {
          this.snapshotStamp = reply.now_ms;
          this.apply(reply);
        }
      }
    } catch (e) {
      this.pollError = e instanceof Error ? e.message : String(e);
    } finally {
      this.busy = false;
    }
  }

  start() {
    addEventListener("resize", () => {
      this.vw = innerWidth;
      this.vh = innerHeight;
    });
    void this.poll();
    setInterval(() => {
      this.tick = performance.now();
      void this.poll();
    }, 1000);
  }

  async setRemoteInterval(seconds: number) {
    if (!inTauri) return;
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke<number>("set_remote_interval", { seconds });
    void this.poll();
  }

  async readClusterNow() {
    if (!inTauri) return;
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("read_cluster_now");
  }
}

export const store = new Store();
store.start();
