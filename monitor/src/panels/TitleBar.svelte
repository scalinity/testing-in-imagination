<script lang="ts">
  import Glyph from "../lib/Glyph.svelte";
  import { fileSource } from "../lib/sources";
  import { glyphFor, look } from "../lib/status";
  import { store } from "../lib/store.svelte";
  import { MILESTONE_NAME } from "../lib/study";
  import { clockS, day, dayClock, span } from "../lib/time";
  import { tip, type Tip } from "../lib/tip.svelte";

  const docs = $derived(store.docs);
  const now = $derived(store.now);
  const milestones = $derived(["M0", "M1", "M2", "M3", "M4", "M5"].map((id) => ({ id, state: docs?.milestones.find((m) => m.id === id)?.state ?? "" })));

  // The app's own heartbeat: the backend answers once a second.
  const silent = $derived(store.sincePoll > 5000);

  type Health = "ok" | "stale" | "bad" | "none";
  interface Lamp {
    name: string;
    health: Health;
    tip: Tip;
  }

  const age = (at: number | null | undefined) => (at == null ? Infinity : now - at);
  const read = (at: number | null | undefined) => (at == null ? "never" : `${dayClock(at)} (${span(now - at)} ago)`);

  const lamps = $derived.by((): Lamp[] => {
    const c = store.checked;
    const w = store.workers;
    const git = store.git;
    const link = store.link;
    const remote = store.remote;
    const meta = store.remoteMeta;
    const interval = (meta?.interval_s ?? 120) * 1000;
    const pick = (ok: boolean, stale: boolean, missing = false): Health => (missing ? "none" : !ok ? "bad" : stale ? "stale" : "ok");
    const jobs = link?.launchd ?? [];
    const remoteHealth: Health = !remote || remote.phase === "never_read" || remote.phase === "" ? "none" : remote.phase === "error" ? "bad" : remote.skipped || age(remote.last_ok_ms) > interval * 2.5 ? "stale" : remote.phase === "not_started" || !remote.status ? "none" : "ok";
    return [
      {
        name: "Transcripts",
        health: pick(Boolean(w?.dir_exists), age(c?.transcripts) > 10_000),
        tip: {
          title: "Claude Code transcripts",
          body: w?.dir_exists ? "Each AI worker writes a transcript as it works. The monitor reads only what was appended since its last look, and keeps tool names, descriptions and file paths, never tool outputs." : "The transcript folder does not exist, so no worker activity can be shown.",
          rows: [["Folder", w?.dir ?? ""], ["Sessions shown", String(w?.workers.length ?? 0)], ["Left out", `${w?.older_sessions_hidden ?? 0} sessions from before the study started`], ["Last read", read(c?.transcripts)], ["Cadence", "every second"]],
        },
      },
      {
        name: "Git",
        health: pick(Boolean(git?.ok), age(c?.git) > 10_000, !git),
        tip: {
          title: "Git, task branch",
          body: git?.ok ? "Read-only git commands on the repository: the branch's commits since the study started, and uncommitted work." : `Git could not be read: ${git?.error ?? "not read yet"}`,
          rows: [["Branch", git?.branch ?? ""], ["Checked out", git?.head_branch ?? ""], ["Commits", String(git?.commits.length ?? 0)], ["Last read", read(c?.git)], ["Cadence", "every 2 seconds"]],
        },
      },
      {
        name: "Documents",
        health: pick(Boolean(docs?.spec.exists) && !(docs?.problems.length ?? 0), age(c?.docs) > 10_000, !docs),
        tip: {
          title: "The study's documents",
          body: docs?.problems.length ? docs.problems.join(" ") : "The specification, authorization, status, blocked items, ledger and experiment cards, read from the repository. Hover text is built from them.",
          rows: [["Specification", docs?.spec_version ? `version ${docs.spec_version}` : "not read"], ["Folder", docs?.spec.path.replace(/\/[^/]+$/, "") ?? ""], ["Last checked", read(c?.docs)], ["Cadence", "every 2 seconds"]],
        },
      },
      {
        name: "Keeper log",
        health: pick(Boolean(link?.keeper_log_exists), age(c?.keeper) > 30_000, !link),
        tip: {
          title: "Connection keeper log",
          body: link?.keeper_log_exists ? "One line per keeper run: whether the SSH master answered." : "The keeper's log file does not exist, so there is no connection history.",
          rows: [["File", link?.keeper_log ?? ""], ["Lines read", String(link?.keeper.length ?? 0)], ["File last written", read(link?.keeper_log_mtime_ms)], ["Last read", read(c?.keeper)], ["Cadence", "every 5 seconds"]],
        },
      },
      {
        name: "Launch agents",
        health: pick(jobs.length > 0 && jobs.every((j) => !j.installed || j.last_exit == null || j.last_exit === 0), age(c?.launchd) > 60_000, !jobs.length),
        tip: {
          title: "launchd jobs on this Mac",
          body: "The keeper, caffeinate and the local loop, as launchctl reports them.",
          rows: [...jobs.map((j): [string, string] => [j.label, j.installed ? j.state : "not installed"]), ["Last read", read(c?.launchd)], ["Cadence", "every 20 seconds"]],
        },
      },
      {
        name: "SSH master",
        health: pick(Boolean(link?.master_up), age(c?.master) > 60_000, !link || c?.master == null),
        tip: {
          title: "SSH master connection",
          body: link?.master_up ? "The signed-in connection to HiPerGator is alive. Every cluster read goes through it." : "The connection is not running, so the monitor makes no ssh call and shows the last known cluster state. To reconnect, sign in from Terminal.",
          rows: [["ssh -O check says", link?.master_detail ?? ""], ["Last checked", read(c?.master)], ["Cadence", "every 20 seconds, and before every cluster read"]],
        },
      },
      {
        name: "HiPerGator files",
        health: remoteHealth,
        tip: {
          title: "Watcher files on HiPerGator",
          body:
            remote?.skipped ??
            (remote?.phase === "error"
              ? `The last read failed: ${remote.detail}`
              : remote?.phase === "not_started" || (remote?.phase === "ok" && !remote.status)
                ? "The watcher has not written its status yet, so there is nothing to show from the cluster."
                : "Read with cat, tail and ls on an allowlist of files, through the existing connection only. No scheduler command is ever sent."),
          rows: [["Last successful read", read(remote?.last_ok_ms)], ["Cadence", `every ${meta?.interval_s ?? 120} seconds, never closer than ${meta?.min_interval_s ?? 60}`], ["Commands in the last read", String(remote?.calls.length ?? 0)]],
        },
      },
    ];
  });

  const SHORT: Record<string, string> = { Documents: "Docs", "Keeper log": "Keeper", "Launch agents": "Agents", "SSH master": "SSH", "HiPerGator files": "HiPerGator" };
  const HEALTH_WORD: Record<Health, string> = { ok: "current", stale: "stale", bad: "failing", none: "not available yet" };
</script>

<header class="bar" class:compact={store.compact} data-tauri-drag-region>
  <h1 data-tauri-drag-region>TII-CEE-1</h1>
  <span class="study muted" data-tauri-drag-region>Testing in Imagination</span>

  <ol class="milestones">
    {#each milestones as m (m.id)}
      <li
        class={m.state ? look(m.state) : "unknown"}
        use:tip={() => ({
          title: `${m.id}: ${MILESTONE_NAME[m.id]}`,
          sub: m.state ? `${m.state.charAt(0)}${m.state.slice(1).toLowerCase()}.` : "STATUS.md does not state this milestone.",
          rows: docs?.status_milestone_line ? [["STATUS.md says", docs.status_milestone_line], ["STATUS.md header", docs.status_updated]] : [],
          source: fileSource(docs?.status_file),
        })}
      >
        <span class="mark" class:breathe={look(m.state) === "running"}><Glyph name={m.state ? glyphFor(m.state) : "none"} size={9} /></span>{m.id}
      </li>
    {/each}
  </ol>

  <span class="grow" data-tauri-drag-region></span>

  <ul class="lamps" aria-label="Sources">
    {#each lamps as l (l.name)}
      <li class={l.health} use:tip={() => ({ ...l.tip, sub: `${l.name}: ${HEALTH_WORD[l.health]}.` })} tabindex="0">
        <Glyph name={l.health === "ok" ? "dot" : l.health === "stale" ? "clock" : l.health === "bad" ? "cross" : "ring"} size={10} />
        <span>{store.compact ? (SHORT[l.name] ?? l.name) : l.name}</span>
      </li>
    {/each}
  </ul>

  <span
    class="beat"
    class:silent
    use:tip={() => ({
      title: silent ? "The monitor stopped updating" : "The monitor is live",
      body: silent
        ? "The window asks its backend for fresh data once a second, and the backend has not answered. Everything on screen is the last known state."
        : "The window asks its backend for fresh data once a second. This dot blinks on each answer.",
      rows: [["Last answer", `${span(store.sincePoll)} ago`], ["Answers so far", String(store.polls)], ...(store.pollError ? ([["Error", store.pollError]] as [string, string][]) : [])],
    })}
  >
    {#key store.polls}<i class="pulse"></i>{/key}
    {silent ? `no update for ${span(store.sincePoll)}` : "live"}
  </span>

  <time class="clock num" data-tauri-drag-region use:tip={{ title: "Now, US Eastern", body: "Every time on screen is US Eastern. In demo mode this is the fixtures' clock, not the real time." }}>
    {day(now)}, {clockS(now)}
  </time>
</header>

<style>
  .bar {
    flex: none;
    display: flex;
    align-items: center;
    gap: 14px;
    height: 34px;
    padding: 0 14px;
    border-bottom: 1px solid var(--line);
    background: var(--raised);
  }
  h1 {
    font-size: var(--fs-4);
    font-weight: 700;
    letter-spacing: 0.01em;
  }
  .study {
    font-size: var(--fs-2);
  }
  .grow {
    flex: 1;
    align-self: stretch;
  }
  .milestones,
  .lamps {
    display: flex;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .milestones {
    gap: 3px;
  }
  .milestones li {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 18px;
    padding: 0 7px 0 5px;
    border: 1px solid var(--line-2);
    border-radius: 9px;
    font-size: var(--fs-1);
    font-weight: 600;
    color: var(--ink-3);
  }
  .milestones li.running {
    border-color: var(--live);
    background: color-mix(in srgb, var(--live) 20%, transparent);
    color: var(--ink);
  }
  .milestones li.running .mark {
    color: var(--live-text);
  }
  .milestones li.done {
    border-color: var(--done);
    background: var(--done);
    color: var(--ink);
  }
  .milestones li.wait {
    border-color: var(--wait);
    color: var(--wait-text);
  }
  .milestones li.unknown {
    border-style: dashed;
  }
  .mark {
    display: inline-flex;
  }
  .lamps {
    gap: 11px;
    font-size: var(--fs-1);
    color: var(--ink-3);
  }
  .lamps li {
    display: flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
  }
  .lamps li.ok {
    color: var(--ink-2);
  }
  .lamps li.stale {
    color: var(--wait-text);
  }
  .lamps li.bad {
    color: var(--fail-text);
  }
  .beat {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: var(--fs-1);
    font-weight: 600;
    color: var(--ink-2);
    white-space: nowrap;
  }
  .beat.silent {
    color: var(--wait-text);
  }
  .pulse {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--live);
    animation: blink 0.9s ease-out;
  }
  .silent .pulse {
    background: var(--wait);
    animation: none;
  }
  @keyframes blink {
    0% {
      box-shadow: 0 0 0 0 color-mix(in srgb, var(--live) 70%, transparent);
    }
    100% {
      box-shadow: 0 0 0 6px transparent;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .pulse {
      animation: none;
    }
  }
  .clock {
    font-size: var(--fs-2);
    font-weight: 600;
    white-space: nowrap;
  }
  /* Small display: shorter source names, tighter spacing. */
  .compact {
    height: 30px;
    gap: 11px;
    padding: 0 10px;
  }
  .compact .study {
    display: none;
  }
  .compact .lamps {
    gap: 8px;
  }
  .compact .milestones li {
    padding: 0 5px 0 4px;
  }
</style>
