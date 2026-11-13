<script lang="ts">
  // The wire between the two machines. On top, the SSH master as the monitor finds it now; below
  // it the keeper's log as a cable, newest at the top. A hatched stretch is a time with no logged
  // check; an amber stretch is a time the connection was down.
  import Glyph from "../lib/Glyph.svelte";
  import { checkedAgo } from "../lib/sources";
  import { store } from "../lib/store.svelte";
  import { terms } from "../lib/study";
  import { clockS, dayClock, easternMidnight, hour, span, zone } from "../lib/time";
  import { tip, type Tip } from "../lib/tip.svelte";
  import type { KeeperEntry } from "../lib/types";

  let height = $state(420);
  const TOP = 72;
  const BOTTOM = 22;
  const W = $derived(store.compact ? 54 : 62);
  const CX = $derived(store.compact ? 36 : 40);

  const link = $derived(store.link);
  const now = $derived(store.now);
  const entries = $derived(link?.keeper ?? []);
  const intervalMs = $derived((link?.launchd.find((j) => j.label.endsWith("hpg-keeper"))?.interval_s ?? 300) * 1000);
  // Show at least 2 hours and at most 24, growing with the log.
  const windowMs = $derived(Math.min(24 * 3600e3, Math.max(2 * 3600e3, entries.length ? now - entries[0].t_ms + intervalMs : 0)));
  const y = (t: number) => TOP + (Math.min(windowMs, Math.max(0, now - t)) / windowMs) * (height - TOP - BOTTOM);

  const MEANING: Record<KeeperEntry["kind"], string> = {
    ok: "The keeper found the connection alive and sent a trivial command through it.",
    reconnected: "The connection was back after a drop.",
    relayed: "The keeper passed watcher events on as macOS notifications; the connection was alive.",
    no_answer: "The master was present but did not answer.",
    closed: "The keeper closed a master that had stopped answering.",
    signin: "The connection was down, and the keeper told Daniel once to sign in. Jobs on HiPerGator keep running meanwhile.",
    down: "Still down; Daniel had already been told.",
    other: "A line the monitor does not recognise.",
  };
  const LOOK: Record<KeeperEntry["kind"], string> = { ok: "ok", relayed: "ok", reconnected: "back", no_answer: "retry", closed: "retry", signin: "wait", down: "wait", other: "other" };

  // Each check stands for half an interval either side of it; neighbours meet in the middle.
  const blocks = $derived.by(() => {
    const from = now - windowMs;
    const half = intervalMs * 0.6;
    return entries
      .map((e, i) => {
        const prev = entries[i - 1];
        const next = entries[i + 1];
        const start = Math.max(prev ? (prev.t_ms + e.t_ms) / 2 : e.t_ms - half, e.t_ms - half);
        const end = Math.min(next ? (next.t_ms + e.t_ms) / 2 : e.t_ms + half, e.t_ms + half, now);
        return { e, start, end };
      })
      .filter((b) => b.end > from);
  });

  // Stretches of the window that no check covers.
  const gaps = $derived.by(() => {
    const out: { start: number; end: number }[] = [];
    let cursor = now - windowMs;
    for (const b of blocks) {
      if (b.start - cursor > 1000) out.push({ start: cursor, end: b.start });
      cursor = Math.max(cursor, b.end);
    }
    if (now - cursor > 1000) out.push({ start: cursor, end: now });
    return out;
  });

  const hours = $derived.by(() => {
    const step = windowMs > 12 * 3600e3 ? 3 * 3600e3 : windowMs > 5 * 3600e3 ? 2 * 3600e3 : 3600e3;
    const anchor = easternMidnight(2026, 11, 9);
    const out: number[] = [];
    for (let t = anchor + Math.ceil((now - windowMs - anchor) / step) * step; t <= now; t += step) out.push(t);
    return out;
  });

  const logSource = $derived(`${link?.keeper_log ?? "keeper log"}, ${checkedAgo(store.checked?.keeper)}.`);
  const last = $derived(entries[entries.length - 1]);
  const keeperLate = $derived(last != null && now - last.t_ms > intervalMs * 2.5);

  const blockTip = (e: KeeperEntry): Tip => ({ title: `Keeper check at ${clockS(e.t_ms)}`, sub: `Logged: "${e.text}"`, body: MEANING[e.kind], terms: terms("keeper"), source: logSource });
</script>

<section class="seam" bind:clientHeight={height} aria-label="Connection between this Mac and HiPerGator">
  <svg width={W} {height}>
    <g
      class="master"
      class:up={link?.master_up}
      tabindex="0"
      role="group"
      aria-label="SSH master"
      use:tip={() => ({
        title: link?.master_up ? "The SSH connection is up" : "The SSH connection is down",
        sub: link?.master_detail ? `ssh -O check says: ${link.master_detail}` : undefined,
        body: link?.master_up
          ? "Every read of HiPerGator goes through this one signed-in connection."
          : "Nothing is sent to HiPerGator while it is down; the cluster panels show the last known state. Jobs there keep running. To reconnect, run sshhiper in Terminal and sign in.",
        terms: terms("master"),
        source: `Asked locally with ssh -O check, ${checkedAgo(store.checked?.master)}. It is asked every 20 seconds and before each cluster read.`,
      })}
    >
      <rect class="hit" x="0" y="0" width={W} height={TOP - 6} />
      <text x={W / 2} y="17" text-anchor="middle" class="name">SSH</text>
      <circle cx={W / 2} cy="34" r="7" class:breathe={link?.master_up} />
      <text x={W / 2} y="55" text-anchor="middle" class="state">{link == null ? "…" : link.master_up ? "up" : "down"}</text>
    </g>

    <text class="hour" x={W / 2} y={TOP - 1} text-anchor="middle">{zone(now)}</text>
    {#each hours as t}
      <line class="hourtick" x1={CX - 8} x2={CX + 8} y1={y(t)} y2={y(t)} />
      <text class="hour num" x={CX - 11} y={y(t) + 3.5} text-anchor="end">{hour(t).replace(" ", "")}</text>
    {/each}

    {#each gaps as g}
      <rect
        class="gap"
        x={CX - 3}
        y={y(g.end)}
        width="6"
        height={Math.max(0, y(g.start) - y(g.end))}
        use:tip={{
          title: "No keeper check logged",
          sub: `${dayClock(g.start)} to ${dayClock(g.end)} (${span(g.end - g.start)})`,
          body: "The log has no line for this stretch: the keeper was not installed yet, or the Mac was asleep, or the keeper did not run. The connection's state then is not known.",
          source: logSource,
        }}
      />
    {/each}
    {#each blocks as b (b.e.t_ms)}
      <rect class="block {LOOK[b.e.kind]}" x={CX - 3} y={y(b.end)} width="6" height={Math.max(1.5, y(b.start) - y(b.end))} use:tip={() => blockTip(b.e)} />
      {#if b.e.kind === "reconnected" || b.e.kind === "signin"}<circle class="event {LOOK[b.e.kind]}" cx={CX} cy={y(b.e.t_ms)} r="5" />{/if}
    {/each}

    <text
      class="foot"
      class:late={keeperLate}
      x={W / 2}
      y={height - 7}
      text-anchor="middle"
      use:tip={{
        title: keeperLate ? "The keeper has gone quiet" : "Keeper log",
        sub: last ? `Last line ${dayClock(last.t_ms)}, ${span(now - last.t_ms)} ago: "${last.text}"` : link?.keeper_log_exists ? "The log is empty." : "The log file does not exist.",
        body: `The cable shows the last ${span(windowMs)}, newest at the top. The keeper runs every ${span(intervalMs)}.`,
        terms: terms("keeper"),
        source: logSource,
      }}>{keeperLate ? "keeper quiet" : "keeper log"}</text
    >
  </svg>
</section>

<style>
  .seam {
    border-left: 1px solid var(--line);
    border-right: 1px solid var(--line);
    background: var(--raised);
    min-height: 0;
    overflow: hidden;
  }
  .hit {
    fill: transparent;
  }
  .name {
    font-size: 11px;
    font-weight: 700;
    fill: var(--ink);
  }
  .state {
    font-size: 10.5px;
    font-weight: 600;
    fill: var(--wait-text);
  }
  .master circle {
    fill: var(--wait);
  }
  .master.up circle {
    fill: var(--live);
  }
  .master.up .state {
    fill: var(--ink-2);
  }
  .hourtick {
    stroke: var(--line-2);
  }
  .hour {
    font-size: 9.5px;
    fill: var(--ink-3);
  }
  .gap {
    fill: url(#hatch);
    stroke: var(--line-2);
    stroke-width: 0.5;
  }
  .block {
    fill: var(--ink-2);
  }
  .block.back {
    fill: var(--live);
  }
  .block.retry {
    fill: var(--retry);
  }
  .block.wait {
    fill: var(--wait);
  }
  .block.other {
    fill: var(--ink-3);
  }
  .event {
    fill: var(--raised);
    stroke-width: 2;
    pointer-events: none;
  }
  .event.back {
    stroke: var(--live);
  }
  .event.wait {
    stroke: var(--wait);
  }
  .foot {
    font-size: 9.5px;
    fill: var(--ink-3);
  }
  .foot.late {
    fill: var(--wait-text);
    font-weight: 600;
  }
</style>
