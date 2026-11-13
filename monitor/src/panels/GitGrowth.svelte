<script lang="ts">
  // Commits and code growth on the task branch: one small chart per top-level area, all on one
  // time axis and one vertical scale. Solid is committed; the hollow bar at the right edge is
  // work that exists on disk and is not committed yet.
  import Age from "../lib/Age.svelte";
  import { checkedAgo } from "../lib/sources";
  import { store } from "../lib/store.svelte";
  import { clock, dayClock, easternMidnight, fmt, hour, shortDay, span, zone } from "../lib/time";
  import { hideTip, showTip, tip } from "../lib/tip.svelte";
  import type { Commit } from "../lib/types";

  let width = $state(520);
  // Raw, so that it stays the same object as the commit it points at and can be compared to it.
  let near = $state.raw<Commit | null>(null);

  const LABEL = $derived(store.compact ? 92 : 150);
  const RIGHT = $derived(store.compact ? 146 : 168);
  const ROWH = $derived(store.compact ? 25 : 34);
  const NAMES = ["tii-cee/", "reports/research-advancement/", "monitor/", "elsewhere"];
  const SHORT = $derived(store.compact ? ["tii-cee/", "reports/…/", "monitor/", "elsewhere"] : ["tii-cee/", "reports/…advancement/", "monitor/", "elsewhere"]);
  const WHAT = [
    "The study's Python package: the CEE library, the twin, the pipeline and their tests.",
    "The study's documents: specification, ledger, status, experiment cards.",
    "This monitor.",
    "Every other path in the repository.",
  ];

  const git = $derived(store.git);
  const now = $derived(store.now);
  const commits = $derived(git?.commits ?? []);
  const plotW = $derived(Math.max(40, width - LABEL - RIGHT));
  const t0 = $derived.by(() => {
    const first = commits[0]?.t_ms ?? now - 3600e3;
    return first - Math.max(60e3, (now - first) * 0.03);
  });
  const x = (t: number) => LABEL + Math.min(1, Math.max(0, (t - t0) / (now - t0))) * plotW;

  interface Area {
    i: number;
    points: { t: number; n: number; c: Commit }[];
    committed: number;
    ins: number;
    del: number;
    pending: number;
    pendingFiles: number;
  }
  const areas = $derived.by((): Area[] => {
    if (!git) return [];
    return [0, 1, 2, 3]
      .map((i) => {
        let n = 0;
        const points = commits.filter((c) => c.areas[i].files > 0).map((c) => ({ t: c.t_ms, n: (n += c.areas[i].ins - c.areas[i].del), c }));
        return {
          i,
          points,
          committed: n,
          ins: commits.reduce((s, c) => s + c.areas[i].ins, 0),
          del: commits.reduce((s, c) => s + c.areas[i].del, 0),
          pending: git.modified[i].ins - git.modified[i].del + git.untracked[i].ins,
          pendingFiles: git.modified[i].files + git.untracked[i].files,
        };
      })
      .filter((a) => a.i < 3 || a.points.length || a.pendingFiles);
  });
  const top = $derived(Math.max(1, ...areas.map((a) => Math.max(a.committed, a.committed + a.pending, ...a.points.map((p) => p.n)))));
  const h = (lines: number) => (Math.max(0, lines) / top) * (ROWH - 8);

  function stepPath(a: Area, base: number): string {
    if (!a.points.length) return "";
    let d = `M${x(a.points[0].t)},${base}`;
    let level = 0;
    for (const p of a.points) {
      d += `L${x(p.t)},${base - h(level)}L${x(p.t)},${base - h(p.n)}`;
      level = p.n;
    }
    return `${d}L${x(now)},${base - h(level)}L${x(now)},${base}Z`;
  }

  const ticks = $derived.by(() => {
    const steps = [5 * 60e3, 15 * 60e3, 30 * 60e3, 3600e3, 3 * 3600e3, 6 * 3600e3, 12 * 3600e3, 86400e3, 2 * 86400e3, 7 * 86400e3];
    const want = (now - t0) / Math.max(2, Math.floor(plotW / 80));
    const step = steps.find((s) => s >= want) ?? steps[steps.length - 1];
    const anchor = easternMidnight(2026, 11, 9);
    const out: { t: number; label: string }[] = [];
    for (let t = anchor + Math.ceil((t0 - anchor) / step) * step; t <= now; t += step) {
      const midnight = (t - anchor) % 86400e3 === 0;
      out.push({ t, label: step >= 86400e3 || midnight ? shortDay(t) : step < 3600e3 ? clock(t).replace(/ [A-Z]+$/, "") : hour(t) });
    }
    return out;
  });

  function moved(e: PointerEvent) {
    const box = (e.currentTarget as SVGElement).getBoundingClientRect();
    const px = e.clientX - box.left;
    let best: Commit | null = null;
    for (const c of commits) if (!best || Math.abs(x(c.t_ms) - px) < Math.abs(x(best.t_ms) - px)) best = c;
    near = best && Math.abs(x(best.t_ms) - px) < 40 ? best : null;
    if (!near) return hideTip();
    const c = near;
    showTip(
      {
        title: c.subject,
        sub: `Commit ${c.hash.slice(0, 7)}, ${dayClock(c.t_ms)} (${span(now - c.t_ms)} ago)${c.hash.startsWith(git?.start_commit ?? "-") ? ". This commit started the study." : ""}`,
        rows: c.areas.map((d, i): [string, string] => [NAMES[i], d.files ? `${d.files} ${d.files === 1 ? "file" : "files"}, +${fmt(d.ins)} −${fmt(d.del)} lines` : ""]).filter(([, v]) => v),
        source: `git log on ${git?.branch}, ${checkedAgo(store.checked?.git)}.`,
      },
      e.clientX,
      e.clientY,
    );
  }
</script>

<section class="git" bind:clientWidth={width}>
  <header>
    <h3 class="panel-name" use:tip={{ title: "Commits and lines", body: `Commits on ${git?.branch ?? "the task branch"} since the study started at ${git?.start_commit ?? "its first commit"}. Each dot is a commit; the solid shape is the net lines committed so far; the hollow bar at the right edge is work on disk that is not committed.` }}>Commits and lines</h3>
    {#if !store.compact}<span class="muted note code">{git?.branch ?? ""}</span>{/if}
    {#if git?.ok && git.head_branch && git.head_branch !== git.branch}
      <span class="warn" use:tip={{ title: "A different branch is checked out", body: `The commits drawn are those of ${git.branch}. Uncommitted work is measured against what is checked out now, ${git.head_branch}.` }}>checked out: {git.head_branch}</span>
    {/if}
    <span class="grow"></span>
    <Age at={store.checked?.git} staleAfter={10_000} what="The repository's git state" cadence="every 2 seconds" />
  </header>

  {#if !git}
    <p class="empty">Reading git…</p>
  {:else if !git.ok}
    <p class="empty"><strong>Git could not be read.</strong> {git.error}</p>
  {:else}
    <svg {width} height={areas.length * ROWH + 18} onpointermove={moved} onpointerleave={() => ((near = null), hideTip())} role="img" aria-label="Lines committed over time by area">
      {#each ticks as tk}<line class="grid" x1={x(tk.t)} x2={x(tk.t)} y1="0" y2={areas.length * ROWH} />{/each}
      {#each areas as a, row (a.i)}
        {@const base = (row + 1) * ROWH - 3}
        <g
          use:tip={() => ({
            title: NAMES[a.i],
            sub: `${fmt(a.committed)} net lines committed since the study started (${fmt(a.ins)} inserted, ${fmt(a.del)} deleted)${a.pendingFiles ? `; ${fmt(a.pending)} more lines in ${a.pendingFiles} uncommitted ${a.pendingFiles === 1 ? "file" : "files"}` : ""}.`,
            body: [WHAT[a.i], "Committed lines are insertions minus deletions, summed over the branch's commits. Uncommitted lines are edits to tracked files plus the lines of files git does not track yet; ignored files are left out."],
            rows: [
              ["Edited, tracked", `${git.modified[a.i].files} files, +${fmt(git.modified[a.i].ins)} −${fmt(git.modified[a.i].del)}`],
              ["Not tracked yet", `${git.untracked[a.i].files} files, ${fmt(git.untracked[a.i].ins)} lines`],
            ],
            source: `git log, git diff and git ls-files on ${git.branch}, ${checkedAgo(store.checked?.git)}.`,
          })}
        >
          <rect class="hit" x="0" y={row * ROWH} width={LABEL} height={ROWH} />
          <text class="area-name code" x={store.compact ? 8 : 10} y={base - (store.compact ? 7 : 9)}>{SHORT[a.i]}</text>
        </g>
        <line class="base" x1={LABEL} x2={LABEL + plotW} y1={base + 0.5} y2={base + 0.5} />
        <path class="lines" d={stepPath(a, base)} />
        {#each a.points as p}<circle class="commit" class:near={near === p.c} cx={x(p.t)} cy={base - h(p.n)} r={near === p.c ? 3.5 : 2} />{/each}
        {#if a.pending > 0}
          <rect class="pending" x={LABEL + plotW + 3} y={base - h(a.committed + a.pending)} width="7" height={Math.max(1.5, h(a.committed + a.pending) - h(a.committed))} />
        {/if}
        <text class="val num" x={LABEL + plotW + 15} y={base - (store.compact ? 12 : 14)}>{fmt(a.committed)}<tspan class="pm" dx="6">+{fmt(a.ins)} −{fmt(a.del)}</tspan></text>
        <text class="val2 num" x={LABEL + plotW + 15} y={base - (store.compact ? 1 : 2)}>{a.pendingFiles ? `+${fmt(a.pending)} uncommitted` : "nothing pending"}</text>
      {/each}
      {#each ticks as tk}<text class="tick num" x={x(tk.t)} y={areas.length * ROWH + 13} text-anchor="middle">{tk.label}</text>{/each}
      <text class="tick" x={LABEL + plotW + 3} y={areas.length * ROWH + 13}>now</text>
      <text class="tick" x={LABEL - 6} y={areas.length * ROWH + 13} text-anchor="end">{zone(now)}</text>
    </svg>
  {/if}
</section>

<style>
  .git {
    flex: none;
    border-top: 1px solid var(--line);
    padding-top: 7px;
    min-width: 0;
  }
  header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 10px 2px;
    min-height: 20px;
  }
  .grow {
    flex: 1;
  }
  .note {
    font-size: 10.5px;
  }
  .warn {
    font-size: var(--fs-1);
    color: var(--wait-text);
  }
  .empty {
    padding: 6px 10px;
  }
  .hit {
    fill: transparent;
  }
  .grid {
    stroke: var(--line);
  }
  .base {
    stroke: var(--line-2);
  }
  .area-name {
    font-size: 10.5px;
    fill: var(--ink-2);
  }
  .lines {
    fill: color-mix(in srgb, var(--ink-2) 38%, transparent);
    stroke: var(--ink-2);
    stroke-width: 1;
    stroke-linejoin: round;
  }
  .commit {
    fill: var(--ink);
    stroke: var(--ground);
    stroke-width: 1;
  }
  .pending {
    fill: none;
    stroke: var(--ink-2);
    stroke-width: 1;
  }
  .val {
    font-size: 11px;
    font-weight: 600;
    fill: var(--ink);
  }
  .val2,
  .pm {
    font-size: 10px;
    font-weight: 400;
    fill: var(--ink-3);
  }
  .tick {
    font-size: 10px;
    fill: var(--ink-3);
  }
</style>
