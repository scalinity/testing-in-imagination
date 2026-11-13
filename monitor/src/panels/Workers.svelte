<script lang="ts">
  // What each AI worker is doing: one row per transcript, with a four-line trace of its tool
  // calls over the chosen window. The line a mark sits on says what kind of call it was.
  import Age from "../lib/Age.svelte";
  import Glyph from "../lib/Glyph.svelte";
  import { checkedAgo } from "../lib/sources";
  import { store } from "../lib/store.svelte";
  import { clock, clockS, dayClock, easternMidnight, hour, span, tailPath, zone } from "../lib/time";
  import { hideTip, showTip, tip, type Tip } from "../lib/tip.svelte";
  import type { Call, Worker } from "../lib/types";

  const WINDOWS: [number, string][] = [
    [900, "15 min"],
    [3600, "1 h"],
    [6 * 3600, "6 h"],
    [24 * 3600, "24 h"],
  ];
  // Trace lines, top to bottom, as indexes into a bucket's counts [read, write, run, talk, other].
  const LINES: { k: number; name: string; what: string }[] = [
    { k: 3, name: "tell", what: "delegating to or messaging another agent, or loading a skill" },
    { k: 1, name: "write", what: "creating or editing a file" },
    { k: 2, name: "run", what: "running a shell command" },
    { k: 0, name: "read", what: "reading a file, searching, or fetching a page" },
  ];
  const ROWH = $derived(store.compact ? 31 : 38);
  const BAND = $derived(store.compact ? 5 : 6);
  const TRACE_TOP = $derived(store.compact ? 1 : 2);

  let traceW = $state(300);
  let hover = $state<{ id: string; i: number } | null>(null);
  /** Workers whose recent calls and files are shown in full. Click a name to toggle. */
  let open = $state<Record<string, boolean>>({});

  const view = $derived(store.workers);
  const now = $derived(store.now);
  const all = $derived(view?.workers ?? []);
  // Quiet workers sink below the others; within each group the order never changes, so a
  // row does not jump when a worker pauses for a moment.
  const shown = $derived(
    all
      .filter((w) => w.calls_in_window > 0 || w.state !== "quiet")
      .map((w, i) => ({ w, i }))
      .sort((a, b) => Number(a.w.state === "quiet") - Number(b.w.state === "quiet") || a.i - b.i)
      .map((x) => x.w),
  );
  const hidden = $derived(all.filter((w) => !shown.includes(w)));
  const n = $derived(view?.workers[0]?.buckets.length ?? 180);
  const bw = $derived(traceW / n);
  const lineY = (line: number) => TRACE_TOP + line * (BAND + 1);
  const markH = (count: number) => (count >= 4 ? BAND : count >= 2 ? 4 : 2.5);

  /** One path per trace line: a mark per bucket, taller when the bucket holds more calls. */
  function linePath(w: Worker, line: number): string {
    const k = LINES[line].k;
    let d = "";
    for (let i = 0; i < w.buckets.length; i++) {
      const c = w.buckets[i][k] + (k === 0 ? w.buckets[i][4] : 0);
      if (!c) continue;
      const h = markH(c);
      d += `M${(i * bw).toFixed(1)},${lineY(line) + BAND - h}h${Math.max(1.4, bw - 0.4).toFixed(1)}v${h}h${-Math.max(1.4, bw - 0.4).toFixed(1)}z`;
    }
    return d;
  }
  function errorPath(w: Worker): string {
    let d = "";
    for (let i = 0; i < w.buckets.length; i++) if (w.buckets[i][5]) d += `M${(i * bw).toFixed(1)},${lineY(4)}h${Math.max(2, bw).toFixed(1)}v3h${-Math.max(2, bw).toFixed(1)}z`;
    return d;
  }
  const kindLine = (kind: Call["kind"]) => LINES.findIndex((l) => l.k === (kind === "other" ? 0 : ["read", "write", "run", "talk"].indexOf(kind)));

  function moved(e: PointerEvent, w: Worker) {
    if (!view) return;
    const box = (e.currentTarget as SVGElement).getBoundingClientRect();
    const i = Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * n)));
    hover = { id: w.id, i };
    const from = view.start_ms + i * view.bucket_s * 1000;
    const to = from + view.bucket_s * 1000;
    const b = w.buckets[i];
    const total = b[0] + b[1] + b[2] + b[3] + b[4];
    const more = total - (w.samples[String(i)]?.length ?? 0);
    showTip(
      {
        title: `${w.name}, ${clockS(from).replace(/ [A-Z]+$/, "")} to ${clockS(to)}`,
        sub: total ? `${total} tool ${total === 1 ? "call" : "calls"}${b[5] ? `, ${b[5]} returned an error` : ""}.` : "No tool call in this stretch.",
        body: [...(w.samples[String(i)] ?? []).map((s) => tailPath(s, 110)), ...(more > 0 ? [`and ${more} more`] : [])],
        rows: LINES.filter((l) => b[l.k] + (l.k === 0 ? b[4] : 0) > 0).map((l): [string, string] => [l.name, `${b[l.k] + (l.k === 0 ? b[4] : 0)}: ${l.what}`]),
        source: `${w.file}, ${checkedAgo(store.checked?.transcripts)}. Tool names and descriptions only; outputs are never read.`,
      },
      e.clientX,
      e.clientY,
    );
  }

  function workerTip(w: Worker): Tip {
    const cur = w.current;
    return {
      title: `${w.name}${w.is_lead ? " (lead session)" : ""}`,
      sub:
        w.state === "working"
          ? cur
            ? `Working: ${cur.tool} has been running for ${span(now - cur.t_ms)}.`
            : `Working: last activity ${span(now - (w.last_ms ?? now))} ago, between tool calls.`
          : w.state === "idle"
            ? `Idle for ${span(now - (w.last_ms ?? now))}.`
            : `Quiet since ${w.last_ms ? dayClock(w.last_ms) : "it started"}.`,
      body: w.description || undefined,
      rows: [
        ["Model", w.model ?? "not recorded"],
        ["Active", w.first_ms && w.last_ms ? `${dayClock(w.first_ms)} to ${clock(w.last_ms)}` : "no timed line yet"],
        ["Tool calls", `${w.calls_total} since it started. In this window: ${w.kinds[0] + w.kinds[4]} read, ${w.kinds[1]} write, ${w.kinds[2]} run, ${w.kinds[3]} tell`],
        ["Errors", `${w.errors_total} tool ${w.errors_total === 1 ? "call" : "calls"} returned an error`],
        ["Files touched", w.files_total ? `${w.files_total}; most recent: ${w.files.slice(0, 6).map((f) => `${tailPath(f.path, 44)} (${f.writes ? `${f.writes} written` : `${f.reads} read`})`).join(", ")}` : "none"],
        ...(w.bad_lines ? ([["Unreadable lines", String(w.bad_lines)]] as [string, string][]) : []),
      ],
      source: `${w.file}, ${checkedAgo(store.checked?.transcripts)}.`,
    };
  }

  const doing = (w: Worker) => {
    const c = w.current ?? w.last_call;
    if (!c) return "no tool call yet";
    return `${c.tool} ${tailPath(c.label, 64)}`;
  };

  const ticks = $derived.by(() => {
    if (!view) return [] as { x: number; label: string }[];
    const w = view.window_s * 1000;
    const step = w <= 900e3 ? 5 * 60e3 : w <= 3600e3 ? 15 * 60e3 : w <= 6 * 3600e3 ? 3600e3 : 6 * 3600e3;
    const anchor = easternMidnight(2026, 11, 9);
    const out: { x: number; label: string }[] = [];
    for (let t = anchor + Math.ceil((view.start_ms - anchor) / step) * step; t <= view.end_ms; t += step) {
      out.push({ x: ((t - view.start_ms) / w) * traceW, label: step >= 3600e3 ? hour(t) : clock(t).replace(/ [A-Z]+$/, "") });
    }
    // A label centred too close to either end would spill out of the plot.
    return out.filter((t) => t.x > 20 && t.x < traceW - 18);
  });
</script>

<section class="workers" class:compact={store.compact}>
  <header>
    <h3 class="panel-name">AI workers</h3>
    <span class="key muted" use:tip={{ title: "Reading a trace", body: "Each worker has four lines. A mark's line says what kind of tool call it was; a taller mark means more calls in that slice of time. A red tick underneath marks a call that returned an error. A blue bar is a call still running.", rows: LINES.map((l): [string, string] => [l.name, l.what]) }}>
      {store.compact ? "lines:" : "lines, top to bottom:"} {LINES.map((l) => l.name).join(", ")}
    </span>
    <span class="grow"></span>
    <div class="ranges" role="group" aria-label="Timeline width">
      {#each WINDOWS as [s, label]}
        <button class:on={store.window === s} onclick={() => ((store.window = s), void store.poll())}>{label}</button>
      {/each}
    </div>
    <Age at={store.checked?.transcripts} staleAfter={10_000} verb={store.compact ? "" : "updated"} what="The Claude Code transcripts" cadence="every second" />
  </header>

  <div class="list scroll">
    {#if !view?.dir_exists}
      <p class="empty"><strong>No transcripts found.</strong> The folder {view?.dir} does not exist, so there is no worker activity to show.</p>
    {:else if !shown.length}
      <p class="empty"><strong>No worker has been active in the last {span((view?.window_s ?? 0) * 1000)}.</strong> Widen the timeline to see earlier sessions.</p>
    {/if}
    {#each shown as w (w.id)}
      <div class="row {w.state}">
        <button class="who" aria-expanded={Boolean(open[w.id])} onclick={() => (open[w.id] = !open[w.id])} use:tip={() => workerTip(w)}>
          <p class="l1">
            <span class="lamp" class:breathe={w.state === "working"}><Glyph name={w.state === "working" ? "dot" : w.state === "idle" ? "ring" : "skip"} size={10} /></span>
            <span class="name">{w.name}</span>
            <span class="model">{(w.model ?? "").replace(/^claude-/, "")}</span>
          </p>
          <p class="l2" class:live={w.current != null}>{w.state === "working" || w.current ? doing(w) : `${w.state} for ${span(now - (w.last_ms ?? now))}`}</p>
        </button>
        <svg class="trace" height={ROWH - 4} width={traceW} onpointermove={(e) => moved(e, w)} onpointerleave={() => ((hover = null), hideTip())} role="img" aria-label="Tool calls by {w.name}">
          {#each LINES as _, line}
            <line class="rule" x1="0" x2={traceW} y1={lineY(line) + BAND + 0.5} y2={lineY(line) + BAND + 0.5} />
            <path class="marks" d={linePath(w, line)} />
          {/each}
          <path class="errors" d={errorPath(w)} />
          {#if w.current && view}
            {@const x0 = Math.max(0, ((w.current.t_ms - view.start_ms) / (view.window_s * 1000)) * traceW)}
            <rect class="inflight breathe" x={x0} y={lineY(kindLine(w.current.kind))} width={Math.max(2, traceW - x0)} height={BAND} rx="1.5" />
          {/if}
          {#if hover?.id === w.id}<rect class="cursor" x={hover.i * bw} y="0" width={Math.max(1.5, bw)} height={ROWH - 4} />{/if}
        </svg>
        <span class="count num" use:tip={() => workerTip(w)}>{w.calls_in_window}<small>{w.files_total} {w.files_total === 1 ? "file" : "files"}</small></span>
      </div>
      {#if open[w.id]}
        <div class="detail">
          <div>
            <h4>Latest tool calls, {zone(now)}</h4>
            {#each w.recent.slice(0, 8) as c}
              <p class:err={c.error} use:tip={{ title: `${c.tool}${c.error ? " returned an error" : ""}`, sub: `${dayClock(c.t_ms)}${c.end_ms ? `, took ${span(c.end_ms - c.t_ms)}` : ", still running"}`, body: [c.label, c.detail].filter(Boolean), source: `${w.file}. Outputs are never read.` }}>
                <span class="num t">{clockS(c.t_ms).replace(/ [A-Z]+$/, "")}</span><span class="tool">{c.tool}</span><span class="what">{tailPath(c.label, 70)}</span>
              </p>
            {:else}<p class="muted">none yet</p>{/each}
          </div>
          <div>
            <h4>Files touched, latest first ({w.files_total})</h4>
            {#each w.files.slice(0, 8) as f}
              <p use:tip={{ title: f.path, rows: [["Read", `${f.reads} times`], ["Written or edited", `${f.writes} times`], ["Last", dayClock(f.last_ms)]], source: w.file }}>
                <span class="num t">{f.writes ? `${f.writes} w` : `${f.reads} r`}</span><span class="what code">{tailPath(f.path, 52)}</span>
              </p>
            {:else}<p class="muted">none yet</p>{/each}
          </div>
        </div>
      {/if}
    {/each}
    {#if hidden.length}
      <p class="hidden muted" use:tip={{ title: "Sessions with no activity in this window", rows: hidden.map((w): [string, string] => [w.name, w.last_ms ? `last active ${dayClock(w.last_ms)}` : "no activity recorded"]) }}>
        {hidden.length} earlier {hidden.length === 1 ? "session" : "sessions"} with no activity in this window
      </p>
    {/if}
  </div>

  <div class="axisrow">
    <span class="who zone muted">times in {zone(now)}</span>
    <svg class="axis" height="16" bind:clientWidth={traceW}>
      {#each ticks as t}
        <line x1={t.x} x2={t.x} y1="0" y2="4" />
        <text class="num" x={t.x} y="14" text-anchor="middle">{t.label}</text>
      {/each}
    </svg>
    <span class="count muted">calls</span>
  </div>
</section>

<style>
  .workers {
    display: flex;
    flex-direction: column;
    min-height: 0;
    flex: 1 1 60%;
  }
  header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 10px 4px;
    min-height: 22px;
  }
  .grow {
    flex: 1;
  }
  .key {
    font-size: var(--fs-1);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .ranges {
    display: flex;
    flex: none;
    white-space: nowrap;
    border: 1px solid var(--line-2);
    border-radius: 5px;
    overflow: hidden;
  }
  .ranges button {
    border: 0;
    background: none;
    padding: 1px 8px;
    font-size: var(--fs-1);
    color: var(--ink-3);
    cursor: pointer;
  }
  .ranges button + button {
    border-left: 1px solid var(--line-2);
  }
  .ranges button.on {
    background: var(--raised-2);
    color: var(--ink);
    font-weight: 600;
  }
  .list {
    flex: 1;
    min-height: 0;
    overflow-x: hidden;
  }
  .empty {
    padding: 8px 10px;
  }
  .row,
  .axisrow {
    display: grid;
    grid-template-columns: 216px minmax(0, 1fr) 44px;
    align-items: center;
    gap: 8px;
    padding: 0 6px 0 10px;
  }
  .row {
    height: 38px;
    border-top: 1px solid var(--line);
  }
  .compact .row,
  .compact .axisrow {
    grid-template-columns: 172px minmax(0, 1fr) 42px;
    gap: 6px;
    padding: 0 4px 0 8px;
  }
  .compact .row {
    height: 31px;
  }
  .compact .l1,
  .compact .l2 {
    line-height: 1.2;
  }
  .compact header {
    gap: 7px;
    padding: 0 6px 3px;
  }
  .who {
    min-width: 0;
  }
  button.who {
    display: block;
    border: 0;
    padding: 0;
    background: none;
    text-align: left;
    cursor: pointer;
  }
  button.who[aria-expanded="true"] .name {
    text-decoration: underline;
    text-decoration-color: var(--ink-3);
    text-underline-offset: 3px;
  }
  .count small {
    display: block;
    font-size: 9.5px;
    color: var(--ink-3);
    white-space: nowrap;
  }
  .detail {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
    gap: 14px;
    padding: 4px 10px 8px 25px;
    font-size: var(--fs-1);
    color: var(--ink-2);
    background: var(--raised);
    border-top: 1px solid var(--line);
  }
  .detail h4 {
    margin: 0 0 2px;
    font-size: var(--fs-1);
    font-weight: 600;
    color: var(--ink-3);
  }
  .detail p {
    display: flex;
    gap: 7px;
    white-space: nowrap;
    line-height: 1.5;
  }
  .detail .t {
    flex: none;
    width: 68px;
    color: var(--ink-3);
  }
  .detail div:last-child .t {
    width: 30px;
  }
  .detail .tool {
    flex: none;
    color: var(--ink);
  }
  .detail .what {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .detail .err .tool {
    color: var(--fail-text);
  }
  .l1,
  .l2 {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .l1 {
    display: flex;
    align-items: baseline;
    gap: 5px;
  }
  .lamp {
    display: inline-flex;
    color: var(--ink-3);
    align-self: center;
  }
  .working .lamp {
    color: var(--live);
  }
  .name {
    font-weight: 600;
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .model {
    font-size: var(--fs-1);
    color: var(--ink-3);
    flex: none;
  }
  .l2 {
    font-size: var(--fs-1);
    color: var(--ink-3);
    padding-left: 15px;
  }
  .working .l2 {
    color: var(--ink-2);
  }
  .quiet .name {
    color: var(--ink-2);
    font-weight: 400;
  }
  .trace {
    width: 100%;
  }
  .rule {
    stroke: var(--line);
  }
  .marks {
    fill: var(--ink-2);
  }
  .quiet .marks,
  .idle .marks {
    fill: var(--ink-3);
  }
  .errors {
    fill: var(--fail);
  }
  .inflight {
    fill: var(--live);
  }
  .cursor {
    fill: var(--ink);
    opacity: 0.22;
    pointer-events: none;
  }
  .count {
    text-align: right;
    font-size: var(--fs-1);
    color: var(--ink-2);
  }
  .hidden {
    padding: 6px 10px;
    font-size: var(--fs-1);
    border-top: 1px solid var(--line);
  }
  .axisrow {
    border-top: 1px solid var(--line-2);
  }
  .axis {
    width: 100%;
  }
  .axis line {
    stroke: var(--line-2);
  }
  .axis text {
    font-size: 10px;
    fill: var(--ink-3);
  }
  .axisrow .count,
  .zone {
    font-size: 10px;
    text-align: right;
  }
</style>
