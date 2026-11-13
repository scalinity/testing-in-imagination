<script lang="ts">
  // Slurm work over time: one row per stage with one bar per attempt, and under it, on the same
  // time axis, the GPUs in use, mirrored: the first account above the zero line, the second below.
  import { gpuSteps, type Attempt, type ClusterWork, type Marker, type StageRow, type Step } from "../lib/derive";
  import { clusterSource } from "../lib/sources";
  import { store } from "../lib/store.svelte";
  import { END_STATE, EXIT_CODE, terms } from "../lib/study";
  import { clock, dayClock, easternMidnight, fmt, hour, shortDay, span, zone } from "../lib/time";
  import { hideTip, showTip, tip, type Tip } from "../lib/tip.svelte";

  let { work, limits, started }: { work: ClusterWork; limits: { account: string; gpus: number; source: string }[]; started: boolean } = $props();

  let width = $state(640);
  let range = $state<"all" | "7d" | "24h" | "6h">("24h");
  let cross = $state<number | null>(null);

  const LABEL = $derived(store.compact ? 122 : 172);
  const RIGHT = 26;
  const ROWH = $derived(store.compact ? 18 : 20);
  const MIRROR = $derived(store.compact ? 74 : 96);
  const idChars = $derived(store.compact ? 12 : 18);
  const RANGES = { "6h": 6 * 3600e3, "24h": 86400e3, "7d": 7 * 86400e3 } as const;

  const now = $derived(store.now);
  const known = $derived(work.knownUntil ?? now);
  const plotW = $derived(Math.max(60, width - LABEL - RIGHT));
  const t1 = $derived(now);
  const t0 = $derived.by(() => {
    if (range !== "all") return t1 - RANGES[range];
    const first = work.first ?? t1 - 3600e3;
    return Math.min(first - (t1 - first) * 0.03, t1 - 30 * 60e3);
  });
  const x = (t: number) => LABEL + Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) * plotW;

  // Tick positions at round Eastern times. Eastern is UTC-5 for the whole study window.
  const ticks = $derived.by(() => {
    const steps = [5 * 60e3, 15 * 60e3, 30 * 60e3, 3600e3, 2 * 3600e3, 3 * 3600e3, 6 * 3600e3, 12 * 3600e3, 86400e3, 2 * 86400e3, 7 * 86400e3];
    const want = (t1 - t0) / Math.max(2, Math.floor(plotW / 86));
    const step = steps.find((s) => s >= want) ?? steps[steps.length - 1];
    const anchor = easternMidnight(2026, 11, 9);
    const out: { t: number; label: string; major: boolean }[] = [];
    for (let t = anchor + Math.ceil((t0 - anchor) / step) * step; t <= t1; t += step) {
      const midnight = (t - anchor) % 86400e3 === 0;
      out.push({ t, label: step >= 86400e3 || midnight ? shortDay(t) : step < 3600e3 ? clock(t).replace(/ [A-Z]+$/, "") : hour(t), major: midnight });
    }
    return out;
  });

  type BarLook = "running" | "done" | "cont" | "retry" | "fail";
  function barLook(a: Attempt, row: StageRow): BarLook {
    if (a.refused || a.exitCode === 20 || a.exitCode === 30) return "fail";
    if (a.ended == null) return "running";
    if (a.exitCode === 10) return "cont";
    if (a.state === "COMPLETED") return "done";
    if (a.state === "FAILED" && row.status === "FAILED" && a === row.attempts[row.attempts.length - 1]) return "fail";
    return "retry";
  }
  const letter = (a: Attempt, l: BarLook) => (l === "cont" ? "›" : l === "done" || l === "running" ? "" : a.refused ? "R" : (END_STATE[a.state]?.letter ?? "?"));

  function attemptTip(a: Attempt, row: StageRow): Tip {
    const l = barLook(a, row);
    const rows: [string, string][] = [["Job", a.jobId ? `${a.jobId} on account ${a.account || "unknown"}` : "no job id recorded"]];
    if (a.submitted != null) rows.push(["Submitted", dayClock(a.submitted)]);
    if (a.started != null) rows.push(["Started", `${dayClock(a.started)}${a.submitted != null ? `, after ${span(a.started - a.submitted)} in the queue` : ""}`]);
    if (a.ended != null) rows.push(["Ended", `${dayClock(a.ended)}${a.started != null ? `, after ${span(a.ended - a.started)}` : ""}`]);
    else if (a.started != null) rows.push(["Running for", `${span(known - a.started)} as of the watcher's last cycle`]);
    else if (a.submitted != null) rows.push(["Queued for", `${span(known - a.submitted)} as of the watcher's last cycle`]);
    if (a.state) rows.push(["Scheduler state", `${a.state}. ${END_STATE[a.state]?.meaning ?? ""}`]);
    if (a.exitCode != null) rows.push(["Exit code", `${a.exitCode}${EXIT_CODE[a.exitCode] ? `: ${EXIT_CODE[a.exitCode]}` : ": a failure"}`]);
    if (a.gpuHours != null) rows.push(["GPU-hours", `${fmt(a.gpuHours, 2)}, charged to ${a.experiment || "its experiment"}'s allowance`]);
    return {
      title: `${row.id}, attempt ${a.attempt}`,
      sub: l === "running" ? "Running." : l === "done" ? "Completed." : l === "cont" ? "Checkpointed at the time limit; it continues in the next attempt." : l === "fail" ? (a.refused || a.exitCode === 20 || a.exitCode === 30 ? "Refused. Never retried; it becomes a BLOCKED item." : "Failed; retries are exhausted.") : "Ended early; the watcher resubmits it from its checkpoint.",
      rows,
      terms: terms("stage"),
      source: clusterSource("state/runlog.jsonl"),
    };
  }

  function rowTip(row: StageRow): Tip {
    const s = row.stage;
    return {
      title: `Stage ${row.id}`,
      sub: `${row.experiment || "No experiment named"}. ${row.status ? row.status.toLowerCase() : "state not reported"}.`,
      rows: [
        ["Attempts", String(Math.max(row.attempts.length, s?.attempts ?? 0))],
        ...(s?.depends_on?.length ? ([["Depends on", s.depends_on.join(", ")]] as [string, string][]) : []),
        ...(s?.gate ? ([["Gate", s.gate]] as [string, string][]) : []),
        ...(s?.estimate_gpu_hours != null ? ([["Estimate", `${fmt(s.estimate_gpu_hours, 1)} GPU-hours`]] as [string, string][]) : []),
        ...(s?.gpu_hours != null ? ([["Spent", `${fmt(s.gpu_hours, 2)} GPU-hours`]] as [string, string][]) : []),
      ],
      terms: terms("stage"),
      source: clusterSource(),
    };
  }

  const markerTip = (m: Marker): Tip => ({
    title: m.label,
    sub: dayClock(m.t),
    body: m.kind === "gate" ? "The watcher evaluated this gate from recorded evidence and took the pre-registered path." : m.kind === "blocked" ? "The watcher raised a BLOCKED item: something only Daniel can resolve." : "A manifest or authorization check refused to run this stage.",
    source: clusterSource("state/runlog.jsonl"),
  });

  // GPU mirror
  const accounts = $derived(limits.slice(0, 2));
  const maxLimit = $derived(Math.max(1, ...accounts.map((a) => a.gpus)));
  const unit = $derived((MIRROR / 2 - 10) / maxLimit);
  const mid = $derived(MIRROR / 2);
  const series = $derived(accounts.map((a) => gpuSteps(work, a.account)));

  function valueAt(steps: Step[], t: number): number {
    let n = 0;
    for (const s of steps) {
      if (s.t > t) break;
      n = s.n;
    }
    return n;
  }

  function areaPath(steps: Step[], sign: 1 | -1): string {
    const end = Math.min(known, t1);
    if (end <= t0) return "";
    let n = valueAt(steps, t0);
    let d = `M${x(t0)},${mid}L${x(t0)},${mid - sign * n * unit}`;
    for (const s of steps) {
      if (s.t <= t0 || s.t > end) continue;
      d += `L${x(s.t)},${mid - sign * n * unit}`;
      n = s.n;
      d += `L${x(s.t)},${mid - sign * n * unit}`;
    }
    return `${d}L${x(end)},${mid - sign * n * unit}L${x(end)},${mid}Z`;
  }

  function mirrorMove(e: PointerEvent) {
    const box = (e.currentTarget as SVGElement).getBoundingClientRect();
    const px = e.clientX - box.left;
    const t = t0 + ((px - LABEL) / plotW) * (t1 - t0);
    cross = px;
    const beyond = t > known;
    showTip(
      {
        title: `GPUs in use at ${dayClock(t)}`,
        sub: beyond ? "After the watcher's last cycle: not known yet." : undefined,
        rows: beyond ? [] : accounts.map((a, i): [string, string] => [a.account, `${valueAt(series[i], t)} of ${a.gpus}`]),
        body: "Counted from each attempt's start and end in the run log, not from a scheduler query. The limit is the class QOS limit and is shared with other students, so fewer may be free.",
        source: clusterSource("state/runlog.jsonl"),
      },
      e.clientX,
      e.clientY,
    );
  }

  const short = (id: string, max: number) => (id.length > max ? `${id.slice(0, max - 1)}…` : id);
</script>

<div class="timeline" class:compact={store.compact} bind:clientWidth={width}>
  <header>
    <h3 class="panel-name" use:tip={{ title: "Jobs", body: "One row per stage, one bar per attempt. A hollow bar is time waiting in the queue; a solid bar is time running. A letter after a bar says how the attempt ended: T time limit, P preempted, N node failure, M out of memory, F failed, C cancelled, R refused, and › for a checkpoint that continues.", terms: terms("stage"), source: clusterSource("state/runlog.jsonl") }}>Jobs</h3>
    <ul class="key">
      {#if !store.compact}<li><i class="k queued"></i>queued</li>{/if}
      <li><i class="k running"></i>running</li>
      {#if !store.compact}<li><i class="k done"></i>done</li>{/if}
      <li><i class="k retry"></i>retried</li>
      <li><i class="k fail"></i>failed</li>
    </ul>
    <span class="grow"></span>
    <div class="ranges" role="group" aria-label="Time range">
      {#each ["6h", "24h", "7d", "all"] as const as r}
        <button class:on={range === r} onclick={() => (range = r)}>{r === "all" ? "all" : r.replace("h", " h").replace("d", " d")}</button>
      {/each}
    </div>
  </header>

  <div class="rows scroll">
    {#if work.rows.length}
      <svg {width} height={work.rows.length * ROWH + 4}>
        {#if known < t1}<rect class="unknown" x={x(known)} y="0" width={x(t1) - x(known)} height={work.rows.length * ROWH + 4} />{/if}
        {#each ticks as tk}<line class="grid" class:major={tk.major} x1={x(tk.t)} x2={x(tk.t)} y1="0" y2={work.rows.length * ROWH + 4} />{/each}
        {#each work.rows as row, i (row.id)}
          {@const y = i * ROWH + ROWH / 2 + 2}
          <g class="row">
            <g use:tip={() => rowTip(row)} tabindex="0" role="group" aria-label="Stage {row.id}">
              <rect class="hit" x="0" y={y - ROWH / 2} width={LABEL - 4} height={ROWH} />
              <text class="exp" x={store.compact ? 6 : 10} y={y + 4}>{row.experiment}</text>
              <text class="stage" x={store.compact ? 34 : 42} y={y + 4}>{short(row.id, idChars)}</text>
              {#if row.attempts.length > 1}<text class="times" x={LABEL - 6} y={y + 4} text-anchor="end">×{row.attempts.length}</text>{/if}
            </g>
            {#each row.attempts as a (a.attempt)}
              {@const l = barLook(a, row)}
              {@const qEnd = a.started ?? a.ended ?? known}
              {@const rEnd = a.ended ?? known}
              <g class="attempt {l}" use:tip={() => attemptTip(a, row)}>
                {#if a.submitted != null && qEnd > t0 && qEnd > a.submitted}
                  <rect class="queue" x={x(a.submitted)} y={y - 3} width={Math.max(1.5, x(qEnd) - x(a.submitted))} height="6" rx="1.5" />
                {/if}
                {#if a.started != null && rEnd > t0}
                  <rect class="run" class:breathe={l === "running"} x={x(a.started)} y={y - 5} width={Math.max(2, x(rEnd) - x(a.started) - 1)} height="10" rx="2" />
                  {#if letter(a, l)}<text class="cap" x={x(rEnd) + 2} y={y + 3.5}>{letter(a, l)}</text>{/if}
                {:else if a.refused && a.ended != null && a.ended > t0}
                  <text class="cap" x={x(a.ended) - 3} y={y + 3.5}>R</text>
                {/if}
              </g>
            {/each}
          </g>
        {/each}
      </svg>
    {:else}
      <div class="nothing" style:left="{LABEL}px" style:right="{RIGHT}px">
        <svg width="100%" height="100%"><rect width="100%" height="100%" fill="url(#hatch)" /></svg>
        <p class="empty">
          {#if started}
            <strong>No job has been submitted yet.</strong> Each stage appears here when the watcher first submits it.
          {:else}
            <strong>Not started yet.</strong> Jobs appear here once the watcher runs its first cycle and writes state/runlog.jsonl.
          {/if}
        </p>
      </div>
    {/if}
  </div>

  <svg class="axis" {width} height={MIRROR + 22} onpointermove={mirrorMove} onpointerleave={() => ((cross = null), hideTip())} role="img" aria-label="GPUs in use over time by account">
    <!-- time axis -->
    <line class="axisline" x1={LABEL} x2={LABEL + plotW} y1="0.5" y2="0.5" />
    {#each ticks as tk}
      <line class="tick" x1={x(tk.t)} x2={x(tk.t)} y1="0" y2="4" />
      <text class="ticklabel num" class:major={tk.major} x={x(tk.t)} y="15" text-anchor="middle">{tk.label}</text>
    {/each}
    <text class="nowlabel" x={LABEL + plotW + 4} y="15">now</text>
    <text class="nowlabel" x={LABEL - 6} y="15" text-anchor="end">{zone(now)}</text>
    {#each work.markers.filter((m) => m.t > t0) as m}
      <g class="marker {m.kind}" use:tip={() => markerTip(m)}>
        <rect class="hit" x={x(m.t) - 6} y="-8" width="12" height="16" />
        <path d="M{x(m.t)},-5l4,4l-4,4l-4,-4z" />
      </g>
    {/each}

    <!-- mirrored GPUs -->
    <g transform="translate(0 22)">
      <text class="panel-label" x={store.compact ? 6 : 10} y="12">GPUs in use</text>
      {#if known < t1}<rect class="unknown" x={x(known)} y="0" width={x(t1) - x(known)} height={MIRROR} />{/if}
      {#each accounts as a, i (a.account)}
        {@const sign = i === 0 ? 1 : -1}
        <line class="limit" x1={LABEL} x2={LABEL + plotW} y1={mid - sign * a.gpus * unit} y2={mid - sign * a.gpus * unit} />
        <text class="acct code" x={LABEL - 10} y={mid - sign * (a.gpus * unit) / 2 + 4} text-anchor="end">{a.account}</text>
        <text class="limitlabel num" x={LABEL + plotW + 4} y={mid - sign * a.gpus * unit + 4}>{a.gpus}</text>
        <path class="area" d={areaPath(series[i], sign)} />
      {/each}
      <line class="zero" x1={LABEL} x2={LABEL + plotW} y1={mid} y2={mid} />
      <text class="limitlabel num" x={LABEL + plotW + 4} y={mid + 4}>0</text>
      {#if !accounts.length}<text class="acct" x={LABEL} y={mid - 8}>{store.docs ? "No GPU limits found in the status file or the addendum." : "Waiting for the study's documents."}</text>{/if}
      {#if cross != null && cross >= LABEL && cross <= LABEL + plotW}<line class="cross" x1={cross} x2={cross} y1="0" y2={MIRROR} />{/if}
    </g>
  </svg>
</div>

<style>
  .timeline {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    flex: 1;
  }
  header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 10px 4px;
    min-height: 22px;
    white-space: nowrap;
  }
  .grow {
    flex: 1;
  }
  .key {
    display: flex;
    min-width: 0;
    overflow: hidden;
    gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: var(--fs-1);
    color: var(--ink-3);
  }
  .key li {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .k {
    width: 12px;
    height: 7px;
    border-radius: 2px;
    background: var(--ink-2);
  }
  .k.queued {
    background: none;
    border: 1px solid var(--ink-3);
    height: 5px;
  }
  .k.running {
    background: var(--live);
  }
  .k.retry {
    background: var(--retry);
  }
  .k.fail {
    background: var(--fail);
  }
  .compact .key {
    gap: 6px;
  }
  .compact header {
    gap: 7px;
    padding: 0 6px 3px;
  }
  .ranges {
    display: flex;
    flex: none;
    border: 1px solid var(--line-2);
    border-radius: 5px;
    overflow: hidden;
  }
  .ranges button {
    border: 0;
    background: none;
    padding: 1px 6px;
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
  .rows {
    flex: 1;
    min-height: 44px;
    overflow-x: hidden;
    position: relative;
  }
  .nothing {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
  }
  .nothing svg {
    position: absolute;
    inset: 0;
  }
  .nothing p {
    position: relative;
    padding: 8px 12px;
    background: var(--ground);
    border-radius: 4px;
    margin: 12px;
  }
  .hit {
    fill: transparent;
  }
  .exp {
    font-size: 11px;
    font-weight: 600;
    fill: var(--ink);
  }
  .stage {
    font-family: var(--code);
    font-size: 10.5px;
    fill: var(--ink-2);
  }
  .times {
    font-size: 10px;
    fill: var(--ink-3);
  }
  .grid {
    stroke: var(--line);
    stroke-width: 1;
  }
  .grid.major {
    stroke: var(--line-2);
  }
  .unknown {
    fill: url(#hatch);
  }
  .queue {
    fill: none;
    stroke: var(--ink-3);
    stroke-width: 1;
  }
  .run {
    fill: var(--ink-2);
  }
  .attempt.running .run {
    fill: var(--live);
  }
  .attempt.cont .run {
    fill: var(--ink-3);
  }
  .attempt.retry .run {
    fill: var(--retry);
  }
  .attempt.fail .run {
    fill: var(--fail);
  }
  .cap {
    font-size: 9.5px;
    font-weight: 700;
    fill: var(--ink-3);
  }
  .attempt.retry .cap {
    fill: var(--retry-text);
  }
  .attempt.fail .cap {
    fill: var(--fail-text);
  }
  .attempt:hover .run {
    filter: brightness(1.2);
  }
  .axis {
    flex: none;
    margin-top: 2px;
  }
  .axisline,
  .tick {
    stroke: var(--line-2);
  }
  .ticklabel,
  .nowlabel {
    font-size: 10.5px;
    fill: var(--ink-3);
  }
  .ticklabel.major {
    fill: var(--ink-2);
    font-weight: 600;
  }
  .marker path {
    fill: var(--ink-2);
  }
  .marker.blocked path {
    fill: var(--wait);
  }
  .marker.refused path {
    fill: var(--fail);
  }
  .panel-label {
    font-size: 12px;
    font-weight: 600;
    fill: var(--ink-2);
  }
  .limit {
    stroke: var(--ink-3);
    stroke-width: 1;
    stroke-dasharray: 4 3;
  }
  .zero {
    stroke: var(--line-2);
  }
  .acct {
    font-size: 10.5px;
    fill: var(--ink-2);
  }
  .limitlabel {
    font-size: 10px;
    fill: var(--ink-3);
  }
  .area {
    fill: color-mix(in srgb, var(--ink-2) 38%, transparent);
    stroke: var(--ink-2);
    stroke-width: 1;
    stroke-linejoin: round;
  }
  .cross {
    stroke: var(--ink-3);
    stroke-width: 1;
  }
</style>
