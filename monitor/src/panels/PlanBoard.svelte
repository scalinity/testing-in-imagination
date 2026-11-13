<script lang="ts">
  // The plan board: the specification's own date windows as columns, experiments as chips,
  // gates as diamonds, and the dependencies between them. Columns follow content, not days;
  // each column carries day ticks so that the different scales are visible.
  import Age from "../lib/Age.svelte";
  import Glyph from "../lib/Glyph.svelte";
  import { acceptedGaps, budgets, experimentStatus, gateMet, gateState, stageCounts, belongsTo } from "../lib/derive";
  import { clusterSource, fileSource } from "../lib/sources";
  import { glyphFor, look, meaning, word, type Look } from "../lib/status";
  import { store } from "../lib/store.svelte";
  import { EDGES, EXPERIMENT_NAME, EXPERIMENT_TERMS, FLAGS, JUNCTIONS, MILESTONE_NAME, NODES, REGIONS, ROWS, WINDOWS, terms, xOfTime, type BoardEdge, type BoardNode } from "../lib/study";
  import { day, dayClock, fmt, span } from "../lib/time";
  import { tip, type Tip } from "../lib/tip.svelte";

  let width = $state(1000);

  const PAD = 12;
  // Two sizes: roomy, and compact for a small display.
  const RAIL = $derived(store.compact ? 36 : 44);
  const ROW = $derived(store.compact ? 22 : 28);
  const CHIP = $derived(store.compact ? 16 : 18);
  const G = $derived(store.compact ? 6 : 7);
  const TOP = $derived(RAIL + (store.compact ? 17 : 22));
  const H = $derived(TOP + (ROWS - 1) * ROW + (store.compact ? 13 : 18));

  const X = (f: number) => PAD + f * (width - 2 * PAD);
  const Y = (row: number) => TOP + row * ROW;
  const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

  const docs = $derived(store.docs);
  const remote = $derived(store.remote);
  const now = $derived(store.now);
  const byId = new Map(NODES.map((n) => [n.id, n]));

  interface View {
    node: BoardNode;
    status: string;
    look: Look;
    satisfied: boolean;
    met: boolean | null;
    /** Gates only: undecided, partial (some records, not the whole gate), or decided. */
    gate?: "undecided" | "partial" | "decided";
  }

  const views = $derived.by(() => {
    const out = new Map<string, View>();
    for (const node of NODES) {
      if (node.kind === "experiment") {
        const st = experimentStatus(node.id, remote, docs);
        out.set(node.id, { node, status: st.status, look: look(st.status), satisfied: st.status === "COMPLETED", met: null });
      } else if (node.kind === "gate") {
        const g = gateState(node.id, remote, docs);
        const met = g.decided ? gateMet(g.result, g.path) : null;
        const gate = g.decided ? "decided" : g.partial ? "partial" : "undecided";
        out.set(node.id, { node, status: gate, look: g.decided ? "done" : "planned", satisfied: g.decided && met !== false, met, gate });
      } else {
        const state = docs?.milestones.find((m) => m.id === node.id)?.state ?? "";
        out.set(node.id, { node, status: state, look: state ? look(state) : "unknown", satisfied: state === "COMPLETE", met: null });
      }
    }
    return out;
  });

  // Which of the plan's rows the live specification still has.
  const missingRows = $derived.by(() => {
    if (!docs?.plan.length) return [] as string[];
    const have = new Set(docs.plan.map((p) => p.date));
    return [...WINDOWS.flatMap((w) => w.rows), ...FLAGS.map((f) => f.row)].filter((r, i, all) => !have.has(r) && all.indexOf(r) === i);
  });

  const planText = (row: string) => docs?.plan.find((p) => p.date === row)?.what ?? "";
  const todayX = $derived(X(xOfTime(now)));
  const activeWindows = $derived(WINDOWS.filter((w) => now >= w.from && now < w.to));
  const beforePlan = $derived(now < WINDOWS[0].from);
  const afterPlan = $derived(now >= WINDOWS[WINDOWS.length - 1].to);

  function dayTicks(w: (typeof WINDOWS)[number]): number[] {
    const days = Math.round((w.to - w.from) / 86_400_000);
    return Array.from({ length: Math.max(0, days - 1) }, (_, i) => w.x0 + ((i + 1) / days) * (w.x1 - w.x0));
  }

  function points(e: BoardEdge): [number, number][] {
    const a = byId.get(e.from)!;
    const b = byId.get(e.to)!;
    const mid = (n: BoardNode) => (n.x0 + n.x1) / 2;
    const half = (n: BoardNode) => (n.kind === "gate" ? G : CHIP / 2);
    let start: [number, number];
    if (e.out === "bottom") start = [X(mid(a)), Y(a.row) + half(a)];
    else if (e.out === "top") start = [X(Math.min(Math.max(mid(b), a.x0), a.x1)), Y(a.row) - half(a)];
    else start = [a.kind === "gate" ? X(a.x0) + G : X(a.x1), Y(a.row)];
    const dy = e.enterDy ?? 0;
    const via = (e.via ?? []).map(([f, r], i, all): [number, number] => [X(f), Y(r) + (i === all.length - 1 && r === b.row ? dy : 0)]);
    const prev = via.length ? via[via.length - 1] : start;
    let end: [number, number];
    if (e.into === "top") end = [prev[0], Y(b.row) - half(b)];
    else if (e.into === "bottom") end = [prev[0], Y(b.row) + half(b)];
    else end = [b.kind === "gate" ? X(b.x0) - G : X(b.x0), Y(b.row) + dy];
    return [start, ...via, end];
  }

  /** A polyline with rounded corners. */
  function path(pts: [number, number][]): string {
    const p = pts.filter((q, i) => i === 0 || q[0] !== pts[i - 1][0] || q[1] !== pts[i - 1][1]);
    let d = `M${p[0][0]},${p[0][1]}`;
    for (let i = 1; i < p.length - 1; i++) {
      const [x0, y0] = p[i - 1];
      const [x1, y1] = p[i];
      const [x2, y2] = p[i + 1];
      const inLen = Math.hypot(x1 - x0, y1 - y0);
      const outLen = Math.hypot(x2 - x1, y2 - y1);
      const r = Math.min(5, inLen / 2, outLen / 2);
      d += `L${x1 - ((x1 - x0) / inLen) * r},${y1 - ((y1 - y0) / inLen) * r}Q${x1},${y1} ${x1 + ((x2 - x1) / outLen) * r},${y1 + ((y2 - y1) / outLen) * r}`;
    }
    const last = p[p.length - 1];
    return `${d}L${last[0]},${last[1]}`;
  }

  const edges = $derived(
    [...EDGES]
      .sort((a, b) => Number(a.over ?? false) - Number(b.over ?? false))
      .map((e) => ({ e, d: path(points(e)), released: views.get(e.from)?.satisfied ?? false })),
  );

  function edgeTip(e: BoardEdge, released: boolean): Tip {
    return {
      title: `${e.from} ${e.modifies ? "conditions" : "releases"} ${e.to}`,
      sub: released ? `${e.from} is done, so this path is open.` : `${e.from} is not done yet.`,
      body: e.why,
      terms: e.modifies ? terms("ceeScale") : undefined,
      source: "Dependency transcribed from the specification; states read live.",
    };
  }

  function experimentTip(id: string, part?: string): Tip {
    const node = byId.get(id)!;
    const st = experimentStatus(id, remote, docs);
    const rows = (docs?.experiments ?? []).filter((e) => (part ? e.id === part : belongsTo(e.id, id)));
    const card = docs?.cards.find((c) => c.id === id);
    const register = docs?.register.find((r) => belongsTo(r.id, id));
    const budget = budgets(remote, docs).find((b) => b.id === id);
    const counts = stageCounts(id, remote);
    const one = rows.length === 1 ? rows[0] : null;
    const detail: [string, string][] = [];
    if (one) {
      detail.push(["Design", one.design], ["Measures", one.measurement], ["Decision rule", one.rule], ["Kills the claim", one.kills], ["Compute", one.compute]);
    }
    if (!part) {
      detail.push(["Frozen card", card ? card.frozen : register?.card ? `${register.card} (no card file yet)` : "none yet"]);
      detail.push(["Plan", `${node.planNote} Spec 17.2, "${node.planRow}": ${planText(node.planRow) || "row not found in the live specification"}`]);
      if (st.note) detail.push(["Ledger note", st.note]);
      if (counts) detail.push(["Stages", `${counts.done} of ${counts.total} done, ${counts.live} running`]);
      if (budget?.allowance) {
        detail.push(["GPU-hours", `${budget.spent == null ? "none reported" : fmt(budget.spent, 1)} spent; estimate ${fmt(budget.estimate ?? 0)}, allowance ${fmt(budget.allowance)}`]);
      }
    }
    return {
      title: part ? `${part}, a part of ${id}` : `${id}: ${EXPERIMENT_NAME[id] ?? ""}`,
      sub: part ? undefined : `${cap(word(st.status))}${st.detail ? ` ${st.detail}` : ""}. ${meaning(st.status)}`,
      body: rows.length ? (one ? one.question : rows.map((r) => `${r.id}: ${r.question}`)) : "The specification's experiment table has no row for this id.",
      rows: detail,
      terms: part ? undefined : terms(...(EXPERIMENT_TERMS[id] ?? []).slice(0, 2)),
      source: `Status from ${st.source}${st.other ? `; ${st.other}` : ""}. Text from ${fileSource(docs?.spec, "section 13")}`,
    };
  }

  function gateTip(id: string): Tip {
    const node = byId.get(id)!;
    const g = gateState(id, remote, docs);
    const spec = docs?.gates.find((x) => x.id === id);
    const met = g.decided ? gateMet(g.result, g.path) : null;
    return {
      title: `Gate ${id}`,
      sub: g.decided
        ? `Decided${met == null ? "" : met ? ": met" : ": not met"}. Record written ${g.decidedAt ? `${dayClock(g.decidedAt)} (${span(now - g.decidedAt)} ago)` : "at an unknown time"}.`
        : g.partial
          ? `Decided in part: ${g.records.length} ${g.records.length === 1 ? "record covers" : "records cover"} some of what this gate releases. The rest is still open.`
          : "Not decided yet. The agent decides it from recorded evidence; it does not wait for a person.",
      rows: [
        ["Rule", spec?.rule ?? "not found in the specification"],
        ["If met", spec?.if_met ?? ""],
        ["If not met", spec?.if_not ?? ""],
        ...(g.decided && !g.records.length ? ([["Result", g.result || "not stated"], ["Path taken", g.path || "not stated"]] as [string, string][]) : []),
        ...g.records.map((r): [string, string] => [r.name, `${r.title}. Result: ${r.result || "not stated"}${r.path ? ` Path taken: ${r.path}` : ""}`]),
        ["Plan", node.planNote],
      ],
      terms: terms("gate"),
      source: `State from ${g.source}. Rule from ${fileSource(docs?.spec, "section 14.1")}`,
    };
  }

  function milestoneTip(id: string): Tip {
    const node = byId.get(id);
    const region = REGIONS.find((r) => r.id === id);
    const state = docs?.milestones.find((m) => m.id === id)?.state;
    return {
      title: `${id}: ${MILESTONE_NAME[id] ?? ""}`,
      sub: state ? `${cap(state.toLowerCase())}.` : "STATUS.md does not state this milestone.",
      body: node?.planNote ?? `Its work is ${region?.members.join(", ")} (${region?.spec}). The tinted region marks where that work sits on the plan.`,
      rows: docs?.status_milestone_line ? [["STATUS.md says", docs.status_milestone_line]] : [],
      source: fileSource(docs?.status_file),
    };
  }

  const nodeTip = (n: BoardNode) => () => (n.kind === "experiment" ? experimentTip(n.id) : n.kind === "gate" ? gateTip(n.id) : milestoneTip(n.id));

  function windowTip(w: (typeof WINDOWS)[number]): Tip {
    const active = now >= w.from && now < w.to;
    const days = Math.round((w.to - w.from) / 86_400_000);
    return {
      title: `${w.label}${active ? ": the plan's current window" : ""}`,
      sub: `${days} ${days === 1 ? "day" : "days"}. Column widths follow content, not days; the ticks mark each day.`,
      rows: w.rows.map((r): [string, string] => [r, planText(r) || "This row is no longer in the specification."]),
      source: fileSource(docs?.spec, "section 17.2"),
    };
  }

  const fits = (text: string, px: number) => text.length * 5.7 <= px;
  const E1_PARTS = ["E1-A", "E1-B", "E1-C", "E1-D"];
  const legend = ["PLANNED", "QUEUED", "RUNNING", "COMPLETED", "BLOCKED", "FAILED", "SKIPPED BY GATE"];
  const gaps = $derived(acceptedGaps(docs));
</script>

<section class="board" class:compact={store.compact}>
  <header>
    <h2 class="panel-name">Plan and dependencies</h2>
    <ul class="legend">
      {#each legend as status}
        <li><span class="swatch {look(status)}"><Glyph name={glyphFor(status)} size={9} /></span>{store.compact && status === "SKIPPED BY GATE" ? "skipped" : word(status)}</li>
      {/each}
      <li><svg width="12" height="12" viewBox="-7 -7 14 14"><path class="gate-swatch" d="M0,-5.5L5.5,0L0,5.5L-5.5,0Z" /></svg>{store.compact ? "gate" : "gate, filled once decided"}</li>
    </ul>
    {#if gaps.length}
      <span
        class="gaps"
        tabindex="0"
        use:tip={() => ({
          title: gaps.length === 1 ? "Accepted gap" : `${gaps.length} accepted gaps`,
          body: "Limits Daniel accepted for the study. Nothing waits on them; each is stated in the paper as a limitation.",
          rows: gaps.map((g): [string, string] => [g.id, `${g.title}. ${g.status}`]),
          source: fileSource(docs?.blocked_file, "Accepted gaps"),
        })}>{gaps.length === 1 ? "1 accepted gap" : `${gaps.length} accepted gaps`}</span
      >
    {/if}
    {#if missingRows.length}
      <span
        class="warn"
        use:tip={{
          title: "The specification's dates changed",
          body: "The columns were laid out from version 3.1 of spec 17.2. These rows are no longer in the live file, so the layout may be out of date:",
          rows: missingRows.map((r): [string, string] => ["Missing row", r]),
          source: fileSource(docs?.spec, "section 17.2"),
        }}><Glyph name="clock" /> plan dates changed in the spec</span
      >
    {/if}
    <span class="grow"></span>
    <Age at={store.checked?.docs} staleAfter={10_000} verb={store.compact ? "checked" : "documents checked"} what="The specification, ledger and status documents" cadence="every 2 seconds" />
  </header>

  <div class="canvas" bind:clientWidth={width}>
    <svg {width} height={H} role="img" aria-label="Plan board: experiments and gates by plan window">
      <!-- Calendar rail -->
      {#each WINDOWS as w (w.key)}
        {@const active = activeWindows.includes(w)}
        {#if active}<rect class="active-col" x={X(w.x0)} y={RAIL} width={X(w.x1) - X(w.x0)} height={H - RAIL} />{/if}
        <g class="window" class:active use:tip={() => windowTip(w)} tabindex="0" role="group" aria-label="Plan window {w.label}">
          <rect class="head" x={X(w.x0)} y={14} width={X(w.x1) - X(w.x0)} height={RAIL - 14} />
          {#if w.key !== "w0"}<text x={X(w.x0) + 6} y={RAIL - (store.compact ? 6 : 8)}>{w.label}</text>{/if}
          {#each dayTicks(w) as f}<line class="daytick" x1={X(f)} x2={X(f)} y1={RAIL - 4} y2={RAIL} />{/each}
        </g>
        <line class="colrule" x1={X(w.x0)} x2={X(w.x0)} y1={14} y2={H} />
      {/each}
      <line class="colrule" x1={X(1)} x2={X(1)} y1={14} y2={H} />
      <line class="railrule" x1={X(0)} x2={X(1)} y1={RAIL} y2={RAIL} />

      {#each FLAGS as f (f.row)}
        <g
          class="flag"
          tabindex="0"
          role="group"
          aria-label={f.name}
          use:tip={() => ({ title: `${f.name}, ${f.row}`, body: planText(f.row) || "This row is no longer in the specification.", source: fileSource(docs?.spec, "section 17.2") })}
        >
          <rect class="hit" x={X(xOfTime(f.at)) - 6} y={10} width={12} height={RAIL - 22} />
          <line x1={X(xOfTime(f.at))} x2={X(xOfTime(f.at))} y1={13} y2={RAIL - 13} />
          <path d="M{X(xOfTime(f.at))},13h6l-2,2.5l2,2.5h-6z" />
        </g>
      {/each}

      <!-- Milestones made of experiments: a tinted region under their work -->
      {#each REGIONS as r (r.id)}
        {@const state = docs?.milestones.find((m) => m.id === r.id)?.state ?? ""}
        <g class="region {state ? look(state) : 'unknown'}" role="group" aria-label="Milestone {r.id}, {word(state)}" use:tip={() => milestoneTip(r.id)}>
          <path d={r.rects.map(([a, b, r0, r1]) => `M${X(a)},${Y(r0) - ROW / 2 + 2}H${X(b)}V${Y(r1) + ROW / 2 - 2}H${X(a)}Z`).join("")} />
          {#each r.labels as [lx, lr]}
            <g class="mark"><Glyph name={glyphFor(state)} size={9} x={X(lx)} y={Y(lr) - 4.5} /></g>
            <text x={X(lx) + 12} y={Y(lr) + 3.5}>{r.id}</text>
          {/each}
        </g>
      {/each}

      <!-- Dependencies -->
      {#each edges as { e, d, released } (e.from + e.to)}
        {#if e.over}<path class="halo" {d} />{/if}
        <path class="edge" class:released class:modifies={e.modifies} {d} />
        <path class="edge-hit" {d} use:tip={() => edgeTip(e, released)} />
      {/each}
      {#each JUNCTIONS as [f, r]}<circle class="junction" cx={X(f)} cy={Y(r)} r="2" />{/each}

      <!-- Nodes -->
      {#each NODES as n (n.id)}
        {@const v = views.get(n.id)!}
        {#if n.kind === "gate"}
          <g class="gate {v.gate}" tabindex="0" role="group" aria-label="Gate {n.id}, {v.status}" use:tip={nodeTip(n)}>
            <rect class="hit" x={X(n.x0) - 12} y={Y(n.row) - 18} width={24} height={30} />
            <path d="M{X(n.x0)},{Y(n.row) - G}l{G},{G}l{-G},{G}l{-G},{-G}z" />
            {#if v.gate === "partial"}<path class="half" d="M{X(n.x0)},{Y(n.row) - G}l{-G},{G}l{G},{G}z" />{/if}
            {#if v.met === false}<line class="notmet" x1={X(n.x0) - 3.5} x2={X(n.x0) + 3.5} y1={Y(n.row)} y2={Y(n.row)} />{/if}
            <text x={X(n.x0)} y={Y(n.row) - G - (store.compact ? 2 : 3.5)} text-anchor="middle">{n.id}</text>
          </g>
        {:else}
          {@const name = n.kind === "experiment" ? (EXPERIMENT_NAME[n.id] ?? "") : ""}
          {@const idW = n.id.length * 6.6 + 18}
          {@const w = Math.max(X(n.x1) - X(n.x0), idW + 2)}
          {@const x = X(n.x1) - w}
          {@const counts = n.kind === "experiment" ? stageCounts(n.id, remote) : null}
          <g
            class="chip {v.look} {n.kind}"
            tabindex="0"
            role="group"
            aria-label="{n.id} {name}, {word(v.status)}"
            use:tip={nodeTip(n)}
          >
            <rect class="body" {x} y={Y(n.row) - CHIP / 2} width={w} height={CHIP} rx={n.kind === "milestone" ? CHIP / 2 : 3} />
            {#if counts && counts.total > 0 && v.look !== "done"}
              <rect class="progress done-part" x={x + 1} y={Y(n.row) + CHIP / 2 - 3} width={Math.max(0, (w - 2) * (counts.done / counts.total))} height="2" />
              <rect class="progress live-part" x={x + 1 + (w - 2) * (counts.done / counts.total)} y={Y(n.row) + CHIP / 2 - 3} width={Math.max(0, (w - 2) * (counts.live / counts.total))} height="2" />
            {/if}
            <g class="mark" class:breathe={v.look === "running"}><Glyph name={glyphFor(v.status)} size={10} x={x + 4} y={Y(n.row) - 5} /></g>
            <text class="id" x={x + (v.look === "unknown" ? 6 : 16)} y={Y(n.row) + 3.8}>{n.id}</text>
            {#if n.id === "E1" && w >= 88}
              {#each E1_PARTS as part, i (part)}
                {@const px = x + w - (4 - i) * 15 - 2}
                <g class="part" use:tip={() => experimentTip("E1", part)}>
                  <rect x={px} y={Y(n.row) - 6} width="13" height="12" rx="2" />
                  <text x={px + 6.5} y={Y(n.row) + 3.6} text-anchor="middle">{part.slice(-1)}</text>
                </g>
              {/each}
            {:else if name && fits(name, w - idW - 6)}
              <text class="name" x={x + idW} y={Y(n.row) + 3.8}>{name}</text>
            {/if}
          </g>
        {/if}
      {/each}

      <!-- Today -->
      {#if !beforePlan && !afterPlan}
        <g
          class="today"
          use:tip={{
            title: `Today is ${day(now)}`,
            body: "The pointer shows where today falls in the plan's dates. Work in columns to its left was planned to be done by now; the specification puts quality before any date, so being behind is not drawn as an alarm.",
            source: "This Mac's clock, shown in US Eastern.",
          }}
        >
          <rect class="hit" x={todayX - 10} y={0} width={20} height={RAIL} />
          <line x1={todayX} x2={todayX} y1={10} y2={RAIL} />
          <path d="M{todayX - 4.5},3h9l-4.5,7z" />
          <text x={todayX + (todayX > width - 120 ? -8 : 8)} y={10} text-anchor={todayX > width - 120 ? "end" : "start"}>today, {day(now)}</text>
        </g>
      {:else}
        <text class="offplan" x={beforePlan ? X(0) : X(1)} y={10} text-anchor={beforePlan ? "start" : "end"}>today, {day(now)}, is {beforePlan ? "before" : "after"} the plan's dates</text>
      {/if}
    </svg>
  </div>
</section>

<style>
  .board {
    display: flex;
    flex-direction: column;
    min-width: 0;
    padding: 8px 4px 0 4px;
  }
  header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 12px 2px;
    min-height: 20px;
  }
  .grow {
    flex: 1;
  }
  .legend {
    display: flex;
    flex-wrap: nowrap;
    overflow: hidden;
    min-width: 0;
    gap: 2px 11px;
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: var(--fs-1);
    color: var(--ink-3);
  }
  .legend li {
    display: flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
  }
  .swatch {
    display: inline-grid;
    place-items: center;
    width: 15px;
    height: 12px;
    border-radius: 3px;
    border: 1px solid var(--ink-3);
    color: var(--ink-3);
  }
  .swatch.queued {
    border-style: dashed;
    border-color: var(--ink-2);
    color: var(--ink-2);
  }
  .swatch.running {
    border-color: var(--live);
    background: color-mix(in srgb, var(--live) 22%, transparent);
    color: var(--live-text);
  }
  .swatch.done {
    border-color: var(--done);
    background: var(--done);
    color: var(--ink);
  }
  .swatch.wait {
    border-color: var(--wait);
    background: color-mix(in srgb, var(--wait) 14%, transparent);
    color: var(--wait-text);
  }
  .swatch.fail {
    border-color: var(--fail);
    background: color-mix(in srgb, var(--fail) 18%, transparent);
    color: var(--fail-text);
  }
  .swatch.skip {
    border-style: dotted;
  }
  .gate-swatch {
    fill: none;
    stroke: var(--ink-3);
    stroke-width: 1.2;
  }
  .warn {
    color: var(--wait-text);
    font-size: var(--fs-1);
  }
  .gaps {
    color: var(--ink-3);
    font-size: var(--fs-1);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .canvas {
    min-width: 0;
  }

  /* Rail */
  .head {
    fill: transparent;
  }
  .window.active .head {
    fill: var(--raised-2);
  }
  .window text {
    font-size: 11px;
    fill: var(--ink-3);
  }
  .window.active text {
    fill: var(--ink);
    font-weight: 600;
  }
  .active-col {
    fill: var(--raised);
  }
  .colrule,
  .railrule {
    stroke: var(--line);
    stroke-width: 1;
  }
  .daytick {
    stroke: var(--line-2);
    stroke-width: 1;
  }
  .flag line,
  .flag path {
    stroke: var(--ink-3);
    fill: var(--ink-3);
    stroke-width: 1;
  }
  .hit {
    fill: transparent;
    stroke: none;
  }
  .today line {
    stroke: var(--live);
    stroke-width: 1.5;
  }
  .today path {
    fill: var(--live);
  }
  .today text,
  .offplan {
    font-size: 11px;
    font-weight: 600;
    fill: var(--ink);
  }

  /* Dependencies */
  .edge {
    fill: none;
    stroke: var(--line-2);
    stroke-width: 1.25;
  }
  .edge.released {
    stroke: var(--ink-2);
  }
  .edge.modifies {
    stroke-dasharray: 3 3;
  }
  .halo {
    fill: none;
    stroke: var(--ground);
    stroke-width: 6;
  }
  .edge-hit {
    fill: none;
    stroke: transparent;
    stroke-width: 9;
  }
  .junction {
    fill: var(--line-2);
  }

  /* Gates */
  .gate path {
    fill: var(--ground);
    stroke: var(--ink-3);
    stroke-width: 1.4;
  }
  .gate.decided path {
    fill: var(--ink-2);
    stroke: var(--ink-2);
  }
  .gate.partial path {
    stroke: var(--ink-2);
  }
  .gate.partial .half {
    fill: var(--ink-2);
    stroke: none;
  }
  .gate .notmet {
    stroke: var(--ground);
    stroke-width: 2;
  }
  .gate text {
    font-size: 10px;
    font-weight: 600;
    fill: var(--ink-3);
  }
  .compact .gate text {
    font-size: 9px;
  }
  .compact .chip text {
    font-size: 10.5px;
  }
  .compact .legend {
    gap: 2px 8px;
  }
  .compact header {
    gap: 10px;
  }
  .gate.decided text,
  .gate.partial text {
    fill: var(--ink);
  }

  /* Chips */
  .chip .body {
    fill: var(--ground);
    stroke: var(--ink-3);
    stroke-width: 1;
  }
  .chip text {
    font-size: 11px;
  }
  .chip .id {
    font-weight: 600;
    fill: var(--ink);
  }
  .chip .name {
    fill: var(--ink-3);
  }
  .chip .mark {
    color: var(--ink-3);
  }
  .chip.queued .body {
    stroke: var(--ink-2);
    stroke-dasharray: 3 2;
  }
  .region path {
    fill: var(--ink-3);
    fill-opacity: 0.07;
  }
  .region.running path {
    fill: var(--live);
    fill-opacity: 0.09;
  }
  .region text {
    fill: var(--ink-3);
    font-size: var(--fs-1);
    font-weight: 700;
  }
  .region .mark {
    color: var(--ink-3);
  }
  .region.running text,
  .region.running .mark {
    fill: var(--live-text);
    color: var(--live-text);
  }
  .chip.running .body {
    fill: color-mix(in srgb, var(--live) 20%, var(--ground));
    stroke: var(--live);
  }
  .chip.running .mark {
    color: var(--live-text);
  }
  .chip.running .name {
    fill: var(--ink-2);
  }
  .chip.done .body {
    fill: var(--done);
    stroke: var(--done);
  }
  .chip.done .id {
    fill: var(--ink);
  }
  .chip.done .name {
    fill: var(--ink-2);
  }
  .chip.done .mark {
    color: var(--ink);
  }
  .chip.wait .body {
    fill: color-mix(in srgb, var(--wait) 13%, var(--ground));
    stroke: var(--wait);
  }
  .chip.wait .mark {
    color: var(--wait-text);
  }
  .chip.fail .body {
    fill: color-mix(in srgb, var(--fail) 18%, var(--ground));
    stroke: var(--fail);
  }
  .chip.fail .mark {
    color: var(--fail-text);
  }
  .chip.retry .body {
    fill: color-mix(in srgb, var(--retry) 13%, var(--ground));
    stroke: var(--retry);
  }
  .chip.retry .mark {
    color: var(--retry-text);
  }
  .chip.skip .body {
    stroke-dasharray: 1 3;
  }
  .chip.skip .id,
  .chip.skip .name {
    text-decoration: line-through;
    fill: var(--ink-3);
  }
  .chip.unknown .body {
    fill: url(#hatch);
    stroke: var(--line-2);
  }
  .chip.milestone .id {
    font-weight: 600;
    fill: var(--ink-2);
  }
  .chip.milestone.done .id {
    fill: var(--ink);
  }
  .progress.done-part {
    fill: var(--ink-2);
  }
  .progress.live-part {
    fill: var(--live);
  }
  .part rect {
    fill: none;
    stroke: var(--line-2);
  }
  .part text {
    font-size: 9.5px;
    font-weight: 600;
    fill: var(--ink-3);
  }
  .chip.done .part rect {
    stroke: var(--ink-3);
  }
  .chip.done .part text {
    fill: var(--ink);
  }
  .chip.running .part rect {
    stroke: var(--live);
  }
  g[tabindex]:focus-visible {
    outline: none;
  }
  g[tabindex]:focus-visible .body,
  g[tabindex]:focus-visible > path,
  g[tabindex]:focus-visible .head {
    stroke: var(--live);
    stroke-width: 2;
  }
</style>
