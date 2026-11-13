<script lang="ts">
  // Figure 1 of the paper (specification 1.7). Each panel is drawn only from its own file under
  // tii-cee/results/figure1/. A panel whose file does not exist yet says which experiments it
  // waits for; nothing is drawn from data that do not exist. On the board it is a thumbnail;
  // a click opens it large (the same component with `big`).
  import { scaleLinear, scaleLog, scaleSqrt } from "d3-scale";
  import Age from "../lib/Age.svelte";
  import Glyph from "../lib/Glyph.svelte";
  import { experimentStatus } from "../lib/derive";
  import { fileSource } from "../lib/sources";
  import { glyphFor, look, word } from "../lib/status";
  import { store } from "../lib/store.svelte";
  import { FIGURE_PANELS, terms } from "../lib/study";
  import { fmt } from "../lib/time";
  import { tip, type Tip } from "../lib/tip.svelte";
  import type { FigureFile } from "../lib/types";

  let { big = false }: { big?: boolean } = $props();

  const docs = $derived(store.docs);
  const file = (name: string): FigureFile | undefined => docs?.figure.find((f) => f.name === name);
  const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const f3 = (v: unknown) => (num(v) == null ? "not given" : fmt(num(v)!, 3));
  const range = (lo: unknown, hi: unknown) => (num(lo) != null && num(hi) != null ? ` [${f3(lo)}, ${f3(hi)}]` : "");

  // Sizes: enlarged, roomy thumbnail, compact thumbnail.
  const size = $derived(
    big
      ? { aw: Math.max(300, Math.min(560, store.vw - 440)), ah: Math.max(300, Math.min(470, store.vh - 110)), bw: 340 }
      : store.compact
        ? { aw: 156, ah: 232, bw: 118 }
        : { aw: 222, ah: 254, bw: 176 },
  );
  const AW = $derived(size.aw);
  const AH = $derived(size.ah);
  const BW = $derived(size.bw);
  const BH = $derived((AH - 4) / 2);

  // Panel A: regime map
  const A = { l: 30, r: 8, t: 34, b: 30 };
  const regime = $derived(file("regime_map.json")?.json ?? null);
  const markers = $derived(((regime?.markers ?? []) as any[]).filter((m) => (num(m.tau_b) ?? 0) > 0 && num(m.kl_inc) != null));
  const harmful = $derived(((regime?.harmful_region ?? []) as any[]).filter((p) => (num(p.tau_b) ?? 0) > 0 && num(p.kl_inc) != null));
  const ax = $derived.by(() => {
    const taus = [...markers.flatMap((m) => [m.tau_b, num(m.tau_b_lo) ?? m.tau_b, num(m.tau_b_hi) ?? m.tau_b]), ...harmful.map((p) => p.tau_b), 1];
    return scaleLog()
      .domain([Math.min(Math.min(...taus) / 1.25, 0.5), Math.max(Math.max(...taus) * 1.25, 2)])
      .range([A.l, AW - A.r]);
  });
  const ay = $derived.by(() => {
    const ys = [...markers.flatMap((m) => [m.kl_inc, num(m.kl_inc_hi) ?? m.kl_inc]), ...harmful.map((p) => p.kl_inc), 0.01];
    return scaleLinear()
      .domain([0, Math.max(...ys) * 1.12])
      .range([AH - A.b, A.t])
      .nice(4);
  });
  const tauTicks = $derived([0.125, 0.25, 0.5, 1, 2, 4, 8, 16].filter((t) => t >= ax.domain()[0] && t <= ax.domain()[1]));
  const sizeOf = $derived(
    scaleSqrt()
      .domain([0, Math.max(1, ...markers.map((m) => num(m.params_b) ?? 0))])
      .range(big ? [4, 10] : [3, 7]),
  );
  const maxDelta = $derived(Math.max(1e-9, ...markers.map((m) => Math.abs(num(m.delta_log_loss) ?? 0))));
  const arrow = (m: any) => ((num(m.delta_log_loss) ?? 0) / maxDelta) * (big ? 34 : 20);
  const datasets = $derived([...new Set(markers.filter((m) => m.kind === "public_dataset").map((m) => String(m.dataset ?? m.label)))]);
  const star = (cx: number, cy: number, r: number) =>
    Array.from({ length: 10 }, (_, i) => {
      const a = (Math.PI / 5) * i - Math.PI / 2;
      const rr = i % 2 ? r * 0.45 : r;
      return `${cx + Math.cos(a) * rr},${cy + Math.sin(a) * rr}`;
    }).join(" ");
  const KIND: Record<string, string> = { simulated: "simulated reader family (twin)", language_model: "language-model reader", trained_policy: "trained policy", second_backbone: "second-backbone policy", public_dataset: "public dataset" };

  const markerTip = (m: any): Tip => ({
    title: String(m.label ?? m.id ?? "reader"),
    sub: `${KIND[m.kind] ?? m.kind ?? "reader"}, from ${m.experiment ?? "an unnamed experiment"}.`,
    rows: [
      ["tau_B", `${f3(m.tau_b)}${range(m.tau_b_lo, m.tau_b_hi)}: ${m.tau_b > 1 ? "reacts more than the test can justify" : m.tau_b < 1 ? "reacts less than the test can justify" : "matched"}`],
      ["Incoherence", `${f3(m.kl_inc)}${range(m.kl_inc_lo, m.kl_inc_hi)} nats per decision, after scaling`],
      ["Change in log loss", num(m.delta_log_loss) == null ? "not given" : `${f3(m.delta_log_loss)} nats under the pre-registered repair (${m.delta_log_loss < 0 ? "lower loss: the arrow points down" : "higher loss: the arrow points up"})`],
      ...(num(m.params_b) != null ? ([["Parameters", `${fmt(m.params_b, 1)} billion`]] as [string, string][]) : []),
    ],
    terms: terms("tauB", "klInc"),
    source: fileSource(file("regime_map.json")),
  });

  // Panel B: predicted against realized gain
  const B = { l: 30, r: 6, t: 30, b: 26 };
  const cells = $derived(((file("gain_accounting.json")?.json?.cells ?? []) as any[]).filter((c) => num(c.predicted_gain) != null && num(c.realized_gain) != null));
  const gain = $derived.by(() => {
    const all = cells.flatMap((c) => [c.predicted_gain, c.realized_gain, num(c.predicted_lo) ?? c.predicted_gain, num(c.predicted_hi) ?? c.predicted_gain, num(c.realized_lo) ?? c.realized_gain, num(c.realized_hi) ?? c.realized_gain, 0]);
    const lo = Math.min(...all);
    const hi = Math.max(...all, lo + 1e-6);
    const pad = (hi - lo) * 0.08;
    return {
      x: scaleLinear()
        .domain([lo - pad, hi + pad])
        .range([B.l, BW - B.r]),
      y: scaleLinear()
        .domain([lo - pad, hi + pad])
        .range([BH - B.b, B.t]),
    };
  });
  const cellTip = (c: any): Tip => ({
    title: `${c.reader ?? "reader"} on ${c.setting ?? "setting"}`,
    sub: `From ${c.experiment ?? "an unnamed experiment"}.`,
    rows: [
      ["Predicted gain", `${f3(c.predicted_gain)}${range(c.predicted_lo, c.predicted_hi)} nats`],
      ["Realized gain", `${f3(c.realized_gain)}${range(c.realized_lo, c.realized_hi)} nats`],
      ["Realized minus predicted", `${f3(c.realized_gain - c.predicted_gain)} nats`],
    ],
    terms: terms("gainIdentity", "realizedGain"),
    source: fileSource(file("gain_accounting.json")),
  });

  // Panel C: regret attribution
  const agents = $derived(
    ((file("regret_attribution.json")?.json?.agents ?? []) as any[])
      .filter((a) => num(a.selection) != null && num(a.reading) != null && num(a.commitment) != null)
      .slice(0, big ? 10 : 6),
  );
  const PARTS = ["selection", "reading", "commitment"] as const;
  const narrow = $derived(BW < 150);
  const nameW = $derived(big ? 110 : narrow ? 48 : 70);
  const agentTip = (a: any): Tip => ({
    title: `${a.label ?? a.id ?? "agent"}, ${a.variant ?? ""}`,
    sub: "Share of exact regret by component.",
    rows: PARTS.map((p): [string, string] => [p, `${fmt(a[p] * 100, 1)}%`]),
    terms: terms("regret"),
    source: fileSource(file("regret_attribution.json")),
  });

  function awaitingTip(p: (typeof FIGURE_PANELS)[number]): Tip {
    const f = file(p.file);
    return {
      title: `Panel ${p.letter}: ${p.name}`,
      sub: f?.error ? `The file exists but could not be read: ${f.error}` : `Awaiting ${p.needs.join(", ")}.`,
      body: [p.what, "Nothing is drawn until the study's figure step writes this panel's file from result manifests."],
      rows: p.needs.map((id): [string, string] => [id, word(experimentStatus(id, store.remote, docs).status)]),
      terms: p.letter === "A" ? terms("regimeMap", "tauB") : p.letter === "B" ? terms("gainIdentity", "realizedGain") : terms("regret"),
      source: fileSource(f),
    };
  }
  const anyData = $derived(markers.length > 0 || cells.length > 0 || agents.length > 0);
</script>

{#snippet awaiting(p: (typeof FIGURE_PANELS)[number], w: number, h: number)}
  {@const perRow = Math.max(1, Math.floor((w - 12) / 40))}
  <g class="awaiting" use:tip={() => awaitingTip(p)} tabindex="0" role="group" aria-label="Panel {p.letter}, awaiting {p.needs.join(', ')}">
    <rect x="0.5" y="0.5" width={w - 1} height={h - 1} rx="4" fill="url(#hatch)" />
    <text class="letter" x="8" y="17">{p.letter}</text>
    <text class="pname" x="22" y="17">{p.name}</text>
    <text class="wait" x="8" y={h / 2 - 2}>{file(p.file)?.error ? "unreadable file" : "awaiting"}</text>
    {#each p.needs as id, i}
      {@const st = experimentStatus(id, store.remote, docs).status}
      <g class="need {look(st)}" transform="translate({8 + (i % perRow) * 40} {h / 2 + 5 + Math.floor(i / perRow) * 20})">
        <rect width="35" height="16" rx="3" />
        <Glyph name={glyphFor(st)} size={9} x={3} y={3.5} />
        <text x="14" y="11.5">{id}</text>
      </g>
    {/each}
  </g>
{/snippet}

<section class="figure" class:big>
  <header>
    <h2 class="panel-name">Figure 1 of the paper</h2>
    <span class="grow"></span>
    {#if big}
      <button onclick={() => (store.figureOpen = false)}>Close</button>
    {:else}
      {#if anyData}<button onclick={() => (store.figureOpen = true)} use:tip={{ title: "Enlarge Figure 1", body: "Opens the three panels large. Escape closes it." }}>Enlarge</button>{/if}
      <Age at={store.checked?.docs} staleAfter={10_000} verb={store.compact ? "checked" : "files checked"} what="The three Figure 1 files in tii-cee/results/figure1/" cadence="every 2 seconds" />
    {/if}
  </header>
  <div class="panels">
    <svg width={AW} height={AH} role="img" aria-label="Panel A, the regime map">
      {#if regime && markers.length}
        <text class="letter" x="2" y="12">A</text>
        <text class="pname" x="16" y="12">The regime map</text>
        {#if harmful.length > 2}
          <polygon
            class="harmful"
            points={harmful.map((p) => `${ax(p.tau_b)},${ay(p.kl_inc)}`).join(" ")}
            use:tip={{ title: "Where projection alone is harmful", body: "From E1's exact twin results: readers in this region get worse when only the projection is applied, because their problem is the size of their reactions, not drift.", terms: terms("projection", "ceeScale"), source: fileSource(file("regime_map.json")) }}
          />
        {/if}
        {#each ay.ticks(4) as t}
          <line class="grid" x1={A.l} x2={AW - A.r} y1={ay(t)} y2={ay(t)} />
          <text class="tick num" x={A.l - 4} y={ay(t) + 3.5} text-anchor="end">{t}</text>
        {/each}
        {#each tauTicks as t}
          <line class="tickmark" x1={ax(t)} x2={ax(t)} y1={AH - A.b} y2={AH - A.b + 3} />
          <text class="tick num" x={ax(t)} y={AH - A.b + 13} text-anchor="middle">{t < 1 ? `1/${1 / t}` : t}</text>
        {/each}
        <line class="matched" x1={ax(1)} x2={ax(1)} y1={A.t} y2={AH - A.b} />
        <text class="axis" x={AW - A.r} y={AH - 3} text-anchor="end" use:tip={{ title: "Horizontal axis", terms: terms("tauB", "budget") }}>{AW < 200 ? "tau_B, log; 1 is matched" : "tau_B, log scale; 1 is matched"}</text>
        <text class="axis" x="2" y={A.t - 8} use:tip={{ title: "Vertical axis", terms: terms("klInc") }}>{AW < 200 ? "incoherence, nats" : "incoherence, nats per decision"}</text>
        {#each markers as m (m.id ?? m.label)}
          {@const cx = ax(m.tau_b)}
          {@const cy = ay(m.kl_inc)}
          <g class="marker {m.kind}" use:tip={() => markerTip(m)} tabindex="0">
            {#if num(m.tau_b_lo) != null && num(m.tau_b_hi) != null}<line class="whisker" x1={ax(m.tau_b_lo)} x2={ax(m.tau_b_hi)} y1={cy} y2={cy} />{/if}
            {#if num(m.kl_inc_lo) != null && num(m.kl_inc_hi) != null}<line class="whisker" x1={cx} x2={cx} y1={ay(m.kl_inc_lo)} y2={ay(m.kl_inc_hi)} />{/if}
            {#if Math.abs(arrow(m)) > 2}
              <line class="arrow" x1={cx} x2={cx} y1={cy} y2={cy - arrow(m)} />
              <path class="arrowhead" d="M{cx - 2.5},{cy - arrow(m) + Math.sign(arrow(m)) * 4}L{cx},{cy - arrow(m)}L{cx + 2.5},{cy - arrow(m) + Math.sign(arrow(m)) * 4}" />
            {/if}
            <circle class="hit" {cx} {cy} r="11" />
            {#if m.kind === "trained_policy" || m.kind === "second_backbone"}
              <polygon class="shape" points={star(cx, cy, big ? 8 : 6)} />
            {:else if m.kind === "public_dataset"}
              {@const k = datasets.indexOf(String(m.dataset ?? m.label)) % 3}
              {#if k === 0}<rect class="shape" x={cx - 3.5} y={cy - 3.5} width="7" height="7" />
              {:else if k === 1}<path class="shape" d="M{cx},{cy - 4.5}L{cx + 4.5},{cy + 3.5}L{cx - 4.5},{cy + 3.5}Z" />
              {:else}<path class="shape" d="M{cx},{cy - 4.8}L{cx + 4.8},{cy}L{cx},{cy + 4.8}L{cx - 4.8},{cy}Z" />{/if}
            {:else}
              <circle class="shape" {cx} {cy} r={m.kind === "language_model" ? sizeOf(num(m.params_b) ?? 0) : big ? 4.5 : 3.5} />
            {/if}
            {#if big}<text class="mlabel" x={Math.max(cx + 10, num(m.tau_b_hi) != null ? ax(m.tau_b_hi) + 5 : 0)} y={cy - 4}>{m.label ?? m.id}</text>{/if}
          </g>
        {/each}
      {:else}
        {@render awaiting(FIGURE_PANELS[0], AW, AH)}
      {/if}
    </svg>

    <div class="stack">
      <svg width={BW} height={BH} role="img" aria-label="Panel B, the accounting">
        {#if cells.length}
          <text class="letter" x="2" y="11">B</text>
          <text class="pname" x="16" y="11">The accounting</text>
          <text class="axis" x="2" y="23">realized gain, nats</text>
          <line class="diagonal" x1={gain.x(gain.x.domain()[0])} y1={gain.y(gain.x.domain()[0])} x2={gain.x(gain.x.domain()[1])} y2={gain.y(gain.x.domain()[1])} />
          <line class="grid" x1={B.l} x2={BW - B.r} y1={gain.y(0)} y2={gain.y(0)} />
          <line class="grid" x1={gain.x(0)} x2={gain.x(0)} y1={B.t} y2={BH - B.b} />
          {#each gain.y.ticks(3) as t}<text class="tick num" x={B.l - 4} y={gain.y(t) + 3.5} text-anchor="end">{t}</text>{/each}
          {#each gain.x.ticks(narrow ? 2 : 3) as t}<text class="tick num" x={gain.x(t)} y={BH - B.b + 12} text-anchor="middle">{t}</text>{/each}
          <text class="axis" x={BW - B.r} y={BH - 2} text-anchor="end">{narrow ? "predicted gain, nats" : "gain predicted by the identity, nats"}</text>
          {#each cells as c (c.id ?? `${c.reader}-${c.setting}`)}
            <g class="marker cell" use:tip={() => cellTip(c)} tabindex="0">
              {#if num(c.predicted_lo) != null && num(c.predicted_hi) != null}<line class="whisker" x1={gain.x(c.predicted_lo)} x2={gain.x(c.predicted_hi)} y1={gain.y(c.realized_gain)} y2={gain.y(c.realized_gain)} />{/if}
              {#if num(c.realized_lo) != null && num(c.realized_hi) != null}<line class="whisker" x1={gain.x(c.predicted_gain)} x2={gain.x(c.predicted_gain)} y1={gain.y(c.realized_lo)} y2={gain.y(c.realized_hi)} />{/if}
              <circle class="hit" cx={gain.x(c.predicted_gain)} cy={gain.y(c.realized_gain)} r="10" />
              <circle class="shape" cx={gain.x(c.predicted_gain)} cy={gain.y(c.realized_gain)} r={big ? 4 : 3.2} />
            </g>
          {/each}
        {:else}
          {@render awaiting(FIGURE_PANELS[1], BW, BH)}
        {/if}
      </svg>

      <svg width={BW} height={BH} role="img" aria-label="Panel C, the attribution">
        {#if agents.length}
          {@const keyX = narrow ? [2, 2, 2] : [2, 58, 106]}
          {@const keyY = narrow ? [17, 26, 35] : [18, 18, 18]}
          {@const top = narrow ? 47 : 32}
          <text class="letter" x="2" y="11">C</text>
          <text class="pname" x="16" y="11">The attribution</text>
          {#each PARTS as p, i}
            <rect class="part p{i}" x={keyX[i]} y={keyY[i]} width="8" height="6" rx="1.5" />
            <text class="tick" x={keyX[i] + 11} y={keyY[i] + 6}>{p}</text>
          {/each}
          {#each agents as a, row (`${a.id}-${a.variant}`)}
            {@const y = top + row * ((BH - top - 2) / agents.length)}
            {@const bar = Math.min(14, (BH - top - 2) / agents.length - 3)}
            {@const total = a.selection + a.reading + a.commitment || 1}
            <g use:tip={() => agentTip(a)} tabindex="0">
              <text class="tick" x="2" y={y + bar - 2}>{String(a.label ?? a.id).slice(0, big ? 16 : narrow ? 5 : 6)} {a.variant === "repaired" ? "rep." : "raw"}</text>
              {#each PARTS as p, i}
                {@const x0 = nameW + (PARTS.slice(0, i).reduce((s, q) => s + a[q], 0) / total) * (BW - nameW - 2)}
                <rect class="part p{i}" x={x0} {y} width={Math.max(0, (a[p] / total) * (BW - nameW - 2) - 2)} height={bar} rx="2" />
              {/each}
            </g>
          {/each}
        {:else}
          {@render awaiting(FIGURE_PANELS[2], BW, BH)}
        {/if}
      </svg>
    </div>
  </div>
</section>

<style>
  .figure {
    border-left: 1px solid var(--line);
    padding: 8px 10px 6px 12px;
    min-width: 0;
  }
  .figure.big {
    border: 0;
    padding: 6px;
  }
  header {
    display: flex;
    align-items: center;
    gap: 8px;
    min-height: 20px;
    padding-bottom: 6px;
    white-space: nowrap;
  }
  .grow {
    flex: 1;
  }
  button {
    font-size: var(--fs-1);
    background: var(--raised);
    border: 1px solid var(--line-2);
    border-radius: 5px;
    padding: 1px 8px;
    color: var(--ink);
    cursor: pointer;
  }
  .panels {
    display: flex;
    gap: 6px;
  }
  .big .panels {
    gap: 18px;
  }
  .stack {
    display: grid;
    gap: 4px;
  }
  .letter {
    font-size: 12px;
    font-weight: 700;
    fill: var(--ink);
  }
  .pname {
    font-size: 11px;
    font-weight: 600;
    fill: var(--ink-2);
  }
  .awaiting > rect {
    stroke: var(--line-2);
  }
  .wait {
    font-size: 11px;
    fill: var(--ink-2);
  }
  .need rect {
    fill: var(--ground);
    stroke: var(--ink-3);
  }
  .need text {
    font-size: 10.5px;
    font-weight: 600;
    fill: var(--ink-2);
  }
  .need {
    color: var(--ink-3);
  }
  .need.running rect {
    stroke: var(--live);
  }
  .need.running {
    color: var(--live-text);
  }
  .need.done rect {
    fill: var(--done);
    stroke: var(--done);
  }
  .need.done text {
    fill: var(--ink);
  }
  .need.done {
    color: var(--ink);
  }
  .need.wait rect {
    stroke: var(--wait);
  }
  .need.wait {
    color: var(--wait-text);
  }
  .need.fail rect {
    stroke: var(--fail);
  }
  .grid {
    stroke: var(--line);
  }
  .tickmark {
    stroke: var(--line-2);
  }
  .tick,
  .axis,
  .mlabel {
    font-size: 9.5px;
    fill: var(--ink-3);
  }
  .mlabel {
    fill: var(--ink-2);
    pointer-events: none;
  }
  .matched,
  .diagonal {
    stroke: var(--ink-3);
    stroke-dasharray: 3 3;
  }
  .harmful {
    fill: color-mix(in srgb, var(--ink-3) 22%, transparent);
    stroke: none;
  }
  .hit {
    fill: transparent;
  }
  .whisker {
    stroke: var(--ink-3);
    stroke-width: 1;
  }
  .arrow,
  .arrowhead {
    stroke: var(--ink);
    stroke-width: 1.2;
    fill: none;
  }
  .shape {
    fill: var(--ink);
    stroke: var(--ground);
    stroke-width: 1.5;
    paint-order: stroke;
  }
  .marker.simulated .shape,
  .marker.second_backbone .shape {
    fill: var(--ground);
    stroke: var(--ink);
    stroke-width: 1.3;
    paint-order: normal;
  }
  .marker:hover .shape,
  .marker:focus-visible .shape {
    stroke: var(--live);
    stroke-width: 2;
    paint-order: normal;
  }
  .part.p0 {
    fill: var(--ink);
  }
  .part.p1 {
    fill: var(--ink-3);
  }
  .part.p2 {
    fill: var(--done);
  }
</style>
