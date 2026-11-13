<script lang="ts">
  // Everything that lives on HiPerGator, as the watcher's files describe it.
  import Age from "../lib/Age.svelte";
  import Glyph from "../lib/Glyph.svelte";
  import { budgets, clusterWork } from "../lib/derive";
  import { clusterSource, fileSource } from "../lib/sources";
  import { glyphFor, look, word } from "../lib/status";
  import { store } from "../lib/store.svelte";
  import { EXPERIMENT_NAME, terms } from "../lib/study";
  import { clock, dayClock, fmt, parseTime, span, zone } from "../lib/time";
  import { tip, type Tip } from "../lib/tip.svelte";
  import ClusterTimeline from "./ClusterTimeline.svelte";

  const remote = $derived(store.remote);
  const meta = $derived(store.remoteMeta);
  const docs = $derived(store.docs);
  const now = $derived(store.now);
  const status = $derived(remote?.status ?? null);
  const work = $derived(clusterWork(remote));

  const pending = $derived(!remote || remote.phase === "never_read" || remote.phase === "");
  const notStarted = $derived(!pending && (remote?.phase === "not_started" || (remote?.phase === "ok" && !status)));
  const failed = $derived(remote?.phase === "error");
  const intervalMs = $derived((meta?.interval_s ?? 120) * 1000);
  const readStale = $derived(remote?.last_ok_ms != null && now - remote.last_ok_ms > intervalMs * 2.5);

  // Watcher heartbeat: time since it last wrote status.json, against its own interval.
  const generated = $derived(parseTime(status?.generated_at));
  const cycleMs = $derived((status?.watcher?.interval_s ?? 900) * 1000);
  const beatAge = $derived(generated == null ? null : Math.max(0, now - generated));
  const watcherDone = $derived(Boolean(status?.watcher?.completed));
  const beatLate = $derived(beatAge != null && !watcherDone && beatAge > cycleMs + 5 * 60e3);

  const auth = $derived(status?.authorization);
  const authOk = $derived(Boolean(auth?.ok));

  const limits = $derived.by(() => {
    const fromStatus = status?.gpu?.limit_by_account;
    if (fromStatus && Object.keys(fromStatus).length) {
      // Keep the addendum's order: the study's own account first.
      const order = (docs?.gpu_limits ?? []).map((l) => l.account);
      const rank = (a: string) => (order.includes(a) ? order.indexOf(a) : order.length);
      return Object.entries(fromStatus)
        .sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b))
        .map(([account, gpus]) => ({ account, gpus, source: "state/status.json on HiPerGator" }));
    }
    return (docs?.gpu_limits ?? []).map((l) => ({ ...l, source: "AUTHORIZATION_ADDENDUM_1.md section 2" }));
  });
  const inUse = (account: string) => status?.gpu?.in_use_by_account?.[account] ?? null;
  const runningOn = (account: string) => work.rows.filter((r) => r.status === "RUNNING" && r.attempts.some((a) => a.ended == null && a.account === account)).flatMap((r) => Array.from({ length: r.stage?.gpus ?? 1 }, () => r.id));

  const budgetRows = $derived(budgets(remote, docs));
  const maxAllowance = $derived(Math.max(1, ...budgetRows.map((b) => b.allowance ?? 0)));
  const totals = $derived({
    spent: budgetRows.some((b) => b.spent != null) ? budgetRows.reduce((s, b) => s + (b.spent ?? 0), 0) : null,
    estimate: budgetRows.reduce((s, b) => s + (b.estimate ?? 0), 0),
    allowance: budgetRows.reduce((s, b) => s + (b.allowance ?? 0), 0),
  });

  // What needs attention first, then what is alive, then the rest.
  const COUNT_ORDER = ["BLOCKED", "FAILED", "REFUSED", "INTERRUPTED", "RUNNING", "QUEUED", "COMPLETED", "PLANNED", "SKIPPED BY GATE"];
  const counts = $derived(
    Object.entries(status?.counts ?? {})
      .filter(([, n]) => n > 0)
      .sort(([a], [b]) => (COUNT_ORDER.indexOf(a) + 99) % 99 - (COUNT_ORDER.indexOf(b) + 99) % 99),
  );
  const budgetName = (id: string) => (/^E\d+$/.test(id) ? id : "bench");

  const progress = $derived(
    Object.values(remote?.progress ?? {})
      .map((f) => ({ file: f, json: f.json ?? {} }))
      .filter((p) => typeof p.json.done === "number" && typeof p.json.total === "number" && p.json.total > 0)
      .map((p) => {
        const stage = String(p.json.stage ?? p.file.name.replace(/\.json$/, ""));
        const row = work.rows.find((r) => r.id === stage);
        return { stage, unit: String(p.json.unit ?? "item"), done: p.json.done as number, total: p.json.total as number, updated: parseTime(p.json.updated_at) ?? p.file.mtime_ms, status: row?.status ?? "", experiment: row?.experiment ?? "" };
      })
      .sort((a, b) => Number(b.status === "RUNNING") - Number(a.status === "RUNNING") || b.updated - a.updated),
  );

  const events = $derived((remote?.events ?? []).slice(-6).reverse());

  const waitWord = $derived.by(() => {
    if (!meta) return "";
    if (meta.reading) return "reading now";
    if (remote?.skipped) return "waiting for the SSH master";
    return meta.next_read_in_s == null ? "" : meta.next_read_in_s === 0 ? "next read due" : `next read in ${span(meta.next_read_in_s * 1000)}`;
  });

  const readTip = (): Tip => ({
    title: "How the cluster is read",
    body: [
      "Only through Daniel's existing signed-in connection, and only with cat, tail and ls on the watcher's files. No scheduler command is ever sent, and no other path on HiPerGator is ever touched. If the connection is down, nothing is sent.",
      remote?.calls.length ? "The last read ran exactly these commands:" : "No command has been sent in this session yet.",
    ],
    rows: (remote?.calls ?? []).map((c): [string, string] => [c.exit === 0 ? "ok" : `exit ${c.exit ?? "none"}`, `${c.command.replace(meta?.root ?? "", "$TII")} (${c.bytes} bytes, ${c.ms} ms)${c.note ? ` ${c.note}` : ""}`]),
    source: `Host ${meta?.host ?? ""}. $TII is ${meta?.root ?? ""}.`,
  });

  const budgetTip = (b: (typeof budgetRows)[number]): Tip => ({
    title: `${b.label}: GPU-hours`,
    sub: b.spent == null ? "Nothing reported as spent yet." : `${fmt(b.spent, 1)} spent, ${fmt(((b.spent ?? 0) / (b.allowance ?? 1)) * 100)}% of the allowance.`,
    rows: [
      ["Spent", b.spent == null ? "none reported" : `${fmt(b.spent, 1)} GPU-hours, failed jobs and retries included`],
      ["Estimate", `${fmt(b.estimate ?? 0)} GPU-hours${docs?.gpu_estimates.find((g) => g.label === b.id)?.basis ? `. Basis: ${docs.gpu_estimates.find((g) => g.label === b.id)?.basis}` : ""}`],
      ["Allowance", `${fmt(b.allowance ?? 0)} GPU-hours`],
    ],
    terms: terms("estimate", "allowance"),
    source: `Spent: ${b.spentSource === "no spending reported yet" ? "no spending reported yet" : clusterSource()} Limits: ${b.limitSource}.`,
  });
</script>

<section class="cluster" class:compact={store.compact} class:stale={readStale || Boolean(remote?.skipped) || failed}>
  <header class="top">
    <h2 class="region-name">HiPerGator</h2>
    <Age at={remote?.last_ok_ms} staleAfter={intervalMs * 2.5} verb="read" what="The watcher's files on HiPerGator" cadence={`every ${meta?.interval_s ?? 120} seconds`} />
    <span class="muted next num" use:tip={readTip}>{waitWord}</span>
    <span class="grow"></span>
    <label class="every" use:tip={{ title: "How often to read HiPerGator", body: "Each read is a handful of cat, tail and ls calls through the existing connection. The floor is 60 seconds and cannot be lowered. The watcher itself only rewrites its files every 15 minutes." }}>
      every
      <select value={String(meta?.interval_s ?? 120)} onchange={(e) => store.setRemoteInterval(Number((e.currentTarget as HTMLSelectElement).value))}>
        {#each [60, 120, 300, 600, 900] as s}<option value={String(s)}>{s < 120 ? `${s} s` : `${s / 60} min`}</option>{/each}
      </select>
    </label>
    <button class="readnow" disabled={!meta || meta.reading || meta.manual_wait_s > 0} onclick={() => store.readClusterNow()} use:tip={readTip}>
      {meta?.reading ? "Reading…" : meta && meta.manual_wait_s > 0 ? `Read again in ${meta.manual_wait_s} s` : "Read now"}
    </button>
  </header>

  {#if remote?.skipped}
    <p class="notice wait"><Glyph name="pause" /> <strong>Connection down.</strong> {remote.skipped} {remote.last_ok_ms ? `Below is the last known state, read ${dayClock(remote.last_ok_ms)} (${span(now - remote.last_ok_ms)} ago).` : "Nothing has been read yet."}</p>
  {:else if failed}
    <p class="notice fail"><Glyph name="cross" /> <strong>The last read failed.</strong> {remote?.detail} {remote?.last_ok_ms ? `Below is the last known state, from ${dayClock(remote.last_ok_ms)}.` : ""}</p>
  {:else if pending}
    <p class="notice"><Glyph name="clock" /> <strong>First read pending.</strong> The monitor reads HiPerGator once the SSH master answers and at most every {meta?.min_interval_s ?? 60} seconds.</p>
  {:else if notStarted}
    <p class="notice">
      <Glyph name="ring" /> <strong>The watcher has not started yet.</strong>
      {remote?.phase === "not_started" ? "Its state folder does not exist on HiPerGator." : "Its state folder exists on HiPerGator, but holds no status.json."}
      {store.compact ? "" : "Jobs, GPU use and gates fill in after its first cycle."} Hollow marks below are limits and plans from the repository, not observations.
    </p>
  {:else}
    <div class="strip">
      <div
        class="beat"
        class:late={beatLate}
        use:tip={() => ({
          title: beatLate ? "The watcher is late" : watcherDone ? "The watcher finished" : "Watcher heartbeat",
          sub: generated == null ? "status.json carries no time." : `Cycle ${status?.watcher?.cycle ?? "?"} wrote status.json at ${dayClock(generated)}, ${span(now - generated)} ago.`,
          body: watcherDone
            ? "The watcher reported the study as complete and disabled its own schedule."
            : `The bar fills over one ${span(cycleMs)} cycle and should reset when the next cycle writes. Past the end, plus 5 minutes of grace, the watcher is late: the cluster keeps running jobs, but nothing new is submitted or resubmitted until it cycles.`,
          rows: [["Lock", status?.watcher?.lock ?? "not stated"], ["Cycle started", parseTime(status?.watcher?.started_at) ? clock(parseTime(status?.watcher?.started_at)!) : "not stated"], ["Cycle finished", parseTime(status?.watcher?.finished_at) ? clock(parseTime(status?.watcher?.finished_at)!) : "not stated"]],
          terms: terms("watcher"),
          source: clusterSource(),
        })}
      >
        <span class="lab">{#if beatLate}<Glyph name="clock" />{/if} watcher</span>
        <span class="meter"><i style:width="{Math.min(100, ((beatAge ?? 0) / cycleMs) * 100)}%"></i></span>
        <span class="num val">{watcherDone ? "finished" : beatAge == null ? "no time" : `cycle ${status?.watcher?.cycle ?? "?"}, ${span(beatAge)} ago`}</span>
      </div>

      <div
        class="auth"
        class:bad={!authOk}
        use:tip={() => ({
          title: authOk ? "Authorization holds" : "Authorization check failed",
          body: authOk
            ? "Every cycle the watcher recomputes the SHA-256 of AUTHORIZATION.md, compares it with the signed value, and checks the expiry date."
            : "The watcher's check of AUTHORIZATION.md failed. It submits nothing new that needs authorization and marks those stages BLOCKED; jobs already running continue.",
          rows: [
            ["Hash matches", auth?.sha256_ok == null ? "not stated" : auth.sha256_ok ? "yes" : "no"],
            ["Signature recorded", auth?.signature_ok == null ? "not stated" : auth.signature_ok ? "yes" : "no"],
            ["Valid until", auth?.expires ?? "not stated"],
            ...(auth?.addenda ?? []).map((a): [string, string] => [a.name ?? "addendum", `${a.sha256_ok ? "hash matches" : "hash does not match"}, ${a.signed ? "signed" : "not signed"}`]),
          ],
          source: clusterSource(),
        })}
      >
        <Glyph name={authOk ? "check" : "cross"} /> authorization {authOk ? `valid to ${auth?.expires ?? "?"}` : "failed"}
      </div>

      <ul class="counts" use:tip={{ title: "Stages by status", body: "How many stages are in each state, as the watcher counted them in its last cycle.", terms: terms("stage"), source: clusterSource() }}>
        {#each counts as [name, n] (name)}
          <li class={look(name)}><Glyph name={glyphFor(name)} size={10} /><b class="num">{n}</b> {word(name)}</li>
        {/each}
      </ul>
    </div>
    {#if remote?.status_note}<p class="notice"><Glyph name="clock" /> {remote.status_note}</p>{/if}
  {/if}

  <div class="body">
    <ClusterTimeline {work} {limits} started={!pending && !notStarted} />

    <div class="side scroll">
      <section>
        <h3 class="panel-name">GPUs now</h3>
        {#each limits as l (l.account)}
          {@const used = inUse(l.account)}
          {@const stages = runningOn(l.account)}
          <div class="rack">
            <span class="code acct">{l.account}</span>
            <span class="slots">
              {#each Array.from({ length: l.gpus }) as _, i}
                {@const on = used != null && i < used}
                <i
                  class:on
                  class:breathe={on}
                  use:tip={() => ({
                    title: on ? `GPU in use on ${l.account}` : `Free slot on ${l.account}`,
                    sub: on ? (stages[i] ? `Held by stage ${stages[i]}.` : "The watcher counts it in use.") : used == null ? "The watcher has not reported GPU use yet." : "Not used by this study.",
                    body: `${l.account} may hold ${l.gpus} GPUs at once. That is the class QOS limit, shared with other students, so fewer may be free. Every job uses one RTX PRO 6000.`,
                    source: used == null ? `Limit from ${l.source}.` : `${clusterSource()} Limit from ${l.source}.`,
                  })}
                ></i>
              {/each}
            </span>
            <span class="num used">{used == null ? "–" : used} of {l.gpus}</span>
          </div>
        {:else}
          <p class="empty">{docs ? "No GPU limit is stated in status.json or in the addendum." : "Waiting for the study's documents."}</p>
        {/each}
      </section>

      <section>
        <h3 class="panel-name withkey">GPU-hours <span class="legendline muted"><i class="lg spent"></i>spent <i class="lg est"></i>estimate <i class="lg cap"></i>allowance</span></h3>
        {#each budgetRows as b (b.label)}
          {@const frac = (b.allowance ?? 0) / maxAllowance}
          {@const over = b.spent != null && b.allowance != null && b.spent > b.allowance}
          <div class="budget" class:over use:tip={() => budgetTip(b)} tabindex="0">
            <span class="bid">{budgetName(b.id)}</span>
            <span class="track" style:width="{frac * 100}%">
              <i class="spent" style:width="{Math.min(100, ((b.spent ?? 0) / (b.allowance ?? 1)) * 100)}%"></i>
              {#if b.estimate && b.allowance}<i class="est" style:left="{(b.estimate / b.allowance) * 100}%"></i>{/if}
            </span>
            <span class="num bval">{b.spent == null ? "–" : fmt(b.spent, b.spent < 100 ? 1 : 0)}<span class="muted of">of {fmt(b.estimate ?? 0)}</span></span>
          </div>
        {/each}
        {#if budgetRows.length}
          <p class="total num" use:tip={{ title: "All experiments", body: "Sum over the rows above. The allowance is about twice the estimate.", terms: terms("estimate", "allowance"), source: fileSource(docs?.authorization, "section 4") }}>
            total {totals.spent == null ? "–" : fmt(totals.spent, 1)} spent, {fmt(totals.estimate)} estimated, {fmt(totals.allowance)} allowed
          </p>
        {:else}
          <p class="empty">{docs ? "AUTHORIZATION.md section 4 could not be read." : "Waiting for the study's documents."}</p>
        {/if}
      </section>

      <section>
        <h3 class="panel-name">Stage progress</h3>
        {#each progress as p (p.stage)}
          <div
            class="prog"
            use:tip={() => ({
              title: `Stage ${p.stage}`,
              sub: `${fmt(p.done)} of ${fmt(p.total)} ${p.unit}s done, ${fmt((p.done / p.total) * 100, 1)}%.`,
              body: "A running stage writes its own progress at most once a minute.",
              rows: [["Stage reported", `${dayClock(p.updated)} (${span(now - p.updated)} ago)`], ["Status", p.status ? p.status.toLowerCase() : "not reported"]],
              source: clusterSource(`state/progress/${p.stage}.json`),
            })}
          >
            <span class="code pstage">{p.stage}</span>
            <span class="meter" class:live={p.status === "RUNNING"}><i style:width="{Math.min(100, (p.done / p.total) * 100)}%"></i></span>
            <span class="num pval">{fmt(p.done)}<span class="muted of">of {fmt(p.total)}</span></span>
          </div>
        {:else}
          <p class="empty">{notStarted || pending ? "Not started yet." : "No running stage has written progress."}</p>
        {/each}
      </section>

      <section>
        <h3 class="panel-name">Latest events, {zone(now)}</h3>
        {#each events as e}
          <p class="event" use:tip={{ title: e.kind, sub: parseTime(e.time) ? dayClock(parseTime(e.time)!) : e.time, body: e.text, source: clusterSource("state/events.tsv") }}>
            <span class="num etime">{parseTime(e.time) ? clock(parseTime(e.time)!).replace(/ [A-Z]+$/, "") : "?"}</span>
            <span class="ekind">{e.kind}</span>
            <span class="etext">{e.text}</span>
          </p>
        {:else}
          <p class="empty">{notStarted || pending ? "Not started yet." : "The watcher has queued no event."}</p>
        {/each}
        {#if remote?.exports_listed}
          <p class="muted exports" use:tip={{ title: "Approved exports", body: "Aggregates that passed the export guard on HiPerGator. The monitor lists them and never reads anything else from the study's storage.", rows: (remote?.exports ?? []).slice(0, 12).map((x): [string, string] => [x.path, `${fmt(x.size)} bytes`]), source: clusterSource("export/approved/") }}>
            {remote.exports.length} approved export {remote.exports.length === 1 ? "file" : "files"}
          </p>
        {/if}
      </section>
    </div>
  </div>
</section>

<style>
  .cluster {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    padding: 9px 4px 6px 4px;
  }
  .top {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 0 10px 6px;
  }
  .grow {
    flex: 1;
  }
  .next {
    font-size: var(--fs-1);
  }
  .every {
    font-size: var(--fs-1);
    color: var(--ink-3);
  }
  select,
  .readnow {
    font-size: var(--fs-1);
    background: var(--raised);
    border: 1px solid var(--line-2);
    border-radius: 5px;
    padding: 2px 8px;
    color: var(--ink-2);
  }
  .readnow {
    cursor: pointer;
    color: var(--ink);
    font-weight: 600;
  }
  .readnow:disabled {
    color: var(--ink-3);
    font-weight: 400;
    cursor: default;
  }
  .notice {
    margin: 0 10px 8px;
    padding: 7px 10px;
    border: 1px solid var(--line-2);
    border-radius: 5px;
    font-size: var(--fs-2);
    color: var(--ink-2);
    line-height: 1.45;
  }
  .notice strong {
    color: var(--ink);
    font-weight: 600;
  }
  .notice.wait {
    border-color: var(--wait);
    background: color-mix(in srgb, var(--wait) 10%, transparent);
  }
  .notice.wait :global(.glyph),
  .notice.wait strong {
    color: var(--wait-text);
  }
  .notice.fail {
    border-color: var(--fail);
    background: color-mix(in srgb, var(--fail) 12%, transparent);
  }
  .notice.fail :global(.glyph),
  .notice.fail strong {
    color: var(--fail-text);
  }
  .strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 20px;
    padding: 0 10px 8px;
    font-size: var(--fs-2);
    color: var(--ink-2);
  }
  .beat {
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .beat .lab {
    font-weight: 600;
    color: var(--ink);
  }
  .meter {
    position: relative;
    display: block;
    width: 84px;
    height: 6px;
    border: 1px solid var(--ink-3);
    border-radius: 3px;
  }
  .meter i {
    position: absolute;
    inset: 0 auto 0 0;
    background: var(--ink-2);
    border-radius: 2px;
  }
  .beat.late {
    color: var(--wait-text);
  }
  .beat.late .lab {
    color: var(--wait-text);
  }
  .beat.late .meter {
    border-color: var(--wait);
  }
  .beat.late .meter i {
    background: var(--wait);
  }
  .auth.bad {
    color: var(--fail-text);
    font-weight: 600;
  }
  .counts {
    display: flex;
    flex-wrap: wrap;
    gap: 2px 13px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .counts li {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--ink-3);
  }
  .counts b {
    color: var(--ink);
    font-weight: 600;
  }
  .counts li.running :global(.glyph) {
    color: var(--live-text);
  }
  .counts li.fail :global(.glyph),
  .counts li.fail b {
    color: var(--fail-text);
  }
  .counts li.wait :global(.glyph),
  .counts li.wait b {
    color: var(--wait-text);
  }
  .counts li.retry :global(.glyph) {
    color: var(--retry-text);
  }
  .body {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 268px;
    gap: 4px;
  }
  .stale .body {
    opacity: 0.6;
  }
  .side {
    padding: 0 10px 0 10px;
    border-left: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
  }
  .side section {
    display: grid;
    gap: 4px;
  }
  .rack {
    display: grid;
    grid-template-columns: 54px 1fr auto;
    align-items: center;
    gap: 6px;
  }
  .acct {
    color: var(--ink-2);
    font-size: 10.5px;
  }
  .slots {
    display: flex;
    gap: 3px;
  }
  .slots i {
    width: 13px;
    height: 13px;
    border: 1px solid var(--ink-3);
    border-radius: 3px;
  }
  .slots i.on {
    background: var(--live);
    border-color: var(--live);
  }
  .used {
    font-size: var(--fs-1);
    color: var(--ink-2);
  }
  .withkey {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .legendline {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: var(--fs-1);
    font-weight: 400;
  }
  .compact .body {
    grid-template-columns: minmax(0, 1fr) 232px;
  }
  .compact .side {
    padding: 0 6px 0 8px;
    gap: 9px;
  }
  .compact .top {
    gap: 9px;
    padding: 0 6px 4px;
  }
  .compact .strip {
    gap: 3px 14px;
    padding: 0 6px 6px;
  }
  .compact.cluster {
    padding-top: 6px;
  }
  .compact .notice {
    margin: 0 6px 6px;
    padding: 5px 8px;
  }
  .compact .legendline {
    font-size: 10px;
    gap: 3px;
  }
  .compact .lg {
    margin-left: 3px;
  }
  .compact .budget {
    grid-template-columns: 30px 1fr 70px;
  }
  .compact .rack {
    grid-template-columns: 50px 1fr auto;
  }
  .compact .slots i {
    width: 12px;
    height: 12px;
  }
  .lg {
    display: inline-block;
    margin-left: 6px;
  }
  .lg:first-child {
    margin-left: 0;
  }
  .lg.spent {
    width: 12px;
    height: 6px;
    background: var(--ink-2);
    border-radius: 2px;
  }
  .lg.est {
    width: 2px;
    height: 10px;
    background: var(--ink);
  }
  .lg.cap {
    width: 12px;
    height: 6px;
    border: 1px solid var(--ink-3);
    border-radius: 2px;
  }
  .budget {
    display: grid;
    grid-template-columns: 34px 1fr 78px;
    align-items: center;
    gap: 6px;
    font-size: var(--fs-1);
  }
  .bid {
    font-weight: 600;
    color: var(--ink);
  }
  .track {
    position: relative;
    height: 7px;
    border: 1px solid var(--ink-3);
    border-radius: 3px;
    min-width: 6px;
  }
  .track .spent {
    position: absolute;
    inset: 0 auto 0 0;
    background: var(--ink-2);
    border-radius: 2px;
  }
  .track .est {
    position: absolute;
    top: -3px;
    bottom: -3px;
    width: 2px;
    margin-left: -1px;
    background: var(--ink);
  }
  .budget.over .track {
    border-color: var(--fail);
  }
  .budget.over .spent {
    background: var(--fail);
  }
  .bval,
  .pval {
    text-align: right;
    color: var(--ink);
    white-space: nowrap;
  }
  .of {
    margin-left: 4px;
  }
  .total {
    font-size: var(--fs-1);
    color: var(--ink-3);
    padding-top: 2px;
  }
  .prog {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 62px 78px;
    align-items: center;
    gap: 6px;
    font-size: var(--fs-1);
  }
  .pstage {
    color: var(--ink-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 10.5px;
  }
  .prog .meter {
    width: auto;
  }
  .prog .meter.live i {
    background: var(--live);
  }
  .event {
    display: grid;
    grid-template-columns: 44px auto minmax(0, 1fr);
    gap: 6px;
    font-size: var(--fs-1);
    color: var(--ink-2);
    white-space: nowrap;
  }
  .etime {
    color: var(--ink-3);
  }
  .ekind {
    color: var(--ink);
    font-weight: 600;
  }
  .etext {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .exports {
    font-size: var(--fs-1);
  }
</style>
