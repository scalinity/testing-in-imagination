<script lang="ts">
  // Everything that lives on this Mac: the launchd jobs, the AI workers, and git.
  import Age from "../lib/Age.svelte";
  import Glyph from "../lib/Glyph.svelte";
  import { checkedAgo } from "../lib/sources";
  import { store } from "../lib/store.svelte";
  import { terms } from "../lib/study";
  import { span } from "../lib/time";
  import { tip, type Tip } from "../lib/tip.svelte";
  import type { LaunchdJob } from "../lib/types";
  import GitGrowth from "./GitGrowth.svelte";
  import Workers from "./Workers.svelte";

  const jobs = $derived(store.link?.launchd ?? []);

  const ABOUT: Record<string, { name: string; what: string; term?: Parameters<typeof terms>[0] }> = {
    "com.tii.hpg-keeper": { name: "keeper", what: "Holds the signed-in connection to HiPerGator open.", term: "keeper" },
    "com.tii.caffeinate": { name: "caffeinate", what: "Keeps this Mac from sleeping, so the connection and the loops stay alive." },
    "com.tii.local-loop": { name: "local loop", what: "Starts an agent session when the cluster finishes a stage or decides a gate.", term: "localLoop" },
  };

  type JobLook = "running" | "waiting" | "failed" | "absent";
  const jobLook = (j: LaunchdJob): JobLook => (!j.installed ? "absent" : j.state === "running" ? "running" : j.last_exit != null && j.last_exit !== 0 ? "failed" : "waiting");
  const WORD: Record<JobLook, string> = { running: "running", waiting: "waits for its next run", failed: "last run failed", absent: "not installed" };

  const jobTip = (j: LaunchdJob): Tip => {
    const about = ABOUT[j.label];
    const l = jobLook(j);
    return {
      title: `${about?.name ?? j.label}: ${WORD[l]}`,
      body: [
        about?.what ?? "",
        l === "waiting" ? "launchd starts it on a timer, so between runs it reports \"not running\". That is normal." : l === "absent" ? "launchd has no job with this label. The study's agent installs it during M0." : "",
      ].filter(Boolean),
      rows: [
        ["Label", j.label],
        ["launchd state", j.installed ? j.state || "not stated" : j.note || "not installed"],
        ...(j.pid != null ? ([["Process", `pid ${j.pid}`]] as [string, string][]) : []),
        ...(j.interval_s != null ? ([["Runs every", span(j.interval_s * 1000)]] as [string, string][]) : []),
        ...(j.runs != null ? ([["Runs so far", String(j.runs)]] as [string, string][]) : []),
        ...(j.last_exit != null ? ([["Last exit code", `${j.last_exit}${j.last_exit === 0 ? " (success)" : " (failure)"}`]] as [string, string][]) : []),
      ],
      terms: about?.term ? terms(about.term) : undefined,
      source: `launchctl print, ${checkedAgo(store.checked?.launchd)}.`,
    };
  };
</script>

<section class="mac">
  <header class="top">
    <h2 class="region-name">This Mac</h2>
    <ul class="jobs">
      {#each jobs as j (j.label)}
        {@const l = jobLook(j)}
        <li class={l} use:tip={() => jobTip(j)} tabindex="0">
          <span class:breathe={l === "running"}><Glyph name={l === "running" ? "dot" : l === "waiting" ? "check" : l === "failed" ? "cross" : "ring"} size={10} /></span>
          {ABOUT[j.label]?.name ?? j.label}
        </li>
      {/each}
    </ul>
    <span class="grow"></span>
    <Age at={store.checked?.launchd} staleAfter={60_000} verb={store.compact ? "checked" : "launch agents checked"} what="The three launchd jobs" cadence="every 20 seconds" />
  </header>
  <Workers />
  <GitGrowth />
</section>

<style>
  .mac {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    padding: 9px 4px 6px 4px;
  }
  .top {
    display: flex;
    align-items: baseline;
    gap: 14px;
    padding: 0 10px 6px;
  }
  .grow {
    flex: 1;
  }
  .jobs {
    display: flex;
    gap: 12px;
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: var(--fs-2);
    color: var(--ink-2);
  }
  .jobs li {
    display: flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
  }
  .jobs li.running :global(.glyph) {
    color: var(--live-text);
  }
  .jobs li.failed {
    color: var(--fail-text);
  }
  .jobs li.absent {
    color: var(--ink-3);
  }
  .jobs span {
    display: inline-flex;
  }
</style>
