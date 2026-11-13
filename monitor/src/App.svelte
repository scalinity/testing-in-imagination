<script lang="ts">
  import Tooltip from "./lib/Tooltip.svelte";
  import { store } from "./lib/store.svelte";
  import BlockedRail from "./panels/BlockedRail.svelte";
  import Cluster from "./panels/Cluster.svelte";
  import Figure1 from "./panels/Figure1.svelte";
  import Mac from "./panels/Mac.svelte";
  import PlanBoard from "./panels/PlanBoard.svelte";
  import Seam from "./panels/Seam.svelte";
  import TitleBar from "./panels/TitleBar.svelte";

  // The repository lives under ~/Documents. The first time the app reads it, macOS asks for
  // permission and the read waits for the answer; a refusal comes back as "not permitted".
  const waitingForRepo = $derived(store.ready && !store.demo && store.checked?.docs == null && store.uptime > 4000);
  const refused = $derived((store.docs?.problems ?? []).some((p) => /not permitted|denied/i.test(p)));
</script>

<!-- Shared fill for "no data": drawn wherever a source is missing or a log has a gap. -->
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <defs>
    <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="var(--hatch)" stroke-width="2.5" />
    </pattern>
  </defs>
</svg>

<svelte:window onkeydown={(e) => e.key === "Escape" && (store.figureOpen = false)} />

<div class="app" class:demo={store.demo} class:compact={store.compact}>
  <TitleBar />
  {#if store.demo}
    <div class="demo-banner" role="alert">
      <span class="stripes" aria-hidden="true"></span>
      <strong>DEMO</strong>
      <span>Fictional data from monitor/fixtures. Nothing on this screen is real, and its clock is not the real time.</span>
      <span class="stripes" aria-hidden="true"></span>
    </div>
  {/if}
  {#if !store.ready}
    <p class="starting">{store.pollError ? `Cannot reach the monitor's backend: ${store.pollError}` : "Reading sources…"}</p>
  {:else}
    {#if waitingForRepo || refused}
      <p class="permission" role="alert">
        {#if refused}
          <strong>macOS refused this app access to the study's folder.</strong> Allow it in System Settings, Privacy and Security, Files and Folders, TII Monitor, Documents Folder. Until then the plan, the documents and git cannot be read.
        {:else}
          <strong>Waiting to read the study's folder.</strong> If macOS is asking whether TII Monitor may access files in your Documents folder, choose Allow: the repository lives there. Worker activity and the connection are already live.
        {/if}
      </p>
    {/if}
    <BlockedRail />
    <div class="upper">
      <PlanBoard />
      <Figure1 />
    </div>
    <div class="lower">
      <Cluster />
      <Seam />
      <Mac />
    </div>
  {/if}
</div>
{#if store.figureOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
  <div class="backdrop" onclick={(e) => e.target === e.currentTarget && (store.figureOpen = false)}>
    <div class="sheet" role="dialog" aria-label="Figure 1, enlarged"><Figure1 big /></div>
  </div>
{/if}
<Tooltip />

<style>
  .app {
    height: 100%;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .upper {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 424px;
    border-bottom: 1px solid var(--line);
    flex: none;
  }
  .lower {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 1.22fr) 62px minmax(0, 1fr);
  }
  .compact .upper {
    grid-template-columns: minmax(0, 1fr) 300px;
  }
  .compact .lower {
    grid-template-columns: minmax(0, 1.3fr) 54px minmax(0, 1fr);
  }
  .compact .demo-banner {
    height: 34px;
    font-size: var(--fs-2);
  }
  .compact .demo-banner strong {
    font-size: 20px;
  }
  .backdrop {
    position: fixed;
    inset: 0;
    z-index: 50;
    display: grid;
    place-items: center;
    background: color-mix(in srgb, var(--ground) 82%, transparent);
  }
  .sheet {
    background: var(--ground);
    border: 1px solid var(--line-2);
    border-radius: 8px;
    box-shadow: var(--shadow);
    padding: 6px 8px 10px;
    max-width: calc(100vw - 32px);
    max-height: calc(100vh - 32px);
    overflow: auto;
  }
  .starting {
    padding: 24px;
    color: var(--ink-2);
  }
  .permission {
    flex: none;
    padding: 8px 14px;
    border-bottom: 1px solid var(--line);
    background: color-mix(in srgb, var(--wait) 11%, var(--ground));
    box-shadow: inset 4px 0 0 var(--wait);
    color: var(--ink-2);
  }
  .permission strong {
    color: var(--wait-text);
    font-weight: 700;
  }
  .demo-banner {
    flex: none;
    display: flex;
    align-items: center;
    gap: 16px;
    height: 46px;
    padding: 0 0;
    background: var(--wait);
    color: #16181b;
    font-size: var(--fs-4);
  }
  .demo-banner strong {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 0.06em;
    line-height: 1;
  }
  .stripes {
    align-self: stretch;
    width: 120px;
    background: repeating-linear-gradient(-45deg, #16181b 0 12px, transparent 12px 24px);
  }
  .stripes:last-child {
    margin-left: auto;
  }
</style>
